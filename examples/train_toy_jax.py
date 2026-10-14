"""Train the toy byte-level Llama with JAX, export MLX-compatible weights.

Why this exists: MLX's Linux CPU backend has no BLAS acceleration (a few
GFLOP/s), so training even a toy model there takes hours. On Apple silicon
just use ``toy_validation.py`` directly, which trains with MLX.

The JAX model mirrors ``mlx_lm.models.llama`` exactly (RMSNorm, non-traditional
RoPE, grouped-query attention, SwiGLU MLP, tied embeddings); ``--check`` loads
the exported weights into MLX and asserts the logits match.

    pip install "jax[cpu]" optax
    python examples/train_toy_jax.py --out results/toy --check
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from toy_validation import load_corpus_np  # noqa: E402  (same corpus + split)


def init_params(key, cfg):
    H, I, L = cfg["hidden_size"], cfg["intermediate_size"], cfg["num_hidden_layers"]
    nh, nkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    hd = H // nh
    keys = iter(jax.random.split(key, 8 * L + 2))
    n = lambda shape, std=0.02: std * jax.random.normal(next(keys), shape)
    out_std = 0.02 / math.sqrt(2 * L)
    p = {"model.embed_tokens.weight": n((cfg["vocab_size"], H)),
         "model.norm.weight": jnp.ones(H)}
    for i in range(L):
        pre = f"model.layers.{i}."
        p.update({
            pre + "input_layernorm.weight": jnp.ones(H),
            pre + "post_attention_layernorm.weight": jnp.ones(H),
            pre + "self_attn.q_proj.weight": n((nh * hd, H)),
            pre + "self_attn.k_proj.weight": n((nkv * hd, H)),
            pre + "self_attn.v_proj.weight": n((nkv * hd, H)),
            pre + "self_attn.o_proj.weight": n((H, nh * hd), out_std),
            pre + "mlp.gate_proj.weight": n((I, H)),
            pre + "mlp.up_proj.weight": n((I, H)),
            pre + "mlp.down_proj.weight": n((H, I), out_std),
        })
    return p


def rms_norm(x, w, eps):
    return w * (x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps))


def rope(x, base=10000.0):
    # non-traditional (rotate-half) RoPE, as mlx nn.RoPE(traditional=False)
    L, d = x.shape[-2], x.shape[-1]
    half = d // 2
    inv = base ** (-jnp.arange(half) / half)
    ang = jnp.arange(L)[:, None] * inv[None]
    cos, sin = jnp.cos(ang), jnp.sin(ang)
    x1, x2 = x[..., :half], x[..., half:]
    return jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


def forward(p, tokens, cfg):
    H, L = cfg["hidden_size"], cfg["num_hidden_layers"]
    nh, nkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    hd, eps = H // nh, cfg["rms_norm_eps"]
    B, T = tokens.shape
    h = p["model.embed_tokens.weight"][tokens]
    mask = jnp.tril(jnp.ones((T, T), dtype=bool))
    for i in range(L):
        pre = f"model.layers.{i}."
        x = rms_norm(h, p[pre + "input_layernorm.weight"], eps)
        q = (x @ p[pre + "self_attn.q_proj.weight"].T).reshape(B, T, nh, hd).transpose(0, 2, 1, 3)
        k = (x @ p[pre + "self_attn.k_proj.weight"].T).reshape(B, T, nkv, hd).transpose(0, 2, 1, 3)
        v = (x @ p[pre + "self_attn.v_proj.weight"].T).reshape(B, T, nkv, hd).transpose(0, 2, 1, 3)
        q, k = rope(q), rope(k)
        rep = nh // nkv
        k, v = jnp.repeat(k, rep, axis=1), jnp.repeat(v, rep, axis=1)
        s = (q @ k.transpose(0, 1, 3, 2)) * hd**-0.5
        s = jnp.where(mask, s, -jnp.inf)
        a = jax.nn.softmax(s, axis=-1) @ v
        a = a.transpose(0, 2, 1, 3).reshape(B, T, nh * hd)
        h = h + a @ p[pre + "self_attn.o_proj.weight"].T
        x = rms_norm(h, p[pre + "post_attention_layernorm.weight"], eps)
        g = jax.nn.silu(x @ p[pre + "mlp.gate_proj.weight"].T) * (x @ p[pre + "mlp.up_proj.weight"].T)
        h = h + g @ p[pre + "mlp.down_proj.weight"].T
    h = rms_norm(h, p["model.norm.weight"], eps)
    return h @ p["model.embed_tokens.weight"].T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="/usr/lib/python3.13")
    ap.add_argument("--max-bytes", type=int, default=8_000_000)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--inter", type=int, default=704)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", default="results/toy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    cfg = {
        "model_type": "llama", "hidden_size": args.hidden,
        "num_hidden_layers": args.layers, "intermediate_size": args.inter,
        "num_attention_heads": args.hidden // 64,
        "num_key_value_heads": max(1, args.hidden // 128),
        "rms_norm_eps": 1e-5, "vocab_size": 256, "tie_word_embeddings": True,
    }
    train, _ = load_corpus_np(Path(args.corpus), args.max_bytes)
    params = init_params(jax.random.PRNGKey(args.seed), cfg)
    n_params = sum(v.size for v in params.values())
    print(f"model: {n_params/1e6:.2f}M params, corpus {train.size/1e6:.1f}M bytes", flush=True)

    sched = optax.warmup_cosine_decay_schedule(
        0.0, args.lr, min(100, args.steps // 10), args.steps, args.lr * 0.05)
    tx = optax.chain(optax.clip_by_global_norm(1.0),
                     optax.adamw(sched, weight_decay=0.1,
                                 mask=lambda p: {k: v.ndim == 2 for k, v in p.items()}))
    opt_state = tx.init(params)
    data = jnp.asarray(train)

    def loss_fn(p, x):
        logits = forward(p, x[:, :-1], cfg)
        return optax.softmax_cross_entropy_with_integer_labels(logits, x[:, 1:]).mean()

    @jax.jit
    def step(p, s, key):
        idx = jax.random.randint(key, (args.batch,), 0, data.size - args.seq_len - 1)
        x = data[idx[:, None] + jnp.arange(args.seq_len + 1)[None]]
        loss, g = jax.value_and_grad(loss_fn)(p, x)
        upd, s = tx.update(g, s, p)
        return optax.apply_updates(p, upd), s, loss

    key = jax.random.PRNGKey(args.seed + 1)
    t0 = time.time()
    log = []
    for i in range(1, args.steps + 1):
        key, sub = jax.random.split(key)
        params, opt_state, loss = step(params, opt_state, sub)
        if i % 100 == 0 or i == 1:
            l = float(loss)
            log.append({"step": i, "loss": l})
            print(f"  step {i:5d}  loss {l:.3f}  bits/byte {l/math.log(2):.3f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "toy_model.npz", **{k: np.asarray(v, dtype=np.float32) for k, v in params.items()})
    (out / "toy_config.json").write_text(json.dumps({**cfg, "train_log": log,
                                                       "train_seconds": time.time() - t0}, indent=1))
    print(f"saved {out}/toy_model.npz")

    if args.check:
        import mlx.core as mx
        from mlx_lm.models import llama

        m = llama.Model(llama.ModelArgs(**{k: v for k, v in cfg.items()}))
        m.load_weights([(k, mx.array(np.asarray(v))) for k, v in params.items()])
        x = np.asarray(train[: 2 * 128]).reshape(2, 128)
        ref = np.asarray(forward(params, jnp.asarray(x), cfg))
        got = np.array(m(mx.array(x)))
        err = np.abs(ref - got).max()
        print(f"JAX vs MLX max |logit diff| = {err:.2e}")
        assert err < 1e-2, "exported weights do not reproduce the JAX model in MLX"


if __name__ == "__main__":
    main()
