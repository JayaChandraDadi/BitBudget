"""Toy-scale validation that runs anywhere MLX runs (including Linux CPU).

Trains a small byte-level Llama-architecture model on a local text corpus
(Python standard-library source by default), then compares quantization
methods at matched memory budgets on held-out files:

    rtn               uniform round-to-nearest
    bitbudget-uniform uniform bits + AWQ scaling + clipping (our AWQ)
    bitbudget-noawq   mixed bits from the allocator, plain RTN (allocation only)
    bitbudget         mixed bits + AWQ + clipping (full method)

This is a *pipeline* check at toy scale: a 5M-parameter model has none of the
large activation outliers that make AWQ shine on billion-parameter LLMs, so do
not read the numbers as predictions for Qwen/Llama. The real experiment is
scripts/run_sweep.sh on Apple silicon.

    python examples/toy_validation.py --out results/toy
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm.models import llama
from mlx_lm.utils import dequantize_model

from bitbudget import Config, allocate, apply_plan, compute_curves, quantize_in_memory
from bitbudget.evaluate import perplexity
from bitbudget.pipeline import uniform_plan
from bitbudget.plotting import plot_bitmap, plot_pareto


def load_corpus_np(root: Path, max_bytes: int):
    """Byte tokens (uint8 -> int32) with a deterministic file-level split:
    ~10% of files are held out for evaluation."""
    train, test = bytearray(), bytearray()
    files = sorted(p for p in root.rglob("*.py") if "test" not in p.parts
                   and "site-packages" not in p.parts)
    for p in files:
        data = p.read_bytes()
        held_out = int(hashlib.md5(str(p.relative_to(root)).encode()).hexdigest(), 16) % 10 == 0
        (test if held_out else train).extend(data + b"\n")
        if len(train) > max_bytes:
            break
    as_np = lambda b: np.frombuffer(bytes(b), dtype=np.uint8).astype(np.int32)
    return as_np(train[:max_bytes]), as_np(test)


def load_corpus(root: Path, max_bytes: int):
    train, test = load_corpus_np(root, max_bytes)
    return mx.array(train), mx.array(test)


def make_model(args):
    margs = llama.ModelArgs(
        model_type="llama",
        hidden_size=args.hidden,
        num_hidden_layers=args.layers,
        intermediate_size=args.inter,
        num_attention_heads=args.hidden // 64,
        num_key_value_heads=max(1, args.hidden // 128),
        rms_norm_eps=1e-5,
        vocab_size=256,
        tie_word_embeddings=True,
    )
    return llama.Model(margs)


def train(model, data, args):
    opt = optim.AdamW(learning_rate=optim.cosine_decay(args.lr, args.steps), weight_decay=0.1)

    def loss_fn(m, x):
        return nn.losses.cross_entropy(m(x[:, :-1]), x[:, 1:], reduction="mean")

    step_fn = nn.value_and_grad(model, loss_fn)
    n = data.size - args.seq_len - 1
    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = mx.random.randint(0, n, (args.batch,))
        x = data[idx[:, None] + mx.arange(args.seq_len + 1)[None]]
        loss, grads = step_fn(model, x)
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state, loss)
        if step % 100 == 0 or step == 1:
            print(f"  step {step:5d}  loss {loss.item():.3f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)


def rows(tokens, seq_len, n=None):
    k = tokens.size // seq_len
    r = tokens[: k * seq_len].reshape(k, seq_len)
    return r[:n] if n else r


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
    ap.add_argument("--n-calib", type=int, default=8)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--budgets", default="3.0,3.5,4.0")
    ap.add_argument("--weights", help="pre-trained toy weights (.npz from train_toy_jax.py)")
    ap.add_argument("--n-grid", type=int, default=10)
    ap.add_argument("--out", default="results/toy")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    mx.random.seed(args.seed)
    train_tok, test_tok = load_corpus(Path(args.corpus), args.max_bytes)
    print(f"corpus: {train_tok.size/1e6:.1f}M train bytes, {test_tok.size/1e6:.2f}M held-out bytes")

    ckpt = out / "toy_model.safetensors"
    model = make_model(args)
    n_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"model: {n_params/1e6:.2f}M params")
    if args.weights:
        w = np.load(args.weights)
        model.load_weights([(k, mx.array(w[k])) for k in w.files])
    elif ckpt.exists():
        model.load_weights(str(ckpt))
    else:
        train(model, train_tok, args)
        model.save_weights(str(ckpt))
    # Quantize from a bf16 checkpoint, as with real models (16-bit scale/bias).
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    base_weights = tree_flatten(model.parameters())

    calib = rows(train_tok[-2_000_000:], args.seq_len)
    mx.random.seed(args.seed + 1)
    calib = calib[mx.random.permutation(calib.shape[0])[: args.n_calib]]
    evals = rows(test_tok, args.seq_len, args.n_eval)

    def fresh():
        m = make_model(args)
        m.set_dtype(mx.bfloat16)
        m.load_weights(base_weights)
        mx.eval(m.parameters())
        return m

    results = []

    def record(method, tag, m, linear_bpw, extra=None):
        # Evaluate the quantized weights in fp32 ("fake quant"): identical
        # weights, but MLX's quantized-matmul kernels are ~13x slower than
        # dense matmul on the Linux CPU backend. Every method, including the
        # bf16 reference, is evaluated the same way.
        dequantize_model(m)
        m.set_dtype(mx.float32)
        ppl = perplexity(m, evals, batch_size=16)
        r = {"tag": tag, "method": method, "bpw": linear_bpw, "ppl": ppl, **(extra or {})}
        results.append(r)
        print(f"  {tag:28s} {linear_bpw:5.3f} bpw  ppl {ppl:.4f}", flush=True)

    print("\nfp16 reference")
    record("fp16", "fp16", fresh(), 16.0)

    embed_bits = 8
    over = 32 / 64  # bf16 scale+bias per group of 64

    t_run = time.time()
    print("\nuniform baselines")
    for b in (2, 3, 4):
        m = fresh()
        quantize_in_memory(m, uniform_plan(m, b, 64), embed_bits=embed_bits)
        record("rtn", f"rtn-{b}bit", m, b + over)
        m = fresh()
        plan = uniform_plan(m, b, 64)
        apply_plan(m, calib, plan, Config(bits=(b,), n_grid=args.n_grid, verbose=False))
        quantize_in_memory(m, plan, embed_bits=embed_bits)
        record("bitbudget-uniform", f"awq-{b}bit", m, b + over)
        print(f"    ({time.time() - t_run:.0f}s elapsed)", flush=True)

    bits = (2, 3, 4, 6, 8)
    print("\nsensitivity curves")
    curves = {}
    for name, cfg in {
        "fisher": Config(bits=bits, n_grid=args.n_grid, verbose=False),
        "fisher-noawq": Config(bits=bits, awq=False, clip=False, verbose=False),
        "output": Config(bits=bits, n_grid=args.n_grid, sensitivity="output", verbose=False),
    }.items():
        t0 = time.time()
        curves[name] = (compute_curves(fresh(), calib, cfg), cfg)
        curves[name][0].save(out / f"curves_{name}.json")
        print(f"  {name}: {time.time() - t0:.0f}s")

    print("\nmixed precision at matched budgets")
    plans = {}
    budgets = [float(b) for b in args.budgets.split(",")]
    for target in budgets:
        for name, method in (("fisher", "bitbudget"), ("fisher-noawq", "bitbudget-noawq"),
                             ("output", "bitbudget-output")):
            if name == "output" and target != 3.5:
                continue  # ablation at a single budget
            c, cfg = curves[name]
            a = allocate(c, target)
            m = fresh()
            apply_plan(m, calib, a["plan"], cfg)
            quantize_in_memory(m, a["plan"], embed_bits=embed_bits)
            record(method, f"{method}-{target}", m, a["bpw"],
                   {"target_bpw": target, "histogram": a["histogram"]})
            if name == "fisher":
                plans[target] = a["plan"]

    (out / "results.jsonl").write_text("\n".join(json.dumps(r) for r in results) + "\n")
    (out / "plans.json").write_text(json.dumps({str(k): v for k, v in plans.items()}, indent=1))
    plot_pareto([r for r in results if r["method"] != "bitbudget-output"],
                out / "toy_pareto.png",
                title=f"Toy byte-level Llama ({n_params/1e6:.1f}M params), held-out ppl",
                xlabel="bits per weight (decoder linear layers, incl. scale/bias)")
    for t, plan in plans.items():
        plot_bitmap(plan, out / f"toy_bitmap_{t}.png", title=f"Toy model: bits per layer at {t} bpw")
    print(f"\nwrote {out}/results.jsonl and figures")


if __name__ == "__main__":
    main()
