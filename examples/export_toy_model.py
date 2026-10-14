"""Package the toy model as a regular Hugging Face / mlx-lm model directory
(config.json + model.safetensors + a byte-level tokenizer), so the *real* CLI
path (``mlx_lm.load`` -> ``bitbudget quantize`` -> ``bitbudget eval``) can be
exercised end to end without downloading anything.

    python examples/export_toy_model.py --weights results/toy/toy_model.npz \\
        --config results/toy/toy_config.json --out results/toy/hf_model
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np


def byte_tokenizer_json() -> dict:
    """A 256-token byte-level tokenizer: token id == byte value."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from tokenizers.pre_tokenizers import ByteLevel

    alphabet = ByteLevel.alphabet()  # 256 printable stand-ins, one per byte
    # map each stand-in char back to the byte it represents
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + \
        list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    byte_of = {chr(c): b for b, c in zip(bs, cs)}
    assert set(byte_of) == set(alphabet)
    vocab = {ch: byte_of[ch] for ch in alphabet}
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tok.decoder = decoders.ByteLevel()
    return json.loads(tok.to_str())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(args.config).read_text())
    cfg = {k: v for k, v in cfg.items() if not k.startswith("train_")}
    cfg["torch_dtype"] = "bfloat16"
    (out / "config.json").write_text(json.dumps(cfg, indent=1))

    w = np.load(args.weights)
    mx.save_safetensors(str(out / "model.safetensors"),
                        {k: mx.array(w[k]).astype(mx.bfloat16) for k in w.files},
                        metadata={"format": "mlx"})

    (out / "tokenizer.json").write_text(json.dumps(byte_tokenizer_json()))
    (out / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": 4096,
        "eos_token": None, "bos_token": None,
    }))
    print(f"wrote {out}")

    # round-trip check through mlx-lm
    from mlx_lm import load

    model, tok = load(str(out))
    text = "def f(x):\n    return x  # ünïcode ✓\n"
    ids = tok.encode(text)
    assert ids == list(text.encode("utf-8")), "tokenizer must map bytes to ids 0-255"
    assert tok.decode(ids) == text
    print("tokenizer round-trip OK; logits shape", model(mx.array([ids])).shape)


if __name__ == "__main__":
    main()
