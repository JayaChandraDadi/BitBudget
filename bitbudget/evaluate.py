"""Quality and systems metrics: perplexity, KL to a reference model, size,
time-to-first-token, decode throughput and peak memory."""

from __future__ import annotations

import math
import statistics
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_reduce



# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_eval_tokens(tokenizer, dataset: str, seq_len: int, n_seqs: int) -> mx.array:
    """Contiguous (not shuffled) evaluation windows.

    ``dataset`` is ``"wikitext2"`` (WikiText-2 raw test split via the
    ``datasets`` package) or a path to a local text file.
    """
    if Path(dataset).exists():
        text = Path(dataset).read_text(encoding="utf-8", errors="ignore")
    elif dataset == "wikitext2":
        from datasets import load_dataset

        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(ds["text"])
    else:
        raise ValueError(f"unknown dataset {dataset!r}")
    tokens = mx.array(tokenizer.encode(text)).reshape(-1)
    n = tokens.size // seq_len
    rows = tokens[: n * seq_len].reshape(n, seq_len)
    return rows[:n_seqs] if n_seqs > 0 else rows


# --------------------------------------------------------------------------- #
# Quality
# --------------------------------------------------------------------------- #
def perplexity(model: nn.Module, tokens: mx.array, batch_size: int = 4) -> float:
    total, count = 0.0, 0
    for i in range(0, tokens.shape[0], batch_size):
        batch = tokens[i : i + batch_size]
        logits = model(batch[:, :-1]).astype(mx.float32)
        loss = nn.losses.cross_entropy(logits, batch[:, 1:], reduction="sum")
        total += loss.item()
        count += batch[:, 1:].size
    return math.exp(total / count)


def kl_to_reference(
    model: nn.Module, reference: nn.Module, tokens: mx.array, batch_size: int = 4
) -> float:
    """Mean per-token KL(reference || model) in nats."""
    total, count = 0.0, 0
    for i in range(0, tokens.shape[0], batch_size):
        batch = tokens[i : i + batch_size, :-1]
        lp = nn.log_softmax(reference(batch).astype(mx.float32), axis=-1)
        lq = nn.log_softmax(model(batch).astype(mx.float32), axis=-1)
        kl = (mx.exp(lp) * (lp - lq)).sum(axis=-1)
        total += kl.sum().item()
        count += kl.size
    return total / count


# --------------------------------------------------------------------------- #
# Size
# --------------------------------------------------------------------------- #
def model_nbytes(model: nn.Module) -> int:
    return tree_reduce(
        lambda acc, x: acc + x.nbytes if isinstance(x, mx.array) else acc,
        model.parameters(),
        0,
    )


def disk_bytes(path: str | Path) -> int | None:
    p = Path(path)
    if not p.is_dir():
        return None
    return sum(f.stat().st_size for f in p.glob("*.safetensors"))


def bits_per_weight(model: nn.Module) -> float:
    """Whole-model bits per weight, measured the same way as mlx-lm
    (``compute_bits_per_weight``) so every method is on the same axis."""
    from mlx_lm.utils import compute_bits_per_weight

    return compute_bits_per_weight(model)


# --------------------------------------------------------------------------- #
# Speed
# --------------------------------------------------------------------------- #
def benchmark_generation(
    model: nn.Module,
    tokenizer,
    prompt_tokens: int = 512,
    gen_tokens: int = 128,
    runs: int = 3,
    seed: int = 0,
) -> dict:
    """TTFT, decode tokens/s and peak memory via ``mlx_lm.stream_generate``.

    Uses a fixed random prompt so every model sees identical work; one warmup
    run is discarded; medians over ``runs``.
    """
    from mlx_lm.generate import stream_generate
    from mlx_lm.sample_utils import make_sampler

    mx.random.seed(seed)
    vocab = getattr(tokenizer, "vocab_size", None) or 32000
    prompt = mx.random.randint(100, min(vocab, 30000), (prompt_tokens,)).tolist()
    sampler = make_sampler(temp=0.0)

    def one_run():
        last = None
        t0 = time.perf_counter()
        for resp in stream_generate(
            model, tokenizer, prompt, max_tokens=gen_tokens, sampler=sampler
        ):
            last = resp
        wall = time.perf_counter() - t0
        return last, wall

    one_run()  # warmup (kernel compilation, allocator)
    mx.reset_peak_memory()
    t_start = time.time()
    ttft, tps, ptps, walls, gen = [], [], [], [], []
    last = None
    for _ in range(runs):
        last, wall = one_run()
        ptps.append(last.prompt_tps)
        ttft.append(1000.0 * last.prompt_tokens / last.prompt_tps)
        tps.append(last.generation_tps)
        walls.append(wall)
        gen.append(last.generation_tokens)
    t_end = time.time()
    return {
        "prompt_tokens": prompt_tokens,
        "gen_tokens": int(statistics.median(gen)),
        "ttft_ms": statistics.median(ttft),
        "prefill_tps": statistics.median(ptps),
        "decode_tps": statistics.median(tps),
        "peak_mem_gb": last.peak_memory,
        "wall_s_total": sum(walls),
        "tokens_total": sum(gen),
        # wall-clock window of the measured runs (for powermetrics alignment)
        "t_start": t_start,
        "t_end": t_end,
    }
