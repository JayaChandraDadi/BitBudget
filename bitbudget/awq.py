"""Activation-aware scale search, scale folding and clipping (AWQ, Lin et al. 2024),
generalized so that every layer in a group can have its *own* bit-width."""

from __future__ import annotations

from typing import Callable, Sequence

import mlx.core as mx
import mlx.nn as nn

from .calibration import run_batched
from .quant import qdq


def activation_magnitude(x: mx.array) -> mx.array:
    """Mean |x| per input channel (the AWQ salience statistic)."""
    x = x.reshape(-1, x.shape[-1]).astype(mx.float32)
    return mx.abs(x).mean(axis=0)


def candidate_scales(x_mag: mx.array, alpha: float) -> mx.array:
    s = mx.maximum(x_mag**alpha, 1e-4)
    # Normalize so the scales are centred around 1 (as in the reference code);
    # this keeps folded norm weights in a well-conditioned range.
    return s / mx.sqrt(s.max() * s.min())


def search_scales(
    layers: Sequence[nn.Linear],
    bits: Sequence[int],
    objective: Callable[[mx.array], mx.array],
    x: mx.array,
    group_size: int,
    n_grid: int = 20,
    batch_size: int = 4,
):
    """Grid-search ``s = mean|x|^alpha`` minimizing the MSE of ``objective(x)``
    when every layer ``l`` is replaced by ``Q_bits[l](W_l * s) / s``.

    ``alpha = 0`` (no scaling, i.e. plain round-to-nearest) is always a
    candidate, so the result is never worse than RTN on the calibration data.

    Returns ``(scales, alpha, error)``. Layer weights are restored on exit.
    """
    x_mag = activation_magnitude(x)
    ref = run_batched(objective, x, batch_size).astype(mx.float32)
    originals = [l.weight for l in layers]

    best = (float("inf"), None, None)
    try:
        for i in range(n_grid + 1):
            alpha = i / n_grid
            s = candidate_scales(x_mag, alpha)
            for layer, w, b in zip(layers, originals, bits):
                ws = (w.astype(mx.float32) * s).astype(w.dtype)
                layer.weight = (qdq(ws, b, group_size).astype(mx.float32) / s).astype(
                    w.dtype
                )
            out = run_batched(objective, x, batch_size).astype(mx.float32)
            err = mx.mean((out - ref) ** 2).item()
            if err < best[0]:
                best = (err, s, alpha)
    finally:
        for layer, w in zip(layers, originals):
            layer.weight = w
    err, s, alpha = best
    return s, alpha, err


def fold_scales(prev: nn.Module, layers: Sequence[nn.Linear], s: mx.array) -> None:
    """Fold per-input-channel scales into ``prev`` and ``layers``.

    ``prev(x) / s`` feeds ``W * s``, so ``(W * s)(x / s) == W x``: the function
    is unchanged, but salient channels are magnified before quantization.
    """
    if isinstance(prev, nn.Linear):
        dt = prev.weight.dtype
        prev.weight = (prev.weight.astype(mx.float32) / s[:, None]).astype(dt)
        if "bias" in prev:
            prev.bias = (prev.bias.astype(mx.float32) / s).astype(prev.bias.dtype)
    elif isinstance(prev, (nn.RMSNorm, nn.LayerNorm)):
        dt = prev.weight.dtype
        prev.weight = (prev.weight.astype(mx.float32) / s).astype(dt)
        if "bias" in prev:
            prev.bias = (prev.bias.astype(mx.float32) / s).astype(prev.bias.dtype)
    else:
        raise NotImplementedError(f"cannot fold scales into {type(prev).__name__}")
    for layer in layers:
        dt = layer.weight.dtype
        layer.weight = (layer.weight.astype(mx.float32) * s).astype(dt)


def search_clip(
    w: mx.array,
    x: mx.array,
    bits: int,
    group_size: int,
    n_grid: int = 20,
    max_shrink: float = 0.5,
    n_tokens: int = 512,
    out_chunk: int = 256,
) -> mx.array:
    """Per-group range clipping (the second half of AWQ).

    For every (output row, input group) pair, search a shrink ratio
    ``r in [1 - max_shrink, 1]`` for the clipping range ``[r*min, r*max]``
    that minimizes ``||(Q(clip(W)) - W) x||^2`` over calibration tokens.
    Returns the clipped (still full-precision) weight; clipping is applied
    before ``mx.quantize`` so it needs no runtime support.
    """
    out_dim, in_dim = w.shape
    n_groups = in_dim // group_size
    xs = x.reshape(-1, in_dim)
    stride = max(1, xs.shape[0] // n_tokens)
    xs = xs[::stride][:n_tokens].astype(mx.float32)
    xg = xs.reshape(xs.shape[0], n_groups, group_size)  # (T, G, gs)

    chunks = []
    for o in range(0, out_dim, out_chunk):
        wc = w[o : o + out_chunk].astype(mx.float32)
        wg = wc.reshape(wc.shape[0], n_groups, group_size)  # (O, G, gs)
        w_max = wg.max(axis=-1, keepdims=True)
        w_min = wg.min(axis=-1, keepdims=True)
        best_err = mx.full((wg.shape[0], n_groups), float("inf"))
        best_w = wg
        for i in range(int(max_shrink * n_grid) + 1):
            r = 1.0 - i / n_grid
            cur = mx.clip(wg, w_min * r, w_max * r)
            q = qdq(cur.reshape(wc.shape).astype(w.dtype), bits, group_size)
            delta = q.astype(mx.float32).reshape(wg.shape) - wg  # (O, G, gs)
            # per-(row, group) output error over calibration tokens
            err = (mx.einsum("ogk,tgk->otg", delta, xg) ** 2).sum(axis=1)
            better = err < best_err
            best_err = mx.where(better, err, best_err)
            best_w = mx.where(better[..., None], cur, best_w)
            mx.eval(best_err, best_w)
        chunks.append(best_w.reshape(wc.shape))
    return mx.concatenate(chunks, axis=0).astype(w.dtype)
