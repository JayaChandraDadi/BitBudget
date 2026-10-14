"""Calibration data and block-by-block activation capture."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterator

import mlx.core as mx
import mlx.nn as nn

from .arch import get_submodule, set_submodule


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def chunk_tokens(tokens: mx.array, seq_len: int, n_samples: int, seed: int = 0):
    """Split a 1-D token stream into ``n_samples`` random non-overlapping rows."""
    tokens = tokens.reshape(-1)
    n_chunks = tokens.size // seq_len
    if n_chunks == 0:
        raise ValueError(f"need at least {seq_len} tokens, got {tokens.size}")
    rows = tokens[: n_chunks * seq_len].reshape(n_chunks, seq_len)
    mx.random.seed(seed)
    order = mx.random.permutation(n_chunks)
    if n_samples > 0:
        order = order[: min(n_samples, n_chunks)]
    return rows[order]


def load_calibration(
    tokenizer, n_samples: int, seq_len: int, path: str | None = None, seed: int = 0
) -> mx.array:
    """Calibration tokens of shape ``(n_samples, seq_len)``.

    With ``path`` a local text file is used; otherwise the same mixed-domain
    calibration text that ``mlx_lm.awq`` / ``mlx_lm.dynamic_quant`` use, so
    comparisons against those baselines share calibration data.
    """
    if path is not None:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
        tokens = mx.array(tokenizer.encode(text))
        return chunk_tokens(tokens, seq_len, n_samples, seed)
    from mlx_lm.quant.utils import load_data

    mx.random.seed(seed)
    return load_data(tokenizer, num_samples=n_samples, sequence_length=seq_len)


# --------------------------------------------------------------------------- #
# Capturing activations
# --------------------------------------------------------------------------- #
class _Stop(Exception):
    pass


class _Store:
    """Plain-Python holder so captured arrays are not registered as parameters."""

    def __init__(self):
        self.items = []
        self.mask = None


class _FirstBlockCatcher(nn.Module):
    """Stands in for block 0; forwards attribute lookups (e.g. ``use_sliding``)
    to the real block so the parent model's forward runs unchanged."""

    def __init__(self, inner: nn.Module, store: _Store):
        super().__init__()
        self._store = store
        self._inner = _Store()
        self._inner.items = [inner]

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self._inner.items[0], key)

    def __call__(self, x, mask=None, cache=None, **kwargs):
        self._store.items.append(x)
        self._store.mask = mask
        raise _Stop


class _InputRecorder(nn.Module):
    def __init__(self, inner: nn.Module, store: _Store):
        super().__init__()
        self.inner = inner
        self._store = store

    def __call__(self, x, *args, **kwargs):
        self._store.items.append(x)
        return self.inner(x, *args, **kwargs)


def capture_block0_inputs(model: nn.Module, tokens: mx.array, batch_size: int):
    """Hidden states entering the first decoder block, plus the attention mask
    the model built for them."""
    store = _Store()
    original = model.layers[0]
    model.layers[0] = _FirstBlockCatcher(original, store)
    try:
        for i in range(0, tokens.shape[0], batch_size):
            try:
                model(tokens[i : i + batch_size])
            except _Stop:
                pass
            mx.eval(store.items[-1])
    finally:
        model.layers[0] = original
    return mx.concatenate(store.items, axis=0), store.mask


def run_batched(fn: Callable, x: mx.array, batch_size: int, *args, **kwargs):
    out = []
    for i in range(0, x.shape[0], batch_size):
        y = fn(x[i : i + batch_size], *args, **kwargs)
        mx.eval(y)
        out.append(y)
    return mx.concatenate(out, axis=0)


def capture_layer_inputs(
    block: nn.Module, h: mx.array, mask, layer_paths, batch_size: int
):
    """Run ``block`` on ``h`` recording the input of each layer in
    ``layer_paths``. Returns ``(block_output, {path: input})``."""
    stores = {}
    originals = {}
    for path in layer_paths:
        layer = get_submodule(block, path)
        stores[path] = _Store()
        originals[path] = layer
        set_submodule(block, path, _InputRecorder(layer, stores[path]))
    try:
        out = run_batched(lambda x: block(x, mask, cache=None), h, batch_size)
    finally:
        for path, layer in originals.items():
            set_submodule(block, path, layer)
    inputs = {p: mx.concatenate(s.items, axis=0) for p, s in stores.items()}
    mx.eval(inputs)
    return out, inputs


def iterate_blocks(
    model: nn.Module, tokens: mx.array, layer_paths, batch_size: int = 4
) -> Iterator[tuple[int, nn.Module, dict, object]]:
    """Yield ``(index, block, {path: input}, mask)`` for every decoder block.

    Inputs are those of the *full-precision* network: block ``i+1`` receives
    the output block ``i`` produced before the caller modified it. AWQ folding
    is function-preserving, so this matches what the quantized model sees up to
    the quantization noise of earlier blocks.
    """
    h, mask = capture_block0_inputs(model, tokens, batch_size)
    for i, block in enumerate(model.layers):
        h_next, inputs = capture_layer_inputs(block, h, mask, layer_paths, batch_size)
        yield i, block, inputs, mask
        h = h_next
        mx.eval(h)
