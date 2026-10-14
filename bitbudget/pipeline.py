"""End-to-end BitBudget pipeline.

    1. curves   = compute_curves(model, calib)       # e_l(b) for every layer l, bit-width b
    2. plan     = allocate(curves, target_bpw)       # multiple-choice knapsack
    3. apply_plan(model, calib, plan)                # AWQ scales + clipping at the planned bits
    4. model    = quantize_with_plan(model, plan)    # standard MLX QuantizedLinear layers

Sensitivity curves do not depend on the budget, so (1) runs once per model and
(2)-(4) can be repeated cheaply for every point on the size/quality curve.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from .allocate import SOLVERS
from .arch import ArchSpec, get_submodule, layers_prefix, spec_for
from .awq import fold_scales, search_clip, search_scales
from .calibration import iterate_blocks
from .quant import SUPPORTED_BITS, group_overhead_bits, is_quantizable, qdq
from .sensitivity import (
    fisher_diagonal,
    fisher_error,
    output_error,
    rescale_fisher_cols,
    rescale_fisher_rows,
)


@dataclass
class Config:
    bits: tuple[int, ...] = (2, 3, 4, 6, 8)
    group_size: int = 64
    n_grid: int = 20
    awq: bool = True
    clip: bool = True
    sensitivity: str = "fisher"  # "fisher" | "output"
    batch_size: int = 4
    fisher_batch_size: int = 1
    verbose: bool = True

    def __post_init__(self):
        self.bits = tuple(sorted(set(int(b) for b in self.bits)))
        bad = [b for b in self.bits if b not in SUPPORTED_BITS]
        if bad:
            raise ValueError(f"unsupported bits {bad}; MLX supports {SUPPORTED_BITS}")
        if self.sensitivity not in ("fisher", "output"):
            raise ValueError("sensitivity must be 'fisher' or 'output'")


@dataclass
class Curves:
    """Estimated loss increase ``errors[path][bits]`` per layer and bit-width."""

    bits: list[int]
    group_size: int
    overhead_bits: int
    params: dict[str, int]
    errors: dict[str, dict[int, float]]
    meta: dict = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        d = asdict(self)
        d["errors"] = {k: {str(b): e for b, e in v.items()} for k, v in self.errors.items()}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(d, indent=1))

    @classmethod
    def load(cls, path: str | Path) -> "Curves":
        d = json.loads(Path(path).read_text())
        d["errors"] = {k: {int(b): e for b, e in v.items()} for k, v in d["errors"].items()}
        return cls(**d)


def _log(cfg: Config, msg: str) -> None:
    if cfg.verbose:
        print(msg, flush=True)


def _quantizable_paths(model: nn.Module, spec: ArchSpec, group_size: int) -> list[str]:
    prefix = layers_prefix(model)
    paths = []
    for i, block in enumerate(model.layers):
        for p in spec.all_layers:
            w = get_submodule(block, p).weight
            if not is_quantizable(w, group_size):
                raise ValueError(
                    f"{prefix}.{i}.{p}: input dim {w.shape[-1]} is not divisible by "
                    f"group size {group_size}"
                )
            paths.append(f"{prefix}.{i}.{p}")
    return paths


def _process_block(
    block: nn.Module,
    inputs: dict[str, mx.array],
    mask,
    spec: ArchSpec,
    bits_of: Callable[[str], int],
    cfg: Config,
    fisher: dict[str, mx.array] | None = None,
    compute_errors: bool = False,
) -> dict:
    """Apply AWQ scaling + clipping to one block *in place*, with the bit-width
    of each layer given by ``bits_of(local_path)``.

    If ``compute_errors``, also return the estimated loss increase of each
    layer once quantized: ``{local_path: error}`` plus the chosen alphas.
    """
    x = dict(inputs)
    F = dict(fisher) if fisher is not None else {}
    errors, alphas = {}, {}

    def finalize(p: str):
        layer = get_submodule(block, p)
        b = bits_of(p)
        w_ref = layer.weight
        if cfg.clip and p not in spec.no_clip:
            layer.weight = search_clip(w_ref, x[p], b, cfg.group_size)
        if compute_errors:
            w_q = qdq(layer.weight, b, cfg.group_size)
            if cfg.sensitivity == "fisher":
                errors[p] = fisher_error(F[p], w_ref, w_q)
            else:
                errors[p] = output_error(x[p], w_ref, w_q)

    for g in spec.groups:
        layers = [get_submodule(block, p) for p in g.layers]
        if cfg.awq:
            if g.module is None:
                objective = layers[0]
            else:
                mod = get_submodule(block, g.module)
                objective = (lambda t, m=mod: m(t, mask)) if g.needs_mask else mod
            s, alpha, _ = search_scales(
                layers,
                [bits_of(p) for p in g.layers],
                objective,
                x[g.layers[0]],
                cfg.group_size,
                n_grid=cfg.n_grid,
                batch_size=cfg.batch_size,
            )
            prev = get_submodule(block, g.prev)
            fold_scales(prev, layers, s)
            for p in g.layers:
                x[p] = (x[p].astype(mx.float32) / s).astype(x[p].dtype)
                if p in F:
                    F[p] = rescale_fisher_cols(F[p], s)
            if isinstance(prev, nn.Linear) and g.prev in F:
                F[g.prev] = rescale_fisher_rows(F[g.prev], s)
            alphas[g.name] = alpha
        for p in g.layers:
            finalize(p)
    for p in spec.unscaled:
        finalize(p)
    mx.eval(block.parameters())
    return {"errors": errors, "alphas": alphas}


def compute_curves(
    model: nn.Module,
    tokens: mx.array,
    cfg: Config,
    fisher: dict[str, mx.array] | None = None,
    fisher_tokens: mx.array | None = None,
) -> Curves:
    """Estimate ``e_l(b)``: loss increase from quantizing layer ``l`` to ``b``
    bits *after* AWQ scaling and clipping searched for that bit-width."""
    spec = spec_for(model)
    prefix = layers_prefix(model)
    paths = _quantizable_paths(model, spec, cfg.group_size)
    t0 = time.time()

    if cfg.sensitivity == "fisher" and fisher is None:
        ft = fisher_tokens if fisher_tokens is not None else tokens
        _log(cfg, f"[bitbudget] Fisher diagonal over {ft.shape[0]} sequences...")
        fisher = fisher_diagonal(model, ft, paths, batch_size=cfg.fisher_batch_size)
        _log(cfg, f"[bitbudget]   done in {time.time() - t0:.1f}s")

    errors: dict[str, dict[int, float]] = {p: {} for p in paths}
    alphas: dict[str, dict[int, dict]] = {}
    params = {}
    dtype = None
    for i, block, inputs, mask in iterate_blocks(
        model, tokens, spec.all_layers, cfg.batch_size
    ):
        tb = time.time()
        snapshot = tree_flatten(block.parameters())
        local_f = None
        if fisher is not None:
            local_f = {p: fisher[f"{prefix}.{i}.{p}"] for p in spec.all_layers}
        for p in spec.all_layers:
            w = get_submodule(block, p).weight
            params[f"{prefix}.{i}.{p}"] = int(w.size)
            dtype = w.dtype
        alphas[str(i)] = {}
        for b in cfg.bits:
            res = _process_block(
                block, inputs, mask, spec, lambda p, b=b: b, cfg, local_f, True
            )
            for p, e in res["errors"].items():
                errors[f"{prefix}.{i}.{p}"][b] = e
            alphas[str(i)][b] = res["alphas"]
            block.update(tree_unflatten(snapshot))  # undo folds / clipping
        mx.eval(block.parameters())
        _log(cfg, f"[bitbudget] block {i:3d}: curves in {time.time() - tb:.1f}s")

    return Curves(
        bits=list(cfg.bits),
        group_size=cfg.group_size,
        overhead_bits=group_overhead_bits(dtype),
        params=params,
        errors=errors,
        meta={
            "sensitivity": cfg.sensitivity,
            "awq": cfg.awq,
            "clip": cfg.clip,
            "n_grid": cfg.n_grid,
            "calib_shape": list(tokens.shape),
            "alphas": alphas,
            "seconds": round(time.time() - t0, 1),
        },
    )


def allocate(curves: Curves, target_bpw: float, solver: str = "dp") -> dict:
    """Pick per-layer bits so the quantized linear layers fit in
    ``target_bpw`` effective bits per weight (including scale/bias overhead).

    Returns ``{"plan": {path: bits}, "bpw": achieved, "predicted_error": ...}``.
    """
    paths = sorted(curves.errors)
    bits = curves.bits
    over = curves.overhead_bits / curves.group_size
    costs = [[curves.params[p] * (b + over) for b in bits] for p in paths]
    errors = [[curves.errors[p][b] for b in bits] for p in paths]
    total = sum(curves.params[p] for p in paths)
    budget = target_bpw * total
    min_bpw = sum(min(c) for c in costs) / total
    if target_bpw < min_bpw:
        raise ValueError(
            f"target {target_bpw} bpw is below the minimum achievable {min_bpw:.3f} "
            f"bpw for bits={bits}, group_size={curves.group_size} "
            f"(includes {over:.2f} bpw of scale/bias overhead)"
        )
    res = SOLVERS[solver](errors, costs, budget)
    plan = {p: bits[j] for p, j in zip(paths, res.choice)}
    return {
        "plan": plan,
        "target_bpw": target_bpw,
        "bpw": res.cost / total,
        "predicted_error": res.error,
        "solver": solver,
        "histogram": {b: sum(1 for v in plan.values() if v == b) for b in bits},
        "param_share": {
            b: sum(curves.params[p] for p, v in plan.items() if v == b) / total
            for b in bits
        },
    }


def uniform_plan(model: nn.Module, bits: int, group_size: int) -> dict[str, int]:
    spec = spec_for(model)
    return {p: bits for p in _quantizable_paths(model, spec, group_size)}


def apply_plan(model: nn.Module, tokens: mx.array, plan: dict[str, int], cfg: Config):
    """Fold AWQ scales and clip weights in place for the given per-layer bits.
    The model stays full precision (and functionally ~unchanged) afterwards;
    call :func:`quantize_with_plan` to actually quantize it."""
    spec = spec_for(model)
    prefix = layers_prefix(model)
    if not (cfg.awq or cfg.clip):
        return {}
    t0 = time.time()
    alphas = {}
    for i, block, inputs, mask in iterate_blocks(
        model, tokens, spec.all_layers, cfg.batch_size
    ):
        res = _process_block(
            block,
            inputs,
            mask,
            spec,
            lambda p, i=i: plan[f"{prefix}.{i}.{p}"],
            cfg,
        )
        alphas[str(i)] = res["alphas"]
        _log(cfg, f"[bitbudget] block {i:3d}: scaled + clipped")
    _log(cfg, f"[bitbudget] apply_plan done in {time.time() - t0:.1f}s")
    return alphas


def quant_predicate(plan: dict[str, int], group_size: int, embed_bits: int):
    def predicate(path: str, module: nn.Module):
        if not hasattr(module, "to_quantized"):
            return False
        if path in plan:
            return {"group_size": group_size, "bits": plan[path], "mode": "affine"}
        if isinstance(module, nn.Embedding) or path.endswith("lm_head"):
            return {"group_size": group_size, "bits": embed_bits, "mode": "affine"}
        return True

    return predicate


def quantize_with_plan(
    model: nn.Module,
    config: dict,
    plan: dict[str, int],
    group_size: int = 64,
    embed_bits: int = 4,
    default_bits: int = 4,
):
    """Quantize with per-layer bit-widths. The resulting config carries one
    entry per layer, which ``mlx_lm.load`` understands natively."""
    from mlx_lm.utils import quantize_model

    config = dict(config)
    # quantize_model switches to per-layer configs only if a quantization key
    # already exists; seed it with the defaults.
    config["quantization"] = {
        "group_size": group_size,
        "bits": default_bits,
        "mode": "affine",
    }
    return quantize_model(
        model,
        config,
        group_size=group_size,
        bits=default_bits,
        quant_predicate=quant_predicate(plan, group_size, embed_bits),
    )


def quantize_in_memory(model: nn.Module, plan: dict[str, int], group_size: int = 64,
                       embed_bits: int = 4) -> nn.Module:
    """Like :func:`quantize_with_plan` but without an mlx-lm config (for tests
    and toy models)."""
    nn.quantize(
        model,
        group_size=group_size,
        bits=4,
        class_predicate=lambda p, m: (
            quant_predicate(plan, group_size, embed_bits)(p, m)
            if (hasattr(m, "to_quantized") and m.weight.shape[-1] % group_size == 0)
            else False
        ),
    )
    return model
