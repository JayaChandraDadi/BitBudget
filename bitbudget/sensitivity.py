"""How much does quantizing one layer to ``b`` bits hurt the model?

Two estimators of the loss increase caused by a weight perturbation ``dW``:

* ``fisher`` (default) -- second-order Taylor expansion with the diagonal
  empirical Fisher ``F = E[g^2]`` as the Hessian approximation
  (Optimal Brain Damage / HAWQ style):  ``dL ~= 1/2 * sum(F * dW^2)``.
  Unlike a layer-local error, this is measured in units of *end-to-end loss*,
  so errors of different layers are comparable and can be traded off against
  each other in the bit allocator.

* ``output`` -- layer-local output error ``E_x ||dW x||^2``. Cheap, needs no
  backward pass, but not comparable across layers (useful as an ablation).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map


def fisher_diagonal(
    model: nn.Module,
    tokens: mx.array,
    layer_paths: list[str],
    batch_size: int = 1,
) -> dict[str, mx.array]:
    """Diagonal empirical Fisher of the next-token loss w.r.t. the weights of
    ``layer_paths`` (model-root paths, e.g. ``model.layers.3.mlp.up_proj``).

    The squared gradient is accumulated per mini-batch (``batch_size=1`` gives
    per-sequence Fisher, the most faithful estimate). Returned in float32.
    """
    wanted = {f"{p}.weight" for p in layer_paths}
    was_training = model.training
    model.freeze()
    for name, mod in model.named_modules():
        if name in layer_paths:
            mod.unfreeze(keys=["weight"], recurse=False)

    def loss_fn(m, x):
        logits = m(x[:, :-1]).astype(mx.float32)
        return nn.losses.cross_entropy(logits, x[:, 1:], reduction="mean")

    grad_fn = nn.value_and_grad(model, loss_fn)
    acc = None
    n = 0
    for i in range(0, tokens.shape[0], batch_size):
        _, grads = grad_fn(model, tokens[i : i + batch_size])
        sq = tree_map(lambda g: g.astype(mx.float32) ** 2, grads)
        acc = sq if acc is None else tree_map(lambda a, b: a + b, acc, sq)
        mx.eval(acc)
        n += 1

    model.unfreeze()
    model.train(was_training)
    flat = dict(tree_flatten(acc))
    missing = wanted - flat.keys()
    if missing:
        raise RuntimeError(f"no gradient for {sorted(missing)[:3]}...")
    return {k[: -len(".weight")]: v / n for k, v in flat.items() if k in wanted}


def fisher_error(fisher: mx.array, w_ref: mx.array, w_q: mx.array) -> float:
    d = w_q.astype(mx.float32) - w_ref.astype(mx.float32)
    return 0.5 * (fisher * d * d).sum().item()


def output_error(x: mx.array, w_ref: mx.array, w_q: mx.array, n_tokens=2048) -> float:
    xs = x.reshape(-1, x.shape[-1])
    stride = max(1, xs.shape[0] // n_tokens)
    xs = xs[::stride][:n_tokens].astype(mx.float32)
    d = w_q.astype(mx.float32) - w_ref.astype(mx.float32)
    return ((xs @ d.T) ** 2).sum(axis=-1).mean().item()


def rescale_fisher_rows(f: mx.array, s: mx.array) -> mx.array:
    """Fisher after ``W' = W / s[:, None]`` (scale folded into output rows)."""
    return f * (s[:, None] ** 2)


def rescale_fisher_cols(f: mx.array, s: mx.array) -> mx.array:
    """Fisher after ``W' = W * s`` (scale folded into input columns).

    The loss is a function of ``W = W' / s``, so ``g' = g / s`` and
    ``F' = F / s^2``; therefore ``1/2 sum F' dW'^2 == 1/2 sum F dW^2``.
    """
    return f / (s**2)
