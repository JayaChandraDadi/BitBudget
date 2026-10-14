"""Architecture descriptions: which linear layers share an input, and where the
AWQ scales for that input can be folded so they cost nothing at inference."""

from __future__ import annotations

from dataclasses import dataclass, field

import mlx.nn as nn


@dataclass(frozen=True)
class ScaleGroup:
    """Linear layers that consume the same activation.

    Attributes:
        name: Human-readable id.
        prev: Block-local path of the op that produces the shared input. The
            per-channel AWQ scale ``s`` is folded into it (``prev /= s``) and
            into the consumers (``W *= s``), so the network function is
            unchanged before quantization.
        layers: Block-local paths of the consuming linear layers.
        module: Block-local path of the sub-module whose output is matched
            during the scale search (e.g. the whole attention block, so that the
            search sees the softmax). ``None`` means "the single layer itself".
        needs_mask: Whether ``module`` must be called with the attention mask.
    """

    name: str
    prev: str
    layers: tuple[str, ...]
    module: str | None = None
    needs_mask: bool = False


@dataclass(frozen=True)
class ArchSpec:
    groups: tuple[ScaleGroup, ...]
    # Linear layers that are quantized but have no foldable predecessor
    # (o_proj: with grouped-query attention the v_proj -> o_proj channel map is
    # not one-to-one, so AWQ leaves it unscaled).
    unscaled: tuple[str, ...] = field(default_factory=tuple)
    # Layers where clipping is skipped (q/k feed a softmax: clipping their
    # outliers hurts attention far more than the local MSE suggests).
    no_clip: tuple[str, ...] = field(default_factory=tuple)

    @property
    def all_layers(self) -> tuple[str, ...]:
        out = []
        for g in self.groups:
            out.extend(g.layers)
        out.extend(self.unscaled)
        return tuple(out)


# Order matters: mlp.down_proj folds into mlp.up_proj, which must happen before
# the gate/up group is searched and quantized.
LLAMA_SPEC = ArchSpec(
    groups=(
        ScaleGroup(
            name="attn_in",
            prev="input_layernorm",
            layers=("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
            module="self_attn",
            needs_mask=True,
        ),
        ScaleGroup(name="mlp_down", prev="mlp.up_proj", layers=("mlp.down_proj",)),
        ScaleGroup(
            name="mlp_in",
            prev="post_attention_layernorm",
            layers=("mlp.gate_proj", "mlp.up_proj"),
            module="mlp",
        ),
    ),
    unscaled=("self_attn.o_proj",),
    no_clip=("self_attn.q_proj", "self_attn.k_proj"),
)

ARCH_SPECS = {
    "llama": LLAMA_SPEC,
    "mistral": LLAMA_SPEC,
    "qwen2": LLAMA_SPEC,
    "qwen3": LLAMA_SPEC,
}


def get_submodule(module: nn.Module, path: str) -> nn.Module:
    for key in path.split("."):
        module = module[key] if not key.isdigit() else module[int(key)]
    return module


def set_submodule(module: nn.Module, path: str, value: nn.Module) -> None:
    *parents, last = path.split(".")
    for key in parents:
        module = module[key] if not key.isdigit() else module[int(key)]
    if last.isdigit():
        module[int(last)] = value
    else:
        setattr(module, last, value)


def spec_for(model: nn.Module, model_type: str | None = None) -> ArchSpec:
    model_type = model_type or getattr(model, "model_type", None)
    if model_type is None and hasattr(model, "args"):
        model_type = getattr(model.args, "model_type", None)
    if model_type not in ARCH_SPECS:
        raise NotImplementedError(
            f"model_type={model_type!r} is not supported yet. "
            f"Supported: {sorted(ARCH_SPECS)}"
        )
    spec = ARCH_SPECS[model_type]
    block = model.layers[0]
    for path in list(spec.all_layers) + [g.prev for g in spec.groups]:
        try:
            get_submodule(block, path)
        except (KeyError, IndexError, AttributeError) as e:
            raise NotImplementedError(
                f"{model_type}: expected sub-module {path!r} in each block"
            ) from e
    return spec


def layers_prefix(model: nn.Module) -> str:
    """Root path of the decoder block list, e.g. ``"model.layers"``."""
    first = model.layers[0]
    for name, mod in model.named_modules():
        if mod is first:
            return name.rsplit(".", 1)[0]
    raise RuntimeError("could not locate model.layers in the module tree")
