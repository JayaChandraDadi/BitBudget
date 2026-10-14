import json

import mlx.core as mx
import mlx.nn as nn
import pytest

from bitbudget.arch import layers_prefix, spec_for
from bitbudget.calibration import capture_block0_inputs, chunk_tokens, iterate_blocks
from bitbudget.pipeline import (
    Config,
    Curves,
    allocate,
    apply_plan,
    compute_curves,
    quantize_in_memory,
    uniform_plan,
)
from bitbudget.sensitivity import fisher_diagonal

from .conftest import make_tiny_llama

# fp32 tiny models: scale+bias overhead is 64 bits per group of 64 -> +1.0 bpw,
# so the cheapest 2-bit allocation is 3.0 bpw.
FAST = dict(n_grid=4, verbose=False)


def test_spec_and_prefix(tiny_llama):
    spec = spec_for(tiny_llama)
    assert "self_attn.o_proj" in spec.all_layers
    assert layers_prefix(tiny_llama) == "model.layers"


def test_unsupported_arch_raises():
    class Fake(nn.Module):
        model_type = "mamba"
        layers = []

    with pytest.raises(NotImplementedError):
        spec_for(Fake())


def test_block_capture_reproduces_forward(tiny_llama, tokens):
    """Running blocks on captured inputs must reproduce the model's hidden states."""
    h, mask = capture_block0_inputs(tiny_llama, tokens, batch_size=3)
    assert h.shape == (8, 64, 128)
    for _, block, _, m in iterate_blocks(tiny_llama, tokens, ["mlp.down_proj"], 4):
        h = block(h, m)
    ref = tiny_llama.model.norm(h)
    full = tiny_llama.model(tokens)
    assert mx.allclose(ref, full, atol=1e-4)


def test_fisher_has_every_layer(tiny_llama, tokens):
    paths = [f"model.layers.0.mlp.up_proj", "model.layers.1.self_attn.q_proj"]
    f = fisher_diagonal(tiny_llama, tokens[:2], paths)
    assert set(f) == set(paths)
    assert all((v >= 0).all().item() for v in f.values())
    assert f[paths[0]].shape == tiny_llama.model.layers[0].mlp.up_proj.weight.shape
    # model is unfrozen again afterwards
    assert len(tiny_llama.trainable_parameters()) > 0


def test_curves_shape_and_monotonic(tiny_llama, tokens):
    cfg = Config(bits=(2, 4, 8), **FAST)
    c = compute_curves(tiny_llama, tokens, cfg)
    assert len(c.errors) == 2 * 7
    for path, e in c.errors.items():
        assert set(e) == {2, 4, 8}
        assert e[2] > e[4] > e[8] >= 0, path


def test_curves_do_not_modify_model(tiny_llama, tokens):
    before = tiny_llama(tokens)
    compute_curves(tiny_llama, tokens, Config(bits=(2, 4), **FAST))
    assert mx.allclose(before, tiny_llama(tokens), atol=1e-5)


def test_curves_roundtrip(tmp_path, tiny_llama, tokens):
    c = compute_curves(tiny_llama, tokens, Config(bits=(3, 4), sensitivity="output", **FAST))
    c.save(tmp_path / "c.json")
    c2 = Curves.load(tmp_path / "c.json")
    assert c2.errors == c.errors and c2.params == c.params


@pytest.mark.parametrize("target", [3.0, 3.5, 4.5, 6.0])
def test_allocation_respects_budget(tiny_llama, tokens, target):
    c = compute_curves(tiny_llama, tokens, Config(bits=(2, 3, 4, 6, 8), **FAST))
    a = allocate(c, target)
    assert a["bpw"] <= target + 1e-9
    assert set(a["plan"].values()) <= {2, 3, 4, 6, 8}


def test_budget_tradeoff_monotone(tiny_llama, tokens):
    c = compute_curves(tiny_llama, tokens, Config(bits=(2, 3, 4, 6, 8), **FAST))
    errs = [allocate(c, t)["predicted_error"] for t in (3.0, 3.5, 4.0, 5.0, 7.0, 9.0)]
    assert all(a >= b for a, b in zip(errs, errs[1:]))


def test_apply_plan_awq_only_is_function_preserving(tokens):
    model = make_tiny_llama()
    before = model(tokens)
    plan = uniform_plan(model, 3, 64)
    apply_plan(model, tokens, plan, Config(bits=(3,), clip=False, **FAST))
    after = model(tokens)
    assert mx.allclose(before, after, atol=2e-3, rtol=1e-3)


def test_end_to_end_mixed_precision_layers(tokens):
    model = make_tiny_llama()
    cfg = Config(bits=(2, 4, 8), **FAST)
    c = compute_curves(model, tokens, cfg)
    a = allocate(c, 4.0)
    apply_plan(model, tokens, a["plan"], cfg)
    quantize_in_memory(model, a["plan"])
    seen = set()
    for name, m in model.named_modules():
        if name in a["plan"]:
            assert isinstance(m, nn.QuantizedLinear)
            assert m.bits == a["plan"][name] and m.group_size == 64
            seen.add(m.bits)
    assert len(seen) >= 2, "expected a genuinely mixed allocation"
    out = model(tokens)
    assert mx.isfinite(out).all().item()


def test_more_bits_lower_loss(tokens):
    """Sanity: the quantized model gets closer to fp as the budget grows."""
    def loss_at(target):
        model = make_tiny_llama()
        ref = model(tokens)
        cfg = Config(bits=(2, 3, 4, 8), **FAST)
        c = compute_curves(model, tokens, cfg)
        plan = allocate(c, target)["plan"]
        apply_plan(model, tokens, plan, cfg)
        quantize_in_memory(model, plan, embed_bits=8)
        return mx.mean((model(tokens) - ref) ** 2).item()

    assert loss_at(3.0) > loss_at(5.0) > loss_at(9.0)


def test_mlx_lm_save_and_load_roundtrip(tmp_path, tokens):
    """Per-layer bit-widths survive mlx-lm's save/load path."""
    from mlx_lm.utils import load_model, save_config, save_model

    from bitbudget.pipeline import quantize_with_plan

    model = make_tiny_llama()
    config = {
        "model_type": "llama", "hidden_size": 128, "num_hidden_layers": 2,
        "intermediate_size": 256, "num_attention_heads": 4,
        "num_key_value_heads": 2, "rms_norm_eps": 1e-5, "vocab_size": 256,
        "tie_word_embeddings": True,
    }
    plan = uniform_plan(model, 4, 64)
    plan["model.layers.0.mlp.down_proj"] = 2
    plan["model.layers.1.self_attn.v_proj"] = 8
    model, config = quantize_with_plan(model, config, plan, 64, embed_bits=6)
    ref = model(tokens)
    save_model(tmp_path, model)
    save_config(config, tmp_path / "config.json")
    loaded, _ = load_model(tmp_path)
    assert loaded.model.layers[0].mlp.down_proj.bits == 2
    assert loaded.model.layers[1].self_attn.v_proj.bits == 8
    assert loaded.model.embed_tokens.bits == 6
    assert mx.allclose(loaded(tokens), ref, atol=1e-5)


def test_chunk_tokens():
    t = mx.arange(1000)
    rows = chunk_tokens(t, 100, 5, seed=0)
    assert rows.shape == (5, 100)
    starts = sorted(int(r[0].item()) for r in rows)
    assert all(s % 100 == 0 for s in starts) and len(set(starts)) == 5
