import mlx.core as mx
import mlx.nn as nn
import pytest

from bitbudget.awq import fold_scales, search_clip, search_scales
from bitbudget.quant import effective_bpw, qdq
from bitbudget.sensitivity import (
    fisher_error,
    rescale_fisher_cols,
    rescale_fisher_rows,
)


def salient_inputs(n=512, d=256, n_salient=4, boost=30.0, seed=0):
    """Activations where a few channels are much larger than the rest -- the
    regime AWQ is designed for (LLM activations have such outlier channels)."""
    mx.random.seed(seed)
    x = mx.random.normal((n, d))
    boost_vec = mx.ones((d,))
    boost_vec[:n_salient] = boost
    return x * boost_vec


def test_qdq_error_decreases_with_bits():
    mx.random.seed(0)
    w = mx.random.normal((64, 256))
    errs = [mx.mean((qdq(w, b, 64) - w) ** 2).item() for b in (2, 3, 4, 5, 6, 8)]
    assert all(a > b for a, b in zip(errs, errs[1:]))


def test_effective_bpw():
    assert effective_bpw(4, 64, mx.bfloat16) == 4.5
    assert effective_bpw(3, 32, mx.float16) == 4.0
    assert effective_bpw(2, 64, mx.float32) == 3.0


def test_scale_search_beats_rtn_on_salient_channels():
    mx.random.seed(0)
    lin = nn.Linear(256, 128, bias=False)
    x = salient_inputs()
    w0 = lin.weight
    ref = lin(x)
    rtn_err = mx.mean((x @ qdq(w0, 3, 64).T - ref) ** 2).item()
    s, alpha, err = search_scales([lin], [3], lin, x, 64, n_grid=20)
    assert alpha > 0
    assert err < 0.8 * rtn_err
    # weights restored after the search
    assert mx.array_equal(lin.weight, w0)


def test_scale_search_never_worse_than_rtn():
    mx.random.seed(3)
    lin = nn.Linear(128, 64, bias=False)
    x = mx.random.normal((256, 128))  # no salient channels
    ref = lin(x)
    rtn_err = mx.mean((x @ qdq(lin.weight, 4, 64).T - ref) ** 2).item()
    _, _, err = search_scales([lin], [4], lin, x, 64, n_grid=10)
    assert err <= rtn_err * (1 + 1e-6)


def test_mixed_bits_in_one_group():
    mx.random.seed(1)
    a, b = nn.Linear(256, 64, bias=False), nn.Linear(256, 64, bias=False)
    x = salient_inputs()

    def objective(t):
        return mx.concatenate([a(t), b(t)], axis=-1)

    s, _, err_mixed = search_scales([a, b], [2, 8], objective, x, 64, n_grid=10)
    _, _, err_low = search_scales([a, b], [2, 2], objective, x, 64, n_grid=10)
    assert err_mixed < err_low


@pytest.mark.parametrize("prev_kind", ["rmsnorm", "linear"])
def test_fold_preserves_function(prev_kind):
    mx.random.seed(0)
    d = 128
    if prev_kind == "rmsnorm":
        prev = nn.RMSNorm(d)
        prev.weight = mx.random.uniform(0.5, 1.5, (d,))
    else:
        prev = nn.Linear(64, d, bias=True)
    nxt = [nn.Linear(d, 32), nn.Linear(d, 48)]
    x = mx.random.normal((16, 64 if prev_kind == "linear" else d))
    before = [l(prev(x)) for l in nxt]
    s = mx.random.uniform(0.2, 5.0, (d,))
    fold_scales(prev, nxt, s)
    after = [l(prev(x)) for l in nxt]
    for u, v in zip(before, after):
        assert mx.allclose(u, v, atol=1e-4, rtol=1e-4)


def test_clip_never_increases_output_error():
    mx.random.seed(2)
    w = mx.random.normal((64, 256))
    w[:, 5] = w[:, 5] * 20  # outliers inside some groups
    x = mx.random.normal((512, 256))
    for bits in (2, 3, 4):
        base = mx.sum(((qdq(w, bits, 64) - w) @ x.T) ** 2).item()
        wc = search_clip(w, x, bits, 64)
        clipped = mx.sum(((qdq(wc, bits, 64) - w) @ x.T) ** 2).item()
        assert clipped <= base * (1 + 1e-5)


def test_fisher_rescaling_is_invariant():
    """1/2 sum F dW^2 must not change when a scale is folded into W."""
    mx.random.seed(4)
    w = mx.random.normal((32, 64))
    f = mx.random.uniform(0, 1, (32, 64))
    dq = qdq(w, 3, 64)
    s_col = mx.random.uniform(0.5, 2.0, (64,))
    s_row = mx.random.uniform(0.5, 2.0, (32,))
    e0 = fisher_error(f, w, dq)
    e_col = fisher_error(rescale_fisher_cols(f, s_col), w * s_col, dq * s_col)
    e_row = fisher_error(rescale_fisher_rows(f, s_row), w / s_row[:, None], dq / s_row[:, None])
    assert e_col == pytest.approx(e0, rel=1e-4)
    assert e_row == pytest.approx(e0, rel=1e-4)
