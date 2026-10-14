import json

import pytest

from bitbudget.energy import energy_per_token, parse_powermetrics
from bitbudget.plotting import load_results, plot_bitmap, plot_pareto

SAMPLE = """Machine model: Mac15,12
*** Sampled system activity (Tue Oct 13 17:45:00 2026 -0500) (200.11ms elapsed) ***

**** Processor usage ****
CPU Power: 1200 mW
GPU Power: 5000 mW
ANE Power: 0 mW
Combined Power (CPU + GPU + ANE): 6200 mW

*** Sampled system activity (Tue Oct 13 17:45:01 2026 -0500) (200.05ms elapsed) ***
CPU Power: 1000 mW
GPU Power: 4000 mW
ANE Power: 0 mW
Combined Power (CPU + GPU + ANE): 5000 mW

*** Sampled system activity (Tue Oct 13 17:45:09 2026 -0500) (200.05ms elapsed) ***
GPU Power: 1 mW
Combined Power (CPU + GPU + ANE): 100 mW
"""


def test_parse_powermetrics():
    s = parse_powermetrics(SAMPLE)
    assert len(s) == 3
    assert s[0]["combined_mw"] == 6200 and s[0]["gpu_mw"] == 5000
    assert s[0]["t"] is not None and s[1]["t"] - s[0]["t"] == pytest.approx(1.0)


def test_energy_windowing():
    s = parse_powermetrics(SAMPLE)
    t0 = s[0]["t"]
    r = energy_per_token(s, t0, t0 + 2.0, tokens=100, idle_mw=600)
    assert r["windowed"] and r["samples"] == 2
    # mean 5600 mW, minus 600 idle = 5 W, for 2 s = 10 J over 100 tokens
    assert r["j_per_token"] == pytest.approx(0.1)


def test_results_merge_and_plots(tmp_path):
    rows = [
        {"tag": "fp", "method": "fp16", "bpw": 16, "ppl": 10.0},
        {"tag": "r3", "method": "rtn", "bpw": 3.5, "ppl": 20.0},
        {"tag": "r4", "method": "rtn", "bpw": 4.5, "ppl": 12.0},
        {"tag": "b3", "method": "bitbudget", "bpw": 3.5, "ppl": 15.0},
        {"tag": "b3", "j_per_token": 0.2},
    ]
    p = tmp_path / "r.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    res = load_results(p)
    assert len(res) == 4
    assert [r for r in res if r["tag"] == "b3"][0]["j_per_token"] == 0.2
    assert plot_pareto(res, tmp_path / "p.png").exists()
    plan = {f"model.layers.{i}.self_attn.q_proj": 2 + i for i in range(3)}
    plan.update({f"model.layers.{i}.mlp.down_proj": 8 for i in range(3)})
    assert plot_bitmap(plan, tmp_path / "b.png").exists()
