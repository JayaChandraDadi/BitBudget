"""Command line interface.

    bitbudget quantize --model Qwen/Qwen2.5-0.5B --target-bpw 3.5 --out runs/q05/bb3.5
    bitbudget eval     --model runs/q05/bb3.5 --method bitbudget --results runs/q05/results.jsonl
    bitbudget bench    --model runs/q05/bb3.5 --json runs/q05/bench_bb3.5.json
    bitbudget energy   --log power.txt --bench-json runs/q05/bench_bb3.5.json --tag bb3.5 ...
    bitbudget plot     --results runs/q05/results.jsonl --plan runs/q05/bb3.5/bitbudget_plan.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _bits(s: str) -> tuple[int, ...]:
    return tuple(int(b) for b in s.split(","))


def _append(path: str | None, record: dict) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(record) + "\n")


# --------------------------------------------------------------------------- #
def cmd_quantize(a):
    import mlx.core as mx
    from mlx_lm.utils import load, save

    from .calibration import load_calibration
    from .pipeline import (
        Config,
        Curves,
        allocate,
        apply_plan,
        compute_curves,
        quantize_with_plan,
        uniform_plan,
    )

    cfg = Config(
        bits=_bits(a.bits),
        group_size=a.group_size,
        n_grid=a.n_grid,
        awq=not a.no_awq,
        clip=not a.no_clip,
        sensitivity=a.sensitivity,
        batch_size=a.batch_size,
    )
    t0 = time.time()
    model, tokenizer, config = load(a.model, return_config=True)
    calib = load_calibration(tokenizer, a.n_samples, a.seq_len, a.calib_file, a.seed)
    print(f"[bitbudget] calibration tokens: {tuple(calib.shape)}")

    if len(cfg.bits) == 1:
        plan = uniform_plan(model, cfg.bits[0], cfg.group_size)
        alloc = {"plan": plan, "target_bpw": None, "solver": "uniform"}
    else:
        if a.curves and Path(a.curves).exists():
            curves = Curves.load(a.curves)
            print(f"[bitbudget] loaded curves from {a.curves}")
            if list(curves.bits) != list(cfg.bits):
                raise SystemExit(f"curves have bits {curves.bits}, asked for {cfg.bits}")
        else:
            curves = compute_curves(
                model, calib, cfg, fisher_tokens=calib[: a.n_fisher]
            )
            if a.save_curves or a.curves:
                curves.save(a.save_curves or a.curves)
        alloc = allocate(curves, a.target_bpw, a.solver)
        print(
            f"[bitbudget] allocation: {alloc['bpw']:.3f} bpw (target {a.target_bpw}), "
            f"layers per bit-width {alloc['histogram']}"
        )
        plan = alloc["plan"]

    alphas = apply_plan(model, calib, plan, cfg)
    model, config = quantize_with_plan(
        model, config, plan, cfg.group_size, a.embed_bits, default_bits=a.embed_bits
    )
    config["bitbudget"] = {
        "target_bpw": alloc.get("target_bpw"),
        "linear_bpw": alloc.get("bpw"),
        "bits": list(cfg.bits),
        "awq": cfg.awq,
        "clip": cfg.clip,
        "sensitivity": cfg.sensitivity,
        "embed_bits": a.embed_bits,
    }
    save(a.out, a.model, model, tokenizer, config)
    report = {**alloc, "alphas": alphas, "seconds": round(time.time() - t0, 1),
              "peak_mem_gb": mx.get_peak_memory() / 1e9}
    (Path(a.out) / "bitbudget_plan.json").write_text(json.dumps(report, indent=1))
    print(f"[bitbudget] saved to {a.out} in {report['seconds']}s")


# --------------------------------------------------------------------------- #
def cmd_eval(a):
    import mlx.core as mx
    from mlx_lm.utils import load

    from .evaluate import (
        benchmark_generation,
        bits_per_weight,
        disk_bytes,
        kl_to_reference,
        load_eval_tokens,
        perplexity,
    )

    model, tokenizer, config = load(a.model, return_config=True)
    tokens = load_eval_tokens(tokenizer, a.dataset, a.seq_len, a.n_seqs)
    rec = {
        "tag": a.tag or Path(a.model).name,
        "method": a.method,
        "model": a.model,
        "bpw": round(bits_per_weight(model), 4),
        "linear_bpw": (config.get("bitbudget") or {}).get("linear_bpw"),
        "target_bpw": a.target_bpw,
        "dataset": a.dataset,
        "eval_tokens": int(tokens.size),
    }
    db = disk_bytes(a.model)
    if db:
        rec["disk_gb"] = db / 1e9
    rec["ppl"] = perplexity(model, tokens, a.batch_size)
    if a.reference:
        ref, _ = load(a.reference)
        rec["kl"] = kl_to_reference(model, ref, tokens[: a.kl_seqs], a.batch_size)
        del ref
    if a.bench:
        mx.clear_cache()
        bench = benchmark_generation(model, tokenizer, a.prompt_len, a.gen_len, a.runs)
        rec.update({k: v for k, v in bench.items() if k not in ("t_start", "t_end")})
    print(json.dumps(rec, indent=1))
    _append(a.results, rec)


def cmd_bench(a):
    from mlx_lm.utils import load

    from .evaluate import benchmark_generation

    model, tokenizer = load(a.model)
    res = benchmark_generation(model, tokenizer, a.prompt_len, a.gen_len, a.runs)
    res["model"] = a.model
    print(json.dumps(res, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))


def cmd_energy(a):
    from .energy import energy_per_token, parse_powermetrics

    bench = json.loads(Path(a.bench_json).read_text())
    samples = parse_powermetrics(Path(a.log).read_text())
    idle = a.idle_mw
    if a.idle_log:
        s = parse_powermetrics(Path(a.idle_log).read_text())
        idle = sum(x["combined_mw"] for x in s) / len(s)
    res = energy_per_token(
        samples, bench["t_start"], bench["t_end"], bench["tokens_total"], idle
    )
    print(json.dumps(res, indent=1))
    if a.results and a.tag:
        _append(a.results, {"tag": a.tag, "j_per_token": res["j_per_token"],
                            "power_w": res["net_power_w"]})


def cmd_plot(a):
    from .plotting import load_results, plot_bitmap, plot_pareto

    out = Path(a.out)
    if a.results:
        res = load_results(a.results)
        print(plot_pareto(res, out / "pareto_ppl.png", "ppl", title=a.title))
        if any("kl" in r for r in res):
            print(plot_pareto(res, out / "pareto_kl.png", "kl", title=a.title))
        if any("j_per_token" in r for r in res):
            print(plot_pareto(res, out / "pareto_energy.png", "j_per_token", title=a.title))
    for p in a.plan or []:
        plan = json.loads(Path(p).read_text())["plan"]
        name = Path(p).parent.name
        print(plot_bitmap(plan, out / f"bitmap_{name}.png", title=f"{name}: bits per layer"))


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(prog="bitbudget")
    sub = ap.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("quantize", help="salience-guided mixed-precision quantization")
    q.add_argument("--model", required=True, help="HF repo or local mlx/hf model path")
    q.add_argument("--out", required=True)
    q.add_argument("--target-bpw", type=float, default=3.5,
                   help="budget for the decoder linear layers, in effective bits "
                        "per weight including scale/bias overhead (4-bit gs64 = 4.5)")
    q.add_argument("--bits", default="2,3,4,6,8", help="candidate bit-widths")
    q.add_argument("--group-size", type=int, default=64)
    q.add_argument("--embed-bits", type=int, default=4,
                   help="bits for embeddings / lm_head (not part of the budget)")
    q.add_argument("--n-samples", type=int, default=32)
    q.add_argument("--seq-len", type=int, default=512)
    q.add_argument("--n-fisher", type=int, default=32)
    q.add_argument("--n-grid", type=int, default=20)
    q.add_argument("--batch-size", type=int, default=4)
    q.add_argument("--sensitivity", choices=["fisher", "output"], default="fisher")
    q.add_argument("--solver", choices=["dp", "lagrangian"], default="dp")
    q.add_argument("--no-awq", action="store_true", help="ablation: no AWQ scaling")
    q.add_argument("--no-clip", action="store_true", help="ablation: no clipping")
    q.add_argument("--curves", help="reuse (or create) a sensitivity-curves cache")
    q.add_argument("--save-curves", help="where to save sensitivity curves")
    q.add_argument("--calib-file", help="local calibration text file")
    q.add_argument("--seed", type=int, default=0)
    q.set_defaults(fn=cmd_quantize)

    e = sub.add_parser("eval", help="perplexity / KL / size (+ optional speed)")
    e.add_argument("--model", required=True)
    e.add_argument("--method", default="unknown")
    e.add_argument("--tag")
    e.add_argument("--target-bpw", type=float)
    e.add_argument("--dataset", default="wikitext2")
    e.add_argument("--seq-len", type=int, default=512)
    e.add_argument("--n-seqs", type=int, default=0, help="0 = whole test split")
    e.add_argument("--batch-size", type=int, default=4)
    e.add_argument("--reference", help="fp16 model for KL divergence")
    e.add_argument("--kl-seqs", type=int, default=64)
    e.add_argument("--bench", action="store_true")
    e.add_argument("--prompt-len", type=int, default=512)
    e.add_argument("--gen-len", type=int, default=128)
    e.add_argument("--runs", type=int, default=3)
    e.add_argument("--results", help="append the record to this JSONL file")
    e.set_defaults(fn=cmd_eval)

    b = sub.add_parser("bench", help="TTFT / decode tok/s / peak memory")
    b.add_argument("--model", required=True)
    b.add_argument("--prompt-len", type=int, default=512)
    b.add_argument("--gen-len", type=int, default=128)
    b.add_argument("--runs", type=int, default=3)
    b.add_argument("--json")
    b.set_defaults(fn=cmd_bench)

    en = sub.add_parser("energy", help="J/token from a powermetrics log")
    en.add_argument("--log", required=True)
    en.add_argument("--bench-json", required=True)
    en.add_argument("--idle-mw", type=float, default=0.0)
    en.add_argument("--idle-log")
    en.add_argument("--results")
    en.add_argument("--tag")
    en.set_defaults(fn=cmd_energy)

    p = sub.add_parser("plot", help="Pareto frontier + per-layer bit map")
    p.add_argument("--results")
    p.add_argument("--plan", nargs="*")
    p.add_argument("--out", default="figures")
    p.add_argument("--title")
    p.set_defaults(fn=cmd_plot)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
