#!/usr/bin/env bash
# Full comparison on Apple silicon: baselines vs BitBudget across memory budgets.
#
#   MODEL=Qwen/Qwen2.5-0.5B ./scripts/run_sweep.sh
#   MODEL=Qwen/Qwen2.5-1.5B BUDGETS="3.0 3.5 4.0" ./scripts/run_sweep.sh
#
# Every method is evaluated with the same command, on the same WikiText-2 test
# split, with bits-per-weight measured the same way (mlx-lm's
# compute_bits_per_weight), and the results land in one JSONL file that
# `bitbudget plot` turns into a Pareto frontier.
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen2.5-0.5B}
NAME=${NAME:-$(basename "$MODEL")}
OUT=${OUT:-runs/$NAME}
BUDGETS=${BUDGETS:-"3.0 3.5 4.0 4.5"}
EVAL_ARGS=${EVAL_ARGS:---bench}
RES="$OUT/results.jsonl"
mkdir -p "$OUT"

ev() {  # ev <model_dir> <method> <tag> [extra eval args...]
  if grep -q "\"tag\": \"$3\"" "$RES" 2>/dev/null; then
    echo "== $3 already evaluated, skipping"; return
  fi
  echo "== eval $3"
  bitbudget eval --model "$1" --method "$2" --tag "$3" --reference "$MODEL" \
    --results "$RES" $EVAL_ARGS "${@:4}"
}

# 1. full-precision reference --------------------------------------------------
ev "$MODEL" fp16 fp16

# 2. uniform round-to-nearest (mlx-lm default quantization) ---------------------
for b in 3 4; do
  [ -d "$OUT/rtn$b" ] || mlx_lm.convert --hf-path "$MODEL" -q --q-bits "$b" \
    --q-group-size 64 --mlx-path "$OUT/rtn$b"
  ev "$OUT/rtn$b" rtn "rtn$b"
done

# 3. mlx-lm fixed mixed-precision recipes ---------------------------------------
for r in mixed_3_4 mixed_3_6; do
  [ -d "$OUT/$r" ] || mlx_lm.convert --hf-path "$MODEL" -q --quant-predicate "$r" \
    --mlx-path "$OUT/$r"
  ev "$OUT/$r" mlx-mixed "$r"
done

# 4. mlx-lm AWQ (uniform bits) ---------------------------------------------------
for b in 3 4; do
  [ -d "$OUT/mlxawq$b" ] || mlx_lm.awq --model "$MODEL" --bits "$b" \
    --mlx-path "$OUT/mlxawq$b"
  ev "$OUT/mlxawq$b" mlx-awq "mlxawq$b"
done

# 5. mlx-lm dynamic quantization (two levels, gradient sensitivity) --------------
for t in 3.75 4.25; do
  [ -d "$OUT/dyn$t" ] || mlx_lm.dynamic_quant --model "$MODEL" --low-bits 3 \
    --high-bits 4 --target-bpw "$t" --mlx-path "$OUT/dyn$t"
  ev "$OUT/dyn$t" mlx-dynamic "dyn$t"
done

# 6. BitBudget: one sensitivity pass, then every budget reuses the curves --------
for t in $BUDGETS; do
  [ -d "$OUT/bb$t" ] || bitbudget quantize --model "$MODEL" --target-bpw "$t" \
    --curves "$OUT/curves.json" --out "$OUT/bb$t"
  ev "$OUT/bb$t" bitbudget "bb$t" --target-bpw "$t"
done

# 7. ablations --------------------------------------------------------------------
#    (a) our AWQ at uniform bits    (b) allocation without AWQ/clipping
for b in 3 4; do
  [ -d "$OUT/bbu$b" ] || bitbudget quantize --model "$MODEL" --bits "$b" \
    --out "$OUT/bbu$b"
  ev "$OUT/bbu$b" bitbudget-uniform "bbu$b"
done
for t in 3.5; do
  [ -d "$OUT/bbnoawq$t" ] || bitbudget quantize --model "$MODEL" --target-bpw "$t" \
    --no-awq --no-clip --curves "$OUT/curves_noawq.json" --out "$OUT/bbnoawq$t"
  ev "$OUT/bbnoawq$t" bitbudget-noawq "bbnoawq$t" --target-bpw "$t"
done

# 8. figures ------------------------------------------------------------------------
bitbudget plot --results "$RES" --out "$OUT/figures" --title "$NAME" \
  --plan $(ls -d "$OUT"/bb[0-9]*/bitbudget_plan.json)
echo "done: $RES and $OUT/figures/"
