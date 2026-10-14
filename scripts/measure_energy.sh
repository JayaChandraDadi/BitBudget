#!/usr/bin/env bash
# Energy per generated token on Apple silicon using powermetrics (needs sudo).
#
#   ./scripts/measure_energy.sh runs/Qwen2.5-0.5B/bb3.5 bb3.5 runs/Qwen2.5-0.5B/results.jsonl
#
# 1. records ~10 s of idle package power (close other apps, plug in power),
# 2. records power while `bitbudget bench` runs,
# 3. subtracts idle power and divides by the generated tokens.
# The J/token value is appended to the results file under the same tag, so
# `bitbudget plot` can draw an energy-vs-size frontier.
set -euo pipefail

MODEL=$1
TAG=$2
RES=$3
TMP=$(mktemp -d)
SAMPLERS=cpu_power,gpu_power,ane_power

echo "Measuring idle power (10 s)..."
sudo powermetrics -i 200 -n 50 --samplers "$SAMPLERS" > "$TMP/idle.txt"

sudo powermetrics -i 200 --samplers "$SAMPLERS" > "$TMP/power.txt" &
PM=$!
trap 'sudo kill $PM 2>/dev/null || true' EXIT
sleep 1

bitbudget bench --model "$MODEL" --prompt-len 512 --gen-len 256 --runs 5 \
  --json "$TMP/bench.json"
sleep 1
sudo kill "$PM" || true

bitbudget energy --log "$TMP/power.txt" --idle-log "$TMP/idle.txt" \
  --bench-json "$TMP/bench.json" --results "$RES" --tag "$TAG"
echo "raw logs in $TMP"
