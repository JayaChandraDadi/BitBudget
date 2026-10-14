#!/usr/bin/env bash
# Toy-scale validation that runs on any machine (no downloads needed).
#   ./scripts/run_toy.sh
# On Apple silicon you can skip the JAX step: toy_validation.py trains with MLX
# when --weights is omitted.
set -euo pipefail
OUT=${OUT:-results/toy}
STEPS=${STEPS:-3000}

if [ ! -f "$OUT/toy_model.npz" ]; then
  (cd examples && python train_toy_jax.py --steps "$STEPS" --out "../$OUT" --check)
fi
python examples/toy_validation.py --weights "$OUT/toy_model.npz" --out "$OUT"

# also exercise the real CLI path on the toy model
python examples/export_toy_model.py --weights "$OUT/toy_model.npz" \
  --config "$OUT/toy_config.json" --out "$OUT/hf_model"
