#!/usr/bin/env bash
# Reproduce the GSM-Symbolic experiment: Inter-Cascade vs. the Jung et al.
# baseline. Needs OPENAI_API_KEY and GEMINI_API_KEY (or a Vertex project_id in
# the configs), plus `pip install -e ".[all]"`.
set -euo pipefail
cd "$(dirname "$0")/.."

CFG=configs/paper/gsm_symbolic.yaml
CFG_BASELINE=configs/paper/gsm_symbolic_jung_baseline.yaml
DATA=data/gsm_symbolic
OUT=results/gsm_symbolic

mkdir -p "$OUT"

# 1. Data: 1250 calibration / 11250 test
inter-cascade prepare --dataset gsm_symbolic --output-dir "$DATA"

# 2. Calibration runs + thresholds (shared by both pipelines)
inter-cascade calibrate --config "$CFG" --scan

# 3. Inter-Cascade (ours)
inter-cascade run --config "$CFG" \
  --input "$DATA/test_gsm_symbolic.json" \
  --output "$OUT/inter_cascade.json" --resume

# 4. Standard cascade baseline (same thresholds, no repository)
inter-cascade run --config "$CFG_BASELINE" \
  --input "$DATA/test_gsm_symbolic.json" \
  --output "$OUT/jung_baseline.json" --resume

# 5. Compare
echo "=== Inter-Cascade ===";    inter-cascade stats --results "$OUT/inter_cascade.json"
echo "=== Jung baseline ===";    inter-cascade stats --results "$OUT/jung_baseline.json"
