#!/usr/bin/env bash
set -euo pipefail
PYTHON="${PYTHON:-python}"
MANIFEST="${MANIFEST:-split_manifest.json}"
DATASET="${DATASET:-data}"
N_TRIALS="${N_TRIALS:-40}"
WORKERS="${WORKERS:-8}"
MEASUREMENT_FRACTION="${MEASUREMENT_FRACTION:-0.20}"

case "${1:-help}" in
  split)
    $PYTHON create_split.py --dataset-path "$DATASET" --output "$MANIFEST" ;;
  physics)
    $PYTHON train.py --pipeline-type DPM --model-type SqrtE --train-deltas --delta-l2 0.01 \
      --manifest "$MANIFEST" --n-trials "$N_TRIALS" --num-workers "$WORKERS" \
      --save-path configs/example_dpm_sqrte_deltas ;;
  spatial)
    for backend in gp_boost graphsage spatial_transformer; do
      $PYTHON train_spatial_ml.py --backend "$backend" --manifest "$MANIFEST" \
        --measurement-fraction "$MEASUREMENT_FRACTION" --save-dir "configs/example_${backend}"
    done ;;
  *) echo "Usage: $0 {split|physics|spatial}" >&2; exit 2 ;;
esac
