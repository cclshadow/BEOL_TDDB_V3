#!/usr/bin/env bash
set -euo pipefail
PYTHON="${PYTHON:-python}"
MANIFEST="${MANIFEST:-split_manifest.json}"
SPLIT="${SPLIT:-test}"
WORKERS="${WORKERS:-8}"

case "${1:-help}" in
  physics)
    $PYTHON inference.py --config-path configs/example_dpm_sqrte_deltas/config.json \
      --manifest "$MANIFEST" --split "$SPLIT" --save-path output/examples/physics --num-workers "$WORKERS" ;;
  gp_boost|graphsage|spatial_transformer)
    backend="$1"
    $PYTHON inference_spatial_ml.py --config-path "configs/example_${backend}/config.json" \
      --manifest "$MANIFEST" --split "$SPLIT" --save-path "output/examples/${backend}" ;;
  *) echo "Usage: $0 {physics|gp_boost|graphsage|spatial_transformer}" >&2; exit 2 ;;
esac
