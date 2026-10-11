# BEOL TDDB Reliability Prediction Guide

## Purpose

This project predicts die-level BEOL TDDB reliability from wafer metrology. It
is calibrated separately for each product or process family: historical labelled
wafers train the model, and a new wafer receives a prediction for every valid
die. The intended deployment case is sparse metrology: only selected test sites
are measured, while reliability is required across the whole wafer.

This is the maintained Markdown successor to the legacy Wafer Mapping Tool
guide. It retains the original physics workflow and adds supervised spatial ML.

## Data contract

```text
data/lot_001/csv/wafer_01/
  Space.csv  MS.csv  ExistenceClass.csv
  VCD.csv    OVL.csv CBCD.csv
```

All files are same-shape two-dimensional die maps. `ExistenceClass.csv`
identifies valid dies: `0`, `NaN`, and `Inf` are background; `1`/`2` are bad
dies; `3` is a good die. `Space` is via-line spacing and `MS` is line-line
spacing. The physics pipeline uses the first two channels; spatial ML uses all
five channels.

Split complete **lots**, never individual wafers, to prevent process-correlation
leakage:

```powershell
python create_split.py --dataset-path data --output split_manifest.json
```

## Physics-informed workflow

The original engine has three layers:

```text
local Space/MS -> unit Weibull model -> wafer weakest-link TTF -> threshold
```

Layer 1 offers DPM `PowerLaw`, `SqrtE`, and `InverseE` models, their scaled and
delta-calibrated variants, a simulation-derived GPR Weibull surrogate, and a
`Linear` (`Space + MS`) baseline. Layer 2 uses spatial GPR/Kriging and a
weakest-link aggregation to form die TTF. Layer 3 applies `TTF >= threshold`
for good/pass. The baseline is important: a complex model should beat it.

```powershell
python train.py --pipeline-type DPM --model-type SqrtE --train-deltas `
  --delta-l2 0.01 --manifest split_manifest.json --n-trials 40 `
  --save-path configs/DPM_SqrtE_deltas

python inference.py --config-path configs/DPM_SqrtE_deltas/config.json `
  --manifest split_manifest.json --split test --save-path output/DPM_SqrtE_test
```

Physics inference writes `ttf.csv`, `binary_label.csv`, and `wafer_maps.png`.

## Spatial ML workflow

The supervised spatial backends output `P(bad)` per valid die. A wafer is a
spatial set/graph, not a single vector. Conceptually each die supplies:

```text
[Space, MS, VCD, OVL, CBCD, x, y, radius, measurement_mask]
```

`x` and `y` are die coordinates centred on the wafer. `measurement_mask`
distinguishes truly unmeasured metrology from a physical value of zero.
`--measurement-fraction` is a deterministic sparse-test simulator; replace its
mask generator with the real test-structure layout before deployment.

| Backend | Method | Appropriate role |
| --- | --- | --- |
| `gp_boost` | Per-channel spatial GP reconstruction plus uncertainty, then LightGBM/sklearn boosting. | First ML baseline; interpretable and uncertainty-aware. |
| `graphsage` | Distance-neighbour die graph with message passing. | Strong local spatial prior. |
| `spatial_transformer` | All dies query measured-die key/value tokens through cross-attention. | Non-local patterns after smaller baselines are established. |

```powershell
python train_spatial_ml.py --backend gp_boost --manifest split_manifest.json `
  --measurement-fraction 0.20 --save-dir configs/spatial_gp_boost

python inference_spatial_ml.py `
  --config-path configs/spatial_gp_boost/config.json `
  --manifest split_manifest.json --split test `
  --save-path output/spatial_gp_boost_test
```

Spatial inference writes `reliability_score.csv` (`1 - P(bad)`),
`binary_label.csv`, and `measurement_mask.csv`; background remains `NaN`.

## Evaluation

Use held-out lots and exactly the measurement budget/layout planned for
deployment. Compare model performance versus measurement budget (for example
5%, 10%, 20%, 40%) and report escape rate at a fixed overkill limit, PR-AUC,
probability calibration, and F-beta. Do not score a sparse-metrology model with
full-wafer measurements.

## Environment and examples

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-ml.txt
```

`run_examples/training.sh` and `run_examples/inference.sh` contain runnable
physics and spatial ML examples. The archived PDF/DOCX retain the predecessor
layout; this guide is the current command reference.
