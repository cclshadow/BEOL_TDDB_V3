# BEOL TDDB Reliability Prediction

This repository predicts die-level BEOL time-dependent dielectric breakdown
(TDDB) reliability from wafer metrology.  It is designed for **product-specific
calibration**: a model is trained on historical labelled wafers from one process
or product family, then scores every valid die on a new wafer.

The repository supports both physics-informed lifetime models and supervised
spatial ML models.  The latter explicitly address the intended deployment case:
only a limited subset of wafer sites is measured, while reliability is required
for every die.

## Features

### Physics-informed pipeline

The original three-layer engine is retained.

1. A unit model maps local via-line (`Space`) and line-line (`MS`) spacing to
   Weibull parameters.
2. A wafer engine spatially interpolates metrology and aggregates microstructure
   failure using a weakest-link lifetime calculation.
3. A threshold turns continuous TTF into a good/bad decision.

Available alternatives are DPM physics models (`PowerLaw`, `SqrtE`, `InverseE`),
scaled and delta-calibrated DPM variants, a GPR Weibull surrogate, and a simple
`Linear` spacing baseline.  The baseline matters: a complex model should beat it
before being trusted.

### Spatial ML backends

The supervised models use historical labels and predict a bad-die probability
for every valid die.

| Backend | Idea | Best use |
| --- | --- | --- |
| `gp_boost` | Reconstruct sparse metrology with Gaussian processes, then classify with LightGBM or sklearn boosting. | First production-quality ML baseline; includes GP uncertainty. |
| `graphsage` | Pass local information across a distance-neighbour die graph. | Local spatial excursions and irregular wafer edges. |
| `spatial_transformer` | Cross-attend from all die queries to the measured die tokens only. | Non-local wafer effects, once sufficient labelled wafers are available. |

The ML path currently consumes `Space.csv`, `MS.csv`, `VCD.csv`, `OVL.csv`, and
`CBCD.csv`.  Missing metrology is represented by an explicit measurement mask;
it is never silently substituted with a physical zero.

## Repository layout

```text
.
├── data/                         # lot_*/csv/wafer_* input data
├── configs/                      # trained physics-model configurations
├── src/                          # physics engine and spatial_ml package
├── docs/                         # legacy guide and its editable source
├── create_split.py               # lot-level train/validation/test split
├── train.py / inference.py       # physics-informed workflow
├── train_spatial_ml.py           # supervised spatial ML calibration
├── inference_spatial_ml.py       # supervised spatial ML scoring
├── SPATIAL_ML.md                 # detailed ML-backend notes
├── requirements.txt              # core dependencies
└── requirements-ml.txt           # optional LightGBM and PyTorch dependencies
```

The prior nested project directory has been removed.  Run all commands from this
repository root.

## Data contract

The data scanner requires this exact structure:

```text
data/
└── lot_001/
    └── csv/
        └── wafer_01/
            ├── Space.csv
            ├── MS.csv
            ├── ExistenceClass.csv
            ├── VCD.csv            # required by spatial ML
            ├── OVL.csv            # required by spatial ML
            └── CBCD.csv           # required by spatial ML
```

All CSVs are same-sized two-dimensional die maps.  A `0`, `NaN`, or `Inf` in
`ExistenceClass.csv` denotes background, not a die.  For valid dies, class `3`
means good/pass and classes `1` or `2` mean bad/intercept.

Split data by **lot**, never individual wafer: wafers from one lot share process
conditions and would otherwise leak information between train and test.

## Setup

Python 3.10+ is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GraphSAGE, the spatial Transformer, or native LightGBM:

```powershell
python -m pip install -r requirements-ml.txt
```

## Physics-informed workflow

Create a reproducible lot-level split:

```powershell
python create_split.py --dataset-path data --output split_manifest.json
```

Train a physics model; the manifest lets the script tune on train/validation
lots and report a held-out test score:

```powershell
python train.py --pipeline-type DPM --model-type SqrtE --train-deltas `
  --delta-l2 0.01 --manifest split_manifest.json --n-trials 40 `
  --save-path configs/DPM_SqrtE_deltas
```

Run it on the held-out test wafers:

```powershell
python inference.py --config-path configs/DPM_SqrtE_deltas/config.json `
  --manifest split_manifest.json --split test --save-path output/DPM_SqrtE
```

Each inferred wafer receives `ttf.csv`, `binary_label.csv`, and a wafer map.

## Spatial ML workflow

Calibrate the GP-plus-boost baseline using only 20% of valid die locations as
measured metrology sites.  Use the same sampling budget during calibration and
evaluation.

```powershell
python train_spatial_ml.py --backend gp_boost --manifest split_manifest.json `
  --measurement-fraction 0.20 --save-dir configs/spatial_gp_boost
```

Train the neural alternatives after installing `requirements-ml.txt`:

```powershell
python train_spatial_ml.py --backend graphsage --manifest split_manifest.json `
  --measurement-fraction 0.20 --save-dir configs/spatial_graphsage

python train_spatial_ml.py --backend spatial_transformer --manifest split_manifest.json `
  --measurement-fraction 0.20 --save-dir configs/spatial_transformer
```

Score new wafers:

```powershell
python inference_spatial_ml.py `
  --config-path configs/spatial_gp_boost/config.json `
  --test-path data --save-path output/spatial_gp_boost
```

For an honest held-out score, use the manifest rather than scanning the entire
dataset:

```powershell
python inference_spatial_ml.py `
  --config-path configs/spatial_gp_boost/config.json `
  --manifest split_manifest.json --split test `
  --save-path output/spatial_gp_boost_test
```

The ML inference output contains `reliability_score.csv`, `binary_label.csv`,
and `measurement_mask.csv`.  The provided mask generator is a deterministic
test-budget simulator.  Replace it with the actual production test-structure
layout before deployment.

## Evaluation guidance

Evaluate only on held-out lots and use the actual sparse-measurement pattern.
F-beta is useful when missed bad dies are expensive, but it is not sufficient on
its own.  Also report escape rate at a fixed overkill budget, PR-AUC, and
probability calibration.  Compare every candidate against the Linear baseline.

## Legacy material

The current detailed guide is [docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md).
The original operating guide is retained in [docs/](docs/README.md), clearly
marked as legacy; this README is the current source of truth for commands and
repository structure.
