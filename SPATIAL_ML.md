# Spatial ML backends

This module trains a **product-specific** reliability model from labelled
wafers.  It is separate from the original physics-only CLI because it learns
weights from the historical labels and saves a model artifact.

## Models

| Backend | Mechanism | Required package |
| --- | --- | --- |
| `gp_boost` | Per-wafer Gaussian-process metrology reconstruction, followed by LightGBM. Falls back to sklearn HistGradientBoosting. | optional `lightgbm` |
| `graphsage` | A small distance-neighbour GraphSAGE classifier. Node features include GP reconstruction and uncertainty. | `torch` |
| `spatial_transformer` | Sparse-measurement encoder / all-die cross-attention decoder. It only uses measured dies as keys and values. | `torch` |

The input channels are `Space`, `MS`, `VCD`, `OVL`, and `CBCD`; `ExistenceClass`
is the target (`3 = good`, `1/2 = bad`).  Positions are centred die-grid
coordinates.  Invalid wafer cells never become model tokens.

## Calibration

First create a lot-level split if one does not exist:

```powershell
py create_split.py --dataset-path data --output split_manifest.json
```

Then fit one model.  The measurement fraction must match the intended
production test budget; it is simulated independently for each wafer during
calibration.

```powershell
py train_spatial_ml.py --backend gp_boost --manifest split_manifest.json `
  --measurement-fraction 0.20 --save-dir configs/spatial_gp_boost
```

For GNN/Transformer, install the optional dependencies first:

```powershell
py -m pip install -r requirements-ml.txt
py train_spatial_ml.py --backend graphsage --manifest split_manifest.json `
  --measurement-fraction 0.20 --save-dir configs/spatial_graphsage
```

## Inference

```powershell
py inference_spatial_ml.py --config-path configs/spatial_gp_boost/config.json `
  --test-path data --save-path output/spatial_gp_boost
```

To infer only a held-out split, use the same manifest used during calibration:

```powershell
py inference_spatial_ml.py --config-path configs/spatial_gp_boost/config.json `
  --manifest split_manifest.json --split test --save-path output/spatial_gp_boost_test
```

Each wafer output contains `reliability_score.csv` (one minus predicted failure
probability), `binary_label.csv`, and `measurement_mask.csv`.  The current mask
is a deterministic random **deployment simulator**.  Before production, replace
it with the actual test-plan mask; this is deliberately isolated in
`spatial_ml.data.measured_mask`.

## Evaluation rule

Never score a model with full-wafer inputs when deployment has sparse inputs.
Use a lot-held-out test split and the identical measurement mask/budget in
train, validation, and test.  Compare escape rate at a fixed overkill budget,
PR-AUC, probability calibration, and F-beta rather than F-beta alone.
