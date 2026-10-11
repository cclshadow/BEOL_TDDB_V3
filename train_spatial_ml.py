"""Train one supervised spatial reliability backend from a lot-level manifest.

Example:
  python train_spatial_ml.py --backend gp_boost --manifest split_manifest.json \
      --measurement-fraction 0.15 --save-dir configs/spatial_gp_boost
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.reliability_classifiers.binary_classifier import (
    evaluate_binary_threshold, optimize_binary_threshold,
)
from src.spatial_ml import (
    GPBoostBackend, GraphSAGEBackend, SpatialTransformerBackend, load_wafers,
)
from src.spatial_ml.data import measured_mask


BACKENDS = {
    "gp_boost": GPBoostBackend,
    "graphsage": GraphSAGEBackend,
    "spatial_transformer": SpatialTransformerBackend,
}


def masks_for(wafers, fraction, seed):
    return [measured_mask(w.n_dies, fraction, seed + i) for i, w in enumerate(wafers)]


def flat_predictions(model, wafers, masks):
    return np.concatenate([model.predict_proba(w, m) for w, m in zip(wafers, masks)])


def main():
    p = argparse.ArgumentParser(description="Train spatial ML reliability backend")
    p.add_argument("--backend", choices=BACKENDS, required=True)
    p.add_argument("--manifest", required=True, help="Lot-level manifest from create_split.py")
    p.add_argument("--measurement-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fscore-beta", type=float, default=2.0)
    p.add_argument("--save-dir", required=True)
    args = p.parse_args()

    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)
    train = load_wafers(manifest["train_wafers"])
    val = load_wafers(manifest["val_wafers"])
    train_masks = masks_for(train, args.measurement_fraction, args.seed)
    val_masks = masks_for(val, args.measurement_fraction, args.seed + 100_000)

    model = BACKENDS[args.backend]().fit(train, train_masks)
    train_y = np.concatenate([w.labels for w in train])
    val_y = np.concatenate([w.labels for w in val])
    # Existing threshold utility assumes a high score means good.  Probability
    # here means bad, so convert it to a reliability score.
    train_reliability = 1.0 - flat_predictions(model, train, train_masks)
    val_reliability = 1.0 - flat_predictions(model, val, val_masks)
    threshold, train_f2 = optimize_binary_threshold(train_reliability, train_y, args.fscore_beta)
    val_f2 = evaluate_binary_threshold(val_reliability, val_y, threshold, args.fscore_beta)

    out = Path(args.save_dir)
    out.mkdir(parents=True, exist_ok=True)
    suffix = ".pt" if args.backend != "gp_boost" else ".pkl"
    artifact = out / f"{args.backend}{suffix}"
    model.save(artifact)
    config = {
        "backend": args.backend, "artifact": artifact.name,
        "measurement_fraction": args.measurement_fraction, "seed": args.seed,
        "threshold": float(threshold), "fscore_beta": args.fscore_beta,
        "validation_f_score": float(val_f2), "train_f_score": float(train_f2),
        "input_features": ["Space", "MS", "VCD", "OVL", "CBCD"],
    }
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
