"""Run a fitted spatial ML backend and write die-level risk/reliability maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.csv_utils import save_matrix_csv, sparse_points_to_matrix
from src.spatial_ml import load_backend, load_wafer
from src.spatial_ml.data import measured_mask


def wafer_paths(root: Path):
    return sorted(p for p in root.glob("lot_*/csv/wafer_*") if p.is_dir())


def main():
    p = argparse.ArgumentParser(description="Infer die reliability from sparse wafer metrology")
    p.add_argument("--config-path", required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--test-path", help="Root containing lot_*/csv/wafer_* directories")
    source.add_argument("--manifest", help="Split manifest from create_split.py")
    p.add_argument("--split", choices=("train", "val", "test"), default="test",
                   help="Manifest split to infer (default: test)")
    p.add_argument("--save-path", required=True)
    p.add_argument("--measurement-fraction", type=float, default=None,
                   help="Overrides calibration fraction; replace with test-plan mask in production")
    args = p.parse_args()
    config_path = Path(args.config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = load_backend(config_path.parent / config["artifact"])
    fraction = args.measurement_fraction or config["measurement_fraction"]
    out_root = Path(args.save_path)
    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        paths = [Path(p) for p in manifest[f"{args.split}_wafers"]]
    else:
        paths = wafer_paths(Path(args.test_path))

    for i, wafer_path in enumerate(paths):
        wafer = load_wafer(wafer_path)
        mask = measured_mask(wafer.n_dies, fraction, config["seed"] + i)
        bad_probability = model.predict_proba(wafer, mask)
        reliability = 1.0 - bad_probability
        labels = (reliability >= config["threshold"]).astype(np.float32)
        relative = Path(*wafer_path.parts[-3:])
        out = out_root / relative
        out.mkdir(parents=True, exist_ok=True)
        # sparse_points_to_matrix expects coordinates centred on the wafer,
        # not raw matrix row/column indices.
        x = wafer.xy[:, 0].astype(np.int32)
        y = wafer.xy[:, 1].astype(np.int32)
        save_matrix_csv(sparse_points_to_matrix(x, y, reliability,
                                                wafer.matrix_shape), out / "reliability_score.csv")
        save_matrix_csv(sparse_points_to_matrix(x, y, labels,
                                                wafer.matrix_shape), out / "binary_label.csv")
        save_matrix_csv(sparse_points_to_matrix(x, y, mask.astype(float),
                                                wafer.matrix_shape), out / "measurement_mask.csv")


if __name__ == "__main__":
    main()
