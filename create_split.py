"""
Lot-level stratified train/validation/test split generator for BEOL TDDB dataset.

Splits at lot granularity to prevent correlation leakage between wafers from the same lot.
Stratifies by per-lot failure rate so each split covers the full range of failure densities.

Usage:
    python create_split.py --dataset-path data/
    python create_split.py --dataset-path data/ --train-ratio 0.6 --val-ratio 0.2 --seed 42 --output split_manifest.json
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple


def find_all_lots(base_path: Path) -> List[Path]:
    return sorted([
        p.resolve() for p in base_path.glob("lot_*")
        if p.is_dir() and (p / "csv").is_dir()
    ])


def find_wafers_in_lot(lot_path: Path) -> List[Path]:
    return sorted([
        w.resolve() for w in (lot_path / "csv").glob("wafer_*")
        if w.is_dir()
    ])


def compute_lot_stats(lot_path: Path) -> Dict:
    """
    Compute failure rate for a lot: fraction of valid dies where ExistenceClass != 3.
    Mirrors the matrix_to_sparse_points logic: ignores NaN, Inf, and zero-valued cells.
    """
    wafer_paths = find_wafers_in_lot(lot_path)
    total_valid = 0
    total_fail = 0

    for wafer_path in wafer_paths:
        class_csv = wafer_path / "ExistenceClass.csv"
        if not class_csv.exists():
            continue
        try:
            matrix = np.loadtxt(class_csv, delimiter=",")
            valid_mask = np.isfinite(matrix) & (matrix != 0)
            values = matrix[valid_mask].astype(np.int32)
            total_valid += len(values)
            total_fail += int(np.sum(values != 3))
        except Exception as e:
            print(f"  [Warning] Could not read {class_csv}: {e}")

    return {
        "lot_path": str(lot_path),
        "lot_name": lot_path.name,
        "n_wafers": len(wafer_paths),
        "n_valid_dies": total_valid,
        "n_fail_dies": total_fail,
        "failure_rate": total_fail / total_valid if total_valid > 0 else 0.0,
        "wafer_paths": [str(w) for w in wafer_paths],
    }


def stratified_lot_split(
    lot_stats: List[Dict],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Sort lots by failure rate, then assign splits in a round-robin pattern across
    the sorted order so each split covers the full failure-rate distribution.

    For 60/20/20 the cycle is 5 slots: [train, train, train, val, test].
    The assignment order within each cycle is shuffled to avoid systematic bias.
    """
    rng = np.random.default_rng(seed)
    test_ratio = round(1.0 - train_ratio - val_ratio, 10)

    # Derive cycle length from the smallest ratio (must divide evenly)
    min_ratio = min(train_ratio, val_ratio, test_ratio)
    cycle = round(1.0 / min_ratio)
    train_slots = round(train_ratio * cycle)
    val_slots = round(val_ratio * cycle)
    test_slots = cycle - train_slots - val_slots

    template = ["train"] * train_slots + ["val"] * val_slots + ["test"] * test_slots

    # Sort lots ascending by failure rate so stratification spans the full range
    sorted_lots = sorted(lot_stats, key=lambda x: x["failure_rate"])

    assignments: List[str] = []
    for i in range(0, len(sorted_lots), cycle):
        chunk = template[: min(cycle, len(sorted_lots) - i)]
        assignments.extend(rng.permuted(chunk).tolist())

    train_lots = [l for l, a in zip(sorted_lots, assignments) if a == "train"]
    val_lots   = [l for l, a in zip(sorted_lots, assignments) if a == "val"]
    test_lots  = [l for l, a in zip(sorted_lots, assignments) if a == "test"]

    return train_lots, val_lots, test_lots


def _split_summary(lots: List[Dict]) -> Dict:
    n_valid = sum(l["n_valid_dies"] for l in lots)
    n_fail  = sum(l["n_fail_dies"]  for l in lots)
    return {
        "n_lots":       len(lots),
        "n_wafers":     sum(l["n_wafers"] for l in lots),
        "n_valid_dies": n_valid,
        "n_fail_dies":  n_fail,
        "failure_rate": n_fail / n_valid if n_valid > 0 else 0.0,
        "lot_names":    [l["lot_name"] for l in lots],
    }


def build_manifest(
    train_lots: List[Dict],
    val_lots:   List[Dict],
    test_lots:  List[Dict],
    train_ratio: float,
    val_ratio:   float,
    seed: int,
    dataset_path: str,
) -> Dict:
    return {
        "split_config": {
            "dataset_path": str(dataset_path),
            "train_ratio":  train_ratio,
            "val_ratio":    val_ratio,
            "test_ratio":   round(1.0 - train_ratio - val_ratio, 6),
            "seed":         seed,
            "split_unit":   "lot",
        },
        "summary": {
            "train": _split_summary(train_lots),
            "val":   _split_summary(val_lots),
            "test":  _split_summary(test_lots),
        },
        "train_wafers": [w for l in train_lots for w in l["wafer_paths"]],
        "val_wafers":   [w for l in val_lots   for w in l["wafer_paths"]],
        "test_wafers":  [w for l in test_lots  for w in l["wafer_paths"]],
    }


def print_summary(manifest: Dict):
    print("\n" + "=" * 60)
    print("  Dataset Split Summary")
    print("=" * 60)
    for split in ("train", "val", "test"):
        s = manifest["summary"][split]
        print(
            f"  {split.upper():5s}  "
            f"{s['n_lots']:2d} lots  "
            f"{s['n_wafers']:3d} wafers  "
            f"{s['n_fail_dies']:6d}/{s['n_valid_dies']:6d} fails  "
            f"({s['failure_rate']*100:.2f}% fail rate)"
        )
        print(f"         lots: {', '.join(s['lot_names'])}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate lot-level stratified train/val/test split")
    parser.add_argument("--dataset-path", type=str, default="data/",
                        help="Root directory containing lot_xxx/ subdirectories")
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio",   type=float, default=0.2)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--output",      type=str,   default="split_manifest.json",
                        help="Output path for the split manifest JSON")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    if test_ratio <= 0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")

    print(f"Scanning lots in: {dataset_path.resolve()}")
    lots = find_all_lots(dataset_path)
    if not lots:
        raise RuntimeError(f"No lot_* directories found under {dataset_path}")
    print(f"Found {len(lots)} lots. Computing failure rates...")

    lot_stats = []
    for lot_path in lots:
        stats = compute_lot_stats(lot_path)
        print(f"  {stats['lot_name']:10s}  {stats['n_wafers']:2d} wafers  "
              f"fail rate: {stats['failure_rate']*100:.2f}%")
        lot_stats.append(stats)

    train_lots, val_lots, test_lots = stratified_lot_split(
        lot_stats,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    manifest = build_manifest(
        train_lots, val_lots, test_lots,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        dataset_path=str(dataset_path.resolve()),
    )

    print_summary(manifest)

    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
