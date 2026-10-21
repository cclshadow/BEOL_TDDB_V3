"""
Map a per-die via-to-line spacing map (Space.csv) to a synthetic Vramp breakdown voltage map (V_Ramp-style CSV).

Option 1 / fixed field strength: breakdown is assumed to occur when the planar-plate field V/S reaches
the DPM strength field E_S(S) = 1 / (a + b/S) (IMEC eq. 17, `BaseDPMModel.calc_Es`), so

    Vbd = S * E_S(S) = S^2 / (a*S + b)

E_S is assumed to be in V/nm, with S in nm. This is the forward direction of vramp_to_space_option_1.py
and ignores ramp rate, test-structure area and Weibull statistics: every die gets a deterministic Vbd.

Special values:
    - Dies outside ExistenceClass are written as 0 (same as the simulator output).
    - Space <= 0 is treated as a short and written as Vbd = 0.
    - Vbd above --compliance is clipped to the compliance voltage, mimicking a ramp that stops
      before breakdown.

Usage (from the repo root, BEOL_TDDB_V3):
    python Vbd_Space_utils/space_to_vramp_option_1.py --input data_2
    python Vbd_Space_utils/space_to_vramp_option_1.py --input data_2/lot_001/csv/wafer_01/Space.csv --out-name V_Ramp_opt1.csv
    python Vbd_Space_utils/space_to_vramp_option_1.py --input data_2 --es-a 0.85 --es-b 2.91 --compliance 20 --overwrite
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Make the repo root importable when this file is run as a script from any folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.csv_utils import load_matrix_csv, save_matrix_csv, valid_die_mask

SPACE_NAME = "Space.csv"


def space_to_vbd(space: np.ndarray, es_a: float = 0.85, es_b: float = 2.91) -> np.ndarray:
    """Vbd = S * E_S(S) = S^2 / (a*S + b)."""
    return space * space / (es_a * space + es_b)


def convert_wafer(space_path: Path, out_path: Path, es_a: float, es_b: float, compliance: float) -> dict:
    space = load_matrix_csv(space_path)

    class_path = space_path.parent / "ExistenceClass.csv"
    mask = valid_die_mask(load_matrix_csv(class_path)) if class_path.exists() else np.isfinite(space) & (space != 0)
    if not np.isfinite(space[mask]).all():
        raise ValueError(f"{space_path}: non-finite Space at valid dies")

    shorted = mask & (space <= 0)
    measured = mask & ~shorted

    vbd = np.zeros_like(space, dtype=float)
    vbd[measured] = space_to_vbd(space[measured], es_a, es_b)
    censored = measured & (vbd >= compliance)
    vbd[censored] = compliance
    save_matrix_csv(vbd, str(out_path))

    return {"n_dies": int(mask.sum()), "n_shorted": int(shorted.sum()), "n_censored": int(censored.sum())}


def main():
    parser = argparse.ArgumentParser(description="Convert Space.csv spacing maps into V_Ramp-style breakdown maps")
    parser.add_argument("--input", type=str, required=True,
                        help=f"A {SPACE_NAME} file, or a root searched recursively for {SPACE_NAME}")
    parser.add_argument("--out-name", type=str, default="V_Ramp_from_Space.csv",
                        help="Output file name, written next to each Space.csv (default: V_Ramp_from_Space.csv)")
    parser.add_argument("--es-a", type=float, default=0.85, help="E_S constant a (0.85 + Es_val_delta_1)")
    parser.add_argument("--es-b", type=float, default=2.91, help="E_S constant b (2.91 + Es_val_delta_2)")
    parser.add_argument("--compliance", type=float, default=20.0,
                        help="Ramp stop voltage; Vbd above it is clipped to it (default: 20)")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output files")
    args = parser.parse_args()

    if args.es_a <= 0 or args.es_b <= 0:
        raise ValueError("--es-a and --es-b must be positive for E_S(S) to stay finite and positive")

    in_path = Path(args.input)
    space_paths = [in_path] if in_path.is_file() else sorted(in_path.rglob(SPACE_NAME))
    if not space_paths:
        raise FileNotFoundError(f"No {SPACE_NAME} found under {in_path}")

    out_paths = [p.parent / args.out_name for p in space_paths]
    existing = [p for p in out_paths if p.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"{len(existing)} output files already exist (e.g. {existing[0]}); "
                              f"pass --overwrite to replace them")

    totals = {"n_dies": 0, "n_shorted": 0, "n_censored": 0}
    for space_path, out_path in zip(space_paths, out_paths):
        stats = convert_wafer(space_path, out_path, args.es_a, args.es_b, args.compliance)
        for key in totals:
            totals[key] += stats[key]

    # Spacing above which the ramp would stop before breakdown
    a, b, v = args.es_a, args.es_b, args.compliance
    s_cap = 0.5 * (a * v + np.sqrt((a * v) ** 2 + 4.0 * b * v))
    print(f"[Done] {len(space_paths)} wafers -> {args.out_name}")
    print(f"       dies: {totals['n_dies']}, shorted (Space<=0 -> Vbd=0): {totals['n_shorted']}, "
          f"clipped (Space>={s_cap:.2f} nm -> Vbd={v:g} V): {totals['n_censored']}")


if __name__ == "__main__":
    main()
