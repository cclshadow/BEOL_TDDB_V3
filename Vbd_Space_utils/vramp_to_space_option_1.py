"""
Map per-die Vramp breakdown voltage (V_Ramp.csv) to an effective via-to-line spacing map (Space-style CSV).

Option 1 / fixed field strength: breakdown is assumed to occur when the planar-plate field V/S reaches
the DPM strength field E_S(S) = 1 / (a + b/S) (IMEC eq. 17, `BaseDPMModel.calc_Es`), so

    Vbd = S * E_S(S) = S^2 / (a*S + b)   =>   S = [a*Vbd + sqrt((a*Vbd)^2 + 4*b*Vbd)] / 2

E_S is assumed to be in V/nm, giving S in nm. This ignores ramp rate, test-structure area and Weibull
statistics; it is a naive deterministic inversion meant for exploration.

Special values:
    - Dies outside ExistenceClass are written as 0 (same as the simulator output).
    - Vbd <= 0 is treated as a short and written as Space = 0, which the engines already treat as TTF = 0.
    - Vbd >= --compliance is censored (no breakdown before the ramp stopped); the spacing at the
      compliance voltage is written, which is only a lower bound.

Usage (from the repo root, BEOL_TDDB_V3):
    python Vbd_Space_utils/vramp_to_space_option_1.py --input data_2
    python Vbd_Space_utils/vramp_to_space_option_1.py --input data_2/lot_001/csv/wafer_01/V_Ramp.csv --out-name Space_opt1.csv
    python Vbd_Space_utils/vramp_to_space_option_1.py --input data_2 --es-a 0.85 --es-b 2.91 --compliance 20 --overwrite
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Make the repo root importable when this file is run as a script from any folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.csv_utils import load_matrix_csv, save_matrix_csv, valid_die_mask

VRAMP_NAME = "V_Ramp.csv"


def vbd_to_space(vbd: np.ndarray, es_a: float = 0.85, es_b: float = 2.91) -> np.ndarray:
    """Positive root of a*V*S + b*V = S^2, i.e. the S where S * E_S(S) = Vbd."""
    av = es_a * vbd
    return 0.5 * (av + np.sqrt(av * av + 4.0 * es_b * vbd))


def convert_wafer(vramp_path: Path, out_path: Path, es_a: float, es_b: float, compliance: float) -> dict:
    vbd = load_matrix_csv(vramp_path)

    class_path = vramp_path.parent / "ExistenceClass.csv"
    mask = valid_die_mask(load_matrix_csv(class_path)) if class_path.exists() else np.isfinite(vbd) & (vbd != 0)
    if not np.isfinite(vbd[mask]).all():
        raise ValueError(f"{vramp_path}: non-finite Vbd at valid dies")

    shorted = mask & (vbd <= 0)
    censored = mask & (vbd >= compliance)
    measured = mask & ~shorted

    space = np.zeros_like(vbd, dtype=float)
    space[measured] = vbd_to_space(np.minimum(vbd[measured], compliance), es_a, es_b)
    save_matrix_csv(space, str(out_path))

    stats = {"n_dies": int(mask.sum()), "n_shorted": int(shorted.sum()), "n_censored": int(censored.sum())}

    # Sanity check against the existing geometry map, if there is one
    space_ref_path = vramp_path.parent / "Space.csv"
    if space_ref_path.exists() and space_ref_path != out_path:
        both = measured & ~censored
        if both.sum() > 2:
            ref = load_matrix_csv(space_ref_path)
            stats["corr_vs_space"] = float(np.corrcoef(space[both], ref[both])[0, 1])
    return stats


def main():
    parser = argparse.ArgumentParser(description="Convert V_Ramp.csv breakdown maps into Space-style spacing maps")
    parser.add_argument("--input", type=str, required=True,
                        help=f"A {VRAMP_NAME} file, or a root searched recursively for {VRAMP_NAME}")
    parser.add_argument("--out-name", type=str, default="Space_from_VRamp.csv",
                        help="Output file name, written next to each V_Ramp.csv (default: Space_from_VRamp.csv)")
    parser.add_argument("--es-a", type=float, default=0.85, help="E_S constant a (0.85 + Es_val_delta_1)")
    parser.add_argument("--es-b", type=float, default=2.91, help="E_S constant b (2.91 + Es_val_delta_2)")
    parser.add_argument("--compliance", type=float, default=20.0,
                        help="Ramp stop voltage; Vbd at or above it is treated as censored (default: 20)")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output files")
    args = parser.parse_args()

    if args.es_a <= 0 or args.es_b <= 0:
        raise ValueError("--es-a and --es-b must be positive for E_S(S) to stay finite and positive")

    in_path = Path(args.input)
    vramp_paths = [in_path] if in_path.is_file() else sorted(in_path.rglob(VRAMP_NAME))
    if not vramp_paths:
        raise FileNotFoundError(f"No {VRAMP_NAME} found under {in_path}")

    out_paths = [p.parent / args.out_name for p in vramp_paths]
    existing = [p for p in out_paths if p.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"{len(existing)} output files already exist (e.g. {existing[0]}); "
                              f"pass --overwrite to replace them")

    totals = {"n_dies": 0, "n_shorted": 0, "n_censored": 0}
    corrs = []
    for vramp_path, out_path in zip(vramp_paths, out_paths):
        stats = convert_wafer(vramp_path, out_path, args.es_a, args.es_b, args.compliance)
        for key in totals:
            totals[key] += stats[key]
        if "corr_vs_space" in stats:
            corrs.append(stats["corr_vs_space"])

    s_cap = vbd_to_space(np.array(args.compliance), args.es_a, args.es_b)
    print(f"[Done] {len(vramp_paths)} wafers -> {args.out_name}")
    print(f"       dies: {totals['n_dies']}, shorted (Vbd<=0 -> Space=0): {totals['n_shorted']}, "
          f"censored (Vbd>={args.compliance:g} V -> Space={s_cap:.2f} nm lower bound): {totals['n_censored']}")
    if corrs:
        print(f"       mean corr vs existing Space.csv (uncensored dies): {np.mean(corrs):+.3f}")


if __name__ == "__main__":
    main()
