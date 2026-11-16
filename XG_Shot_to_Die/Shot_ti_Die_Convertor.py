from pathlib import Path
import csv
import numpy as np

# ============================================================
# Config
# ============================================================
INPUT_ROOT = Path("A")
OUTPUT_ROOT = Path("D")
PATTERN = "C.csv"  # e.g. "*_OVL.csv", "*.csv"

# Wafer geometry CSVs
DIE_GEO_CSV = Path("wafer_geo_13x10_shot_3x4_die.csv")
SHOT_GEO_CSV = Path("shot_layout_13x10_reference.csv")

SHOT_ROWS = 4
SHOT_COLS = 3

ZERO_IS_MISSING = False
OUTPUT_SUFFIX = "_die"

# ============================================================
# Load geometry
# ============================================================
die_geo = np.loadtxt(DIE_GEO_CSV, delimiter=",", dtype=int)
shot_geo = np.loadtxt(SHOT_GEO_CSV, delimiter=",", dtype=int)

if die_geo.ndim != 2 or shot_geo.ndim != 2:
    raise ValueError("Geometry CSVs must be 2D.")

if not np.all(np.isin(die_geo, [0, 1])) or not np.all(np.isin(shot_geo, [0, 1])):
    raise ValueError("Geometry CSVs must contain only 0/1.")

H, W = die_geo.shape
SH, SW = shot_geo.shape

if (H, W) != (SH * SHOT_ROWS, SW * SHOT_COLS):
    raise ValueError("Die and shot geometry dimensions do not match.")

# Check shot/die geometry consistency
inferred = die_geo.reshape(SH, SHOT_ROWS, SW, SHOT_COLS).any(axis=(1, 3))
if not np.array_equal(inferred, shot_geo.astype(bool)):
    raise ValueError("Shot and die geometry masks are inconsistent.")

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# ============================================================
# Batch conversion
# ============================================================
files = sorted(p for p in INPUT_ROOT.rglob(PATTERN) if p.is_file())

for src in files:
    data = np.genfromtxt(src, delimiter=",", dtype=float)

    if data.shape != die_geo.shape:
        raise ValueError(f"{src}: data shape mismatch.")

    if ZERO_IS_MISSING:
        data[data == 0] = np.nan

    if np.isfinite(data[die_geo == 0]).any():
        raise ValueError(f"{src}: data outside valid die geometry.")

    result = np.full((H, W), np.nan)

    for sr in range(SH):
        for sc in range(SW):

            # Skip invalid shots
            if shot_geo[sr, sc] == 0:
                continue

            r = sr * SHOT_ROWS
            c = sc * SHOT_COLS

            rs = slice(r, r + SHOT_ROWS)
            cs = slice(c, c + SHOT_COLS)

            valid = die_geo[rs, cs] == 1
            values = data[rs, cs][valid]
            values = values[np.isfinite(values)]

            if len(values) == 0:
                continue

            if not np.allclose(values, values[0]):
                raise ValueError(
                    f"{src}: conflicting values in shot ({sr}, {sc})"
                )

            result[rs, cs] = np.where(valid, values[0], np.nan)

    # Preserve original subfolder structure
    relative = src.relative_to(INPUT_ROOT)
    dst = OUTPUT_ROOT / relative.parent / f"{src.stem}{OUTPUT_SUFFIX}.csv"

    dst.parent.mkdir(parents=True, exist_ok=True)

    with dst.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows([
            [f"{v:.6g}" if np.isfinite(v) else "" for v in row]
            for row in result
        ])

    print(f"Converted: {src} -> {dst}")

print(f"Finished: {len(files)} files.")