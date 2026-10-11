"""Wafer I/O and leakage-safe sparse-measurement simulation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from ..csv_utils import load_matrix_csv


# Keep this list in one place.  The legacy pipeline uses only Space/MS, while
# the ML backends can exploit every supplied metrology channel.
FEATURE_NAMES = ("Space", "MS", "VCD", "OVL", "CBCD")


@dataclass(frozen=True)
class WaferData:
    """Valid dies of one wafer, represented as an unordered spatial point set."""

    wafer_path: Path
    xy: np.ndarray                 # (n_die, 2), centred die coordinates
    features: np.ndarray           # (n_die, n_feature)
    labels: np.ndarray             # 0=bad, 1=good
    matrix_shape: tuple[int, int]
    rows: np.ndarray
    cols: np.ndarray

    @property
    def n_dies(self) -> int:
        return len(self.labels)


def load_wafer(wafer_path: str | Path, feature_names=FEATURE_NAMES) -> WaferData:
    """Load one wafer and align every channel against ExistenceClass.

    A die is valid only where ExistenceClass is nonzero and finite.  This is
    intentional: a physical measurement value of zero is not used to decide
    whether a die exists, avoiding channel-specific masking mistakes.
    """
    wafer_path = Path(wafer_path)
    classes = load_matrix_csv(wafer_path / "ExistenceClass.csv")
    valid = np.isfinite(classes) & (classes != 0)
    rows, cols = np.where(valid)
    h, w = classes.shape
    xy = np.column_stack((cols - w // 2, rows - h // 2)).astype(np.float32)

    channels = []
    for name in feature_names:
        values = load_matrix_csv(wafer_path / f"{name}.csv")
        if values.shape != classes.shape:
            raise ValueError(f"{wafer_path}: {name}.csv shape {values.shape} != {classes.shape}")
        channels.append(values[valid])
    features = np.column_stack(channels).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError(f"{wafer_path}: metrology contains non-finite values at valid dies")
    labels = (classes[valid].astype(np.int32) == 3).astype(np.int64)
    return WaferData(wafer_path, xy, features, labels, classes.shape, rows, cols)


def load_wafers(paths: Iterable[str | Path]) -> list[WaferData]:
    return [load_wafer(p) for p in paths]


def measured_mask(n_dies: int, fraction: float, seed: int) -> np.ndarray:
    """Deterministically select sparse metrology sites for one wafer.

    This is a deployment simulator, not random feature dropout: all channels at
    an unselected die are unavailable.  A production test-plan mask can be
    supplied to the backends in place of this helper.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError("measurement fraction must be in (0, 1]")
    n = max(2, int(np.ceil(n_dies * fraction)))
    rng = np.random.default_rng(seed)
    mask = np.zeros(n_dies, dtype=bool)
    mask[rng.choice(n_dies, size=min(n, n_dies), replace=False)] = True
    return mask
