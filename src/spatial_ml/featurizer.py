"""GP spatial reconstruction used by all supervised spatial backends."""

from __future__ import annotations

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

from .data import WaferData


class SpatialFeaturizer:
    """Turn sparse die metrology into dense, uncertainty-aware features."""

    def __init__(self, length_scale: float = 3.0, noise: float = 0.03):
        self.length_scale = length_scale
        self.noise = noise

    def transform(self, wafer: WaferData, observed: np.ndarray) -> np.ndarray:
        """Return [imputed channels, GP std channels, x, y, radius, observed]."""
        observed = np.asarray(observed, dtype=bool)
        if observed.shape != (wafer.n_dies,):
            raise ValueError("observed mask must have one entry per valid die")
        if observed.sum() < 2:
            raise ValueError("at least two measured dies are required for GP reconstruction")

        x_train, x_all = wafer.xy[observed], wafer.xy
        imputed, uncertainty = [], []
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
            self.length_scale, (0.25, 30.0)
        ) + WhiteKernel(self.noise, (1e-6, 1.0))
        for channel in wafer.features.T:
            gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=0,
                                          n_restarts_optimizer=1)
            gp.fit(x_train, channel[observed])
            mean, std = gp.predict(x_all, return_std=True)
            imputed.append(mean)
            uncertainty.append(std)
        radius = np.linalg.norm(wafer.xy, axis=1, keepdims=True)
        return np.column_stack([
            np.column_stack(imputed), np.column_stack(uncertainty),
            wafer.xy, radius, observed.astype(np.float32),
        ]).astype(np.float32)

    def token_features(self, wafer: WaferData, observed: np.ndarray) -> np.ndarray:
        """Raw sparse tokens for neural models; unavailable values are zero + mask."""
        raw = wafer.features.copy()
        raw[~observed] = 0.0
        radius = np.linalg.norm(wafer.xy, axis=1, keepdims=True)
        return np.column_stack([raw, wafer.xy, radius, observed.astype(np.float32)]).astype(np.float32)
