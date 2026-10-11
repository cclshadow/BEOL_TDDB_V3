import numpy as np
from .wafer_reliability_engines import BaseWaferEngine


class LinearWaferEngine(BaseWaferEngine):
    """
    Zero-physics baseline engine. TTF proxy = VL_spacing + LL_spacing per die.

    Bypasses all Weibull physics, spatial kriging, and weakest-link Monte Carlo.
    M_structures, F_target, and N_samples_per_dim are accepted for interface
    compatibility but have no effect on predictions.

    If this baseline matches DPM/GPR performance, the physics machinery adds no
    predictive value beyond the raw spacing signal.
    """

    def predict_die_lifetimes(
        self,
        x_die: np.ndarray,
        y_die: np.ndarray,
        vl_data: np.ndarray,
        ll_data: np.ndarray,
        N_samples_per_dim: int = 10,
    ) -> np.ndarray:
        return vl_data + ll_data
