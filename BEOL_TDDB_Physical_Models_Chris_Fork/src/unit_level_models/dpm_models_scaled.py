import numpy as np
from abc import abstractmethod

from .base_class import BaseTDDBModel

V_OP = 0.7  # Fallback default value / 兜底默认值

# ==============================================================================
# Physics Base Model Parent Class (Scaled) / 物理基础模型母类（含间距缩放）
# ==============================================================================
class BaseDPMModel(BaseTDDBModel):
    """
    Scaled variant of BaseDPMModel.

    Adds a `spacing_scale` factor k that maps measured spacings into the
    calibrated regime of the DPM physics equations before any calculation:

        S_eff = S_measured / k

    Motivation: the DPM constants (C_sc, P_sc, W_mu, …) were calibrated for
    sub-5 nm geometries.  Real datasets with spacings in the 9–22 nm range
    produce compressed TTF distributions that prevent threshold-based
    discrimination.  Dividing by k re-centres the physics in its valid range
    without modifying any equation coefficients.

    Suggested starting point for a 9–22 nm dataset: k = 3.0
        9 nm  →  3.0 nm  (high-field, short-lifetime regime)
        15 nm →  5.0 nm  (calibration boundary)
        22 nm →  7.3 nm  (moderate-field regime)

    k = 1.0 reproduces the original unscaled behaviour exactly.
    """

    def __init__(self,
                 target_component: str = "line",
                 v_op: float = V_OP,
                 spacing_scale: float = 1.0):
        """
        Args:
            target_component: "via" or "line".
            v_op:             Operating voltage (V).
            spacing_scale:    Divisor k applied to raw spacings before physics.
                              k > 1 compresses spacings into a smaller effective
                              range; k = 1 is the original behaviour.
        """
        self.target_component = target_component.lower()
        assert self.target_component in ["via", "line"], \
            "target_component must be 'via' or 'line'"

        self.v_op = v_op
        self.spacing_scale = float(spacing_scale)
        assert self.spacing_scale > 0, "spacing_scale must be positive"

        # Table 1 Shared Physical Constants / Table 1 共有物理常数
        self.C_sc = 2.134
        self.P_sc = 0.847
        self.W_mu = -0.3665
        self.mu_sc = 0.914

    def _extract_spacing(self, inputs: np.ndarray) -> np.ndarray:
        arr = np.asarray(inputs, dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 2:
            return arr[:, 0] if self.target_component == "via" else arr[:, 1]
        elif arr.ndim == 2 and arr.shape[1] == 1:
            return arr.flatten()
        return arr

    def calc_beta_DOT(self, S: np.ndarray) -> np.ndarray:
        return self.C_sc * (S ** self.P_sc)

    def calc_eta_DOT(self, S: np.ndarray) -> np.ndarray:
        beta_DOT = self.calc_beta_DOT(S)
        return self.mu_sc * np.exp(-self.W_mu / beta_DOT)

    def calc_Es(self, S: np.ndarray) -> np.ndarray:
        return 1.0 / (0.85 + 2.91 / S)

    def calc_m(self, S: np.ndarray) -> np.ndarray:
        return 0.5659 * (S ** (-0.455))

    def calc_beta_tBD(self, S: np.ndarray) -> np.ndarray:
        return self.calc_m(S) * self.calc_beta_DOT(S)

    @abstractmethod
    def calc_ln_eta_tBD(self, S: np.ndarray) -> np.ndarray:
        pass

    def predict_weibull_log_params(self, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return beta and ln(eta) without risking exponential overflow."""
        S_raw = self._extract_spacing(inputs)

        # Apply spacing normalisation: map measured spacing into calibrated regime
        S = S_raw / self.spacing_scale

        beta_tBD = self.calc_beta_tBD(S)
        ln_eta_tBD = self.calc_ln_eta_tBD(S)

        if inputs.ndim == 2:
            beta_tBD = beta_tBD.reshape(-1, 1)
            ln_eta_tBD = ln_eta_tBD.reshape(-1, 1)

        return beta_tBD, ln_eta_tBD

    def predict_weibull_params(self, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        beta_tBD, ln_eta_tBD = self.predict_weibull_log_params(inputs)

        # Overflow-safe exponentiation
        eta_tBD = np.full_like(ln_eta_tBD, np.inf, dtype=float)
        safe_mask = ln_eta_tBD <= 709.0
        eta_tBD[safe_mask] = np.exp(ln_eta_tBD[safe_mask])

        return beta_tBD, eta_tBD


# ==============================================================================
# Derived Subclasses
# ==============================================================================

class PowerLawDPMModel(BaseDPMModel):
    """Classical Power-Law E-field acceleration model with spacing scale."""

    def calc_ma(self, S: np.ndarray) -> np.ndarray:
        return 20.66 / np.tanh(0.073 * S)

    def calc_ln_eta_tBD(self, S: np.ndarray) -> np.ndarray:
        eta_DOT = self.calc_eta_DOT(S)
        E_local = self.v_op / S
        m_val = self.calc_m(S)
        Es_val = self.calc_Es(S)
        ma_val = self.calc_ma(S)

        ln_AF = -ma_val * np.log(E_local / Es_val)
        ln_eta_supercell = ln_AF + (1.0 / m_val) * np.log(eta_DOT)

        beta_tBD = self.calc_beta_tBD(S)
        return ln_eta_supercell + (1.0 / beta_tBD) * np.log(S ** 2)


class SqrtEDPMModel(BaseDPMModel):
    """Sqrt(E) E-field acceleration model with spacing scale."""

    def __init__(self,
                 target_component: str = "line",
                 v_op: float = V_OP,
                 spacing_scale: float = 1.0):
        super().__init__(target_component, v_op, spacing_scale)
        self.gamma_coeff = 49.14

    def calc_gamma_sqrtE(self, S: np.ndarray) -> np.ndarray:
        return self.gamma_coeff / np.tanh(0.073 * S)

    def calc_ln_eta_tBD(self, S: np.ndarray) -> np.ndarray:
        eta_DOT = self.calc_eta_DOT(S)
        E_local = self.v_op / S
        m_val = self.calc_m(S)
        Es_val = self.calc_Es(S)
        gamma = self.calc_gamma_sqrtE(S)

        ln_AF = gamma * (np.sqrt(Es_val) - np.sqrt(E_local))
        ln_eta_supercell = ln_AF + (1.0 / m_val) * np.log(eta_DOT)

        beta_tBD = self.calc_beta_tBD(S)
        return ln_eta_supercell + (1.0 / beta_tBD) * np.log(S ** 2)


class InverseEDPMModel(BaseDPMModel):
    """1/E E-field acceleration model with spacing scale."""

    def __init__(self,
                 target_component: str = "line",
                 v_op: float = V_OP,
                 spacing_scale: float = 1.0):
        super().__init__(target_component, v_op, spacing_scale)
        self.G_1E_coeff = 14.39

    def calc_G_1E(self, S: np.ndarray) -> np.ndarray:
        return self.G_1E_coeff / np.tanh(0.073 * S)

    def calc_ln_eta_tBD(self, S: np.ndarray) -> np.ndarray:
        eta_DOT = self.calc_eta_DOT(S)
        E_local = self.v_op / S
        m_val = self.calc_m(S)
        Es_val = self.calc_Es(S)
        G_val = self.calc_G_1E(S)

        ln_AF = G_val * (1.0 / E_local - 1.0 / Es_val)
        ln_eta_supercell = ln_AF + (1.0 / m_val) * np.log(eta_DOT)

        beta_tBD = self.calc_beta_tBD(S)
        return ln_eta_supercell + (1.0 / beta_tBD) * np.log(S ** 2)
