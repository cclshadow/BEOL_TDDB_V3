import numpy as np
from abc import abstractmethod

from .base_class import BaseTDDBModel

V_OP = 0.7  # Fallback default value / 兜底默认值

# ==============================================================================
# Physics Base Model Parent Class: Base DPM / 物理基础模型母类：Base DPM
# ==============================================================================
class BaseDPMModel(BaseTDDBModel):
    """
    Base class for Dynamic Percolation Model (DPM) physics.
    Upgraded to support unified 2D inputs [vl_space, ll_space] to perfectly align with GPR.

    动态渗透模型 (DPM) 物理公式的基类。
    升级后支持统一的 2维输入 [vl_space, ll_space]，从而与 GPR 模型实现完美对齐。

    Trainable deltas / 可训练的 delta 微调量
    -----------------------------------------
    The empirical, spacing-dependent sub-equations (m, E_s, and the per-model
    acceleration factor) each expose additive `delta` corrections that default
    to 0.0 — with every delta at 0.0 the model reproduces the original IMEC fit
    exactly. A second team fits these deltas to their own process/dielectric via
    the training pipeline; the fitted values ride inside `unit_model_kwargs`
    ("deltas") and are therefore saved to / loaded from config.json automatically.

    Note: the simulation backbone constants (C_sc, P_sc, W_mu, mu_sc of the DOT
    distribution) are intentionally NOT exposed — the deltas perturb the
    empirical fits, not the simulated percolation backbone.

    经验的、间距相关的子方程（m、E_s 以及各模型特有的加速因子）都开放了加性的
    `delta` 修正量，默认全部为 0.0 —— 当所有 delta 均为 0.0 时，模型与原始 IMEC
    拟合完全一致。接手的团队可通过训练流水线将这些 delta 拟合到自家工艺/介质上；
    拟合值存放在 `unit_model_kwargs` 的 "deltas" 中，因此会随 config.json 自动
    保存与加载。仿真骨架常数（DOT 分布的 C_sc、P_sc、W_mu、mu_sc）有意不开放 ——
    delta 修正的是经验拟合，而非仿真渗透骨架。
    """

    # Shared trainable deltas: {name: (default, low, high)}.
    # Bounds are the physically-informed ranges validated with the interactive
    # delta sweep tool (each range keeps its sub-equation on the well-behaved
    # side of a singularity/sign-flip over the plotted 0-20 nm spacing window).
    # 共享可训练 delta：{名称: (默认值, 下界, 上界)}。边界取自交互式扫掠工具中
    # 经物理校验的范围（保证子方程在 0-20nm 窗口内不越过奇点/变号）。
    _SHARED_DELTA_SPEC = {
        "m_val_delta_1":  (0.0, -0.5, 2.0),   # keeps (0.5659 + d) > 0
        "m_val_delta_2":  (0.0, -0.3, 0.3),   # exponent shift -- highly sensitive
        "Es_val_delta_1": (0.0, -0.5, 1.0),   # keeps Es denominator > 0
        "Es_val_delta_2": (0.0, -1.5, 5.0),   # keeps the fitted numerator (2.91 + d) > 0
    }
    # Per-model deltas; overridden by each subclass. Holds each model's own
    # acceleration-factor deltas AND may re-tighten a shared bound for that
    # model (an entry here overrides the same key in _SHARED_DELTA_SPEC).
    # 各模型特有的 delta；由子类覆盖。既存放本模型的加速因子 delta，也可覆盖
    # 收紧某个共享参数的边界（此处的同名键会覆盖 _SHARED_DELTA_SPEC 中的项）。
    _MODEL_DELTA_SPEC: dict = {}

    def __init__(self, target_component: str = "line", v_op: float = V_OP,
                 deltas: dict = None):
        """
        Args:
            target_component (str): "via" or "line". Determines which spacing column to route.
                                    "via" 或 "line"。决定自动路由并提取哪一列间距数据。
            v_op (float): Operating voltage.
                          工作电压。
            deltas (dict): Optional {name: value} overrides for the trainable
                           corrections. Any omitted name defaults to 0.0.
                           可选的 {名称: 值} 覆盖表；未给出的项默认为 0.0。
        """
        self.target_component = target_component.lower()
        assert self.target_component in ["via", "line"], "target_component must be 'via' or 'line'"

        self.v_op = v_op

        # Table 1 Shared Physical Constants / Table 1 共有物理常数
        self.C_sc = 2.134   # Weibull slope coefficient / Weibull 斜率系数
        self.P_sc = 0.847   # Weibull slope exponent / Weibull 斜率指数
        self.W_mu = -0.3665 # Scale parameter for DOT distribution / DOT 分布的尺度参数
        self.mu_sc = 0.914  # Scale factor for eta_DOT / eta_DOT 的缩放因子

        # Merge caller-supplied deltas onto the zero-defaults for this model.
        # 将调用方传入的 delta 覆盖到本模型的零默认值之上。
        self.deltas = self._init_deltas(deltas)

    # --------------------------------------------------------------------------
    # Trainable-parameter plumbing / 可训练参数管线
    # --------------------------------------------------------------------------
    @classmethod
    def _full_spec(cls) -> dict:
        """Shared deltas, with this model's deltas / bound-overrides applied on top.
        共享 delta，叠加本模型自有的 delta 及边界覆盖。"""
        spec = dict(BaseDPMModel._SHARED_DELTA_SPEC)
        spec.update(cls._MODEL_DELTA_SPEC)
        return spec

    def trainable_param_spec(self) -> dict:
        """{name: (default, low, high)} for every delta this model can train."""
        return dict(self._full_spec())

    def _init_deltas(self, deltas: dict) -> dict:
        spec = self._full_spec()
        merged = {name: default for name, (default, _lo, _hi) in spec.items()}
        if deltas:
            for name, value in deltas.items():
                if name not in spec:
                    raise ValueError(
                        f"Unknown delta '{name}' for {type(self).__name__}. "
                        f"Valid deltas: {list(spec.keys())}"
                    )
                merged[name] = float(value)
        return merged

    def _extract_spacing(self, inputs: np.ndarray) -> np.ndarray:
        """
        Feature Routing Layer: Extract the corresponding 1D spacing based on target_component.
        Supports both unified 2D matrix (n_samples, 2) and fallback 1D array (n_samples,).

        特征路由层：根据目标组件自动提取对应的 1维间距数组。
        同时支持统一的 2维特征矩阵 (n_samples, 2) 和向后兼容的 1维数组 (n_samples,)。
        """
        arr = np.asarray(inputs, dtype=float)

        if arr.ndim == 2 and arr.shape[1] == 2:
            # Column 0: vl_space (via spacing), Column 1: ll_space (line spacing / MS)
            # 第 0 列：vl_space (via间距)，第 1 列：ll_space (line间距 / MS)
            if self.target_component == "via":
                return arr[:, 0]
            else:
                return arr[:, 1]
        elif arr.ndim == 2 and arr.shape[1] == 1:
            return arr.flatten()

        return arr

    def calc_beta_DOT(self, S: np.ndarray) -> np.ndarray:
        """Equation (2): Calculates the Weibull slope of the DOT distribution."""
        """公式 (2): 计算 DOT 分布的 Weibull 斜率。"""
        return self.C_sc * (S ** self.P_sc)

    def calc_eta_DOT(self, S: np.ndarray) -> np.ndarray:
        """Equation (3): Calculates the local Weibull scale parameter for DOT distribution."""
        """公式 (3): 计算 DOT 分布的局部 Weibull 尺度参数。"""
        beta_DOT = self.calc_beta_DOT(S)
        return self.mu_sc * np.exp(-self.W_mu / beta_DOT)

    def calc_Es(self, S: np.ndarray) -> np.ndarray:
        """Equation (17): Characteristic E-field (Es); trainable deltas on the fit constants."""
        """公式 (17): 特征电场 (Es)；拟合常数上带可训练 delta。"""
        return 1.0 / (0.85 + self.deltas["Es_val_delta_1"]
                      + (2.91 + self.deltas["Es_val_delta_2"]) / S)

    def calc_m(self, S: np.ndarray) -> np.ndarray:
        """Equation (18): Spacing-dependent exponent m; trainable deltas on coeff and exponent."""
        """公式 (18): 间距相关指数 m；系数与指数上带可训练 delta。"""
        return (0.5659 + self.deltas["m_val_delta_1"]) * (S ** (-0.455 + self.deltas["m_val_delta_2"]))

    def calc_beta_tBD(self, S: np.ndarray) -> np.ndarray:
        """Calculates the local Weibull shape parameter beta for time-to-breakdown."""
        """计算最终的 Time-to-breakdown Weibull 形状参数 beta。"""
        return self.calc_m(S) * self.calc_beta_DOT(S)

    @abstractmethod
    def calc_ln_eta_tBD(self, S: np.ndarray) -> np.ndarray:
        """Calculate the natural log of the final scale parameter: ln(eta_tBD)."""
        """计算最终的弹性尺度参数的自然对数 ln(eta_tBD)。"""
        pass

    def predict_weibull_log_params(self, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return beta and ln(eta) without risking exponential overflow."""
        # Automatically route and extract the correct physical spacing
        # 自动路由并提取出正确的物理间距
        S = self._extract_spacing(inputs)

        beta_tBD = self.calc_beta_tBD(S)
        ln_eta_tBD = self.calc_ln_eta_tBD(S)

        if inputs.ndim == 2:
            beta_tBD = beta_tBD.reshape(-1, 1)
            ln_eta_tBD = ln_eta_tBD.reshape(-1, 1)

        return beta_tBD, ln_eta_tBD

    def predict_weibull_params(self, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Unified interface implementation: seamlessly process 2D/1D inputs and return parameters.

        统一接口实现：无缝处理 2D/1D 输入，并返回对应的 Weibull 参数。
        """
        beta_tBD, ln_eta_tBD = self.predict_weibull_log_params(inputs)

        # Secure handling to prevent float64 overflow
        # 安全防溢出处理
        eta_tBD = np.full_like(ln_eta_tBD, np.inf, dtype=float)
        safe_mask = ln_eta_tBD <= 709.0
        eta_tBD[safe_mask] = np.exp(ln_eta_tBD[safe_mask])

        return beta_tBD, eta_tBD


# ==============================================================================
# Derived Subclasses (Automatically Inherit the Routing + Delta Feature)
#    派生子类（自动继承特征路由与 delta 功能）
# ==============================================================================

class PowerLawDPMModel(BaseDPMModel):
    """1) Classical Power-Law Acceleration Model. / 1) 经典幂律加速模型。"""
    # m_a = (20.66 + d1) / tanh((0.073 + d2) * S)
    _MODEL_DELTA_SPEC = {
        "ma_val_delta_1": (0.0, -15.0, 30.0),  # keeps (20.66+d1)>0 (floor ~5.66, ceil ~2.45x)
        "ma_val_delta_2": (0.0, -0.06, 0.3),   # keeps (0.073+d2)>0 so tanh can't flip sign
    }

    def calc_ma(self, S: np.ndarray) -> np.ndarray:
        return (20.66 + self.deltas["ma_val_delta_1"]) / np.tanh((0.073 + self.deltas["ma_val_delta_2"]) * S)

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
    """2) Sqrt(E) Acceleration Model. / 2) Sqrt(E) 电场加速模型。"""
    # gamma = (49.14 + d1) / tanh((0.073 + d2) * S); d1 range = PowerLaw's
    # -72.6%/+145% ratios applied to 49.14 -> [-35, 70].
    _MODEL_DELTA_SPEC = {
        "gamma_val_delta_1": (0.0, -35.0, 70.0),  # floor ~14.14 (positive), ceil ~2.4x
        "gamma_val_delta_2": (0.0, -0.06, 0.3),   # keeps (0.073+d2)>0
    }

    def __init__(self, target_component: str = "line", v_op: float = V_OP,
                 deltas: dict = None):
        super().__init__(target_component, v_op, deltas)
        self.gamma_coeff = 49.14

    def calc_gamma_sqrtE(self, S: np.ndarray) -> np.ndarray:
        return (self.gamma_coeff + self.deltas["gamma_val_delta_1"]) / np.tanh(
            (0.073 + self.deltas["gamma_val_delta_2"]) * S)

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
    """3) 1/E Acceleration Model. / 3) 1/E 电场加速模型。"""
    # G = (14.39 + d1) / tanh((0.073 + d2) * S); d1 range = PowerLaw's
    # -72.6%/+145% ratios applied to 14.39 -> [-10, 20].
    _MODEL_DELTA_SPEC = {
        "G_val_delta_1":   (0.0, -10.0, 20.0),  # floor ~4.39 (positive), ceil ~2.4x
        "G_val_delta_2":   (0.0, -0.06, 0.3),   # keeps (0.073+d2)>0
    }

    def __init__(self, target_component: str = "line", v_op: float = V_OP,
                 deltas: dict = None):
        super().__init__(target_component, v_op, deltas)
        self.G_1E_coeff = 14.39

    def calc_G_1E(self, S: np.ndarray) -> np.ndarray:
        return (self.G_1E_coeff + self.deltas["G_val_delta_1"]) / np.tanh(
            (0.073 + self.deltas["G_val_delta_2"]) * S)

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
