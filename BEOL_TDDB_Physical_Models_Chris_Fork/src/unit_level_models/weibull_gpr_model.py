import numpy as np
import pathlib
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from .base_class import BaseTDDBModel

# ==============================================================================
# Machine Learning Base Model: GPR Weibull Model / 机器学习基础模型：GPR Weibull 模型
# ==============================================================================
class WeibullGPRModel(BaseTDDBModel):
    def __init__(self, 
                 actual_vl_max: float = 25.0, 
                 actual_ll_max: float = 12.5, 
                 vl_bounds: tuple[float, float] = (0.0, 25.0), 
                 ll_bounds: tuple[float, float] = (5.0, 12.5)):
        """
        Gaussian Process Regression (GPR) model for Weibull parameter interpolation.
        
        基于高斯过程回归（GPR）的 Weibull 参数插值模型。
        """
        self.vl_bounds = vl_bounds
        self.ll_bounds = ll_bounds
        
        # Auto-calculate scaling coefficients (prevent division by zero)
        # 自动计算缩放系数（防止除以0）
        self.vl_scale = self.vl_bounds[1] / max(actual_vl_max, 1e-9)
        self.ll_scale = self.ll_bounds[1] / max(actual_ll_max, 1e-9)

        # Construct the kernel function / 构造核函数
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3)) *
            RBF(length_scale=[1.0, 1.0], length_scale_bounds=(1e-2, 1e2))
            + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e-2))
        )

        self.gp_beta = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        self.gp_eta = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)

    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        """
        Feature preprocessing: scaling and clipping.
        
        特征预处理：缩放与裁剪。
        """
        X_processed = np.copy(X)
        X_processed[:, 0] *= self.vl_scale
        X_processed[:, 1] *= self.ll_scale
        X_processed[:, 0] = np.clip(X_processed[:, 0], self.vl_bounds[0], self.vl_bounds[1])
        X_processed[:, 1] = np.clip(X_processed[:, 1], self.ll_bounds[0], self.ll_bounds[1])
        return X_processed

    def fit(self, X: np.ndarray, beta: np.ndarray, eta: np.ndarray, train_save_path: str = None):
        """
        Train GPR models in log-domain to ensure strictly positive predictions.
        
        训练 GPR 模型（在 Log 空间训练以保证预测值严格大于 0）。
        """
        self.gp_beta.fit(X, np.log(beta))
        self.gp_eta.fit(X, np.log(eta))

        if train_save_path is not None:
            np.savez(train_save_path, X=X, beta=beta, eta=eta)
            
    def predict_weibull_params(self, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Unified interface implementation: predict beta and eta for given coordinates/features.
        
        统一接口实现：预测指定坐标/特征下的 beta 和 eta。
        """
        X_processed = self._preprocess(inputs)
        beta_pred = np.exp(self.gp_beta.predict(X_processed))
        eta_pred = np.exp(self.gp_eta.predict(X_processed))
        return beta_pred, eta_pred

    def predict_with_std(self, X: np.ndarray, conf_factor: float = 2.0):
        """
        Retain original function for predictions with confidence intervals (for advanced debugging).
        
        保留原代码中带置信区间的预测功能（高级调试使用）。
        """
        X_processed = self._preprocess(X)
        log_beta, std_beta = self.gp_beta.predict(X_processed, return_std=True)
        log_eta,  std_eta  = self.gp_eta.predict(X_processed, return_std=True)

        log_beta, std_beta = np.expand_dims(log_beta, -1), np.expand_dims(std_beta, -1)
        log_eta,  std_eta  = np.expand_dims(log_eta, -1), np.expand_dims(std_eta, -1)

        beta_intervals = np.exp(np.stack([log_beta, log_beta - std_beta * conf_factor, log_beta + std_beta * conf_factor], axis=-1))
        eta_intervals = np.exp(np.stack([log_eta, log_eta - std_eta * conf_factor, log_eta + std_eta * conf_factor], axis=-1))
        return beta_intervals, eta_intervals


# Configuration parameters from original code
# 沿用原代码中的配置参数
DEFAULT_PATH = pathlib.Path('./src/unit_level_models/weibull_gpr_model_simulated_data/')
VIA_DIM_Y = 10.5
VIA_DIM_Z = 21.0
LINE_DIM_X = 10.5
LINE_DIM_Y = 21.0
LINE_DIM_Z = 21.0
RADIUS_N = 2.0


def load_weibull_gpr_data(
        m_param: float = 5.0,
        radius: float = 0.45,
        train_data_path: str = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract raw training data (X, beta, eta) from simulated .npz files.
    This pure data-loading function is perfectly suited for pipeline training.
    
    从仿真的 .npz 文件中提取原始训练数据 (X, beta, eta)。
    该纯数据加载函数非常适合流水线（Pipeline）训练使用。
    """
    # Validate simulated parameter bounds / 验证仿真参数边界
    assert m_param in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    assert radius in [0.45, 0.65]

    # Resolve data path / 解析数据路径
    if train_data_path is not None: 
        data_file = pathlib.Path(train_data_path)
    else:
        dir_name = f'vy{VIA_DIM_Y:.2f}_vz{VIA_DIM_Z:.2f}_lx{LINE_DIM_X:.2f}_ly{LINE_DIM_Y:.2f}_lz{LINE_DIM_Z:.2f}_r{radius:.2f}'
        data_file = DEFAULT_PATH / dir_name / f"weibull_gpr_m{m_param:.2f}_rN{RADIUS_N:.2f}.npz"

    # Load and extract arrays / 加载并提取数组
    data = np.load(data_file)
    X = data["X"]
    beta = data["beta"]
    eta = data["eta"]

    return X, beta, eta


def load_weibull_gpr_model(
        m_param: float = 5.0,
        radius: float = 0.45,
        train_data_path: str = None,
        actual_vl_max: float = 25.0,
        actual_ll_max: float = 12.5
    ):
    """
    Load data and return an instantiated, pre-fitted refactored WeibullGPRModel.
    Maintains backwards compatibility with the original standalone convenience loading feature.
    
    加载数据并返回一个实例化且预训练好的重构 WeibullGPRModel。
    保持与原有独立便捷加载功能的向后兼容性。
    """
    # Reuse the pure data extraction function / 复用纯数据提取函数
    X, beta, eta = load_weibull_gpr_data(m_param, radius, train_data_path)

    # Instantiate the newly refactored model class / 实例化新重构的模型类
    model = WeibullGPRModel(actual_vl_max=actual_vl_max, actual_ll_max=actual_ll_max)
    
    # Train the model on the extracted data / 在提取的数据上训练模型
    model.fit(X=X, beta=beta, eta=eta)

    return model