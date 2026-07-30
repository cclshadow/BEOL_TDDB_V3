import numpy as np
import pathlib
from abc import ABC, abstractmethod

# ==============================================================================
# Unified Base Model Abstract Class / 统一基础模型抽象基类
# ==============================================================================
class BaseTDDBModel(ABC):
    """
    Abstract base class for TDDB base models.
    All underlying reliability models (GPR or physics-based) must inherit from 
    this class and implement a unified interface.
    
    TDDB 基础模型的抽象基类。
    所有底层可靠性模型（无论是基于GPR插值还是纯物理推导）都必须继承此类并实现统一的接口。
    """
    @abstractmethod
    def predict_weibull_params(self, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pass

    def predict_weibull_log_params(self, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return Weibull ``beta`` and ``ln(eta)``.

        Models with a native log-scale calculation should override this method
        so very large scale parameters are never exponentiated first.
        """
        beta, eta = self.predict_weibull_params(inputs)
        return beta, np.log(eta)

    # ==========================================================================
    # Trainable-parameter interface / 可训练参数接口
    # ==========================================================================
    def trainable_param_spec(self) -> dict:
        """
        Declare which internal parameters this model exposes for training.
        声明本模型对外开放、可供训练器搜索的内部参数。

        Returns an ordered mapping ``{name: (default, low, high)}`` where ``low``
        and ``high`` are physically-informed bounds. The optimizer treats every
        entry as one search dimension and hands the fitted values back to the
        constructor via ``unit_model_kwargs["deltas"]``. Models with no trainable
        parameters (e.g. the GPR surrogate) return an empty dict and are simply
        left un-searched.

        返回一个有序字典 ``{参数名: (默认值, 下界, 上界)}``，下界/上界为物理上
        合理的范围。优化器会把每一项当作一个搜索维度，并通过
        ``unit_model_kwargs["deltas"]`` 把拟合结果回传给构造函数。没有可训练参数
        的模型（例如 GPR 代理模型）返回空字典，训练器会自动跳过它。
        """
        return {}
