from .weibull_gpr_model import load_weibull_gpr_model
from .dpm_models import (PowerLawDPMModel, SqrtEDPMModel, InverseEDPMModel)
from .dpm_models_scaled import (PowerLawDPMModel as PowerLawDPMModelScaled,
                                SqrtEDPMModel    as SqrtEDPMModelScaled,
                                InverseEDPMModel as InverseEDPMModelScaled)

# ==============================================================================
# Unified Factory Registry / 大一统工厂模型注册表
# ==============================================================================
# "PowerLaw" / "SqrtE" / "InverseE"        — original unscaled models
# "PowerLawScaled" / "SqrtEScaled" / "InverseEScaled" — scaled variants;
#     pass spacing_scale=k in unit_model_kwargs to set the divisor
MODEL_REGISTRY = {
    "GPR":             load_weibull_gpr_model,
    "PowerLaw":        PowerLawDPMModel,
    "SqrtE":           SqrtEDPMModel,
    "InverseE":        InverseEDPMModel,
    "PowerLawScaled":  PowerLawDPMModelScaled,
    "SqrtEScaled":     SqrtEDPMModelScaled,
    "InverseEScaled":  InverseEDPMModelScaled,
}


# ==============================================================================
# Factory Function Implementation / 工厂函数核心实现
# ==============================================================================
def unit_model_factory(model_type: str, target_component: str = None, **kwargs):
    """
    Unified entry point to fetch Layer 1 unit-level models.
    - If model_type is 'GPR', it directly calls 'load_weibull_gpr_model(...)' to return a fitted model.
    - If model_type is any DPM variant, it instantiates the physics class directly.
    
    统一的 Unit 级别微观模型获取入口。
    - 如果 model_type 是 'GPR'，则直接调用你提供的 'load_weibull_gpr_model' 函数，
      它会在内部完成数据读取和 `.fit()` 训练，返回一个已训练好的可用模型实例。
    - 如果 model_type 是任何 DPM 变体，则直接实例化对应的纯物理机制类。
    
    Args:
        model_type: Key name from MODEL_REGISTRY (e.g., "GPR", "InverseE").
                    来自注册表的键名。
        target_component: Optional string for DPM models ("via" or "line"). 
                         GPR joint models will automatically ignore this.
                         DPM 模型可选的子组件参数（"via" 或 "line"）。GPR 联合模型会自动忽略。
        **kwargs: Flexible pass-through parameters (e.g., actual_vl_max, m_param, radius, etc.).
                  透传给加载函数或类构造函数的灵活参数。
                  
    Returns:
        An operational unit-level model instance with 'predict_weibull_params' method.
        一个带有 'predict_weibull_params' 方法的、立即可用的第一层微观模型实例。
    """
    normalized_type = model_type.strip()
    if normalized_type not in MODEL_REGISTRY:
        raise ValueError(f"Model type '{model_type}' is not registered in factory! "
                         f"Available types: {list(MODEL_REGISTRY.keys())}")
                         
    target_entity = MODEL_REGISTRY[normalized_type]

    # --- 条件 1：是 GPR 模型（需要调用加载函数，执行你的自动化数据加载+训练逻辑） ---
    if normalized_type == "GPR":
        # 直接执行函数，由于你写的函数支持 actual_vl_max、m_param 等作为参数，
        # 用 **kwargs 透传过去刚好契合。
        return target_entity(**kwargs)

    # --- 条件 2：是 DPM 物理模型（不需要加载训练数据，直接实例化类） ---
    else:
        if target_component is None:
            raise ValueError(f"DPM physics models require a specified 'target_component' (via/line), got None.")
        
        # 直接实例化对应的物理类，并传入当前流水线组装所需的子组件参数（'via' 或 'line'）
        return target_entity(target_component=target_component, **kwargs)