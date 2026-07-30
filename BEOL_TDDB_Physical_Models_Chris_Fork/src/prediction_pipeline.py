import numpy as np
import json

from .unit_level_models.model_factory import unit_model_factory
from .wafer_level_models.wafer_reliability_engines import GPRWaferEngine, DPMWaferEngine
from .wafer_level_models.linear_engine import LinearWaferEngine
from .reliability_classifiers.binary_classifier import BinaryReliabilityClassifier

# ==============================================================================
# Top-Level Assembly Pipeline (Refactored) / 顶层组装流水线类（重构版）
# ==============================================================================
class IntegratedWaferPipeline:
    """
    Unified 3-Layer Reliability Pipeline.
    Dynamically parses a configuration dictionary to assemble Layer 1, 2, and 3.
    Supports running continuous lifetime simulation independently of decision thresholds.
    
    大一统三层可靠性流水线。
    动态解析配置字典，一键组装第一、二、三层模型。
    支持在没有设置决策阈值的情况下，独立运行连续寿命仿真。
    """
    def __init__(self, config: dict):
        """
        Args:
            config: A comprehensive dictionary defining architecture types, factory functions,
                    and initial tunable parameters. 'threshold' can be omitted or set to None.
                    包含架构类型、工厂函数以及初始可调参数的完整配置字典。'threshold' 可省略或设为 None。
        """
        # 1. Parse meta architectural configurations / 解析底层模型元配置
        self.pipeline_type = config["pipeline_type"]            # "GPR" or "DPM"
        self.unit_model_kwargs = config.get("unit_model_kwargs", {}) # 可选参数
        
        # 2. Extract Core Tunable Parameters / 提取核心可调参数
        self.M_structures = config["M_structures"]
        self.F_target = config["F_target"]
        self.N_samples_per_dim = config["N_samples_per_dim"]
        
        # Safe extraction of threshold (Defaults to None if missing)
        # 安全提取阈值（若缺省则默认为 None）
        self.threshold = config.get("threshold", None)

        # 3. Trigger dynamic assembly / 触发动态组装
        self._build_pipeline_layers()

    def _build_pipeline_layers(self):
        """Internal factory method to instantiate the 3-layer architecture."""
        # ----------------------------------------------------------------------
        # Assembly Layer 1 & 2: Physics/ML Engine / 组装第一、二层：物理/机器学习引擎
        # ----------------------------------------------------------------------
        if self.pipeline_type == "GPR":
            self.unit_model_joint = unit_model_factory(**self.unit_model_kwargs)
            self.engine = GPRWaferEngine(
                unit_model_joint=self.unit_model_joint,
                M_structures=self.M_structures,
                F_target=self.F_target
            )
            
        elif self.pipeline_type == "DPM":
            self.unit_model_via = unit_model_factory(target_component='via', **self.unit_model_kwargs)
            self.unit_model_line = unit_model_factory(target_component='line', **self.unit_model_kwargs)
            self.engine = DPMWaferEngine(
                unit_model_via=self.unit_model_via,
                unit_model_line=self.unit_model_line,
                M_structures=self.M_structures,
                F_target=self.F_target
            )
        elif self.pipeline_type == "Linear":
            self.engine = LinearWaferEngine(
                M_structures=self.M_structures,
                F_target=self.F_target
            )

        else:
            raise ValueError(f"Unsupported pipeline type: {self.pipeline_type}")

        # ----------------------------------------------------------------------
        # Assembly Layer 3: Classifier Boundary / 组装第三层：分类器边界网闸
        # ----------------------------------------------------------------------
        # Only instantiate the classifier if threshold is explicitly provided
        # 只有在明确提供了阈值的情况下，才实例化第三层分类器
        if self.threshold is not None:
            self.classifier = BinaryReliabilityClassifier(threshold=self.threshold)
        else:
            self.classifier = None

    # ==============================================================================
    # NEW FUNCTION: Pure TTF Lifetime Simulation / 仅预测连续寿命（新设函数）
    # ==============================================================================
    def predict_ttf(self, x_die: np.ndarray, y_die: np.ndarray, 
                    vl_data: np.ndarray, ll_data: np.ndarray) -> np.ndarray:
        """
        Pure continuous simulation interface (Layer 1 + Layer 2 only).
        Runs successfully even if 'threshold' is not configured.
        
        纯连续仿真接口（仅运行第一层+第二层）。
        即使没有配置 'threshold'（阈值），该方法也能完全正常运行。
        
        Returns:
            np.ndarray: Continuous time-to-failure (TTF) arrays for each die.
                        各个 Die 的连续失效寿命（TTF）数组。
        """
        # Execute spatial interpolation + weakest link equation solving
        # 直接执行空间插值与最薄弱环节方程求解，返回寿命数组
        ttf_lifetimes = self.engine.predict_die_lifetimes(
            x_die=x_die, y_die=y_die, vl_data=vl_data, ll_data=ll_data,
            N_samples_per_dim=self.N_samples_per_dim
        )
        return ttf_lifetimes

    # ==============================================================================
    # MODIFIED FUNCTION: Binary Classification Decision / 二分类决策（修改后）
    # ==============================================================================
    def predict(self, x_die: np.ndarray, y_die: np.ndarray, 
                vl_data: np.ndarray, ll_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Executes end-to-end inference and returns BOTH quantitative lifetimes and binary decisions.
        执行端到端推理，同时输出定量的连续失效时间（Breakdown Time）与定性的二分类决策。
        
        Returns:
            tuple (ttf_lifetimes, binary_labels)
        """
        if self.classifier is None:
            raise ValueError("Pipeline Error: 'threshold' has not been set yet! Cannot run predict().")
            
        # 1. Get breakdown times / 获取击穿寿命值
        ttf_lifetimes = self.predict_ttf(x_die, y_die, vl_data, ll_data)
        
        # 2. Get binary classification flags / 获取分类标记
        binary_labels = self.classifier.classify(ttf_lifetimes)
        
        return ttf_lifetimes, binary_labels

    # ==============================================================================
    # Parameters Tuning Interface / 参数更新接口（联动热更新决策层）
    # ==============================================================================
    def update_tunable_parameters(self, M_structures: float = None, F_target: float = None, 
                                   N_samples_per_dim: int = None, threshold: float = None):
        """
        Dynamically hot-swap parameters and automatically manage Classifier lifecycles.
        """
        if M_structures is not None:
            self.M_structures = M_structures
            self.engine.M = self.M_structures
        if F_target is not None:
            self.F_target = F_target
            self.engine.target_hazard = -np.log(1.0 - self.F_target)
        if N_samples_per_dim is not None:
            self.N_samples_per_dim = N_samples_per_dim

        if threshold is not None:
            self.threshold = threshold
            # Dynamic Hot-Swapping Logic for Layer 3 / 第三层分类器的动态热重载逻辑
            if self.classifier is None:
                # If it was previously unconfigured, instantiate it now to activate the gate
                # 如果此前未配置阈值，现在立马实例化它，激活决策门
                self.classifier = BinaryReliabilityClassifier(threshold=self.threshold)
            else:
                # If it already exists, simply refresh its threshold value
                # 如果已经存在，则直接更新其内部阈值
                self.classifier.threshold = self.threshold

    # ==============================================================================
    # Parameters Persistence (Save & Load) / 参数持久化接口
    # ==============================================================================
    def save_tuned_parameters(self, file_path: str):
        tuned_params = {
            "pipeline_type": self.pipeline_type,
            "M_structures": float(self.M_structures),
            "F_target": float(self.F_target),
            "N_samples_per_dim": int(self.N_samples_per_dim),
            "threshold": float(self.threshold) if self.threshold is not None else None
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(tuned_params, f, indent=4, ensure_ascii=False)

    def load_tuned_parameters(self, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as f:
            tuned_params = json.load(f)
            
        if tuned_params["pipeline_type"] != self.pipeline_type:
            raise ValueError(f"Type mismatch!")
            
        self.update_tunable_parameters(
            M_structures=tuned_params["M_structures"],
            F_target=tuned_params["F_target"],
            N_samples_per_dim=tuned_params["N_samples_per_dim"],
            threshold=tuned_params["threshold"]
        )