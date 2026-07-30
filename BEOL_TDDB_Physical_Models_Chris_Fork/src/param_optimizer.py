import copy
import numpy as np
import matplotlib.pyplot as plt
import optuna
from typing import List, Dict, Any, Tuple, Callable

from .unit_level_models.model_factory import unit_model_factory

class TwoStageWaferOptimizer:
    """
    Production-Grade Two-Stage Parameter Optimizer.
    Separates production configs from optimization telemetry/metadata.
    
    工业级双阶段参数优化器。
    将生产环境配置与优化过程的元数据/历史日志完全分离开来。
    """
    def __init__(self, 
                 train_jobs: List[Dict[str, Any]], 
                 val_jobs: List[Dict[str, Any]], 
                 train_labels: List[np.ndarray], 
                 val_labels: List[np.ndarray],
                 batch_process_fn: Callable, 
                 optimize_fn: Callable, 
                 evaluate_fn: Callable,
                 base_config: Dict[str, Any],               # 🌟 注入基础配置
                 search_spaces: Dict[str, Tuple] = None,    # 🌟 注入搜索空间定义
                 batch_size: int = 5,                       # 🌟 注入批大小控制
                 num_workers: int = 4,                      # 🌟 注入并行核心数控制
                 beta: float = 2.0,
                 train_deltas: bool = False,                # 🌟 是否一并训练物理 delta
                 delta_l2: float = 0.0,                     # 🌟 delta 向 0 收缩的 L2 正则权重
                 anchor_ttf: float = None,                  # 🌟 绝对尺度锚点 (秒)；None 关闭
                 anchor_weight: float = 0.0):               # 🌟 锚点惩罚权重
        """
        Args:
            train_jobs / val_jobs: Task paths dictionaries for train/validation. / 训练/验证任务字典列表。
            train_labels / val_labels: 1D ground-truth vectors per wafer. / 按晶圆排列的 1D 真实标签向量列表。
            batch_process_fn: Multi-processing batch manager function. / 顶层多进程调度大总管函数。
            optimize_fn: Function to solve optimal threshold on train set. / 训练集最佳阈值解算函数。
            evaluate_fn: Function to evaluate performance on validation set. / 验证集指标评估函数。
            base_config: The initial pipeline configuration dictionary to be cloned and updated. 
                         初始的流水线配置字典（后续将基于此进行复制与参数替换）。
            search_spaces: User defined search space (人工定义的搜索空间)
            batch_size: Number of wafers per batch during processing. / 每一批处理的晶圆数量。
            num_workers: CPU workers assigned for interpolation. / 分配给插值计算的 CPU 核心数。
            beta: F-beta score weight (e.g., beta=2.0 privileges Recall). / F-beta 评分的权重因子。
        """
        self.train_jobs = train_jobs
        self.val_jobs = val_jobs
        self.train_labels = train_labels
        self.val_labels = val_labels
        self.batch_process_fn = batch_process_fn
        self.optimize_fn = optimize_fn
        self.evaluate_fn = evaluate_fn
        
        # Store execution configurations / 存储运行时控制参数
        self.base_config = base_config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.beta = beta

        # ------------------------------------------------------------------
        # Optional physics-delta training / 可选的物理 delta 训练
        # ------------------------------------------------------------------
        # We ask the unit model itself which parameters it exposes (model-agnostic).
        # A model with no trainable deltas (e.g. GPR) transparently disables this.
        # 直接询问单元模型自身开放了哪些可训练参数（与具体模型解耦）。
        # 没有可训练 delta 的模型（如 GPR）会自动关闭该功能。
        self.train_deltas = train_deltas
        self.delta_l2 = delta_l2
        self.anchor_ttf = anchor_ttf
        self.anchor_weight = anchor_weight
        self.delta_spec: Dict[str, Tuple[float, float, float]] = {}
        if self.train_deltas:
            probe_kwargs = dict(base_config.get("unit_model_kwargs", {}))
            model_type = probe_kwargs.get("model_type")
            if not model_type:
                # e.g. the Linear pipeline has no unit model to train deltas on
                print("[Deltas] Pipeline has no unit model with trainable deltas "
                      "-- delta training disabled.")
                self.train_deltas = False
            else:
                probe = unit_model_factory(target_component="line", **probe_kwargs)
                self.delta_spec = probe.trainable_param_spec()
                if not self.delta_spec:
                    print(f"[Deltas] Model '{model_type}' exposes no trainable deltas "
                          f"-- delta training disabled.")
                    self.train_deltas = False

            if self.train_deltas:
                print(f"[Deltas] Training {len(self.delta_spec)} deltas: "
                      f"{list(self.delta_spec.keys())}")
                if self.anchor_ttf is not None and self.anchor_weight > 0.0:
                    print(f"[Deltas] Anchoring swept threshold -> {self.anchor_ttf:.3e} s "
                          f"(weight {self.anchor_weight}).")
                if self.delta_l2 > 0.0:
                    print(f"[Deltas] L2 shrink toward 0 with weight {self.delta_l2}.")

        # Search space
        default_spaces = {
            "M_structures": (100_000, 5_000_000, 100_000),  # low, high, step
            "F_target": (1e-6, 1e-2),                       # low, high
            "N_samples_per_dim": (1, 16, 1)                 # low, high, step
        }
        if search_spaces:
            default_spaces.update(search_spaces)
        self.search_spaces = default_spaces
        
        # Pre-concatenate full ground-truth vectors / 预先拼装全量真实标签向量
        self.y_train_true_flat = np.concatenate(train_labels)
        self.y_val_true_flat = np.concatenate(val_labels)
        
        # Metrics logger / 历史轨迹追踪器
        self.history = {
            "iteration": [],
            "train_f_score": [],
            "val_f_score": [],
            "solutions": [],
            "params": []
        }
        self.iter_counter = 0

    def _align_and_flatten_predictions(self, raw_outputs: List[Tuple[np.ndarray, Any]], 
                                        reference_labels: List[np.ndarray], stage_name: str) -> np.ndarray:
        """Align predictions wafer-by-wafer with reference labels and perform strict length assertions."""
        pred_vectors = []
        for idx, (ttf_pred_vector, _) in enumerate(raw_outputs):
            true_vector = reference_labels[idx]
            # CRITICAL ASSERTION: Ensure spatial consistency / 核心断言：严格确保点数完全一致
            assert len(ttf_pred_vector) == len(true_vector), (
                f"[{stage_name} Error] Wafer index {idx} length mismatch! "
                f"Predicted: {len(ttf_pred_vector)}, Reference: {len(true_vector)}."
            )
            pred_vectors.append(ttf_pred_vector)
        return np.concatenate(pred_vectors)

    def objective(self, trial: optuna.Trial) -> float:
        self.iter_counter += 1
        
        # 1. Clone the base configuration and overlay the proposed hyper-parameters
        # 深度复制原始 config，并用当前轮次外层推荐的 3D 参数进行覆盖替换
        current_config = copy.deepcopy(self.base_config)
        m_space = self.search_spaces["M_structures"]
        f_space = self.search_spaces["F_target"]
        n_space = self.search_spaces["N_samples_per_dim"]
        
        current_config.update({
            "M_structures": trial.suggest_int("M_structures", m_space[0], m_space[1], step=m_space[2]),
            "F_target": trial.suggest_float("F_target", f_space[0], f_space[1], log=True),
            "N_samples_per_dim": trial.suggest_int("N_samples_per_dim", n_space[0], n_space[1], step=n_space[2])
        })

        # Add the physics deltas as extra search dimensions (if enabled)
        # 若开启，则把物理 delta 作为额外的搜索维度加入本轮试验
        trial_deltas = {}
        if self.train_deltas:
            deltas = current_config["unit_model_kwargs"].setdefault("deltas", {})
            for name, (_default, low, high) in self.delta_spec.items():
                deltas[name] = trial.suggest_float(name, low, high)
            trial_deltas = deltas
        
        # Force all background tasks into lightweight 'ttf_only' mode
        # 强制将本轮调度任务指定为轻量级内存级计算模式
        for job in self.train_jobs + self.val_jobs:
            job["mode"] = "ttf_only"
            job["output_ttf_path"] = None  
            job["output_binary_path"] = None

        # 2. Drive the simulation engine using the injected batching parameters
        # 使用类内部注入的批大小与核心数控制，并行驱动物理模型跑批
        train_outputs = self.batch_process_fn(
            self.train_jobs, current_config, batch_size=self.batch_size, num_workers=self.num_workers
        )
        val_outputs = self.batch_process_fn(
            self.val_jobs, current_config, batch_size=self.batch_size, num_workers=self.num_workers
        )
        
        # 3. Extract and align predictions / 抽取并对齐预测向量
        y_train_pred_flat = self._align_and_flatten_predictions(train_outputs, self.train_labels, "Train-Stage")
        y_val_pred_flat = self._align_and_flatten_predictions(val_outputs, self.val_labels, "Validation-Stage")
        
        # 4. Second stage: Solve for the optimal solution analytically on Train Set
        # 第二阶段：在训练集上直接解出最佳决策参数（如二分类最佳门限）
        best_sol, train_f = self.optimize_fn(y_train_pred_flat, self.y_train_true_flat, self.beta)
        
        # 5. Evaluate generalization performance on Validation Set
        # 使用该方案在验证集上泛化评估，作为 Optuna 的唯一反馈反馈指标
        val_f = self.evaluate_fn(y_val_pred_flat, self.y_val_true_flat, best_sol, self.beta)
        
        # 6. Record tracking history / 记录本轮追踪指标
        #    history keeps the RAW validation F-score for interpretability;
        #    the value returned to Optuna may additionally carry soft penalties.
        #    历史里保存原始验证 F 分数以便解读；返回给 Optuna 的值可能额外带软惩罚。
        self.history["iteration"].append(self.iter_counter)
        self.history["train_f_score"].append(train_f)
        self.history["val_f_score"].append(val_f)
        self.history["solutions"].append(best_sol)
        self.history["params"].append(current_config)

        # 7. Apply optional soft objective terms (delta training only)
        # 7. 施加可选的软目标项（仅在训练 delta 时生效）
        score = val_f
        if self.train_deltas and self.delta_l2 > 0.0:
            # L2 shrink toward the physical baseline (delta=0), normalised per bound range
            # 向物理基线 (delta=0) 收缩的 L2 正则，按各自边界范围归一化
            norm_sq = []
            for name, (default, low, high) in self.delta_spec.items():
                rng = (high - low) or 1.0
                norm_sq.append(((trial_deltas[name] - default) / rng) ** 2)
            score -= self.delta_l2 * float(np.mean(norm_sq))
        if self.anchor_ttf is not None and self.anchor_weight > 0.0 and best_sol > 0.0:
            # Anchor the swept decision boundary to a physical lifetime (log-space)
            # 将扫掠得到的决策边界锚定到某个物理寿命上（对数空间）
            score -= self.anchor_weight * (np.log(best_sol) - np.log(self.anchor_ttf)) ** 2

        return score

    def run_optimization(self, n_trials: int = 30, seed: int = 42) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Launch the Bayesian optimization pipeline and return decoupled result files.
        启动贝叶斯寻优流水线。
        
        Args:
            n_trials: 优化的迭代轮数。
            seed: 随机种子，保证优化轨迹绝对可复现。
        """
        # 🌟 关键修改：显式创建一个带有固定种子的 TPE 采样器
        sampler = optuna.samplers.TPESampler(seed=seed)
        
        # 将采样器注入到 study 中
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(self.objective, n_trials=n_trials)
        
        best_trial_idx = study.best_trial.number
        opt_solution = self.history["solutions"][best_trial_idx]
        
        # ======================================================================
        # FILE 1: Optimized Config / 文件 1：优化后的纯净配置（直接用于生产投产）
        # ======================================================================
        optimized_config = copy.deepcopy(self.base_config)
        optimized_config.update({
            "M_structures": study.best_params["M_structures"],
            "F_target": study.best_params["F_target"],
            "N_samples_per_dim": study.best_params["N_samples_per_dim"],
            "threshold": opt_solution  # Injecting the analytically resolved decision boundary
        })

        # Persist the winning physics deltas so config.json round-trips into inference
        # 将胜出的物理 delta 写入配置，使 config.json 能无缝还原到推理阶段
        best_deltas = {}
        if self.train_deltas:
            optimized_config.setdefault("unit_model_kwargs", {})
            best_deltas = {name: float(study.best_params[name]) for name in self.delta_spec}
            optimized_config["unit_model_kwargs"]["deltas"] = best_deltas
        
        # ======================================================================
        # FILE 2: Optimization Metadata & History / 文件 2：优化元数据与历史追踪日志
        # ======================================================================
        optimization_metadata = {
            "optimization_method_summary": {
                "algorithm": "Two-Stage Profile Likelihood (Optuna + Exact Threshold Sweeping)",
                "total_iterations": n_trials,
                "beta_parameter": self.beta,
                "best_trial_number": best_trial_idx + 1,
                "best_train_f_score": float(self.history["train_f_score"][best_trial_idx]),
                "best_val_f_score": float(self.history["val_f_score"][best_trial_idx]),
                "best_objective_value": float(study.best_value),  # F minus any soft penalties
                "trained_deltas": self.train_deltas,
                "best_deltas": best_deltas,
                "delta_l2": self.delta_l2,
                "anchor_ttf_seconds": self.anchor_ttf,
                "anchor_weight": self.anchor_weight
            },
            "history_telemetry": {
                "iteration": list(self.history["iteration"]),
                "train_f_scores": [float(x) for x in self.history["train_f_score"]],
                "val_f_scores": [float(x) for x in self.history["val_f_score"]],
                "resolved_solutions": [float(x) for x in self.history["solutions"]]
            }
        }
        
        return optimized_config, optimization_metadata

    def plot_history(self, save_image_path: str = "optimization_curve.png"):
        """Generate and save the train vs validation convergence curve."""
        plt.figure(figsize=(10, 5), dpi=150)
        plt.plot(self.history["iteration"], self.history["train_f_score"], 
                 'o-', color='#1f77b4', label=f'Train F-{self.beta} Score')
        plt.plot(self.history["iteration"], self.history["val_f_score"], 
                 's--', color='#ff7f0e', label=f'Validation F-{self.beta} Score')
        
        plt.title("Parameter Optimization Convergence History", fontsize=12, fontweight='bold')
        plt.xlabel("Iteration / Trial", fontsize=10)
        plt.ylabel("F-score Performance", fontsize=10)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc="lower right", fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_image_path)
        plt.close()
        print(f"[Visualization] Optimization curve saved successfully to: {save_image_path}")