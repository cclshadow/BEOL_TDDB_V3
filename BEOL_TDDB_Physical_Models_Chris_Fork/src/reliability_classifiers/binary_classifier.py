import numpy as np
from typing import Tuple

# ==============================================================================
# 1. Standalone Metric Function / 独立的评估指标函数
# ==============================================================================
def calculate_f_beta_score(y_true: np.ndarray, y_pred: np.ndarray, 
                             beta: float = 2.0, pos_label: int = 0) -> float:
    """
    Calculate the F-beta score for binary classification.
    Designed for highly imbalanced semiconductor data where catching rare failures is critical.
    
    计算二分类的 F-beta 分数。
    专为极度不平衡的半导体数据设计，其中捕获罕见的失效芯片（Bad Die）是核心任务。
    """
    tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
    fp = np.sum((y_true != pos_label) & (y_pred == pos_label)) # Overkill / 误杀
    fn = np.sum((y_true == pos_label) & (y_pred != pos_label)) # Escape / 漏检
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    beta_sq = beta ** 2
    denominator = (beta_sq * precision) + recall
    
    if denominator == 0:
        return 0.0
        
    f_beta = (1 + beta_sq) * (precision * recall) / denominator
    return float(f_beta)


# ==============================================================================
# 2. Layer 3 Classifier Class / 第三层分类器类
# ==============================================================================
class BinaryReliabilityClassifier:
    """
    Layer 3 Classifier: Maps continuous Time-to-Failure (TTF) into binary decisions.
    Class 0: Low Reliability / Bad Die (TTF < threshold) -> Intercept
    Class 1: High Reliability / Good Die (TTF >= threshold) -> Pass
    
    第三层分类器：将连续的失效时间 (TTF) 映射为二分类决策。
    分类 0：低可靠性 / 坏芯片 (TTF < 阈值) -> 拦截
    分类 1：高可靠性 / 好芯片 (TTF >= 阈值) -> 放行
    """
    def __init__(self, threshold: float):
        """
        Args:
            threshold: The cutoff lifetime separating bad and good dies.
                       区分好坏芯片的寿命截断阈值。
        """
        self.threshold = threshold

    def classify(self, ttf_array: np.ndarray) -> np.ndarray:
        """
        Execute binary thresholding. / 执行二分类阈值切分。
        """
        return np.where(ttf_array >= self.threshold, 1, 0)


# ==============================================================================
# 3. Abstracted Optimizer Inputs / 抽象出的优化器输入函数（基于二分类前提）
# ==============================================================================
def optimize_binary_threshold(ttf_flat: np.ndarray, labels_flat: np.ndarray, beta: float = 2.0) -> Tuple[float, float]:
    """
    O(N log N) Global Optimal Threshold Sweeper matching the pos_label=0 (Fail) convention.
    O(N log N) 的全局最优阈值扫掠器，完美匹配 pos_label=0 (Fail) 的拦截约定。
    
    Args:
        ttf_flat: Flattened 1D array of valid TTF predictions. / 展平的有效 TTF 预测 1D 数组。
        labels_flat: Flattened 1D array of reference binary labels. / 展平的参考二分类标签 1D 数组。
        beta: F-beta weight parameter. / F-beta 权重参数。
        
    Returns:
        Tuple[float, float]: (best_threshold, best_f_score)
    """
    # Sort predictions and labels synchronously based on TTF ascending order
    # 根据 TTF 升序对预测值和真实标签进行同步排序
    sort_idx = np.argsort(ttf_flat)
    ttf_sorted = ttf_flat[sort_idx]
    labels_sorted = labels_flat[sort_idx]
    
    total_fails = np.sum(labels_sorted == 0) # Target class / 核心截留目标数量
    if total_fails == 0:
        return float(np.max(ttf_flat) + 1.0), 0.0

    # Initial state: Threshold is set lower than all values -> All predicted as Pass (1).
    # 初始状态：阈值设得比所有预测值都小 -> 所有人预测为放行 (1)。
    # Therefore, for pos_label=0: TP=0, FP=0, FN=total_fails
    tp = 0
    fp = 0
    fn = total_fails
    
    best_f_score = -1.0
    best_threshold = float(ttf_sorted[0]) - 1.0
    beta_sq = beta ** 2
    
    # Sweep through each potential split point from left to right
    # 从左到右扫掠每一个可能的阈值切分点
    for i in range(len(ttf_sorted) - 1):
        if labels_sorted[i] == 0: # An actual failure enters the intercept zone (prediction becomes 0)
            tp += 1               # 真实的失效芯片进入了拦截区（预测变为了 0，TP 增加）
            fn -= 1
        else:                     # An actual good pass enters the intercept zone (prediction becomes 0)
            fp += 1               # 真实的好芯片误入了拦截区（预测变为了 0，FP 增加）
            
        # Avoid splitting between identical continuous values / 避免在连续相同预测值中间画刀
        if ttf_sorted[i] == ttf_sorted[i+1]:
            continue
            
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        denominator = (beta_sq * precision) + recall
        f_score = (1 + beta_sq) * (precision * recall) / denominator if denominator > 0 else 0.0
            
        if f_score > best_f_score:
            best_f_score = f_score
            best_threshold = float((ttf_sorted[i] + ttf_sorted[i+1]) / 2.0)
            
    return best_threshold, max(0.0, best_f_score)


def evaluate_binary_threshold(ttf_flat: np.ndarray, labels_flat: np.ndarray, threshold: float, beta: float = 2.0) -> float:
    """
    Evaluate the F-beta score using a fixed threshold.
    使用固定的阈值计算二分类的 F-beta 分数。
    """
    classifier = BinaryReliabilityClassifier(threshold)
    y_pred = classifier.classify(ttf_flat)
    return calculate_f_beta_score(y_true=labels_flat, y_pred=y_pred, beta=beta, pos_label=0)