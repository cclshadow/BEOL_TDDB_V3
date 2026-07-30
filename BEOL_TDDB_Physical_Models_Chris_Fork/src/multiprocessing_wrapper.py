from multiprocessing import Pool, cpu_count as get_cpu_count
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from tqdm import tqdm

from .csv_utils import (
    load_matrix_csv,
    matrix_to_sparse_points,
    sparse_points_to_matrix,
    save_matrix_csv
)
from .prediction_pipeline import IntegratedWaferPipeline

# ==============================================================================
# Global Worker Memory / 子进程全局内存空间
# ==============================================================================
# Placeholder for the singleton pipeline instance inside each worker process.
# 每个子进程内部单例流水线对象的占位符，避免重复加载模型。
_worker_pipeline = None

def _init_worker_process(config: dict):
    """
    Worker Initializer: Executed exactly once per CPU core upon process creation.
    Instantiates the IntegratedWaferPipeline in the worker's memory space.
    
    子进程初始化器：在创建进程池时，每个 CPU 核心仅执行一次。
    在子进程内存空间中实例化 IntegratedWaferPipeline。
    """
    global _worker_pipeline
    _worker_pipeline = IntegratedWaferPipeline(config)


def _process_and_save_by_path(path_job: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Core Path Worker: Operates purely on file paths to decouple from business logic.
    Supports flexible execution modes and safe fallback on exceptions.
    
    纯路径工作器：完全基于文件路径操作以解耦业务逻辑。
    支持灵活的执行模式（仅寿命或双输出），并在发生异常时安全降级。
    
    Args:
        path_job (Dict[str, Any]): A dictionary containing input paths, optional output paths, and execution mode.
                                   包含输入路径、可选输出路径以及执行模式的字典。
                                   
    Returns:
        Tuple[Optional[np.ndarray], Optional[np.ndarray]]: 
            A tuple of (ttf_matrix, binary_matrix). Returns (None, None) if processing fails.
            包含 (连续寿命矩阵, 二分类矩阵) 的元组。如果处理失败则返回 (None, None)。
    """
    global _worker_pipeline
    
    # Extract paths and mode (defaulting to 'both' if not specified)
    # 提取输入、输出路径及运行模式（如果未指定，默认执行全套预测）
    vl_path = path_job["vl_path"]
    ll_path = path_job["ll_path"]
    output_ttf_path = path_job.get("output_ttf_path", None)
    output_binary_path = path_job.get("output_binary_path", None)
    mode = path_job.get("mode", "both") 

    try:
        # 1. Read and parse original CSV matrices / 读取并解析原始 CSV 矩阵
        vl_matrix = load_matrix_csv(vl_path)
        ll_matrix = load_matrix_csv(ll_path)
        
        # 2. Convert to sparse spatial points and capture original shape 
        #    空间稀疏化转换，并捕获原始矩阵形状用于后续重建
        x, y, vl_val, orig_shape = matrix_to_sparse_points(vl_matrix)
        _, _, ll_val, _ = matrix_to_sparse_points(ll_matrix)
        
        # 3. Execute pipeline prediction based on the selected mode
        #    根据指定的 mode 执行对应的流水线预测
        if mode == "ttf_only":
            # Run Layer 1 & 2 only (No threshold constraints) / 仅调用第一、二层（无需阈值限制）
            ttf_lifetimes = _worker_pipeline.predict_ttf(
                x_die=x, y_die=y, vl_data=vl_val, ll_data=ll_val
            )
            binary_labels = None
            
        elif mode == "both":
            # Run full pipeline: Layer 1, 2 & 3 / 双输出模式，调用全套流水线
            ttf_lifetimes, binary_labels = _worker_pipeline.predict(
                x_die=x, y_die=y, vl_data=vl_val, ll_data=ll_val
            )
        else:
            raise ValueError(f"Unknown job mode: {mode}")
        
        # 4. Inverse reconstruction to dense 2D matrices and optional disk storage
        #    逆向还原为 2D 密集矩阵，并进行可选的磁盘存储
        
        # Reconstruct and save TTF matrix (save_matrix_csv safely ignores None paths)
        # 重建并保存连续寿命矩阵（save_matrix_csv 函数内部已处理路径为 None 的跳过逻辑）
        dense_ttf_matrix = sparse_points_to_matrix(x, y, ttf_lifetimes, orig_shape)
        save_matrix_csv(dense_ttf_matrix, output_ttf_path)

        # Reconstruct and save Binary matrix if applicable
        # 如果存在二分类结果，则重建并保存二分类矩阵
        dense_binary_matrix = None
        if binary_labels is not None:
            dense_binary_matrix = sparse_points_to_matrix(x, y, binary_labels, orig_shape)
            save_matrix_csv(dense_binary_matrix, output_binary_path)
        
        return ttf_lifetimes, binary_labels

    except Exception as e:
        # Robust error handling: Return None tuple to prevent worker crash and pool hang.
        # 强健的异常处理：遇到脏数据等错误时返回 None 元组，保障进程池整体队列的稳定性。
        print(f"[Worker Error] Failed to process {vl_path}: {str(e)}")
        return None, None


# ==============================================================================
# Top-Level Parallel Dispatcher / 顶层并行分发包装器
# ==============================================================================
def batch_process_by_path(jobs_list: List[Dict[str, Any]], 
                          pipeline_config: dict, 
                          batch_size: int = 20,
                          num_workers: int = 0) -> List[Tuple[Optional[np.ndarray], Optional[np.ndarray]]]:
    """
    Pure path-based parallel dispatcher with progress bar. 
    Strictly preserves input order and returns reconstructed matrices.
    
    集成进度条的纯路径并行分发器。
    支持自定义 CPU 核心数，并严格保序返回重建后的结果矩阵列表。
    
    Args:
        jobs_list (List[Dict[str, Any]]): List of task dictionaries. / 任务字典列表。
        pipeline_config (dict): Unified pipeline configuration. / 统一的流水线配置字典。
        batch_size (int): Task chunk size per worker. / 每个子进程一次领取的任务粒度。
        num_workers (int): Number of CPU cores to use. 0 defaults to all available cores. 
                           使用的 CPU 核心数。传入 0 则默认拉满所有可用核心。
                           
    Returns:
        List[Tuple[Optional[np.ndarray], Optional[np.ndarray]]]: Ordered list of output matrices.
                                                                 按输入顺序排列的输出矩阵列表。
    """
    # Determine the actual number of workers to deploy / 动态决定调用的物理核心数
    max_workers = num_workers if num_workers > 0 else get_cpu_count()
    results = []

    print(f"[Parallel Engine] Launching path-based pool with {max_workers} cores...")
    print(f"[Parallel Engine] Total files to process: {len(jobs_list)}")

    # Launch multiprocessing Pool with persistent initializer
    # 启动多进程池，并为每个核心注入初始化配置
    with Pool(processes=max_workers, initializer=_init_worker_process, initargs=(pipeline_config,)) as pool:
        
        # Stream tasks in chunks via imap to preserve strict ordering, wrapped in tqdm for UI progress
        # 通过 imap 以 chunk 为单位分发任务以严格保序，并使用 tqdm 包装以显示进度 UI
        path_iterator = pool.imap(_process_and_save_by_path, jobs_list, chunksize=batch_size)
        
        for result in tqdm(path_iterator, total=len(jobs_list), desc="Processing Wafers"):
            results.append(result)
                
    return results