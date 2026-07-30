import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Dict
import multiprocessing

from train import find_all_wafer_paths
from src.multiprocessing_wrapper import batch_process_by_path
from src.csv_utils import (
	load_matrix_csv,
	matrix_to_sparse_points,
	)
from src.plot_utils.inference_wafer_maps import plot_five_trends


def generate_and_create_mirrored_paths(data_in: str, data_out: str) -> List[Dict[str, str]]:
    """
    Scan data_in for wafer directories, check and create mirrored paths under data_out 
    (with a creation message), and return a list of paired input/output path dictionaries.
    
    扫描 data_in 下的晶圆目录，检查并在 data_out 下按需创建对应的镜像路径（附带创建提示），
    同时收集并返回包含输入和输出路径对的任务字典列表。
    
    Args:
        data_in (str): Root directory of the input source dataset (e.g., "data_in/").
                       输入的数据集源根目录（例如 "data_in/"）。
        data_out (str): Root directory for the desired output (e.g., "data_out/").
                        期望输出的目标根目录（例如 "data_out/"）。
                        
    Returns:
        List[Dict[str, str]]: A list of paired paths for jobs, even if directories already existed.
                              成对的绝对路径任务列表（无论文件夹是新创建的还是原本就存在的都会返回）。
                              Example / 示例: [{'input_path': '...', 'output_path': '...'}, ...]
    """
    # Convert string paths to pathlib.Path objects for robust cross-platform path manipulation.
    # 将字符串路径转换为 pathlib.Path 对象，以实现健壮的跨平台路径操作。
    in_root = Path(data_in)
    out_root = Path(data_out)
    
    # Ensure the source input directory actually exists before proceeding.
    # 在继续执行之前，确保输入的源目录真实存在。
    if not in_root.exists():
        raise FileNotFoundError(f"Source directory does not exist / 源目录不存在: {data_in}")
        
    # Define the structural wildcard pattern to match the target wafer directories.
    # 定义匹配晶圆文件夹层级的通配符结构规则。
    search_pattern = "lot_*/csv/wafer_*"
    
    jobs_paths = []
    
    # Traverse all directories matching the pattern. sorted() ensures deterministic order.
    # 遍历所有符合规则的目录。sorted() 保证在不同系统下的遍历顺序完全一致（保证实验复现性）。
    for in_wafer_path in sorted(in_root.glob(search_pattern)):
        if in_wafer_path.is_dir():
            
            # Step 1: Calculate the relative sub-path from data_in (e.g., 'lot_001/csv/wafer_01').
            # 步骤 1：计算出当前 wafer 目录相对于输入根目录的后半段相对路径。
            relative_path = in_wafer_path.relative_to(in_root)
            
            # Step 2: Combine the output root with the relative sub-path to form the target path.
            # 步骤 2：将输出根目录与该相对路径拼接，组合出目标输出的完整路径。
            out_wafer_path = out_root / relative_path
            
            # Step 3: Check existence and create conditionally.
            # 步骤 3：显式判断该镜像路径是否已经存在，从而决定是否创建和打印。
            if not out_wafer_path.exists():
                # Recursively create the target directory and all missing parent folders.
                # 递归创建目标目录以及所有缺失的中间父文件夹（等同于 Linux 的 `mkdir -p`）。
                out_wafer_path.mkdir(parents=True, exist_ok=True)
                
                # Output a real-time message indicating the new directory creation.
                # 📢 只有当文件夹原本不存在、真正执行新建时，才会打印此条消息，避免刷屏。
                print(f"[Create] New directory built: {out_wafer_path}")
            
            # Step 4: Always append the path pair to the job list (whether it was just created or already existed).
            # 步骤 4：无论文件夹是刚刚新建的，还是之前就存在的，都必须把这对绝对路径安全地追加到任务列表中。
            jobs_paths.append((in_wafer_path.resolve(), out_wafer_path.resolve()))
            
    return jobs_paths

def resolve_wafer_io_pairs(args) -> List:
    """
    Produce the ordered list of (input_wafer_dir, output_wafer_dir) pairs to run.

    Two sources, mutually exclusive:
      * --manifest : run ONLY the wafers of the chosen split (default 'test') from a
                     split_manifest.json. This is the honest, held-out evaluation --
                     it excludes every wafer the model was trained/tuned on.
      * --test-path: scan a directory for lot_*/csv/wafer_* and run all of them.

    Output dirs mirror the lot_*/csv/wafer_* layout under --save-path in both modes.
    """
    if args.manifest:
        with open(args.manifest, "r") as f:
            manifest = json.load(f)
        if f"{args.split}_wafers" not in manifest:
            raise KeyError(f"Manifest has no '{args.split}_wafers' key: {args.manifest}")
        dataset_root = Path(manifest.get("split_config", {}).get("dataset_path", ""))
        out_root = Path(args.save_path)
        pairs = []
        for p in manifest[f"{args.split}_wafers"]:
            in_wafer = Path(p)
            # Mirror lot_xxx/csv/wafer_yyy under save-path. Fall back to the last 3
            # path parts if the manifest was written on a machine with a different root.
            try:
                rel = in_wafer.relative_to(dataset_root)
            except ValueError:
                rel = Path(*in_wafer.parts[-3:])
            out_wafer = out_root / rel
            if not out_wafer.exists():
                out_wafer.mkdir(parents=True, exist_ok=True)
                print(f"[Create] New directory built: {out_wafer}")
            pairs.append((in_wafer.resolve(), out_wafer.resolve()))
        print(f"[Manifest] Running '{args.split}' split: {len(pairs)} wafers from {args.manifest}")
        return pairs
    return generate_and_create_mirrored_paths(args.test_path, args.save_path)


def build_test_job_and_label(io_pairs):
    job_list = []
    labels = []
    for (in_path, out_path) in io_pairs:
        job_list.append(
                {
                    "vl_path": in_path / 'Space.csv',
                    "ll_path": in_path / 'MS.csv',
                    "mode": "both",
                    "output_ttf_path": out_path / 'ttf.csv',
                    "output_binary_path": out_path / 'binary_label.csv'
                }
            )
        class_val_matrix = load_matrix_csv(in_path / 'ExistenceClass.csv').astype(np.int32)
        _, _, class_val, _ = matrix_to_sparse_points(class_val_matrix)
        labels.append(np.where(class_val == 3, 1, 0))		# 1 or 2: bad chip (0); 3: good chip (1)

    return job_list, labels


def plot_wafer_worker(job_kwargs: dict) -> str:
    """
    Independent worker function for parallel plotting.
    用于并行绘图的独立工作函数。
    
    Args:
        job_kwargs (dict): A dictionary containing all necessary data for one wafer.
                           包含单个晶圆所有绘图所需数据的字典。
    """
    idx = job_kwargs['idx']
    try:
        # 1. 独立加载数据 / Load data independently in the child process
        vl_matrix = load_matrix_csv(job_kwargs['vl_path'])
        ll_matrix = load_matrix_csv(job_kwargs['ll_path'])
        
        # 2. 转换为稀疏点 / Convert to sparse points
        x, y, vl_val, _ = matrix_to_sparse_points(vl_matrix)
        _, _, ll_val, _ = matrix_to_sparse_points(ll_matrix)
        
        # 3. 执行重型绘图任务 / Execute the heavy plotting function
        plot_five_trends(
            x_die=x, 
            y_die=y, 
            vl_val=vl_val,
            ll_val=ll_val, 
            t_scores=job_kwargs['ttf_res'], 
            y_true=job_kwargs['test_label'],
            y_pred=job_kwargs['label_res'],
            title='Wafer Spatial Trends, Lifetime & Binary Classification',
            cmap='turbo',
            output_path=job_kwargs['output_image_path'],
            use_log_scale_for_t=True
        )
        return f"[Success] Wafer {idx} plotted successfully."
        
    except Exception as e:
        # 捕获异常防止单个晶圆的数据错误导致整个进程池崩溃
        return f"[Error] Failed on Wafer {idx}: {str(e)}"


def main():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--config-path', type=str, default=None, help='Path to load a config')
    parser.add_argument('--pipeline-type', type=str, choices=['GPR', 'DPM'], help='Prediction Pipeline Type')
    parser.add_argument('--model-type', type=str, default=None, choices=['PowerLaw', 'SqrtE', 'InverseE'], help='Only for DPM pipelines')
    parser.add_argument('--M-structures', type=int, default=None, help='Parameter: M structures')
    parser.add_argument('--F-target', type=float, default=None, help='Parameter: F target')
    parser.add_argument('--N-samples-per-dim', type=int, default=None, help='Parameter: N samples per dim')
    parser.add_argument('--threshold', type=float, default=None, help='Parameter: threshold')
    parser.add_argument('--test-path', type=str, help='Directory to scan for lot_*/csv/wafer_* (ignored when --manifest is set)')
    parser.add_argument('--manifest', type=str, default=None, help='Path to split_manifest.json; run only the chosen --split (held-out evaluation)')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'], help="Which manifest split to run when --manifest is set (default: test)")
    parser.add_argument('--save-path', type=str, help='Output path for test results')
    parser.add_argument('--fscore-beta', type=float, default=2.0, help='The beta parameter of F score')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size of wafers')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of CPUs')
    args = parser.parse_args()
    if args.pipeline_type == 'GPR': args.model_type = 'GPR'
    print(args)

    if not args.manifest and not args.test_path:
        parser.error("provide either --manifest (with --split) or --test-path")
    if not args.save_path:
        parser.error("--save-path is required")

    io_pairs = resolve_wafer_io_pairs(args)
    test_jobs, test_labels = build_test_job_and_label(io_pairs)
    if args.config_path is not None:
        with open(args.config_path, "r") as f:
            config = json.load(f)
    else:
        config = {
        "pipeline_type": None,
        "M_structures": None,
        "F_target": None,
        "N_samples_per_dim": None,
        "threshold": None,
        "unit_model_kwargs":\
            {
                "model_type": None
            }
        }
    if args.pipeline_type is not None: config["pipeline_type"] = args.pipeline_type
    if args.M_structures is not None: config["M_structures"] = args.M_structures
    if args.F_target is not None: config["F_target"] = args.F_target
    if args.N_samples_per_dim is not None: config["N_samples_per_dim"] = args.N_samples_per_dim
    if args.threshold is not None: config["threshold"] = args.threshold
    if config["pipeline_type"] == 'GPR':
        config["unit_model_kwargs"]["actual_vl_max"] = 40.0
        config["unit_model_kwargs"]["actual_ll_max"] = 20.0

    results = batch_process_by_path(
        jobs_list=test_jobs,
        pipeline_config=config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # job_path = generate_and_create_mirrored_paths(args.test_path, args.save_path)
    # for idx, (ttf_res, label_res) in enumerate(results):
    #     vl_matrix = load_matrix_csv(test_jobs[idx]['vl_path'])
    #     ll_matrix = load_matrix_csv(test_jobs[idx]['ll_path'])
    #     x, y, vl_val, _ = matrix_to_sparse_points(vl_matrix)
    #     _, _, ll_val, _ = matrix_to_sparse_points(ll_matrix)

    #     plot_five_trends(
    #         x, 
    #         y, 
    #         vl_val,
    #         ll_val, 
    #         ttf_res, 
    #         test_labels[idx],
    #         label_res,
    #         title='Wafer Spatial Trends, Lifetime & Binary Classification',
    #         cmap='turbo',
    #         output_path=job_path[idx][1] / 'wafer_maps.png',
    #         use_log_scale_for_t=True)


    plot_jobs = []
    for idx, (ttf_res, label_res) in enumerate(results):
        base_out_dir = Path(io_pairs[idx][1])
        output_img = str(base_out_dir / 'wafer_maps.png')
        job_dict = {
            'idx': idx,
            'vl_path': test_jobs[idx]['vl_path'],
            'll_path': test_jobs[idx]['ll_path'],
            'ttf_res': ttf_res,
            'test_label': test_labels[idx],
            'label_res': label_res,
            'output_image_path': output_img
        }
        plot_jobs.append(job_dict)
        pool = multiprocessing.Pool(processes=args.num_workers, maxtasksperchild=10)
    
    try:
        completed_results = pool.imap(plot_wafer_worker, plot_jobs, chunksize=1)     
        for i, msg in enumerate(completed_results):
            print(f"[{i+1}/{len(plot_jobs)}] {msg}")
            
    except KeyboardInterrupt:
        print("\n[Warning] Caught Ctrl+C! Terminating all plotting processes...")
        pool.terminate()
        
    finally:
        pool.close()
        pool.join()
        print("\nAll wafer plotting jobs finished successfully.")


if __name__ == "__main__":
	main()