import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import numpy as np
from pathlib import Path
from typing import List
import json
import sys

from src.reliability_classifiers.binary_classifier import (
	optimize_binary_threshold,
	evaluate_binary_threshold,
	)
from src.multiprocessing_wrapper import batch_process_by_path
from src.param_optimizer import TwoStageWaferOptimizer
from src.csv_utils import (
	load_matrix_csv,
	matrix_to_sparse_points,
	)

def _build_jobs_and_labels(wafer_paths: List[Path]):
	"""Shared core: build job dicts and labels from an explicit list of wafer paths."""
	job_list = []
	labels = []
	for wafer_path in wafer_paths:
		job_list.append(
			{
				"vl_path": wafer_path / "Space.csv",
				"ll_path": wafer_path / "MS.csv",
				"mode": "ttf_only",
				"output_ttf_path": None,
				"output_binary_path": None,
			}
		)
		class_val_matrix = load_matrix_csv(wafer_path / "ExistenceClass.csv").astype(np.int32)
		_, _, class_val, _ = matrix_to_sparse_points(class_val_matrix)
		labels.append(np.where(class_val == 3, 1, 0))
	return job_list, labels


def build_jobs_from_manifest(manifest: dict, split: str):
	"""Load jobs and labels for a named split ('train', 'val', 'test') from a manifest."""
	wafer_paths = [Path(p) for p in manifest[f"{split}_wafers"]]
	return _build_jobs_and_labels(wafer_paths)


def find_all_wafer_paths(base_path: str) -> List[Path]:
    """
    Find all wafer paths in the dataset matching the pattern: path/lot_xxx/csv/wafer_yyy
    
    遍历数据集目录，寻找所有符合层级规则的晶圆文件夹路径。
    匹配规则：根目录/lot_xxx/csv/wafer_yyy
    
    Args:
        base_path (str): The root directory of the dataset. 
                         数据集的根目录路径。
        
    Returns:
        List[str]: A sorted list of absolute or relative paths to the wafers. 
                   按字母顺序排序的满足条件的晶圆路径列表（字符串格式）。
    """
    root_dir = Path(base_path)
    
    # 确保根目录存在 / Ensure the base directory exists
    if not root_dir.exists() or not root_dir.is_dir():
        raise FileNotFoundError(f"The dataset path does not exist or is not a directory: {base_path}")

    wafer_paths = []
    
    # 使用 glob 进行严格的层级模式匹配 / Use glob for strict hierarchical pattern matching
    # "lot_*" matches lot_123, lot_ABC, etc.
    # "csv" is the exact folder name required.
    # "wafer_*" matches wafer_01, wafer_test, etc.
    search_pattern = "lot_*/csv/wafer_*"
    
    for path in root_dir.glob(search_pattern):
        # 通常 wafer_yyy 是一个包含数据的文件夹，做一次 is_dir() 断言可以过滤掉同名的意外文件
        # Ensure the matched path is actually a directory, not a file
        if path.is_dir():
            # 保留 Path 对象
            wafer_paths.append(path.resolve())
            
    # 返回排序后的列表，保证不同操作系统下读取的顺序一致，这对机器学习复现非常重要
    # Return sorted paths to guarantee deterministic behavior across different OS
    return sorted(wafer_paths)

def build_train_job_and_label(dataset_path):
	return _build_jobs_and_labels(find_all_wafer_paths(dataset_path))

def main():
	parser = argparse.ArgumentParser(description='Train a model')
	parser.add_argument('--pipeline-type', type=str, choices=['GPR', 'DPM', 'Linear'], help='Prediction Pipeline Type')
	parser.add_argument('--model-type', type=str, default=None, choices=['PowerLaw', 'SqrtE', 'InverseE', 'PowerLawScaled', 'SqrtEScaled', 'InverseEScaled'], help='Only for DPM pipelines')
	parser.add_argument('--spacing-scale', type=float, default=1.0, help='Spacing divisor k for scaled DPM models (e.g. 3.0 maps 9-22nm into 3-7nm)')
	# --- Physics-delta training (DPM models only; GPR/Linear transparently ignore these) ---
	parser.add_argument('--train-deltas', action='store_true', help='Also fit the physics deltas (empirical-fit corrections) during optimisation.')
	parser.add_argument('--delta-l2', type=float, default=0.0, help='L2 shrink of deltas toward the physical baseline (delta=0). 0 = off.')
	parser.add_argument('--anchor-years', type=float, default=None, help='Anchor the learned decision boundary to this lifetime (years) as an absolute-scale prior. None = off.')
	parser.add_argument('--anchor-weight', type=float, default=0.0, help='Weight of the anchor penalty (log-space). 0 = off.')
	parser.add_argument('--manifest', type=str, default=None, help='Path to split_manifest.json from create_split.py (overrides --training-path / --validation-path)')
	parser.add_argument('--training-path', type=str, help='Path for training set (ignored when --manifest is set)')
	parser.add_argument('--validation-path', type=str, help='Path for validation set (ignored when --manifest is set)')
	parser.add_argument('--fscore-beta', type=float, default=2.0, help='The beta parameter of F score')
	parser.add_argument('--n-trials', type=int, default=20, help='Training iterations')
	parser.add_argument('--batch-size', type=int, default=1, help='Batch size of wafers')
	parser.add_argument('--num-workers', type=int, default=4, help='Number of CPUs')
	parser.add_argument('--save-path', type=str, default=None, help='Output path for training results')
	args = parser.parse_args()
	if args.pipeline_type == 'GPR': args.model_type = 'GPR'
	if args.pipeline_type == 'Linear': args.model_type = 'Linear'
	print(args)

	# build job list and label
	if args.manifest:
		with open(args.manifest, "r") as f:
			manifest = json.load(f)
		print(f"[Split] Loaded manifest: {args.manifest}")
		print(f"[Split] Train: {manifest['summary']['train']['n_lots']} lots / {manifest['summary']['train']['n_wafers']} wafers")
		print(f"[Split]   Val: {manifest['summary']['val']['n_lots']} lots / {manifest['summary']['val']['n_wafers']} wafers")
		print(f"[Split]  Test: {manifest['summary']['test']['n_lots']} lots / {manifest['summary']['test']['n_wafers']} wafers")
		train_jobs, train_labels = build_jobs_from_manifest(manifest, "train")
		val_jobs,   val_labels   = build_jobs_from_manifest(manifest, "val")
		test_jobs,  test_labels  = build_jobs_from_manifest(manifest, "test")
	elif args.training_path and args.validation_path:
		train_jobs, train_labels = build_train_job_and_label(args.training_path)
		val_jobs,   val_labels   = build_train_job_and_label(args.validation_path)
		test_jobs,  test_labels  = None, None
	else:
		print("[Error] Provide either --manifest or both --training-path and --validation-path.")
		sys.exit(1)

	# build config
	config = {
	"pipeline_type": args.pipeline_type,
	"M_structures": 1_000_000,
	"F_target": 1e-4,
	"N_samples_per_dim": 8,
	}
	if args.pipeline_type != 'Linear':
		config["unit_model_kwargs"] = {"model_type": args.model_type}
		if args.pipeline_type == 'GPR':
			config["unit_model_kwargs"]["actual_vl_max"] = 40.0
			config["unit_model_kwargs"]["actual_ll_max"] = 20.0
		if args.spacing_scale != 1.0:
			config["unit_model_kwargs"]["spacing_scale"] = args.spacing_scale

	# Convert the anchor lifetime from years to seconds (matches physics TTF units)
	anchor_ttf = args.anchor_years * 365.25 * 24 * 3600 if args.anchor_years else None

	# optimization
	optimizer = TwoStageWaferOptimizer(
					train_jobs=train_jobs,
					val_jobs=val_jobs,
					train_labels=train_labels,
					val_labels=val_labels,
					batch_process_fn=batch_process_by_path,
					optimize_fn=optimize_binary_threshold,
					evaluate_fn=evaluate_binary_threshold,
					base_config=config,
					batch_size=args.batch_size,
					num_workers=args.num_workers,
					beta=args.fscore_beta,
					train_deltas=args.train_deltas,
					delta_l2=args.delta_l2,
					anchor_ttf=anchor_ttf,
					anchor_weight=args.anchor_weight
					)

	optimized_config, optimization_metadata = optimizer.run_optimization(
														n_trials=args.n_trials
														)

	save_path = Path(args.save_path) if args.save_path else Path(f'./configs/{args.pipeline_type}_{args.model_type}_{args.fscore_beta:.1f}_{args.n_trials}/')
	save_path.mkdir(parents=True, exist_ok=True)

	with open(save_path / 'config.json', "w") as f:
		json.dump(optimized_config, f)

	with open(save_path / 'metadata.json', "w") as f:
		json.dump(optimization_metadata, f)

	optimizer.plot_history(save_image_path=save_path/'metric_curve.png')

	# Final held-out test evaluation (only available when --manifest is used)
	if test_jobs is not None:
		print("\n[Test] Running final evaluation on held-out test set...")
		test_outputs = batch_process_by_path(
			test_jobs, optimized_config, batch_size=args.batch_size, num_workers=args.num_workers
		)
		y_test_true = np.concatenate(test_labels)
		y_test_pred_flat = np.concatenate([ttf for ttf, _ in test_outputs if ttf is not None])
		test_f = evaluate_binary_threshold(
			y_test_pred_flat, y_test_true, optimized_config["threshold"], beta=args.fscore_beta
		)
		print(f"[Test] Final held-out F-{args.fscore_beta} score: {test_f:.4f}")
		optimization_metadata["test_evaluation"] = {
			"test_f_score": float(test_f),
			"n_lots":   manifest["summary"]["test"]["n_lots"],
			"n_wafers": manifest["summary"]["test"]["n_wafers"],
		}
		with open(save_path / 'metadata.json', "w") as f:
			json.dump(optimization_metadata, f)


if __name__ == "__main__":
	main()