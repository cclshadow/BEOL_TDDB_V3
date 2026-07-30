import numpy as np
import pathlib
from typing import Tuple

# ==============================================================================
# Spatial Data Parsers & Reconstructors / 空间数据解析与重建工具
# ==============================================================================

def load_matrix_csv(csv_path: str) -> np.ndarray:
    """
    Load raw 2D wafer matrix from a CSV file.
    从 CSV 文件读取原始二维晶圆矩阵。
    
    Args:
        csv_path (str): Path to the CSV file. / CSV 文件路径。
        
    Returns:
        np.ndarray: The loaded 2D numpy array. / 加载的二维 numpy 数组。
    """
    return np.loadtxt(csv_path, delimiter=',')


def matrix_to_sparse_points(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, tuple]:
    """
    Extract valid die values and map them to zero-centered coordinates.
    提取有效的 Die 数据，并将其映射为以晶圆中心为原点的相对坐标。
    
    Non-finite values (NaN, Inf) and absolute zeros are ignored.
    非有限数（NaN, Inf）及绝对 0 值将被视为无效区域并被忽略。
    
    Args:
        matrix (np.ndarray): The dense 2D wafer matrix. / 密集的二维晶圆矩阵。
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, tuple]: 
            (x_coordinates, y_coordinates, valid_values, original_shape)
            返回稀疏的 (x 坐标集, y 坐标集, 对应的值集合, 原始矩阵形状)。
    """
    rows, cols = matrix.shape
    x_list, y_list, value_list = [], [], []
    
    # Define the physical center of the wafer / 定义晶圆的物理中心坐标
    center_x = cols // 2
    center_y = rows // 2

    for row in range(rows):
        for col in range(cols):
            value = matrix[row, col]
            
            # Filter out invalid or zero-value background regions
            # 过滤掉无效值或数值为 0 的背景区域
            if not np.isfinite(value) or value == 0:
                continue
                
            # Convert absolute indices to relative coordinates
            # 将绝对行列索引转换为相对中心坐标
            x = col - center_x
            y = row - center_y
            
            x_list.append(x)
            y_list.append(y)
            value_list.append(value)

    return np.array(x_list), np.array(y_list), np.array(value_list), matrix.shape


def sparse_points_to_matrix(x_arr: np.ndarray, y_arr: np.ndarray, 
                            values_arr: np.ndarray, original_shape: tuple) -> np.ndarray:
    """
    Inverse reconstruction: Rebuilds a dense 2D matrix from sparse predictions.
    逆向重建：将稀疏的预测结果还原为密集的二维矩阵。
    
    Unassigned areas automatically retain their default `np.nan` values.
    未赋值的背景区域将自动保持默认的 `np.nan` 填充。
    
    Args:
        x_arr (np.ndarray): Relative X coordinates. / 相对 X 坐标集。
        y_arr (np.ndarray): Relative Y coordinates. / 相对 Y 坐标集。
        values_arr (np.ndarray): Values corresponding to the coordinates. / 坐标对应的值集合。
        original_shape (tuple): The (rows, cols) shape to reconstruct. / 需要重建的原始形状 (行数, 列数)。
        
    Returns:
        np.ndarray: Reconstructed 2D matrix with NaNs for background. / 重建后带 NaN 背景的二维矩阵。
    """
    rows, cols = original_shape
    
    # Initialize the base canvas completely with NaNs
    # 使用 NaN 初始化整块画布底板
    reconstructed_matrix = np.full((rows, cols), np.nan)
    
    center_x = cols // 2
    center_y = rows // 2

    # Map relative coordinates back to absolute matrix indices
    # 将相对坐标映射回绝对的矩阵行列索引
    for x, y, val in zip(x_arr, y_arr, values_arr):
        col = x + center_x
        row = y + center_y
        
        # Safety bound check to prevent index out-of-range errors
        # 边界安全检查，防止索引越界
        if 0 <= row < rows and 0 <= col < cols:
            reconstructed_matrix[row, col] = val

    return reconstructed_matrix


def save_matrix_csv(matrix: np.ndarray, save_path: str = None):
    """
    Save the 2D matrix to disk as a CSV format.
    将二维矩阵以 CSV 格式落盘保存。
    
    Safely skips operations if `save_path` is None, making it ideal for memory-only runs.
    如果 `save_path` 为 None 则安全跳过，非常适合纯内存计算（不写盘）的场景。
    
    Args:
        matrix (np.ndarray): The 2D matrix to save. / 需要保存的二维矩阵。
        save_path (str, optional): Target file path. Defaults to None. / 目标文件路径，默认为 None。
    """
    if save_path is None:
        return
        
    path = pathlib.Path(save_path)
    
    # Automatically create any missing parent directories cascaded
    # 自动创建任意缺失的级联父目录
    path.parent.mkdir(parents=True, exist_ok=True) 
    
    # Save using 6 decimal places for clean, standardized formatting
    # 保留 6 位小数以确保格式规范整洁
    np.savetxt(save_path, matrix, delimiter=',', fmt='%.6f')