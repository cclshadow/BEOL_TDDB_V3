from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle, Circle

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    WhiteKernel,
)


# ============================================================
# LOAD MATRIX CSV
# ============================================================

def load_matrix_csv(csv_path):

    matrix = np.loadtxt(
        csv_path,
        delimiter=',',
    )

    return matrix


# ============================================================
# CREATE SPARSE DIE DATA
# ============================================================

def matrix_to_sparse_points(matrix):
    """
    Convert wafer matrix to sparse die-center points.
    """

    rows, cols = matrix.shape

    x_list = []
    y_list = []
    value_list = []

    center_x = cols // 2
    center_y = rows // 2

    for row in range(rows):
        for col in range(cols):

            value = matrix[row, col]

            # skip invalid
            if not np.isfinite(value):
                continue

            if value == 0:
                continue

            x = col - center_x
            y = row - center_y

            x_list.append(x)
            y_list.append(y)
            value_list.append(value)

    return (
        np.array(x_list),
        np.array(y_list),
        np.array(value_list),
    )


# ============================================================
# WAFER MASK
# ============================================================

def create_wafer_mask(xx, yy, radius):

    distance = np.sqrt(xx**2 + yy**2)

    return distance <= radius


# ============================================================
# DRAW DIE GRID
# ============================================================

def draw_die_grid(
    ax,
    x_min,
    x_max,
    y_min,
    y_max,
):

    for x in range(x_min, x_max + 1):

        ax.plot(
            [x - 0.5, x - 0.5],
            [y_min - 0.5, y_max + 0.5],
            color='black',
            linewidth=0.5,
            alpha=0.7,
        )

    for y in range(y_min, y_max + 1):

        ax.plot(
            [x_min - 0.5, x_max + 0.5],
            [y - 0.5, y - 0.5],
            color='black',
            linewidth=0.5,
            alpha=0.7,
        )


# ============================================================
# GPR
# ============================================================

def perform_gpr(
    x,
    y,
    value,
    resolution=300,
):

    X_train = np.column_stack([x, y])

    kernel = (
        ConstantKernel(1.0)
        * RBF(length_scale=3.0)
        + WhiteKernel(noise_level=0.05)
    )

    gpr = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
    )

    gpr.fit(X_train, value)

    x_min = int(np.floor(x.min())) - 1
    x_max = int(np.ceil(x.max())) + 1

    y_min = int(np.floor(y.min())) - 1
    y_max = int(np.ceil(y.max())) + 1

    x_lin = np.linspace(x_min, x_max, resolution)
    y_lin = np.linspace(y_min, y_max, resolution)

    xx, yy = np.meshgrid(x_lin, y_lin)

    X_pred = np.column_stack([
        xx.ravel(),
        yy.ravel(),
    ])

    # 修改：要求模型返回标准差 (std)
    zz, std = gpr.predict(X_pred, return_std=True)

    zz = zz.reshape(xx.shape)
    std = std.reshape(xx.shape)

    return xx, yy, zz, std


# ============================================================
# MAIN
# ============================================================

def plot_wafer_gpr(
    csv_path,
    title='Wafer GPR',
    cmap='turbo',
    output_path=None,
    k=2.0,  # 新增：不确定性系数k，默认2.0 (即2*sigma)
):

    # --------------------------------------------------------
    # Load matrix
    # --------------------------------------------------------

    matrix = load_matrix_csv(csv_path)

    # --------------------------------------------------------
    # Sparse points
    # --------------------------------------------------------

    x, y, value = matrix_to_sparse_points(matrix)

    # --------------------------------------------------------
    # GPR
    # --------------------------------------------------------

    # 修改：接收返回的 std
    xx, yy, zz, std = perform_gpr(
        x,
        y,
        value,
    )

    # --------------------------------------------------------
    # Wafer mask
    # --------------------------------------------------------

    valid_distance = np.sqrt(x**2 + y**2)

    wafer_radius = valid_distance.max()

    wafer_mask = create_wafer_mask(
        xx + 0.5,
        yy + 0.5,
        wafer_radius,
    )

    zz = np.where(
        wafer_mask,
        zz,
        np.nan,
    )

    # 新增：计算不确定性 k*sigma 并应用掩膜
    uncertainty = np.where(
        wafer_mask,
        k * std,
        np.nan,
    )

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    # 修改：将子图数量从2改为3，并加宽画布
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4.5),
    )

    # ========================================================
    # LEFT: SPARSE POINTS
    # ========================================================

    ax = axes[0]

    scatter = ax.scatter(
        x,
        y,
        c=value,
        cmap=cmap,
        s=10,
        edgecolors='black',
        linewidths=0.5,
        zorder=3,
    )

    draw_die_grid(
        ax,
        int(x.min()),
        int(x.max()),
        int(y.min()),
        int(y.max()),
    )

    wafer_circle = Circle(
        (-0.5, -0.5),
        wafer_radius,
        fill=False,
        color='black',
        linewidth=2,
    )

    ax.add_patch(wafer_circle)

    ax.set_title(
        'Measured Data\n'
        '(One Center Point per Die)',
        fontsize=14,
    )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    ax.set_aspect('equal')

    cbar = fig.colorbar(
        scatter,
        ax=ax,
        shrink=0.75,
    )

    cbar.set_label('Measured Value (nm)')

    # ========================================================
    # MIDDLE: GPR
    # ========================================================

    ax = axes[1]

    image = ax.imshow(
        zz,
        extent=[
            xx.min(),
            xx.max(),
            yy.min(),
            yy.max(),
        ],
        origin='lower',
        cmap=cmap,
        aspect='equal',
    )

    draw_die_grid(
        ax,
        int(x.min()),
        int(x.max()),
        int(y.min()),
        int(y.max()),
    )

    wafer_circle = Circle(
        (-0.5, -0.5),
        wafer_radius,
        fill=False,
        color='black',
        linewidth=2,
    )

    ax.add_patch(wafer_circle)

    ax.scatter(
        x,
        y,
        c='black',
        s=5,
        alpha=0.5,
    )

    ax.set_title(
        'Gaussian Process Regression\n'
        '(Spatial Trend Prediction)',
        fontsize=14,
    )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    cbar = fig.colorbar(
        image,
        ax=ax,
        shrink=0.75,
    )

    cbar.set_label('GPR Prediction Value (nm)')

    # ========================================================
    # RIGHT: GPR UNCERTAINTY
    # ========================================================

    ax = axes[2]

    # 使用不同色系(如热力图常用的 YlOrRd 或 magma)以区分预测值和不确定性
    img_unc = ax.imshow(
        uncertainty,
        extent=[
            xx.min(),
            xx.max(),
            yy.min(),
            yy.max(),
        ],
        origin='lower',
        cmap='YlOrRd', 
        aspect='equal',
    )

    draw_die_grid(
        ax,
        int(x.min()),
        int(x.max()),
        int(y.min()),
        int(y.max()),
    )

    wafer_circle = Circle(
        (-0.5, -0.5),
        wafer_radius,
        fill=False,
        color='black',
        linewidth=2,
    )

    ax.add_patch(wafer_circle)

    # 绘制已知点，可以更直观地看到有数据点的地方不确定性低
    ax.scatter(
        x,
        y,
        c='black',
        s=5,
        alpha=0.5,
    )

    ax.set_title(
        f'GPR Uncertainty ({k}σ)\n'
        '(Prediction Confidence)',
        fontsize=14,
    )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    cbar = fig.colorbar(
        img_unc,
        ax=ax,
        shrink=0.75,
    )

    cbar.set_label(f'Uncertainty Value ({k}σ, nm)')


    fig.tight_layout(w_pad=0.5)

    # --------------------------------------------------------
    # Save / show
    # --------------------------------------------------------

    if output_path is not None:

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches='tight',
        )

        print(f'Saved to: {output_path}')

        plt.close(fig)

    else:
        plt.show()


# ============================================================
# EXAMPLE
# ============================================================

if __name__ == '__main__':

    plot_wafer_gpr(
        csv_path='data/lot_001/csv/wafer_14/Space.csv',
        title='Wafer GPR Example',
        output_path=None,
        k=2.0,  # 在这里调整 k 值
    )