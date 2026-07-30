import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Patch
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
from pathlib import Path

from .wafer_plot import create_wafer_mask, draw_die_grid, perform_gpr

def plot_five_trends(
    x_die, 
    y_die, 
    vl_val, 
    ll_val, 
    t_scores, 
    y_true,          
    y_pred,          
    title='Wafer Spatial Trends, Lifetime & Binary Classification',
    cmap='turbo',
    output_path=None,
    use_log_scale_for_t=True
):
    """
    绘制晶圆综合分析图 (2行3列布局)：
    - [0, 0] / [0, 1]：Space 和 MS 的连续 GPR 空间趋势插值图
    - [0, 2]：预测寿命 t 的矢量方格连续图
    - [1, 0]：真实好/坏芯片分类图 (Per-Die 矢量方格) -> 0画冷色, 1画暖色
    - [1, 1]：模型预测好/坏分类图 (Per-Die 矢量方格) -> 0画冷色, 1画暖色
    - [1, 2]：留空隐藏 (保持版面整洁)
    """
    # --------------------------------------------------------
    # 1. 工艺尺寸 vl 和 ll 的连续空间 GPR 插值
    # --------------------------------------------------------
    xx, yy, zz_vl, _ = perform_gpr(x_die, y_die, vl_val)
    
    _, _, zz_ll, _ = perform_gpr(x_die, y_die, ll_val)

    # --------------------------------------------------------
    # 2. 晶圆掩膜 (仅用于 GPR 连续插值图)
    # --------------------------------------------------------
    valid_distance = np.sqrt(x_die**2 + y_die**2)
    wafer_radius = valid_distance.max()

    wafer_mask = create_wafer_mask(xx + 0.5, yy + 0.5, wafer_radius)
    zz_vl = np.where(wafer_mask, zz_vl, np.nan)
    zz_ll = np.where(wafer_mask, zz_ll, np.nan)

    # --------------------------------------------------------
    # 3. 画布布局初始化 (🌟 更改为 2 行 3 列)
    # --------------------------------------------------------
    # figsize 从 (28, 5.5) 改为 (18, 11) 以适应 2x3 的纵横比
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(title, fontsize=18, fontweight='bold', y=1.02)

    # 提取当前 cmap 的两端颜色用于好坏标签图
    colormap_ref = plt.get_cmap(cmap)
    color_bad = colormap_ref(0.0)   # 坏芯片颜色 (极小值/冷色)
    color_good = colormap_ref(1.0)  # 好芯片颜色 (极大值/暖色)

    # 辅助函数：绘制连续趋势子图 (GPR)
    def draw_gpr_subplot(ax, zz_data, subplot_title, cbar_label):
        image = ax.imshow(
            zz_data,
            extent=[xx.min(), xx.max(), yy.min(), yy.max()],
            origin='lower',
            cmap=cmap,
            aspect='equal'
        )
        draw_die_grid(ax, int(x_die.min()), int(x_die.max()), int(y_die.min()), int(y_die.max()))
        wafer_circle = Circle((-0.5, -0.5), wafer_radius, fill=False, color='black', linewidth=2)
        ax.add_patch(wafer_circle)
        
        ax.scatter(x_die, y_die, c='black', s=5, alpha=0.5)
        ax.set_title(subplot_title, fontsize=14)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        cbar = fig.colorbar(image, ax=ax, shrink=0.75)
        cbar.set_label(cbar_label)

    # 🌟 [第一行] 图1、图2、图3
    # 图1：Space (vl)
    draw_gpr_subplot(axes[0, 0], zz_vl, 'Space (vl) Spatial Trend\n(GPR Continuous)', 'Space Measurement (nm)')

    # 图2：MS (ll)
    draw_gpr_subplot(axes[0, 1], zz_ll, 'MS (ll) Spatial Trend\n(GPR Continuous)', 'MS Measurement (nm)')

    # 图3：预测寿命 t (矢量矩形块连续图)
    ax_t = axes[0, 2]
    norm = LogNorm(vmin=np.nanmin(t_scores), vmax=np.nanmax(t_scores)) if use_log_scale_for_t else plt.Normalize(vmin=np.nanmin(t_scores), vmax=np.nanmax(t_scores))
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    
    for x, y, score in zip(x_die, y_die, t_scores):
        if np.isfinite(score):
            rect = Rectangle(
                xy=(x - 0.5, y - 0.5),
                width=1.0, height=1.0,
                facecolor=mapper.to_rgba(score),
                edgecolor='none', zorder=2
            )
            ax_t.add_patch(rect)

    ax_t.set_xlim(xx.min(), xx.max())
    ax_t.set_ylim(yy.min(), yy.max())
    draw_die_grid(ax_t, int(x_die.min()), int(x_die.max()), int(y_die.min()), int(y_die.max()))
    ax_t.add_patch(Circle((-0.5, -0.5), wafer_radius, fill=False, color='black', linewidth=2))
    ax_t.set_title('Predicted Lifetime (t)\n(Per-Die Vector Grid)', fontsize=14)
    ax_t.set_xlabel('X')
    ax_t.set_ylabel('Y')
    ax_t.set_aspect('equal')
    cbar = fig.colorbar(mapper, ax=ax_t, shrink=0.75)
    cbar.set_label('Reliability Lifetime Score (t)')

    # --------------------------------------------------------
    # 🌟 新增辅助函数：用于绘制二分类标签图
    # --------------------------------------------------------
    def draw_binary_label_subplot(ax, labels, subplot_title):
        for x, y, label in zip(x_die, y_die, labels):
            if np.isfinite(label):
                current_color = color_bad if int(label) == 0 else color_good
                rect = Rectangle(
                    xy=(x - 0.5, y - 0.5), 
                    width=1.0, height=1.0,
                    facecolor=current_color,
                    edgecolor='none',
                    zorder=2
                )
                ax.add_patch(rect)
                
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        draw_die_grid(ax, int(x_die.min()), int(x_die.max()), int(y_die.min()), int(y_die.max()))
        ax.add_patch(Circle((-0.5, -0.5), wafer_radius, fill=False, color='black', linewidth=2))
        ax.set_title(subplot_title, fontsize=14)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal')
        
        # 绘制离散型图例
        legend_elements = [
            Patch(facecolor=color_bad, edgecolor='black', label='Bad Die (0 / Intercept)'),
            Patch(facecolor=color_good, edgecolor='black', label='Good Die (1 / Pass)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.8)

    # 🌟 [第二行] 图4、图5、以及隐藏的空图
    # 图4：真实标签
    draw_binary_label_subplot(axes[1, 0], y_true, 'Ground Truth Reliability\n(Binary Labels)')

    # 图5：预测标签
    draw_binary_label_subplot(axes[1, 1], y_pred, 'Model Predicted Decision\n(Binary Classification)')

    # 图6：彻底隐藏不需要的右下角空坐标轴
    axes[1, 2].axis('off')

    # --------------------------------------------------------
    # 4. 调整间距与保存
    # --------------------------------------------------------
    fig.tight_layout(w_pad=2.0, h_pad=2.0)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f'Saved to: {output_path}')
        plt.close(fig)
    else:
        plt.show()