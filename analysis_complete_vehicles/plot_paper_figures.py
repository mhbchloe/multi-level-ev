"""
顶刊风格的聚类结果可视化
包含多种专业图表组合
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
from matplotlib.patches import Rectangle
import os
import json

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
rcParams['font.size'] = 10
rcParams['axes.linewidth'] = 1.5
rcParams['grid.linewidth'] = 0.8

# ============================================================
# 配置
# ============================================================
CONFIG = {
    'clustering_results': './analysis_complete_vehicles/results/clustering_v3/clustering_v3_results.npz',
    'clustering_summary': './analysis_complete_vehicles/results/clustering_v3/clustering_v3_summary.json',
    'save_dir': './analysis_complete_vehicles/results/clustering_v3',
}

CLUSTER_COLORS = {0: '#5B9BD5', 1: '#70AD47', 2: '#C0504D', 3: '#FFC000'}

# 特征配置
FEATURES_CONFIG = [
    ('avg_speed_mov',     'Speed (moving)',        'km/h',   24.2276, 41.8020, 72.8773, 41.5640),
    ('speed_std',         'Speed Variation',       'km/h',   6.0934,  13.5917, 25.3141, 14.6288),
    ('acc_std_mov',       'Acceleration Var.',     'm/s²',   0.0019,  0.0036,  0.0081,  0.0039),
    ('heading_change',    'Turning Intensity',     'deg',    1222.07, 555.80,  570.62,  1022.88),
    ('soc_rate',          'Energy Consumption',    '%/min',  2.0835,  11.1219, 18.7522, 7.9352),
    ('idle_ratio',        'Idle Ratio',            '',       0.9473,  0.7939,  0.2329,  0.8135),
    ('seg_length',        'Segment Duration',      'steps',  645.55,  30.28,   13.33,   42.08),
]


# ============================================================
# 图1：Heatmap（热力图）
# ============================================================
def plot_heatmap(labels, phys, stats, save_dir):
    """绘制特征热力图 - 论文常用"""
    print("绘制热力图...")
    
    unique = sorted(np.unique(labels))
    feat_names = [f[1] for f in FEATURES_CONFIG]
    
    # 准备数据矩阵
    data_matrix = []
    for feat_key, _, _, *values in FEATURES_CONFIG:
        data_matrix.append(values[:len(unique)])
    
    data_matrix = np.array(data_matrix)
    
    # 行归一化（每行独立归一化到 [0, 1]）
    data_norm = np.zeros_like(data_matrix, dtype=float)
    for i in range(data_matrix.shape[0]):
        row_min, row_max = data_matrix[i].min(), data_matrix[i].max()
        if row_max - row_min > 0:
            data_norm[i] = (data_matrix[i] - row_min) / (row_max - row_min)
        else:
            data_norm[i] = 0.5
    
    # 绘制
    fig, ax = plt.subplots(figsize=(10, 7))
    
    im = ax.imshow(data_norm, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # 设置标签
    ax.set_xticks(range(len(unique)))
    ax.set_xticklabels([f'C{c}: {stats[c].get("label", f"C{c}")}' 
                        for c in unique], fontsize=11, fontweight='bold')
    ax.set_yticks(range(len(feat_names)))
    ax.set_yticklabels(feat_names, fontsize=10)
    
    # 添加数值标注
    for i in range(len(feat_names)):
        for j in range(len(unique)):
            text = ax.text(j, i, f'{data_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=9,
                          fontweight='bold')
    
    ax.set_title('Cluster Feature Profiles (Normalized Heatmap)', 
                fontsize=13, fontweight='bold', pad=15)
    
    # 颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Value', fontsize=10)
    
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'paper_heatmap.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(save_dir, 'paper_heatmap.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"✓ 已保存: paper_heatmap.png/pdf")


# ============================================================
# 图2：分组柱状图（多特征对比）
# ============================================================
def plot_grouped_bars(labels, phys, stats, save_dir):
    """绘制分组柱状图 - 展示主要特征对比"""
    print("绘制分组柱状图...")
    
    unique = sorted(np.unique(labels))
    colors = {c: CLUSTER_COLORS.get(c, f'C{c}') for c in unique}
    
    # 选择关键特征
    key_features = [
        ('avg_speed_mov',     'Speed (moving)', 'km/h'),
        ('heading_change',    'Turning', 'deg'),
        ('soc_rate',          'Energy Rate', '%/min'),
        ('idle_ratio',        'Idle Ratio', ''),
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (feat_key, feat_name, unit) in enumerate(key_features):
        ax = axes[idx]
        
        # 提取数据
        values = [phys[feat_key][labels == c].mean() for c in unique]
        
        # 绘制柱子
        x_pos = np.arange(len(unique))
        bars = ax.bar(x_pos, values, color=[colors[c] for c in unique],
                     edgecolor='black', linewidth=1.5, alpha=0.85, width=0.6)
        
        # 在柱子上标注数值
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}', ha='center', va='bottom',
                   fontsize=10, fontweight='bold')
        
        # 设置标签
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'C{c}' for c in unique], fontsize=11, fontweight='bold')
        ax.set_ylabel(f'{feat_name} ({unit})', fontsize=11, fontweight='bold')
        ax.set_title(f'({chr(97+idx)}) {feat_name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 美化
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.suptitle('Cluster Feature Comparison (Key Metrics)', 
                fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'paper_bars.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(save_dir, 'paper_bars.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"✓ 已保存: paper_bars.png/pdf")


# ============================================================
# 图3：Box Plot（箱线图）
# ============================================================
def plot_boxplots(labels, phys, stats, save_dir):
    """绘制箱线图 - 显示分布"""
    print("���制箱线图...")
    
    unique = sorted(np.unique(labels))
    colors = {c: CLUSTER_COLORS.get(c, f'C{c}') for c in unique}
    
    # 选择关键特征
    key_features = [
        ('avg_speed_mov',     'Speed (moving)', 'km/h'),
        ('soc_rate',          'Energy Consumption', '%/min'),
        ('idle_ratio',        'Idle Ratio', ''),
        ('seg_length',        'Segment Duration', 'steps'),
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (feat_key, feat_name, unit) in enumerate(key_features):
        ax = axes[idx]
        
        # 准备数据
        data_list = [phys[feat_key][labels == c] for c in unique]
        
        # 绘制箱线图
        bp = ax.boxplot(data_list, labels=[f'C{c}' for c in unique],
                       patch_artist=True, widths=0.6,
                       medianprops=dict(color='#E67E22', linewidth=2.5),
                       whiskerprops=dict(linewidth=1.2),
                       capprops=dict(linewidth=1.2),
                       boxprops=dict(linewidth=1.5))
        
        # 着色
        for patch, c in zip(bp['boxes'], unique):
            patch.set_facecolor(colors[c])
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
        
        # 设置标签
        ax.set_ylabel(f'{feat_name} ({unit})', fontsize=11, fontweight='bold')
        ax.set_title(f'({chr(97+idx)}) {feat_name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.suptitle('Cluster Feature Distribution (Box Plots)', 
                fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'paper_boxplots.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(save_dir, 'paper_boxplots.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"✓ 已保存: paper_boxplots.png/pdf")


# ============================================================
# 图4：Summary Table（汇总表格）
# ============================================================
def plot_summary_table(labels, phys, stats, save_dir):
    """绘制汇总表格"""
    print("绘制汇总表格...")
    
    unique = sorted(np.unique(labels))
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # 准备表格数据
    rows = []
    
    # 表头
    header = ['Feature', 'Unit'] + [f'C{c}: {stats[c].get("label", f"C{c}")}' for c in unique]
    
    # 数据行
    for feat_key, feat_name, unit, *values in FEATURES_CONFIG:
        row = [feat_name, unit] + [f'{v:.4f}' for v in values[:len(unique)]]
        rows.append(row)
    
    # 创建表格
    table = ax.table(cellText=rows, colLabels=header, cellLoc='center',
                    loc='center', colWidths=[0.25, 0.1] + [0.15]*len(unique))
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # 美化表头
    for i in range(len(header)):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white', fontsize=11)
    
    # 美化数据行
    for i in range(1, len(rows) + 1):
        for j in range(len(header)):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#E7E6E6')
            else:
                cell.set_facecolor('#F2F2F2')
            cell.set_text_props(fontsize=10)
    
    # 美化聚类列
    for i in range(1, len(rows) + 1):
        for j in range(2, len(header)):
            cell = table[(i, j)]
            cell.set_facecolor('#D9E8F5')
    
    ax.set_title('Cluster Feature Summary (Mean Values)', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'paper_table.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(save_dir, 'paper_table.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"✓ 已保存: paper_table.png/pdf")


# ============================================================
# 图5：PCA 散点图 + 聚类标注
# ============================================================
def plot_pca_clusters(labels, z_pca, phys, stats, save_dir):
    """绘制 PCA 降维后的聚类散点图"""
    print("绘制 PCA 聚类散点图...")
    
    from sklearn.decomposition import PCA
    
    unique = sorted(np.unique(labels))
    colors = {c: CLUSTER_COLORS.get(c, f'C{c}') for c in unique}
    
    # PCA 到 2D
    pca = PCA(n_components=2, random_state=42)
    z_2d = pca.fit_transform(z_pca)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 绘制每个聚类
    np.random.seed(42)
    for c in unique:
        mask = labels == c
        n_show = min(5000, mask.sum())
        idx = np.random.choice(np.where(mask)[0], n_show, replace=False)
        
        lbl = stats[c].get('label', f'C{c}')
        ax.scatter(z_2d[idx, 0], z_2d[idx, 1],
                  c=colors[c], s=20, alpha=0.5, 
                  label=f'C{c}: {lbl} (n={mask.sum():,})',
                  edgecolors='none')
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', 
                 fontsize=12, fontweight='bold')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', 
                 fontsize=12, fontweight='bold')
    ax.set_title('Latent Space Clustering (PCA Visualization)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11, framealpha=0.95, edgecolor='black')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'paper_pca.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(save_dir, 'paper_pca.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"✓ 已保存: paper_pca.png/pdf")


# ============================================================
# 主函数
# ============================================================
def main():
    cfg = CONFIG
    os.makedirs(cfg['save_dir'], exist_ok=True)
    
    print("="*70)
    print("生成顶刊风格聚类可视化")
    print("="*70)
    
    # 加载数据
    print("加载聚类结果...")
    data = np.load(cfg['clustering_results'])
    labels = data['labels']
    z_pca = data['z_pca']
    
    phys = {}
    for key in data.files:
        if key not in ['labels', 'seg_types', 'z_pca']:
            phys[key] = data[key]
    
    # 加载统计信息
    with open(cfg['clustering_summary'], 'r') as f:
        summary = json.load(f)
    
    stats = {}
    for k_str, v in summary['cluster_stats'].items():
        k = int(k_str)
        stats[k] = v
    
    # 生成所有图表
    print("\n生成图表...")
    plot_heatmap(labels, phys, stats, cfg['save_dir'])
    plot_grouped_bars(labels, phys, stats, cfg['save_dir'])
    plot_boxplots(labels, phys, stats, cfg['save_dir'])
    plot_summary_table(labels, phys, stats, cfg['save_dir'])
    plot_pca_clusters(labels, z_pca, phys, stats, cfg['save_dir'])
    
    print("\n" + "="*70)
    print("✓ 所有图表生成完成!")
    print("="*70)
    print("\n生成的文件:")
    for fn in sorted(os.listdir(cfg['save_dir'])):
        if fn.startswith('paper_'):
            fp = os.path.join(cfg['save_dir'], fn)
            if os.path.isfile(fp):
                print(f"   {fn:<40} {os.path.getsize(fp)/1024:>8.1f} KB")


if __name__ == '__main__':
    main()