"""
优化布局的排名式雷达图 - 完全重设计
- 排名数字放在最外圈（单独一圈）
- 特征名字放在中间
- 标题不重叠
- 文字完全分离
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
from matplotlib.lines import Line2D
import os
import json

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
rcParams['font.size'] = 11
rcParams['axes.linewidth'] = 1.8
rcParams['grid.linewidth'] = 1.0

# ============================================================
# 配置
# ============================================================
CONFIG = {
    'clustering_results': './analysis_complete_vehicles/results/clustering_v3/clustering_v3_results.npz',
    'clustering_summary': './analysis_complete_vehicles/results/clustering_v3/clustering_v3_summary.json',
    'save_dir': './analysis_complete_vehicles/results/clustering_v3',
}

CLUSTER_COLORS = {0: '#5B9BD5', 1: '#70AD47', 2: '#C0504D', 3: '#FFC000'}


# ============================================================
# 优化布局的雷达图 - 重新设计
# ============================================================
def plot_radar_optimized(labels, phys, stats, save_dir):
    """
    绘制优化布局的雷达图
    - 排名数字在最外圈
    - 特征名字在中间圈
    - 避免所有重叠
    """
    print("="*70)
    print("绘制优化布局的排名式雷达图...")
    print("="*70)
    
    unique = sorted(np.unique(labels))
    colors = {c: CLUSTER_COLORS.get(c, f'C{c}') for c in unique}

    # 定义特征
    features = [
        ('avg_speed_mov',     'Speed\n(moving)'),
        ('speed_std',         'Speed\nVariation'),
        ('acc_std_mov',       'Acceleration\nVariation'),
        ('heading_change',    'Turning\nIntensity'),
        ('soc_rate',          'Energy\nConsumption'),
        ('idle_ratio',        'Idle\nRatio'),
        ('seg_length',        'Segment\nDuration'),
    ]
    
    feat_keys = [f[0] for f in features]
    feat_labels = [f[1] for f in features]
    n_feats = len(feat_keys)
    
    # ========== 计算排名 ==========
    raw_values = {}
    for feat_key in feat_keys:
        raw_values[feat_key] = []
        for c in unique:
            mask = labels == c
            mean_val = float(np.mean(phys[feat_key][mask]))
            raw_values[feat_key].append(mean_val)
    
    ranks = {c: [] for c in unique}
    for feat_key in feat_keys:
        values = np.array(raw_values[feat_key])
        rank_order = np.argsort(values)
        for rank_idx, cluster_idx in enumerate(rank_order):
            c = unique[cluster_idx]
            ranks[c].append(rank_idx + 1)

    # ========== 绘制 ==========
    angles = np.linspace(0, 2 * np.pi, n_feats, endpoint=False).tolist()
    angles += angles[:1]
    
    # 使用 GridSpec 精确控制布局
    fig = plt.figure(figsize=(16, 14))
    
    # 创建 GridSpec
    gs = gridspec.GridSpec(3, 3, 
                          height_ratios=[0.8, 4.5, 0.5],
                          width_ratios=[0.3, 3.5, 0.5],
                          hspace=0.12, wspace=0.18,
                          left=0.08, right=0.92, top=0.94, bottom=0.06)
    
    # 标题区
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')
    ax_title.text(0.5, 0.5, 'Cluster Feature Profiles', 
                 fontsize=18, fontweight='bold', 
                 ha='center', va='center')
    ax_title.text(0.5, 0.0, '(Ranking: 1=Lowest, 4=Highest)', 
                 fontsize=13, ha='center', va='center', 
                 style='italic', color='#555')
    
    # 主图表区 (polar)
    ax_radar = fig.add_subplot(gs[1, 1], projection='polar')
    
    # 绘制所有聚类
    for c in unique:
        vals = ranks[c] + [ranks[c][0]]
        lbl = stats[c].get('label', f'C{c}')
        n_samples = stats[c]['size']
        
        ax_radar.plot(angles, vals, 'o-', lw=3.5, ms=13, 
                     label=f'C{c}: {lbl}\n(n={n_samples:,})', 
                     color=colors[c], 
                     markeredgewidth=2.5, markeredgecolor='white', 
                     alpha=0.9, zorder=5-c)
        ax_radar.fill(angles, vals, alpha=0.09, color=colors[c], zorder=1)
    
    # ========== 设置标签 - 分离排名和特征名 ==========
    ax_radar.set_xticks(angles[:-1])
    
    # 先设置空标签，然后手动添加
    ax_radar.set_xticklabels(['' for _ in angles[:-1]])
    
    # 手动添加特征标签和排名
    for i, angle in enumerate(angles[:-1]):
        # 特征标签位置（在 y=2.8 的圆周上）
        radius_feat = 2.85
        x_feat = radius_feat * np.cos(angle - np.pi/2)
        y_feat = radius_feat * np.sin(angle - np.pi/2)
        
        # 用 text 添加特征名
        ax_radar.text(angle - np.pi/2, radius_feat, feat_labels[i],
                     fontsize=11, fontweight='bold',
                     ha='center', va='center',
                     bbox=dict(boxstyle='round,pad=0.3', 
                              facecolor='white', edgecolor='none', 
                              alpha=0.85),
                     zorder=10)
        
        # 排名标签位置（在 y=4.8 的圆周上，外圈）
        radius_rank = 4.85
        ax_radar.text(angle - np.pi/2, radius_rank, 
                     f'{int((i + 1))}',
                     fontsize=10, fontweight='bold',
                     ha='center', va='center',
                     color='#666',
                     bbox=dict(boxstyle='round,pad=0.2',
                              facecolor='#f0f0f0', edgecolor='#ccc',
                              alpha=0.9, linewidth=1),
                     zorder=11)
    
    # 设置径向范围 - 扩大以容纳外部标签
    ax_radar.set_ylim(0.5, 5.2)
    ax_radar.set_yticks([1, 2, 3, 4])
    
    # 美化纵向标签
    ax_radar.set_yticklabels(['1', '2', '3', '4'], 
                            fontsize=10, fontweight='bold',
                            color='#888')
    
    # 美化网格
    ax_radar.grid(True, alpha=0.4, linewidth=1.1, linestyle='-', 
                 color='#bbb', zorder=0)
    ax_radar.set_rgrids([1, 2, 3, 4], angle=22.5, fontsize=9.5, 
                       fontweight='bold', color='#888')
    
    # 修改外围圆 - 用浅灰色
    ax_radar.spines['polar'].set_linewidth(2.2)
    ax_radar.spines['polar'].set_color('#333')
    
    # ========== 图例区 (右侧) ==========
    ax_legend = fig.add_subplot(gs[1, 2])
    ax_legend.axis('off')
    
    # 创建自定义图例
    legend_elements = []
    for c in unique:
        lbl = stats[c].get('label', f'C{c}')
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', 
                  markerfacecolor=colors[c], markersize=12,
                  markeredgecolor='white', markeredgewidth=2,
                  label=f'C{c}: {lbl}', linewidth=3, linestyle='-')
        )
    
    leg = ax_legend.legend(handles=legend_elements, 
                          loc='upper left', fontsize=12,
                          framealpha=0.98, edgecolor='black',
                          fancybox=True, shadow=False,
                          frameon=True)
    leg.get_frame().set_linewidth(1.5)
    
    # ========== 底部统计区 ==========
    ax_stats = fig.add_subplot(gs[2, :])
    ax_stats.axis('off')
    
    stats_text = "  |  ".join([
        f"C{c}: {stats[c].get('label', f'C{c}')} (n={stats[c]['size']:,}, {stats[c]['driving_pct']:.0f}% driving)"
        for c in unique
    ])
    ax_stats.text(0.5, 0.5, stats_text, 
                 fontsize=11, ha='center', va='center',
                 bbox=dict(boxstyle='round,pad=0.8', facecolor='#f5f5f5', 
                          edgecolor='#ccc', linewidth=1.5, alpha=0.8))

    # 保存
    os.makedirs(save_dir, exist_ok=True)
    for fmt, dpi in [('.png', 300), ('.pdf', None)]:
        filepath = os.path.join(save_dir, f'cluster_radar_optimized{fmt}')
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"✓ 已保存: {filepath}")
    
    plt.close(fig)

    # ========== 打印排名表 ==========
    print("\n" + "="*70)
    print("特征排名表 (1=最低, 4=最高)")
    print("="*70 + "\n")
    
    hdr = f"   {'Feature':>25}"
    for c in unique:
        hdr += f"  {'C' + str(c):>10}"
    print(hdr)
    print(f"   {'─' * 25}  " + "  ".join(['─' * 10] * len(unique)))
    
    for i, (feat_key, feat_name) in enumerate(features):
        line = f"   {feat_name.replace(chr(10), ' '):>25}"
        for c in unique:
            line += f"  {ranks[c][i]:>10}"
        print(line)
    
    # ========== 打印原始值 ==========
    print("\n" + "="*70)
    print("原始值 (用于计算排名)")
    print("="*70 + "\n")
    
    hdr = f"   {'Feature':>25} {'Unit':>12}"
    for c in unique:
        hdr += f"  {'C' + str(c):>15}"
    print(hdr)
    print(f"   {'─' * 25} {'─' * 12}  " + "  ".join(['─' * 15] * len(unique)))
    
    units = [
        ('avg_speed_mov',     'km/h'),
        ('speed_std',         'km/h'),
        ('acc_std_mov',       'm/s²'),
        ('heading_change',    'deg'),
        ('soc_rate',          '%/min'),
        ('idle_ratio',        ''),
        ('seg_length',        'steps'),
    ]
    
    for (feat_key, feat_name), (_, unit) in zip(features, units):
        line = f"   {feat_name.replace(chr(10), ' '):>25} {unit:>12}"
        for c in unique:
            mask = labels == c
            mean_val = float(np.mean(phys[feat_key][mask]))
            if mean_val < 1:
                line += f"  {mean_val:>15.5f}"
            elif mean_val < 100:
                line += f"  {mean_val:>15.2f}"
            else:
                line += f"  {mean_val:>15.1f}"
        print(line)


# ============================================================
# 主函数
# ============================================================
def main():
    cfg = CONFIG
    
    print("加载聚类结果...")
    data = np.load(cfg['clustering_results'])
    labels = data['labels']
    
    # 重新构建 phys 字典
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
    
    # 绘制优化的雷达图
    plot_radar_optimized(labels, phys, stats, cfg['save_dir'])
    
    print("\n" + "="*70)
    print("✓ 完成!")
    print("="*70)


if __name__ == '__main__':
    main()