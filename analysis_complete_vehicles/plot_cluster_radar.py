"""
绘制 Radar 图 (图1)
基于聚类结果生成归一化的雷达图
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
import os
import json

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
rcParams['font.size'] = 11

# ============================================================
# 配置
# ============================================================
CONFIG = {
    'clustering_results': './analysis_complete_vehicles/results/clustering_v3/clustering_v3_results.npz',
    'clustering_summary': './analysis_complete_vehicles/results/clustering_v3/clustering_v3_summary.json',
    'save_dir': './analysis_complete_vehicles/results/clustering_v3',
    'output_name': 'cluster_radar_chart',
}

CLUSTER_COLORS = {0: '#5B9BD5', 1: '#70AD47', 2: '#C0504D', 3: '#FFC000'}

# ============================================================
# 绘制 Radar 图
# ============================================================
def plot_radar_chart(labels, phys, stats, save_dir, output_name='cluster_radar_chart'):
    """
    绘制归一化的雷达图 (Rank-based: Low → High)
    """
    print("="*70)
    print("绘制 Radar 雷达图...")
    print("="*70)
    
    unique = sorted(np.unique(labels))
    colors = {c: CLUSTER_COLORS.get(c, f'C{c}') for c in unique}

    # 定义雷达图的特征
    radar_cfg = [
        ('avg_speed_mov',     'Speed\n(moving)',        'km/h'),
        ('speed_std',         'Speed\nVariation',       'km/h'),
        ('acc_std_mov',       'Acceleration\nVariation', 'm/s^2'),
        ('heading_change',    'Turning\nIntensity',     'deg'),
        ('soc_rate',          'Energy\nConsumption',    '%/min'),
        ('idle_ratio',        'Idle\nRatio',            ''),
        ('seg_length',        'Segment\nDuration',      'steps'),
    ]
    
    feat_keys = [r[0] for r in radar_cfg]
    feat_labels = [r[1] for r in radar_cfg]
    n_feats = len(feat_keys)
    
    # 计算角度
    angles = np.linspace(0, 2 * np.pi, n_feats, endpoint=False).tolist()
    angles += angles[:1]  # 闭合雷达图

    # 提取原始值
    raw = {}
    for c in unique:
        mask = labels == c
        raw[c] = [float(np.mean(phys[k][mask])) for k in feat_keys]
    
    # 归一化到 [0, 1]
    arr = np.array([raw[c] for c in unique])
    fmin, fmax = arr.min(0), arr.max(0)
    frng = fmax - fmin
    frng[frng < 1e-10] = 1.0
    norm = {c: ((np.array(raw[c]) - fmin) / frng).tolist() for c in unique}

    # 绘制
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    for c in unique:
        vals = norm[c] + norm[c][:1]  # 闭合
        lbl = stats[c].get('label', f'C{c}')
        ax.plot(angles, vals, 'o-', lw=2.5, ms=8, 
                label=f'C{c}: {lbl}', color=colors[c])
        ax.fill(angles, vals, alpha=0.1, color=colors[c])
    
    # 设置标签和范围
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feat_labels, fontsize=11)
    ax.set_ylim(-0.05, 1.15)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['Low', '', '', '', 'High'], fontsize=9)
    
    # 图例和标题
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), 
              fontsize=11, framealpha=0.95)
    ax.set_title('Cluster Feature Profiles\n(Rank-based: Low → High)', 
                 fontsize=14, fontweight='bold', pad=25)
    ax.grid(True, alpha=0.3)

    # 保存
    os.makedirs(save_dir, exist_ok=True)
    for fmt, dpi in [('.png', 300), ('.pdf', None)]:
        filepath = os.path.join(save_dir, f'{output_name}{fmt}')
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        print(f"✓ 已保存: {filepath}")
    
    plt.close(fig)

    # 打印原始值表
    print("\n   原始值 (Mean):")
    hdr = f"   {'Feature':>20} {'Unit':>10}"
    for c in unique:
        hdr += f"  {'C' + str(c):>12}"
    print(hdr)
    print(f"   {'─' * 20} {'─' * 10}  " + "  ".join(['─' * 12] * len(unique)))
    
    for i, (key, name, unit) in enumerate(radar_cfg):
        line = f"   {name.replace(chr(10), ' '):>20} {unit:>10}"
        for c in unique:
            line += f"  {raw[c][i]:>12.4f}"
        print(line)


# ============================================================
# 主函数
# ============================================================
def main():
    cfg = CONFIG
    
    # 加载数据
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
    
    # 转换 stats 格式
    stats = {}
    for k_str, v in summary['cluster_stats'].items():
        k = int(k_str)
        stats[k] = v
    
    # 绘制 Radar 图
    plot_radar_chart(labels, phys, stats, cfg['save_dir'], cfg['output_name'])
    
    print("\n" + "="*70)
    print("✓ 完成!")
    print("="*70)


if __name__ == '__main__':
    main()