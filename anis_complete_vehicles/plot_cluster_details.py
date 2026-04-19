"""
绘制详细特征对比图 (图2)
每个聚类一个子图，显示所有特征的实际值
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
import json

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
rcParams['font.size'] = 10

# ============================================================
# 配置
# ============================================================
CONFIG = {
    'clustering_results': './analysis_complete_vehicles/results/clustering_v3/clustering_v3_results.npz',
    'clustering_summary': './analysis_complete_vehicles/results/clustering_v3/clustering_v3_summary.json',
    'save_dir': './analysis_complete_vehicles/results/clustering_v3',
    'output_name': 'cluster_features_detail',
}

CLUSTER_COLORS = {0: '#5B9BD5', 1: '#70AD47', 2: '#C0504D', 3: '#FFC000'}

# ============================================================
# 绘制详细特征对比
# ============================================================
def plot_cluster_features_detail(labels, phys, stats, save_dir, output_name='cluster_features_detail'):
    """
    绘制 2x2 子图，每个聚类显示所有特征的实际值
    """
    print("="*70)
    print("绘制详细特征对比图...")
    print("="*70)
    
    unique = sorted(np.unique(labels))
    colors = {c: CLUSTER_COLORS.get(c, f'C{c}') for c in unique}
    
    # 定义要显示的特征
    features_cfg = [
        ('avg_speed_mov',     'Speed (moving)',     'km/h'),
        ('speed_std',         'Acc. Var.',          'm/s^2'),
        ('heading_change',    'Heading Change',     'deg'),
        ('soc_rate',          'Energy Rate',        '%/min'),
        ('idle_ratio',        'Idle Ratio',         ''),
        ('seg_length',        'Duration',           'steps'),
    ]
    
    # 创建 2x2 子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    os.makedirs(save_dir, exist_ok=True)
    
    for ax_idx, c in enumerate(unique):
        ax = axes[ax_idx]
        mask = labels == c
        
        # 获取聚类信息
        size = stats[c]['size']
        driving_pct = stats[c]['driving_pct']
        label = stats[c].get('label', f'C{c}')
        
        # 准备数据
        feature_names = []
        feature_values = []
        
        for feat_key, feat_name, unit in features_cfg:
            val = stats[c].get(f'{feat_key}_mean', 0)
            if unit:
                feature_names.append(f'{feat_name}\n({unit})')
            else:
                feature_names.append(feat_name)
            feature_values.append(val)
        
        # 绘制横向柱状图
        bars = ax.barh(feature_names, feature_values, color=colors[c], 
                       edgecolor='#333', linewidth=1.5, alpha=0.85)
        
        # 在柱子上显示数值
        for i, (bar, val) in enumerate(zip(bars, feature_values)):
            ax.text(val, bar.get_y() + bar.get_height()/2, 
                   f' {val:.4f}' if val < 1 else f' {val:.2f}',
                   va='center', ha='left', fontsize=9, fontweight='bold')
        
        # 设置标题和标签
        ax.set_title(f'C{c}: {label}\n(n={size:,}, {driving_pct:.0f}% driving)', 
                    fontsize=12, fontweight='bold', color=colors[c])
        ax.set_xlabel('Value', fontsize=10)
        ax.grid(True, alpha=0.15, axis='x')
        ax.set_xlim(left=0)
    
    plt.suptitle('Cluster Feature Profiles (colored = this cluster, gray = others)', 
                 fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # 保存
    for fmt, dpi in [('.png', 300), ('.pdf', None)]:
        filepath = os.path.join(save_dir, f'{output_name}{fmt}')
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        print(f"✓ 已保存: {filepath}")
    
    plt.close(fig)


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
    
    # 绘制详细特征对比图
    plot_cluster_features_detail(labels, phys, stats, cfg['save_dir'], cfg['output_name'])
    
    print("\n" + "="*70)
    print("✓ 完成!")
    print("="*70)


if __name__ == '__main__':
    main()