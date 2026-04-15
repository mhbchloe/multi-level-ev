"""
Step 9: Generate Radar Chart + Energy-Speed Analysis Figures
仿照参考图表风格，生成雷达图和能量-速度分析图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import seaborn as sns
from scipy import stats
import pickle
import os
from math import pi

# ============ 设置论文风格 ============
plt.style.use('default')
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
})

# 论文配色方案（与参考图相似）
colors_cluster = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 蓝、橙、绿、红

results_dir = './analysis_complete_vehicles/results/'
paper_fig_dir = os.path.join(results_dir, 'paper_figures_radar')
os.makedirs(paper_fig_dir, exist_ok=True)

print("="*70)
print("📊 Generating Radar + Energy-Speed Figures")
print("="*70)

# ============ 1. 加载数据 ============
print("\n📂 Loading data...")

df = pd.read_csv(os.path.join(results_dir, 'segments_with_clusters_labeled.csv'))
embeddings = np.load(os.path.join(results_dir, 'segment_embeddings.npy'))

print(f"✅ Loaded {len(df):,} segments")

# ============ 图表 1: 雷达图 (4 个聚类并排显示) ============
print("\n📊 Figure 1: Radar Charts for All Clusters (4-in-1)")

fig = plt.figure(figsize=(18, 16))

# 选择关键特征（8 个，仿照参考图）
radar_features = ['speed_mean', 'power_mean', 'current_mean', 'voltage_mean', 
                  'soc_drop', 'duration_seconds', 'acc_std', 'speed_std']
radar_labels = ['Speed\n(km/h)', 'Power\n(W)', 'Current\n(A)', 'Voltage\n(V)', 
                'SOC Drop\n(%)', 'Duration\n(min)', 'Acc Std\n(m/s²)', 'Speed Std\n(km/h)']

num_vars = len(radar_features)
angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
angles += angles[:1]

# 为每个聚类创建子图
cluster_names = [
    'C0: Highway\nDynamic',
    'C1: Congestion\n/AC High',
    'C2: City\nModerate',
    'C3: Parking\n/Idle'
]

for cluster in range(4):
    ax = fig.add_subplot(2, 2, cluster + 1, projection='polar')
    
    cluster_df = df[df['cluster'] == cluster]
    
    # 提取该聚类的特征值
    values = []
    for feat in radar_features:
        if feat == 'duration_seconds':
            val = cluster_df[feat].mean() / 60  # 转换为分钟
        else:
            val = cluster_df[feat].mean()
        values.append(val)
    
    # 归一化（按全数据集）
    values_norm = []
    for i, feat in enumerate(radar_features):
        feat_data = []
        for c in range(4):
            c_df = df[df['cluster'] == c]
            if feat == 'duration_seconds':
                feat_data.append(c_df[feat].mean() / 60)
            else:
                feat_data.append(c_df[feat].mean())
        
        min_val, max_val = min(feat_data), max(feat_data)
        if max_val - min_val == 0:
            norm_val = 0
        else:
            norm_val = (values[i] - min_val) / (max_val - min_val)
        values_norm.append(norm_val)
    
    values_norm += values_norm[:1]
    
    # 绘制雷达图
    ax.plot(angles, values_norm, 'o-', linewidth=2.5, 
            color=colors_cluster[cluster], markersize=7, label='Actual')
    ax.fill(angles, values_norm, alpha=0.25, color=colors_cluster[cluster])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_title(f'{cluster_names[cluster]}\nSamples: {len(cluster_df):,}', 
                 fontsize=12, fontweight='bold', pad=20)

fig.suptitle('Discharge Characteristics Radar Charts by Cluster (Normalized)', 
             fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig(os.path.join(paper_fig_dir, 'Fig1_Radar_4Clusters.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(paper_fig_dir, 'Fig1_Radar_4Clusters.pdf'), dpi=300, bbox_inches='tight')
print("✅ Saved: Fig1_Radar_4Clusters.png/pdf")
plt.close()

# ============ 图表 2: 单独的叠加雷达图 (用于对比) ============
print("\n📊 Figure 2: Overlaid Radar Chart (All Clusters)")

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='polar')

for cluster in range(4):
    cluster_df = df[df['cluster'] == cluster]
    
    values = []
    for feat in radar_features:
        if feat == 'duration_seconds':
            val = cluster_df[feat].mean() / 60
        else:
            val = cluster_df[feat].mean()
        values.append(val)
    
    # 归一化
    values_norm = []
    for i, feat in enumerate(radar_features):
        feat_data = []
        for c in range(4):
            c_df = df[df['cluster'] == c]
            if feat == 'duration_seconds':
                feat_data.append(c_df[feat].mean() / 60)
            else:
                feat_data.append(c_df[feat].mean())
        
        min_val, max_val = min(feat_data), max(feat_data)
        if max_val - min_val == 0:
            norm_val = 0
        else:
            norm_val = (values[i] - min_val) / (max_val - min_val)
        values_norm.append(norm_val)
    
    values_norm += values_norm[:1]
    
    ax.plot(angles, values_norm, 'o-', linewidth=2.5,
            color=colors_cluster[cluster], markersize=7, label=f'C{cluster}')
    ax.fill(angles, values_norm, alpha=0.15, color=colors_cluster[cluster])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_labels, fontsize=11, fontweight='bold')
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=11, framealpha=0.95)
ax.set_title('Overlaid Discharge Characteristics Radar Chart', 
             fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(os.path.join(paper_fig_dir, 'Fig2_Radar_Overlaid.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(paper_fig_dir, 'Fig2_Radar_Overlaid.pdf'), dpi=300, bbox_inches='tight')
print("✅ Saved: Fig2_Radar_Overlaid.png/pdf")
plt.close()

# ============ 图表 3: 能量-速度关系的多维分析 ============
print("\n📊 Figure 3: Energy-Speed Relationship Analysis (Multi-dimensional)")

fig = plt.figure(figsize=(18, 12))

# 子图 1: 速度 vs 能量消耗（散点图，按聚类着色）
ax1 = fig.add_subplot(2, 3, 1)
for cluster in range(4):
    cluster_data = df[df['cluster'] == cluster]
    ax1.scatter(cluster_data['speed_mean'], cluster_data['power_mean'],
               alpha=0.4, s=20, color=colors_cluster[cluster], label=f'C{cluster}',
               edgecolors='none')
ax1.set_xlabel('Average Speed (km/h)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Average Power (W)', fontsize=11, fontweight='bold')
ax1.set_title('Speed vs Power Consumption', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)

# 添加聚类中心
for cluster in range(4):
    cluster_data = df[df['cluster'] == cluster]
    center_speed = cluster_data['speed_mean'].mean()
    center_power = cluster_data['power_mean'].mean()
    ax1.scatter(center_speed, center_power, marker='*', s=500,
               color=colors_cluster[cluster], edgecolors='black', linewidths=2,
               zorder=10)

# 子图 2: 速度 vs 能效（能量/速度）
ax2 = fig.add_subplot(2, 3, 2)
df['energy_efficiency'] = df['power_mean'] / (df['speed_mean'] + 0.1)

for cluster in range(4):
    cluster_data = df[df['cluster'] == cluster]
    ax2.scatter(cluster_data['speed_mean'], cluster_data['energy_efficiency'],
               alpha=0.4, s=20, color=colors_cluster[cluster], label=f'C{cluster}',
               edgecolors='none')
ax2.set_xlabel('Average Speed (km/h)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Energy per Unit Speed (W·h/km)', fontsize=11, fontweight='bold')
ax2.set_title('Speed vs Energy Efficiency', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)

# 子图 3: 速度 vs 电流
ax3 = fig.add_subplot(2, 3, 3)
for cluster in range(4):
    cluster_data = df[df['cluster'] == cluster]
    ax3.scatter(cluster_data['speed_mean'], cluster_data['current_mean'],
               alpha=0.4, s=20, color=colors_cluster[cluster], label=f'C{cluster}',
               edgecolors='none')
ax3.set_xlabel('Average Speed (km/h)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Average Current (A)', fontsize=11, fontweight='bold')
ax3.set_title('Speed vs Current Draw', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(alpha=0.3)

# 子图 4: 聚类特征对比（功率）
ax4 = fig.add_subplot(2, 3, 4)
cluster_stats = []
for cluster in range(4):
    cluster_data = df[df['cluster'] == cluster]
    cluster_stats.append({
        'speed': cluster_data['speed_mean'].mean(),
        'power': cluster_data['power_mean'].mean(),
        'current': cluster_data['current_mean'].mean(),
        'soc_drop': cluster_data['soc_drop'].mean(),
    })

x = np.arange(4)
width = 0.2

powers = [cs['power'] for cs in cluster_stats]
ax4.bar(x, powers, width, label='Power', color=colors_cluster, alpha=0.8, edgecolor='black')
ax4.set_xlabel('Cluster', fontsize=11, fontweight='bold')
ax4.set_ylabel('Average Power (W)', fontsize=11, fontweight='bold')
ax4.set_title('Power Consumption by Cluster', fontsize=12, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels([f'C{i}' for i in range(4)])
ax4.grid(alpha=0.3, axis='y')

# 添加数值标签
for i, (bar, power) in enumerate(zip(ax4.patches, powers)):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f'{power:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 子图 5: 聚类特征对比（速度）
ax5 = fig.add_subplot(2, 3, 5)
speeds = [cs['speed'] for cs in cluster_stats]
ax5.bar(x, speeds, width, label='Speed', color=colors_cluster, alpha=0.8, edgecolor='black')
ax5.set_xlabel('Cluster', fontsize=11, fontweight='bold')
ax5.set_ylabel('Average Speed (km/h)', fontsize=11, fontweight='bold')
ax5.set_title('Driving Speed by Cluster', fontsize=12, fontweight='bold')
ax5.set_xticks(x)
ax5.set_xticklabels([f'C{i}' for i in range(4)])
ax5.grid(alpha=0.3, axis='y')

for i, (bar, speed) in enumerate(zip(ax5.patches, speeds)):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f'{speed:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 子图 6: 能量消耗等级分布
ax6 = fig.add_subplot(2, 3, 6)
power_bins = [0, 50, 100, 150, 200, 300]
power_labels = ['<50W', '50-100W', '100-150W', '150-200W', '>200W']

x_pos = np.arange(len(power_labels))
width_group = 0.2

for cluster in range(4):
    cluster_data = df[df['cluster'] == cluster]['power_mean'].values
    power_dist = np.histogram(cluster_data, bins=power_bins)[0]
    ax6.bar(x_pos + cluster * width_group, power_dist, width_group,
           label=f'C{cluster}', color=colors_cluster[cluster], alpha=0.8, edgecolor='black')

ax6.set_xlabel('Power Consumption Level', fontsize=11, fontweight='bold')
ax6.set_ylabel('Number of Segments', fontsize=11, fontweight='bold')
ax6.set_title('Distribution of Power Levels', fontsize=12, fontweight='bold')
ax6.set_xticks(x_pos + width_group * 1.5)
ax6.set_xticklabels(power_labels, rotation=15, ha='right')
ax6.legend(fontsize=10, loc='upper right')
ax6.grid(alpha=0.3, axis='y')

fig.suptitle('Energy-Speed Relationship Analysis', fontsize=15, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(os.path.join(paper_fig_dir, 'Fig3_Energy_Speed_Analysis.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(paper_fig_dir, 'Fig3_Energy_Speed_Analysis.pdf'), dpi=300, bbox_inches='tight')
print("✅ Saved: Fig3_Energy_Speed_Analysis.png/pdf")
plt.close()

# ============ 图表 4: 详细的能量-速度关系（参考图参考设计） ============
print("\n📊 Figure 4: Detailed Energy-Speed Curves")

fig = plt.figure(figsize=(16, 12))

# 创建 2x2 的子图布局
for cluster in range(4):
    ax = fig.add_subplot(2, 2, cluster + 1)
    
    cluster_data = df[df['cluster'] == cluster].copy()
    
    # 按速度分组统计
    speed_bins = np.linspace(0, cluster_data['speed_mean'].max(), 20)
    cluster_data['speed_bin'] = pd.cut(cluster_data['speed_mean'], bins=speed_bins)
    
    grouped = cluster_data.groupby('speed_bin').agg({
        'power_mean': ['mean', 'std', 'count'],
        'speed_mean': 'mean'
    }).reset_index(drop=True)
    
    grouped = grouped.dropna()
    
    # 绘制曲线
    ax.plot(grouped[('speed_mean', 'mean')], grouped[('power_mean', 'mean')],
           'o-', linewidth=2.5, markersize=8, color=colors_cluster[cluster], label='Mean')
    
    # 添加误差带
    ax.fill_between(grouped[('speed_mean', 'mean')],
                   grouped[('power_mean', 'mean')] - grouped[('power_mean', 'std')],
                   grouped[('power_mean', 'mean')] + grouped[('power_mean', 'std')],
                   alpha=0.2, color=colors_cluster[cluster], label='±1 Std Dev')
    
    ax.set_xlabel('Average Speed (km/h)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Average Power (W)', fontsize=11, fontweight='bold')
    ax.set_title(f'{cluster_names[cluster]}\nSpeed-Power Relationship', 
                fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

fig.suptitle('Speed-Power Curves for Each Cluster', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(paper_fig_dir, 'Fig4_Speed_Power_Curves.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(paper_fig_dir, 'Fig4_Speed_Power_Curves.pdf'), dpi=300, bbox_inches='tight')
print("✅ Saved: Fig4_Speed_Power_Curves.png/pdf")
plt.close()

# ============ 图表 5: 能量效率综合对比 ============
print("\n📊 Figure 5: Comprehensive Energy Efficiency Comparison")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Energy Efficiency Metrics Comparison', fontsize=15, fontweight='bold')

# 指标 1: 能效（功率/速度）
ax = axes[0, 0]
efficiency_data = []
for cluster in range(4):
    cluster_data = df[df['cluster'] == cluster]
    eff = (cluster_data['power_mean'] / (cluster_data['speed_mean'] + 0.1)).mean()
    efficiency_data.append(eff)

bars = ax.bar(range(4), efficiency_data, color=colors_cluster, alpha=0.8, edgecolor='black', linewidth=2)
ax.set_ylabel('Energy per Unit Speed (W·h/km)', fontsize=11, fontweight='bold')
ax.set_title('Energy Efficiency (Power/Speed)', fontsize=12, fontweight='bold')
ax.set_xticks(range(4))
ax.set_xticklabels([f'C{i}' for i in range(4)])
ax.grid(alpha=0.3, axis='y')
for bar, eff in zip(bars, efficiency_data):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
           f'{eff:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 指标 2: SOC 下降率
ax = axes[0, 1]
soc_rate_data = []
for cluster in range(4):
    cluster_data = df[df['cluster'] == cluster]
    rate = (cluster_data['soc_drop'] / (cluster_data['duration_seconds'] / 60 + 0.1)).mean()
    soc_rate_data.append(rate)

bars = ax.bar(range(4), soc_rate_data, color=colors_cluster, alpha=0.8, edgecolor='black', linewidth=2)
ax.set_ylabel('SOC Drop Rate (%/min)', fontsize=11, fontweight='bold')
ax.set_title('Battery Discharge Rate', fontsize=12, fontweight='bold')
ax.set_xticks(range(4))
ax.set_xticklabels([f'C{i}' for i in range(4)])
ax.grid(alpha=0.3, axis='y')
for bar, rate in zip(bars, soc_rate_data):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
           f'{rate:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 指标 3: 能量消耗强度（功率）
ax = axes[1, 0]
power_data = [df[df['cluster'] == c]['power_mean'].mean() for c in range(4)]
bars = ax.bar(range(4), power_data, color=colors_cluster, alpha=0.8, edgecolor='black', linewidth=2)
ax.set_ylabel('Average Power (W)', fontsize=11, fontweight='bold')
ax.set_title('Power Consumption Intensity', fontsize=12, fontweight='bold')
ax.set_xticks(range(4))
ax.set_xticklabels([f'C{i}' for i in range(4)])
ax.grid(alpha=0.3, axis='y')
for bar, power in zip(bars, power_data):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
           f'{power:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 指标 4: 驾驶速度
ax = axes[1, 1]
speed_data = [df[df['cluster'] == c]['speed_mean'].mean() for c in range(4)]
bars = ax.bar(range(4), speed_data, color=colors_cluster, alpha=0.8, edgecolor='black', linewidth=2)
ax.set_ylabel('Average Speed (km/h)', fontsize=11, fontweight='bold')
ax.set_title('Driving Speed', fontsize=12, fontweight='bold')
ax.set_xticks(range(4))
ax.set_xticklabels([f'C{i}' for i in range(4)])
ax.grid(alpha=0.3, axis='y')
for bar, speed in zip(bars, speed_data):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
           f'{speed:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(paper_fig_dir, 'Fig5_Efficiency_Metrics.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(paper_fig_dir, 'Fig5_Efficiency_Metrics.pdf'), dpi=300, bbox_inches='tight')
print("✅ Saved: Fig5_Efficiency_Metrics.png/pdf")
plt.close()

print(f"\n{'='*70}")
print(f"✅ All Radar + Energy-Speed Figures Generated!")
print(f"{'='*70}")
print(f"\n📁 Output Directory: {paper_fig_dir}")
print(f"\n📊 Figures Generated:")
print(f"   ✓ Fig1_Radar_4Clusters.png/pdf (4-in-1 radar charts)")
print(f"   ✓ Fig2_Radar_Overlaid.png/pdf (overlaid comparison)")
print(f"   ✓ Fig3_Energy_Speed_Analysis.png/pdf (6-panel analysis)")
print(f"   ✓ Fig4_Speed_Power_Curves.png/pdf (detailed curves)")
print(f"   ✓ Fig5_Efficiency_Metrics.png/pdf (comprehensive metrics)")
print(f"\n✨ Ready for SCI Journal Publication!")
print(f"{'='*70}\n")