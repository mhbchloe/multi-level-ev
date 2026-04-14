"""
生成综合分析图 - Cross-Attention K=4聚类
包含：PCA空间分布、能效、速度-能耗关系、簇大小、功率、行程时长
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("🎨 Generating Comprehensive Analysis Plots")
print("="*70)

# ==================== 加载数据 ====================
print("\n📂 Loading data...")

labels = np.load('./results/labels_k4_crossattn.npy')
features = np.load('./results/features_k4_crossattn.npy')
driving_seqs = np.load('./results/temporal_soc_full/driving_sequences.npy', allow_pickle=True)
energy_seqs = np.load('./results/temporal_soc_full/energy_sequences.npy', allow_pickle=True)

# 确保长度一致
min_len = min(len(labels), len(features), len(driving_seqs), len(energy_seqs))
labels = labels[:min_len]
features = features[:min_len]
driving_seqs = driving_seqs[:min_len]
energy_seqs = energy_seqs[:min_len]

print(f"✅ Loaded {len(labels):,} samples")
print(f"   Feature dimension: {features.shape[1]}D")

# ==================== 提取详细特征 ====================
print("\n📊 Extracting detailed features for each sample...")

# 为每个样本计算特征
sample_features = []

for i in range(len(labels)):
    feat = {}
    
    # 驾驶特征
    spd = driving_seqs[i][:, 0]
    acc = driving_seqs[i][:, 1]
    
    feat['avg_speed'] = np.mean(spd)
    feat['max_speed'] = np.max(spd)
    feat['speed_std'] = np.std(spd)
    
    # 能量特征
    soc = energy_seqs[i][:, 0]
    v = energy_seqs[i][:, 1]
    current = energy_seqs[i][:, 2]
    
    feat['soc_drop'] = soc[0] - soc[-1] if len(soc) > 1 else 0
    feat['avg_power'] = np.mean(np.abs(v * current))
    
    # 行程特征
    feat['duration'] = len(driving_seqs[i])  # 采样点数
    feat['distance'] = np.sum(spd) / 3600 * 10  # 近似距离 (km)
    
    # 能效 (km/kWh) 或者 (%/km)
    if feat['distance'] > 0:
        feat['energy_rate'] = feat['soc_drop'] / feat['distance']  # %/km
    else:
        feat['energy_rate'] = 0
    
    sample_features.append(feat)

df_samples = pd.DataFrame(sample_features)
df_samples['cluster'] = labels

print(f"✅ Extracted features for {len(df_samples)} samples")

# ==================== 计算每个簇的统计 ====================
print("\n📊 Computing cluster statistics...")

cluster_stats = []
for cluster_id in range(4):
    cluster_data = df_samples[df_samples['cluster'] == cluster_id]
    
    stats = {
        'cluster': cluster_id,
        'count': len(cluster_data),
        'avg_speed_mean': cluster_data['avg_speed'].mean(),
        'energy_rate_mean': cluster_data['energy_rate'].mean(),
        'energy_rate_median': cluster_data['energy_rate'].median(),
        'avg_power_mean': cluster_data['avg_power'].mean(),
        'duration_mean': cluster_data['duration'].mean()
    }
    cluster_stats.append(stats)

df_stats = pd.DataFrame(cluster_stats)

print("\n📊 Cluster Statistics:")
print(df_stats)

# ==================== PCA降维 ====================
print("\n🔍 Performing PCA for visualization...")

pca = PCA(n_components=2)
features_2d = pca.fit_transform(features)

print(f"✅ PCA completed")
print(f"   Explained variance: {pca.explained_variance_ratio_.sum()*100:.1f}%")

# ==================== 创建6子图 ====================
print("\n🎨 Creating comprehensive plot...")

fig = plt.figure(figsize=(20, 12))

colors = ['#3498DB', '#2ECC71', '#E74C3C', '#F39C12']
color_names = ['C0', 'C1', 'C2', 'C3']

# ==================== 子图1: PCA散点图 ====================
ax1 = plt.subplot(2, 3, 1)

for cluster_id in range(4):
    mask = labels == cluster_id
    ax1.scatter(features_2d[mask, 0], features_2d[mask, 1],
               c=colors[cluster_id], label=f'Cluster {cluster_id}',
               alpha=0.6, s=20, edgecolors='none')

ax1.set_xlabel('PC1', fontsize=12, fontweight='bold')
ax1.set_ylabel('PC2', fontsize=12, fontweight='bold')
ax1.set_title('Driving Style Clusters (PCA Space)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10, loc='upper right')
ax1.grid(True, alpha=0.3)

# ==================== 子图2: 能效箱线图 ====================
ax2 = plt.subplot(2, 3, 2)

# 过滤异常值
energy_data = []
positions = []
for cluster_id in range(4):
    cluster_data = df_samples[df_samples['cluster'] == cluster_id]['energy_rate']
    # 去除极端异常值
    q1, q3 = cluster_data.quantile([0.25, 0.75])
    iqr = q3 - q1
    filtered = cluster_data[(cluster_data >= q1 - 1.5*iqr) & (cluster_data <= q3 + 1.5*iqr)]
    energy_data.append(filtered)
    positions.append(cluster_id)

bp = ax2.boxplot(energy_data, positions=positions, widths=0.6,
                patch_artist=True, showfliers=False)

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax2.set_xlabel('Cluster', fontsize=12, fontweight='bold')
ax2.set_ylabel('Energy Rate (%/km)', fontsize=12, fontweight='bold')
ax2.set_title('Energy Efficiency by Cluster', fontsize=14, fontweight='bold')
ax2.set_xticks(range(4))
ax2.set_xticklabels(['C0', 'C1', 'C2', 'C3'])
ax2.grid(True, alpha=0.3, axis='y')

# ==================== 子图3: 速度vs能耗散点图 ====================
ax3 = plt.subplot(2, 3, 3)

for cluster_id in range(4):
    avg_speed = df_stats.loc[cluster_id, 'avg_speed_mean']
    energy_rate = df_stats.loc[cluster_id, 'energy_rate_mean']
    
    ax3.scatter(avg_speed, energy_rate, 
               c=colors[cluster_id], s=500, alpha=0.8,
               edgecolors='black', linewidth=3, zorder=10)
    
    ax3.text(avg_speed, energy_rate, f'C{cluster_id}',
            ha='center', va='center', fontsize=14, fontweight='bold',
            color='white', zorder=11)

ax3.set_xlabel('Average Speed (km/h)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Energy Rate (%/km)', fontsize=12, fontweight='bold')
ax3.set_title('Speed vs Energy Consumption', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)

# ==================== 子图4: 簇大小分布 ====================
ax4 = plt.subplot(2, 3, 4)

cluster_counts = [df_stats.loc[i, 'count'] for i in range(4)]

bars = ax4.bar(range(4), cluster_counts, color=colors, alpha=0.8,
              edgecolor='black', linewidth=2)

ax4.set_xlabel('Cluster', fontsize=12, fontweight='bold')
ax4.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
ax4.set_title('Cluster Size Distribution', fontsize=14, fontweight='bold')
ax4.set_xticks(range(4))
ax4.set_xticklabels(['C0', 'C1', 'C2', 'C3'])
ax4.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar, count in zip(bars, cluster_counts):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(count):,}', ha='center', va='bottom',
            fontsize=11, fontweight='bold')

# ==================== 子图5: 功率特征 ====================
ax5 = plt.subplot(2, 3, 5)

power_values = [df_stats.loc[i, 'avg_power_mean'] for i in range(4)]

bars = ax5.bar(range(4), power_values, color=colors, alpha=0.8,
              edgecolor='black', linewidth=2)

ax5.set_xlabel('Cluster', fontsize=12, fontweight='bold')
ax5.set_ylabel('Average Max Power (W)', fontsize=12, fontweight='bold')
ax5.set_title('Power Characteristics', fontsize=14, fontweight='bold')
ax5.set_xticks(range(4))
ax5.set_xticklabels(['C0', 'C1', 'C2', 'C3'])
ax5.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar, val in zip(bars, power_values):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1f}', ha='center', va='bottom',
            fontsize=11, fontweight='bold')

# ==================== 子图6: 行程时长 ====================
ax6 = plt.subplot(2, 3, 6)

duration_values = [df_stats.loc[i, 'duration_mean'] for i in range(4)]

bars = ax6.bar(range(4), duration_values, color=colors, alpha=0.8,
              edgecolor='black', linewidth=2)

ax6.set_xlabel('Cluster', fontsize=12, fontweight='bold')
ax6.set_ylabel('Average Duration (points)', fontsize=12, fontweight='bold')
ax6.set_title('Trip Duration', fontsize=14, fontweight='bold')
ax6.set_xticks(range(4))
ax6.set_xticklabels(['C0', 'C1', 'C2', 'C3'])
ax6.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar, val in zip(bars, duration_values):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.0f}', ha='center', va='bottom',
            fontsize=11, fontweight='bold')

# ==================== 总标题 ====================
fig.suptitle('Comprehensive Cluster Analysis (Cross-Attention, K=4)', 
            fontsize=18, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.99])

output = './results/comprehensive_analysis_k4_crossattn.png'
plt.savefig(output, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✅ Saved: {output}")

# ==================== 保存统计数据 ====================
df_stats.to_csv('./results/cluster_statistics_comprehensive.csv', 
               encoding='utf-8-sig', index=False)
print(f"✅ Saved: cluster_statistics_comprehensive.csv")

# ==================== 生成详细报告 ====================
print("\n" + "="*70)
print("📊 Cluster Analysis Summary")
print("="*70)

for cluster_id in range(4):
    stats = df_stats.loc[cluster_id]
    print(f"\n🔷 Cluster {cluster_id}:")
    print(f"   Sample count: {stats['count']:,} ({stats['count']/len(labels)*100:.1f}%)")
    print(f"   Avg Speed: {stats['avg_speed_mean']:.2f} km/h")
    print(f"   Energy Rate: {stats['energy_rate_mean']:.3f} %/km")
    print(f"   Avg Power: {stats['avg_power_mean']:.2f} W")
    print(f"   Avg Duration: {stats['duration_mean']:.0f} points")
    
    # 简单解释
    if stats['avg_speed_mean'] > 50:
        print(f"   💡 Interpretation: Highway driving")
    elif stats['avg_speed_mean'] < 30:
        print(f"   💡 Interpretation: Urban/congested")
    else:
        print(f"   💡 Interpretation: Mixed conditions")

print("\n" + "="*70)
print("✅ Comprehensive Analysis Complete!")
print("="*70)
print(f"\n📁 Generated:")
print(f"   1. {output}")
print(f"   2. cluster_statistics_comprehensive.csv")
print("="*70)