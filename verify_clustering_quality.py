"""
验证Direct Concatenation (K=4)的聚类清晰度
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 加载结果
features_df = pd.read_csv('./results/reloaded_full/dual_channel_full_results.csv')

# 假设用的是method1 (Direct)
labels = features_df['cluster_method1_k4'].values

print("="*70)
print("🔍 Clustering Quality Analysis")
print("="*70)

# 1. 每个簇的特征统计
print(f"\n📊 Cluster Characteristics:")

key_features = ['speed_mean', 'distance_total', 'duration_minutes', 
                'soc_drop_total', 'moving_ratio']

for cid in range(4):
    cluster_data = features_df[features_df['cluster_method1_k4'] == cid]
    
    print(f"\n{'='*50}")
    print(f"Cluster {cid} ({len(cluster_data):,} trips, {len(cluster_data)/len(features_df)*100:.1f}%)")
    print("="*50)
    
    for feat in key_features:
        mean = cluster_data[feat].mean()
        std = cluster_data[feat].std()
        print(f"  {feat:20s}: {mean:7.1f} ± {std:6.1f}")

# 2. 簇间差异分析
print(f"\n{'='*70}")
print(f"🎯 Inter-Cluster Differences (how distinct are clusters?)")
print("="*70)

cluster_means = []
for cid in range(4):
    cluster_data = features_df[features_df['cluster_method1_k4'] == cid]
    means = [cluster_data[feat].mean() for feat in key_features]
    cluster_means.append(means)

cluster_means = np.array(cluster_means)

# 计算簇间距离
from scipy.spatial.distance import pdist, squareform

distances = squareform(pdist(cluster_means, metric='euclidean'))

print(f"\nCluster Distance Matrix (Euclidean):")
print("        C0      C1      C2      C3")
for i in range(4):
    print(f"C{i}  ", end='')
    for j in range(4):
        if i == j:
            print("  -    ", end='')
        else:
            print(f"{distances[i,j]:6.1f}", end=' ')
    print()

print(f"\nAverage inter-cluster distance: {distances[np.triu_indices_from(distances, k=1)].mean():.1f}")

# 3. 簇内紧凑度
print(f"\n{'='*70}")
print(f"📐 Intra-Cluster Compactness (how tight are clusters?)")
print("="*70)

for cid in range(4):
    cluster_data = features_df[features_df['cluster_method1_k4'] == cid]
    cluster_subset = cluster_data[key_features].values
    
    # 计算簇内平均距离
    if len(cluster_subset) > 1:
        intra_distances = pdist(cluster_subset[:1000], metric='euclidean')  # 采样1000个点
        avg_intra = intra_distances.mean()
        print(f"Cluster {cid}: Average intra-cluster distance = {avg_intra:.2f}")

# 4. 可视化：簇的分离度
print(f"\n📊 Creating separation visualization...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']

# 速度 vs 距离
ax = axes[0, 0]
for cid in range(4):
    cluster_data = features_df[features_df['cluster_method1_k4'] == cid]
    ax.scatter(cluster_data['speed_mean'], cluster_data['distance_total'],
              alpha=0.4, s=10, c=colors[cid], label=f'C{cid}')
ax.set_xlabel('Speed (km/h)', fontsize=12, fontweight='bold')
ax.set_ylabel('Distance (km)', fontsize=12, fontweight='bold')
ax.set_title('Speed vs Distance', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
ax.set_xlim(0, 80)
ax.set_ylim(0, 100)

# 速度 vs 持续时间
ax = axes[0, 1]
for cid in range(4):
    cluster_data = features_df[features_df['cluster_method1_k4'] == cid]
    ax.scatter(cluster_data['speed_mean'], cluster_data['duration_minutes'],
              alpha=0.4, s=10, c=colors[cid], label=f'C{cid}')
ax.set_xlabel('Speed (km/h)', fontsize=12, fontweight='bold')
ax.set_ylabel('Duration (min)', fontsize=12, fontweight='bold')
ax.set_title('Speed vs Duration', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
ax.set_xlim(0, 80)
ax.set_ylim(0, 200)

# 距离 vs 持续时间
ax = axes[0, 2]
for cid in range(4):
    cluster_data = features_df[features_df['cluster_method1_k4'] == cid]
    ax.scatter(cluster_data['distance_total'], cluster_data['duration_minutes'],
              alpha=0.4, s=10, c=colors[cid], label=f'C{cid}')
ax.set_xlabel('Distance (km)', fontsize=12, fontweight='bold')
ax.set_ylabel('Duration (min)', fontsize=12, fontweight='bold')
ax.set_title('Distance vs Duration', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
ax.set_xlim(0, 100)
ax.set_ylim(0, 200)

# 速度分布（小提琴图）
ax = axes[1, 0]
speed_data = [features_df[features_df['cluster_method1_k4'] == c]['speed_mean'] for c in range(4)]
parts = ax.violinplot(speed_data, positions=range(4), showmeans=True, showmedians=True)
for pc, color in zip(parts['bodies'], colors):
    pc.set_facecolor(color)
    pc.set_alpha(0.7)
ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
ax.set_ylabel('Speed (km/h)', fontsize=12, fontweight='bold')
ax.set_title('Speed Distribution by Cluster', fontsize=14, fontweight='bold')
ax.set_xticks(range(4))
ax.grid(axis='y', alpha=0.3)

# SOC消耗分布
ax = axes[1, 1]
soc_data = [features_df[features_df['cluster_method1_k4'] == c]['soc_drop_total'] for c in range(4)]
bp = ax.boxplot(soc_data, positions=range(4), patch_artist=True, showfliers=False)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
ax.set_ylabel('SOC Drop (%)', fontsize=12, fontweight='bold')
ax.set_title('Energy Consumption by Cluster', fontsize=14, fontweight='bold')
ax.set_xticks(range(4))
ax.grid(axis='y', alpha=0.3)

# 移动比例分布
ax = axes[1, 2]
moving_data = [features_df[features_df['cluster_method1_k4'] == c]['moving_ratio'] for c in range(4)]
bp2 = ax.boxplot(moving_data, positions=range(4), patch_artist=True, showfliers=False)
for patch, color in zip(bp2['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
ax.set_ylabel('Moving Ratio', fontsize=12, fontweight='bold')
ax.set_title('Movement Activity by Cluster', fontsize=14, fontweight='bold')
ax.set_xticks(range(4))
ax.grid(axis='y', alpha=0.3)

plt.suptitle('Direct Concatenation (K=4) - Cluster Separation Analysis', 
            fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('./results/reloaded_full/cluster_separation_analysis.png', dpi=300, bbox_inches='tight')

print(f"✅ Visualization saved!")

print("\n" + "="*70)
print("✅ Analysis Complete!")
print("="*70)

print(f"\n💡 Interpretation:")
print(f"   - Large inter-cluster distances = Clear separation ✅")
print(f"   - Small intra-cluster distances = Tight clusters ✅")
print(f"   - Non-overlapping scatter plots = Distinct patterns ✅")