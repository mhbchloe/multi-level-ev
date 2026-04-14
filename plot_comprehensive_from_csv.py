"""
直接使用已有的聚类统计结果绘制综合分析图
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("🎨 Generating Comprehensive Analysis from Existing Stats")
print("="*70)

# ==================== 加载已有的聚类统计 ====================
print("\n📂 Loading existing cluster statistics...")

csv_file = './results/cluster_features_k4_crossattn.csv'
if not Path(csv_file).exists():
    print(f"❌ File not found: {csv_file}")
    print("   Please run gru_clustering_k4_crossattn.py first")
    exit(1)

df_stats = pd.read_csv(csv_file, index_col=0)

print("\n✅ Loaded cluster statistics:")
print(df_stats)

# ==================== 加载其他必要数据 ====================
print("\n📂 Loading additional data for PCA plot...")

labels = np.load('./results/labels_k4_crossattn.npy')
features = np.load('./results/features_k4_crossattn.npy')

print(f"✅ Loaded labels and features: {len(labels):,} samples")

# ==================== PCA降维 ====================
print("\n🔍 Performing PCA...")

pca = PCA(n_components=2)
features_2d = pca.fit_transform(features)

print(f"✅ PCA completed, explained variance: {pca.explained_variance_ratio_.sum()*100:.1f}%")

# ==================== 计算每个簇的样本数 ====================
unique, counts = np.unique(labels, return_counts=True)
cluster_counts = dict(zip(unique, counts))

print(f"\n📊 Cluster distribution:")
for cluster_id, count in cluster_counts.items():
    print(f"   Cluster {cluster_id}: {count:,} ({count/len(labels)*100:.1f}%)")

# ==================== 创建综合分析图 ====================
print("\n🎨 Creating comprehensive plot...")

fig = plt.figure(figsize=(20, 12))

colors = ['#3498DB', '#2ECC71', '#E74C3C', '#F39C12']

# ==================== 子图1: PCA空间分布 ====================
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

# ==================== 子图2: 能效（SOC Drop Rate） ====================
ax2 = plt.subplot(2, 3, 2)

# 使用SOC Drop Rate作为能效指标
if 'SOC Drop Rate' in df_stats.columns:
    energy_values = df_stats['SOC Drop Rate'].values
    ylabel = 'SOC Drop Rate (%/trip)'
    title = 'Energy Consumption by Cluster'
else:
    # 如果没有这个字段，用Avg Power替代
    energy_values = df_stats['Avg Power'].values
    ylabel = 'Avg Power (W)'
    title = 'Power Consumption by Cluster'

bars = ax2.bar(range(4), energy_values, color=colors, alpha=0.8,
              edgecolor='black', linewidth=2)

ax2.set_xlabel('Cluster', fontsize=12, fontweight='bold')
ax2.set_ylabel(ylabel, fontsize=12, fontweight='bold')
ax2.set_title(title, fontsize=14, fontweight='bold')
ax2.set_xticks(range(4))
ax2.set_xticklabels(['C0', 'C1', 'C2', 'C3'])
ax2.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars, energy_values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.2f}', ha='center', va='bottom',
            fontsize=11, fontweight='bold')

# ==================== 子图3: 速度 vs 功率 ====================
ax3 = plt.subplot(2, 3, 3)

avg_speeds = df_stats['Avg Speed'].values
avg_powers = df_stats['Avg Power'].values

for cluster_id in range(4):
    ax3.scatter(avg_speeds[cluster_id], avg_powers[cluster_id],
               c=colors[cluster_id], s=500, alpha=0.8,
               edgecolors='black', linewidth=3, zorder=10)
    
    ax3.text(avg_speeds[cluster_id], avg_powers[cluster_id], f'C{cluster_id}',
            ha='center', va='center', fontsize=14, fontweight='bold',
            color='white', zorder=11)

ax3.set_xlabel('Average Speed (km/h)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Average Power (W)', fontsize=12, fontweight='bold')
ax3.set_title('Speed vs Power Consumption', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)

# ==================== 子图4: 簇大小分布 ====================
ax4 = plt.subplot(2, 3, 4)

counts_list = [cluster_counts[i] for i in range(4)]
bars = ax4.bar(range(4), counts_list, color=colors, alpha=0.8,
              edgecolor='black', linewidth=2)

ax4.set_xlabel('Cluster', fontsize=12, fontweight='bold')
ax4.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
ax4.set_title('Cluster Size Distribution', fontsize=14, fontweight='bold')
ax4.set_xticks(range(4))
ax4.set_xticklabels(['C0', 'C1', 'C2', 'C3'])
ax4.grid(True, alpha=0.3, axis='y')

for bar, count in zip(bars, counts_list):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(count):,}', ha='center', va='bottom',
            fontsize=11, fontweight='bold')

# ==================== 子图5: Max Speed对比 ====================
ax5 = plt.subplot(2, 3, 5)

max_speeds = df_stats['Max Speed'].values
bars = ax5.bar(range(4), max_speeds, color=colors, alpha=0.8,
              edgecolor='black', linewidth=2)

ax5.set_xlabel('Cluster', fontsize=12, fontweight='bold')
ax5.set_ylabel('Max Speed (km/h)', fontsize=12, fontweight='bold')
ax5.set_title('Maximum Speed by Cluster', fontsize=14, fontweight='bold')
ax5.set_xticks(range(4))
ax5.set_xticklabels(['C0', 'C1', 'C2', 'C3'])
ax5.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars, max_speeds):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.2f}', ha='center', va='bottom',
            fontsize=11, fontweight='bold')

# ==================== 子图6: Trip Length ====================
ax6 = plt.subplot(2, 3, 6)

trip_lengths = df_stats['Trip Length'].values
bars = ax6.bar(range(4), trip_lengths, color=colors, alpha=0.8,
              edgecolor='black', linewidth=2)

ax6.set_xlabel('Cluster', fontsize=12, fontweight='bold')
ax6.set_ylabel('Average Trip Length (points)', fontsize=12, fontweight='bold')
ax6.set_title('Trip Duration', fontsize=14, fontweight='bold')
ax6.set_xticks(range(4))
ax6.set_xticklabels(['C0', 'C1', 'C2', 'C3'])
ax6.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars, trip_lengths):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1f}', ha='center', va='bottom',
            fontsize=11, fontweight='bold')

# ==================== 总标题 ====================
fig.suptitle('Comprehensive Cluster Analysis (Cross-Attention, K=4)', 
            fontsize=18, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.99])

output = './results/comprehensive_analysis_k4_crossattn_simple.png'
plt.savefig(output, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✅ Saved: {output}")

# ==================== 生成文字报告 ====================
print("\n" + "="*70)
print("📊 Cluster Analysis Summary")
print("="*70)

for cluster_id in range(4):
    print(f"\n🔷 Cluster {cluster_id}:")
    print(f"   Sample count: {cluster_counts[cluster_id]:,} ({cluster_counts[cluster_id]/len(labels)*100:.1f}%)")
    print(f"   Avg Speed: {df_stats.loc[f'Cluster {cluster_id}', 'Avg Speed']:.2f} km/h")
    print(f"   Max Speed: {df_stats.loc[f'Cluster {cluster_id}', 'Max Speed']:.2f} km/h")
    print(f"   Avg Power: {df_stats.loc[f'Cluster {cluster_id}', 'Avg Power']:.2f} W")
    print(f"   Trip Length: {df_stats.loc[f'Cluster {cluster_id}', 'Trip Length']:.1f} points")
    
    # 简单解释
    avg_spd = df_stats.loc[f'Cluster {cluster_id}', 'Avg Speed']
    if avg_spd > 50:
        print(f"   💡 Interpretation: Highway/Fast driving")
    elif avg_spd < 25:
        print(f"   💡 Interpretation: Urban/Congested")
    else:
        print(f"   💡 Interpretation: Mixed conditions")

print("\n" + "="*70)
print("✅ Comprehensive Analysis Complete!")
print("="*70)
print(f"\n📁 Generated: {output}")
print("="*70)