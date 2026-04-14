"""
Step 11: Fix Speed vs Energy Consumption Plot
修复右上角图表，显示每个聚类的代表性均值
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import pickle
import os

# ============ 设置论文风格 ============
plt.style.use('default')
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

# 配色方案
colors_cluster = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 蓝、橙、绿、红

results_dir = './analysis_complete_vehicles/results/'
paper_fig_dir = os.path.join(results_dir, 'paper_figures_fixed')
os.makedirs(paper_fig_dir, exist_ok=True)

print("="*70)
print("📊 Fixing Speed vs Energy Consumption Plot")
print("="*70)

# ============ 1. 加载数据 ============
print("\n📂 Loading data...")

df = pd.read_csv(os.path.join(results_dir, 'segments_with_clusters_labeled.csv'))
embeddings = np.load(os.path.join(results_dir, 'segment_embeddings.npy'))

print(f"✅ Loaded {len(df):,} segments")

# ============ 2. 创建修复后的主图表（6 个子图） ============
print("\n📊 Figure: Fixed 6-Panel Cluster Analysis")

fig = plt.figure(figsize=(18, 10))

# ========== 子图 1: PCA 空间 ==========
ax1 = plt.subplot(2, 3, 1)

pca = PCA(n_components=2)
embeddings_pca = pca.fit_transform(embeddings)

for cluster in range(4):
    mask = df['cluster'].values == cluster
    ax1.scatter(embeddings_pca[mask, 0], embeddings_pca[mask, 1],
               alpha=0.6, s=15, color=colors_cluster[cluster],
               label=f'Cluster {cluster}', edgecolors='none')

ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=10, fontweight='bold')
ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=10, fontweight='bold')
ax1.set_title('Driving Style Clusters (PCA Space)', fontsize=11, fontweight='bold')
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, alpha=0.3)

# ========== 子图 2: 能效分布（箱线图） ==========
ax2 = plt.subplot(2, 3, 2)

# 计算能效（SOC drop rate %/min）
df['energy_rate'] = df['soc_drop'] / (df['duration_seconds'] / 60 + 0.1)

data_energy = [df[df['cluster'] == c]['energy_rate'].values for c in range(4)]

bp = ax2.boxplot(data_energy, patch_artist=True, widths=0.6, showfliers=False,
                 labels=[f'C{i}' for i in range(4)],
                 medianprops=dict(color='red', linewidth=2, linestyle='-'),
                 boxprops=dict(linewidth=1.5),
                 whiskerprops=dict(linewidth=1.5),
                 capprops=dict(linewidth=1.5))

for patch, color in zip(bp['boxes'], colors_cluster):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax2.set_ylabel('Energy Rate (%/min)', fontsize=10, fontweight='bold')
ax2.set_xlabel('Cluster', fontsize=10, fontweight='bold')
ax2.set_title('Energy Efficiency by Cluster', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# ========== 子图 3: 速度 vs 能效（只显示聚类均值） ============
ax3 = plt.subplot(2, 3, 3)

# 计算每个聚类的均值
cluster_centers = []
for cluster in range(4):
    cluster_data = df[df['cluster'] == cluster]
    center_speed = cluster_data['speed_mean'].mean()
    center_energy = cluster_data['energy_rate'].mean()
    cluster_centers.append((center_speed, center_energy))

# 绘制每个聚类的均值点（大的圆点）
for cluster in range(4):
    center_speed, center_energy = cluster_centers[cluster]
    
    # 绘制大的代表点
    ax3.scatter(center_speed, center_energy, 
               s=800, color=colors_cluster[cluster],
               edgecolors='black', linewidth=3, zorder=10,
               alpha=0.9)
    
    # 添加聚类标签
    ax3.text(center_speed, center_energy, f'C{cluster}',
            ha='center', va='center', fontsize=11, fontweight='bold', color='white',
            zorder=11)

ax3.set_xlabel('Average Speed (km/h)', fontsize=10, fontweight='bold')
ax3.set_ylabel('Energy Rate (%/min)', fontsize=10, fontweight='bold')
ax3.set_title('Speed vs Energy Consumption', fontsize=11, fontweight='bold')
ax3.grid(True, alpha=0.3)

# 设置合理的轴范围
ax3.set_xlim(0, max([c[0] for c in cluster_centers]) + 5)
ax3.set_ylim(0, max([c[1] for c in cluster_centers]) + 0.05)

# ========== 子图 4: 聚类大小分布 ==========
ax4 = plt.subplot(2, 3, 4)

cluster_counts = [len(df[df['cluster'] == c]) for c in range(4)]
bars = ax4.bar(range(4), cluster_counts, color=colors_cluster, alpha=0.8, 
               edgecolor='black', linewidth=1.5, width=0.6)

ax4.set_ylabel('Number of Samples', fontsize=10, fontweight='bold')
ax4.set_xlabel('Cluster', fontsize=10, fontweight='bold')
ax4.set_title('Cluster Size Distribution', fontsize=11, fontweight='bold')
ax4.set_xticks(range(4))
ax4.set_xticklabels([f'C{i}' for i in range(4)])
ax4.set_ylim(0, max(cluster_counts) * 1.15)
ax4.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar, count in zip(bars, cluster_counts):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{count:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# ========== 子图 5: 功率特征 ==========
ax5 = plt.subplot(2, 3, 5)

power_means = [df[df['cluster'] == c]['power_mean'].mean() for c in range(4)]
bars = ax5.bar(range(4), power_means, color=colors_cluster, alpha=0.8,
              edgecolor='black', linewidth=1.5, width=0.6)

ax5.set_ylabel('Average Power (W)', fontsize=10, fontweight='bold')
ax5.set_xlabel('Cluster', fontsize=10, fontweight='bold')
ax5.set_title('Power Characteristics', fontsize=11, fontweight='bold')
ax5.set_xticks(range(4))
ax5.set_xticklabels([f'C{i}' for i in range(4)])
ax5.set_ylim(0, max(power_means) * 1.15)
ax5.grid(True, alpha=0.3, axis='y')

for bar, power in zip(bars, power_means):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
            f'{power:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# ========== 子图 6: 持续时间 ==========
ax6 = plt.subplot(2, 3, 6)

duration_means = [df[df['cluster'] == c]['duration_seconds'].mean() / 60 for c in range(4)]  # 转换为分钟
bars = ax6.bar(range(4), duration_means, color=colors_cluster, alpha=0.8,
              edgecolor='black', linewidth=1.5, width=0.6)

ax6.set_ylabel('Average Duration (min)', fontsize=10, fontweight='bold')
ax6.set_xlabel('Cluster', fontsize=10, fontweight='bold')
ax6.set_title('Trip Duration', fontsize=11, fontweight='bold')
ax6.set_xticks(range(4))
ax6.set_xticklabels([f'C{i}' for i in range(4)])
ax6.set_ylim(0, max(duration_means) * 1.15)
ax6.grid(True, alpha=0.3, axis='y')

for bar, duration in zip(bars, duration_means):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
            f'{duration:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# 整体标题
fig.suptitle('EV Driving Behavior Clustering Analysis', 
             fontsize=14, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(paper_fig_dir, 'Figure_6Panel_Fixed.png'), 
            dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(paper_fig_dir, 'Figure_6Panel_Fixed.pdf'), 
            dpi=300, bbox_inches='tight')
print("✅ Saved: Figure_6Panel_Fixed.png/pdf")
plt.close()

# ============ 3. 生成详细的聚类统计表 ============
print("\n📊 Generating Detailed Statistics...")

print("\n" + "="*100)
print("CLUSTER CENTER VALUES (Representative Mean Values)")
print("="*100)

summary_data = []
for cluster in range(4):
    cluster_df = df[df['cluster'] == cluster]
    
    summary_data.append({
        'Cluster': f'C{cluster}',
        'Samples': len(cluster_df),
        'Avg Speed (km/h)': f'{cluster_df["speed_mean"].mean():.2f}',
        'Avg Power (W)': f'{cluster_df["power_mean"].mean():.2f}',
        'Avg Voltage (V)': f'{cluster_df["voltage_mean"].mean():.2f}',
        'Avg Current (A)': f'{cluster_df["current_mean"].mean():.2f}',
        'Avg SOC Drop (%)': f'{cluster_df["soc_drop"].mean():.2f}',
        'Avg Duration (min)': f'{cluster_df["duration_seconds"].mean()/60:.1f}',
        'Energy Rate (%/min)': f'{cluster_df["energy_rate"].mean():.4f}',
    })

df_summary = pd.DataFrame(summary_data)
print(df_summary.to_string(index=False))
print("="*100)

# 保存为 CSV
df_summary.to_csv(os.path.join(paper_fig_dir, 'Cluster_Center_Values.csv'), index=False)

# ============ 4. 绘制额外的对比图 ============
print("\n📊 Generating Speed-Energy Detailed Analysis...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Cluster Characteristics - Detailed Comparison', fontsize=14, fontweight='bold')

# 子图 1: 速度均值对比
ax = axes[0, 0]
speeds = [df[df['cluster'] == c]['speed_mean'].mean() for c in range(4)]
bars = ax.bar(range(4), speeds, color=colors_cluster, alpha=0.8, edgecolor='black', linewidth=2)
ax.set_ylabel('Average Speed (km/h)', fontsize=11, fontweight='bold')
ax.set_xlabel('Cluster', fontsize=11, fontweight='bold')
ax.set_title('Speed Characteristics', fontsize=12, fontweight='bold')
ax.set_xticks(range(4))
ax.set_xticklabels([f'C{i}' for i in range(4)])
ax.grid(True, alpha=0.3, axis='y')
for bar, speed in zip(bars, speeds):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{speed:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# 子图 2: 能效率均值对比
ax = axes[0, 1]
energies = [df[df['cluster'] == c]['energy_rate'].mean() for c in range(4)]
bars = ax.bar(range(4), energies, color=colors_cluster, alpha=0.8, edgecolor='black', linewidth=2)
ax.set_ylabel('Energy Rate (%/min)', fontsize=11, fontweight='bold')
ax.set_xlabel('Cluster', fontsize=11, fontweight='bold')
ax.set_title('Energy Efficiency Characteristics', fontsize=12, fontweight='bold')
ax.set_xticks(range(4))
ax.set_xticklabels([f'C{i}' for i in range(4)])
ax.grid(True, alpha=0.3, axis='y')
for bar, energy in zip(bars, energies):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{energy:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# 子图 3: 功率均值对比
ax = axes[1, 0]
powers = [df[df['cluster'] == c]['power_mean'].mean() for c in range(4)]
bars = ax.bar(range(4), powers, color=colors_cluster, alpha=0.8, edgecolor='black', linewidth=2)
ax.set_ylabel('Average Power (W)', fontsize=11, fontweight='bold')
ax.set_xlabel('Cluster', fontsize=11, fontweight='bold')
ax.set_title('Power Consumption Characteristics', fontsize=12, fontweight='bold')
ax.set_xticks(range(4))
ax.set_xticklabels([f'C{i}' for i in range(4)])
ax.grid(True, alpha=0.3, axis='y')
for bar, power in zip(bars, powers):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{power:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# 子图 4: 持续时间均值对比
ax = axes[1, 1]
durations = [df[df['cluster'] == c]['duration_seconds'].mean() / 60 for c in range(4)]
bars = ax.bar(range(4), durations, color=colors_cluster, alpha=0.8, edgecolor='black', linewidth=2)
ax.set_ylabel('Average Duration (min)', fontsize=11, fontweight='bold')
ax.set_xlabel('Cluster', fontsize=11, fontweight='bold')
ax.set_title('Trip Duration Characteristics', fontsize=12, fontweight='bold')
ax.set_xticks(range(4))
ax.set_xticklabels([f'C{i}' for i in range(4)])
ax.grid(True, alpha=0.3, axis='y')
for bar, duration in zip(bars, durations):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{duration:.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(paper_fig_dir, 'Figure_Cluster_Centers_Detailed.png'), 
            dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(paper_fig_dir, 'Figure_Cluster_Centers_Detailed.pdf'), 
            dpi=300, bbox_inches='tight')
print("✅ Saved: Figure_Cluster_Centers_Detailed.png/pdf")
plt.close()

# ============ 5. 生成聚类中心点对比图 ============
print("\n📊 Generating Cluster Centers Comparison...")

fig, ax = plt.subplots(figsize=(12, 8))

# 提取每个聚类的中心点
speeds_center = [df[df['cluster'] == c]['speed_mean'].mean() for c in range(4)]
energies_center = [df[df['cluster'] == c]['energy_rate'].mean() for c in range(4)]

# 绘制大的中心点
for cluster in range(4):
    ax.scatter(speeds_center[cluster], energies_center[cluster],
              s=1000, color=colors_cluster[cluster],
              edgecolors='black', linewidth=3, zorder=10,
              alpha=0.85, label=f'C{cluster}')
    
    # 添加标签
    ax.text(speeds_center[cluster], energies_center[cluster],
           f'C{cluster}\n({speeds_center[cluster]:.2f} km/h\n{energies_center[cluster]:.4f} %/min)',
           ha='center', va='center', fontsize=10, fontweight='bold', color='white',
           zorder=11)

ax.set_xlabel('Average Speed (km/h)', fontsize=12, fontweight='bold')
ax.set_ylabel('Energy Rate (%/min)', fontsize=12, fontweight='bold')
ax.set_title('Cluster Representative Centers\n(Speed vs Energy Efficiency)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11, loc='upper right', framealpha=0.95)

# 设置轴范围
ax.set_xlim(-5, max(speeds_center) + 10)
ax.set_ylim(-0.01, max(energies_center) + 0.03)

plt.tight_layout()
plt.savefig(os.path.join(paper_fig_dir, 'Figure_Cluster_Centers_Focus.png'), 
            dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(paper_fig_dir, 'Figure_Cluster_Centers_Focus.pdf'), 
            dpi=300, bbox_inches='tight')
print("✅ Saved: Figure_Cluster_Centers_Focus.png/pdf")
plt.close()

# ============ 6. 打印详细对比 ============
print("\n" + "="*100)
print("CLUSTER COMPARISON SUMMARY")
print("="*100)

print("\n【速度特征】")
for cluster in range(4):
    cluster_df = df[df['cluster'] == cluster]
    print(f"  C{cluster}: {cluster_df['speed_mean'].mean():.2f} km/h "
          f"(range: {cluster_df['speed_mean'].min():.2f} - {cluster_df['speed_mean'].max():.2f})")

print("\n【能效特征】")
for cluster in range(4):
    cluster_df = df[df['cluster'] == cluster]
    print(f"  C{cluster}: {cluster_df['energy_rate'].mean():.4f} %/min "
          f"(range: {cluster_df['energy_rate'].min():.4f} - {cluster_df['energy_rate'].max():.4f})")

print("\n【功率特征】")
for cluster in range(4):
    cluster_df = df[df['cluster'] == cluster]
    print(f"  C{cluster}: {cluster_df['power_mean'].mean():.1f} W "
          f"(range: {cluster_df['power_mean'].min():.1f} - {cluster_df['power_mean'].max():.1f})")

print("\n【时长特征】")
for cluster in range(4):
    cluster_df = df[df['cluster'] == cluster]
    print(f"  C{cluster}: {cluster_df['duration_seconds'].mean()/60:.1f} min "
          f"(range: {cluster_df['duration_seconds'].min()/60:.1f} - {cluster_df['duration_seconds'].max()/60:.1f})")

print("\n" + "="*100)

# ============ 最终总结 ============
print(f"\n{'='*70}")
print(f"✅ Fixed Speed vs Energy Plot Complete!")
print(f"{'='*70}")
print(f"\n📁 Output Directory: {paper_fig_dir}")
print(f"\n📊 Generated Figures:")
print(f"   ✓ Figure_6Panel_Fixed.png/pdf (MAIN - Fixed version)")
print(f"   ✓ Figure_Cluster_Centers_Detailed.png/pdf (Detailed metrics)")
print(f"   ✓ Figure_Cluster_Centers_Focus.png/pdf (Centers focus)")
print(f"   ✓ Cluster_Center_Values.csv (Data table)")
print(f"\n✨ Ready for Publication!")
print(f"{'='*70}\n")