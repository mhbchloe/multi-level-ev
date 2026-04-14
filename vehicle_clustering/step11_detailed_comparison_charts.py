"""
Step 11: Detailed Clustering Comparison Charts
为每种聚类算法生成详细的K=3聚类分析图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['font.size'] = 10

print("="*70)
print("📊 Generating Detailed Clustering Comparison Charts")
print("="*70)

results_dir = "./vehicle_clustering/results/"

# 加载数据
df_features = pd.read_csv(os.path.join(results_dir, 'vehicle_advanced_features_v3.csv'))
print(f"✅ Vehicles: {len(df_features):,}")

# 特征标准化
feature_cols = [col for col in df_features.columns 
                if col not in ['vehicle_id', 'n_segments', 'n_events', 'n_charging']]

X = df_features[feature_cols].copy()
X = X.fillna(X.median())

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"✅ Features standardized and PCA computed")

# ============ 定义聚类方法 ============
algorithms = {
    'K-means': KMeans(n_clusters=3, n_init=100, random_state=42, max_iter=500),
    'GMM': GaussianMixture(n_components=3, n_init=150, random_state=42, max_iter=500),
    'Hierarchical': AgglomerativeClustering(n_clusters=3, linkage='ward'),
    'Spectral': SpectralClustering(n_clusters=3, affinity='rbf', random_state=42, n_init=100),
}

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
seg_colors = ['#FF9999', '#99CCFF', '#FF6B6B', '#66B2FF']  # Segment cluster colors
seg_labels = ['Moderate', 'Conservative', 'Aggressive', 'Highway']

# ============ 为每种算法生成详细图表 ============
for algo_name, algorithm in algorithms.items():
    print(f"\n{'='*70}")
    print(f"🎯 Generating detailed chart for {algo_name}")
    print(f"{'='*70}")
    
    # 执行聚类
    labels = algorithm.fit_predict(X_scaled)
    
    # 计算指标
    sil = silhouette_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)
    ch = calinski_harabasz_score(X_scaled, labels)
    
    # 创建子集数据
    df_cluster = df_features.copy()
    df_cluster['cluster'] = labels
    
    print(f"Silhouette: {sil:.4f}")
    print(f"Davies-Bouldin: {db:.4f}")
    print(f"Calinski-Harabasz: {ch:.1f}")
    
    # ========== 创建详细图表 ==========
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.35)
    
    unique_clusters = sorted(set(labels))
    n_clusters = len(unique_clusters)
    
    # 1. 车辆类型分布
    ax1 = fig.add_subplot(gs[0, 0])
    cluster_counts = [sum(labels == i) for i in unique_clusters]
    bars = ax1.bar(range(n_clusters), cluster_counts, color=colors[:n_clusters], 
                   alpha=0.85, edgecolor='black', linewidth=2)
    
    for bar, count in zip(bars, cluster_counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count):,}\n({count/len(labels)*100:.1f}%)',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax1.set_ylabel('Number of Vehicles', fontweight='bold')
    ax1.set_title('Vehicle Type Distribution', fontweight='bold', fontsize=12)
    ax1.set_xticks(range(n_clusters))
    ax1.set_xticklabels([f'Type {i}' for i in unique_clusters])
    ax1.grid(alpha=0.3, axis='y')
    ax1.set_ylim(0, max(cluster_counts) * 1.15)
    
    # 2. Segment群构成
    ax2 = fig.add_subplot(gs[0, 1])
    
    seg_ratios = []
    for cluster in unique_clusters:
        cluster_data = df_cluster[df_cluster['cluster'] == cluster]
        ratios = [
            cluster_data['cluster_0_ratio'].mean(),
            cluster_data['cluster_1_ratio'].mean(),
            cluster_data['cluster_2_ratio'].mean(),
            cluster_data['cluster_3_ratio'].mean()
        ]
        seg_ratios.append(ratios)
    
    bottom = np.zeros(n_clusters)
    for seg_idx, seg_color in enumerate(seg_colors):
        values = [seg_ratios[c][seg_idx] for c in range(n_clusters)]
        ax2.bar(range(n_clusters), values, bottom=bottom, color=seg_color, alpha=0.8,
               label=seg_labels[seg_idx], edgecolor='black', linewidth=0.5)
        bottom += values
    
    ax2.set_ylabel('Proportion', fontweight='bold')
    ax2.set_title('Segment Cluster Composition', fontweight='bold', fontsize=12)
    ax2.set_xticks(range(n_clusters))
    ax2.set_xticklabels([f'Type {i}' for i in unique_clusters])
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(alpha=0.3, axis='y')
    
    # 3. 能耗对比
    ax3 = fig.add_subplot(gs[0, 2])
    power_data = [df_cluster[df_cluster['cluster'] == i]['weighted_power'].values 
                  for i in unique_clusters]
    bp = ax3.boxplot(power_data, patch_artist=True, 
                     labels=[f'Type {i}' for i in unique_clusters],
                     widths=0.6, showfliers=True)
    
    for patch, color in zip(bp['boxes'], colors[:n_clusters]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax3.set_ylabel('Weighted Power (kW)', fontweight='bold')
    ax3.set_title('Energy Consumption', fontweight='bold', fontsize=12)
    ax3.grid(alpha=0.3, axis='y')
    
    # 4. Moderate Urban比例
    ax4 = fig.add_subplot(gs[1, 0])
    data = [df_cluster[df_cluster['cluster'] == i]['cluster_0_ratio'].values 
            for i in unique_clusters]
    bp = ax4.boxplot(data, patch_artist=True, 
                     labels=[f'Type {i}' for i in unique_clusters],
                     widths=0.6, showfliers=False)
    
    for patch, color in zip(bp['boxes'], colors[:n_clusters]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax4.set_ylabel('Proportion', fontweight='bold')
    ax4.set_title('Moderate Urban Driving', fontweight='bold', fontsize=11)
    ax4.grid(alpha=0.3, axis='y')
    
    # 5. Aggressive Urban比例
    ax5 = fig.add_subplot(gs[1, 1])
    data = [df_cluster[df_cluster['cluster'] == i]['cluster_2_ratio'].values 
            for i in unique_clusters]
    bp = ax5.boxplot(data, patch_artist=True, 
                     labels=[f'Type {i}' for i in unique_clusters],
                     widths=0.6, showfliers=False)
    
    for patch, color in zip(bp['boxes'], colors[:n_clusters]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax5.set_ylabel('Proportion', fontweight='bold')
    ax5.set_title('Aggressive Urban Driving', fontweight='bold', fontsize=11)
    ax5.grid(alpha=0.3, axis='y')
    
    # 6. Low SOC风险
    ax6 = fig.add_subplot(gs[1, 2])
    data = [df_cluster[df_cluster['cluster'] == i]['low_soc_ratio'].values 
            for i in unique_clusters]
    bp = ax6.boxplot(data, patch_artist=True, 
                     labels=[f'Type {i}' for i in unique_clusters],
                     widths=0.6, showfliers=False)
    
    for patch, color in zip(bp['boxes'], colors[:n_clusters]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax6.set_ylabel('Low SOC Risk Ratio', fontweight='bold')
    ax6.set_title('Low SOC Risk', fontweight='bold', fontsize=11)
    ax6.grid(alpha=0.3, axis='y')
    
    # 7. 雷达图
    ax7 = fig.add_subplot(gs[2, 0:2], projection='polar')
    
    radar_features = ['cluster_2_ratio', 'weighted_power', 'low_soc_ratio',
                     'charging_urgency', 'usage_freq', 'charging_freq']
    radar_labels = ['Aggressive\nDriving', 'Energy\nConsumption', 'Low SOC\nRisk',
                   'Charging\nUrgency', 'Usage\nFrequency', 'Charging\nFrequency']
    
    cluster_radar_data = []
    for cluster in unique_clusters:
        cluster_data = df_cluster[df_cluster['cluster'] == cluster]
        values = [
            cluster_data['cluster_2_ratio'].mean(),
            cluster_data['weighted_power'].mean(),
            cluster_data['low_soc_ratio'].mean(),
            cluster_data['charging_urgency'].mean(),
            cluster_data['usage_freq'].mean(),
            cluster_data['charging_freq'].mean()
        ]
        cluster_radar_data.append(values)
    
    cluster_radar_norm = (np.array(cluster_radar_data) - np.array(cluster_radar_data).min(axis=0)) / \
                        (np.array(cluster_radar_data).max(axis=0) - np.array(cluster_radar_data).min(axis=0) + 1e-10)
    
    angles = np.linspace(0, 2*np.pi, len(radar_labels), endpoint=False).tolist()
    angles += angles[:1]
    
    for cluster, color in zip(unique_clusters, colors[:n_clusters]):
        idx = unique_clusters.index(cluster)
        values = cluster_radar_norm[idx].tolist()
        values += values[:1]
        ax7.plot(angles, values, 'o-', linewidth=2.5, label=f'Type {cluster}',
                color=color, markersize=8)
        ax7.fill(angles, values, alpha=0.15, color=color)
    
    ax7.set_xticks(angles[:-1])
    ax7.set_xticklabels(radar_labels, fontsize=9, fontweight='bold')
    ax7.set_ylim(0, 1)
    ax7.set_title('Multi-Dimensional Profile Comparison', fontsize=12, fontweight='bold', pad=20)
    ax7.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax7.grid(True)
    
    # 8. PCA投影
    ax8 = fig.add_subplot(gs[2, 2])
    
    for cluster, color in zip(unique_clusters, colors[:n_clusters]):
        mask = labels == cluster
        ax8.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=color, label=f'Type {cluster}',
                   alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    ax8.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontweight='bold')
    ax8.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontweight='bold')
    ax8.set_title('Vehicle Clustering (PCA Projection)', fontweight='bold', fontsize=12)
    ax8.legend(fontsize=10, loc='best')
    ax8.grid(alpha=0.3)
    
    # 9. 聚类结果信息框
    ax9 = fig.add_subplot(gs[0, 2])
    ax9.axis('off')
    
    metrics_text = f"""CLUSTERING RESULTS

K: {n_clusters} clusters
Silhouette: {sil:.4f}
Davies-Bouldin: {db:.4f}
Calinski-Harabasz: {ch:.1f}

Total Vehicles: {len(labels):,}
Total Features: {len(feature_cols)}

Vehicle Types:
"""
    
    for cluster in unique_clusters:
        count = sum(labels == cluster)
        metrics_text += f"\n  Type {cluster}: {count:,}"
    
    ax9.text(0.05, 0.95, metrics_text, transform=ax9.transAxes,
            fontfamily='monospace', fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle(f'Vehicle Clustering Analysis (K={n_clusters}) - {algo_name}', 
                fontsize=16, fontweight='bold', y=0.995)
    
    # 保存图表
    filename = f'clustering_detailed_{algo_name.lower().replace("-", "_")}.png'
    plt.savefig(os.path.join(results_dir, filename), dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {filename}")
    
    plt.close()

# ============ DBSCAN特殊处理 ============
print(f"\n{'='*70}")
print(f"🎯 Generating detailed chart for DBSCAN")
print(f"{'='*70}")

from sklearn.neighbors import NearestNeighbors

k = 5
neighbors = NearestNeighbors(n_neighbors=k)
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)
distances = np.sort(distances[:, k-1], axis=0)
eps = distances[int(len(distances) * 0.9)]

dbscan = DBSCAN(eps=eps, min_samples=10)
labels = dbscan.fit_predict(X_scaled)

sil = silhouette_score(X_scaled[labels != -1], labels[labels != -1]) if len(set(labels[labels != -1])) > 1 else 0
db = davies_bouldin_score(X_scaled[labels != -1], labels[labels != -1]) if len(set(labels[labels != -1])) > 1 else 0
ch = calinski_harabasz_score(X_scaled[labels != -1], labels[labels != -1]) if len(set(labels[labels != -1])) > 1 else 0

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

df_cluster = df_features.copy()
df_cluster['cluster'] = labels

print(f"Silhouette: {sil:.4f}")
print(f"Davies-Bouldin: {db:.4f}")
print(f"Calinski-Harabasz: {ch:.1f}")
print(f"Noise Points: {n_noise}")

# 创建DBSCAN图表
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.35)

unique_clusters = sorted(set(labels[labels != -1]))
n_real_clusters = len(unique_clusters)

# 1. 车辆类型分布
ax1 = fig.add_subplot(gs[0, 0])
cluster_counts = [sum(labels == i) for i in unique_clusters]
noise_count = sum(labels == -1)

all_counts = cluster_counts + ([noise_count] if noise_count > 0 else [])
all_labels = [f'Cluster {i}' for i in unique_clusters] + (['Noise'] if noise_count > 0 else [])
colors_dbscan = colors[:n_real_clusters] + (['gray'] if noise_count > 0 else [])

bars = ax1.bar(range(len(all_counts)), all_counts, color=colors_dbscan, 
              alpha=0.85, edgecolor='black', linewidth=2)

for bar, count in zip(bars, all_counts):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(count):,}\n({count/len(labels)*100:.1f}%)',
            ha='center', va='bottom', fontweight='bold', fontsize=10)

ax1.set_ylabel('Number of Vehicles', fontweight='bold')
ax1.set_title('Vehicle Type Distribution (with Noise)', fontweight='bold', fontsize=12)
ax1.set_xticks(range(len(all_labels)))
ax1.set_xticklabels(all_labels, rotation=45, ha='right')
ax1.grid(alpha=0.3, axis='y')
ax1.set_ylim(0, max(all_counts) * 1.15)

# 2-6. 其他图表（跳过对噪声点的详细分析）
ax2 = fig.add_subplot(gs[0, 1])
ax2.text(0.5, 0.5, f"DBSCAN found:\n{n_real_clusters} clusters\n{n_noise:,} noise points\n\nNote: Noise points\nexcluded from\ndetailed analysis",
        ha='center', va='center', fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
        transform=ax2.transAxes)
ax2.axis('off')

ax3 = fig.add_subplot(gs[0, 2])
ax3.axis('off')
metrics_text = f"""CLUSTERING RESULTS

Algorithm: DBSCAN
eps: {eps:.4f}
min_samples: 10

Clusters: {n_real_clusters}
Noise Points: {n_noise:,}

Silhouette: {sil:.4f}
Davies-Bouldin: {db:.4f}
Calinski-Harabasz: {ch:.1f}

Total Vehicles: {len(labels):,}
"""

ax3.text(0.05, 0.95, metrics_text, transform=ax3.transAxes,
        fontfamily='monospace', fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# PCA投影
ax_pca = fig.add_subplot(gs[1:, :2])

for cluster in unique_clusters:
    mask = labels == cluster
    ax_pca.scatter(X_pca[mask, 0], X_pca[mask, 1],
                  c=colors[cluster % len(colors)], label=f'Cluster {cluster}',
                  alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

if noise_count > 0:
    noise_mask = labels == -1
    ax_pca.scatter(X_pca[noise_mask, 0], X_pca[noise_mask, 1],
                  marker='X', c='gray', alpha=0.3, s=100, label='Noise')

ax_pca.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontweight='bold')
ax_pca.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontweight='bold')
ax_pca.set_title('Vehicle Clustering (PCA Projection)', fontweight='bold', fontsize=12)
ax_pca.legend(fontsize=10)
ax_pca.grid(alpha=0.3)

# 警告信息
ax_warning = fig.add_subplot(gs[1:, 2])
ax_warning.axis('off')
warning_text = """⚠️ DBSCAN Assessment

Strengths:
✅ Automatic outlier
   detection
✅ No K specification

Weaknesses:
❌ Many noise points
❌ Difficult parameter
   tuning
❌ Not suitable for
   this dataset

Recommendation:
🔴 Not recommended
   for vehicle clustering
"""

ax_warning.text(0.05, 0.95, warning_text, transform=ax_warning.transAxes,
               fontfamily='monospace', fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='#FFE0E0', alpha=0.8))

plt.suptitle(f'Vehicle Clustering Analysis - DBSCAN', 
            fontsize=16, fontweight='bold', y=0.995)

plt.savefig(os.path.join(results_dir, 'clustering_detailed_dbscan.png'), dpi=300, bbox_inches='tight')
print(f"✅ Saved: clustering_detailed_dbscan.png")

plt.close('all')

print(f"\n{'='*70}")
print(f"✅ Step 11 Complete!")
print(f"{'='*70}")
print(f"\n📁 Generated Files:")
print(f"   ✅ clustering_detailed_k_means.png")
print(f"   ✅ clustering_detailed_gmm.png")
print(f"   ✅ clustering_detailed_hierarchical.png")
print(f"   ✅ clustering_detailed_spectral.png")
print(f"   ✅ clustering_detailed_dbscan.png")
print(f"\n🎉 Detailed comparison charts complete!")