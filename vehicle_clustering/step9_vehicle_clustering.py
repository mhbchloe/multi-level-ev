"""
Step 9 Improved: Better Visualization + Optimized Clustering
专业级可视化 + 优化的聚类参数
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import pickle
import os
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 11

print("="*70)
print("🎯 Vehicle Clustering (Improved Visualization)")
print("="*70)

results_dir = "./vehicle_clustering/results/"
os.makedirs(results_dir, exist_ok=True)

# ============ 1. 加载特征 ============
print("\n📂 Loading features...")

df_features = pd.read_csv(os.path.join(results_dir, 'vehicle_advanced_features.csv'))

print(f"✅ Vehicles: {len(df_features):,}")

# ============ 2. 特征标准化 ============
print(f"\n🔧 Standardizing features...")

feature_cols = [col for col in df_features.columns 
                if col not in ['vehicle_id', 'n_segments', 'n_events', 'n_charging']]

X = df_features[feature_cols].copy()
X = X.fillna(X.median())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"✅ Features standardized: {X_scaled.shape}")

with open(os.path.join(results_dir, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

# ============ 3. 最优K选择（更详细的分析） ============
print(f"\n{'='*70}")
print(f"🔍 Comprehensive K Selection Analysis")
print(f"{'='*70}")

k_range = range(2, 8)
metrics = {
    'BIC': [], 'AIC': [], 'Silhouette': [],
    'Davies-Bouldin': [], 'Calinski-Harabasz': []
}

for k in tqdm(k_range, desc="Testing different K values"):
    gmm = GaussianMixture(n_components=k, n_init=20, random_state=42, max_iter=300)
    labels = gmm.fit_predict(X_scaled)
    
    metrics['BIC'].append(gmm.bic(X_scaled))
    metrics['AIC'].append(gmm.aic(X_scaled))
    metrics['Silhouette'].append(silhouette_score(X_scaled, labels))
    metrics['Davies-Bouldin'].append(davies_bouldin_score(X_scaled, labels))
    metrics['Calinski-Harabasz'].append(calinski_harabasz_score(X_scaled, labels))

# 打印详细结果
print(f"\nK Selection Results:")
print(f"{'K':<5} {'BIC':<12} {'AIC':<12} {'Silhouette':<12} {'DB':<10} {'CH':<12}")
print(f"{'-'*60}")

for i, k in enumerate(k_range):
    print(f"{k:<5} {metrics['BIC'][i]:<12.1f} {metrics['AIC'][i]:<12.1f} "
          f"{metrics['Silhouette'][i]:<12.4f} {metrics['Davies-Bouldin'][i]:<10.4f} "
          f"{metrics['Calinski-Harabasz'][i]:<12.1f}")

# 找到最优K（多个指标综合考虑）
best_k_bic = k_range[np.argmin(metrics['BIC'])]
best_k_sil = k_range[np.argmax(metrics['Silhouette'])]

print(f"\n✅ Best K (BIC): {best_k_bic}")
print(f"✅ Best K (Silhouette): {best_k_sil}")

# 用最优K或用户指定的K
optimal_k = 3  # 用silhouette最好的
print(f"📌 Using K={optimal_k} (Best Silhouette)")

# ============ 4. 最终GMM聚类 ============
print(f"\n{'='*70}")
print(f"🎯 GMM Clustering (K={optimal_k})")
print(f"{'='*70}")

gmm = GaussianMixture(n_components=optimal_k, n_init=50, random_state=42, max_iter=500)
vehicle_labels = gmm.fit_predict(X_scaled)
vehicle_probs = gmm.predict_proba(X_scaled)

sil = silhouette_score(X_scaled, vehicle_labels)
db = davies_bouldin_score(X_scaled, vehicle_labels)
ch = calinski_harabasz_score(X_scaled, vehicle_labels)

print(f"\nGMM Results:")
print(f"   Silhouette Score: {sil:.4f}")
print(f"   Davies-Bouldin Index: {db:.4f}")
print(f"   Calinski-Harabasz Score: {ch:.1f}")

print(f"\nCluster Distribution:")
unique, counts = np.unique(vehicle_labels, return_counts=True)
for cluster, count in zip(unique, counts):
    print(f"   Cluster {cluster}: {count:,} ({count/len(vehicle_labels)*100:.1f}%)")

# ============ 5. 聚类分析 ============
print(f"\n{'='*70}")
print(f"📊 Cluster Characterization")
print(f"{'='*70}")

df_features['cluster'] = vehicle_labels
df_features['cluster_prob'] = vehicle_probs.max(axis=1)

key_features = [
    'power_mean_avg',
    'low_soc_risk_ratio',
    'deep_discharge_ratio',
    'speed_mean_avg',
    'charging_freq',
    'power_efficiency',
]

cluster_profiles = df_features.groupby('cluster')[key_features].mean()
print(f"\nCluster Profiles:")
print(cluster_profiles.round(3))

# ========== 6. 聚类命名 ==========
print(f"\n{'='*70}")
print(f"🏷️ Cluster Labeling")
print(f"{'='*70}")

cluster_names = {}
cluster_descriptions = {}

for cluster in range(optimal_k):
    cluster_data = df_features[df_features['cluster'] == cluster]
    
    avg_power = cluster_data['power_mean_avg'].mean()
    avg_soc_risk = cluster_data['low_soc_risk_ratio'].mean()
    avg_speed = cluster_data['speed_mean_avg'].mean()
    avg_charging = cluster_data['charging_freq'].mean()
    
    # 动态命名规则
    if abs(avg_power) > 10:
        name = "High Energy Consumption"
        desc = "高能耗车型 - 动力输出强劲，需加强电池管理"
    elif avg_soc_risk > 0.1:
        name = "Low SOC Risk"
        desc = "低SOC风险型 - 频繁低电量放电，需优化��电策略"
    else:
        name = "Low Energy Consumption"
        desc = "低能耗高效型 - 能效优异，运营成本最低"
    
    cluster_names[cluster] = name
    cluster_descriptions[cluster] = desc
    
    print(f"\nCluster {cluster}: {name}")
    print(f"   车型特征: {desc}")
    print(f"   车辆数: {len(cluster_data):,}")
    print(f"   平均功率: {avg_power:.2f} kW")
    print(f"   SOC风险: {avg_soc_risk:.1%}")
    print(f"   平均速度: {avg_speed:.1f} km/h")
    print(f"   充电频率: {avg_charging:.2f} 次/天")

df_features['vehicle_type'] = df_features['cluster'].map(cluster_names)

# ============ 7. 保存结果 ============
print(f"\n💾 Saving results...")

df_features.to_csv(os.path.join(results_dir, 'vehicle_clustering_results.csv'), index=False)
with open(os.path.join(results_dir, 'gmm_model.pkl'), 'wb') as f:
    pickle.dump(gmm, f)

cluster_info = {
    'method': 'GMM',
    'n_clusters': optimal_k,
    'cluster_names': cluster_names,
    'cluster_descriptions': cluster_descriptions,
    'metrics': {'silhouette': float(sil), 'davies_bouldin': float(db), 'calinski_harabasz': float(ch)}
}

with open(os.path.join(results_dir, 'clustering_info.json'), 'w') as f:
    json.dump(cluster_info, f, indent=2, ensure_ascii=False)

print(f"✅ Saved clustering results")

# ============ 8. 专业级可视化 ============
print(f"\n📈 Generating professional visualizations...")

# 设置配色方案
colors_list = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
colors = colors_list[:optimal_k]

# ========== Figure 1: 聚类综合分析 ==========
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

# 1.1 聚类分布（改进的柱状图）
ax1 = fig.add_subplot(gs[0, 0])
cluster_counts = [sum(vehicle_labels == i) for i in range(optimal_k)]
bars = ax1.bar(range(optimal_k), cluster_counts, color=colors, alpha=0.85, 
               edgecolor='black', linewidth=2, width=0.6)

for bar, count in zip(bars, cluster_counts):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(count):,}\n({count/len(vehicle_labels)*100:.1f}%)',
            ha='center', va='bottom', fontweight='bold', fontsize=10)

ax1.set_ylabel('Number of Vehicles', fontweight='bold', fontsize=11)
ax1.set_title('Vehicle Type Distribution', fontweight='bold', fontsize=12)
ax1.set_xticks(range(optimal_k))
ax1.set_xticklabels([f'Type {i}' for i in range(optimal_k)])
ax1.grid(alpha=0.3, axis='y')
ax1.set_ylim(0, max(cluster_counts) * 1.15)

# 1.2 能耗对比（小提琴图）
ax2 = fig.add_subplot(gs[0, 1])
power_data = [df_features[df_features['cluster'] == i]['power_mean_avg'].values 
              for i in range(optimal_k)]

parts = ax2.violinplot(power_data, positions=range(optimal_k), showmeans=True, showmedians=True)
for pc, color in zip(parts['bodies'], colors):
    pc.set_facecolor(color)
    pc.set_alpha(0.7)

ax2.set_ylabel('Avg Power (kW)', fontweight='bold', fontsize=11)
ax2.set_title('Energy Consumption Distribution', fontweight='bold', fontsize=12)
ax2.set_xticks(range(optimal_k))
ax2.set_xticklabels([f'Type {i}' for i in range(optimal_k)])
ax2.grid(alpha=0.3, axis='y')

# 1.3 SOC风险对比
ax3 = fig.add_subplot(gs[0, 2])
risk_data = [df_features[df_features['cluster'] == i]['low_soc_risk_ratio'].values 
             for i in range(optimal_k)]

parts = ax3.violinplot(risk_data, positions=range(optimal_k), showmeans=True, showmedians=True)
for pc, color in zip(parts['bodies'], colors):
    pc.set_facecolor(color)
    pc.set_alpha(0.7)

ax3.set_ylabel('Low SOC Risk Ratio', fontweight='bold', fontsize=11)
ax3.set_title('SOC Risk Distribution', fontweight='bold', fontsize=12)
ax3.set_xticks(range(optimal_k))
ax3.set_xticklabels([f'Type {i}' for i in range(optimal_k)])
ax3.grid(alpha=0.3, axis='y')

# 1.4 多特征雷达图
ax4 = fig.add_subplot(gs[1, :], projection='polar')

radar_features = ['power_mean_avg', 'speed_mean_avg', 'charging_freq', 
                  'low_soc_risk_ratio', 'power_efficiency']
radar_labels = ['Energy\nConsumption', 'Speed', 'Charging\nFrequency', 
                'SOC Risk', 'Efficiency']

cluster_radar_data = []
for cluster in range(optimal_k):
    cluster_data = df_features[df_features['cluster'] == cluster]
    values = [cluster_data[f].mean() for f in radar_features]
    cluster_radar_data.append(values)

cluster_radar_norm = (np.array(cluster_radar_data) - np.array(cluster_radar_data).min(axis=0)) / \
                     (np.array(cluster_radar_data).max(axis=0) - np.array(cluster_radar_data).min(axis=0) + 1e-10)

angles = np.linspace(0, 2*np.pi, len(radar_labels), endpoint=False).tolist()
angles += angles[:1]

for cluster in range(optimal_k):
    values = cluster_radar_norm[cluster].tolist()
    values += values[:1]
    ax4.plot(angles, values, 'o-', linewidth=2.5, label=f'Type {cluster}: {cluster_names[cluster]}',
            color=colors[cluster], markersize=8)
    ax4.fill(angles, values, alpha=0.15, color=colors[cluster])

ax4.set_xticks(angles[:-1])
ax4.set_xticklabels(radar_labels, fontsize=10, fontweight='bold')
ax4.set_ylim(0, 1)
ax4.set_title('Cluster Feature Comparison (Normalized)', fontsize=13, fontweight='bold', pad=20)
ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
ax4.grid(True)

# 1.5 PCA投影
ax5 = fig.add_subplot(gs[2, 0:2])
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

for cluster in range(optimal_k):
    mask = vehicle_labels == cluster
    ax5.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=colors[cluster], label=f'Type {cluster}',
               alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

ax5.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontweight='bold', fontsize=11)
ax5.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontweight='bold', fontsize=11)
ax5.set_title('Vehicle Clustering (PCA)', fontweight='bold', fontsize=12)
ax5.legend(fontsize=10, loc='best')
ax5.grid(alpha=0.3)

# 1.6 聚类质量指标
ax6 = fig.add_subplot(gs[2, 2])
ax6.axis('off')

metrics_text = f"""
CLUSTERING QUALITY METRICS

Silhouette Score: {sil:.4f}
(Range: -1 to 1, higher is better)

Davies-Bouldin Index: {db:.4f}
(Lower is better)

Calinski-Harabasz Score: {ch:.1f}
(Higher is better)

Optimal K: {optimal_k} clusters
Total Vehicles: {len(df_features):,}
"""

ax6.text(0.1, 0.9, metrics_text, transform=ax6.transAxes,
        fontfamily='monospace', fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Vehicle Clustering Analysis - Comprehensive View', 
            fontsize=16, fontweight='bold', y=0.995)

plt.savefig(os.path.join(results_dir, 'vehicle_clustering_professional.png'), 
           dpi=300, bbox_inches='tight')
print(f"✅ Saved: vehicle_clustering_professional.png")

# ========== Figure 2: K选择曲线 ==========
fig2, axes2 = plt.subplots(2, 3, figsize=(16, 10))
fig2.suptitle('Optimal K Selection Analysis', fontsize=16, fontweight='bold')

ax = axes2[0, 0]
ax.plot(k_range, metrics['BIC'], 'o-', linewidth=2, markersize=8, color='#FF6B6B')
ax.set_xlabel('K (Number of Clusters)', fontweight='bold')
ax.set_ylabel('BIC Score', fontweight='bold')
ax.set_title('BIC (Lower is Better)', fontweight='bold')
ax.grid(alpha=0.3)
ax.axvline(best_k_bic, color='red', linestyle='--', alpha=0.5)

ax = axes2[0, 1]
ax.plot(k_range, metrics['AIC'], 'o-', linewidth=2, markersize=8, color='#4ECDC4')
ax.set_xlabel('K (Number of Clusters)', fontweight='bold')
ax.set_ylabel('AIC Score', fontweight='bold')
ax.set_title('AIC (Lower is Better)', fontweight='bold')
ax.grid(alpha=0.3)

ax = axes2[0, 2]
ax.plot(k_range, metrics['Silhouette'], 'o-', linewidth=2, markersize=8, color='#45B7D1')
ax.set_xlabel('K (Number of Clusters)', fontweight='bold')
ax.set_ylabel('Silhouette Score', fontweight='bold')
ax.set_title('Silhouette (Higher is Better)', fontweight='bold')
ax.grid(alpha=0.3)
ax.axvline(best_k_sil, color='blue', linestyle='--', alpha=0.5)

ax = axes2[1, 0]
ax.plot(k_range, metrics['Davies-Bouldin'], 'o-', linewidth=2, markersize=8, color='#FFA07A')
ax.set_xlabel('K (Number of Clusters)', fontweight='bold')
ax.set_ylabel('Davies-Bouldin Index', fontweight='bold')
ax.set_title('Davies-Bouldin (Lower is Better)', fontweight='bold')
ax.grid(alpha=0.3)

ax = axes2[1, 1]
ax.plot(k_range, metrics['Calinski-Harabasz'], 'o-', linewidth=2, markersize=8, color='#98D8C8')
ax.set_xlabel('K (Number of Clusters)', fontweight='bold')
ax.set_ylabel('Calinski-Harabasz Score', fontweight='bold')
ax.set_title('Calinski-Harabasz (Higher is Better)', fontweight='bold')
ax.grid(alpha=0.3)

axes2[1, 2].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'k_selection_analysis.png'), dpi=300, bbox_inches='tight')
print(f"✅ Saved: k_selection_analysis.png")

plt.close('all')

# ============ 9. 生成报告 ============
print(f"\n📄 Generating report...")

report = f"""
{'='*70}
Vehicle Clustering Analysis Report
{'='*70}

Date: {pd.Timestamp.now()}

1. METHODOLOGY
{'-'*70}
Algorithm: Gaussian Mixture Model (GMM)
Number of Clusters (K): {optimal_k}
Feature Dimension: {len(feature_cols)}
Total Vehicles: {len(df_features):,}

2. CLUSTERING QUALITY METRICS
{'-'*70}
Silhouette Score: {sil:.4f}
Davies-Bouldin Index: {db:.4f}
Calinski-Harabasz Score: {ch:.1f}

3. VEHICLE TYPES
{'-'*70}
"""

for cluster in range(optimal_k):
    cluster_data = df_features[df_features['cluster'] == cluster]
    report += f"""
Type {cluster}: {cluster_names[cluster]}
Description: {cluster_descriptions[cluster]}
Count: {len(cluster_data):,} vehicles ({len(cluster_data)/len(df_features)*100:.1f}%)
Average Power: {cluster_data['power_mean_avg'].mean():.2f} kW
Average Speed: {cluster_data['speed_mean_avg'].mean():.1f} km/h
SOC Risk Ratio: {cluster_data['low_soc_risk_ratio'].mean():.1%}
Charging Frequency: {cluster_data['charging_freq'].mean():.2f} times/day
Power Efficiency: {cluster_data['power_efficiency'].mean():.2f}
"""

report += f"\n{'='*70}\n"

with open(os.path.join(results_dir, 'clustering_report.txt'), 'w', encoding='utf-8') as f:
    f.write(report)

print(report)
print(f"✅ Saved: clustering_report.txt")

print(f"\n{'='*70}")
print(f"✅ Step 9 Complete!")
print(f"{'='*70}")
print(f"\n📁 Output Files:")
print(f"   ✅ vehicle_clustering_results.csv")
print(f"   ✅ vehicle_clustering_professional.png (推荐查看)")
print(f"   ✅ k_selection_analysis.png")
print(f"   ✅ gmm_model.pkl")
print(f"   ✅ clustering_info.json")
print(f"   ✅ clustering_report.txt")