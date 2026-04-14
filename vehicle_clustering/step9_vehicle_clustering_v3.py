"""
Step 9 v3: K=3 Clustering with Segment Integration
整合Segment信息的K=3聚类
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import pickle
import os
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['figure.figsize'] = (18, 13)
plt.rcParams['font.size'] = 10

print("="*70)
print("🎯 Vehicle Clustering v3 (K=3, Segment-Integrated)")
print("="*70)

results_dir = "./vehicle_clustering/results/"

# 加载特征
df_features = pd.read_csv(os.path.join(results_dir, 'vehicle_advanced_features_v3.csv'))
print(f"✅ Vehicles: {len(df_features):,}")

# 特征标准化
feature_cols = [col for col in df_features.columns 
                if col not in ['vehicle_id', 'n_segments', 'n_events', 'n_charging']]

X = df_features[feature_cols].copy()
X = X.fillna(X.median())

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

with open(os.path.join(results_dir, 'scaler_v3.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

print(f"✅ Features standardized: {X_scaled.shape}")

# ========== K=3 聚类 ==========
print(f"\n{'='*70}")
print(f"🎯 GMM Clustering with K=3")
print(f"{'='*70}")

optimal_k = 3

gmm = GaussianMixture(n_components=optimal_k, n_init=150, random_state=42, max_iter=500)
vehicle_labels = gmm.fit_predict(X_scaled)
vehicle_probs = gmm.predict_proba(X_scaled)

sil = silhouette_score(X_scaled, vehicle_labels)
db = davies_bouldin_score(X_scaled, vehicle_labels)

print(f"\n✅ Clustering Results:")
print(f"   Silhouette Score: {sil:.4f}")
print(f"   Davies-Bouldin Index: {db:.4f}")

print(f"\n📊 Cluster Distribution:")
unique, counts = np.unique(vehicle_labels, return_counts=True)
for cluster, count in zip(unique, counts):
    print(f"   Cluster {cluster}: {count:,} ({count/len(vehicle_labels)*100:.1f}%)")

# ========== 分析 ==========
df_features['cluster'] = vehicle_labels
df_features['cluster_prob'] = vehicle_probs.max(axis=1)

print(f"\n{'='*70}")
print(f"📊 Cluster Characteristics")
print(f"{'='*70}")

# Segment聚类分布对比
seg_features = ['cluster_0_ratio', 'cluster_1_ratio', 'cluster_2_ratio', 'cluster_3_ratio']
seg_profile = df_features.groupby('cluster')[seg_features].mean()

print(f"\nSegment Cluster Distribution by Vehicle Type:")
print(seg_profile.round(3))

# 能耗对比
energy_features = ['weighted_power', 'overall_avg_power', 'energy_intensity']
energy_profile = df_features.groupby('cluster')[energy_features].mean()

print(f"\nEnergy Profile by Vehicle Type:")
print(energy_profile.round(3))

# 风险对比
risk_features = ['low_soc_ratio', 'deep_discharge_ratio', 'charging_urgency']
risk_profile = df_features.groupby('cluster')[risk_features].mean()

print(f"\nRisk Profile by Vehicle Type:")
print(risk_profile.round(3))

# ========== 命名 ==========
cluster_names = {}
cluster_descriptions = {}

for cluster in range(optimal_k):
    cluster_data = df_features[df_features['cluster'] == cluster]
    
    # 获取该簇的关键特征
    avg_aggressive = cluster_data['aggressive_driving_ratio'].mean()
    avg_conservative = cluster_data['conservative_driving_ratio'].mean()
    avg_highway = cluster_data['highway_driving_ratio'].mean()
    avg_power = cluster_data['weighted_power'].mean()
    avg_low_soc = cluster_data['low_soc_ratio'].mean()
    avg_charging_urgency = cluster_data['charging_urgency'].mean()
    
    # 智能分类
    if avg_aggressive > 0.3 and avg_power < -12:
        name = "高能耗激进型"
        desc = "驾驶激进、能耗高、功率输出强，需要加强电池管理和安全驾驶引导"
    elif avg_low_soc > 0.15 or avg_charging_urgency > 0.25:
        name = "低SOC风险型"
        desc = "低SOC运行频繁、充电管理不当、电池寿命风险高，需优化充电策略"
    else:
        name = "高效经济型"
        desc = "驾驶稳定、能耗低、充电管理良好，运营成本最优"
    
    cluster_names[cluster] = name
    cluster_descriptions[cluster] = desc
    
    print(f"\n🚗 Type {cluster}: {name}")
    print(f"   {desc}")
    print(f"   车辆数: {len(cluster_data):,} ({len(cluster_data)/len(df_features)*100:.1f}%)")
    print(f"   驾驶模式: 激进={avg_aggressive:.1%}, 保守={avg_conservative:.1%}, 高速={avg_highway:.1%}")
    print(f"   平均功率: {avg_power:.2f} kW")
    print(f"   低SOC风险: {avg_low_soc:.1%}")
    print(f"   充电迫切度: {avg_charging_urgency:.1%}")

df_features['vehicle_type'] = df_features['cluster'].map(cluster_names)

# ========== 保存 ==========
df_features.to_csv(os.path.join(results_dir, 'vehicle_clustering_results_v3.csv'), index=False)

with open(os.path.join(results_dir, 'gmm_model_v3.pkl'), 'wb') as f:
    pickle.dump(gmm, f)

cluster_info = {
    'method': 'GMM',
    'n_clusters': optimal_k,
    'cluster_names': cluster_names,
    'cluster_descriptions': cluster_descriptions,
    'metrics': {'silhouette': float(sil), 'davies_bouldin': float(db)}
}

with open(os.path.join(results_dir, 'clustering_info_v3.json'), 'w') as f:
    json.dump(cluster_info, f, indent=2, ensure_ascii=False)

# ========== 可视化 ==========
print(f"\n📈 Generating visualizations...")

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

fig = plt.figure(figsize=(18, 13))
gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.35)

# 1. 分布
ax1 = fig.add_subplot(gs[0, 0])
cluster_counts = [sum(vehicle_labels == i) for i in range(optimal_k)]
bars = ax1.bar(range(optimal_k), cluster_counts, color=colors, alpha=0.85, edgecolor='black', linewidth=2)

for bar, count in zip(bars, cluster_counts):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(count):,}\n({count/len(vehicle_labels)*100:.1f}%)',
            ha='center', va='bottom', fontweight='bold', fontsize=10)

ax1.set_ylabel('Number of Vehicles', fontweight='bold')
ax1.set_title('Vehicle Type Distribution', fontweight='bold', fontsize=12)
ax1.set_xticks(range(optimal_k))
ax1.set_xticklabels([f'Type {i}' for i in range(optimal_k)])
ax1.grid(alpha=0.3, axis='y')

# 2. Segment分布（堆积柱）
ax2 = fig.add_subplot(gs[0, 1])
seg_cluster_names = ['Moderate', 'Conservative', 'Aggressive', 'Highway']
seg_colors = ['#FF9999', '#99CCFF', '#FF6B6B', '#66B2FF']

for cluster in range(optimal_k):
    cluster_data = df_features[df_features['cluster'] == cluster]
    seg_dist = [cluster_data['cluster_0_ratio'].mean(),
                cluster_data['cluster_1_ratio'].mean(),
                cluster_data['cluster_2_ratio'].mean(),
                cluster_data['cluster_3_ratio'].mean()]
    
    ax2.bar(cluster, seg_dist[0], color=seg_colors[0], alpha=0.8, label='Moderate' if cluster == 0 else '')
    ax2.bar(cluster, seg_dist[1], bottom=seg_dist[0], color=seg_colors[1], alpha=0.8, label='Conservative' if cluster == 0 else '')
    ax2.bar(cluster, seg_dist[2], bottom=sum(seg_dist[:2]), color=seg_colors[2], alpha=0.8, label='Aggressive' if cluster == 0 else '')
    ax2.bar(cluster, seg_dist[3], bottom=sum(seg_dist[:3]), color=seg_colors[3], alpha=0.8, label='Highway' if cluster == 0 else '')

ax2.set_ylabel('Proportion', fontweight='bold')
ax2.set_title('Segment Cluster Composition', fontweight='bold', fontsize=12)
ax2.set_xticks(range(optimal_k))
ax2.set_xticklabels([f'Type {i}' for i in range(optimal_k)])
ax2.legend(fontsize=8, loc='upper right')
ax2.grid(alpha=0.3, axis='y')

# 3. 能耗对比
ax3 = fig.add_subplot(gs[0, 2])
power_data = [df_features[df_features['cluster'] == i]['weighted_power'].values for i in range(optimal_k)]
bp = ax3.boxplot(power_data, patch_artist=True, labels=[f'Type {i}' for i in range(optimal_k)])
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax3.set_ylabel('Weighted Power (kW)', fontweight='bold')
ax3.set_title('Energy Consumption', fontweight='bold', fontsize=12)
ax3.grid(alpha=0.3, axis='y')

# 4. Segment 0 (Moderate)
ax4 = fig.add_subplot(gs[1, 0])
data = [df_features[df_features['cluster'] == i]['cluster_0_ratio'].values for i in range(optimal_k)]
bp = ax4.boxplot(data, patch_artist=True, labels=[f'Type {i}' for i in range(optimal_k)])
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax4.set_ylabel('Proportion', fontweight='bold')
ax4.set_title('Moderate Urban Driving', fontweight='bold', fontsize=11)
ax4.grid(alpha=0.3, axis='y')

# 5. Segment 2 (Aggressive)
ax5 = fig.add_subplot(gs[1, 1])
data = [df_features[df_features['cluster'] == i]['cluster_2_ratio'].values for i in range(optimal_k)]
bp = ax5.boxplot(data, patch_artist=True, labels=[f'Type {i}' for i in range(optimal_k)])
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax5.set_ylabel('Proportion', fontweight='bold')
ax5.set_title('Aggressive Urban Driving', fontweight='bold', fontsize=11)
ax5.grid(alpha=0.3, axis='y')

# 6. SOC风险
ax6 = fig.add_subplot(gs[1, 2])
data = [df_features[df_features['cluster'] == i]['low_soc_ratio'].values for i in range(optimal_k)]
bp = ax6.boxplot(data, patch_artist=True, labels=[f'Type {i}' for i in range(optimal_k)])
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax6.set_ylabel('Low SOC Ratio', fontweight='bold')
ax6.set_title('Low SOC Risk', fontweight='bold', fontsize=11)
ax6.grid(alpha=0.3, axis='y')

# 7. 雷达图
ax7 = fig.add_subplot(gs[2, :], projection='polar')

radar_features = ['cluster_2_ratio', 'weighted_power', 'low_soc_ratio', 
                  'charging_urgency', 'usage_freq', 'charging_freq']
radar_labels = ['Aggressive\nDriving', 'Energy\nConsumption', 'Low SOC\nRisk', 
                'Charging\nUrgency', 'Usage\nFrequency', 'Charging\nFrequency']

cluster_radar_data = []
for cluster in range(optimal_k):
    cluster_data = df_features[df_features['cluster'] == cluster]
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

for cluster in range(optimal_k):
    values = cluster_radar_norm[cluster].tolist()
    values += values[:1]
    ax7.plot(angles, values, 'o-', linewidth=2.5, label=cluster_names[cluster],
            color=colors[cluster], markersize=8)
    ax7.fill(angles, values, alpha=0.15, color=colors[cluster])

ax7.set_xticks(angles[:-1])
ax7.set_xticklabels(radar_labels, fontsize=10, fontweight='bold')
ax7.set_ylim(0, 1)
ax7.set_title('Multi-Dimensional Profile Comparison', fontsize=13, fontweight='bold', pad=20)
ax7.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
ax7.grid(True)

# 8. PCA投影
ax8 = fig.add_subplot(gs[3, 0:2])
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

for cluster in range(optimal_k):
    mask = vehicle_labels == cluster
    ax8.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=colors[cluster], label=cluster_names[cluster],
               alpha=0.6, s=60, edgecolors='black', linewidth=0.5)

ax8.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontweight='bold')
ax8.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontweight='bold')
ax8.set_title('Vehicle Clustering (PCA Projection)', fontweight='bold', fontsize=12)
ax8.legend(fontsize=10)
ax8.grid(alpha=0.3)

# 9. 信息
ax9 = fig.add_subplot(gs[3, 2])
ax9.axis('off')

metrics_text = f"""CLUSTERING RESULTS

K: {optimal_k} clusters
Silhouette: {sil:.4f}
Davies-Bouldin: {db:.4f}

Total Vehicles: {len(df_features):,}
Total Features: {len(feature_cols)}

Vehicle Types:
"""

for cluster in range(optimal_k):
    count = sum(vehicle_labels == cluster)
    metrics_text += f"\n  {cluster_names[cluster]}: {count:,}"

ax9.text(0.05, 0.95, metrics_text, transform=ax9.transAxes,
        fontfamily='monospace', fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.suptitle('Vehicle Clustering Analysis (K=3) - Segment-Integrated Features', 
            fontsize=16, fontweight='bold', y=0.995)

plt.savefig(os.path.join(results_dir, 'vehicle_clustering_v3_professional.png'), 
           dpi=300, bbox_inches='tight')
print(f"✅ Saved: vehicle_clustering_v3_professional.png")

# ========== 报告 ==========
report = f"""
{'='*70}
Vehicle Clustering Report (K=3) - Segment-Integrated
{'='*70}

Date: {pd.Timestamp.now()}

1. CLUSTERING QUALITY
{'-'*70}
Silhouette Score: {sil:.4f}
Davies-Bouldin Index: {db:.4f}
Number of Clusters: {optimal_k}

2. VEHICLE TYPES
{'-'*70}
"""

for cluster in range(optimal_k):
    cluster_data = df_features[df_features['cluster'] == cluster]
    report += f"""
Type {cluster}: {cluster_names[cluster]}
Description: {cluster_descriptions[cluster]}
Count: {len(cluster_data):,} ({len(cluster_data)/len(df_features)*100:.1f}%)

Segment Composition:
  Moderate Urban: {cluster_data['cluster_0_ratio'].mean():.1%}
  Conservative Urban: {cluster_data['cluster_1_ratio'].mean():.1%}
  Aggressive Urban: {cluster_data['cluster_2_ratio'].mean():.1%}
  Highway Efficient: {cluster_data['cluster_3_ratio'].mean():.1%}

Energy & Risk Metrics:
  Weighted Power: {cluster_data['weighted_power'].mean():.2f} kW
  Low SOC Ratio: {cluster_data['low_soc_ratio'].mean():.1%}
  Charging Urgency: {cluster_data['charging_urgency'].mean():.1%}
  Usage Frequency: {cluster_data['usage_freq'].mean():.2f} times/day
"""

report += f"\n{'='*70}\n"

with open(os.path.join(results_dir, 'clustering_report_v3.txt'), 'w', encoding='utf-8') as f:
    f.write(report)

print(report)

print(f"\n{'='*70}")
print(f"✅ Step 9 v3 Complete!")
print(f"{'='*70}")
print(f"\n📁 Output Files (v3):")
print(f"   ✅ vehicle_advanced_features_v3.csv")
print(f"   ✅ vehicle_clustering_results_v3.csv")
print(f"   ✅ vehicle_clustering_v3_professional.png")
print(f"   ✅ clustering_report_v3.txt")