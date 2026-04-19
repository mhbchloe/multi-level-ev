"""
Step 9: Feature Dimension Analysis for Vehicle Clusters
特征维度分析 - 深度理解各车辆聚类的特征特性
"""

import numpy as np
import pandas as pd
import os
import json
import pickle
import warnings
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib import rcParams
from matplotlib.patches import Rectangle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import entropy, spearmanr
from scipy.spatial.distance import pdist, squareform
from sklearn.feature_selection import mutual_info_classif
from collections import defaultdict
from tqdm import tqdm

warnings.filterwarnings('ignore')

rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 11
rcParams['figure.dpi'] = 150

# ============================================================
# 自定义 JSON 编码器处理 NumPy 类型
# ============================================================
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

print("=" * 80)
print("🚀 STEP 9: FEATURE DIMENSION ANALYSIS FOR VEHICLE CLUSTERS")
print("=" * 80)

# ============================================================
# 配置
# ============================================================
CONFIG = {
    'vehicle_clustering_path': './vehicle_clustering/results/vehicle_clustering_gmm_k4.csv',
    'vehicle_features_path': './coupling_analysis/results/vehicles_aggregated_features.csv',
    'clustering_summary_path': './vehicle_clustering/results/vehicle_clustering_gmm_k4_summary.json',
    'save_dir': './vehicle_clustering/results/feature_analysis/',
    'seed': 42,
}

os.makedirs(CONFIG['save_dir'], exist_ok=True)

# ============================================================
# 1. 加载数据
# ============================================================
print(f"\n【STEP 1】Loading Data")
print("=" * 80)

# 加载聚类结果
vehicle_data = pd.read_csv(CONFIG['vehicle_clustering_path'])
vehicle_features_full = pd.read_csv(CONFIG['vehicle_features_path'])

with open(CONFIG['clustering_summary_path'], 'r') as f:
    summary = json.load(f)

print(f"   ✓ Vehicle clustering: {len(vehicle_data):,} vehicles")
print(f"   ✓ Clusters: {summary['n_clusters']}")
print(f"   ✓ Features: {summary['n_features']}")

# ============================================================
# 2. 特征准备
# ============================================================
print(f"\n【STEP 2】Feature Preparation")
print("=" * 80)

# 获取所有特征列
feature_names = summary['feature_names']
cluster_ratio_cols = [f'cluster_{c}_ratio' for c in range(4)]
behavior_cols = ['high_energy_ratio', 'idle_dominant_ratio']
phys_cols = [c for c in feature_names if c.startswith('avg_')]

print(f"\n   Feature Categories:")
print(f"      Cluster composition: {len(cluster_ratio_cols)} features")
print(f"      Driving behavior: {len(behavior_cols)} features")
print(f"      Physical features: {len(phys_cols)} features")

# 提取特征矩阵
X = vehicle_data[feature_names].copy()
X = X.fillna(0).astype(np.float32)
y = vehicle_data['vehicle_cluster'].values

print(f"\n   Data shape: {X.shape}")
print(f"   Clusters: {sorted(np.unique(y))}")

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

print(f"   ✓ Standardized")

# ============================================================
# 3. 特征重要性分析
# ============================================================
print(f"\n【STEP 3】Feature Importance Analysis")
print("=" * 80)

# 3.1 方差分析 (Variance)
print(f"   3.1 Variance Analysis...")
feature_variance = X_scaled.var(axis=0)
variance_rank = np.argsort(-feature_variance)

# 3.2 相互信息 (Mutual Information)
print(f"   3.2 Mutual Information Analysis...")
mi_scores = mutual_info_classif(X_scaled, y, random_state=CONFIG['seed'])
mi_rank = np.argsort(-mi_scores)

# 3.3 相关性分析 (Correlation with cluster membership)
print(f"   3.3 Correlation Analysis...")
correlations = []
for i, fname in enumerate(feature_names):
    corr, _ = spearmanr(X_scaled[:, i], y)
    correlations.append(abs(corr))
correlations = np.array(correlations)
corr_rank = np.argsort(-correlations)

# 综合排名
print(f"   3.4 Composite Ranking...")
combined_scores = (
    0.35 * (len(feature_names) - np.argsort(np.argsort(-feature_variance))) +
    0.40 * (len(feature_names) - np.argsort(np.argsort(-mi_scores))) +
    0.25 * (len(feature_names) - np.argsort(np.argsort(-correlations)))
)
composite_rank = np.argsort(combined_scores)

# 保存特征重要性
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'variance': feature_variance,
    'variance_rank': np.argsort(np.argsort(-feature_variance)) + 1,
    'mi_score': mi_scores,
    'mi_rank': np.argsort(np.argsort(-mi_scores)) + 1,
    'correlation': correlations,
    'correlation_rank': np.argsort(np.argsort(-correlations)) + 1,
    'composite_score': combined_scores,
    'composite_rank': np.argsort(combined_scores) + 1,
})

feature_importance = feature_importance.sort_values('composite_rank')
print(f"\n   Top 10 Features by Composite Score:")
for idx, row in feature_importance.head(10).iterrows():
    print(f"      #{row['composite_rank']:<2.0f} {row['feature']:<30} | "
          f"MI={row['mi_score']:.4f} | Var={row['variance']:.4f} | Corr={row['correlation']:.4f}")

feature_importance.to_csv(
    os.path.join(CONFIG['save_dir'], 'feature_importance.csv'),
    index=False
)
print(f"   ✓ Saved: feature_importance.csv")

# ============================================================
# 4. 聚类特征轮廓 (Cluster Feature Profiles)
# ============================================================
print(f"\n【STEP 4】Cluster Feature Profiles")
print("=" * 80)

unique_clusters = sorted(np.unique(y))
cluster_profiles = {}

print(f"\n   Cluster Profiles:")
for cluster_id in unique_clusters:
    mask = y == cluster_id
    X_cluster = X[mask]
    
    # 计算统计量
    mean_vals = X_cluster.mean(axis=0)
    std_vals = X_cluster.std(axis=0)
    median_vals = X_cluster.median(axis=0)
    
    cluster_profiles[cluster_id] = {
        'size': mask.sum(),
        'mean': mean_vals,
        'std': std_vals,
        'median': median_vals,
    }
    
    print(f"\n   V{cluster_id}: {mask.sum():,} vehicles")
    print(f"      Cluster Composition:")
    for c in range(4):
        print(f"         C{c}: {mean_vals[c]:.3f} ± {std_vals[c]:.3f}")

# 保存聚类特征
cluster_profiles_df = pd.DataFrame()
for cluster_id in unique_clusters:
    profile = cluster_profiles[cluster_id]
    row_data = {'cluster': cluster_id, 'size': profile['size']}
    
    for i, fname in enumerate(feature_names):
        row_data[f'{fname}_mean'] = profile['mean'][i]
        row_data[f'{fname}_std'] = profile['std'][i]
        row_data[f'{fname}_median'] = profile['median'][i]
    
    cluster_profiles_df = pd.concat(
        [cluster_profiles_df, pd.DataFrame([row_data])],
        ignore_index=True
    )

cluster_profiles_df.to_csv(
    os.path.join(CONFIG['save_dir'], 'cluster_profiles.csv'),
    index=False
)
print(f"\n   ✓ Saved: cluster_profiles.csv")

# ============================================================
# 5. 聚类特征差异分析
# ============================================================
print(f"\n【STEP 5】Cluster Feature Differentiation")
print("=" * 80)

# 计算每个特征的簇间差异度 (Between-cluster variance / Within-cluster variance)
feature_discrimination = []

for i, fname in enumerate(feature_names):
    # 全局方差
    total_var = X_scaled[:, i].var()
    
    # 簇内方差
    within_var = 0
    for cluster_id in unique_clusters:
        mask = y == cluster_id
        within_var += (X_scaled[mask, i].var() * mask.sum()) / len(X_scaled)
    
    # 簇间方差
    between_var = total_var - within_var
    
    # 差异比率
    if total_var > 1e-8:
        discrimination_ratio = between_var / total_var if within_var > 1e-8 else 0
    else:
        discrimination_ratio = 0
    
    feature_discrimination.append({
        'feature': fname,
        'within_var': within_var,
        'between_var': between_var,
        'total_var': total_var,
        'discrimination_ratio': discrimination_ratio,
    })

feature_discrimination_df = pd.DataFrame(feature_discrimination).sort_values(
    'discrimination_ratio', ascending=False
)

print(f"\n   Top 15 Differentiating Features:")
for idx, row in feature_discrimination_df.head(15).iterrows():
    print(f"      {row['feature']:<30} | Discrimination={row['discrimination_ratio']:.4f}")

feature_discrimination_df.to_csv(
    os.path.join(CONFIG['save_dir'], 'feature_discrimination.csv'),
    index=False
)
print(f"   ✓ Saved: feature_discrimination.csv")

# ============================================================
# 6. 聚类特征标签优化
# ============================================================
print(f"\n【STEP 6】Optimized Cluster Labeling")
print("=" * 80)

optimized_labels = {}

print(f"\n   Cluster Label Generation:")
for cluster_id in unique_clusters:
    profile = cluster_profiles[cluster_id]
    mean_vals = profile['mean']
    
    # 获取特征值
    comp_0 = mean_vals[0]  # cluster_0_ratio
    comp_1 = mean_vals[1]  # cluster_1_ratio
    comp_2 = mean_vals[2]  # cluster_2_ratio
    comp_3 = mean_vals[3]  # cluster_3_ratio
    high_energy_idx = feature_names.index('high_energy_ratio')
    idle_idx = feature_names.index('idle_dominant_ratio')
    high_energy = mean_vals[high_energy_idx]
    idle_dominant = mean_vals[idle_idx]
    
    # 生成标签
    characteristics = []
    
    if comp_2 > 0.30:
        characteristics.append("Highway-driven")
    if comp_1 > 0.35:
        characteristics.append("Mixed-urban")
    if idle_dominant > 0.55:
        characteristics.append("Parking-heavy")
    if high_energy > 0.35:
        characteristics.append("Aggressive")
    if comp_0 > 0.25:
        characteristics.append("Idle-prone")
    
    if not characteristics:
        characteristics = ["Balanced"]
    
    label = " + ".join(characteristics)
    
    optimized_labels[int(cluster_id)] = {
        'label': label,
        'characteristics': characteristics,
        'composition': {
            'C0_idle': float(comp_0),
            'C1_mixed': float(comp_1),
            'C2_highway': float(comp_2),
            'C3_mixed': float(comp_3),
        },
        'driving_behavior': {
            'high_energy_ratio': float(high_energy),
            'idle_dominant_ratio': float(idle_dominant),
        }
    }
    
    print(f"   V{cluster_id}: {label}")
    print(f"      Composition: C0={comp_0:.1%} C1={comp_1:.1%} C2={comp_2:.1%} C3={comp_3:.1%}")
    print(f"      Behavior: High-energy={high_energy:.1%} Idle={idle_dominant:.1%}")

# 保存优化的标签
with open(os.path.join(CONFIG['save_dir'], 'optimized_cluster_labels.json'), 'w') as f:
    json.dump(optimized_labels, f, indent=2, cls=NumpyEncoder)
print(f"   ✓ Saved: optimized_cluster_labels.json")

# ============================================================
# 7. 可视化 - 特征重要性
# ============================================================
print(f"\n【STEP 7】Visualization - Feature Importance")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 7.1 方差
ax = axes[0, 0]
top_n = 20
top_features_var = feature_importance.head(top_n)
colors_var = plt.cm.Blues(np.linspace(0.4, 0.9, len(top_features_var)))
bars = ax.barh(range(len(top_features_var)), top_features_var['variance'].values, color=colors_var)
ax.set_yticks(range(len(top_features_var)))
ax.set_yticklabels(top_features_var['feature'].values, fontsize=9)
ax.set_xlabel('Variance', fontweight='bold')
ax.set_title('(a) Top Features by Variance', fontweight='bold', fontsize=12)
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

# 7.2 互信息
ax = axes[0, 1]
top_features_mi = feature_importance.sort_values('mi_score', ascending=False).head(top_n)
colors_mi = plt.cm.Greens(np.linspace(0.4, 0.9, len(top_features_mi)))
bars = ax.barh(range(len(top_features_mi)), top_features_mi['mi_score'].values, color=colors_mi)
ax.set_yticks(range(len(top_features_mi)))
ax.set_yticklabels(top_features_mi['feature'].values, fontsize=9)
ax.set_xlabel('Mutual Information Score', fontweight='bold')
ax.set_title('(b) Top Features by Mutual Information', fontweight='bold', fontsize=12)
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

# 7.3 相关性
ax = axes[1, 0]
top_features_corr = feature_importance.sort_values('correlation', ascending=False).head(top_n)
colors_corr = plt.cm.Oranges(np.linspace(0.4, 0.9, len(top_features_corr)))
bars = ax.barh(range(len(top_features_corr)), top_features_corr['correlation'].values, color=colors_corr)
ax.set_yticks(range(len(top_features_corr)))
ax.set_yticklabels(top_features_corr['feature'].values, fontsize=9)
ax.set_xlabel('|Correlation with Cluster|', fontweight='bold')
ax.set_title('(c) Top Features by Cluster Correlation', fontweight='bold', fontsize=12)
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

# 7.4 综合排名
ax = axes[1, 1]
top_features_comp = feature_importance.head(top_n)
colors_comp = plt.cm.Purples(np.linspace(0.4, 0.9, len(top_features_comp)))
ranks = np.arange(1, len(top_features_comp) + 1)
scores = 21 - ranks  # 反向分数
bars = ax.barh(range(len(top_features_comp)), scores, color=colors_comp)
ax.set_yticks(range(len(top_features_comp)))
ax.set_yticklabels(top_features_comp['feature'].values, fontsize=9)
ax.set_xlabel('Composite Score (Higher is Better)', fontweight='bold')
ax.set_title('(d) Top Features by Composite Score', fontweight='bold', fontsize=12)
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
for fmt, dpi in [('.png', 300), ('.pdf', None)]:
    path = os.path.join(CONFIG['save_dir'], f'feature_importance_analysis{fmt}')
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: feature_importance_analysis.png/pdf")

# ============================================================
# 8. 可视化 - 聚类特征热力图
# ============================================================
print(f"\n【STEP 8】Visualization - Cluster Feature Heatmaps")
print("=" * 80)

# 准备数据：每个聚类的特征均值
top_features_list = feature_importance.head(20)['feature'].tolist()
heatmap_data = []

for cluster_id in unique_clusters:
    profile = cluster_profiles[cluster_id]
    mean_vals = profile['mean']
    row = []
    for fname in top_features_list:
        idx = feature_names.index(fname)
        row.append(mean_vals[idx])
    heatmap_data.append(row)

heatmap_data = np.array(heatmap_data)

# 标准化用于热力图
heatmap_normalized = (heatmap_data - heatmap_data.mean(axis=0)) / (heatmap_data.std(axis=0) + 1e-8)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

# 原始值
sns.heatmap(heatmap_data, 
           xticklabels=top_features_list,
           yticklabels=[f"V{c}\n{optimized_labels[c]['label'][:30]}" for c in unique_clusters],
           cmap='RdYlGn',
           annot=True, 
           fmt='.3f',
           ax=ax1,
           cbar_kws={'label': 'Feature Value'})
ax1.set_title('(a) Cluster Feature Profiles (Raw Values)', fontweight='bold', fontsize=12)
ax1.set_xticklabels(top_features_list, rotation=45, ha='right', fontsize=9)

# 标准化值
sns.heatmap(heatmap_normalized,
           xticklabels=top_features_list,
           yticklabels=[f"V{c}" for c in unique_clusters],
           cmap='coolwarm',
           center=0,
           annot=True,
           fmt='.2f',
           ax=ax2,
           cbar_kws={'label': 'Standardized Value'})
ax2.set_title('(b) Cluster Feature Profiles (Standardized)', fontweight='bold', fontsize=12)
ax2.set_xticklabels(top_features_list, rotation=45, ha='right', fontsize=9)

plt.tight_layout()
for fmt, dpi in [('.png', 300), ('.pdf', None)]:
    path = os.path.join(CONFIG['save_dir'], f'cluster_feature_heatmap{fmt}')
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: cluster_feature_heatmap.png/pdf")

# ============================================================
# 9. 可视化 - 雷达图
# ============================================================
print(f"\n【STEP 9】Visualization - Radar Charts")
print("=" * 80)

# 选择关键特征用于雷达图
key_features = ['cluster_0_ratio', 'cluster_1_ratio', 'cluster_2_ratio', 'cluster_3_ratio',
                'high_energy_ratio', 'idle_dominant_ratio']
key_indices = [feature_names.index(f) for f in key_features if f in feature_names]
key_features_filtered = [feature_names[i] for i in key_indices]

n_clusters = len(unique_clusters)
n_features = len(key_features_filtered)

fig = plt.figure(figsize=(4*n_clusters, 4*n_clusters))
angles = np.linspace(0, 2*np.pi, n_features, endpoint=False).tolist()
angles += angles[:1]

colors_radar = plt.cm.Set3(np.linspace(0, 1, n_clusters))

for plot_idx, cluster_id in enumerate(unique_clusters):
    ax = fig.add_subplot(2, 2, plot_idx + 1, projection='polar')
    
    profile = cluster_profiles[cluster_id]
    values = [profile['mean'][i] for i in key_indices]
    values += values[:1]
    
    ax.plot(angles, values, 'o-', linewidth=2, color=colors_radar[plot_idx], label='Values')
    ax.fill(angles, values, alpha=0.25, color=colors_radar[plot_idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(key_features_filtered, fontsize=9)
    ax.set_ylim(0, max([profile['mean'][i] for i in key_indices]) * 1.2)
    ax.set_title(f"V{cluster_id}: {optimized_labels[cluster_id]['label'][:40]}", 
                fontweight='bold', fontsize=11, pad=20)
    ax.grid(True)

plt.tight_layout()
for fmt, dpi in [('.png', 300), ('.pdf', None)]:
    path = os.path.join(CONFIG['save_dir'], f'cluster_radar_charts{fmt}')
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: cluster_radar_charts.png/pdf")

# ============================================================
# 10. 可视化 - 特征判别度
# ============================================================
print(f"\n【STEP 10】Visualization - Feature Discrimination")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 10.1 判别度排名
ax = axes[0, 0]
top_disc = feature_discrimination_df.head(15)
colors_disc = plt.cm.Reds(np.linspace(0.4, 0.9, len(top_disc)))
bars = ax.barh(range(len(top_disc)), top_disc['discrimination_ratio'].values, color=colors_disc)
ax.set_yticks(range(len(top_disc)))
ax.set_yticklabels(top_disc['feature'].values, fontsize=9)
ax.set_xlabel('Discrimination Ratio (Between/Total Variance)', fontweight='bold')
ax.set_title('(a) Top Features by Discrimination Ratio', fontweight='bold', fontsize=12)
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

# 10.2 方差分解
ax = axes[0, 1]
top_disc_plot = feature_discrimination_df.head(10)
between_vars = top_disc_plot['between_var'].values
within_vars = top_disc_plot['within_var'].values
x_pos = np.arange(len(top_disc_plot))
ax.bar(x_pos, between_vars, label='Between-cluster', color='#3498db', edgecolor='black', linewidth=1)
ax.bar(x_pos, within_vars, bottom=between_vars, label='Within-cluster', color='#e74c3c', edgecolor='black', linewidth=1)
ax.set_xticks(x_pos)
ax.set_xticklabels(top_disc_plot['feature'].values, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Variance', fontweight='bold')
ax.set_title('(b) Variance Decomposition', fontweight='bold', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# 10.3 聚类大小分布
ax = axes[1, 0]
cluster_sizes = [cluster_profiles[c]['size'] for c in unique_clusters]
cluster_labels_list = [f"V{c}\n{optimized_labels[c]['label'][:25]}" for c in unique_clusters]
colors_size = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))
bars = ax.bar(range(len(unique_clusters)), cluster_sizes, color=colors_size, edgecolor='black', linewidth=1.5)
for i, (bar, size) in enumerate(zip(bars, cluster_sizes)):
    ax.text(bar.get_x() + bar.get_width()/2, size + max(cluster_sizes)*0.02,
           f'{size:,}', ha='center', fontweight='bold', fontsize=10)
ax.set_xticks(range(len(unique_clusters)))
ax.set_xticklabels(cluster_labels_list, fontsize=10)
ax.set_ylabel('Number of Vehicles', fontweight='bold')
ax.set_title('(c) Cluster Size Distribution', fontweight='bold', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# 10.4 特征类别分布
ax = axes[1, 1]
category_counts = defaultdict(int)
for fname in feature_names:
    if 'cluster_' in fname:
        category_counts['Cluster Composition'] += 1
    elif fname in ['high_energy_ratio', 'idle_dominant_ratio']:
        category_counts['Driving Behavior'] += 1
    elif fname.startswith('avg_'):
        category_counts['Physical Features'] += 1

categories = list(category_counts.keys())
counts = list(category_counts.values())
colors_cat = ['#3498db', '#2ecc71', '#e74c3c']
bars = ax.bar(categories, counts, color=colors_cat, edgecolor='black', linewidth=1.5)
for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, count + max(counts)*0.02,
           f'{count}', ha='center', fontweight='bold', fontsize=11)
ax.set_ylabel('Feature Count', fontweight='bold')
ax.set_title('(d) Feature Category Distribution', fontweight='bold', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
for fmt, dpi in [('.png', 300), ('.pdf', None)]:
    path = os.path.join(CONFIG['save_dir'], f'feature_discrimination_analysis{fmt}')
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: feature_discrimination_analysis.png/pdf")

# ============================================================
# 11. 相关性矩阵
# ============================================================
print(f"\n【STEP 11】Correlation Matrix")
print("=" * 80)

# 计算相关性矩阵
corr_matrix = np.corrcoef(X_scaled.T)

fig, ax = plt.subplots(figsize=(16, 14))
sns.heatmap(corr_matrix,
           xticklabels=feature_names,
           yticklabels=feature_names,
           cmap='coolwarm',
           center=0,
           square=True,
           linewidths=0.5,
           cbar_kws={'label': 'Correlation Coefficient'},
           ax=ax,
           vmin=-1, vmax=1)
ax.set_title('Feature Correlation Matrix', fontweight='bold', fontsize=13, pad=20)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()

for fmt, dpi in [('.png', 300), ('.pdf', None)]:
    path = os.path.join(CONFIG['save_dir'], f'feature_correlation_matrix{fmt}')
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: feature_correlation_matrix.png/pdf")

# ============================================================
# 12. 综合报告生成
# ============================================================
print(f"\n【STEP 12】Generating Comprehensive Report")
print("=" * 80)

report = {
    'title': 'Vehicle Cluster Feature Dimension Analysis Report',
    'timestamp': pd.Timestamp.now().isoformat(),
    'summary': {
        'total_vehicles': int(len(vehicle_data)),
        'n_clusters': len(unique_clusters),
        'n_features': len(feature_names),
        'n_features_by_category': dict(category_counts),
    },
    'feature_importance_top10': feature_importance.head(10).to_dict('records'),
    'feature_discrimination_top10': feature_discrimination_df.head(10).to_dict('records'),
    'cluster_characteristics': optimized_labels,
    'cluster_statistics': {},
}

# 添加集群统计
for cluster_id in unique_clusters:
    cluster_id_int = int(cluster_id)
    report['cluster_statistics'][f'V{cluster_id_int}'] = {
        'size': int(cluster_profiles[cluster_id]['size']),
        'label': optimized_labels[cluster_id_int]['label'],
        'composition': optimized_labels[cluster_id_int]['composition'],
        'driving_behavior': optimized_labels[cluster_id_int]['driving_behavior'],
    }

with open(os.path.join(CONFIG['save_dir'], 'feature_analysis_report.json'), 'w') as f:
    json.dump(report, f, indent=2, cls=NumpyEncoder)
print(f"   ✓ Saved: feature_analysis_report.json")

# 生成文本报告
report_text = f"""
{'='*80}
VEHICLE CLUSTER FEATURE DIMENSION ANALYSIS REPORT
{'='*80}

Generated: {pd.Timestamp.now()}

1. SUMMARY
{'-'*80}
Total Vehicles: {len(vehicle_data):,}
Number of Clusters: {len(unique_clusters)}
Total Features: {len(feature_names)}

Feature Categories:
  - Cluster Composition: {category_counts['Cluster Composition']}
  - Driving Behavior: {category_counts['Driving Behavior']}
  - Physical Features: {category_counts['Physical Features']}

2. TOP 10 MOST IMPORTANT FEATURES
{'-'*80}
"""

for idx, row in feature_importance.head(10).iterrows():
    report_text += f"  #{row['composite_rank']:<2.0f} {row['feature']:<30} | MI={row['mi_score']:.4f} | Var={row['variance']:.4f}\n"

report_text += f"""
3. TOP 10 MOST DISCRIMINATING FEATURES
{'-'*80}
"""

for idx, row in feature_discrimination_df.head(10).iterrows():
    report_text += f"  {row['feature']:<30} | Discrimination Ratio={row['discrimination_ratio']:.4f}\n"

report_text += f"""
4. CLUSTER CHARACTERISTICS
{'-'*80}
"""

for cluster_id in unique_clusters:
    cluster_id_int = int(cluster_id)
    label_info = optimized_labels[cluster_id_int]
    size = cluster_profiles[cluster_id]['size']
    report_text += f"\nV{cluster_id_int}: {label_info['label']}\n"
    report_text += f"  Size: {size:,} vehicles ({size/len(vehicle_data)*100:.1f}%)\n"
    report_text += f"  Composition:\n"
    for comp_key, comp_val in label_info['composition'].items():
        report_text += f"    {comp_key}: {comp_val:.1%}\n"
    report_text += f"  Driving Behavior:\n"
    for behavior_key, behavior_val in label_info['driving_behavior'].items():
        report_text += f"    {behavior_key}: {behavior_val:.1%}\n"

report_text += f"""
5. OUTPUT FILES
{'-'*80}
  1. feature_importance.csv - Comprehensive feature importance rankings
  2. cluster_profiles.csv - Detailed cluster feature profiles
  3. feature_discrimination.csv - Feature discrimination analysis
  4. optimized_cluster_labels.json - Optimized cluster labels and characteristics
  5. feature_analysis_report.json - Structured analysis report
  6. feature_importance_analysis.png/pdf - Feature importance visualizations
  7. cluster_feature_heatmap.png/pdf - Cluster feature heatmaps
  8. cluster_radar_charts.png/pdf - Radar charts for each cluster
  9. feature_discrimination_analysis.png/pdf - Discrimination analysis plots
  10. feature_correlation_matrix.png/pdf - Feature correlation heatmap

{'='*80}
END OF REPORT
{'='*80}
"""

with open(os.path.join(CONFIG['save_dir'], 'feature_analysis_report.txt'), 'w') as f:
    f.write(report_text)
print(f"   ✓ Saved: feature_analysis_report.txt")

# ============================================================
# 完成
# ============================================================
print("\n" + "=" * 80)
print("✅ STEP 9 COMPLETE!")
print("=" * 80)

print(f"""
Analysis Summary:
  - Feature Importance Analysis: ✓
  - Cluster Feature Profiles: ✓
  - Feature Discrimination Analysis: ✓
  - Cluster Label Optimization: ✓
  - Comprehensive Visualizations: ✓

Output Files Generated:
  {len([f for f in os.listdir(CONFIG['save_dir']) if f.endswith(('.csv', '.json', '.txt', '.png', '.pdf'))])} files

Key Insights:
  - Top Differentiating Feature: {feature_discrimination_df.iloc[0]['feature']}
  - Most Important Feature: {feature_importance.iloc[0]['feature']}
  - Clusters Identified: {len(unique_clusters)}

Next Steps:
  - Use optimized labels for charging analysis
  - Apply feature importance for downstream modeling
  - Integrate with inter-charge trips analysis
""")

print("=" * 80)