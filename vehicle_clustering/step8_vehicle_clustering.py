"""
Step 8: Vehicle-Level Clustering (GMM with automatic K selection)
使用GMM，通过多指标综合选择最优K，完成车辆级聚类
"""

import numpy as np
import pandas as pd
import os
import json
import pickle
import warnings
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
from sklearn.preprocessing import RobustScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from scipy.stats import entropy
from collections import Counter
from tqdm import tqdm

warnings.filterwarnings('ignore')

rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 11
rcParams['figure.dpi'] = 150

print("=" * 80)
print("🚀 STEP 8: VEHICLE-LEVEL CLUSTERING (GMM)")
print("=" * 80)

# ============================================================
# 配置
# ============================================================
CONFIG = {
    'vehicle_features_path': './coupling_analysis/results/vehicles_aggregated_features.csv',
    'segments_path': './coupling_analysis/results/segments_integrated_complete.csv',
    'save_dir': './vehicle_clustering/results/',
    'seed': 42,
    'k_range': range(2, 8),  # 自动搜索 K
}

os.makedirs(CONFIG['save_dir'], exist_ok=True)

# ============================================================
# 1. 加载数据
# ============================================================
print(f"\n【STEP 1】Loading Data")
print("=" * 80)

# 1.1 车辆级特征
vehicle_features = pd.read_csv(CONFIG['vehicle_features_path'])
print(f"   ✓ Vehicle features: {len(vehicle_features):,} vehicles, {len(vehicle_features.columns)} features")

# 1.2 段级数据（用于补充信息）
segments_df = pd.read_csv(CONFIG['segments_path'])
print(f"   ✓ Segments: {len(segments_df):,} segments")

# ============================================================
# 2. 准备聚类特征
# ============================================================
print(f"\n【STEP 2】Preparing Clustering Features")
print("=" * 80)

# 选择用于聚类的特征 (去除冗余特征)
cluster_ratio_cols = [f'cluster_{c}_ratio' for c in range(4)]

# 新增的丰富特征
diversity_cols = [c for c in vehicle_features.columns if c in [
    'mode_diversity', 'mode_switch_rate', 'transition_entropy', 
    'self_loop_ratio', 'avg_run_length'
]]

# 完整 4×4 转移矩阵特征
trans_cols = [c for c in vehicle_features.columns if c.startswith('trans_')]

# 物理特征（均值和标准差）
phys_cols = [c for c in vehicle_features.columns if c.startswith('avg_') or c.startswith('std_')]

# SOC/能耗特征
soc_cols = [c for c in vehicle_features.columns if c in [
    'avg_soc_drop_per_segment', 'max_soc_drop', 'soc_consumption_rate', 'total_duration_hrs'
]]

# 注意：不使用 high_energy_ratio 和 idle_dominant_ratio (它们与 cluster ratios 线性冗余)
feature_cols = cluster_ratio_cols + diversity_cols + trans_cols + phys_cols + soc_cols
# 去重并保持顺序
seen = set()
feature_cols = [c for c in feature_cols if c in vehicle_features.columns and not (c in seen or seen.add(c))]

print(f"\n   Feature Selection:")
print(f"      Cluster composition: {len(cluster_ratio_cols)} features")
print(f"      Diversity/dynamics: {len(diversity_cols)} features")
print(f"      Transition matrix: {len(trans_cols)} features")
print(f"      Physical features: {len(phys_cols)} features")
print(f"      SOC/energy features: {len(soc_cols)} features")
print(f"      Total: {len(feature_cols)} features")

# 提取特征矩阵
X = vehicle_features[feature_cols].copy()

# 处理缺失值 (使用中位数填充代替0, 避免偏差)
missing = X.isna().sum().sum()
if missing > 0:
    print(f"\n   ⚠️  Found {missing} missing values, filling with median...")
    X = X.fillna(X.median())

X = X.astype(np.float32)
print(f"\n   Feature matrix: {X.shape}")

# ============================================================
# 3. 标准化 + PCA 降维
# ============================================================
print(f"\n【STEP 3】Feature Standardization + PCA")
print("=" * 80)

# 使用 RobustScaler 代替 StandardScaler (对异常值更鲁棒)
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

print(f"   ✓ RobustScaler: median-centered, IQR-scaled")

# 移除零方差特征
var = X_scaled.var(axis=0)
active = var > 1e-8
n_removed = (~active).sum()
if n_removed > 0:
    print(f"   ⚠️  Removing {n_removed} zero-variance features")
    X_active = X_scaled[:, active]
    active_feature_cols = [feature_cols[i] for i in range(len(feature_cols)) if active[i]]
else:
    X_active = X_scaled
    active_feature_cols = feature_cols

print(f"   ✓ Active features: {X_active.shape[1]}")

# PCA 降维 (保留 95% 方差，减少噪声维度)
pca_pre = PCA(n_components=0.95, random_state=CONFIG['seed'])
X_cluster = pca_pre.fit_transform(X_active)
n_pca_dims = X_cluster.shape[1]
pca_var_retained = pca_pre.explained_variance_ratio_.sum()
print(f"   ✓ PCA: {X_active.shape[1]} → {n_pca_dims} dims (variance retained: {pca_var_retained:.2%})")

# ============================================================
# 4. 自动 K 选择 + GMM 聚类
# ============================================================
print(f"\n【STEP 4】Automatic K Selection + GMM Clustering")
print("=" * 80)

k_range = CONFIG['k_range']
k_results = {}

print(f"\n   Testing K from {k_range.start} to {k_range.stop - 1}...")
print(f"   {'K':<5} {'BIC':<12} {'Silhouette':<12} {'CH':<12} {'DB':<10}")
print(f"   {'-'*50}")

for K in k_range:
    best_sil_k = -1
    best_result_k = None
    
    for cov_type in ['full', 'tied', 'diag']:
        try:
            gmm = GaussianMixture(
                n_components=K, 
                covariance_type=cov_type,
                n_init=10,
                random_state=CONFIG['seed'], 
                max_iter=500,
                reg_covar=1e-4,
                verbose=0
            )
            gmm.fit(X_cluster)
            labels = gmm.predict(X_cluster)
            
            sil = silhouette_score(X_cluster, labels, 
                                  sample_size=min(5000, len(X_cluster)), 
                                  random_state=CONFIG['seed'])
            ch = calinski_harabasz_score(X_cluster, labels)
            db = davies_bouldin_score(X_cluster, labels)
            bic = gmm.bic(X_cluster)
            
            if sil > best_sil_k:
                best_sil_k = sil
                best_result_k = {
                    'gmm': gmm, 'labels': labels, 'cov_type': cov_type,
                    'sil': sil, 'ch': ch, 'db': db, 'bic': bic
                }
        except Exception:
            pass
    
    if best_result_k is not None:
        k_results[K] = best_result_k
        r = best_result_k
        print(f"   {K:<5} {r['bic']:<12.1f} {r['sil']:<12.4f} {r['ch']:<12.1f} {r['db']:<10.4f}")

# 选择最优 K (基于 Silhouette)
best_K = max(k_results, key=lambda k: k_results[k]['sil'])
best_result = k_results[best_K]
best_labels = best_result['labels']
best_gmm = best_result['gmm']
best_cov_type = best_result['cov_type']
best_score = best_result['sil']
best_ch = best_result['ch']
best_db = best_result['db']
K = best_K

print(f"\n   🏆 Best: K={K}, cov_type='{best_cov_type}' (Silhouette={best_score:.4f})")

# ============================================================
# 5. 分析车辆聚类
# ============================================================
print(f"\n【STEP 5】Analyzing Vehicle Clusters")
print("=" * 80)

unique_clusters = sorted(np.unique(best_labels))
v_stats = {}

print(f"\n   Cluster Distribution:")
for vc in unique_clusters:
    mask = best_labels == vc
    n = mask.sum()
    pct = n / len(best_labels) * 100
    print(f"      V{vc}: {n:>6,} vehicles ({pct:>5.1f}%)")

    # 统计 - 使用原始 vehicle_features 数据进行分析
    vf_vc = vehicle_features[mask]
    
    comp_mean = vf_vc[cluster_ratio_cols].mean()
    # 从 cluster ratios 计算 (避免依赖冗余特征列)
    high_energy = comp_mean['cluster_2_ratio'] + comp_mean['cluster_3_ratio']
    idle_dominant = comp_mean['cluster_0_ratio']
    
    v_stats[vc] = {
        'size': int(n),
        'pct': float(pct),
        'composition': {f'C{c}': float(comp_mean[f'cluster_{c}_ratio']) 
                       for c in range(4)},
        'high_energy_ratio': float(high_energy),
        'idle_dominant_ratio': float(idle_dominant),
    }

# 自动标记
print(f"\n   Cluster Labels (Auto-generated):")
for vc in unique_clusters:
    comp = v_stats[vc]['composition']
    high_e = v_stats[vc]['high_energy_ratio']
    idle_d = v_stats[vc]['idle_dominant_ratio']
    
    # 根据特征自动标记
    if comp['C2'] > 0.25:
        label = "Highway-focused"
    elif comp['C1'] > 0.40:
        label = "City-mixed"
    elif idle_d > 0.60:
        label = "Parking-heavy"
    elif high_e > 0.35:
        label = "Aggressive-mixed"
    else:
        label = "Balanced"
    
    v_stats[vc]['label'] = label
    print(f"      V{vc}: {label:<20} | C0:{comp['C0']:>5.1%} C1:{comp['C1']:>5.1%} C2:{comp['C2']:>5.1%} C3:{comp['C3']:>5.1%}")

# ============================================================
# 6. 可视化
# ============================================================
print(f"\n【STEP 6】Visualization")
print("=" * 80)

# 6.1 GMM 聚类质量
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Silhouette
ax = axes[0]
ax.text(0.5, 0.7, f'{best_score:.4f}', ha='center', va='center', 
       fontsize=48, fontweight='bold', color='#2ecc71')
ax.text(0.5, 0.3, 'Silhouette Score', ha='center', va='center',
       fontsize=14, fontweight='bold')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.axis('off')

# Calinski-Harabasz
ax = axes[1]
ax.text(0.5, 0.7, f'{best_ch:.1f}', ha='center', va='center', 
       fontsize=48, fontweight='bold', color='#3498db')
ax.text(0.5, 0.3, 'Calinski-Harabasz', ha='center', va='center',
       fontsize=14, fontweight='bold')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.axis('off')

# Davies-Bouldin
ax = axes[2]
ax.text(0.5, 0.7, f'{best_db:.4f}', ha='center', va='center', 
       fontsize=48, fontweight='bold', color='#e74c3c')
ax.text(0.5, 0.3, 'Davies-Bouldin (↓)', ha='center', va='center',
       fontsize=14, fontweight='bold')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.axis('off')

plt.suptitle(f'GMM Clustering (K={K}, cov_type={best_cov_type})', 
            fontweight='bold', fontsize=13)
plt.tight_layout()

for fmt, dpi in [('.png', 300), ('.pdf', None)]:
    path = os.path.join(CONFIG['save_dir'], f'gmm_clustering_metrics{fmt}')
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: gmm_clustering_metrics.png/pdf")

# 6.2 PCA 可视化
pca_viz = PCA(n_components=2, random_state=CONFIG['seed'])
X_pca = pca_viz.fit_transform(X_active)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# PCA 散点图
colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))
for vi, vc in enumerate(unique_clusters):
    mask = best_labels == vc
    label = v_stats[vc]['label']
    ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], 
               c=[colors[vi]], s=50, alpha=0.6, 
               label=f'V{vc}: {label} (n={mask.sum()})',
               edgecolors='black', linewidth=0.5)

ev = pca_viz.explained_variance_ratio_
ax1.set_xlabel(f'PC1 ({ev[0]:.1%})', fontweight='bold', fontsize=11)
ax1.set_ylabel(f'PC2 ({ev[1]:.1%})', fontweight='bold', fontsize=11)
ax1.set_title('(a) Vehicle Clusters (PCA)', fontweight='bold', fontsize=12)
ax1.legend(fontsize=10, loc='best')
ax1.grid(True, alpha=0.2)

# 簇大小
sizes = [v_stats[vc]['size'] for vc in unique_clusters]
labels_list = [f"V{vc}\n{v_stats[vc]['label']}" for vc in unique_clusters]
bars = ax2.bar(range(len(unique_clusters)), sizes, color=colors, 
              edgecolor='black', linewidth=1.5)
for i, (bar, s) in enumerate(zip(bars, sizes)):
    ax2.text(bar.get_x() + bar.get_width() / 2, s + max(sizes) * 0.02,
            f'{s:,}', ha='center', fontsize=11, fontweight='bold')
ax2.set_xticks(range(len(unique_clusters)))
ax2.set_xticklabels(labels_list, fontsize=11)
ax2.set_ylabel('Number of Vehicles', fontweight='bold', fontsize=11)
ax2.set_title('(b) Cluster Sizes', fontweight='bold', fontsize=12)
ax2.grid(True, alpha=0.2, axis='y')

plt.tight_layout()
for fmt, dpi in [('.png', 300), ('.pdf', None)]:
    path = os.path.join(CONFIG['save_dir'], f'gmm_clustering_overview{fmt}')
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: gmm_clustering_overview.png/pdf")

# 6.3 聚类特征热力图
fig, axes = plt.subplots(1, len(unique_clusters), figsize=(4*len(unique_clusters), 5))
if len(unique_clusters) == 1:
    axes = [axes]

for vi, vc in enumerate(unique_clusters):
    ax = axes[vi]
    
    comp = v_stats[vc]['composition']
    comp_values = [comp[f'C{c}'] for c in range(4)]
    
    bars = ax.bar(['C0\nLong\nIdle', 'C1\nMixed\nDriving', 'C2\nHighway\nDriving', 'C3\nMixed\nDriving'],
                  comp_values, color=['#5B9BD5', '#70AD47', '#C0504D', '#FFC000'],
                  edgecolor='black', linewidth=1.5)
    
    for i, v in enumerate(comp_values):
        ax.text(i, v + 0.02, f'{v:.1%}', ha='center', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Composition Ratio', fontweight='bold')
    ax.set_title(f"V{vc}: {v_stats[vc]['label']}\n(n={v_stats[vc]['size']:,})", 
                fontweight='bold', fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.2, axis='y')

plt.tight_layout()
for fmt, dpi in [('.png', 300), ('.pdf', None)]:
    path = os.path.join(CONFIG['save_dir'], f'gmm_cluster_composition{fmt}')
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: gmm_cluster_composition.png/pdf")

# ============================================================
# 7. 保存结果
# ============================================================
print(f"\n【STEP 7】Saving Results")
print("=" * 80)

# 7.1 添加聚类标签到原始数据
vehicle_features['vehicle_cluster'] = best_labels
vehicle_features['cluster_label'] = vehicle_features['vehicle_cluster'].map(
    {vc: v_stats[vc]['label'] for vc in unique_clusters}
)

vehicle_results_path = os.path.join(CONFIG['save_dir'], 'vehicle_clustering_gmm_k4.csv')
vehicle_features.to_csv(vehicle_results_path, index=False)
print(f"   ✓ vehicle_clustering_gmm_k4.csv ({len(vehicle_features):,} vehicles)")

# 7.2 NPZ 格式
np.savez(os.path.join(CONFIG['save_dir'], 'vehicle_clustering_gmm_k4.npz'),
         X=X.values, X_scaled=X_scaled, best_labels=best_labels,
         vehicle_ids=vehicle_features['vehicle_id'].values,
         X_pca=X_pca, feature_names=np.array(active_feature_cols),
         gmm_model=best_gmm)
print(f"   ✓ vehicle_clustering_gmm_k4.npz")

# 7.3 摘要
k_selection_info = {}
for k_val, r in k_results.items():
    k_selection_info[str(k_val)] = {
        'silhouette': float(r['sil']),
        'calinski_harabasz': float(r['ch']),
        'davies_bouldin': float(r['db']),
        'bic': float(r['bic']),
        'cov_type': r['cov_type'],
    }

summary = {
    'timestamp': pd.Timestamp.now().isoformat(),
    'method': 'GMM',
    'n_clusters': K,
    'covariance_type': best_cov_type,
    'silhouette_score': float(best_score),
    'calinski_harabasz_score': float(best_ch),
    'davies_bouldin_score': float(best_db),
    'n_vehicles': len(vehicle_features),
    'n_features': len(active_feature_cols),
    'pca_dims': int(n_pca_dims),
    'pca_variance_retained': float(pca_var_retained),
    'k_selection': k_selection_info,
    'cluster_stats': {str(k): v for k, v in v_stats.items()},
    'feature_names': active_feature_cols,
}

summary_path = os.path.join(CONFIG['save_dir'], 'vehicle_clustering_gmm_k4_summary.json')
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2, default=str)
print(f"   ✓ vehicle_clustering_gmm_k4_summary.json")

# ============================================================
# 完成
# ============================================================
print("\n" + "=" * 80)
print("✅ STEP 8 COMPLETE!")
print("=" * 80)

print(f"""
Summary:
  - Vehicles: {len(vehicle_features):,}
  - Clusters: {K} (auto-selected from {CONFIG['k_range'].start}-{CONFIG['k_range'].stop - 1})
  - Method: GMM
  - Covariance Type: {best_cov_type}
  - PCA Dims: {n_pca_dims} (variance: {pca_var_retained:.2%})
  - Silhouette Score: {best_score:.4f}
  - Calinski-Harabasz: {best_ch:.1f}
  - Davies-Bouldin: {best_db:.4f}

Cluster Breakdown:
""")

for vc in sorted(unique_clusters):
    comp = v_stats[vc]['composition']
    print(f"  V{vc} ({v_stats[vc]['label']:<20}): {v_stats[vc]['size']:>6,} vehicles ({v_stats[vc]['pct']:>5.1f}%)")
    print(f"     C0:{comp['C0']:>5.1%} C1:{comp['C1']:>5.1%} C2:{comp['C2']:>5.1%} C3:{comp['C3']:>5.1%}")

print(f"""
Output Files:
  1. vehicle_clustering_gmm_k4.csv
  2. vehicle_clustering_gmm_k4.npz
  3. vehicle_clustering_gmm_k4_summary.json
  4. gmm_clustering_metrics.png/pdf
  5. gmm_clustering_overview.png/pdf
  6. gmm_cluster_composition.png/pdf

Next Steps:
  - Use vehicle clusters for charging analysis
  - Integrate with inter_charge_trips data
  - Run coupling analysis (Step 3.3)
""")

print("=" * 80)