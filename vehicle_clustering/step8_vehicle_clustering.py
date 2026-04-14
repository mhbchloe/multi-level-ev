"""
Step 8: Vehicle-Level Clustering (GMM K=4)
使用GMM，固定K=4，完成车辆级聚类
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
from sklearn.preprocessing import StandardScaler
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
print("🚀 STEP 8: VEHICLE-LEVEL CLUSTERING (GMM K=4)")
print("=" * 80)

# ============================================================
# 配置
# ============================================================
CONFIG = {
    'vehicle_features_path': './coupling_analysis/results/vehicles_aggregated_features.csv',
    'segments_path': './coupling_analysis/results/segments_integrated_complete.csv',
    'save_dir': './vehicle_clustering/results/',
    'seed': 42,
    'n_clusters': 4,  # 固定 K=4
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

# 选择用于聚类的特征（三维特征体系）
# ① 分布 (Distribution)
cluster_ratio_cols = [f'cluster_{c}_ratio' for c in range(4)]
distribution_cols = cluster_ratio_cols + ['mode_diversity']

# ② 转移 (Transition)
transition_cols = [f'trans_{i}_to_{j}' for i in range(4) for j in range(4)]
transition_cols += ['mode_switch_rate', 'transition_entropy', 'self_loop_ratio']

# ③ 演化 (Evolution)
evolution_cols = [
    'mode_drift', 'soc_trend',                    # 时序累积
    'mode_autocorr_lag1',                          # 节奏
    'interval_regularity', 'interval_cv',
    'temporal_concentration',
    'mode_entropy_stability', 'avg_run_length',    # 稳定性
]

# 辅助特征
behavior_cols = ['high_energy_ratio', 'idle_dominant_ratio']
phys_cols = [c for c in vehicle_features.columns if c.startswith('avg_')]

feature_cols = distribution_cols + transition_cols + evolution_cols + behavior_cols + list(phys_cols)

print(f"\n   Feature Selection (Three-Dimension Framework):")
print(f"      ① Distribution: {len([f for f in distribution_cols if f in vehicle_features.columns])} features")
print(f"      ② Transition:   {len([f for f in transition_cols if f in vehicle_features.columns])} features")
print(f"      ③ Evolution:    {len([f for f in evolution_cols if f in vehicle_features.columns])} features")
print(f"      Behavior:       {len([f for f in behavior_cols if f in vehicle_features.columns])} features")
print(f"      Physical:       {len([f for f in phys_cols if f in vehicle_features.columns])} features")

# 只使用数据中实际存在的特征
feature_cols = [f for f in feature_cols if f in vehicle_features.columns]
missing_cols = set(distribution_cols + transition_cols + evolution_cols) - set(feature_cols)
if missing_cols:
    print(f"\n   ⚠️  Missing features (run integrate_clustering_complete.py first): {missing_cols}")
print(f"      Total available: {len(feature_cols)} features")

# 提取特征矩阵
X = vehicle_features[feature_cols].copy()

# 处理缺失值
missing = X.isna().sum().sum()
if missing > 0:
    print(f"\n   ⚠️  Found {missing} missing values, filling with 0...")
    X = X.fillna(0)

X = X.astype(np.float32)
print(f"\n   Feature matrix: {X.shape}")

# ============================================================
# 3. 标准化
# ============================================================
print(f"\n【STEP 3】Feature Standardization")
print("=" * 80)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

print(f"   ✓ Standardized: mean=0, std=1")

# 移除零方差特征
var = X_scaled.var(axis=0)
active = var > 1e-8
n_removed = (~active).sum()
if n_removed > 0:
    print(f"   ⚠️  Removing {n_removed} zero-variance features")
    X_cluster = X_scaled[:, active]
    active_feature_cols = [feature_cols[i] for i in range(len(feature_cols)) if active[i]]
else:
    X_cluster = X_scaled
    active_feature_cols = feature_cols

print(f"   ✓ Active features: {X_cluster.shape[1]}")

# ============================================================
# 4. GMM 聚类 (K=4)
# ============================================================
print(f"\n【STEP 4】GMM Clustering (K={CONFIG['n_clusters']})")
print("=" * 80)

K = CONFIG['n_clusters']
best_labels = None
best_score = -1
best_cov_type = None

print(f"\n   Testing covariance types...")

for cov_type in ['full', 'tied', 'diag', 'spherical']:
    print(f"\n   Testing cov_type='{cov_type}'...", end=" ")
    try:
        gmm = GaussianMixture(
            n_components=K, 
            covariance_type=cov_type,
            n_init=10,  # 更多初始化
            random_state=CONFIG['seed'], 
            max_iter=500,  # 更多迭代
            reg_covar=1e-4,
            verbose=0
        )
        gmm.fit(X_cluster)
        labels = gmm.predict(X_cluster)
        
        # 计算评分
        sil = silhouette_score(X_cluster, labels, 
                              sample_size=min(5000, len(X_cluster)), 
                              random_state=CONFIG['seed'])
        ch = calinski_harabasz_score(X_cluster, labels)
        db = davies_bouldin_score(X_cluster, labels)
        
        print(f"✓ Sil={sil:.4f}, CH={ch:.1f}, DB={db:.4f}")
        
        if sil > best_score:
            best_score = sil
            best_labels = labels
            best_cov_type = cov_type
            best_gmm = gmm
            best_ch = ch
            best_db = db
    
    except Exception as e:
        print(f"✗ Failed: {str(e)[:50]}")

print(f"\n   🏆 Best: cov_type='{best_cov_type}' (Silhouette={best_score:.4f})")

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

    # 统计
    X_vc = X[mask]
    
    comp_mean = X_vc[cluster_ratio_cols].mean()
    high_energy = X_vc['high_energy_ratio'].mean()
    idle_dominant = X_vc['idle_dominant_ratio'].mean()
    
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

plt.suptitle(f'GMM Clustering (K=4, cov_type={best_cov_type})', 
            fontweight='bold', fontsize=13)
plt.tight_layout()

for fmt, dpi in [('.png', 300), ('.pdf', None)]:
    path = os.path.join(CONFIG['save_dir'], f'gmm_clustering_metrics{fmt}')
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: gmm_clustering_metrics.png/pdf")

# 6.2 PCA 可视化
pca = PCA(n_components=2, random_state=CONFIG['seed'])
X_pca = pca.fit_transform(X_scaled)

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

ev = pca.explained_variance_ratio_
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
         X=X, X_scaled=X_scaled, best_labels=best_labels,
         vehicle_ids=vehicle_features['vehicle_id'].values,
         X_pca=X_pca, feature_names=np.array(active_feature_cols),
         gmm_model=best_gmm)
print(f"   ✓ vehicle_clustering_gmm_k4.npz")

# 7.3 摘要
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
  - Clusters: {K}
  - Method: GMM
  - Covariance Type: {best_cov_type}
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