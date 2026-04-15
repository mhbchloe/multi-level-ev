"""
Step 8 Improved: Vehicle-Level Clustering with 3D Feature Framework
改进版车辆聚类：整合 分布(Distribution)、转移(Transition)、演化(Evolution) 三个维度

三维特征框架:
  ① 分布 (4 features): 四种驾驶模式(C0-C3)各占多少比例
  ② 转移 (16+3 features): 4×4转移矩阵 + 转移熵 + 模式切换率 + 稳定性指数
  ③ 演化 (~8 features): 时序特性：序列长度、切换频率、稳定性、累积时间、节奏规律性、模式熵
"""

import numpy as np
import pandas as pd
import os
import json
import warnings
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
from sklearn.preprocessing import RobustScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from scipy.stats import entropy as sp_entropy
import seaborn as sns
from tqdm import tqdm

warnings.filterwarnings('ignore')

rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 11
rcParams['figure.dpi'] = 150

print("=" * 80)
print("🚀 STEP 8 IMPROVED: VEHICLE CLUSTERING WITH 3D FEATURES")
print("   (Distribution + Transition + Evolution)")
print("=" * 80)

# ============================================================
# 配置
# ============================================================
CONFIG = {
    'segments_path': './coupling_analysis/results/segments_integrated_complete.csv',
    'vehicle_features_path': './coupling_analysis/results/vehicles_aggregated_features.csv',
    'save_dir': './vehicle_clustering/results/',
    'seed': 42,
    'n_modes': 4,       # C0-C3
    'k_range': (2, 8),  # K range for evaluation
}

os.makedirs(CONFIG['save_dir'], exist_ok=True)

# ============================================================
# 1. 加载数据
# ============================================================
print(f"\n{'='*80}")
print(f"【STEP 1】Loading Data")
print("=" * 80)

segments_df = pd.read_csv(CONFIG['segments_path'])
print(f"   ✓ Segments: {len(segments_df):,} rows, {segments_df['vehicle_id'].nunique():,} vehicles")

# Load existing vehicle features as baseline for comparison
vehicle_baseline = None
if os.path.exists(CONFIG['vehicle_features_path']):
    vehicle_baseline = pd.read_csv(CONFIG['vehicle_features_path'])
    print(f"   ✓ Baseline vehicle features: {len(vehicle_baseline):,} vehicles")

# Ensure required columns
required_cols = ['vehicle_id', 'cluster_id']
for col in required_cols:
    if col not in segments_df.columns:
        raise ValueError(f"Missing required column: {col}")

# Sort segments by vehicle and time
if 'start_dt' in segments_df.columns:
    segments_df['start_dt'] = pd.to_datetime(segments_df['start_dt'])
    segments_df = segments_df.sort_values(['vehicle_id', 'start_dt']).reset_index(drop=True)
elif 'segment_id' in segments_df.columns:
    segments_df = segments_df.sort_values(['vehicle_id', 'segment_id']).reset_index(drop=True)

N_MODES = CONFIG['n_modes']

print(f"\n   Cluster distribution in segments:")
for c in range(N_MODES):
    count = (segments_df['cluster_id'] == c).sum()
    print(f"      C{c}: {count:>10,} ({count/len(segments_df)*100:>5.2f}%)")


# ============================================================
# 2. 计算三维特征
# ============================================================
print(f"\n{'='*80}")
print(f"【STEP 2】Computing 3D Features (Distribution + Transition + Evolution)")
print("=" * 80)


def compute_3d_features(vehicle_id, v_group, n_modes=4):
    """计算单个车辆的三维特征."""
    clusters = v_group['cluster_id'].values.astype(int)
    n_segs = len(clusters)

    feat = {'vehicle_id': vehicle_id}

    # ============================================================
    # ① 分布维度 (Distribution): 4 features
    # ============================================================
    cluster_dist = pd.Series(clusters).value_counts(normalize=True).to_dict()
    for c in range(n_modes):
        feat[f'cluster_{c}_ratio'] = cluster_dist.get(c, 0.0)

    # Mode diversity (entropy of distribution)
    ratios = [feat[f'cluster_{c}_ratio'] for c in range(n_modes)]
    ratios_arr = np.array(ratios)
    ratios_arr = ratios_arr[ratios_arr > 0]
    feat['mode_diversity'] = float(sp_entropy(ratios_arr)) if len(ratios_arr) > 0 else 0.0

    # ============================================================
    # ② 转移维度 (Transition): 16 + 3 features
    # ============================================================
    # 4×4 transition matrix (normalized)
    trans_matrix = np.zeros((n_modes, n_modes))
    if n_segs > 1:
        for i in range(n_segs - 1):
            from_c = clusters[i]
            to_c = clusters[i + 1]
            if 0 <= from_c < n_modes and 0 <= to_c < n_modes:
                trans_matrix[from_c, to_c] += 1

        # Normalize to probabilities
        total_transitions = trans_matrix.sum()
        if total_transitions > 0:
            trans_prob = trans_matrix / total_transitions
        else:
            trans_prob = np.zeros((n_modes, n_modes))
    else:
        trans_prob = np.zeros((n_modes, n_modes))

    # Store all 16 transition probabilities
    for i in range(n_modes):
        for j in range(n_modes):
            feat[f'trans_{i}_to_{j}'] = trans_prob[i, j]

    # Transition entropy: uncertainty of transitions
    trans_flat = trans_prob.flatten()
    trans_flat_pos = trans_flat[trans_flat > 0]
    feat['transition_entropy'] = float(sp_entropy(trans_flat_pos)) if len(trans_flat_pos) > 0 else 0.0

    # Mode switch rate: how often the mode changes
    if n_segs > 1:
        switches = sum(1 for i in range(n_segs - 1) if clusters[i] != clusters[i + 1])
        feat['mode_switch_rate'] = switches / (n_segs - 1)
    else:
        feat['mode_switch_rate'] = 0.0

    # Self-loop ratio: probability of staying in the same mode
    self_loop_total = sum(trans_matrix[i, i] for i in range(n_modes))
    feat['self_loop_ratio'] = self_loop_total / max(total_transitions, 1) if n_segs > 1 else 1.0

    # ============================================================
    # ③ 演化维度 (Evolution): ~8-10 features
    # ============================================================

    # 3.1 Sequence length
    feat['sequence_length'] = n_segs

    # 3.2 Mode switching frequency (switches / total length)
    feat['mode_switching_freq'] = feat['mode_switch_rate']  # same as switch rate

    # 3.3 Pattern stability: max consecutive same-mode run / total segments
    if n_segs > 0:
        max_run = 1
        current_run = 1
        for i in range(1, n_segs):
            if clusters[i] == clusters[i - 1]:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 1
        feat['pattern_stability'] = max_run / n_segs
    else:
        feat['pattern_stability'] = 0.0

    # 3.4 Average run length: average consecutive same-mode segments
    if n_segs > 0:
        runs = []
        current_run = 1
        for i in range(1, n_segs):
            if clusters[i] == clusters[i - 1]:
                current_run += 1
            else:
                runs.append(current_run)
                current_run = 1
        runs.append(current_run)
        feat['avg_run_length'] = np.mean(runs)
    else:
        feat['avg_run_length'] = 0.0

    # 3.5 Cumulative time per mode (from duration_seconds if available)
    if 'duration_seconds' in v_group.columns:
        total_duration = v_group['duration_seconds'].sum()
        for c in range(n_modes):
            mode_mask = clusters == c
            mode_duration = v_group['duration_seconds'].values[mode_mask].sum()
            feat[f'cumulative_time_C{c}'] = mode_duration / max(total_duration, 1)
    else:
        for c in range(n_modes):
            feat[f'cumulative_time_C{c}'] = feat[f'cluster_{c}_ratio']

    # 3.6 Rhythm regularity: proportion of repeated adjacent mode pairs
    if n_segs > 2:
        pairs = [(clusters[i], clusters[i + 1]) for i in range(n_segs - 1)]
        pair_counts = {}
        for p in pairs:
            pair_counts[p] = pair_counts.get(p, 0) + 1
        total_pairs = len(pairs)
        repeated_pairs = sum(1 for count in pair_counts.values() if count > 1)
        feat['rhythm_regularity'] = repeated_pairs / max(len(pair_counts), 1)
    else:
        feat['rhythm_regularity'] = 0.0

    # 3.7 Mode entropy: Shannon entropy of mode distribution
    feat['mode_entropy'] = feat['mode_diversity']  # same computation

    # 3.8 Dominant mode fraction: fraction of the most frequent mode
    if n_segs > 0:
        mode_counts = np.bincount(clusters, minlength=n_modes)
        feat['dominant_mode_fraction'] = mode_counts.max() / n_segs
    else:
        feat['dominant_mode_fraction'] = 0.0

    return feat


# Compute features for all vehicles
print(f"\n   Computing 3D features for each vehicle...")
vehicle_features_list = []

for vehicle_id, v_group in tqdm(segments_df.groupby('vehicle_id'),
                                desc="   🔄 Feature extraction", ncols=80):
    feat = compute_3d_features(vehicle_id, v_group, N_MODES)
    vehicle_features_list.append(feat)

vehicle_3d_df = pd.DataFrame(vehicle_features_list)
print(f"   ✓ Computed features for {len(vehicle_3d_df):,} vehicles")

# ============================================================
# 3. Feature Engineering & Selection
# ============================================================
print(f"\n{'='*80}")
print(f"【STEP 3】Feature Engineering & Selection")
print("=" * 80)

# Define feature groups
distribution_cols = [f'cluster_{c}_ratio' for c in range(N_MODES)] + ['mode_diversity']
transition_cols = [f'trans_{i}_to_{j}' for i in range(N_MODES) for j in range(N_MODES)] + \
                  ['transition_entropy', 'mode_switch_rate', 'self_loop_ratio']
evolution_cols = ['sequence_length', 'mode_switching_freq', 'pattern_stability',
                  'avg_run_length', 'rhythm_regularity', 'mode_entropy',
                  'dominant_mode_fraction']

# Add cumulative time columns
cumulative_time_cols = [f'cumulative_time_C{c}' for c in range(N_MODES)]
evolution_cols += cumulative_time_cols

# All feature columns for clustering
all_feature_cols = distribution_cols + transition_cols + evolution_cols

# Verify all columns exist
missing_cols = [c for c in all_feature_cols if c not in vehicle_3d_df.columns]
if missing_cols:
    print(f"   ⚠️  Missing columns (will be filled with 0): {missing_cols}")
    for c in missing_cols:
        vehicle_3d_df[c] = 0.0

print(f"\n   Feature Dimensions:")
print(f"      ① Distribution: {len(distribution_cols)} features")
print(f"         {distribution_cols}")
print(f"      ② Transition:   {len(transition_cols)} features")
print(f"         {transition_cols[:5]}... + {len(transition_cols)-5} more")
print(f"      ③ Evolution:    {len(evolution_cols)} features")
print(f"         {evolution_cols}")
print(f"      Total: {len(all_feature_cols)} features")

# Extract feature matrix
X = vehicle_3d_df[all_feature_cols].copy()
X = X.fillna(0).astype(np.float64)

print(f"\n   Feature matrix: {X.shape}")
print(f"   Missing values after fill: {X.isna().sum().sum()}")

# ============================================================
# 4. Standardization
# ============================================================
print(f"\n{'='*80}")
print(f"【STEP 4】Feature Standardization")
print("=" * 80)

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

# Remove zero-variance features
var = X_scaled.var(axis=0)
active = var > 1e-8
n_removed = (~active).sum()
if n_removed > 0:
    print(f"   ⚠️  Removing {n_removed} zero-variance features:")
    removed_features = [all_feature_cols[i] for i in range(len(all_feature_cols)) if not active[i]]
    print(f"      {removed_features}")
    X_cluster = X_scaled[:, active]
    active_feature_cols = [all_feature_cols[i] for i in range(len(all_feature_cols)) if active[i]]
else:
    X_cluster = X_scaled
    active_feature_cols = all_feature_cols.copy()

print(f"   ✓ Active features after filtering: {X_cluster.shape[1]}")
print(f"   ✓ Standardized with RobustScaler")

# ============================================================
# 5. Optimal K Selection
# ============================================================
print(f"\n{'='*80}")
print(f"【STEP 5】Optimal K Selection (K={CONFIG['k_range'][0]}..{CONFIG['k_range'][1]-1})")
print("=" * 80)

k_results = {}
k_min, k_max = CONFIG['k_range']

for k in range(k_min, k_max):
    best_sil_k = -1
    best_result_k = None

    for cov_type in ['full', 'tied', 'diag', 'spherical']:
        try:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type=cov_type,
                n_init=10,
                random_state=CONFIG['seed'],
                max_iter=500,
                reg_covar=1e-4,
            )
            gmm.fit(X_cluster)
            labels = gmm.predict(X_cluster)

            # Skip if only 1 cluster is actually used
            if len(np.unique(labels)) < 2:
                continue

            sil = silhouette_score(X_cluster, labels,
                                   sample_size=min(5000, len(X_cluster)),
                                   random_state=CONFIG['seed'])
            ch = calinski_harabasz_score(X_cluster, labels)
            db = davies_bouldin_score(X_cluster, labels)

            if sil > best_sil_k:
                best_sil_k = sil
                best_result_k = {
                    'k': k,
                    'cov_type': cov_type,
                    'silhouette': sil,
                    'calinski_harabasz': ch,
                    'davies_bouldin': db,
                    'labels': labels,
                    'gmm': gmm,
                    'bic': gmm.bic(X_cluster),
                }
        except Exception:
            continue

    if best_result_k is not None:
        k_results[k] = best_result_k
        r = best_result_k
        print(f"   K={k}: Sil={r['silhouette']:.4f}, CH={r['calinski_harabasz']:.1f}, "
              f"DB={r['davies_bouldin']:.4f}, BIC={r['bic']:.1f} (cov={r['cov_type']})")

# Select best K by silhouette score
if not k_results:
    raise RuntimeError("No valid clustering found")

best_k = max(k_results, key=lambda k: k_results[k]['silhouette'])
best_result = k_results[best_k]
best_labels = best_result['labels']
best_gmm = best_result['gmm']

print(f"\n   🏆 Best K={best_k} (Silhouette={best_result['silhouette']:.4f})")

# ============================================================
# 6. Analyze Vehicle Clusters
# ============================================================
print(f"\n{'='*80}")
print(f"【STEP 6】Analyzing Vehicle Clusters (K={best_k})")
print("=" * 80)

unique_clusters = sorted(np.unique(best_labels))
v_stats = {}

print(f"\n   Cluster Distribution:")
for vc in unique_clusters:
    mask = best_labels == vc
    n = mask.sum()
    pct = n / len(best_labels) * 100
    print(f"      V{vc}: {n:>6,} vehicles ({pct:>5.1f}%)")

    # Compute cluster statistics
    X_vc = vehicle_3d_df[mask]

    comp_mean = {f'C{c}': float(X_vc[f'cluster_{c}_ratio'].mean()) for c in range(N_MODES)}
    trans_entropy_mean = float(X_vc['transition_entropy'].mean())
    mode_switch_mean = float(X_vc['mode_switch_rate'].mean())
    pattern_stability_mean = float(X_vc['pattern_stability'].mean())

    v_stats[vc] = {
        'size': int(n),
        'pct': float(pct),
        'composition': comp_mean,
        'transition_entropy': trans_entropy_mean,
        'mode_switch_rate': mode_switch_mean,
        'pattern_stability': pattern_stability_mean,
        'self_loop_ratio': float(X_vc['self_loop_ratio'].mean()),
        'avg_run_length': float(X_vc['avg_run_length'].mean()),
        'rhythm_regularity': float(X_vc['rhythm_regularity'].mean()),
    }

# Auto-label clusters
print(f"\n   Cluster Labels (Auto-generated):")
for vc in unique_clusters:
    comp = v_stats[vc]['composition']
    switch_rate = v_stats[vc]['mode_switch_rate']
    stability = v_stats[vc]['pattern_stability']

    characteristics = []
    if comp['C2'] > 0.25:
        characteristics.append("Highway-focused")
    if comp['C1'] > 0.40:
        characteristics.append("City-mixed")
    if comp['C0'] > 0.30:
        characteristics.append("Idle-prone")
    if comp['C3'] > 0.20:
        characteristics.append("Short-stop")
    if switch_rate > 0.7:
        characteristics.append("High-switching")
    if stability > 0.3:
        characteristics.append("Stable-pattern")

    if not characteristics:
        characteristics = ["Balanced"]

    label = " + ".join(characteristics[:2])
    v_stats[vc]['label'] = label
    print(f"      V{vc}: {label:<30} | C0:{comp['C0']:>5.1%} C1:{comp['C1']:>5.1%} "
          f"C2:{comp['C2']:>5.1%} C3:{comp['C3']:>5.1%} | Switch:{switch_rate:.2f} Stab:{stability:.2f}")

# ============================================================
# 7. Visualization
# ============================================================
print(f"\n{'='*80}")
print(f"【STEP 7】Visualization")
print("=" * 80)

K = best_k
colors_map = plt.cm.Set2(np.linspace(0, 1, max(K, 4)))[:K]

# --- 7.1 Distribution Dimension ---
print("\n   [7.1] Distribution Features...")
fig, axes = plt.subplots(1, K, figsize=(4 * K, 4.5), squeeze=False)
axes = axes[0]

colors_segment = ['#5B9BD5', '#70AD47', '#C0504D', '#FFC000']

for vi, vc in enumerate(unique_clusters):
    ax = axes[vi]
    vehicles_in_cluster = vehicle_3d_df[best_labels == vc]

    means = [vehicles_in_cluster[f'cluster_{c}_ratio'].mean() for c in range(N_MODES)]
    stds = [vehicles_in_cluster[f'cluster_{c}_ratio'].std() for c in range(N_MODES)]

    x_pos = np.arange(N_MODES)
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5,
                  color=colors_segment, edgecolor='black', linewidth=1.5,
                  alpha=0.8, error_kw={'linewidth': 2, 'ecolor': 'gray'})

    for i, (bar, m) in enumerate(zip(bars, means)):
        ax.text(bar.get_x() + bar.get_width() / 2, m + stds[i] + 0.03,
                f'{m:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=9)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(['C0\nLong Idle', 'C1\nUrban', 'C2\nHighway', 'C3\nShort Idle'], fontsize=8)
    ax.set_ylabel('Mean Ratio', fontweight='bold', fontsize=10)
    ax.set_title(f'V{vc}: {v_stats[vc]["label"]}\n(n={v_stats[vc]["size"]:,})',
                 fontweight='bold', fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.2, axis='y')

plt.suptitle('① Distribution Features: Segment Mode Composition',
             fontweight='bold', fontsize=13)
plt.tight_layout()
for fmt, dpi in [('.png', 300), ('.pdf', None)]:
    fig.savefig(os.path.join(CONFIG['save_dir'], f'improved_3d_distribution{fmt}'),
                dpi=dpi, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: improved_3d_distribution.png/pdf")

# --- 7.2 Transition Dimension ---
print("\n   [7.2] Transition Features (Heatmaps)...")
fig, axes = plt.subplots(1, K, figsize=(4.5 * K, 4.5), squeeze=False)
axes = axes[0]

for vi, vc in enumerate(unique_clusters):
    ax = axes[vi]
    vehicles_in_cluster = vehicle_3d_df[best_labels == vc]

    # Build average transition matrix
    T_avg = np.zeros((N_MODES, N_MODES))
    for i in range(N_MODES):
        for j in range(N_MODES):
            col = f'trans_{i}_to_{j}'
            T_avg[i, j] = vehicles_in_cluster[col].mean()

    # Row-normalize for display
    row_sums = T_avg.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    T_display = T_avg / row_sums

    sns.heatmap(T_display, annot=True, fmt='.3f', cmap='YlOrRd',
                ax=ax, cbar_kws={'label': 'Probability'},
                xticklabels=['C0', 'C1', 'C2', 'C3'],
                yticklabels=['C0', 'C1', 'C2', 'C3'],
                linewidths=1, linecolor='black', vmin=0, vmax=1)

    ax.set_title(f'V{vc}: {v_stats[vc]["label"]}\nEntropy={v_stats[vc]["transition_entropy"]:.3f}',
                 fontweight='bold', fontsize=10)
    ax.set_xlabel('To →', fontweight='bold')
    ax.set_ylabel('From ↓', fontweight='bold')

plt.suptitle('② Transition Features: Mode Transition Probabilities',
             fontweight='bold', fontsize=13)
plt.tight_layout()
for fmt, dpi in [('.png', 300), ('.pdf', None)]:
    fig.savefig(os.path.join(CONFIG['save_dir'], f'improved_3d_transition{fmt}'),
                dpi=dpi, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: improved_3d_transition.png/pdf")

# --- 7.3 Evolution Dimension ---
print("\n   [7.3] Evolution Features (Box plots)...")
evo_plot_features = [
    ('mode_switch_rate', 'Mode Switch Rate'),
    ('pattern_stability', 'Pattern Stability'),
    ('avg_run_length', 'Avg Run Length'),
    ('rhythm_regularity', 'Rhythm Regularity'),
    ('mode_entropy', 'Mode Entropy'),
    ('dominant_mode_fraction', 'Dominant Mode Frac.'),
]

n_evo = len(evo_plot_features)
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, (feat_col, feat_name) in enumerate(evo_plot_features):
    ax = axes[idx]

    data_for_box = []
    labels_for_box = []

    for vc in unique_clusters:
        mask = best_labels == vc
        data = vehicle_3d_df.loc[mask, feat_col].dropna().values
        if len(data) > 0:
            data_for_box.append(data)
            labels_for_box.append(f"V{vc}")

    if len(data_for_box) > 0:
        bp = ax.boxplot(data_for_box, labels=labels_for_box, patch_artist=True,
                        medianprops=dict(color='red', linewidth=2),
                        boxprops=dict(linewidth=1.5),
                        whiskerprops=dict(linewidth=1.2),
                        capprops=dict(linewidth=1.2),
                        flierprops=dict(markersize=3, alpha=0.5))

        for patch_idx, patch in enumerate(bp['boxes']):
            patch.set_facecolor(colors_map[patch_idx % len(colors_map)])
            patch.set_alpha(0.7)

        ax.set_ylabel(feat_name, fontweight='bold')
        ax.set_title(f'({chr(97 + idx)}) {feat_name}', fontweight='bold', fontsize=11)
        ax.grid(True, alpha=0.2, axis='y')

for idx in range(n_evo, len(axes)):
    axes[idx].axis('off')

plt.suptitle('③ Evolution Features: Temporal Driving Characteristics',
             fontweight='bold', fontsize=13)
plt.tight_layout()
for fmt, dpi in [('.png', 300), ('.pdf', None)]:
    fig.savefig(os.path.join(CONFIG['save_dir'], f'improved_3d_evolution{fmt}'),
                dpi=dpi, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: improved_3d_evolution.png/pdf")

# --- 7.4 PCA Visualization ---
print("\n   [7.4] PCA Visualization...")
pca = PCA(n_components=min(3, X_cluster.shape[1]), random_state=CONFIG['seed'])
X_pca = pca.fit_transform(X_cluster)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

for vi, vc in enumerate(unique_clusters):
    mask = best_labels == vc
    label = v_stats[vc]['label']
    ax1.scatter(X_pca[mask, 0], X_pca[mask, 1],
                c=[colors_map[vi]], s=50, alpha=0.6,
                label=f'V{vc}: {label} (n={mask.sum()})',
                edgecolors='black', linewidth=0.5)

ev = pca.explained_variance_ratio_
ax1.set_xlabel(f'PC1 ({ev[0]:.1%})', fontweight='bold', fontsize=11)
ax1.set_ylabel(f'PC2 ({ev[1]:.1%})', fontweight='bold', fontsize=11)
ax1.set_title('(a) Vehicle Clusters (PCA, 3D Features)', fontweight='bold', fontsize=12)
ax1.legend(fontsize=9, loc='best')
ax1.grid(True, alpha=0.2)

# Cluster sizes bar chart
sizes = [v_stats[vc]['size'] for vc in unique_clusters]
labels_list = [f"V{vc}\n{v_stats[vc]['label']}" for vc in unique_clusters]
bars = ax2.bar(range(len(unique_clusters)), sizes, color=colors_map,
               edgecolor='black', linewidth=1.5)
for i, (bar, s) in enumerate(zip(bars, sizes)):
    ax2.text(bar.get_x() + bar.get_width() / 2, s + max(sizes) * 0.02,
             f'{s:,}', ha='center', fontsize=11, fontweight='bold')
ax2.set_xticks(range(len(unique_clusters)))
ax2.set_xticklabels(labels_list, fontsize=9)
ax2.set_ylabel('Number of Vehicles', fontweight='bold', fontsize=11)
ax2.set_title('(b) Cluster Sizes', fontweight='bold', fontsize=12)
ax2.grid(True, alpha=0.2, axis='y')

plt.suptitle(f'Improved 3D Clustering Overview (K={best_k})', fontweight='bold', fontsize=13)
plt.tight_layout()
for fmt, dpi in [('.png', 300), ('.pdf', None)]:
    fig.savefig(os.path.join(CONFIG['save_dir'], f'improved_3d_overview{fmt}'),
                dpi=dpi, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: improved_3d_overview.png/pdf")

# --- 7.5 K Selection Plot ---
print("\n   [7.5] K Selection Plot...")
if len(k_results) > 1:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ks = sorted(k_results.keys())
    sils = [k_results[k]['silhouette'] for k in ks]
    chs = [k_results[k]['calinski_harabasz'] for k in ks]
    dbs = [k_results[k]['davies_bouldin'] for k in ks]

    axes[0].plot(ks, sils, 'o-', color='#2ecc71', linewidth=2, markersize=8)
    axes[0].axvline(best_k, color='red', linestyle='--', alpha=0.7, label=f'Best K={best_k}')
    axes[0].set_xlabel('K', fontweight='bold')
    axes[0].set_ylabel('Silhouette Score', fontweight='bold')
    axes[0].set_title('(a) Silhouette Score (↑)', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.2)

    axes[1].plot(ks, chs, 'o-', color='#3498db', linewidth=2, markersize=8)
    axes[1].axvline(best_k, color='red', linestyle='--', alpha=0.7, label=f'Best K={best_k}')
    axes[1].set_xlabel('K', fontweight='bold')
    axes[1].set_ylabel('Calinski-Harabasz', fontweight='bold')
    axes[1].set_title('(b) Calinski-Harabasz Index (↑)', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.2)

    axes[2].plot(ks, dbs, 'o-', color='#e74c3c', linewidth=2, markersize=8)
    axes[2].axvline(best_k, color='red', linestyle='--', alpha=0.7, label=f'Best K={best_k}')
    axes[2].set_xlabel('K', fontweight='bold')
    axes[2].set_ylabel('Davies-Bouldin', fontweight='bold')
    axes[2].set_title('(c) Davies-Bouldin Index (↓)', fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.2)

    plt.suptitle('Clustering Quality vs. Number of Clusters', fontweight='bold', fontsize=13)
    plt.tight_layout()
    for fmt, dpi in [('.png', 300), ('.pdf', None)]:
        fig.savefig(os.path.join(CONFIG['save_dir'], f'improved_3d_k_selection{fmt}'),
                    dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: improved_3d_k_selection.png/pdf")

# ============================================================
# 8. Save Results
# ============================================================
print(f"\n{'='*80}")
print(f"【STEP 8】Saving Results")
print("=" * 80)

# 8.1 Add cluster labels to vehicle dataframe
vehicle_3d_df['vehicle_cluster'] = best_labels
vehicle_3d_df['cluster_label'] = vehicle_3d_df['vehicle_cluster'].map(
    {vc: v_stats[vc]['label'] for vc in unique_clusters}
)

# 8.2 Save CSV
csv_path = os.path.join(CONFIG['save_dir'], 'vehicle_clustering_improved_3d.csv')
vehicle_3d_df.to_csv(csv_path, index=False)
print(f"   ✓ vehicle_clustering_improved_3d.csv ({len(vehicle_3d_df):,} vehicles)")

# 8.3 Save feature matrix as NPZ
npz_path = os.path.join(CONFIG['save_dir'], 'features_3d_matrix.npz')
np.savez(npz_path,
         X=X.values,
         X_scaled=X_scaled,
         X_cluster=X_cluster,
         labels=best_labels,
         vehicle_ids=vehicle_3d_df['vehicle_id'].values,
         feature_names=np.array(all_feature_cols),
         active_feature_names=np.array(active_feature_cols),
         distribution_cols=np.array(distribution_cols),
         transition_cols=np.array(transition_cols),
         evolution_cols=np.array(evolution_cols),
         X_pca=X_pca,
         pca_explained_variance=pca.explained_variance_ratio_)
print(f"   ✓ features_3d_matrix.npz")

# 8.4 Save summary JSON


class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types in JSON serialization."""

    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


summary = {
    'timestamp': pd.Timestamp.now().isoformat(),
    'method': 'GMM (Improved 3D)',
    'n_clusters': int(best_k),
    'covariance_type': best_result['cov_type'],
    'silhouette_score': float(best_result['silhouette']),
    'calinski_harabasz_score': float(best_result['calinski_harabasz']),
    'davies_bouldin_score': float(best_result['davies_bouldin']),
    'bic': float(best_result['bic']),
    'n_vehicles': len(vehicle_3d_df),
    'n_features_total': len(all_feature_cols),
    'n_features_active': len(active_feature_cols),
    'feature_dimensions': {
        'distribution': {
            'count': len(distribution_cols),
            'features': distribution_cols,
        },
        'transition': {
            'count': len(transition_cols),
            'features': transition_cols,
        },
        'evolution': {
            'count': len(evolution_cols),
            'features': evolution_cols,
        },
    },
    'k_selection': {
        str(k): {
            'silhouette': float(r['silhouette']),
            'calinski_harabasz': float(r['calinski_harabasz']),
            'davies_bouldin': float(r['davies_bouldin']),
            'bic': float(r['bic']),
            'cov_type': r['cov_type'],
        }
        for k, r in k_results.items()
    },
    'cluster_stats': {str(k): v for k, v in v_stats.items()},
}

summary_path = os.path.join(CONFIG['save_dir'], 'vehicle_clustering_improved_summary.json')
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2, cls=NumpyEncoder)
print(f"   ✓ vehicle_clustering_improved_summary.json")

# ============================================================
# 9. Quality Summary
# ============================================================
print(f"\n{'='*80}")
print("✅ STEP 8 IMPROVED COMPLETE!")
print("=" * 80)

print(f"""
Summary:
  - Vehicles: {len(vehicle_3d_df):,}
  - Clusters: {best_k}
  - Method: GMM (3D Features)
  - Covariance Type: {best_result['cov_type']}
  - Silhouette Score: {best_result['silhouette']:.4f}
  - Calinski-Harabasz: {best_result['calinski_harabasz']:.1f}
  - Davies-Bouldin: {best_result['davies_bouldin']:.4f}
  - BIC: {best_result['bic']:.1f}

Feature Dimensions:
  ① Distribution: {len(distribution_cols)} features
  ② Transition:   {len(transition_cols)} features
  ③ Evolution:    {len(evolution_cols)} features
  Total:          {len(all_feature_cols)} features (active: {len(active_feature_cols)})

Cluster Breakdown:
""")

for vc in sorted(unique_clusters):
    comp = v_stats[vc]['composition']
    print(f"  V{vc} ({v_stats[vc]['label']:<30}): {v_stats[vc]['size']:>6,} vehicles ({v_stats[vc]['pct']:>5.1f}%)")
    print(f"     Distribution: C0:{comp['C0']:>5.1%} C1:{comp['C1']:>5.1%} C2:{comp['C2']:>5.1%} C3:{comp['C3']:>5.1%}")
    print(f"     Transition:   entropy={v_stats[vc]['transition_entropy']:.3f}  switch_rate={v_stats[vc]['mode_switch_rate']:.3f}")
    print(f"     Evolution:    stability={v_stats[vc]['pattern_stability']:.3f}  rhythm={v_stats[vc]['rhythm_regularity']:.3f}")

print(f"""
Output Files:
  1. vehicle_clustering_improved_3d.csv
  2. vehicle_clustering_improved_summary.json
  3. features_3d_matrix.npz
  4. improved_3d_distribution.png/pdf
  5. improved_3d_transition.png/pdf
  6. improved_3d_evolution.png/pdf
  7. improved_3d_overview.png/pdf
  8. improved_3d_k_selection.png/pdf

Next Steps:
  - Compare with baseline clustering results
  - Use improved clusters for coupling analysis
  - Run: python vehicle_clustering/step8_vehicle_clustering_analysis_figures.py
""")

print("=" * 80)
