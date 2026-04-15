"""
Step 10: Improved 3D Vehicle Clustering
Integrates Distribution, Transition, and Evolution features for
enhanced vehicle-level clustering using GMM.
"""

import numpy as np
import pandas as pd
import os
import json
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
import seaborn as sns

warnings.filterwarnings('ignore')

rcParams['font.sans-serif'] = ['DejaVu Sans']
rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 11
rcParams['figure.dpi'] = 150

print("=" * 80)
print("🚀 STEP 10: IMPROVED 3D VEHICLE CLUSTERING")
print("   (Distribution + Transition + Evolution)")
print("=" * 80)

# ============================================================
# Configuration
# ============================================================
CONFIG = {
    'vehicle_features_path': './coupling_analysis/results/vehicles_aggregated_features.csv',
    'segments_path': './coupling_analysis/results/segments_integrated_complete.csv',
    'save_dir': './vehicle_clustering/results/',
    'figures_dir': './vehicle_clustering/results/figures_improved_3d/',
    'seed': 42,
    'k_range': [3, 4, 5],
    'n_modes': 4,  # number of segment-level modes (cluster_id 0-3)
}

os.makedirs(CONFIG['save_dir'], exist_ok=True)
os.makedirs(CONFIG['figures_dir'], exist_ok=True)

# ============================================================
# 【STEP 1】 Load Data
# ============================================================
print(f"\n【STEP 1】Loading Data")
print("=" * 80)

vehicle_features = pd.read_csv(CONFIG['vehicle_features_path'])
print(f"   ✓ Vehicle features: {len(vehicle_features):,} vehicles, "
      f"{len(vehicle_features.columns)} columns")

segments_df = pd.read_csv(CONFIG['segments_path'])
print(f"   ✓ Segments: {len(segments_df):,} rows")

vehicle_ids = vehicle_features['vehicle_id'].unique()
print(f"   ✓ Unique vehicles in features file: {len(vehicle_ids):,}")

seg_vehicle_ids = segments_df['vehicle_id'].unique()
print(f"   ✓ Unique vehicles in segments file: {len(seg_vehicle_ids):,}")

common_ids = np.intersect1d(vehicle_ids, seg_vehicle_ids)
print(f"   ✓ Common vehicle IDs: {len(common_ids):,}")

# ============================================================
# 【STEP 2】 Extract Three-Dimensional Features
# ============================================================
print(f"\n【STEP 2】Extracting 3D Features")
print("=" * 80)

N_MODES = CONFIG['n_modes']

# ----------------------------------------------------------
# Dimension ① : Distribution Features (from aggregated file)
# ----------------------------------------------------------
print(f"\n   🔍 Dimension ①: Distribution Features")
cluster_ratio_cols = [f'cluster_{c}_ratio' for c in range(N_MODES)]

# Verify columns exist
missing_ratio_cols = [c for c in cluster_ratio_cols if c not in vehicle_features.columns]
if missing_ratio_cols:
    raise ValueError(f"Missing columns in vehicle features: {missing_ratio_cols}")

dist_df = vehicle_features[['vehicle_id'] + cluster_ratio_cols].copy()

# Shannon entropy of mode distribution
ratios = dist_df[cluster_ratio_cols].values
ratios_safe = np.clip(ratios, 1e-10, None)
dist_df['distribution_entropy'] = entropy(ratios_safe, axis=1)

# Dominant mode ratio
dist_df['dominant_mode_ratio'] = ratios.max(axis=1)

dist_feature_cols = cluster_ratio_cols + ['distribution_entropy', 'dominant_mode_ratio']
print(f"      ✓ {len(dist_feature_cols)} distribution features extracted")

# ----------------------------------------------------------
# Dimension ② & ③ : Transition & Evolution Features (per vehicle)
# ----------------------------------------------------------
print(f"\n   🔍 Dimension ② & ③: Transition & Evolution Features")
print(f"      Computing per-vehicle features from {len(segments_df):,} segments...")

segments_sorted = segments_df.sort_values(['vehicle_id', 'segment_id']).reset_index(drop=True)

trans_cols = [f'trans_{i}_to_{j}_prob' for i in range(N_MODES) for j in range(N_MODES)]
trans_extra_cols = ['transition_entropy', 'self_transition_ratio',
                    'transition_diversity', 'dominant_transition']
evol_cols = ['sequence_length', 'mode_switching_freq', 'stability_index',
             'mode_entropy', 'rhythm_regularity', 'volatility_index']
cumulative_cols = [f'cumulative_hours_C{c}' for c in range(N_MODES)]
evol_cols_all = evol_cols + cumulative_cols

all_new_cols = trans_cols + trans_extra_cols + evol_cols_all

records = []
grouped = segments_sorted.groupby('vehicle_id')

for vid, grp in tqdm(grouped, desc="      Processing vehicles", ncols=80):
    row = {'vehicle_id': vid}
    modes = grp['cluster_id'].values.astype(int)
    n_segs = len(modes)

    # --- Transition features ---
    trans_matrix = np.zeros((N_MODES, N_MODES), dtype=np.float64)
    if n_segs > 1:
        for k in range(n_segs - 1):
            src, dst = modes[k], modes[k + 1]
            if 0 <= src < N_MODES and 0 <= dst < N_MODES:
                trans_matrix[src, dst] += 1

    # Row-normalize
    row_sums = trans_matrix.sum(axis=1, keepdims=True)
    row_sums_safe = np.where(row_sums == 0, 1.0, row_sums)
    trans_prob = trans_matrix / row_sums_safe

    for i in range(N_MODES):
        for j in range(N_MODES):
            row[f'trans_{i}_to_{j}_prob'] = trans_prob[i, j]

    # Transition entropy
    flat_probs = trans_prob.flatten()
    flat_probs_safe = flat_probs[flat_probs > 0]
    row['transition_entropy'] = float(entropy(flat_probs_safe)) if len(flat_probs_safe) > 0 else 0.0

    # Self-transition ratio
    total_trans = trans_matrix.sum()
    if total_trans > 0:
        self_trans = np.trace(trans_matrix)
        row['self_transition_ratio'] = float(self_trans / total_trans)
    else:
        row['self_transition_ratio'] = 1.0  # single segment stays in its mode

    # Transition diversity
    row['transition_diversity'] = float(np.count_nonzero(trans_matrix))

    # Dominant transition
    if total_trans > 0:
        row['dominant_transition'] = float(trans_matrix.max() / total_trans)
    else:
        row['dominant_transition'] = 1.0

    # --- Evolution features ---
    row['sequence_length'] = n_segs

    # Mode switching frequency
    if n_segs > 1:
        switches = np.sum(modes[:-1] != modes[1:])
        row['mode_switching_freq'] = float(switches / (n_segs - 1))
    else:
        row['mode_switching_freq'] = 0.0

    # Stability index: max consecutive same-mode / total
    if n_segs > 0:
        max_run = 1
        current_run = 1
        for k in range(1, n_segs):
            if modes[k] == modes[k - 1]:
                current_run += 1
                if current_run > max_run:
                    max_run = current_run
            else:
                current_run = 1
        row['stability_index'] = float(max_run / n_segs)
    else:
        row['stability_index'] = 0.0

    # Mode entropy (same as distribution entropy but from raw sequence)
    mode_counts = np.bincount(modes, minlength=N_MODES).astype(float)
    mode_freq = mode_counts / mode_counts.sum() if mode_counts.sum() > 0 else mode_counts
    row['mode_entropy'] = float(entropy(mode_freq + 1e-10))

    # Cumulative hours per mode
    if 'duration_seconds' in grp.columns:
        for c in range(N_MODES):
            mask_c = modes == c
            total_sec = grp.loc[grp.index[mask_c], 'duration_seconds'].sum()
            row[f'cumulative_hours_C{c}'] = float(total_sec / 3600.0)
    else:
        for c in range(N_MODES):
            row[f'cumulative_hours_C{c}'] = 0.0

    # Rhythm regularity: ratio of repeated consecutive pairs
    if n_segs >= 3:
        pairs = list(zip(modes[:-1], modes[1:]))
        pair_counts = Counter(pairs)
        repeated_pairs = sum(1 for k in range(1, len(pairs)) if pairs[k] == pairs[k - 1])
        row['rhythm_regularity'] = float(repeated_pairs / (len(pairs) - 1)) if len(pairs) > 1 else 0.0
    else:
        row['rhythm_regularity'] = 0.0

    # Volatility index: average absolute mode jump magnitude
    if n_segs > 1:
        jumps = np.abs(np.diff(modes).astype(float))
        row['volatility_index'] = float(jumps.mean())
    else:
        row['volatility_index'] = 0.0

    records.append(row)

trans_evol_df = pd.DataFrame(records)
print(f"      ✓ Transition features: {len(trans_cols) + len(trans_extra_cols)} per vehicle")
print(f"      ✓ Evolution features:  {len(evol_cols_all)} per vehicle")
print(f"      ✓ Computed for {len(trans_evol_df):,} vehicles")

# ============================================================
# 【STEP 3】 Merge & Process Features
# ============================================================
print(f"\n【STEP 3】Feature Processing")
print("=" * 80)

# Merge distribution features with transition/evolution features
features_3d = pd.merge(dist_df, trans_evol_df, on='vehicle_id', how='inner')
print(f"   ✓ Merged features for {len(features_3d):,} vehicles")

# Define all feature columns
all_feature_cols = dist_feature_cols + trans_cols + trans_extra_cols + evol_cols_all
print(f"   ✓ Total feature dimensions: {len(all_feature_cols)}")
print(f"      - Distribution: {len(dist_feature_cols)}")
print(f"      - Transition:   {len(trans_cols) + len(trans_extra_cols)}")
print(f"      - Evolution:    {len(evol_cols_all)}")

# Extract feature matrix
X_raw = features_3d[all_feature_cols].copy()

# Handle missing values with median imputation
missing_count = X_raw.isna().sum().sum()
if missing_count > 0:
    print(f"   ⚠️  Found {missing_count} missing values, imputing with median...")
    X_raw = X_raw.fillna(X_raw.median())

# Handle infinities
X_raw = X_raw.replace([np.inf, -np.inf], np.nan).fillna(0)
X_raw = X_raw.astype(np.float64)

# Outlier clipping (clip at 1st and 99th percentile)
for col in all_feature_cols:
    p01 = X_raw[col].quantile(0.01)
    p99 = X_raw[col].quantile(0.99)
    if p01 < p99:
        X_raw[col] = X_raw[col].clip(p01, p99)
print(f"   ✓ Outliers clipped at 1st/99th percentiles")

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw.values)
X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
print(f"   ✓ Features standardized (mean=0, std=1)")

# Remove zero-variance features
var = X_scaled.var(axis=0)
active_mask = var > 1e-8
n_removed = (~active_mask).sum()
if n_removed > 0:
    print(f"   ⚠️  Removing {n_removed} zero-variance features")
    X_cluster = X_scaled[:, active_mask]
    active_feature_cols = [all_feature_cols[i] for i in range(len(all_feature_cols)) if active_mask[i]]
else:
    X_cluster = X_scaled
    active_feature_cols = list(all_feature_cols)

print(f"   ✓ Active features for clustering: {X_cluster.shape[1]}")

# ============================================================
# 【STEP 4】 GMM Clustering (K=3,4,5)
# ============================================================
print(f"\n【STEP 4】GMM Clustering (K={CONFIG['k_range']})")
print("=" * 80)

k_results = {}

for K in CONFIG['k_range']:
    print(f"\n   📊 Testing K={K}:")
    best_k_score = -1
    best_k_result = None

    for cov_type in ['full', 'tied', 'diag', 'spherical']:
        print(f"      cov_type='{cov_type}'...", end=" ")
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

            n_unique = len(np.unique(labels))
            if n_unique < 2:
                print(f"✗ Only {n_unique} cluster(s)")
                continue

            sil = silhouette_score(X_cluster, labels,
                                   sample_size=min(5000, len(X_cluster)),
                                   random_state=CONFIG['seed'])
            ch = calinski_harabasz_score(X_cluster, labels)
            db = davies_bouldin_score(X_cluster, labels)

            print(f"✓ Sil={sil:.4f}, CH={ch:.1f}, DB={db:.4f}")

            if sil > best_k_score:
                best_k_score = sil
                best_k_result = {
                    'K': K,
                    'cov_type': cov_type,
                    'labels': labels.copy(),
                    'gmm': gmm,
                    'silhouette': sil,
                    'calinski_harabasz': ch,
                    'davies_bouldin': db,
                    'bic': gmm.bic(X_cluster),
                    'aic': gmm.aic(X_cluster),
                }

        except Exception as e:
            print(f"✗ Failed: {str(e)[:60]}")

    if best_k_result is not None:
        k_results[K] = best_k_result
        print(f"      🏆 Best for K={K}: cov='{best_k_result['cov_type']}' "
              f"(Sil={best_k_result['silhouette']:.4f})")

# Select optimal K
print(f"\n   Selecting optimal K...")
if not k_results:
    raise RuntimeError("All GMM fits failed!")

optimal_K = max(k_results.keys(), key=lambda k: k_results[k]['silhouette'])
best = k_results[optimal_K]
print(f"   🏆 Optimal K={optimal_K} (Silhouette={best['silhouette']:.4f})")

best_labels = best['labels']
best_gmm = best['gmm']
best_cov_type = best['cov_type']
best_sil = best['silhouette']
best_ch = best['calinski_harabasz']
best_db = best['davies_bouldin']

# ============================================================
# 【STEP 5】 Analyze Vehicle Clusters
# ============================================================
print(f"\n【STEP 5】Analyzing Improved Vehicle Clusters")
print("=" * 80)

unique_clusters = sorted(np.unique(best_labels))
u_stats = {}

print(f"\n   Cluster Distribution (Improved):")
for uc in unique_clusters:
    mask = best_labels == uc
    n = mask.sum()
    pct = n / len(best_labels) * 100
    print(f"      U{uc}: {n:>6,} vehicles ({pct:>5.1f}%)")

    # Compute per-cluster statistics
    X_uc = X_raw[mask]
    feat_df_uc = features_3d[mask]

    comp_mean = {}
    for c in range(N_MODES):
        col = f'cluster_{c}_ratio'
        comp_mean[f'C{c}'] = float(feat_df_uc[col].mean()) if col in feat_df_uc.columns else 0.0

    avg_trans_entropy = float(feat_df_uc['transition_entropy'].mean())
    avg_self_trans = float(feat_df_uc['self_transition_ratio'].mean())
    avg_switching = float(feat_df_uc['mode_switching_freq'].mean())
    avg_stability = float(feat_df_uc['stability_index'].mean())
    avg_volatility = float(feat_df_uc['volatility_index'].mean())
    avg_seq_len = float(feat_df_uc['sequence_length'].mean())

    # Transition matrix average
    avg_trans_matrix = np.zeros((N_MODES, N_MODES))
    for i in range(N_MODES):
        for j in range(N_MODES):
            avg_trans_matrix[i, j] = float(feat_df_uc[f'trans_{i}_to_{j}_prob'].mean())

    u_stats[uc] = {
        'size': int(n),
        'pct': float(pct),
        'composition': comp_mean,
        'transition_entropy': avg_trans_entropy,
        'self_transition_ratio': avg_self_trans,
        'mode_switching_freq': avg_switching,
        'stability_index': avg_stability,
        'volatility_index': avg_volatility,
        'avg_sequence_length': avg_seq_len,
        'avg_transition_matrix': avg_trans_matrix.tolist(),
    }

# Auto-label clusters
print(f"\n   Auto-labeling clusters:")
for uc in unique_clusters:
    s = u_stats[uc]
    comp = s['composition']
    switching = s['mode_switching_freq']
    stability = s['stability_index']
    volatility = s['volatility_index']

    dominant_mode = max(comp, key=comp.get)
    dominant_ratio = comp[dominant_mode]

    if stability > 0.5 and switching < 0.3:
        label = f"Stable-{dominant_mode}-dominant"
    elif switching > 0.7:
        label = "High-switching"
    elif volatility > 1.0:
        label = "Volatile-mixed"
    elif comp['C0'] > 0.35:
        label = "Idle-heavy"
    elif comp['C2'] > 0.25:
        label = "Highway-oriented"
    elif dominant_ratio > 0.45:
        label = f"{dominant_mode}-focused"
    elif s['transition_entropy'] > 2.0:
        label = "Diverse-transitions"
    else:
        label = "Balanced-mixed"

    u_stats[uc]['label'] = label
    print(f"      U{uc}: {label:<25} | "
          f"C0:{comp['C0']:>5.1%} C1:{comp['C1']:>5.1%} "
          f"C2:{comp['C2']:>5.1%} C3:{comp['C3']:>5.1%} | "
          f"switch={switching:.2f} stab={stability:.2f}")

# ============================================================
# 【STEP 6】 Visualization (7 figures)
# ============================================================
print(f"\n【STEP 6】Visualization")
print("=" * 80)

cluster_colors = plt.cm.Set2(np.linspace(0, 1, len(unique_clusters)))

# ----------------------------------------------------------
# Figure 1: Distribution of three-dimensional features
# ----------------------------------------------------------
print(f"   📊 Figure 1: Feature distributions...")

fig, axes = plt.subplots(3, 3, figsize=(18, 14))

# Row 1: Distribution features
dist_plot_cols = ['distribution_entropy', 'dominant_mode_ratio',
                  'cluster_0_ratio']
dist_plot_labels = ['Distribution Entropy', 'Dominant Mode Ratio',
                    'Cluster 0 (Idle) Ratio']

# Row 2: Transition features
trans_plot_cols = ['transition_entropy', 'self_transition_ratio',
                   'transition_diversity']
trans_plot_labels = ['Transition Entropy', 'Self-Transition Ratio',
                     'Transition Diversity']

# Row 3: Evolution features
evol_plot_cols = ['mode_switching_freq', 'stability_index',
                  'volatility_index']
evol_plot_labels = ['Mode Switching Freq', 'Stability Index',
                    'Volatility Index']

all_plot_cols = dist_plot_cols + trans_plot_cols + evol_plot_cols
all_plot_labels = dist_plot_labels + trans_plot_labels + evol_plot_labels
dim_titles = ['Distribution', 'Distribution', 'Distribution',
              'Transition', 'Transition', 'Transition',
              'Evolution', 'Evolution', 'Evolution']

for idx, (col, label, dim) in enumerate(zip(all_plot_cols, all_plot_labels, dim_titles)):
    r, c_idx = divmod(idx, 3)
    ax = axes[r, c_idx]
    for ui, uc in enumerate(unique_clusters):
        mask = best_labels == uc
        vals = features_3d.loc[mask, col].dropna()
        ax.hist(vals, bins=40, alpha=0.5, color=cluster_colors[ui],
                label=f'U{uc}', density=True)
    ax.set_title(f'{label}\n({dim})', fontweight='bold', fontsize=10)
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

row_labels = ['① Distribution', '② Transition', '③ Evolution']
for r_idx, rl in enumerate(row_labels):
    axes[r_idx, 0].set_ylabel(f'{rl}\nDensity', fontweight='bold', fontsize=10)

plt.suptitle('Three-Dimensional Feature Distributions by Cluster',
             fontweight='bold', fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(CONFIG['figures_dir'], '01_features_3d_distributions.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print(f"      ✓ 01_features_3d_distributions.png")

# ----------------------------------------------------------
# Figure 2: Transition matrices heatmaps (per cluster)
# ----------------------------------------------------------
print(f"   📊 Figure 2: Transition matrices...")

n_cls = len(unique_clusters)
fig, axes = plt.subplots(1, n_cls, figsize=(5 * n_cls, 4.5))
if n_cls == 1:
    axes = [axes]

mode_labels = [f'C{c}' for c in range(N_MODES)]

for ui, uc in enumerate(unique_clusters):
    ax = axes[ui]
    mat = np.array(u_stats[uc]['avg_transition_matrix'])

    sns.heatmap(mat, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=mode_labels, yticklabels=mode_labels,
                ax=ax, vmin=0, vmax=1, linewidths=0.5,
                cbar_kws={'shrink': 0.8})
    ax.set_title(f"U{uc}: {u_stats[uc]['label']}\n(n={u_stats[uc]['size']:,})",
                 fontweight='bold', fontsize=11)
    ax.set_xlabel('To Mode')
    ax.set_ylabel('From Mode')

plt.suptitle('Average Transition Matrices per Cluster',
             fontweight='bold', fontsize=14, y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(CONFIG['figures_dir'], '02_transition_matrices_heatmaps.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print(f"      ✓ 02_transition_matrices_heatmaps.png")

# ----------------------------------------------------------
# Figure 3: Evolution features comparison
# ----------------------------------------------------------
print(f"   📊 Figure 3: Evolution features comparison...")

evol_compare_cols = ['sequence_length', 'mode_switching_freq', 'stability_index',
                     'mode_entropy', 'rhythm_regularity', 'volatility_index']
evol_compare_labels = ['Sequence Length', 'Mode Switching Freq', 'Stability Index',
                       'Mode Entropy', 'Rhythm Regularity', 'Volatility Index']

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes_flat = axes.flatten()

for idx, (col, label) in enumerate(zip(evol_compare_cols, evol_compare_labels)):
    ax = axes_flat[idx]
    data_per_cluster = []
    tick_labels = []
    for uc in unique_clusters:
        mask = best_labels == uc
        vals = features_3d.loc[mask, col].dropna().values
        data_per_cluster.append(vals)
        tick_labels.append(f'U{uc}')

    bp = ax.boxplot(data_per_cluster, patch_artist=True, labels=tick_labels,
                    showfliers=False, widths=0.6)
    for patch, color in zip(bp['boxes'], cluster_colors[:n_cls]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_title(label, fontweight='bold', fontsize=11)
    ax.grid(True, alpha=0.2, axis='y')

plt.suptitle('Evolution Features Comparison Across Clusters',
             fontweight='bold', fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(CONFIG['figures_dir'], '03_evolution_features_comparison.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print(f"      ✓ 03_evolution_features_comparison.png")

# ----------------------------------------------------------
# Figure 4: Clustering quality comparison (K=3,4,5)
# ----------------------------------------------------------
print(f"   📊 Figure 4: Clustering quality comparison...")

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
k_vals = sorted(k_results.keys())
sil_vals = [k_results[k]['silhouette'] for k in k_vals]
ch_vals = [k_results[k]['calinski_harabasz'] for k in k_vals]
db_vals = [k_results[k]['davies_bouldin'] for k in k_vals]
bic_vals = [k_results[k]['bic'] for k in k_vals]

# Silhouette
ax = axes[0]
bars = ax.bar(k_vals, sil_vals, color=['#2ecc71' if k == optimal_K else '#95a5a6' for k in k_vals],
              edgecolor='black', linewidth=1.5)
for bar, v in zip(bars, sil_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005, f'{v:.4f}',
            ha='center', fontsize=10, fontweight='bold')
ax.set_xlabel('K')
ax.set_ylabel('Silhouette Score (↑)')
ax.set_title('Silhouette Score', fontweight='bold')
ax.set_xticks(k_vals)
ax.grid(True, alpha=0.2, axis='y')

# Calinski-Harabasz
ax = axes[1]
bars = ax.bar(k_vals, ch_vals, color=['#3498db' if k == optimal_K else '#95a5a6' for k in k_vals],
              edgecolor='black', linewidth=1.5)
for bar, v in zip(bars, ch_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, v + max(ch_vals) * 0.02, f'{v:.1f}',
            ha='center', fontsize=10, fontweight='bold')
ax.set_xlabel('K')
ax.set_ylabel('CH Score (↑)')
ax.set_title('Calinski-Harabasz', fontweight='bold')
ax.set_xticks(k_vals)
ax.grid(True, alpha=0.2, axis='y')

# Davies-Bouldin
ax = axes[2]
bars = ax.bar(k_vals, db_vals, color=['#e74c3c' if k == optimal_K else '#95a5a6' for k in k_vals],
              edgecolor='black', linewidth=1.5)
for bar, v in zip(bars, db_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, v + max(db_vals) * 0.02, f'{v:.4f}',
            ha='center', fontsize=10, fontweight='bold')
ax.set_xlabel('K')
ax.set_ylabel('DB Score (↓)')
ax.set_title('Davies-Bouldin', fontweight='bold')
ax.set_xticks(k_vals)
ax.grid(True, alpha=0.2, axis='y')

# BIC
ax = axes[3]
bars = ax.bar(k_vals, bic_vals, color=['#9b59b6' if k == optimal_K else '#95a5a6' for k in k_vals],
              edgecolor='black', linewidth=1.5)
for bar, v in zip(bars, bic_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, v + abs(max(bic_vals)) * 0.005,
            f'{v:.0f}', ha='center', fontsize=9, fontweight='bold')
ax.set_xlabel('K')
ax.set_ylabel('BIC (↓)')
ax.set_title('BIC', fontweight='bold')
ax.set_xticks(k_vals)
ax.grid(True, alpha=0.2, axis='y')

plt.suptitle(f'Clustering Quality Comparison (Optimal K={optimal_K})',
             fontweight='bold', fontsize=14, y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(CONFIG['figures_dir'], '04_clustering_quality_comparison.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print(f"      ✓ 04_clustering_quality_comparison.png")

# ----------------------------------------------------------
# Figure 5: PCA 3D projection
# ----------------------------------------------------------
print(f"   📊 Figure 5: PCA projection...")

pca_3d = PCA(n_components=3, random_state=CONFIG['seed'])
X_pca_3d = pca_3d.fit_transform(X_cluster)

pca_2d = PCA(n_components=2, random_state=CONFIG['seed'])
X_pca_2d = pca_2d.fit_transform(X_cluster)

fig = plt.figure(figsize=(18, 7))

# 2D projection
ax1 = fig.add_subplot(1, 2, 1)
ev = pca_2d.explained_variance_ratio_
for ui, uc in enumerate(unique_clusters):
    mask = best_labels == uc
    ax1.scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1],
                c=[cluster_colors[ui]], s=30, alpha=0.5,
                label=f"U{uc}: {u_stats[uc]['label']} (n={mask.sum():,})",
                edgecolors='white', linewidth=0.3)
ax1.set_xlabel(f'PC1 ({ev[0]:.1%})', fontweight='bold')
ax1.set_ylabel(f'PC2 ({ev[1]:.1%})', fontweight='bold')
ax1.set_title('(a) PCA 2D Projection', fontweight='bold', fontsize=12)
ax1.legend(fontsize=9, loc='best', framealpha=0.9)
ax1.grid(True, alpha=0.2)

# 3D projection
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ev3 = pca_3d.explained_variance_ratio_
for ui, uc in enumerate(unique_clusters):
    mask = best_labels == uc
    ax2.scatter(X_pca_3d[mask, 0], X_pca_3d[mask, 1], X_pca_3d[mask, 2],
                c=[cluster_colors[ui]], s=15, alpha=0.4,
                label=f'U{uc}')
ax2.set_xlabel(f'PC1 ({ev3[0]:.1%})', fontsize=9)
ax2.set_ylabel(f'PC2 ({ev3[1]:.1%})', fontsize=9)
ax2.set_zlabel(f'PC3 ({ev3[2]:.1%})', fontsize=9)
ax2.set_title('(b) PCA 3D Projection', fontweight='bold', fontsize=12)
ax2.legend(fontsize=8, loc='best')

plt.suptitle(f'PCA Visualization (Improved 3D Clustering, K={optimal_K})',
             fontweight='bold', fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(CONFIG['figures_dir'], '05_pca_3d_projection.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print(f"      ✓ 05_pca_3d_projection.png")

# ----------------------------------------------------------
# Figure 6: Cluster sizes comparison (old K4 vs improved)
# ----------------------------------------------------------
print(f"   📊 Figure 6: Cluster sizes comparison...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Improved clustering sizes
ax = axes[0]
sizes = [u_stats[uc]['size'] for uc in unique_clusters]
labels_list = [f"U{uc}\n{u_stats[uc]['label']}" for uc in unique_clusters]
bars = ax.bar(range(len(unique_clusters)), sizes, color=cluster_colors[:n_cls],
              edgecolor='black', linewidth=1.5)
for i, (bar, s) in enumerate(zip(bars, sizes)):
    ax.text(bar.get_x() + bar.get_width() / 2, s + max(sizes) * 0.02,
            f'{s:,}\n({s / sum(sizes):.1%})', ha='center', fontsize=10, fontweight='bold')
ax.set_xticks(range(len(unique_clusters)))
ax.set_xticklabels(labels_list, fontsize=9)
ax.set_ylabel('Number of Vehicles', fontweight='bold')
ax.set_title(f'(a) Improved 3D Clustering (K={optimal_K})', fontweight='bold', fontsize=12)
ax.grid(True, alpha=0.2, axis='y')

# Old K=4 comparison (from vehicle features if available)
ax = axes[1]
old_cluster_col = None
if 'vehicle_cluster' in vehicle_features.columns:
    old_cluster_col = 'vehicle_cluster'

if old_cluster_col is not None:
    old_labels = vehicle_features.loc[
        vehicle_features['vehicle_id'].isin(features_3d['vehicle_id']),
        old_cluster_col
    ].values
    old_unique = sorted(np.unique(old_labels))
    old_sizes = [np.sum(old_labels == vc) for vc in old_unique]
    old_colors = plt.cm.Set3(np.linspace(0, 1, len(old_unique)))
    old_labels_list = [f'V{vc}' for vc in old_unique]

    bars = ax.bar(range(len(old_unique)), old_sizes, color=old_colors,
                  edgecolor='black', linewidth=1.5)
    for i, (bar, s) in enumerate(zip(bars, old_sizes)):
        ax.text(bar.get_x() + bar.get_width() / 2, s + max(old_sizes) * 0.02,
                f'{s:,}\n({s / sum(old_sizes):.1%})', ha='center', fontsize=10, fontweight='bold')
    ax.set_xticks(range(len(old_unique)))
    ax.set_xticklabels(old_labels_list, fontsize=11)
    ax.set_ylabel('Number of Vehicles', fontweight='bold')
    ax.set_title('(b) Previous Clustering (K=4)', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.2, axis='y')
else:
    # No old clustering available — show placeholder
    ax.text(0.5, 0.5, 'Previous K=4 clustering\nnot available\n(run step8 first)',
            ha='center', va='center', fontsize=12, color='gray',
            transform=ax.transAxes)
    ax.set_title('(b) Previous Clustering (K=4)', fontweight='bold', fontsize=12)
    ax.axis('off')

plt.suptitle('Cluster Sizes: Improved vs Previous',
             fontweight='bold', fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(CONFIG['figures_dir'], '06_cluster_sizes_comparison.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print(f"      ✓ 06_cluster_sizes_comparison.png")

# ----------------------------------------------------------
# Figure 7: Transition paths flow diagram
# ----------------------------------------------------------
print(f"   📊 Figure 7: Transition paths flow diagram...")

fig, axes = plt.subplots(1, n_cls, figsize=(6 * n_cls, 5))
if n_cls == 1:
    axes = [axes]

for ui, uc in enumerate(unique_clusters):
    ax = axes[ui]
    mat = np.array(u_stats[uc]['avg_transition_matrix'])

    # Draw flow diagram using arrows between nodes
    # Position modes in a circle
    n_modes = N_MODES
    angles = np.linspace(0, 2 * np.pi, n_modes, endpoint=False) - np.pi / 2
    node_x = np.cos(angles) * 0.35 + 0.5
    node_y = np.sin(angles) * 0.35 + 0.5

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.axis('off')

    mode_colors = ['#5B9BD5', '#70AD47', '#C0504D', '#FFC000']

    # Draw edges (transitions with probability > 0.05)
    for i in range(n_modes):
        for j in range(n_modes):
            prob = mat[i, j]
            if prob < 0.05:
                continue
            if i == j:
                # Self-loop — draw as a small arc annotation
                offset = 0.06
                ax.annotate('', xy=(node_x[i] + offset * 0.7, node_y[i] + offset),
                            xytext=(node_x[i] - offset * 0.7, node_y[i] + offset),
                            arrowprops=dict(arrowstyle='->', color=mode_colors[i],
                                            lw=max(1, prob * 5), alpha=0.6,
                                            connectionstyle='arc3,rad=-0.8'))
                ax.text(node_x[i], node_y[i] + offset + 0.04, f'{prob:.0%}',
                        ha='center', va='bottom', fontsize=7, color=mode_colors[i])
            else:
                lw = max(0.5, prob * 6)
                ax.annotate('', xy=(node_x[j], node_y[j]),
                            xytext=(node_x[i], node_y[i]),
                            arrowprops=dict(arrowstyle='->', color='gray',
                                            lw=lw, alpha=min(1.0, prob * 2),
                                            connectionstyle='arc3,rad=0.15'))
                mid_x = (node_x[i] + node_x[j]) / 2
                mid_y = (node_y[i] + node_y[j]) / 2
                if prob >= 0.10:
                    ax.text(mid_x, mid_y, f'{prob:.0%}', ha='center', va='center',
                            fontsize=7, color='black', alpha=0.8,
                            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.7))

    # Draw nodes
    for i in range(n_modes):
        circle = plt.Circle((node_x[i], node_y[i]), 0.08,
                             color=mode_colors[i], ec='black', lw=1.5, zorder=5)
        ax.add_patch(circle)
        ax.text(node_x[i], node_y[i], f'C{i}', ha='center', va='center',
                fontsize=11, fontweight='bold', zorder=6)

    ax.set_title(f"U{uc}: {u_stats[uc]['label']}\n(n={u_stats[uc]['size']:,})",
                 fontweight='bold', fontsize=11)

plt.suptitle('Mode Transition Flow Diagrams per Cluster',
             fontweight='bold', fontsize=14, y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(CONFIG['figures_dir'], '07_transition_paths_sankey.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print(f"      ✓ 07_transition_paths_sankey.png")

print(f"\n   ✓ All 7 figures saved to {CONFIG['figures_dir']}")

# ============================================================
# 【STEP 7】 Save Results
# ============================================================
print(f"\n【STEP 7】Saving Results")
print("=" * 80)

# 7.1 Improved clustering CSV
result_df = features_3d.copy()
result_df['improved_cluster'] = best_labels
result_df['improved_cluster_label'] = result_df['improved_cluster'].map(
    {uc: u_stats[uc]['label'] for uc in unique_clusters}
)

result_path = os.path.join(CONFIG['save_dir'], 'vehicle_clustering_improved_3d.csv')
result_df.to_csv(result_path, index=False)
print(f"   ✓ vehicle_clustering_improved_3d.csv ({len(result_df):,} vehicles)")

# 7.2 Complete feature matrix
np.savez(os.path.join(CONFIG['save_dir'], 'features_3d_complete.npz'),
         X_raw=X_raw.values,
         X_scaled=X_scaled,
         X_cluster=X_cluster,
         labels=best_labels,
         vehicle_ids=features_3d['vehicle_id'].values,
         feature_names=np.array(all_feature_cols),
         active_feature_names=np.array(active_feature_cols),
         X_pca_2d=X_pca_2d,
         X_pca_3d=X_pca_3d,
         scaler_mean=scaler.mean_,
         scaler_scale=scaler.scale_)
print(f"   ✓ features_3d_complete.npz")

# 7.3 Clustering comparison metrics
comparison_metrics = {}
for K in sorted(k_results.keys()):
    kr = k_results[K]
    comparison_metrics[str(K)] = {
        'covariance_type': kr['cov_type'],
        'silhouette': float(kr['silhouette']),
        'calinski_harabasz': float(kr['calinski_harabasz']),
        'davies_bouldin': float(kr['davies_bouldin']),
        'bic': float(kr['bic']),
        'aic': float(kr['aic']),
    }
comparison_metrics['optimal_K'] = int(optimal_K)

comp_metrics_path = os.path.join(CONFIG['save_dir'], 'clustering_comparison_metrics.json')
with open(comp_metrics_path, 'w') as f:
    json.dump(comparison_metrics, f, indent=2, default=str)
print(f"   ✓ clustering_comparison_metrics.json")

# 7.4 Summary JSON
summary = {
    'timestamp': pd.Timestamp.now().isoformat(),
    'method': 'GMM (Improved 3D)',
    'optimal_K': int(optimal_K),
    'covariance_type': best_cov_type,
    'silhouette_score': float(best_sil),
    'calinski_harabasz_score': float(best_ch),
    'davies_bouldin_score': float(best_db),
    'n_vehicles': len(features_3d),
    'n_features_total': len(all_feature_cols),
    'n_features_active': len(active_feature_cols),
    'feature_dimensions': {
        'distribution': len(dist_feature_cols),
        'transition': len(trans_cols) + len(trans_extra_cols),
        'evolution': len(evol_cols_all),
    },
    'cluster_stats': {
        f'U{k}': {key: val for key, val in v.items() if key != 'avg_transition_matrix'}
        for k, v in u_stats.items()
    },
    'feature_names': all_feature_cols,
    'active_feature_names': active_feature_cols,
    'k_comparison': comparison_metrics,
}

summary_path = os.path.join(CONFIG['save_dir'], 'vehicle_clustering_improved_summary.json')
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2, default=str)
print(f"   ✓ vehicle_clustering_improved_summary.json")

# ============================================================
# Complete
# ============================================================
print("\n" + "=" * 80)
print("✅ STEP 10 COMPLETE!")
print("=" * 80)

print(f"""
Summary:
  - Vehicles: {len(features_3d):,}
  - Optimal K: {optimal_K}
  - Method: GMM (Improved 3D)
  - Covariance Type: {best_cov_type}
  - Silhouette Score: {best_sil:.4f}
  - Calinski-Harabasz: {best_ch:.1f}
  - Davies-Bouldin: {best_db:.4f}

Feature Dimensions:
  - Distribution: {len(dist_feature_cols)} features
  - Transition:   {len(trans_cols) + len(trans_extra_cols)} features
  - Evolution:    {len(evol_cols_all)} features
  - Total:        {len(all_feature_cols)} features (active: {len(active_feature_cols)})

Cluster Breakdown:
""")

for uc in sorted(unique_clusters):
    comp = u_stats[uc]['composition']
    s = u_stats[uc]
    print(f"  U{uc} ({s['label']:<25}): {s['size']:>6,} vehicles ({s['pct']:>5.1f}%)")
    print(f"     C0:{comp['C0']:>5.1%} C1:{comp['C1']:>5.1%} "
          f"C2:{comp['C2']:>5.1%} C3:{comp['C3']:>5.1%} | "
          f"switch={s['mode_switching_freq']:.2f} stab={s['stability_index']:.2f}")

print(f"""
Output Files:
  1. vehicle_clustering_improved_3d.csv
  2. features_3d_complete.npz
  3. clustering_comparison_metrics.json
  4. vehicle_clustering_improved_summary.json
  5. figures_improved_3d/ (7 figures)

Figures:
  01_features_3d_distributions.png
  02_transition_matrices_heatmaps.png
  03_evolution_features_comparison.png
  04_clustering_quality_comparison.png
  05_pca_3d_projection.png
  06_cluster_sizes_comparison.png
  07_transition_paths_sankey.png

Next Steps:
  - Compare improved clusters with step8 results
  - Use improved clusters for charging behavior analysis
  - Run step11 for detailed comparison charts
""")

print("=" * 80)
