"""
Step 12: Feature Diagnosis and Optimization Analysis
特征诊断与优化分析

Tasks:
  1. Feature diagnosis (variance, Fisher score, correlation, PCA)
  2. Feature engineering (composite features)
  3. Feature selection (Fisher, correlation dedup, RFE, Random Forest)
  4. Algorithm comparison (GMM, HDBSCAN, Spectral, KMeans)
  5. Optimized clustering with best features
  6. Visualizations (8 figures)
  7. Diagnostic report (diagnostic_report.md)

Input:
  - coupling_analysis/results/segments_integrated_complete.csv
  - coupling_analysis/results/vehicles_aggregated_features.csv
  - vehicle_clustering/results/vehicle_clustering_gmm_k4.csv (optional)

Output:
  - vehicle_clustering/results/diagnostic_report.md
  - vehicle_clustering/results/features_optimized_candidates.csv
  - vehicle_clustering/results/feature_importance_scores.json
  - vehicle_clustering/results/clustering_algorithm_comparison.json
  - vehicle_clustering/results/vehicle_clustering_optimal.csv
  - vehicle_clustering/results/figures_diagnosis/*.png
"""

import numpy as np
import pandas as pd
import os
import json
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime
from scipy.stats import entropy
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score)
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

try:
    import hdbscan as _hdbscan_module
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 150

# ============================================================
# Custom JSON encoder for NumPy types
# ============================================================
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


# ============================================================
# Configuration
# ============================================================
CONFIG = {
    'segments_path': './coupling_analysis/results/segments_integrated_complete.csv',
    'vehicles_path': './coupling_analysis/results/vehicles_aggregated_features.csv',
    'existing_clustering_path': './vehicle_clustering/results/vehicle_clustering_gmm_k4.csv',
    'save_dir': './vehicle_clustering/results/',
    'fig_dir': './vehicle_clustering/results/figures_diagnosis/',
    'seed': 42,
    'n_clusters_range': range(2, 7),  # K=2..6
}

os.makedirs(CONFIG['save_dir'], exist_ok=True)
os.makedirs(CONFIG['fig_dir'], exist_ok=True)

np.random.seed(CONFIG['seed'])

# Number of segment-level clusters (C0-C3)
N_SEG_CLUSTERS = 4

print("=" * 80)
print("STEP 12: FEATURE DIAGNOSIS AND OPTIMIZATION ANALYSIS")
print("=" * 80)


# ============================================================
# Helper functions
# ============================================================
def compute_fisher_score(X, y):
    """Compute Fisher discriminant score for each feature.

    Fisher score = inter-class variance / intra-class variance.
    Higher is better for cluster discrimination.
    """
    classes = np.unique(y)
    n_features = X.shape[1]
    scores = np.zeros(n_features)

    overall_mean = X.mean(axis=0)
    for j in range(n_features):
        inter_var = 0.0
        intra_var = 0.0
        for c in classes:
            mask = y == c
            n_c = mask.sum()
            if n_c == 0:
                continue
            class_mean = X[mask, j].mean()
            class_var = X[mask, j].var()
            inter_var += n_c * (class_mean - overall_mean[j]) ** 2
            intra_var += n_c * class_var
        scores[j] = inter_var / (intra_var + 1e-10)
    return scores


def compute_gini_coefficient(values):
    """Compute Gini coefficient for a distribution of values."""
    values = np.sort(np.abs(values))
    n = len(values)
    if n == 0 or values.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * values) / (n * values.sum())) - (n + 1) / n


# ============================================================
# STEP 1: Load Data
# ============================================================
print("\n[STEP 1] Loading Data")
print("=" * 80)

segments_df = pd.read_csv(CONFIG['segments_path'])
print(f"   Segments: {len(segments_df):,} rows, {len(segments_df.columns)} columns")

vehicles_df = pd.read_csv(CONFIG['vehicles_path'])
print(f"   Vehicles: {len(vehicles_df):,} rows, {len(vehicles_df.columns)} columns")

existing_labels = None
if os.path.exists(CONFIG['existing_clustering_path']):
    existing_df = pd.read_csv(CONFIG['existing_clustering_path'])
    if 'vehicle_cluster' in existing_df.columns:
        existing_labels = existing_df.set_index('vehicle_id')['vehicle_cluster']
        print(f"   Existing clustering: {len(existing_labels)} vehicles, "
              f"{existing_labels.nunique()} clusters")
else:
    print("   No existing clustering found, will use initial GMM labels")


# ============================================================
# STEP 2: Diagnose Original Features
# ============================================================
print("\n[STEP 2] Diagnosing Original Features")
print("=" * 80)

# Identify feature columns from vehicles_df
meta_cols = ['vehicle_id', 'n_segments', 'n_trips', 'vehicle_cluster', 'cluster_label']
original_feature_cols = [c for c in vehicles_df.columns if c not in meta_cols]
print(f"   Original feature count: {len(original_feature_cols)}")

X_orig = vehicles_df[original_feature_cols].copy()
X_orig = X_orig.fillna(X_orig.median())
X_orig = X_orig.astype(np.float64)

scaler_orig = StandardScaler()
X_orig_scaled = scaler_orig.fit_transform(X_orig)
X_orig_scaled = np.nan_to_num(X_orig_scaled, nan=0.0, posinf=0.0, neginf=0.0)

# Generate initial labels for diagnosis if none exist
if existing_labels is not None:
    y_diag = vehicles_df['vehicle_id'].map(existing_labels).fillna(0).astype(int).values
else:
    gmm_init = GaussianMixture(n_components=3, n_init=10, random_state=CONFIG['seed'],
                               max_iter=300, reg_covar=1e-4)
    y_diag = gmm_init.fit_predict(X_orig_scaled)

n_orig_clusters = len(np.unique(y_diag))
print(f"   Diagnosis clusters: {n_orig_clusters}")

# 2.1 Variance contribution
print("   2.1 Variance contribution...")
feature_variance = X_orig_scaled.var(axis=0)

# 2.2 Fisher discriminant score
print("   2.2 Fisher discriminant scores...")
fisher_scores = compute_fisher_score(X_orig_scaled, y_diag)

# 2.3 Correlation matrix
print("   2.3 Feature correlation matrix...")
corr_matrix = np.corrcoef(X_orig_scaled.T)

# 2.4 PCA explained variance
print("   2.4 PCA explained variance...")
pca_full = PCA(random_state=CONFIG['seed'])
pca_full.fit(X_orig_scaled)
explained_var = pca_full.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)

# 2.5 Silhouette of original features
sil_orig = silhouette_score(X_orig_scaled, y_diag,
                            sample_size=min(5000, len(X_orig_scaled)),
                            random_state=CONFIG['seed'])
print(f"   Original Silhouette Score: {sil_orig:.4f}")

# Classify features
fisher_threshold_high = np.percentile(fisher_scores, 75)
fisher_threshold_low = np.percentile(fisher_scores, 25)
var_threshold_low = np.percentile(feature_variance, 10)

effective_features = []
redundant_features = []
noise_features = []

for i, fname in enumerate(original_feature_cols):
    if feature_variance[i] < var_threshold_low:
        noise_features.append(fname)
    elif fisher_scores[i] >= fisher_threshold_high:
        effective_features.append(fname)
    else:
        # Check redundancy via high correlation with a better feature
        is_redundant = False
        for j in range(len(original_feature_cols)):
            if j != i and abs(corr_matrix[i, j]) > 0.90 and fisher_scores[j] > fisher_scores[i]:
                is_redundant = True
                break
        if is_redundant:
            redundant_features.append(fname)
        elif fisher_scores[i] < fisher_threshold_low:
            noise_features.append(fname)
        else:
            effective_features.append(fname)

print(f"\n   Feature classification:")
print(f"      Effective:  {len(effective_features)} features")
print(f"      Redundant:  {len(redundant_features)} features")
print(f"      Noise/Low:  {len(noise_features)} features")


# ============================================================
# STEP 3: Feature Engineering - Optimized Composite Features
# ============================================================
print("\n[STEP 3] Feature Engineering - Composite Features")
print("=" * 80)

segments_df['start_dt'] = pd.to_datetime(segments_df['start_dt'], errors='coerce')
segments_df['end_dt'] = pd.to_datetime(segments_df['end_dt'], errors='coerce')

optimized_features = []
vehicle_ids_ordered = []

for vehicle_id, v_group in tqdm(segments_df.groupby('vehicle_id'),
                                desc="   Computing optimized features"):
    v_sorted = v_group.sort_values('segment_id')
    clusters = v_sorted['cluster_id'].values.astype(int)
    durations = v_sorted['duration_seconds'].values.astype(float)
    n_segs = len(clusters)

    feat = {'vehicle_id': vehicle_id}

    # --- Distribution features (from cluster ratios) ---
    cluster_counts = np.bincount(clusters, minlength=N_SEG_CLUSTERS)
    cluster_ratios = cluster_counts / max(n_segs, 1)
    for c in range(N_SEG_CLUSTERS):
        feat[f'cluster_{c}_ratio'] = cluster_ratios[c]

    # Distribution entropy
    nonzero = cluster_ratios[cluster_ratios > 0]
    feat['distribution_entropy'] = float(entropy(nonzero))

    # Dominant mode ratio
    feat['dominant_mode_ratio'] = float(cluster_ratios.max())

    # --- Transition matrix features (compact: 5-6 instead of 16) ---
    T = np.zeros((N_SEG_CLUSTERS, N_SEG_CLUSTERS))
    if n_segs > 1:
        for i in range(n_segs - 1):
            T[clusters[i], clusters[i + 1]] += 1
        T_norm = T.copy()
        row_sums = T_norm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        T_norm = T_norm / row_sums
    else:
        T_norm = np.eye(N_SEG_CLUSTERS) * (1.0 / N_SEG_CLUSTERS)

    # Highway preference: tendency to enter/leave highway mode (cluster 2)
    feat['highway_preference'] = float(
        (T_norm[2, :].sum() + T_norm[:, 2].sum()) / 2)
    # Urban affinity: tendency for city driving (cluster 1)
    feat['urban_affinity'] = float(
        (T_norm[1, :].sum() + T_norm[:, 1].sum()) / 2)
    # Idle preference: tendency for idle/parking (cluster 0)
    feat['idle_preference'] = float(
        (T_norm[0, :].sum() + T_norm[:, 0].sum()) / 2)
    # Mode stability: diagonal dominance
    feat['mode_stability'] = float(np.trace(T_norm) / N_SEG_CLUSTERS)
    # Transition diversity: fraction of active transitions
    feat['transition_diversity'] = float(
        len(np.where(T_norm > 0.01)[0]) / T_norm.size)
    # Max transition probability
    feat['max_transition_prob'] = float(np.max(T_norm))

    # --- Driving behavior features ---
    total_time = durations.sum()
    if total_time > 0:
        # Time proportions by mode
        for c in range(N_SEG_CLUSTERS):
            mode_time = durations[clusters == c].sum()
            feat[f'time_proportion_C{c}'] = float(mode_time / total_time)

        # Idle time concentration (Gini coefficient)
        idle_durations = durations[clusters == 0]
        feat['idle_concentration'] = float(
            compute_gini_coefficient(idle_durations) if len(idle_durations) > 1 else 0.0)
    else:
        for c in range(N_SEG_CLUSTERS):
            feat[f'time_proportion_C{c}'] = 0.0
        feat['idle_concentration'] = 0.0

    # SOC drop rate
    if 'soc_start' in v_sorted.columns and 'soc_end' in v_sorted.columns:
        soc_drops = (v_sorted['soc_start'].values - v_sorted['soc_end'].values)
        total_hours = total_time / 3600.0
        feat['avg_soc_drop_rate'] = float(
            soc_drops.sum() / total_hours if total_hours > 0 else 0.0)
    else:
        feat['avg_soc_drop_rate'] = 0.0

    # Speed variance (from physical features if available)
    speed_col = 'phys_avg_speed' if 'phys_avg_speed' in v_sorted.columns else None
    if speed_col and not v_sorted[speed_col].isna().all():
        feat['speed_variance'] = float(v_sorted[speed_col].var())
    else:
        feat['speed_variance'] = 0.0

    # --- Evolution stability features ---
    if n_segs > 1:
        # Mode switching frequency
        transitions_count = sum(
            1 for i in range(n_segs - 1) if clusters[i] != clusters[i + 1])
        feat['mode_switching_freq'] = float(transitions_count / (n_segs - 1))

        # Rhythm consistency: fraction of repeated transition pairs
        pair_counts = {}
        for i in range(n_segs - 1):
            pair = (clusters[i], clusters[i + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
        repeated_pairs = sum(1 for v in pair_counts.values() if v > 1)
        feat['rhythm_consistency'] = float(
            repeated_pairs / max(len(pair_counts), 1))

        # Mode persistence: average consecutive run length
        run_lengths = []
        current_run = 1
        for i in range(1, n_segs):
            if clusters[i] == clusters[i - 1]:
                current_run += 1
            else:
                run_lengths.append(current_run)
                current_run = 1
        run_lengths.append(current_run)
        feat['mode_persistence'] = float(np.mean(run_lengths))

        # Volatility score: average magnitude of mode jumps
        jumps = [abs(int(clusters[i + 1]) - int(clusters[i]))
                 for i in range(n_segs - 1)
                 if clusters[i] != clusters[i + 1]]
        feat['volatility_score'] = float(
            np.mean(jumps) if jumps else 0.0)

        # Long-term stability: similarity between first and second half
        mid = n_segs // 2
        first_half = np.bincount(clusters[:mid], minlength=N_SEG_CLUSTERS) / max(mid, 1)
        second_half = np.bincount(clusters[mid:], minlength=N_SEG_CLUSTERS) / max(n_segs - mid, 1)
        feat['long_term_stability'] = float(
            1.0 - np.sqrt(np.sum((first_half - second_half) ** 2) / N_SEG_CLUSTERS))
    else:
        feat['mode_switching_freq'] = 0.0
        feat['rhythm_consistency'] = 0.0
        feat['mode_persistence'] = 1.0
        feat['volatility_score'] = 0.0
        feat['long_term_stability'] = 1.0

    # Sequence length (log-scaled to reduce skew)
    feat['log_sequence_length'] = float(np.log1p(n_segs))

    # Mode entropy
    feat['mode_entropy'] = float(entropy(cluster_ratios[cluster_ratios > 0]))

    optimized_features.append(feat)
    vehicle_ids_ordered.append(vehicle_id)

opt_df = pd.DataFrame(optimized_features)
print(f"   Optimized features: {opt_df.shape[1] - 1} features for "
      f"{len(opt_df)} vehicles")

# Save optimized feature candidates
opt_df.to_csv(os.path.join(CONFIG['save_dir'], 'features_optimized_candidates.csv'),
              index=False)
print(f"   Saved: features_optimized_candidates.csv")


# ============================================================
# STEP 4: Feature Selection
# ============================================================
print("\n[STEP 4] Feature Selection")
print("=" * 80)

opt_feature_cols = [c for c in opt_df.columns if c != 'vehicle_id']
X_opt = opt_df[opt_feature_cols].fillna(0).astype(np.float64).values
scaler_opt = RobustScaler()
X_opt_scaled = scaler_opt.fit_transform(X_opt)
X_opt_scaled = np.nan_to_num(X_opt_scaled, nan=0.0, posinf=0.0, neginf=0.0)

# Remove zero-variance features
opt_var = X_opt_scaled.var(axis=0)
active_mask = opt_var > 1e-8
X_opt_active = X_opt_scaled[:, active_mask]
active_opt_names = [opt_feature_cols[i] for i in range(len(opt_feature_cols))
                    if active_mask[i]]
print(f"   Active features after zero-variance removal: {len(active_opt_names)}")

# Initial clustering for feature selection scoring
gmm_sel = GaussianMixture(n_components=3, n_init=10, random_state=CONFIG['seed'],
                          max_iter=300, reg_covar=1e-4)
y_sel = gmm_sel.fit_predict(X_opt_active)

# 4.1 Fisher score
print("   4.1 Fisher score ranking...")
fisher_opt = compute_fisher_score(X_opt_active, y_sel)
fisher_rank = np.argsort(-fisher_opt)

# 4.2 Correlation-based redundancy removal
print("   4.2 Correlation deduplication...")
corr_opt = np.corrcoef(X_opt_active.T)
corr_threshold = 0.90
to_remove_corr = set()
for i in range(len(active_opt_names)):
    if i in to_remove_corr:
        continue
    for j in range(i + 1, len(active_opt_names)):
        if j in to_remove_corr:
            continue
        if abs(corr_opt[i, j]) > corr_threshold:
            # Remove the one with lower Fisher score
            if fisher_opt[i] >= fisher_opt[j]:
                to_remove_corr.add(j)
            else:
                to_remove_corr.add(i)

kept_after_corr = [i for i in range(len(active_opt_names))
                   if i not in to_remove_corr]
print(f"      Removed {len(to_remove_corr)} highly correlated features, "
      f"kept {len(kept_after_corr)}")

# 4.3 RFE with Logistic Regression
print("   4.3 RFE with Logistic Regression...")
n_select_rfe = min(15, len(kept_after_corr))
X_for_rfe = X_opt_active[:, kept_after_corr]
names_for_rfe = [active_opt_names[i] for i in kept_after_corr]

lr = LogisticRegression(max_iter=1000, random_state=CONFIG['seed'],
                        solver='lbfgs')
rfe = RFE(lr, n_features_to_select=n_select_rfe, step=1)
rfe.fit(X_for_rfe, y_sel)
rfe_support = rfe.support_
rfe_ranking = rfe.ranking_

rfe_selected = [names_for_rfe[i] for i in range(len(names_for_rfe))
                if rfe_support[i]]
print(f"      RFE selected: {len(rfe_selected)} features")

# 4.4 Random Forest importance
print("   4.4 Random Forest importance...")
rf = RandomForestClassifier(n_estimators=200, random_state=CONFIG['seed'],
                            max_depth=10, n_jobs=-1)
rf.fit(X_for_rfe, y_sel)
rf_importance = rf.feature_importances_
rf_rank = np.argsort(-rf_importance)

# 4.5 Composite scoring
print("   4.5 Composite feature scoring...")
feature_scores_dict = {}
for i, fname in enumerate(names_for_rfe):
    orig_idx = active_opt_names.index(fname)
    score = {
        'fisher_score': float(fisher_opt[orig_idx]),
        'rf_importance': float(rf_importance[i]),
        'rfe_selected': bool(rfe_support[i]),
        'rfe_rank': int(rfe_ranking[i]),
    }
    # Composite: normalize each metric to [0,1] then weight
    score['composite'] = 0.0
    feature_scores_dict[fname] = score

# Normalize and compute composite
fisher_vals = np.array([feature_scores_dict[f]['fisher_score']
                        for f in names_for_rfe])
rf_vals = np.array([feature_scores_dict[f]['rf_importance']
                    for f in names_for_rfe])

fisher_norm = (fisher_vals - fisher_vals.min()) / (
    fisher_vals.max() - fisher_vals.min() + 1e-10)
rf_norm = (rf_vals - rf_vals.min()) / (
    rf_vals.max() - rf_vals.min() + 1e-10)

for i, fname in enumerate(names_for_rfe):
    rfe_bonus = 0.3 if feature_scores_dict[fname]['rfe_selected'] else 0.0
    feature_scores_dict[fname]['composite'] = float(
        0.35 * fisher_norm[i] + 0.35 * rf_norm[i] + rfe_bonus)

# Select top features by composite score
sorted_features = sorted(feature_scores_dict.items(),
                         key=lambda x: x[1]['composite'], reverse=True)
n_final = min(18, len(sorted_features))
final_feature_names = [f[0] for f in sorted_features[:n_final]]

print(f"\n   Final selected features ({n_final}):")
for rank, (fname, scores) in enumerate(sorted_features[:n_final], 1):
    print(f"      #{rank:<2} {fname:<30} composite={scores['composite']:.4f} "
          f"fisher={scores['fisher_score']:.4f} rf={scores['rf_importance']:.4f}")

# Save feature importance scores
with open(os.path.join(CONFIG['save_dir'], 'feature_importance_scores.json'), 'w') as f:
    json.dump({
        'original_diagnosis': {
            'n_features': len(original_feature_cols),
            'silhouette_score': float(sil_orig),
            'effective_features': effective_features,
            'redundant_features': redundant_features,
            'noise_features': noise_features,
            'fisher_scores': {original_feature_cols[i]: float(fisher_scores[i])
                              for i in range(len(original_feature_cols))},
        },
        'optimized_scores': feature_scores_dict,
        'final_selected': final_feature_names,
    }, f, indent=2, cls=NumpyEncoder)
print(f"   Saved: feature_importance_scores.json")


# ============================================================
# STEP 5: Algorithm Comparison
# ============================================================
print("\n[STEP 5] Algorithm Comparison")
print("=" * 80)

# Prepare final feature matrix
final_idx = [opt_feature_cols.index(f) for f in final_feature_names]
X_final = X_opt[:, final_idx]
scaler_final = RobustScaler()
X_final_scaled = scaler_final.fit_transform(X_final)
X_final_scaled = np.nan_to_num(X_final_scaled, nan=0.0, posinf=0.0, neginf=0.0)

algo_results = {}

for K in CONFIG['n_clusters_range']:
    print(f"\n   K={K}:")
    k_results = {}

    # GMM
    try:
        gmm = GaussianMixture(n_components=K, n_init=20,
                              random_state=CONFIG['seed'], max_iter=500,
                              reg_covar=1e-4, covariance_type='full')
        labels_gmm = gmm.fit_predict(X_final_scaled)
        sil_gmm = silhouette_score(X_final_scaled, labels_gmm,
                                   sample_size=min(5000, len(X_final_scaled)),
                                   random_state=CONFIG['seed'])
        db_gmm = davies_bouldin_score(X_final_scaled, labels_gmm)
        ch_gmm = calinski_harabasz_score(X_final_scaled, labels_gmm)
        k_results['GMM'] = {
            'silhouette': float(sil_gmm),
            'davies_bouldin': float(db_gmm),
            'calinski_harabasz': float(ch_gmm),
        }
        print(f"      GMM:      Sil={sil_gmm:.4f}  DB={db_gmm:.4f}  CH={ch_gmm:.1f}")
    except Exception as e:
        print(f"      GMM:      Failed - {e}")

    # KMeans
    try:
        km = KMeans(n_clusters=K, n_init=20, random_state=CONFIG['seed'],
                    max_iter=500)
        labels_km = km.fit_predict(X_final_scaled)
        sil_km = silhouette_score(X_final_scaled, labels_km,
                                  sample_size=min(5000, len(X_final_scaled)),
                                  random_state=CONFIG['seed'])
        db_km = davies_bouldin_score(X_final_scaled, labels_km)
        ch_km = calinski_harabasz_score(X_final_scaled, labels_km)
        k_results['KMeans'] = {
            'silhouette': float(sil_km),
            'davies_bouldin': float(db_km),
            'calinski_harabasz': float(ch_km),
        }
        print(f"      KMeans:   Sil={sil_km:.4f}  DB={db_km:.4f}  CH={ch_km:.1f}")
    except Exception as e:
        print(f"      KMeans:   Failed - {e}")

    # Spectral Clustering (skip large K or large data to avoid OOM)
    try:
        n_samples = len(X_final_scaled)
        if n_samples <= 10000 and K <= 5:
            sc = SpectralClustering(n_clusters=K, affinity='rbf',
                                   random_state=CONFIG['seed'], n_init=10)
            labels_sc = sc.fit_predict(X_final_scaled)
            sil_sc = silhouette_score(X_final_scaled, labels_sc,
                                     sample_size=min(5000, n_samples),
                                     random_state=CONFIG['seed'])
            db_sc = davies_bouldin_score(X_final_scaled, labels_sc)
            ch_sc = calinski_harabasz_score(X_final_scaled, labels_sc)
            k_results['Spectral'] = {
                'silhouette': float(sil_sc),
                'davies_bouldin': float(db_sc),
                'calinski_harabasz': float(ch_sc),
            }
            print(f"      Spectral: Sil={sil_sc:.4f}  DB={db_sc:.4f}  CH={ch_sc:.1f}")
        else:
            print(f"      Spectral: Skipped (data too large: {n_samples} samples)")
    except Exception as e:
        print(f"      Spectral: Failed - {e}")

    # HDBSCAN
    if HAS_HDBSCAN:
        try:
            hdb = _hdbscan_module.HDBSCAN(
                min_cluster_size=max(50, len(X_final_scaled) // 20),
                min_samples=10, cluster_selection_method='eom')
            labels_hdb = hdb.fit_predict(X_final_scaled)
            n_hdb_clusters = len(set(labels_hdb)) - (1 if -1 in labels_hdb else 0)
            if n_hdb_clusters >= 2:
                valid = labels_hdb != -1
                sil_hdb = silhouette_score(X_final_scaled[valid], labels_hdb[valid],
                                          sample_size=min(5000, valid.sum()),
                                          random_state=CONFIG['seed'])
                db_hdb = davies_bouldin_score(X_final_scaled[valid], labels_hdb[valid])
                ch_hdb = calinski_harabasz_score(X_final_scaled[valid], labels_hdb[valid])
                noise_pct = float((~valid).sum() / len(labels_hdb))
                k_results['HDBSCAN'] = {
                    'silhouette': float(sil_hdb),
                    'davies_bouldin': float(db_hdb),
                    'calinski_harabasz': float(ch_hdb),
                    'n_clusters_found': n_hdb_clusters,
                    'noise_fraction': noise_pct,
                }
                print(f"      HDBSCAN:  Sil={sil_hdb:.4f}  DB={db_hdb:.4f}  "
                      f"CH={ch_hdb:.1f}  K_found={n_hdb_clusters}  "
                      f"noise={noise_pct:.1%}")
            else:
                print(f"      HDBSCAN:  Only {n_hdb_clusters} cluster(s) found")
        except Exception as e:
            print(f"      HDBSCAN:  Failed - {e}")
    else:
        print("      HDBSCAN:  Not installed, skipping")

    algo_results[str(K)] = k_results

# Save algorithm comparison
with open(os.path.join(CONFIG['save_dir'], 'clustering_algorithm_comparison.json'),
          'w') as f:
    json.dump(algo_results, f, indent=2, cls=NumpyEncoder)
print(f"\n   Saved: clustering_algorithm_comparison.json")


# ============================================================
# STEP 6: Find Optimal Clustering
# ============================================================
print("\n[STEP 6] Optimal Clustering")
print("=" * 80)

# Find best combination by Silhouette score
best_sil = -1
best_algo = None
best_K = None

for K_str, k_res in algo_results.items():
    for algo_name, metrics in k_res.items():
        if 'silhouette' in metrics and metrics['silhouette'] > best_sil:
            best_sil = metrics['silhouette']
            best_algo = algo_name
            best_K = int(K_str)

print(f"   Best: {best_algo} with K={best_K}, Silhouette={best_sil:.4f}")

# Re-run best algorithm
if best_algo == 'GMM':
    best_model = GaussianMixture(n_components=best_K, n_init=50,
                                 random_state=CONFIG['seed'], max_iter=500,
                                 reg_covar=1e-4, covariance_type='full')
    optimal_labels = best_model.fit_predict(X_final_scaled)
elif best_algo == 'KMeans':
    best_model = KMeans(n_clusters=best_K, n_init=50,
                        random_state=CONFIG['seed'], max_iter=500)
    optimal_labels = best_model.fit_predict(X_final_scaled)
elif best_algo == 'Spectral':
    best_model = SpectralClustering(n_clusters=best_K, affinity='rbf',
                                   random_state=CONFIG['seed'], n_init=10)
    optimal_labels = best_model.fit_predict(X_final_scaled)
else:
    # Fallback to GMM
    best_model = GaussianMixture(n_components=best_K, n_init=50,
                                 random_state=CONFIG['seed'], max_iter=500,
                                 reg_covar=1e-4, covariance_type='full')
    optimal_labels = best_model.fit_predict(X_final_scaled)

# Compute metrics for optimal
sil_optimal = silhouette_score(X_final_scaled, optimal_labels,
                               sample_size=min(5000, len(X_final_scaled)),
                               random_state=CONFIG['seed'])
db_optimal = davies_bouldin_score(X_final_scaled, optimal_labels)
ch_optimal = calinski_harabasz_score(X_final_scaled, optimal_labels)

print(f"   Optimal metrics: Sil={sil_optimal:.4f}  DB={db_optimal:.4f}  "
      f"CH={ch_optimal:.1f}")
print(f"   Improvement: Sil {sil_orig:.4f} -> {sil_optimal:.4f} "
      f"({(sil_optimal - sil_orig) / abs(sil_orig + 1e-10) * 100:+.1f}%)")

# Save optimal clustering results
opt_result_df = opt_df[['vehicle_id']].copy()
opt_result_df['optimal_cluster'] = optimal_labels
for i, fname in enumerate(final_feature_names):
    opt_result_df[fname] = X_final[:, i]

opt_result_df.to_csv(
    os.path.join(CONFIG['save_dir'], 'vehicle_clustering_optimal.csv'),
    index=False)
print(f"   Saved: vehicle_clustering_optimal.csv")


# ============================================================
# STEP 7: Visualizations
# ============================================================
print("\n[STEP 7] Generating Visualizations")
print("=" * 80)

# --- Figure 01: Feature Importance Ranking ---
print("   01_feature_importance_ranking.png")
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Original features - Fisher scores
ax = axes[0]
sorted_idx = np.argsort(fisher_scores)[::-1][:20]
ax.barh(range(len(sorted_idx)),
        fisher_scores[sorted_idx],
        color='steelblue', edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(sorted_idx)))
ax.set_yticklabels([original_feature_cols[i] for i in sorted_idx], fontsize=8)
ax.set_xlabel('Fisher Score', fontweight='bold')
ax.set_title('Original Features - Fisher Score (Top 20)', fontweight='bold')
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

# Optimized features - composite score
ax = axes[1]
sorted_opt = sorted_features[:20]
ax.barh(range(len(sorted_opt)),
        [s[1]['composite'] for s in sorted_opt],
        color='coral', edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(sorted_opt)))
ax.set_yticklabels([s[0] for s in sorted_opt], fontsize=8)
ax.set_xlabel('Composite Score', fontweight='bold')
ax.set_title('Optimized Features - Composite Score (Top 20)', fontweight='bold')
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
fig.savefig(os.path.join(CONFIG['fig_dir'], '01_feature_importance_ranking.png'),
            dpi=200, bbox_inches='tight')
plt.close()

# --- Figure 02: Feature Correlation Heatmap ---
print("   02_feature_correlation_heatmap.png")
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Original correlation (subsample to top features)
top_orig_idx = np.argsort(fisher_scores)[::-1][:15]
top_corr = corr_matrix[np.ix_(top_orig_idx, top_orig_idx)]
top_names = [original_feature_cols[i] for i in top_orig_idx]

ax = axes[0]
sns.heatmap(top_corr, annot=True, fmt='.2f', cmap='RdBu_r',
            xticklabels=top_names, yticklabels=top_names,
            ax=ax, vmin=-1, vmax=1, annot_kws={'size': 6},
            cbar_kws={'shrink': 0.8})
ax.set_title('Original Features Correlation\n(Top 15 by Fisher Score)',
             fontweight='bold')
ax.tick_params(axis='both', labelsize=7)

# Optimized features correlation
X_sel_for_corr = X_opt_scaled[:, [active_opt_names.index(f) for f in final_feature_names
                                  if f in active_opt_names]]
sel_names_corr = [f for f in final_feature_names if f in active_opt_names]
corr_sel = np.corrcoef(X_sel_for_corr.T)

ax = axes[1]
sns.heatmap(corr_sel, annot=True, fmt='.2f', cmap='RdBu_r',
            xticklabels=sel_names_corr, yticklabels=sel_names_corr,
            ax=ax, vmin=-1, vmax=1, annot_kws={'size': 5},
            cbar_kws={'shrink': 0.8})
ax.set_title('Selected Features Correlation', fontweight='bold')
ax.tick_params(axis='both', labelsize=6)

plt.tight_layout()
fig.savefig(os.path.join(CONFIG['fig_dir'], '02_feature_correlation_heatmap.png'),
            dpi=200, bbox_inches='tight')
plt.close()

# --- Figure 03: PCA Explained Variance ---
print("   03_pca_explained_variance.png")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Original features PCA
ax = axes[0]
n_show = min(15, len(explained_var))
ax.bar(range(1, n_show + 1), explained_var[:n_show] * 100,
       color='steelblue', alpha=0.7, edgecolor='black', label='Individual')
ax.plot(range(1, n_show + 1), cumulative_var[:n_show] * 100,
        'ro-', linewidth=2, markersize=6, label='Cumulative')
ax.axhline(y=95, color='gray', linestyle='--', alpha=0.5, label='95% threshold')
ax.set_xlabel('Principal Component', fontweight='bold')
ax.set_ylabel('Explained Variance (%)', fontweight='bold')
ax.set_title('Original Features - PCA Variance', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Optimized features PCA
pca_opt = PCA(random_state=CONFIG['seed'])
pca_opt.fit(X_final_scaled)
ev_opt = pca_opt.explained_variance_ratio_
cv_opt = np.cumsum(ev_opt)

ax = axes[1]
n_show_opt = min(15, len(ev_opt))
ax.bar(range(1, n_show_opt + 1), ev_opt[:n_show_opt] * 100,
       color='coral', alpha=0.7, edgecolor='black', label='Individual')
ax.plot(range(1, n_show_opt + 1), cv_opt[:n_show_opt] * 100,
        'ro-', linewidth=2, markersize=6, label='Cumulative')
ax.axhline(y=95, color='gray', linestyle='--', alpha=0.5, label='95% threshold')
ax.set_xlabel('Principal Component', fontweight='bold')
ax.set_ylabel('Explained Variance (%)', fontweight='bold')
ax.set_title('Optimized Features - PCA Variance', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(CONFIG['fig_dir'], '03_pca_explained_variance.png'),
            dpi=200, bbox_inches='tight')
plt.close()

# --- Figure 04: Fisher Score Distribution ---
print("   04_fisher_score_distribution.png")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.hist(fisher_scores, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
ax.axvline(x=fisher_threshold_high, color='red', linestyle='--',
           label=f'75th pctl = {fisher_threshold_high:.3f}')
ax.axvline(x=fisher_threshold_low, color='orange', linestyle='--',
           label=f'25th pctl = {fisher_threshold_low:.3f}')
ax.set_xlabel('Fisher Score', fontweight='bold')
ax.set_ylabel('Count', fontweight='bold')
ax.set_title('Original Features - Fisher Score Distribution', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.hist(fisher_opt, bins=20, color='coral', edgecolor='black', alpha=0.7)
ax.set_xlabel('Fisher Score', fontweight='bold')
ax.set_ylabel('Count', fontweight='bold')
ax.set_title('Optimized Features - Fisher Score Distribution', fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(CONFIG['fig_dir'], '04_fisher_score_distribution.png'),
            dpi=200, bbox_inches='tight')
plt.close()

# --- Figure 05: Original vs Optimized PCA ---
print("   05_original_vs_optimized_pca.png")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Original PCA
pca2_orig = PCA(n_components=2, random_state=CONFIG['seed'])
X_pca_orig = pca2_orig.fit_transform(X_orig_scaled)

ax = axes[0]
scatter = ax.scatter(X_pca_orig[:, 0], X_pca_orig[:, 1],
                     c=y_diag, cmap='Set2', s=30, alpha=0.5,
                     edgecolors='black', linewidth=0.3)
ax.set_xlabel(f'PC1 ({pca2_orig.explained_variance_ratio_[0]:.1%})',
              fontweight='bold')
ax.set_ylabel(f'PC2 ({pca2_orig.explained_variance_ratio_[1]:.1%})',
              fontweight='bold')
ax.set_title(f'Original Features (Sil={sil_orig:.4f})\n'
             f'{len(original_feature_cols)} features', fontweight='bold')
ax.grid(True, alpha=0.2)
plt.colorbar(scatter, ax=ax, label='Cluster')

# Optimized PCA
pca2_opt = PCA(n_components=2, random_state=CONFIG['seed'])
X_pca_opt = pca2_opt.fit_transform(X_final_scaled)

ax = axes[1]
scatter = ax.scatter(X_pca_opt[:, 0], X_pca_opt[:, 1],
                     c=optimal_labels, cmap='Set2', s=30, alpha=0.5,
                     edgecolors='black', linewidth=0.3)
ax.set_xlabel(f'PC1 ({pca2_opt.explained_variance_ratio_[0]:.1%})',
              fontweight='bold')
ax.set_ylabel(f'PC2 ({pca2_opt.explained_variance_ratio_[1]:.1%})',
              fontweight='bold')
ax.set_title(f'Optimized Features (Sil={sil_optimal:.4f})\n'
             f'{len(final_feature_names)} features', fontweight='bold')
ax.grid(True, alpha=0.2)
plt.colorbar(scatter, ax=ax, label='Cluster')

plt.tight_layout()
fig.savefig(os.path.join(CONFIG['fig_dir'], '05_original_vs_optimized_pca.png'),
            dpi=200, bbox_inches='tight')
plt.close()

# --- Figure 06: Algorithm Comparison ---
print("   06_algorithm_comparison.png")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

algo_names_all = set()
for k_res in algo_results.values():
    algo_names_all.update(k_res.keys())
algo_names_all = sorted(algo_names_all)

colors_algo = plt.cm.Set2(np.linspace(0, 1, len(algo_names_all)))
algo_color_map = {name: colors_algo[i] for i, name in enumerate(algo_names_all)}

metrics_to_plot = [
    ('silhouette', 'Silhouette Score (higher is better)', axes[0]),
    ('davies_bouldin', 'Davies-Bouldin Index (lower is better)', axes[1]),
    ('calinski_harabasz', 'Calinski-Harabasz Score (higher is better)', axes[2]),
]

for metric_key, metric_title, ax in metrics_to_plot:
    for algo_name in algo_names_all:
        ks = []
        vals = []
        for K_str in sorted(algo_results.keys(), key=int):
            if algo_name in algo_results[K_str]:
                if metric_key in algo_results[K_str][algo_name]:
                    ks.append(int(K_str))
                    vals.append(algo_results[K_str][algo_name][metric_key])
        if ks:
            ax.plot(ks, vals, 'o-', label=algo_name, linewidth=2,
                    markersize=8, color=algo_color_map[algo_name])
    ax.set_xlabel('K (Number of Clusters)', fontweight='bold')
    ax.set_ylabel(metric_key.replace('_', ' ').title(), fontweight='bold')
    ax.set_title(metric_title, fontweight='bold', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(sorted([int(k) for k in algo_results.keys()]))

plt.tight_layout()
fig.savefig(os.path.join(CONFIG['fig_dir'], '06_algorithm_comparison.png'),
            dpi=200, bbox_inches='tight')
plt.close()

# --- Figure 07: Optimal Clustering Result ---
print("   07_optimal_clustering_result.png")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

unique_opt = sorted(np.unique(optimal_labels))
colors_cluster = plt.cm.Set2(np.linspace(0, 1, len(unique_opt)))

# PCA scatter
ax = axes[0]
for vi, vc in enumerate(unique_opt):
    mask = optimal_labels == vc
    ax.scatter(X_pca_opt[mask, 0], X_pca_opt[mask, 1],
               c=[colors_cluster[vi]], s=30, alpha=0.5,
               label=f'C{vc} (n={mask.sum()})',
               edgecolors='black', linewidth=0.3)
ax.set_xlabel(f'PC1 ({pca2_opt.explained_variance_ratio_[0]:.1%})',
              fontweight='bold')
ax.set_ylabel(f'PC2 ({pca2_opt.explained_variance_ratio_[1]:.1%})',
              fontweight='bold')
ax.set_title(f'Optimal Clustering: {best_algo} K={best_K}',
             fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

# Cluster sizes
ax = axes[1]
sizes = [np.sum(optimal_labels == vc) for vc in unique_opt]
bars = ax.bar(range(len(unique_opt)), sizes, color=colors_cluster,
              edgecolor='black', linewidth=1.5)
for i, (bar, s) in enumerate(zip(bars, sizes)):
    ax.text(bar.get_x() + bar.get_width() / 2, s + max(sizes) * 0.02,
            f'{s:,}', ha='center', fontsize=10, fontweight='bold')
ax.set_xticks(range(len(unique_opt)))
ax.set_xticklabels([f'C{vc}' for vc in unique_opt])
ax.set_ylabel('Number of Vehicles', fontweight='bold')
ax.set_title('Cluster Sizes', fontweight='bold')
ax.grid(True, alpha=0.2, axis='y')

# Metrics summary
ax = axes[2]
ax.axis('off')
metrics_text = (
    f"OPTIMAL CLUSTERING\n"
    f"{'=' * 30}\n"
    f"Algorithm: {best_algo}\n"
    f"K: {best_K}\n"
    f"Features: {len(final_feature_names)}\n\n"
    f"METRICS\n"
    f"{'-' * 30}\n"
    f"Silhouette: {sil_optimal:.4f}\n"
    f"Davies-Bouldin: {db_optimal:.4f}\n"
    f"Calinski-Harabasz: {ch_optimal:.1f}\n\n"
    f"IMPROVEMENT\n"
    f"{'-' * 30}\n"
    f"Orig Silhouette: {sil_orig:.4f}\n"
    f"New Silhouette:  {sil_optimal:.4f}\n"
    f"Change: {(sil_optimal - sil_orig) / abs(sil_orig + 1e-10) * 100:+.1f}%"
)
ax.text(0.1, 0.95, metrics_text, transform=ax.transAxes,
        fontfamily='monospace', fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
fig.savefig(os.path.join(CONFIG['fig_dir'], '07_optimal_clustering_result.png'),
            dpi=200, bbox_inches='tight')
plt.close()

# --- Figure 08: Cluster Characteristics ---
print("   08_cluster_characteristics.png")

# Compute per-cluster feature means for the optimized features
cluster_means = {}
for vc in unique_opt:
    mask = optimal_labels == vc
    cluster_means[vc] = X_final[mask].mean(axis=0)

n_feat_show = min(10, len(final_feature_names))
top_feat_for_char = final_feature_names[:n_feat_show]
top_feat_idx = list(range(n_feat_show))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Radar-style bar comparison
ax = axes[0]
x_pos = np.arange(n_feat_show)
width = 0.8 / len(unique_opt)
for vi, vc in enumerate(unique_opt):
    vals = [cluster_means[vc][idx] for idx in top_feat_idx]
    ax.barh(x_pos + vi * width, vals, height=width,
            color=colors_cluster[vi], alpha=0.8,
            label=f'C{vc}', edgecolor='black', linewidth=0.5)
ax.set_yticks(x_pos + width * (len(unique_opt) - 1) / 2)
ax.set_yticklabels(top_feat_for_char, fontsize=8)
ax.set_xlabel('Feature Value (raw)', fontweight='bold')
ax.set_title('Cluster Characteristics - Top Features', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()

# Heatmap of normalized cluster profiles
ax = axes[1]
profile_data = np.array([cluster_means[vc][:n_feat_show] for vc in unique_opt])
# Normalize per feature for comparison
profile_norm = profile_data.copy()
for j in range(profile_norm.shape[1]):
    col_range = profile_norm[:, j].max() - profile_norm[:, j].min()
    if col_range > 1e-10:
        profile_norm[:, j] = (profile_norm[:, j] - profile_norm[:, j].min()) / col_range
    else:
        profile_norm[:, j] = 0.5

sns.heatmap(profile_norm, annot=True, fmt='.2f', cmap='YlOrRd',
            xticklabels=top_feat_for_char,
            yticklabels=[f'C{vc}' for vc in unique_opt],
            ax=ax, cbar_kws={'shrink': 0.8, 'label': 'Normalized'},
            annot_kws={'size': 7})
ax.set_title('Cluster Profiles (Normalized)', fontweight='bold')
ax.tick_params(axis='x', labelsize=7)
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
fig.savefig(os.path.join(CONFIG['fig_dir'], '08_cluster_characteristics.png'),
            dpi=200, bbox_inches='tight')
plt.close()

print("   All 8 figures generated.")


# ============================================================
# STEP 8: Generate Diagnostic Report
# ============================================================
print("\n[STEP 8] Generating Diagnostic Report")
print("=" * 80)

# Cluster distribution summary
cluster_dist_str = ""
for vc in unique_opt:
    mask = optimal_labels == vc
    n = mask.sum()
    pct = n / len(optimal_labels) * 100
    cluster_dist_str += f"  - C{vc}: {n:,} vehicles ({pct:.1f}%)\n"

# Effective features list
effective_str = "\n".join(f"  - {f}" for f in effective_features[:10])
if len(effective_features) > 10:
    effective_str += f"\n  - ... ({len(effective_features) - 10} more)"

# Noise features list
noise_str = "\n".join(f"  - {f}" for f in noise_features[:10])
if len(noise_features) > 10:
    noise_str += f"\n  - ... ({len(noise_features) - 10} more)"

# Top selected features
selected_str = ""
for i, fname in enumerate(final_feature_names, 1):
    sc = feature_scores_dict.get(fname, {})
    selected_str += (f"  {i:>2}. {fname:<30} "
                     f"composite={sc.get('composite', 0):.4f}\n")

# Best algorithm per K
best_per_k_str = ""
for K_str in sorted(algo_results.keys(), key=int):
    best_sil_k = -1
    best_name_k = ""
    for algo_name, metrics in algo_results[K_str].items():
        if metrics.get('silhouette', -1) > best_sil_k:
            best_sil_k = metrics['silhouette']
            best_name_k = algo_name
    best_per_k_str += f"  - K={K_str}: {best_name_k} (Sil={best_sil_k:.4f})\n"

report_md = f"""# Feature Diagnosis and Optimization Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. Feature Diagnosis Summary

### Original Feature Analysis
- **Total features**: {len(original_feature_cols)}
- **Original Silhouette Score**: {sil_orig:.4f}
- **Effective features**: {len(effective_features)}
- **Redundant features**: {len(redundant_features)}
- **Noise/Low-value features**: {len(noise_features)}

### PCA Analysis (Original)
- PC1 explains {explained_var[0]:.1%} of variance
- Top 5 PCs explain {cumulative_var[min(4, len(cumulative_var)-1)]:.1%} cumulative
- Top 10 PCs explain {cumulative_var[min(9, len(cumulative_var)-1)]:.1%} cumulative

### Effective Features (Top by Fisher Score)
{effective_str}

### Noise/Low-Value Features
{noise_str}

---

## 2. Optimized Feature Set

### Feature Engineering
Computed **{len(opt_feature_cols)}** composite features from raw segments:
- **Distribution features**: cluster ratios, entropy, dominant mode
- **Transition features**: highway/urban/idle preference, stability, diversity
- **Driving behavior**: time proportions, SOC drop rate, speed variance
- **Evolution features**: switching frequency, persistence, volatility, stability

### Feature Selection Pipeline
1. Zero-variance removal: {len(active_opt_names)} features kept
2. Fisher score ranking
3. Correlation deduplication (threshold={corr_threshold}): {len(to_remove_corr)} removed
4. RFE with Logistic Regression: {len(rfe_selected)} selected
5. Random Forest importance ranking
6. Composite scoring (Fisher 35% + RF 35% + RFE 30%)

### Recommended Feature Subset ({len(final_feature_names)} features)
{selected_str}

---

## 3. Algorithm Comparison

### Best Algorithm per K
{best_per_k_str}

### Overall Best
- **Algorithm**: {best_algo}
- **K**: {best_K}
- **Silhouette**: {sil_optimal:.4f}
- **Davies-Bouldin**: {db_optimal:.4f}
- **Calinski-Harabasz**: {ch_optimal:.1f}

---

## 4. Optimal Clustering

### Cluster Distribution
{cluster_dist_str}

### Metrics Comparison
| Metric | Original | Optimized | Change |
|--------|----------|-----------|--------|
| Silhouette | {sil_orig:.4f} | {sil_optimal:.4f} | {(sil_optimal - sil_orig) / abs(sil_orig + 1e-10) * 100:+.1f}% |
| Features | {len(original_feature_cols)} | {len(final_feature_names)} | {len(final_feature_names) - len(original_feature_cols):+d} |

---

## 5. Recommendations

1. **Use optimized features**: The {len(final_feature_names)}-feature subset provides
   better cluster separation than the original {len(original_feature_cols)} features.
2. **Best algorithm**: {best_algo} with K={best_K} achieves the highest Silhouette score.
3. **Remove redundant features**: {len(redundant_features)} features were identified as
   redundant due to high correlation (>{corr_threshold:.0%}).
4. **Focus on discriminative features**: Transition and evolution features provide
   additional discrimination power beyond simple cluster ratios.

---

## 6. Output Files

- `diagnostic_report.md` - This report
- `features_optimized_candidates.csv` - All optimized feature candidates
- `feature_importance_scores.json` - Detailed feature scoring
- `clustering_algorithm_comparison.json` - Algorithm comparison results
- `vehicle_clustering_optimal.csv` - Optimal clustering assignments
- `figures_diagnosis/` - 8 diagnostic figures
"""

report_path = os.path.join(CONFIG['save_dir'], 'diagnostic_report.md')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_md)
print(f"   Saved: diagnostic_report.md")


# ============================================================
# COMPLETE
# ============================================================
print("\n" + "=" * 80)
print("STEP 12 COMPLETE!")
print("=" * 80)
print(f"""
Summary:
  Original: {len(original_feature_cols)} features, Silhouette={sil_orig:.4f}
  Optimized: {len(final_feature_names)} features, Silhouette={sil_optimal:.4f}
  Best algorithm: {best_algo} (K={best_K})
  Improvement: {(sil_optimal - sil_orig) / abs(sil_orig + 1e-10) * 100:+.1f}%

Output Files:
  1. diagnostic_report.md
  2. features_optimized_candidates.csv
  3. feature_importance_scores.json
  4. clustering_algorithm_comparison.json
  5. vehicle_clustering_optimal.csv
  6. figures_diagnosis/
     - 01_feature_importance_ranking.png
     - 02_feature_correlation_heatmap.png
     - 03_pca_explained_variance.png
     - 04_fisher_score_distribution.png
     - 05_original_vs_optimized_pca.png
     - 06_algorithm_comparison.png
     - 07_optimal_clustering_result.png
     - 08_cluster_characteristics.png
""")
print("=" * 80)
