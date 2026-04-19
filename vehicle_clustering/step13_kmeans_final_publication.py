"""
Step 13: Final KMeans K=4 Vehicle Clustering — Publication-Quality Visualizations
===================================================================================
Three-Dimensional Feature Framework:
  ① Distribution  : 4 mode ratios + distribution_entropy + dominant_mode_ratio  (6 features)
  ② Transition    : 4×4=16 trans_probs + transition_entropy + self_trans +
                    trans_diversity + dominant_trans                              (20 features)
  ③ Evolution     : sequence_length, mode_switching_freq, stability_index,
                    avg_run_length, mode_entropy, 4 cumulative_hours,
                    rhythm_regularity, volatility_index                          (11 features)

Input  : coupling_analysis/results/segments_integrated_complete.csv
Output : vehicle_clustering/results/
           vehicle_clustering_kmeans_final_k4.csv
           clustering_k4_final_summary.json
           clustering_k4_final_report.txt
           feature_matrices_for_analysis.npz
           figures_kmeans_final/
             01_three_dimensions_overview.png
             02_distribution_features_heatmap.png
             03_transition_matrices_4clusters.png
             04_evolution_features_comparison.png
             05_cluster_characteristics_radar.png
             06_kmeans_pca_projection.png
             07_cluster_profiles_detailed.png
             08_vehicle_archetype_summary.png
"""

import os
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib import rcParams

from scipy.stats import entropy as scipy_entropy
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (silhouette_score,
                             davies_bouldin_score,
                             calinski_harabasz_score)
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ── Typography ────────────────────────────────────────────────────────────────
rcParams['font.family']     = 'DejaVu Sans'
rcParams['font.size']       = 11
rcParams['axes.titlesize']  = 12
rcParams['axes.labelsize']  = 11
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['figure.dpi']      = 150

# ── Colour palette (publication-quality) ─────────────────────────────────────
CLUSTER_COLORS = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A']   # 4 vehicle types
SEG_COLORS     = ['#6C8EBF', '#82B366', '#D79B00', '#AE4132']   # 4 segment modes
SEG_LABELS     = ['C0 Moderate-Urban', 'C1 Conservative', 'C2 Aggressive-Urban', 'C3 Highway']

print("=" * 80)
print("🚀  STEP 13: Final KMeans K=4 — Publication-Quality Vehicle Clustering")
print("=" * 80)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
SEG_PATH     = './coupling_analysis/results/segments_integrated_complete.csv'
SAVE_DIR     = './vehicle_clustering/results/'
FIG_DIR      = os.path.join(SAVE_DIR, 'figures_kmeans_final')
os.makedirs(SAVE_DIR,  exist_ok=True)
os.makedirs(FIG_DIR,   exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Load Segments
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n【STEP 0】Loading segment data …")
df = pd.read_csv(SEG_PATH)
print(f"   ✓ Loaded {len(df):,} segments  |  columns: {list(df.columns)}")

# Normalise column names
df.columns = [c.strip() for c in df.columns]

# Ensure datetime columns are parsed
for col in ['start_dt', 'end_dt']:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

# Determine cluster column
cluster_col = None
for candidate in ['cluster_id', 'cluster', 'label']:
    if candidate in df.columns:
        cluster_col = candidate
        break
if cluster_col is None:
    raise ValueError("Cannot find cluster column in segments file.")
print(f"   ✓ Cluster column: '{cluster_col}'")

n_vehicles_total = df['vehicle_id'].nunique()
print(f"   ✓ Unique vehicles: {n_vehicles_total:,}")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Extract Three-Dimensional Features (per vehicle)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n【STEP 1】Extracting 3D features per vehicle …")

N_MODES = 4   # segment clusters C0..C3

records = []

for vehicle_id, grp in tqdm(df.groupby('vehicle_id'),
                             desc='   🔄 Feature extraction', ncols=80):
    grp = grp.sort_values('start_dt').reset_index(drop=True)
    clusters = grp[cluster_col].values.astype(int)
    n        = len(clusters)

    if n < 2:
        continue

    feat = {'vehicle_id': vehicle_id, 'n_segments': n}

    # ── Dimension ①: Distribution ──────────────────────────────────────────
    counts = np.bincount(clusters, minlength=N_MODES)
    ratios = counts / n
    for c in range(N_MODES):
        feat[f'dist_C{c}_ratio'] = ratios[c]

    p = ratios[ratios > 0]
    feat['dist_entropy']          = float(-np.sum(p * np.log(p + 1e-12)))
    feat['dist_dominant_ratio']   = float(ratios.max())

    # ── Dimension ②: Transition ────────────────────────────────────────────
    trans_mat = np.zeros((N_MODES, N_MODES), dtype=np.float64)
    for i in range(n - 1):
        trans_mat[clusters[i], clusters[i + 1]] += 1

    row_sums = trans_mat.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    trans_prob = trans_mat / row_sums

    for r in range(N_MODES):
        for c in range(N_MODES):
            feat[f'trans_{r}_to_{c}'] = float(trans_prob[r, c])

    # Transition entropy
    flat = trans_prob.flatten()
    flat_nz = flat[flat > 0]
    feat['trans_entropy']         = float(-np.sum(flat_nz * np.log(flat_nz + 1e-12)))
    feat['trans_self_ratio']      = float(np.trace(trans_prob) / N_MODES)
    feat['trans_diversity']       = float((trans_prob > 0.05).sum() / (N_MODES * N_MODES))

    # Dominant transition (max off-diagonal probability)
    trans_off = trans_prob.copy()
    np.fill_diagonal(trans_off, 0.0)
    feat['trans_dominant']        = float(trans_off.max())

    # ── Dimension ③: Evolution ─────────────────────────────────────────────
    feat['evo_sequence_length']   = int(n)

    # Mode switching frequency (switches / segment)
    switches = int((np.diff(clusters) != 0).sum())
    feat['evo_switching_freq']    = switches / max(n - 1, 1)

    # Stability index (avg run-length normalised)
    run_lengths = []
    current_run = 1
    for i in range(1, n):
        if clusters[i] == clusters[i - 1]:
            current_run += 1
        else:
            run_lengths.append(current_run)
            current_run = 1
    run_lengths.append(current_run)
    avg_run = float(np.mean(run_lengths))
    feat['evo_stability_index']   = avg_run / max(n, 1)
    feat['evo_avg_run_length']    = avg_run

    # Mode entropy of the sequence (temporal)
    feat['evo_mode_entropy']      = float(scipy_entropy(counts + 1e-12, base=2))

    # Cumulative time per mode (hours)
    if 'duration_seconds' in grp.columns:
        dur = grp['duration_seconds'].values
    else:
        dur = np.ones(n) * 60.0   # assume 1-min segments if not available

    for c in range(N_MODES):
        feat[f'evo_cum_hours_C{c}'] = float(dur[clusters == c].sum() / 3600.0)

    # Rhythm regularity (coefficient of variation of inter-switch gaps)
    switch_positions = np.where(np.diff(clusters) != 0)[0]
    if len(switch_positions) > 1:
        gaps    = np.diff(switch_positions).astype(float)
        cv      = gaps.std() / (gaps.mean() + 1e-12)
        feat['evo_rhythm_regularity'] = float(1.0 / (1.0 + cv))
    else:
        feat['evo_rhythm_regularity'] = 1.0

    # Volatility index (std of mode at each third of the sequence)
    thirds = np.array_split(clusters, 3)
    third_dominant = [np.bincount(t, minlength=N_MODES).argmax() for t in thirds if len(t) > 0]
    feat['evo_volatility_index']  = float(len(set(third_dominant)) / N_MODES)

    records.append(feat)

df_veh = pd.DataFrame(records)
print(f"   ✓ Extracted features for {len(df_veh):,} vehicles")

# Feature column groups
dist_cols  = ([f'dist_C{c}_ratio' for c in range(N_MODES)]
              + ['dist_entropy', 'dist_dominant_ratio'])
trans_cols = ([f'trans_{r}_to_{c}' for r in range(N_MODES) for c in range(N_MODES)]
              + ['trans_entropy', 'trans_self_ratio', 'trans_diversity', 'trans_dominant'])
evo_cols   = (['evo_sequence_length', 'evo_switching_freq', 'evo_stability_index',
               'evo_avg_run_length', 'evo_mode_entropy']
              + [f'evo_cum_hours_C{c}' for c in range(N_MODES)]
              + ['evo_rhythm_regularity', 'evo_volatility_index'])

all_feat_cols = dist_cols + trans_cols + evo_cols
print(f"   ✓ Features: {len(dist_cols)} Dist + {len(trans_cols)} Trans + {len(evo_cols)} Evo = {len(all_feat_cols)} total")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Preprocessing
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n【STEP 2】Preprocessing …")

X = df_veh[all_feat_cols].copy()
X = X.fillna(X.median())

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

# Remove zero-variance features
var = X_scaled.var(axis=0)
active = var > 1e-8
if (~active).sum() > 0:
    print(f"   ⚠  Removing {(~active).sum()} zero-variance features")
X_cluster = X_scaled[:, active]
active_cols = [all_feat_cols[i] for i in range(len(all_feat_cols)) if active[i]]

# PCA for visualisation (keep 2 components)
pca2 = PCA(n_components=2, random_state=42)
X_pca = pca2.fit_transform(X_cluster)
print(f"   ✓ PCA explained variance: {pca2.explained_variance_ratio_.sum()*100:.1f}%  "
      f"(PC1={pca2.explained_variance_ratio_[0]*100:.1f}%, "
      f"PC2={pca2.explained_variance_ratio_[1]*100:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  KMeans K=4 Clustering
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n【STEP 3】KMeans K=4 Clustering …")

kmeans = KMeans(n_clusters=4, n_init=100, max_iter=500, random_state=42)
labels = kmeans.fit_predict(X_cluster)

sil  = silhouette_score(X_cluster, labels, sample_size=min(5000, len(X_cluster)), random_state=42)
db   = davies_bouldin_score(X_cluster, labels)
ch   = calinski_harabasz_score(X_cluster, labels)

print(f"   ✓ Silhouette Score    : {sil:.4f}")
print(f"   ✓ Davies-Bouldin Index: {db:.4f}")
print(f"   ✓ Calinski-Harabasz   : {ch:.1f}")

df_veh['vehicle_cluster'] = labels

print(f"\n   Cluster Distribution:")
for k in range(4):
    n_k  = (labels == k).sum()
    pct  = n_k / len(labels) * 100
    print(f"      K{k}: {n_k:>6,} vehicles ({pct:>5.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Auto-label clusters based on dominant features
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n【STEP 4】Auto-labelling clusters …")

ARCHETYPE_NAMES = {}
ARCHETYPE_DESC  = {}

for k in range(4):
    mask = labels == k
    sub  = df_veh[mask]
    r    = {c: sub[f'dist_C{c}_ratio'].mean() for c in range(N_MODES)}
    sf   = sub['evo_switching_freq'].mean()
    si   = sub['evo_stability_index'].mean()

    if r[2] >= max(r.values()):
        name = "Aggressive-Urban Type"
        desc = "Dominant aggressive-urban segments; high energy consumption; frequent mode switching"
    elif r[3] >= max(r.values()):
        name = "Highway Type"
        desc = "Dominant highway segments; high speed sustained; stable driving rhythm"
    elif si > 0.06 and sf < 0.35:
        name = "Stable-Conservative Type"
        desc = "Low mode-switching frequency; long stable runs; energy-efficient"
    else:
        name = "Mixed-Urban Type"
        desc = "Balanced segment composition; moderate switching; typical commuter pattern"

    ARCHETYPE_NAMES[k] = name
    ARCHETYPE_DESC[k]  = desc
    print(f"      K{k}: {name}")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Save outputs
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n【STEP 5】Saving result files …")

# 5.1  CSV
out_csv = os.path.join(SAVE_DIR, 'vehicle_clustering_kmeans_final_k4.csv')
df_veh.to_csv(out_csv, index=False)
print(f"   ✓ {out_csv}")

# 5.2  NPZ feature matrices
npz_path = os.path.join(SAVE_DIR, 'feature_matrices_for_analysis.npz')
np.savez_compressed(npz_path,
                    X_dist=df_veh[dist_cols].values,
                    X_trans=df_veh[trans_cols].values,
                    X_evo=df_veh[evo_cols].values,
                    X_all=X.values,
                    X_scaled=X_scaled,
                    X_pca=X_pca,
                    labels=labels)
print(f"   ✓ {npz_path}")

# 5.3  JSON summary
summary = {
    'timestamp'    : datetime.now().isoformat(),
    'algorithm'    : 'KMeans',
    'n_clusters'   : 4,
    'n_vehicles'   : int(len(df_veh)),
    'metrics'      : {'silhouette': round(float(sil), 4),
                      'davies_bouldin': round(float(db), 4),
                      'calinski_harabasz': round(float(ch), 1)},
    'feature_dims' : {'distribution': len(dist_cols),
                      'transition'  : len(trans_cols),
                      'evolution'   : len(evo_cols),
                      'total'       : len(all_feat_cols)},
    'clusters'     : {}
}
for k in range(4):
    mask = labels == k
    sub  = df_veh[mask]
    summary['clusters'][f'K{k}'] = {
        'name'       : ARCHETYPE_NAMES[k],
        'description': ARCHETYPE_DESC[k],
        'n_vehicles' : int(mask.sum()),
        'proportion' : round(float(mask.sum() / len(labels)), 4),
        'dist'       : {f'C{c}': round(float(sub[f'dist_C{c}_ratio'].mean()), 4)
                        for c in range(N_MODES)},
        'evo'        : {
            'switching_freq'   : round(float(sub['evo_switching_freq'].mean()), 4),
            'stability_index'  : round(float(sub['evo_stability_index'].mean()), 4),
            'rhythm_regularity': round(float(sub['evo_rhythm_regularity'].mean()), 4),
        }
    }
json_path = os.path.join(SAVE_DIR, 'clustering_k4_final_summary.json')
with open(json_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"   ✓ {json_path}")

# 5.4  Text report
report_lines = [
    "=" * 70,
    "  Step 13: KMeans K=4 Vehicle Clustering — Final Report",
    f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "=" * 70,
    "",
    f"Total vehicles  : {len(df_veh):,}",
    f"Algorithm       : KMeans (n_init=100, max_iter=500)",
    f"Silhouette      : {sil:.4f}",
    f"Davies-Bouldin  : {db:.4f}",
    f"Calinski-Harabasz: {ch:.1f}",
    "",
    "Feature Dimensions:",
    f"  ① Distribution : {len(dist_cols):2d} features  {dist_cols}",
    f"  ② Transition   : {len(trans_cols):2d} features",
    f"  ③ Evolution    : {len(evo_cols):2d} features  {evo_cols}",
    "",
    "─" * 70,
    "Cluster Profiles:",
    "",
]
for k in range(4):
    mask = labels == k
    sub  = df_veh[mask]
    report_lines += [
        f"K{k}: {ARCHETYPE_NAMES[k]}  (n={mask.sum():,}, {mask.sum()/len(labels)*100:.1f}%)",
        f"     {ARCHETYPE_DESC[k]}",
        f"     Dist : C0={sub['dist_C0_ratio'].mean():.2%}  C1={sub['dist_C1_ratio'].mean():.2%}"
        f"  C2={sub['dist_C2_ratio'].mean():.2%}  C3={sub['dist_C3_ratio'].mean():.2%}",
        f"     Evo  : switch_freq={sub['evo_switching_freq'].mean():.3f}"
        f"  stability={sub['evo_stability_index'].mean():.3f}"
        f"  rhythm={sub['evo_rhythm_regularity'].mean():.3f}",
        "",
    ]
report_path = os.path.join(SAVE_DIR, 'clustering_k4_final_report.txt')
with open(report_path, 'w') as f:
    f.write('\n'.join(report_lines))
print(f"   ✓ {report_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Helper: save figure
# ─────────────────────────────────────────────────────────────────────────────
def savefig(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"   ✓ Saved: {name}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 01 — Three-Dimensions Overview
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n【FIGURE 01】Three-Dimensions Overview …")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# ① Distribution bar
ax = axes[0]
bottoms = np.zeros(4)
for c in range(N_MODES):
    vals = [df_veh[labels == k][f'dist_C{c}_ratio'].mean() for k in range(4)]
    ax.bar(range(4), vals, bottom=bottoms, color=SEG_COLORS[c],
           label=SEG_LABELS[c], alpha=0.88, edgecolor='white', linewidth=0.8)
    bottoms += np.array(vals)
ax.set_xticks(range(4))
ax.set_xticklabels([f'K{k}\n{ARCHETYPE_NAMES[k].split()[0]}' for k in range(4)], fontsize=9)
ax.set_ylabel('Average Proportion', fontweight='bold')
ax.set_title('① Distribution\n(Segment Mode Composition)', fontweight='bold')
ax.set_ylim(0, 1.05)
ax.legend(fontsize=8, loc='upper right')
ax.grid(axis='y', alpha=0.3)

# ② Transition entropy
ax = axes[1]
te_vals = [df_veh[labels == k]['trans_entropy'].mean() for k in range(4)]
bars = ax.bar(range(4), te_vals, color=CLUSTER_COLORS, alpha=0.88,
              edgecolor='black', linewidth=1.2)
for bar, v in zip(bars, te_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
            f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_xticks(range(4))
ax.set_xticklabels([f'K{k}' for k in range(4)])
ax.set_ylabel('Transition Entropy', fontweight='bold')
ax.set_title('② Transition\n(Mode-Switch Entropy)', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# ③ Evolution — switching frequency & stability
ax = axes[2]
sw = [df_veh[labels == k]['evo_switching_freq'].mean() for k in range(4)]
st = [df_veh[labels == k]['evo_stability_index'].mean() for k in range(4)]
x  = np.arange(4)
w  = 0.35
b1 = ax.bar(x - w/2, sw, w, label='Switch Freq',  color='#A8DADC', edgecolor='black', linewidth=1)
b2 = ax.bar(x + w/2, st, w, label='Stability Idx',color='#457B9D', edgecolor='black', linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels([f'K{k}' for k in range(4)])
ax.set_ylabel('Value', fontweight='bold')
ax.set_title('③ Evolution\n(Switching Freq & Stability)', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

plt.suptitle('Three-Dimensional Feature Framework Overview (KMeans K=4)',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
savefig(fig, '01_three_dimensions_overview.png')

# ─────────────────────────────────────────────────────────────────────────────
# Figure 02 — Distribution Features Heatmap
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n【FIGURE 02】Distribution Features Heatmap …")

heat_cols = dist_cols
heat_data = np.array([[df_veh[labels == k][c].mean() for c in heat_cols] for k in range(4)])

fig, ax = plt.subplots(figsize=(14, 6))
im = ax.imshow(heat_data, cmap='YlOrRd', aspect='auto', vmin=0)
ax.set_xticks(range(len(heat_cols)))
ax.set_xticklabels(heat_cols, rotation=35, ha='right', fontsize=8)
ax.set_yticks(range(4))
ax.set_yticklabels([f'K{k}: {ARCHETYPE_NAMES[k]}' for k in range(4)], fontsize=10)
ax.set_title('② Distribution Features — Cluster Mean Heatmap', fontweight='bold', fontsize=13)
cbar = plt.colorbar(im, ax=ax, fraction=0.03)
cbar.set_label('Mean Value', fontsize=10)
for i in range(4):
    for j in range(len(heat_cols)):
        v = heat_data[i, j]
        text_color = 'white' if v > heat_data.max() * 0.65 else 'black'
        ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                fontsize=7.5, color=text_color, fontweight='bold')
plt.tight_layout()
savefig(fig, '02_distribution_features_heatmap.png')

# ─────────────────────────────────────────────────────────────────────────────
# Figure 03 — Transition Matrices for 4 Clusters
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n【FIGURE 03】Transition Matrices …")

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for k in range(4):
    ax   = axes[k]
    mat  = np.zeros((N_MODES, N_MODES))
    for r in range(N_MODES):
        for c in range(N_MODES):
            mat[r, c] = df_veh[labels == k][f'trans_{r}_to_{c}'].mean()
    im = ax.imshow(mat, cmap='Blues', vmin=0, vmax=mat.max() + 0.05)
    ax.set_xticks(range(N_MODES))
    ax.set_yticks(range(N_MODES))
    mode_ticks = ['C0', 'C1', 'C2', 'C3']
    ax.set_xticklabels(mode_ticks, fontsize=9)
    ax.set_yticklabels(mode_ticks, fontsize=9)
    ax.set_xlabel('To Mode',   fontsize=9)
    ax.set_ylabel('From Mode', fontsize=9)
    ax.set_title(f'K{k}: {ARCHETYPE_NAMES[k]}\n'
                 f'(n={int((labels==k).sum()):,}, {(labels==k).mean()*100:.1f}%)',
                 fontweight='bold', fontsize=10)
    for r in range(N_MODES):
        for c in range(N_MODES):
            v = mat[r, c]
            text_color = 'white' if v > mat.max() * 0.6 else 'black'
            ax.text(c, r, f'{v:.2f}', ha='center', va='center',
                    fontsize=9, color=text_color, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.suptitle('② Transition Matrices — Mode-Transition Probabilities by Vehicle Type',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
savefig(fig, '03_transition_matrices_4clusters.png')

# ─────────────────────────────────────────────────────────────────────────────
# Figure 04 — Evolution Features Comparison
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n【FIGURE 04】Evolution Features Comparison …")

evo_plot_cols = ['evo_switching_freq', 'evo_stability_index',
                 'evo_rhythm_regularity', 'evo_volatility_index']
evo_plot_labs = ['Mode-Switching\nFrequency', 'Stability\nIndex',
                 'Rhythm\nRegularity', 'Volatility\nIndex']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for idx, (col, lab) in enumerate(zip(evo_plot_cols, evo_plot_labs)):
    ax   = axes[idx]
    data = [df_veh[labels == k][col].dropna().values for k in range(4)]
    bp   = ax.boxplot(data, patch_artist=True,
                      labels=[f'K{k}' for k in range(4)],
                      widths=0.55, showfliers=False,
                      medianprops=dict(color='black', linewidth=2.5))
    for patch, color in zip(bp['boxes'], CLUSTER_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    means = [np.mean(d) for d in data]
    ax.plot(range(1, 5), means, 'D--', color='navy', markersize=7,
            linewidth=1.5, label='Mean')
    ax.set_ylabel(lab, fontweight='bold')
    ax.set_title(f'③ {lab}', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=9)

plt.suptitle('③ Evolution Features — Temporal Dynamics Comparison (KMeans K=4)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
savefig(fig, '04_evolution_features_comparison.png')

# ─────────────────────────────────────────────────────────────────────────────
# Figure 05 — Radar Chart
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n【FIGURE 05】Cluster Characteristics Radar …")

radar_cols = ['dist_C0_ratio', 'dist_C2_ratio', 'dist_C3_ratio',
              'trans_entropy', 'evo_switching_freq', 'evo_stability_index',
              'evo_rhythm_regularity']
radar_labs = ['C0-Moderate\nRatio', 'C2-Aggressive\nRatio', 'C3-Highway\nRatio',
              'Trans.\nEntropy', 'Switching\nFreq', 'Stability\nIndex',
              'Rhythm\nReg.']

n_r = len(radar_cols)
angles = np.linspace(0, 2 * np.pi, n_r, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

for k in range(4):
    vals = [df_veh[labels == k][c].mean() for c in radar_cols]
    # Normalise to [0, 1]
    arr  = np.array(vals)
    all_max = np.array([df_veh[c].max() for c in radar_cols])
    all_min = np.array([df_veh[c].min() for c in radar_cols])
    norm = (arr - all_min) / (all_max - all_min + 1e-12)
    norm_list = norm.tolist() + [norm[0]]
    ax.plot(angles, norm_list, 'o-', linewidth=2.5,
            color=CLUSTER_COLORS[k], label=f'K{k}: {ARCHETYPE_NAMES[k]}',
            markersize=8)
    ax.fill(angles, norm_list, alpha=0.12, color=CLUSTER_COLORS[k])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_labs, fontsize=10, fontweight='bold')
ax.set_ylim(0, 1)
ax.set_title('Multi-Dimensional Cluster Characteristics (KMeans K=4)',
             fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=10)
ax.grid(True)

plt.tight_layout()
savefig(fig, '05_cluster_characteristics_radar.png')

# ─────────────────────────────────────────────────────────────────────────────
# Figure 06 — PCA Projection
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n【FIGURE 06】KMeans PCA Projection …")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

ax = axes[0]
for k in range(4):
    mask = labels == k
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=CLUSTER_COLORS[k], label=f'K{k}: {ARCHETYPE_NAMES[k]} (n={mask.sum():,})',
               alpha=0.55, s=40, edgecolors='none')
# Plot cluster centres in PCA space
# kmeans.cluster_centers_ has shape (n_clusters, X_cluster.shape[1]) because
# KMeans was fitted on X_cluster, so pca2.transform can be applied directly.
centres_pca = pca2.transform(kmeans.cluster_centers_)
ax.scatter(centres_pca[:, 0], centres_pca[:, 1],
           marker='*', s=300, c='black', zorder=5, label='Centroids')
ax.set_xlabel(f'PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)', fontweight='bold')
ax.set_ylabel(f'PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)', fontweight='bold')
ax.set_title('(a) Vehicle Clusters — PCA Projection', fontweight='bold')
ax.legend(fontsize=9, loc='best')
ax.grid(alpha=0.25)

# Size pie chart
ax2 = axes[1]
sizes = [(labels == k).sum() for k in range(4)]
explode = [0.04] * 4
wedges, texts, autotexts = ax2.pie(
    sizes, explode=explode, labels=[f'K{k}\n{ARCHETYPE_NAMES[k]}' for k in range(4)],
    colors=CLUSTER_COLORS, autopct='%1.1f%%',
    startangle=90, pctdistance=0.78,
    textprops={'fontsize': 10},
    wedgeprops=dict(linewidth=1.5, edgecolor='white'))
for at in autotexts:
    at.set_fontsize(10)
    at.set_fontweight('bold')
ax2.set_title('(b) Cluster Size Distribution', fontweight='bold')

plt.suptitle('KMeans K=4 Clustering Results — PCA Projection & Distribution',
             fontsize=14, fontweight='bold')
plt.tight_layout()
savefig(fig, '06_kmeans_pca_projection.png')

# ─────────────────────────────────────────────────────────────────────────────
# Figure 07 — Detailed Cluster Profiles
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n【FIGURE 07】Cluster Profiles Detailed …")

fig = plt.figure(figsize=(20, 14))
gs  = gridspec.GridSpec(3, 4, hspace=0.45, wspace=0.35)

# Row 0: Segment mode composition stacked bars
for k in range(4):
    ax = fig.add_subplot(gs[0, k])
    vals = [df_veh[labels == k][f'dist_C{c}_ratio'].mean() for c in range(N_MODES)]
    bars = ax.bar(range(N_MODES), vals, color=SEG_COLORS, alpha=0.88,
                  edgecolor='black', linewidth=0.8)
    for bar, v in zip(bars, vals):
        if v > 0.04:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + 0.01, f'{v:.1%}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(range(N_MODES))
    ax.set_xticklabels(['C0', 'C1', 'C2', 'C3'], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Mean Ratio', fontweight='bold')
    n_k = int((labels == k).sum())
    ax.set_title(f'K{k}: {ARCHETYPE_NAMES[k]}\n(n={n_k:,}, {n_k/len(labels)*100:.1f}%)',
                 fontweight='bold', fontsize=10, color=CLUSTER_COLORS[k])
    ax.grid(axis='y', alpha=0.3)

# Row 1: Transition entropy vs self-transition ratio (scatter per vehicle)
for k in range(4):
    ax   = fig.add_subplot(gs[1, k])
    sub  = df_veh[labels == k]
    ax.scatter(sub['trans_entropy'], sub['trans_self_ratio'],
               c=CLUSTER_COLORS[k], alpha=0.45, s=20, edgecolors='none')
    ax.axvline(sub['trans_entropy'].median(), color='navy', ls='--', lw=1.5, label='Median')
    ax.axhline(sub['trans_self_ratio'].median(), color='gray', ls=':', lw=1.5)
    ax.set_xlabel('Transition Entropy', fontsize=9)
    ax.set_ylabel('Self-Trans. Ratio',  fontsize=9)
    ax.set_title(f'K{k} Transition Profile', fontweight='bold', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

# Row 2: Evolution kde-like (histogram) for switching frequency
for k in range(4):
    ax   = fig.add_subplot(gs[2, k])
    data = df_veh[labels == k]['evo_switching_freq'].dropna()
    ax.hist(data, bins=30, color=CLUSTER_COLORS[k], alpha=0.75,
            edgecolor='black', linewidth=0.5, density=True)
    ax.axvline(data.median(), color='navy',  ls='--', lw=2, label=f'Median={data.median():.3f}')
    ax.axvline(data.mean(),   color='red',   ls=':',  lw=2, label=f'Mean={data.mean():.3f}')
    ax.set_xlabel('Mode-Switching Frequency', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.set_title(f'K{k} Switching Freq Distribution', fontweight='bold', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

plt.suptitle('Detailed Cluster Profile Analysis — Distribution · Transition · Evolution',
             fontsize=15, fontweight='bold', y=1.01)
savefig(fig, '07_cluster_profiles_detailed.png')

# ─────────────────────────────────────────────────────────────────────────────
# Figure 08 — Vehicle Archetype Summary
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n【FIGURE 08】Vehicle Archetype Summary …")

fig, ax = plt.subplots(figsize=(18, 10))
ax.axis('off')

col_headers = ['Cluster', 'Archetype Name', 'Count (%)',
               'Dist: C0 | C1 | C2 | C3',
               'Trans Entropy', 'Switch Freq', 'Stability', 'Rhythm', 'Description']

table_data = [col_headers]
for k in range(4):
    mask = labels == k
    sub  = df_veh[mask]
    n_k  = int(mask.sum())
    dist_str = (f"{sub['dist_C0_ratio'].mean():.2f} | "
                f"{sub['dist_C1_ratio'].mean():.2f} | "
                f"{sub['dist_C2_ratio'].mean():.2f} | "
                f"{sub['dist_C3_ratio'].mean():.2f}")
    row = [
        f'K{k}',
        ARCHETYPE_NAMES[k],
        f"{n_k:,} ({n_k/len(labels)*100:.1f}%)",
        dist_str,
        f"{sub['trans_entropy'].mean():.3f}",
        f"{sub['evo_switching_freq'].mean():.3f}",
        f"{sub['evo_stability_index'].mean():.4f}",
        f"{sub['evo_rhythm_regularity'].mean():.3f}",
        ARCHETYPE_DESC[k],
    ]
    table_data.append(row)

# Determine column widths
col_widths = [0.04, 0.14, 0.08, 0.15, 0.07, 0.07, 0.07, 0.07, 0.31]

y_pos  = 0.92
x_start = 0.01
row_h  = 0.14

for ri, row in enumerate(table_data):
    is_header = (ri == 0)
    bg_color  = '#2C3E50' if is_header else (CLUSTER_COLORS[ri - 1] + '33')  # '33' = 20% opacity in hex
    txt_color = 'white' if is_header else 'black'
    fontw     = 'bold'

    x = x_start
    for ci, (cell, w) in enumerate(zip(row, col_widths)):
        rect = FancyBboxPatch((x, y_pos - row_h), w - 0.003, row_h,
                              boxstyle="round,pad=0.005",
                              facecolor=bg_color, edgecolor='white', linewidth=1.5,
                              transform=ax.transAxes, clip_on=False)
        ax.add_patch(rect)
        ax.text(x + w / 2, y_pos - row_h / 2, str(cell),
                transform=ax.transAxes,
                ha='center', va='center',
                fontsize=8.5 if ci == len(row) - 1 else 9,
                fontweight=fontw, color=txt_color,
                wrap=True)
        x += w

    y_pos -= row_h + 0.01

ax.set_title('Vehicle Archetype Summary — KMeans K=4 Final Clustering',
             fontsize=16, fontweight='bold', pad=20)
fig.text(0.5, 0.01,
         f"Total vehicles: {len(df_veh):,}  |  "
         f"Silhouette={sil:.4f}  |  Davies-Bouldin={db:.4f}  |  "
         f"Calinski-Harabasz={ch:.1f}  |  "
         f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
         ha='center', fontsize=9, color='gray')
savefig(fig, '08_vehicle_archetype_summary.png')

# ─────────────────────────────────────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'=' * 80}")
print(f"✅  Step 13 Complete!")
print(f"{'=' * 80}")
print(f"\n📁  Output Directory: {SAVE_DIR}")
print(f"    vehicle_clustering_kmeans_final_k4.csv")
print(f"    clustering_k4_final_summary.json")
print(f"    clustering_k4_final_report.txt")
print(f"    feature_matrices_for_analysis.npz")
print(f"\n📊  Figures: {FIG_DIR}")
for n in ['01_three_dimensions_overview.png',
          '02_distribution_features_heatmap.png',
          '03_transition_matrices_4clusters.png',
          '04_evolution_features_comparison.png',
          '05_cluster_characteristics_radar.png',
          '06_kmeans_pca_projection.png',
          '07_cluster_profiles_detailed.png',
          '08_vehicle_archetype_summary.png']:
    print(f"    {n}")
print(f"\n🎉  Run:  python vehicle_clustering/step13_kmeans_final_publication.py")
