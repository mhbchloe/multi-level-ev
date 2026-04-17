"""
Step 14: Four-Archetype Charging Pattern Analysis (KMeans K=4)
四类车辆充电模式对比分析

Based on KMeans K=4 clustering of vehicle features using three-dimension framework:
  ① Distribution: cluster_0~3 ratios (idle / urban / highway / mixed)
  ② Transition:   4×4 transition matrix (how modes switch)
  ③ Evolution:    temporal accumulation, rhythm, stability

Outputs:
  vehicle_clustering/results/charging_archetype_analysis/
    charging_characteristics_c0.csv  (per-vehicle charging stats for archetype C0)
    charging_characteristics_c1.csv
    charging_characteristics_c2.csv
    charging_characteristics_c3.csv
    charging_archetype_summary.json
    grid_impact_analysis.json
    charging_archetype_report.txt

  vehicle_clustering/results/figures_charging_archetypes/
    01_four_archetypes_overview.png
    02_charging_demand_profile.png
    03_grid_impact_comparison.png
    04_charging_infrastructure_requirement.png
"""

import numpy as np
import pandas as pd
import os
import json
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.stats import entropy as scipy_entropy
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ── publication-quality global settings ──────────────────────────────────────
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
rcParams['font.size'] = 11
rcParams['axes.linewidth'] = 1.2
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
rcParams['xtick.major.width'] = 1.2
rcParams['ytick.major.width'] = 1.2
rcParams['figure.dpi'] = 150

print("=" * 80)
print("🔋 STEP 14: FOUR-ARCHETYPE CHARGING PATTERN ANALYSIS (KMeans K=4)")
print("=" * 80)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    'vehicle_features_path': './coupling_analysis/results/vehicles_aggregated_features.csv',
    'segments_path':         './coupling_analysis/results/segments_integrated_complete.csv',
    'save_dir':              './vehicle_clustering/results/charging_archetype_analysis/',
    'fig_dir':               './vehicle_clustering/results/figures_charging_archetypes/',
    'seed': 42,
    'n_clusters': 4,
}

os.makedirs(CONFIG['save_dir'], exist_ok=True)
os.makedirs(CONFIG['fig_dir'], exist_ok=True)

# Archetype color palette (publication-friendly)
ARCHETYPE_COLORS = {
    'C0': '#4C72B0',   # blue   – Mixed-Pattern
    'C1': '#DD8452',   # orange – Highway-Intensive
    'C2': '#55A868',   # green  – Idle-Stable
    'C3': '#C44E52',   # red    – Urban-Dynamic
}
ARCHETYPE_LABELS = {
    'C0': 'C0: Mixed-Pattern',
    'C1': 'C1: Highway-Intensive',
    'C2': 'C2: Idle-Stable',
    'C3': 'C3: Urban-Dynamic',
}
ARCHETYPE_ROLES = {
    'C0': 'Mixed-Pattern Charger',
    'C1': 'Highway Power Charger',
    'C2': 'Deep-Idle Charger',
    'C3': 'Urban-Quick Charger',
}

# ── Thresholds & normalization constants ─────────────────────────────────────
# A SOC rise of at least 2 percentage-points within a segment is treated as a
# charging event (distinguishes genuine top-ups from sensor noise).
CHARGING_SOC_THRESHOLD    = 2.0

# A SOC drop of more than 1 percentage-point marks the segment as discharging.
DISCHARGING_SOC_THRESHOLD = -1.0

# In a 4×4 mode-transition matrix, 16 cells exist; this divisor normalises
# "transition diversity" to the [0, 1] range.
TOTAL_TRANSITIONS = 16.0

# Threshold above which a single transition count is counted as "present"
# (used in transition_diversity calculation).
TRANSITION_THRESHOLD = 0.5

# Composite grid-stress score weights: peak demand (0.4), uncontrollability (0.3),
# peak-valley ratio (0.3).  Peak demand is normalised against an expected maximum
# of 5×; peak-valley ratio against an expected maximum of 20.
GRID_STRESS_WEIGHTS  = (0.4, 0.3, 0.3)
PEAK_DEMAND_MAX      = 5.0    # normalisation ceiling for peak_demand_index
PEAK_VALLEY_MAX      = 20.0   # normalisation ceiling for peak_valley_ratio

# Radar chart: peak-power-handling dimension is normalised by this factor so
# that a peak_demand_index of ~5× maps to a score of 1.0.
PEAK_POWER_NORMALIZATION_FACTOR = 5.0

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1  Load data
# ─────────────────────────────────────────────────────────────────────────────
print("\n【STEP 1】Loading Data")
print("-" * 80)

vehicle_features = pd.read_csv(CONFIG['vehicle_features_path'])
segments_df      = pd.read_csv(CONFIG['segments_path'])

print(f"   ✓ Vehicle features : {len(vehicle_features):,} vehicles, "
      f"{len(vehicle_features.columns)} columns")
print(f"   ✓ Segments         : {len(segments_df):,} segments")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2  Extract three-dimension features
# ─────────────────────────────────────────────────────────────────────────────
print("\n【STEP 2】Extracting Three-Dimension Features")
print("-" * 80)

features_dict = {}

for vid in tqdm(vehicle_features['vehicle_id'].values, desc="   Feature extraction"):
    v_segs = segments_df[segments_df['vehicle_id'] == vid].sort_values('segment_id')

    if len(v_segs) < 2:
        continue

    clusters  = v_segs['cluster_id'].values.astype(int)
    durations = v_segs['duration_seconds'].values
    n_segs    = len(clusters)
    total_dur = durations.sum()

    # ── ① Distribution (4 ratios + entropy + dominant ratio) ──────────────
    dist = {}
    for c in range(4):
        dist[f'cluster_{c}_ratio'] = float((clusters == c).sum() / n_segs)

    dist_vals = np.array([dist[f'cluster_{c}_ratio'] for c in range(4)])
    dist['distribution_entropy']  = float(scipy_entropy(dist_vals + 1e-9))
    dist['dominant_mode_ratio']   = float(dist_vals.max())

    # ── ② Transition (4×4 matrix + summary stats) ─────────────────────────
    T = np.zeros((4, 4), dtype=float)
    for i in range(n_segs - 1):
        T[clusters[i], clusters[i + 1]] += 1.0

    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    T_norm = T / row_sums

    trans = {}
    for r in range(4):
        for c in range(4):
            trans[f'trans_{r}_to_{c}'] = float(T_norm[r, c])

    flat = T_norm.flatten()
    trans['transition_entropy']    = float(scipy_entropy(flat + 1e-9))
    trans['self_transition_ratio'] = float(np.trace(T_norm) / 4.0)
    trans['transition_diversity']  = float(np.count_nonzero(T > TRANSITION_THRESHOLD) / TOTAL_TRANSITIONS)
    trans['dominant_transition']   = float(flat.max())

    # ── ③ Evolution (temporal + stability) ────────────────────────────────
    switches   = (clusters[:-1] != clusters[1:]).sum()
    switch_freq = switches / (n_segs - 1) if n_segs > 1 else 0.0

    # stability = longest run / n_segs
    max_run = 1
    cur_run = 1
    for i in range(1, n_segs):
        if clusters[i] == clusters[i - 1]:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 1
    stability = max_run / n_segs

    # rhythm regularity: coefficient of variation of run lengths
    run_lengths = []
    cur_run = 1
    for i in range(1, n_segs):
        if clusters[i] == clusters[i - 1]:
            cur_run += 1
        else:
            run_lengths.append(cur_run)
            cur_run = 1
    run_lengths.append(cur_run)
    rl = np.array(run_lengths, dtype=float)
    rhythm = 1.0 / (1.0 + (rl.std() / (rl.mean() + 1e-9)))

    evo = {
        'sequence_length':      float(n_segs),
        'mode_switching_freq':  float(switch_freq),
        'stability_index':      float(stability),
        'rhythm_regularity':    float(rhythm),
        'cumulative_hours':     float(total_dur / 3600.0),
    }
    for c in range(4):
        evo[f'cumulative_hours_C{c}'] = float(
            durations[clusters == c].sum() / 3600.0
        )

    features_dict[vid] = {**dist, **trans, **evo}

features_df = pd.DataFrame.from_dict(features_dict, orient='index')
features_df.index.name = 'vehicle_id'
features_df = features_df.reset_index()

print(f"   ✓ {len(features_df):,} vehicles × {len(features_df.columns) - 1} features extracted")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3  Standardise features and run KMeans K=4
# ─────────────────────────────────────────────────────────────────────────────
print("\n【STEP 3】KMeans K=4 Clustering")
print("-" * 80)

feat_cols = [c for c in features_df.columns if c != 'vehicle_id']
X = features_df[feat_cols].fillna(0).values.astype(np.float64)

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

# Remove near-zero variance features
var    = X_scaled.var(axis=0)
active = var > 1e-8
X_clus = X_scaled[:, active]
print(f"   ✓ Active features after variance filter: {active.sum()} / {len(active)}")

kmeans = KMeans(n_clusters=CONFIG['n_clusters'], n_init=20,
                max_iter=500, random_state=CONFIG['seed'])
raw_labels = kmeans.fit_predict(X_clus)

sil = silhouette_score(X_clus, raw_labels,
                       sample_size=min(5000, len(X_clus)),
                       random_state=CONFIG['seed'])
ch  = calinski_harabasz_score(X_clus, raw_labels)
db  = davies_bouldin_score(X_clus, raw_labels)

print(f"   ✓ Silhouette={sil:.4f}  |  Calinski-Harabasz={ch:.1f}  |  "
      f"Davies-Bouldin={db:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4  Map raw cluster indices → C0/C1/C2/C3 archetypes
#   C2: highest idle ratio (停歇占比最高)
#   C1: highest highway ratio (高速占比最高)
#   C3: highest urban ratio (城市占比最高)
#   C0: remaining (mixed)
# ─────────────────────────────────────────────────────────────────────────────
print("\n【STEP 4】Mapping Clusters to Archetypes")
print("-" * 80)

cluster_means = {}
for k in range(CONFIG['n_clusters']):
    mask = raw_labels == k
    cluster_means[k] = {
        'idle_ratio':    features_df.loc[mask, 'cluster_0_ratio'].mean(),
        'urban_ratio':   features_df.loc[mask, 'cluster_1_ratio'].mean(),
        'highway_ratio': features_df.loc[mask, 'cluster_2_ratio'].mean(),
        'switch_freq':   features_df.loc[mask, 'mode_switching_freq'].mean(),
        'stability':     features_df.loc[mask, 'stability_index'].mean(),
        'size':          int(mask.sum()),
    }

# Assign archetypes greedily by strongest characteristic
remaining = set(range(CONFIG['n_clusters']))
archetype_map = {}   # raw_k → 'C0' | 'C1' | 'C2' | 'C3'

# C2: highest idle_ratio
best = max(remaining, key=lambda k: cluster_means[k]['idle_ratio'])
archetype_map[best] = 'C2'
remaining.remove(best)

# C1: highest highway_ratio
best = max(remaining, key=lambda k: cluster_means[k]['highway_ratio'])
archetype_map[best] = 'C1'
remaining.remove(best)

# C3: highest urban_ratio (among remaining)
best = max(remaining, key=lambda k: cluster_means[k]['urban_ratio'])
archetype_map[best] = 'C3'
remaining.remove(best)

# C0: the last one (mixed)
archetype_map[remaining.pop()] = 'C0'

# Apply mapping
archetype_labels = np.array([archetype_map[k] for k in raw_labels])
features_df['archetype'] = archetype_labels

print("   Archetype mapping:")
for raw_k, arch in sorted(archetype_map.items(), key=lambda x: x[1]):
    cm = cluster_means[raw_k]
    print(f"     raw-{raw_k} → {arch}  "
          f"(n={cm['size']:,}  idle={cm['idle_ratio']:.2f}  "
          f"urban={cm['urban_ratio']:.2f}  "
          f"highway={cm['highway_ratio']:.2f}  "
          f"switch={cm['switch_freq']:.2f})")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5  Extract charging characteristics from segments
# ─────────────────────────────────────────────────────────────────────────────
print("\n【STEP 5】Extracting Charging Characteristics from Segments")
print("-" * 80)

# Build vehicle → archetype lookup
vid_to_arch = dict(zip(features_df['vehicle_id'], features_df['archetype']))

# Parse timestamps (flexible – try common formats)
for dt_col in ['start_dt', 'end_dt']:
    if dt_col in segments_df.columns:
        segments_df[dt_col] = pd.to_datetime(segments_df[dt_col], errors='coerce')

# Detect charging events: SOC rises (or large SOC recovery within a segment)
segments_df['soc_delta'] = segments_df['soc_end'] - segments_df['soc_start']
charging_mask    = segments_df['soc_delta'] > CHARGING_SOC_THRESHOLD    # > 2 SOC-points rise
discharging_mask = segments_df['soc_delta'] < DISCHARGING_SOC_THRESHOLD

charging_events  = segments_df[charging_mask].copy()
discharging_segs = segments_df[discharging_mask].copy()

print(f"   ✓ Charging events detected  : {len(charging_events):,}")
print(f"   ✓ Discharging segments       : {len(discharging_segs):,}")

def extract_charging_features(vehicle_id, v_charge, v_discharge):
    """Derive per-vehicle charging metrics."""
    feats = {'vehicle_id': vehicle_id}

    if len(v_charge) == 0:
        feats['n_charge_events']       = 0
        feats['avg_charge_duration_h'] = np.nan
        feats['total_charge_time_h']   = 0.0
        feats['avg_soc_gain']          = np.nan
        feats['peak_soc_gain']         = np.nan
        feats['charge_rate_soc_per_h'] = np.nan
        feats['charge_entropy']        = np.nan
        feats['charge_regularity']     = np.nan
        feats['discharge_rate_soc_per_h'] = np.nan
        feats['peak_discharge_rate']   = np.nan
        for h in range(24):
            feats[f'charge_hour_{h}'] = 0.0
        return feats

    # Basic charging stats
    feats['n_charge_events'] = len(v_charge)

    dur_h = v_charge['duration_seconds'] / 3600.0
    feats['avg_charge_duration_h'] = float(dur_h.mean())
    feats['total_charge_time_h']   = float(dur_h.sum())

    feats['avg_soc_gain']  = float(v_charge['soc_delta'].mean())
    feats['peak_soc_gain'] = float(v_charge['soc_delta'].max())

    # Charging rate (SOC-points per hour)
    rates = v_charge['soc_delta'] / (dur_h.replace(0, np.nan))
    feats['charge_rate_soc_per_h'] = float(rates.mean())

    # Hourly distribution of charging start times
    hour_counts = np.zeros(24)
    if 'start_dt' in v_charge.columns:
        valid_times = v_charge['start_dt'].dropna()
        if len(valid_times) > 0:
            for h in valid_times.dt.hour:
                hour_counts[int(h)] += 1

    feats['charge_entropy'] = float(
        scipy_entropy(hour_counts / (hour_counts.sum() + 1e-9) + 1e-9)
    )
    # Regularity: inverse of normalised entropy
    max_ent = np.log(24)
    feats['charge_regularity'] = float(1.0 - feats['charge_entropy'] / max_ent)

    for h in range(24):
        feats[f'charge_hour_{h}'] = float(hour_counts[h])

    # Discharge rate (proxy for driving demand on battery)
    if len(v_discharge) > 0:
        d_dur_h = v_discharge['duration_seconds'] / 3600.0
        d_rates = v_discharge['soc_delta'].abs() / (d_dur_h.replace(0, np.nan))
        feats['discharge_rate_soc_per_h'] = float(d_rates.mean())
        feats['peak_discharge_rate']       = float(d_rates.max())
    else:
        feats['discharge_rate_soc_per_h'] = np.nan
        feats['peak_discharge_rate']       = np.nan

    return feats


# Group by vehicle
charge_by_vid    = charging_events.groupby('vehicle_id')
discharge_by_vid = discharging_segs.groupby('vehicle_id')

all_charge_feats = []
for vid in tqdm(features_df['vehicle_id'].values, desc="   Charging feature extraction"):
    v_charge    = charge_by_vid.get_group(vid) if vid in charge_by_vid.groups else pd.DataFrame()
    v_discharge = discharge_by_vid.get_group(vid) if vid in discharge_by_vid.groups else pd.DataFrame()
    feat = extract_charging_features(vid, v_charge, v_discharge)
    feat['archetype'] = vid_to_arch.get(vid, 'C0')
    all_charge_feats.append(feat)

charge_feats_df = pd.DataFrame(all_charge_feats)
print(f"   ✓ Charging features extracted for {len(charge_feats_df):,} vehicles")

# Save per-archetype CSV files
for arch in ['C0', 'C1', 'C2', 'C3']:
    arch_df = charge_feats_df[charge_feats_df['archetype'] == arch].copy()
    path = os.path.join(CONFIG['save_dir'], f'charging_characteristics_{arch.lower()}.csv')
    arch_df.to_csv(path, index=False)
    print(f"   ✓ Saved: charging_characteristics_{arch.lower()}.csv "
          f"({len(arch_df):,} vehicles)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6  Compute per-archetype aggregate statistics
# ─────────────────────────────────────────────────────────────────────────────
print("\n【STEP 6】Computing Archetype Aggregate Statistics")
print("-" * 80)

hour_cols = [f'charge_hour_{h}' for h in range(24)]

arch_stats = {}
for arch in ['C0', 'C1', 'C2', 'C3']:
    sub = charge_feats_df[charge_feats_df['archetype'] == arch]
    n_vehicles = len(sub)

    # Charging frequency (events per vehicle per day)
    # Use total charge events / total cumulative hours * 24
    total_hours_col = 'total_charge_time_h'
    total_drive_h = (
        features_df[features_df['archetype'] == arch]['cumulative_hours'].mean()
    )
    avg_events = sub['n_charge_events'].mean()
    charge_freq = avg_events / (total_drive_h / 24.0 + 1e-9)  # events/day

    # Hourly demand profile (averaged across vehicles, normalised)
    hour_profile = sub[hour_cols].mean().values
    hour_profile_norm = hour_profile / (hour_profile.sum() + 1e-9)

    # Grid-impact proxies
    avg_charge_rate = sub['charge_rate_soc_per_h'].median()
    peak_charge_rate = sub['charge_rate_soc_per_h'].quantile(0.95)
    rate_std = sub['charge_rate_soc_per_h'].std()
    peak_demand_index = float(peak_charge_rate / (avg_charge_rate + 1e-9))

    # Peak-to-valley ratio on hourly profile
    p_max = hour_profile_norm.max()
    p_min = hour_profile_norm.min()
    peak_valley_ratio = float(p_max / (p_min + 1e-9))

    # Controllability: based on regularity
    controllability = float(sub['charge_regularity'].median())

    # Infrastructure proxies
    fast_charge_ratio  = float(
        (sub['charge_rate_soc_per_h'] > sub['charge_rate_soc_per_h'].quantile(0.5)).mean()
    )
    slow_charge_ratio  = 1.0 - fast_charge_ratio
    coverage_need      = 1.0 - controllability        # irregular → more locations needed
    temporal_regularity = controllability
    peak_power_handling = float(peak_demand_index)

    # Transition features for archetype
    arch_vids = features_df[features_df['archetype'] == arch]['vehicle_id']
    arch_trans = features_df[features_df['archetype'] == arch]

    arch_stats[arch] = {
        'n_vehicles':           n_vehicles,
        'pct':                  100.0 * n_vehicles / len(features_df),
        'charge_freq_per_day':  float(charge_freq),
        'avg_charge_duration_h': float(sub['avg_charge_duration_h'].median()),
        'avg_soc_gain':         float(sub['avg_soc_gain'].median()),
        'charge_regularity':    float(sub['charge_regularity'].median()),
        'charge_entropy':       float(sub['charge_entropy'].median()),
        'avg_charge_rate':      float(avg_charge_rate),
        'peak_charge_rate':     float(peak_charge_rate),
        'charge_rate_std':      float(rate_std),
        'peak_demand_index':    peak_demand_index,
        'peak_valley_ratio':    peak_valley_ratio,
        'controllability':      controllability,
        'fast_charge_ratio':    fast_charge_ratio,
        'slow_charge_ratio':    slow_charge_ratio,
        'coverage_need':        coverage_need,
        'temporal_regularity':  temporal_regularity,
        'peak_power_handling':  peak_power_handling,
        'hour_profile':         hour_profile_norm.tolist(),
        'idle_ratio':           float(arch_trans['cluster_0_ratio'].mean()),
        'urban_ratio':          float(arch_trans['cluster_1_ratio'].mean()),
        'highway_ratio':        float(arch_trans['cluster_2_ratio'].mean()),
        'switch_freq':          float(arch_trans['mode_switching_freq'].mean()),
        'stability_index':      float(arch_trans['stability_index'].mean()),
        'cumulative_hours':     float(arch_trans['cumulative_hours'].mean()),
    }

    print(f"   {arch}: n={n_vehicles:,} ({arch_stats[arch]['pct']:.1f}%)  "
          f"freq={charge_freq:.2f}/day  dur={arch_stats[arch]['avg_charge_duration_h']:.2f}h  "
          f"regularity={controllability:.2f}")

# Save summary JSON
summary_path = os.path.join(CONFIG['save_dir'], 'charging_archetype_summary.json')
with open(summary_path, 'w', encoding='utf-8') as f:
    json.dump(arch_stats, f, indent=2, ensure_ascii=False)
print(f"\n   ✓ Saved: charging_archetype_summary.json")

# Grid impact JSON
grid_impact = {}
for arch in ['C0', 'C1', 'C2', 'C3']:
    grid_impact[arch] = {
        'role':                   ARCHETYPE_ROLES[arch],
        'peak_demand_index':      arch_stats[arch]['peak_demand_index'],
        'charge_rate_volatility': arch_stats[arch]['charge_rate_std'],
        'peak_valley_ratio':      arch_stats[arch]['peak_valley_ratio'],
        'controllability':        arch_stats[arch]['controllability'],
        'grid_stress_score': float(
            GRID_STRESS_WEIGHTS[0] * arch_stats[arch]['peak_demand_index'] / PEAK_DEMAND_MAX +
            GRID_STRESS_WEIGHTS[1] * (1.0 - arch_stats[arch]['controllability']) +
            GRID_STRESS_WEIGHTS[2] * min(arch_stats[arch]['peak_valley_ratio'] / PEAK_VALLEY_MAX, 1.0)
        ),
    }

grid_path = os.path.join(CONFIG['save_dir'], 'grid_impact_analysis.json')
with open(grid_path, 'w', encoding='utf-8') as f:
    json.dump(grid_impact, f, indent=2, ensure_ascii=False)
print(f"   ✓ Saved: grid_impact_analysis.json")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7  Visualization — Figure 1: Four Archetypes Overview
# ─────────────────────────────────────────────────────────────────────────────
print("\n【STEP 7】Figure 1: Four Archetypes Overview")
print("-" * 80)

archs = ['C0', 'C1', 'C2', 'C3']
colors_list = [ARCHETYPE_COLORS[a] for a in archs]
arch_name_list = [ARCHETYPE_LABELS[a] for a in archs]

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
axes = axes.flatten()

# ── Sub-figure 1a: Charging frequency per day ─────────────────────────────
ax = axes[0]
vals = [arch_stats[a]['charge_freq_per_day'] for a in archs]
bars = ax.bar(archs, vals, color=colors_list, edgecolor='black',
              linewidth=1.2, width=0.55, zorder=3)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2, v + max(vals) * 0.02,
            f'{v:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylabel('Charging Events per Day', fontsize=12, fontweight='bold')
ax.set_title('(a) Charging Frequency', fontsize=13, fontweight='bold')
ax.set_xticks(range(4))
ax.set_xticklabels(arch_name_list, fontsize=9, rotation=12)
ax.yaxis.grid(True, alpha=0.3, zorder=0)
ax.set_axisbelow(True)

# ── Sub-figure 1b: Average charging duration ─────────────────────────────
ax = axes[1]
vals = [arch_stats[a]['avg_charge_duration_h'] for a in archs]
bars = ax.bar(archs, vals, color=colors_list, edgecolor='black',
              linewidth=1.2, width=0.55, zorder=3)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2, v + max(vals) * 0.02,
            f'{v:.2f}h', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylabel('Average Charging Duration (hours)', fontsize=12, fontweight='bold')
ax.set_title('(b) Average Charging Duration', fontsize=13, fontweight='bold')
ax.set_xticks(range(4))
ax.set_xticklabels(arch_name_list, fontsize=9, rotation=12)
ax.yaxis.grid(True, alpha=0.3, zorder=0)
ax.set_axisbelow(True)

# ── Sub-figure 1c: Time-slot regularity (entropy) ────────────────────────
ax = axes[2]
vals_reg = [arch_stats[a]['charge_regularity'] for a in archs]
vals_ent = [arch_stats[a]['charge_entropy'] for a in archs]
x = np.arange(4)
width = 0.35
b1 = ax.bar(x - width / 2, vals_reg, width, label='Regularity (↑ better)',
            color=colors_list, edgecolor='black', linewidth=1.2, zorder=3)
b2 = ax.bar(x + width / 2, vals_ent, width, label='Entropy (↓ more regular)',
            color=colors_list, edgecolor='black', linewidth=1.2, alpha=0.45, zorder=3,
            hatch='//')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('(c) Charging Time-Slot Regularity', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(arch_name_list, fontsize=9, rotation=12)
ax.legend(fontsize=9)
ax.yaxis.grid(True, alpha=0.3, zorder=0)
ax.set_axisbelow(True)

# ── Sub-figure 1d: Driving mode composition stacked bar ──────────────────
ax = axes[3]
mode_labels   = ['Idle (C0-seg)', 'Urban (C1-seg)', 'Highway (C2-seg)', 'Mixed (C3-seg)']
mode_colors   = ['#A8D8EA', '#AA96DA', '#FCBAD3', '#FFFFD2']
idle_vals     = [arch_stats[a]['idle_ratio']    for a in archs]
urban_vals    = [arch_stats[a]['urban_ratio']   for a in archs]
highway_vals  = [arch_stats[a]['highway_ratio'] for a in archs]
mixed_vals    = [
    1.0 - idle_vals[i] - urban_vals[i] - highway_vals[i]
    for i in range(4)
]
mixed_vals = [max(0, v) for v in mixed_vals]

x = np.arange(4)
bottom = np.zeros(4)
for mode_v, mc, ml in zip(
    [idle_vals, urban_vals, highway_vals, mixed_vals],
    mode_colors, mode_labels
):
    ax.bar(x, mode_v, bottom=bottom, color=mc, edgecolor='black',
           linewidth=0.8, label=ml, zorder=3)
    bottom += np.array(mode_v)

ax.set_ylabel('Proportion', fontsize=12, fontweight='bold')
ax.set_title('(d) Driving Mode Composition', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(arch_name_list, fontsize=9, rotation=12)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=9, loc='lower right', ncol=2)
ax.yaxis.grid(True, alpha=0.3, zorder=0)
ax.set_axisbelow(True)

plt.suptitle(
    'Figure 1: Charging Archetype Overview — Four Vehicle Types\n'
    '(KMeans K=4 | Distribution × Transition × Evolution Features)',
    fontsize=14, fontweight='bold', y=1.01
)
plt.tight_layout()
fig_path = os.path.join(CONFIG['fig_dir'], '01_four_archetypes_overview.png')
fig.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: 01_four_archetypes_overview.png")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8  Visualization — Figure 2: Charging Demand Time Profile
# ─────────────────────────────────────────────────────────────────────────────
print("\n【STEP 8】Figure 2: Charging Demand Time Profile")
print("-" * 80)

hours = np.arange(24)
fig, ax = plt.subplots(figsize=(13, 6))

line_styles = ['-', '--', '-.', ':']
markers     = ['o', 's', '^', 'D']

for i, arch in enumerate(archs):
    profile = np.array(arch_stats[arch]['hour_profile'])
    # Smooth with a rolling average (window=3, circular)
    profile_smooth = np.convolve(
        np.tile(profile, 3), np.ones(3) / 3, mode='same'
    )[24:48]

    n = arch_stats[arch]['n_vehicles']
    ax.plot(hours, profile_smooth * 100,
            color=ARCHETYPE_COLORS[arch],
            linestyle=line_styles[i],
            linewidth=2.2,
            marker=markers[i], markersize=5,
            label=f"{ARCHETYPE_LABELS[arch]} (n={n:,})",
            zorder=3)

# Shade night/off-peak period
ax.axvspan(0, 6,   alpha=0.07, color='navy', label='Night (00–06 h)')
ax.axvspan(22, 24, alpha=0.07, color='navy')
ax.axvspan(8, 12,  alpha=0.05, color='red',  label='Morning peak (08–12 h)')
ax.axvspan(17, 20, alpha=0.05, color='orange', label='Evening peak (17–20 h)')

ax.set_xlabel('Hour of Day', fontsize=13, fontweight='bold')
ax.set_ylabel('Normalised Charging Demand (%)', fontsize=13, fontweight='bold')
ax.set_title(
    'Figure 2: Charging Demand Temporal Distribution by Archetype\n'
    '(Proportion of daily charging events per hour)',
    fontsize=14, fontweight='bold'
)
ax.set_xticks(range(0, 24, 2))
ax.set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 2)], rotation=30, fontsize=9)
ax.legend(fontsize=10, loc='upper left', framealpha=0.9)
ax.yaxis.grid(True, alpha=0.3, zorder=0)
ax.set_axisbelow(True)

plt.tight_layout()
fig_path = os.path.join(CONFIG['fig_dir'], '02_charging_demand_profile.png')
fig.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: 02_charging_demand_profile.png")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9  Visualization — Figure 3: Grid Impact Comparison
# ─────────────────────────────────────────────────────────────────────────────
print("\n【STEP 9】Figure 3: Grid Impact Comparison")
print("-" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

# ── 3a: Peak power demand index ───────────────────────────────────────────
ax = axes[0]
vals = [arch_stats[a]['peak_demand_index'] for a in archs]
bars = ax.barh(arch_name_list, vals, color=colors_list, edgecolor='black',
               linewidth=1.2, height=0.55, zorder=3)
for bar, v in zip(bars, vals):
    ax.text(v + max(vals) * 0.01, bar.get_y() + bar.get_height() / 2,
            f'{v:.2f}×', va='center', fontsize=11, fontweight='bold')
ax.set_xlabel('Peak-to-Average Charging Rate Ratio', fontsize=11, fontweight='bold')
ax.set_title('(a) Peak Power Demand Index', fontsize=12, fontweight='bold')
ax.xaxis.grid(True, alpha=0.3, zorder=0)
ax.set_axisbelow(True)

# ── 3b: Charging rate volatility (std) ───────────────────────────────────
ax = axes[1]
vals = [arch_stats[a]['charge_rate_std'] for a in archs]
bars = ax.barh(arch_name_list, vals, color=colors_list, edgecolor='black',
               linewidth=1.2, height=0.55, zorder=3)
for bar, v in zip(bars, vals):
    ax.text(v + max(vals) * 0.01, bar.get_y() + bar.get_height() / 2,
            f'{v:.2f}', va='center', fontsize=11, fontweight='bold')
ax.set_xlabel('Std of Charging Rate (SOC %/h)', fontsize=11, fontweight='bold')
ax.set_title('(b) Charging Power Volatility', fontsize=12, fontweight='bold')
ax.xaxis.grid(True, alpha=0.3, zorder=0)
ax.set_axisbelow(True)

# ── 3c: Peak-to-valley ratio on daily demand curve ───────────────────────
ax = axes[2]
vals = [arch_stats[a]['peak_valley_ratio'] for a in archs]
bars = ax.barh(arch_name_list, vals, color=colors_list, edgecolor='black',
               linewidth=1.2, height=0.55, zorder=3)
for bar, v in zip(bars, vals):
    ax.text(v + max(vals) * 0.01, bar.get_y() + bar.get_height() / 2,
            f'{v:.1f}', va='center', fontsize=11, fontweight='bold')
ax.set_xlabel('Peak / Valley Ratio (hourly demand)', fontsize=11, fontweight='bold')
ax.set_title('(c) Daily Load Peak-to-Valley Ratio', fontsize=12, fontweight='bold')
ax.xaxis.grid(True, alpha=0.3, zorder=0)
ax.set_axisbelow(True)

# ── 3d: Controllability index ─────────────────────────────────────────────
ax = axes[3]
vals = [arch_stats[a]['controllability'] for a in archs]
bars = ax.barh(arch_name_list, vals, color=colors_list, edgecolor='black',
               linewidth=1.2, height=0.55, zorder=3)
for bar, v in zip(bars, vals):
    ax.text(v + max(vals) * 0.01, bar.get_y() + bar.get_height() / 2,
            f'{v:.2f}', va='center', fontsize=11, fontweight='bold')
ax.set_xlabel('Controllability Index (0–1, higher = more controllable)',
              fontsize=11, fontweight='bold')
ax.set_title('(d) Grid Controllability Index', fontsize=12, fontweight='bold')
ax.xaxis.grid(True, alpha=0.3, zorder=0)
ax.set_axisbelow(True)

plt.suptitle(
    'Figure 3: Grid Impact Comparison Across Charging Archetypes',
    fontsize=14, fontweight='bold', y=1.01
)
plt.tight_layout()
fig_path = os.path.join(CONFIG['fig_dir'], '03_grid_impact_comparison.png')
fig.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: 03_grid_impact_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 10  Visualization — Figure 4: Infrastructure Requirements (Radar)
# ─────────────────────────────────────────────────────────────────────────────
print("\n【STEP 10】Figure 4: Infrastructure Requirements Radar Chart")
print("-" * 80)

radar_dimensions = [
    'Fast\nCharging\nDemand',
    'Slow\nCharging\nDemand',
    'Location\nCoverage\nNeed',
    'Time-slot\nRegularity',
    'Peak Power\nHandling\nCapacity',
]
N_dim = len(radar_dimensions)
angles = np.linspace(0, 2 * np.pi, N_dim, endpoint=False).tolist()
angles += angles[:1]   # close the polygon


def get_radar_values(arch):
    s = arch_stats[arch]
    raw = [
        s['fast_charge_ratio'],            # Fast charging demand
        s['slow_charge_ratio'],            # Slow charging demand
        s['coverage_need'],                # Location coverage need
        s['temporal_regularity'],          # Time-slot regularity
        min(s['peak_power_handling'] / PEAK_POWER_NORMALIZATION_FACTOR, 1.0),  # Peak power (normalised)
    ]
    return [min(max(v, 0.0), 1.0) for v in raw]


fig = plt.figure(figsize=(13, 10))
fig.suptitle(
    'Figure 4: Charging Infrastructure Requirements by Archetype\n'
    '(Radar chart — values normalised 0–1)',
    fontsize=14, fontweight='bold', y=0.98
)

for idx, arch in enumerate(archs):
    ax = fig.add_subplot(2, 2, idx + 1, polar=True)

    values      = get_radar_values(arch)
    values_plot = values + values[:1]

    ax.plot(angles, values_plot, color=ARCHETYPE_COLORS[arch],
            linewidth=2.5, linestyle='-', zorder=3)
    ax.fill(angles, values_plot, color=ARCHETYPE_COLORS[arch], alpha=0.25, zorder=2)

    # Gridlines and labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_dimensions, fontsize=9, fontweight='bold')
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], fontsize=7, color='grey')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.35)

    n = arch_stats[arch]['n_vehicles']
    pct = arch_stats[arch]['pct']
    ax.set_title(
        f"{ARCHETYPE_LABELS[arch]}\n"
        f"({ARCHETYPE_ROLES[arch]})\n"
        f"n = {n:,}  ({pct:.1f}%)",
        fontsize=10, fontweight='bold', pad=16,
        color=ARCHETYPE_COLORS[arch]
    )

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig_path = os.path.join(CONFIG['fig_dir'], '04_charging_infrastructure_requirement.png')
fig.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: 04_charging_infrastructure_requirement.png")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 11  Generate comparison report
# ─────────────────────────────────────────────────────────────────────────────
print("\n【STEP 11】Generating Comparison Report")
print("-" * 80)

report_lines = []
report_lines.append("=" * 80)
report_lines.append("       FOUR-ARCHETYPE CHARGING ROLE COMPARISON ANALYSIS")
report_lines.append("       (KMeans K=4 | Three-Dimension Vehicle Features)")
report_lines.append("=" * 80)
report_lines.append("")

# ── 1. Clustering Quality ─────────────────────────────────────────────────
report_lines.append("1. CLUSTERING QUALITY (KMeans K=4)")
report_lines.append("-" * 80)
report_lines.append(f"   Silhouette Score      : {sil:.4f}")
report_lines.append(f"   Calinski-Harabasz     : {ch:.1f}")
report_lines.append(f"   Davies-Bouldin        : {db:.4f} (lower = better)")
report_lines.append("")

# ── 2. Charging Mode Comparison ───────────────────────────────────────────
report_lines.append("2. CHARGING MODE COMPARISON")
report_lines.append("-" * 80)
header = f"  {'Metric':<32}  {'C0':>10}  {'C1':>10}  {'C2':>10}  {'C3':>10}"
report_lines.append(header)
report_lines.append("  " + "-" * (len(header) - 2))

metrics_map = [
    ('Vehicles (n)',          'n_vehicles',           '{:>10,.0f}'),
    ('Share (%)',             'pct',                  '{:>10.1f}'),
    ('Charge freq (ev/day)',  'charge_freq_per_day',  '{:>10.3f}'),
    ('Avg charge dur (h)',    'avg_charge_duration_h','{:>10.3f}'),
    ('Avg SOC gain (%)',      'avg_soc_gain',         '{:>10.2f}'),
    ('Charge rate (SOC%/h)', 'avg_charge_rate',       '{:>10.2f}'),
    ('Charge regularity',    'charge_regularity',     '{:>10.3f}'),
    ('Idle ratio',           'idle_ratio',            '{:>10.3f}'),
    ('Urban ratio',          'urban_ratio',           '{:>10.3f}'),
    ('Highway ratio',        'highway_ratio',         '{:>10.3f}'),
    ('Switch frequency',     'switch_freq',           '{:>10.3f}'),
    ('Stability index',      'stability_index',       '{:>10.3f}'),
    ('Cumulative hours',     'cumulative_hours',      '{:>10.1f}'),
]

for label, key, fmt in metrics_map:
    row = f"  {label:<32}"
    for arch in archs:
        v = arch_stats[arch].get(key, float('nan'))
        row += '  ' + fmt.format(v)
    report_lines.append(row)
report_lines.append("")

# ── 3. Grid Impact Analysis ───────────────────────────────────────────────
report_lines.append("3. GRID IMPACT ANALYSIS")
report_lines.append("-" * 80)
for arch in archs:
    g = grid_impact[arch]
    report_lines.append(f"  {arch}: {ARCHETYPE_ROLES[arch]}")
    report_lines.append(f"     Peak demand index      : {g['peak_demand_index']:.3f}")
    report_lines.append(f"     Charging rate volatility: {g['charge_rate_volatility']:.3f}")
    report_lines.append(f"     Peak-to-valley ratio   : {g['peak_valley_ratio']:.2f}")
    report_lines.append(f"     Controllability        : {g['controllability']:.3f}")
    report_lines.append(f"     Grid stress score      : {g['grid_stress_score']:.3f}")
    report_lines.append("")

# ── 4. Infrastructure Needs ───────────────────────────────────────────────
report_lines.append("4. CHARGING INFRASTRUCTURE NEEDS")
report_lines.append("-" * 80)
infra_metrics = [
    ('Fast charge demand',   'fast_charge_ratio',   '{:>8.3f}'),
    ('Slow charge demand',   'slow_charge_ratio',   '{:>8.3f}'),
    ('Location coverage',    'coverage_need',       '{:>8.3f}'),
    ('Temporal regularity',  'temporal_regularity', '{:>8.3f}'),
    ('Peak power handling',  'peak_power_handling', '{:>8.3f}'),
]
header = f"  {'Metric':<28}  {'C0':>8}  {'C1':>8}  {'C2':>8}  {'C3':>8}"
report_lines.append(header)
report_lines.append("  " + "-" * (len(header) - 2))
for label, key, fmt in infra_metrics:
    row = f"  {label:<28}"
    for arch in archs:
        v = arch_stats[arch].get(key, float('nan'))
        row += '  ' + fmt.format(v)
    report_lines.append(row)
report_lines.append("")

# ── 5. Conclusions and Recommendations ───────────────────────────────────
report_lines.append("5. CONCLUSIONS AND RECOMMENDATIONS")
report_lines.append("-" * 80)
report_lines.append("""
  C0 — Mixed-Pattern Charger
    • Charging strategy : Provide distributed Level-2 AC chargers at residential
      and workplace locations to support frequent, moderate-duration top-ups.
    • Grid management   : Smooth load profile; leverage V1G smart scheduling
      to shift charging to off-peak windows.
    • Infrastructure    : Dense deployment of slow chargers across suburban areas.

  C1 — Highway Power Charger
    • Charging strategy : High-power DC fast chargers (≥ 120 kW) at motorway
      service areas with short inter-charger spacing (≤ 60 km).
    • Grid management   : Anticipate midday demand spikes on major corridors;
      battery-backed BESS buffers at highway hubs are recommended.
    • Infrastructure    : Prioritise expressway/national-road corridor coverage;
      few but high-capacity chargers per location.

  C2 — Deep-Idle Charger
    • Charging strategy : Overnight slow chargers (Level-1/Level-2 AC, ≤ 11 kW)
      at home or long-term parking; highly predictable load.
    • Grid management   : Excellent candidate for V2G / smart-grid participation;
      stable and controllable — can support base-load balancing.
    • Infrastructure    : Prioritise residential/carpark slow-charger installation;
      focus on coverage rather than fast-charge capability.

  C3 — Urban-Quick Charger
    • Charging strategy : Dense network of rapid chargers (22–50 kW AC or DC)
      at urban hotspots: malls, office blocks, transit hubs.
    • Grid management   : Highest grid stress; demand-response incentives and
      dynamic tariffs needed to flatten intra-day spikes.
    • Infrastructure    : Highest location-coverage requirement; smart booking
      and dynamic pricing can reduce congestion.
""")

report_lines.append("=" * 80)
report_lines.append("OUTPUT FILES")
report_lines.append("-" * 80)
report_lines.append(f"  Results : {CONFIG['save_dir']}")
report_lines.append(f"  Figures : {CONFIG['fig_dir']}")
report_lines.append("")

report_text = '\n'.join(report_lines)
report_path = os.path.join(CONFIG['save_dir'], 'charging_archetype_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_text)
print(f"   ✓ Saved: charging_archetype_report.txt")
print()
print(report_text)

# ─────────────────────────────────────────────────────────────────────────────
# Complete
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("✅  STEP 14 COMPLETE — Four-Archetype Charging Analysis")
print("=" * 80)
print(f"""
Output structure:
  {CONFIG['save_dir']}
    charging_characteristics_c0.csv
    charging_characteristics_c1.csv
    charging_characteristics_c2.csv
    charging_characteristics_c3.csv
    charging_archetype_summary.json
    grid_impact_analysis.json
    charging_archetype_report.txt

  {CONFIG['fig_dir']}
    01_four_archetypes_overview.png
    02_charging_demand_profile.png
    03_grid_impact_comparison.png
    04_charging_infrastructure_requirement.png
""")
