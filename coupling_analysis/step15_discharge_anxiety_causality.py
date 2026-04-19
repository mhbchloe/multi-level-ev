"""
Step 15: Discharge-Vehicle-Anxiety-Charging Causality Analysis
==============================================================
Analyzes how vehicle discharge patterns relate to range anxiety
and subsequent charging behavior.

Produces 15 visualizations (all English labels) saved to:
  ./coupling_analysis/results/figures_step15/

Key fixes vs prior version:
- SHAP: custom bar chart (NOT shap.summary_plot which fails with rendering)
- All English labels (no Chinese characters)
- Robust try/except per visualization - failures skip gracefully
- Graceful missing-data handling with informative messages

Dependencies:
  matplotlib seaborn pandas numpy scipy scikit-learn shap plotly networkx kaleido
"""

import os
import sys
import warnings
import traceback

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

# ── Output directories ──────────────────────────────────────────────────────
RESULTS_DIR = "./coupling_analysis/results/"
FIGURE_DIR = os.path.join(RESULTS_DIR, "figures_step15")
os.makedirs(FIGURE_DIR, exist_ok=True)

# ── Matplotlib style ────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'figure.dpi': 150,
})

COLORS = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6',
          '#1ABC9C', '#E67E22', '#34495E']

print("=" * 70)
print("Step 15: Discharge-Vehicle-Anxiety-Charging Causality Analysis")
print("=" * 70)

# ============================================================
# STEP 1 — Load data
# ============================================================
print("\n[1/6] Loading data...")

def load_csv_safe(path, label):
    """Load a CSV, return None with a clear message if missing."""
    if not os.path.exists(path):
        print(f"   WARNING: {label} not found at {path}")
        return None
    df = pd.read_csv(path)
    print(f"   OK  {label}: {len(df):,} rows, {df.shape[1]} cols")
    return df


# Candidate filenames for trips data
trips_candidates = [
    'inter_charge_trips_v2.csv',
    'inter_charge_trips.csv',
]
df_trips = None
for fn in trips_candidates:
    path = os.path.join(RESULTS_DIR, fn)
    df_trips = load_csv_safe(path, f"trips ({fn})")
    if df_trips is not None:
        break

# Candidate filenames for charging events
charge_candidates = [
    'charging_events_stationary_only.csv',
    'charging_events_stationary_meaningful.csv',
    'charging_events_meaningful.csv',
    'charging_events_raw_extracted.csv',
]
df_charge = None
for fn in charge_candidates:
    path = os.path.join(RESULTS_DIR, fn)
    df_charge = load_csv_safe(path, f"charging ({fn})")
    if df_charge is not None:
        break

# Vehicle clustering
vehicle_candidates = [
    os.path.join('./vehicle_clustering/results/', 'vehicle_clustering_gmm_k4.csv'),
    os.path.join(RESULTS_DIR, 'vehicle_clustering_gmm_k4.csv'),
    os.path.join('./vehicle_clustering/results/', 'vehicle_clustering_optimal.csv'),
]
df_vehicles = None
for path in vehicle_candidates:
    df_vehicles = load_csv_safe(path, f"vehicle clustering ({os.path.basename(path)})")
    if df_vehicles is not None:
        break

# ── Datetime parsing ─────────────────────────────────────────────────────────
for df, time_cols in [
    (df_charge, ['start_time', 'end_time']),
    (df_trips,  ['trip_start', 'trip_end', 'start_time', 'end_time']),
]:
    if df is None:
        continue
    for col in time_cols:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except Exception:
                pass

# ============================================================
# STEP 2 — Build feature matrix
# ============================================================
print("\n[2/6] Building feature matrix...")

if df_trips is not None:
    df = df_trips.copy()

    # Standardise column names
    column_rename_map = {}
    for col in df.columns:
        lc = col.lower()
        if 'soc_drop' in lc and 'trip_total' not in lc:
            column_rename_map[col] = 'soc_drop'
        if lc in ('chg_duration', 'charge_duration', 'charging_duration'):
            column_rename_map[col] = 'charge_duration_min'
    if column_rename_map:
        df.rename(columns=column_rename_map, inplace=True)

    # Infer vehicle_type from vehicle cluster data if not present
    if 'vehicle_type' not in df.columns and df_vehicles is not None:
        vt_col = next(
            (c for c in df_vehicles.columns
             if c.lower() in ('vehicle_type', 'archetype', 'cluster_label', 'cluster')),
            None)
        id_col = next(
            (c for c in df_vehicles.columns if c.lower() == 'vehicle_id'),
            None)
        if vt_col and id_col:
            df = df.merge(
                df_vehicles[[id_col, vt_col]].rename(
                    columns={id_col: 'vehicle_id', vt_col: 'vehicle_type'}),
                on='vehicle_id', how='left')
            print(f"   Merged vehicle_type from clustering data")

    # Fall back: create generic vehicle_type
    if 'vehicle_type' not in df.columns:
        df['vehicle_type'] = 'Unknown'
        print("   NOTE: vehicle_type not found; using 'Unknown'")

    # ── Discharge / anxiety feature proxies ──────────────────────────────────
    # soc_drop: primary discharge proxy
    soc_col = next(
        (c for c in ['trip_total_soc_drop', 'soc_drop', 'total_soc_drop']
         if c in df.columns), None)
    if soc_col:
        df['soc_discharge'] = df[soc_col]
    elif 'soc_start' in df.columns and 'soc_end' in df.columns:
        df['soc_discharge'] = df['soc_start'] - df['soc_end']
    else:
        df['soc_discharge'] = np.nan
        print("   WARNING: Cannot derive soc_discharge")

    # charge_trigger_soc proxy
    trig_cols = [c for c in ['charge_trigger_soc', 'soc_start', 'charge_start_soc'] if c in df.columns]
    df['charge_trigger_soc'] = df[trig_cols[0]] if trig_cols else np.nan

    # aggression proxy
    agg_cols = [c for c in ['ratio_aggressive', 'aggressiveness_index', 'aggressive_ratio'] if c in df.columns]
    df['aggression'] = df[agg_cols[0]] if agg_cols else np.nan

    # conservative proxy
    con_cols = [c for c in ['ratio_conservative', 'conservative_ratio'] if c in df.columns]
    df['conservatism'] = df[con_cols[0]] if con_cols else np.nan

    # trip duration
    dur_cols = [c for c in ['trip_duration_hrs', 'trip_duration', 'duration_hrs'] if c in df.columns]
    df['trip_duration'] = df[dur_cols[0]] if dur_cols else np.nan

    # charging gain
    gain_cols = [c for c in ['charge_gain_soc', 'soc_gain', 'charge_amount_soc'] if c in df.columns]
    df['charge_gain'] = df[gain_cols[0]] if gain_cols else np.nan

    # charge duration
    cdur_cols = [c for c in ['charge_duration_min', 'charge_duration_hrs', 'charging_duration'] if c in df.columns]
    df['charge_duration'] = df[cdur_cols[0]] if cdur_cols else np.nan

    # fast-charge flag proxy (trigger SOC < 30% = emergency / fast charge intent)
    if 'charge_trigger_soc' in df.columns:
        df['fast_charge_rate'] = (df['charge_trigger_soc'] < 30).astype(float)
    else:
        df['fast_charge_rate'] = np.nan

    # ── Composite Anxiety Score ────────────────────────────────────────────────
    # Anxiety = weighted sum of normalised components:
    #   0.40 * discharge_anxiety  (high soc drop → high anxiety)
    #   0.30 * trigger_anxiety    (low trigger SOC → high anxiety)
    #   0.20 * aggression_anxiety (high aggression → high anxiety)
    #   0.10 * speed_anxiety      (optional: high speed → anxiety)

    def safe_norm(series):
        """Min-max normalise; returns 0.5 for constant series."""
        s = series.dropna()
        rng = s.max() - s.min()
        if rng == 0:
            return pd.Series(0.5, index=series.index)
        return (series - s.min()) / rng

    components = {}

    if df['soc_discharge'].notna().sum() > 10:
        components['discharge'] = safe_norm(df['soc_discharge'])   # higher discharge → higher anxiety

    if df['charge_trigger_soc'].notna().sum() > 10:
        components['trigger'] = 1 - safe_norm(df['charge_trigger_soc'])  # lower trigger SOC → higher anxiety

    if df['aggression'].notna().sum() > 10:
        components['aggression'] = safe_norm(df['aggression'])

    speed_cols = [c for c in ['trip_avg_speed', 'avg_speed', 'speed_mean'] if c in df.columns]
    if speed_cols:
        components['speed'] = safe_norm(df[speed_cols[0]])

    if components:
        weights = {'discharge': 0.40, 'trigger': 0.30, 'aggression': 0.20, 'speed': 0.10}
        active = [k for k in weights if k in components]
        total_w = sum(weights[k] for k in active)
        df['anxiety_score'] = sum(
            (weights[k] / total_w) * components[k] for k in active
        )
        print(f"   Anxiety score built from: {active} (total weight={total_w:.2f})")
    else:
        df['anxiety_score'] = np.random.uniform(0, 1, len(df))  # fallback demo
        print("   WARNING: Using random anxiety_score (no input features found)")

    # Anxiety level categorical
    df['anxiety_level'] = pd.cut(
        df['anxiety_score'],
        bins=[0, 0.33, 0.67, 1.0],
        labels=['Low', 'Medium', 'High'],
        include_lowest=True
    )

    print(f"   Feature matrix: {len(df):,} records, {df['vehicle_type'].nunique()} vehicle types")
    print(f"   Anxiety levels: {df['anxiety_level'].value_counts().to_dict()}")

else:
    print("   ERROR: No trips data available. Creating synthetic demo data.")
    n = 2000
    np.random.seed(42)
    df = pd.DataFrame({
        'vehicle_id': np.random.randint(1, 201, n),
        'vehicle_type': np.random.choice(['Efficient-Economic', 'Low-SOC-Risk'], n),
        'soc_discharge': np.random.normal(25, 10, n).clip(0, 100),
        'charge_trigger_soc': np.random.normal(35, 15, n).clip(5, 90),
        'aggression': np.random.uniform(0, 1, n),
        'conservatism': np.random.uniform(0, 1, n),
        'trip_duration': np.random.exponential(0.5, n).clip(0.1, 5),
        'charge_gain': np.random.normal(40, 15, n).clip(5, 90),
        'charge_duration': np.random.normal(45, 20, n).clip(5, 180),
        'fast_charge_rate': np.random.uniform(0, 1, n),
    })
    df['anxiety_score'] = (
        0.4 * (df['soc_discharge'] / 100) +
        0.3 * (1 - df['charge_trigger_soc'] / 100) +
        0.2 * df['aggression'] +
        0.1 * np.random.uniform(0, 1, n)
    )
    df['anxiety_level'] = pd.cut(
        df['anxiety_score'],
        # Upper bound is 1.01 (not 1.0) so that the maximum value 1.0 falls
        # inside the last bin rather than being excluded as NaN.
        bins=[0, 0.33, 0.67, 1.01],
        labels=['Low', 'Medium', 'High']
    )

# Save feature matrix
feat_csv = os.path.join(RESULTS_DIR, 'step15_feature_matrix.csv')
df.to_csv(feat_csv, index=False)
print(f"   Saved: {feat_csv}")

# ============================================================
# STEP 3 — Statistical analysis
# ============================================================
print("\n[3/6] Statistical analysis...")

stat_results = {}

# Pearson correlations between anxiety score and key features
corr_targets = ['soc_discharge', 'charge_trigger_soc', 'aggression',
                'charge_gain', 'charge_duration', 'trip_duration']
corr_results = []
for feat in corr_targets:
    if feat in df.columns and df[feat].notna().sum() > 30:
        r, p = stats.pearsonr(
            df['anxiety_score'].fillna(0),
            df[feat].fillna(df[feat].median())
        )
        corr_results.append({'feature': feat, 'r': round(r, 4), 'p': round(p, 6)})

df_corr = pd.DataFrame(corr_results)
stat_results['correlations'] = df_corr
print(f"   Pearson correlations computed for {len(df_corr)} features")

# ANOVA: anxiety across vehicle types
vtypes = df['vehicle_type'].dropna().unique()
if len(vtypes) >= 2:
    groups = [df[df['vehicle_type'] == v]['anxiety_score'].dropna().values for v in vtypes]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) >= 2:
        f_stat, p_anova = stats.f_oneway(*groups)
        stat_results['anova'] = {'F': f_stat, 'p': p_anova}
        print(f"   ANOVA anxiety ~ vehicle_type: F={f_stat:.3f}, p={p_anova:.4e}")

# ============================================================
# STEP 4 — SHAP feature importance (via XGBoost + TreeExplainer)
# ============================================================
print("\n[4/6] Computing SHAP values...")

shap_feature_importance = {}
shap_available = False

try:
    from sklearn.preprocessing import LabelEncoder
    import xgboost as xgb
    import shap

    feature_cols = [c for c in
                    ['soc_discharge', 'charge_trigger_soc', 'aggression',
                     'conservatism', 'trip_duration', 'charge_gain', 'charge_duration',
                     'fast_charge_rate']
                    if c in df.columns and df[c].notna().sum() > 100]

    if len(feature_cols) >= 2:
        df_model = df[feature_cols + ['anxiety_score']].dropna()
        if len(df_model) > 200:
            X_m = df_model[feature_cols].values
            y_m = df_model['anxiety_score'].values

            model_xgb = xgb.XGBRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                random_state=42, verbosity=0
            )
            model_xgb.fit(X_m, y_m)

            # Use TreeExplainer (not shap.Explainer) for reliability
            explainer = shap.TreeExplainer(model_xgb)
            # Cap SHAP sample count at 2000: computing SHAP values is O(n·d),
            # so large datasets are expensive; 2000 samples is sufficient for
            # stable mean |SHAP| estimates.
            n_shap = min(2000, len(X_m))
            idx_s = np.random.RandomState(42).choice(len(X_m), n_shap, replace=False)
            shap_vals = explainer.shap_values(X_m[idx_s])

            mean_abs_shap = np.abs(shap_vals).mean(axis=0)
            for fname, imp in zip(feature_cols, mean_abs_shap):
                shap_feature_importance[fname] = float(imp)

            shap_available = True
            print(f"   SHAP computed on {n_shap} samples, {len(feature_cols)} features")
        else:
            print(f"   SHAP skipped: too few records ({len(df_model)} < 200)")
    else:
        print(f"   SHAP skipped: insufficient feature columns ({len(feature_cols)})")
except ImportError as e:
    print(f"   SHAP/XGBoost not available: {e}")
except Exception as e:
    print(f"   SHAP computation failed: {e}")
    traceback.print_exc()

# ============================================================
# STEP 5 — Archetype comparison & save
# ============================================================
print("\n[5/6] Archetype comparison...")

archetype_metrics = ['anxiety_score', 'soc_discharge', 'charge_trigger_soc',
                     'aggression', 'charge_gain', 'charge_duration']
archetype_metrics = [c for c in archetype_metrics if c in df.columns]

if archetype_metrics and 'vehicle_type' in df.columns:
    df_arch = df.groupby('vehicle_type')[archetype_metrics].agg(['mean', 'std']).round(3)
    arch_csv = os.path.join(RESULTS_DIR, 'step15_archetype_comparison.csv')
    df_arch.to_csv(arch_csv)
    print(f"   Saved: {arch_csv}")

# ============================================================
# STEP 6 — Save causality report
# ============================================================
print("\n[6/6] Saving causality report...")

report_path = os.path.join(RESULTS_DIR, 'step15_causality_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("Step 15: Discharge-Vehicle-Anxiety-Charging Causality Report\n")
    f.write("=" * 70 + "\n\n")

    f.write("DATASET SUMMARY\n")
    f.write("-" * 40 + "\n")
    f.write(f"  Total records: {len(df):,}\n")
    if 'vehicle_id' in df.columns:
        f.write(f"  Unique vehicles: {df['vehicle_id'].nunique():,}\n")
    f.write(f"  Vehicle types: {list(df['vehicle_type'].unique())}\n\n")

    f.write("ANXIETY SCORE STATISTICS\n")
    f.write("-" * 40 + "\n")
    f.write(df['anxiety_score'].describe().to_string() + "\n\n")

    f.write("PEARSON CORRELATIONS WITH ANXIETY SCORE\n")
    f.write("-" * 40 + "\n")
    if not df_corr.empty:
        f.write(df_corr.to_string(index=False) + "\n\n")

    if 'anova' in stat_results:
        f.write("ANOVA: ANXIETY ~ VEHICLE TYPE\n")
        f.write("-" * 40 + "\n")
        f.write(f"  F-statistic: {stat_results['anova']['F']:.4f}\n")
        f.write(f"  P-value:     {stat_results['anova']['p']:.4e}\n\n")

    if shap_feature_importance:
        f.write("SHAP FEATURE IMPORTANCE (mean |SHAP|)\n")
        f.write("-" * 40 + "\n")
        for fname, imp in sorted(shap_feature_importance.items(), key=lambda x: -x[1]):
            f.write(f"  {fname:<28}: {imp:.6f}\n")
        f.write("\n")

    f.write("OUTPUT FILES\n")
    f.write("-" * 40 + "\n")
    f.write(f"  Feature matrix: {feat_csv}\n")
    f.write(f"  Figures: {FIGURE_DIR}\n")

print(f"   Saved: {report_path}")

# ============================================================
# VISUALIZATIONS (01 – 15)
# ============================================================
print("\n[Viz] Generating 15 figures...")

fig_count = {'ok': 0, 'fail': 0}

def savefig(fig, name, fmt='png', close=True):
    """Save figure with robust error handling."""
    path = os.path.join(FIGURE_DIR, name)
    try:
        fig.savefig(path, dpi=150, bbox_inches='tight')
        fig_count['ok'] += 1
        print(f"   [OK]  {name}")
    except Exception as e:
        print(f"   [FAIL] {name}: {e}")
        fig_count['fail'] += 1
    if close:
        plt.close(fig)


# ── 01: Boxplot – Anxiety distribution by vehicle type ──────────────────────
print("\n  Fig 01: Anxiety distribution by vehicle type")
try:
    vtypes_sorted = sorted(df['vehicle_type'].dropna().unique())
    n_types = len(vtypes_sorted)
    pal = COLORS[:n_types]

    fig, ax = plt.subplots(figsize=(max(8, n_types * 2), 6))
    data_by_type = [df[df['vehicle_type'] == v]['anxiety_score'].dropna().values
                    for v in vtypes_sorted]
    bp = ax.boxplot(data_by_type, labels=vtypes_sorted, patch_artist=True,
                    widths=0.6, notch=False)
    for patch, color in zip(bp['boxes'], pal):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_title('Anxiety Score Distribution by Vehicle Type', fontweight='bold')
    ax.set_xlabel('Vehicle Type')
    ax.set_ylabel('Composite Anxiety Score (0–1)')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    savefig(fig, 'fig01_anxiety_by_vehicle_type.png')
except Exception as e:
    print(f"   [SKIP] Fig 01: {e}")
    fig_count['fail'] += 1

# ── 02: Scatter – Discharge SOC vs Charging Duration ────────────────────────
print("\n  Fig 02: Discharge SOC vs Charging Duration")
try:
    x_col = 'soc_discharge' if 'soc_discharge' in df.columns else None
    y_col = 'charge_duration' if 'charge_duration' in df.columns else None

    if x_col and y_col:
        sub = df[[x_col, y_col, 'vehicle_type']].dropna()
        if len(sub) > 30:
            fig, ax = plt.subplots(figsize=(8, 6))
            vtypes_s = sorted(sub['vehicle_type'].unique())
            for i, vt in enumerate(vtypes_s):
                m = sub['vehicle_type'] == vt
                ax.scatter(sub.loc[m, x_col], sub.loc[m, y_col],
                           alpha=0.35, s=15, color=COLORS[i % len(COLORS)], label=vt)
            # Overall trend line
            x_v = sub[x_col].values
            y_v = sub[y_col].values
            slope, intercept, r, p, _ = stats.linregress(x_v, y_v)
            xl = np.linspace(x_v.min(), x_v.max(), 100)
            ax.plot(xl, slope * xl + intercept, 'k--', lw=2,
                    label=f'Trend  r={r:.2f}, p={p:.3f}')
            ax.set_title('Discharge SOC vs Charging Duration', fontweight='bold')
            ax.set_xlabel('SOC Discharged (%)')
            ax.set_ylabel('Charging Duration (min)')
            ax.legend(fontsize=9, loc='upper left')
            ax.grid(alpha=0.3)
            plt.tight_layout()
            savefig(fig, 'fig02_discharge_vs_charging_duration.png')
        else:
            print("   [SKIP] Fig 02: insufficient data")
            fig_count['fail'] += 1
    else:
        print(f"   [SKIP] Fig 02: missing columns ({x_col}, {y_col})")
        fig_count['fail'] += 1
except Exception as e:
    print(f"   [SKIP] Fig 02: {e}")
    fig_count['fail'] += 1

# ── 03: Scatter – Anxiety score vs Fast charge rate ─────────────────────────
print("\n  Fig 03: Anxiety score vs Fast charge rate")
try:
    sub = df[['anxiety_score', 'fast_charge_rate', 'vehicle_type']].dropna()
    if len(sub) > 30:
        fig, ax = plt.subplots(figsize=(8, 6))
        vtypes_s = sorted(sub['vehicle_type'].unique())
        for i, vt in enumerate(vtypes_s):
            m = sub['vehicle_type'] == vt
            ax.scatter(sub.loc[m, 'anxiety_score'], sub.loc[m, 'fast_charge_rate'],
                       alpha=0.4, s=18, color=COLORS[i % len(COLORS)], label=vt)
        x_v = sub['anxiety_score'].values
        y_v = sub['fast_charge_rate'].values
        slope, intercept, r, p, _ = stats.linregress(x_v, y_v)
        xl = np.linspace(x_v.min(), x_v.max(), 100)
        ax.plot(xl, slope * xl + intercept, 'k--', lw=2,
                label=f'Trend  r={r:.2f}, p={p:.3f}')
        ax.set_title('Anxiety Score vs Fast Charge Rate', fontweight='bold')
        ax.set_xlabel('Composite Anxiety Score (0–1)')
        ax.set_ylabel('Fast Charge Propensity')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        savefig(fig, 'fig03_anxiety_vs_fast_charge.png')
    else:
        print("   [SKIP] Fig 03: insufficient data")
        fig_count['fail'] += 1
except Exception as e:
    print(f"   [SKIP] Fig 03: {e}")
    fig_count['fail'] += 1

# ── 04: Heatmap – Correlation matrix ────────────────────────────────────────
print("\n  Fig 04: Correlation matrix heatmap")
try:
    num_cols = [c for c in
                ['anxiety_score', 'soc_discharge', 'charge_trigger_soc',
                 'aggression', 'conservatism', 'trip_duration',
                 'charge_gain', 'charge_duration', 'fast_charge_rate']
                if c in df.columns and df[c].notna().sum() > 30]
    if len(num_cols) >= 3:
        corr_mat = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(max(8, len(num_cols)), max(7, len(num_cols) - 1)))
        sns.heatmap(corr_mat, annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, vmin=-1, vmax=1, linewidths=0.5, ax=ax,
                    annot_kws={'size': 9})
        ax.set_title('Feature Correlation Matrix', fontweight='bold')
        plt.tight_layout()
        savefig(fig, 'fig04_correlation_heatmap.png')
    else:
        print(f"   [SKIP] Fig 04: only {len(num_cols)} numeric columns")
        fig_count['fail'] += 1
except Exception as e:
    print(f"   [SKIP] Fig 04: {e}")
    fig_count['fail'] += 1

# ── 05: Boxplots – Anxiety components by vehicle type (grid) ────────────────
print("\n  Fig 05: Anxiety components by vehicle type")
try:
    comp_cols = [c for c in
                 ['soc_discharge', 'charge_trigger_soc', 'aggression',
                  'conservatism', 'charge_gain']
                 if c in df.columns and df[c].notna().sum() > 30]
    if comp_cols and 'vehicle_type' in df.columns:
        nc = len(comp_cols)
        ncols_g = min(3, nc)
        nrows_g = (nc + ncols_g - 1) // ncols_g
        fig, axes = plt.subplots(nrows_g, ncols_g,
                                  figsize=(5 * ncols_g, 4 * nrows_g))
        axes_flat = axes.flatten() if nc > 1 else [axes]
        vtypes_s = sorted(df['vehicle_type'].dropna().unique())
        for i, col in enumerate(comp_cols):
            ax = axes_flat[i]
            data = [df[df['vehicle_type'] == v][col].dropna().values for v in vtypes_s]
            bp = ax.boxplot(data, labels=vtypes_s, patch_artist=True, widths=0.5)
            for patch, color in zip(bp['boxes'], COLORS):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax.set_title(col.replace('_', ' ').title(), fontweight='bold')
            ax.set_ylabel('Value')
            ax.grid(axis='y', alpha=0.3)
            ax.tick_params(axis='x', rotation=20)
        # Hide unused subplots
        for j in range(nc, len(axes_flat)):
            axes_flat[j].set_visible(False)
        fig.suptitle('Anxiety Components by Vehicle Type', fontweight='bold', fontsize=14)
        plt.tight_layout()
        savefig(fig, 'fig05_anxiety_components_by_type.png')
    else:
        print("   [SKIP] Fig 05: insufficient component data")
        fig_count['fail'] += 1
except Exception as e:
    print(f"   [SKIP] Fig 05: {e}")
    fig_count['fail'] += 1

# ── 06: Bar – Anxiety level vs charging behavior ────────────────────────────
print("\n  Fig 06: Anxiety level vs charging behavior")
try:
    behav_cols = [c for c in ['charge_gain', 'charge_duration', 'charge_trigger_soc']
                  if c in df.columns]
    if behav_cols and 'anxiety_level' in df.columns:
        levels = ['Low', 'Medium', 'High']
        nc = len(behav_cols)
        fig, axes = plt.subplots(1, nc, figsize=(5 * nc, 5))
        axes_l = axes if nc > 1 else [axes]
        for i, col in enumerate(behav_cols):
            ax = axes_l[i]
            means = [df[df['anxiety_level'] == lvl][col].mean() for lvl in levels]
            sds = [df[df['anxiety_level'] == lvl][col].std() for lvl in levels]
            bars = ax.bar(levels, means, yerr=sds, capsize=5,
                          color=['#2ECC71', '#F39C12', '#E74C3C'], alpha=0.85,
                          edgecolor='black')
            ax.set_title(col.replace('_', ' ').title(), fontweight='bold')
            ax.set_xlabel('Anxiety Level')
            ax.set_ylabel('Mean Value')
            ax.grid(axis='y', alpha=0.3)
        fig.suptitle('Charging Behavior by Anxiety Level', fontweight='bold', fontsize=14)
        plt.tight_layout()
        savefig(fig, 'fig06_anxiety_level_vs_charging.png')
    else:
        print("   [SKIP] Fig 06: missing required columns")
        fig_count['fail'] += 1
except Exception as e:
    print(f"   [SKIP] Fig 06: {e}")
    fig_count['fail'] += 1

# ── 07: Boxplot + scatter – Charge start SOC analysis ───────────────────────
print("\n  Fig 07: Charge start SOC analysis")
try:
    if 'charge_trigger_soc' in df.columns and 'vehicle_type' in df.columns:
        vtypes_s = sorted(df['vehicle_type'].dropna().unique())
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Boxplot
        ax = axes[0]
        data = [df[df['vehicle_type'] == v]['charge_trigger_soc'].dropna().values
                for v in vtypes_s]
        bp = ax.boxplot(data, labels=vtypes_s, patch_artist=True, widths=0.6)
        for patch, color in zip(bp['boxes'], COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title('Charge Start SOC by Vehicle Type', fontweight='bold')
        ax.set_xlabel('Vehicle Type')
        ax.set_ylabel('Charge Trigger SOC (%)')
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=20)

        # Scatter vs anxiety
        ax = axes[1]
        for i, vt in enumerate(vtypes_s):
            m = df['vehicle_type'] == vt
            ax.scatter(df.loc[m, 'anxiety_score'],
                       df.loc[m, 'charge_trigger_soc'],
                       alpha=0.3, s=12, color=COLORS[i % len(COLORS)], label=vt)
        ax.set_title('Charge Start SOC vs Anxiety Score', fontweight='bold')
        ax.set_xlabel('Composite Anxiety Score (0–1)')
        ax.set_ylabel('Charge Trigger SOC (%)')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        savefig(fig, 'fig07_charge_start_soc_analysis.png')
    else:
        print("   [SKIP] Fig 07: missing charge_trigger_soc or vehicle_type")
        fig_count['fail'] += 1
except Exception as e:
    print(f"   [SKIP] Fig 07: {e}")
    fig_count['fail'] += 1

# ── 08: SHAP Feature Importance (custom bar chart) ───────────────────────────
# IMPORTANT: We deliberately avoid shap.summary_plot() which causes rendering
# failures. Instead we use a plain matplotlib bar chart of mean |SHAP| values.
print("\n  Fig 08: SHAP Feature Importance (custom bar chart)")
try:
    if shap_available and shap_feature_importance:
        sorted_feats = sorted(shap_feature_importance.items(), key=lambda x: x[1])
        labels = [k.replace('_', ' ').title() for k, _ in sorted_feats]
        values = [v for _, v in sorted_feats]

        fig, ax = plt.subplots(figsize=(9, max(4, len(labels) * 0.5)))
        bars = ax.barh(labels, values, color='#3498DB', edgecolor='black', alpha=0.8)
        ax.set_xlabel('Mean |SHAP Value| (Feature Importance)', fontweight='bold')
        ax.set_title('SHAP Feature Importance\n(XGBoost → Anxiety Score Prediction)',
                     fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        for bar, val in zip(bars, values):
            ax.text(val + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{val:.4f}', va='center', fontsize=9)
        plt.tight_layout()
        savefig(fig, 'fig08_shap_feature_importance.png')
    else:
        # Fallback: use correlation-based importance
        if not df_corr.empty:
            df_importance = df_corr.sort_values('r')
            labels = [s.replace('_', ' ').title() for s in df_importance['feature'].tolist()]
            values = df_importance['r'].abs().tolist()

            fig, ax = plt.subplots(figsize=(9, max(4, len(labels) * 0.5)))
            colors_bar = ['#E74C3C' if r > 0 else '#3498DB'
                          for r in df_importance['r'].tolist()]
            ax.barh(labels, df_importance['r'].tolist(), color=colors_bar,
                    edgecolor='black', alpha=0.8)
            ax.axvline(0, color='black', lw=0.8)
            ax.set_xlabel('Pearson Correlation with Anxiety Score', fontweight='bold')
            ax.set_title('Feature Importance (Correlation-based)\n'
                         '(SHAP not available)', fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            savefig(fig, 'fig08_shap_feature_importance.png')
        else:
            print("   [SKIP] Fig 08: SHAP not available and no correlation data")
            fig_count['fail'] += 1
except Exception as e:
    print(f"   [SKIP] Fig 08: {e}")
    fig_count['fail'] += 1

# ── 09: Violin + KDE – Anxiety distribution ─────────────────────────────────
print("\n  Fig 09: Violin + KDE – Anxiety distribution")
try:
    vtypes_s = sorted(df['vehicle_type'].dropna().unique())
    if vtypes_s:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Violin by vehicle type
        ax = axes[0]
        plot_df = df[['anxiety_score', 'vehicle_type']].dropna()
        if not plot_df.empty:
            sns.violinplot(data=plot_df, x='vehicle_type', y='anxiety_score',
                           palette=COLORS[:len(vtypes_s)], inner='quartile',
                           order=vtypes_s, ax=ax)
        ax.set_title('Anxiety Score Distribution by Vehicle Type\n(Violin Plot)',
                     fontweight='bold')
        ax.set_xlabel('Vehicle Type')
        ax.set_ylabel('Anxiety Score (0–1)')
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=20)

        # KDE overlay
        ax = axes[1]
        for i, vt in enumerate(vtypes_s):
            sub = df[df['vehicle_type'] == vt]['anxiety_score'].dropna()
            if len(sub) > 10:
                sub.plot.kde(ax=ax, label=vt, color=COLORS[i % len(COLORS)], lw=2)
        ax.set_title('Anxiety Score KDE by Vehicle Type', fontweight='bold')
        ax.set_xlabel('Anxiety Score (0–1)')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        savefig(fig, 'fig09_violin_kde_anxiety.png')
    else:
        print("   [SKIP] Fig 09: no vehicle type data")
        fig_count['fail'] += 1
except Exception as e:
    print(f"   [SKIP] Fig 09: {e}")
    fig_count['fail'] += 1

# ── 10: 3D interactive scatter (Plotly) ─────────────────────────────────────
print("\n  Fig 10: 3D interactive scatter (Plotly)")
try:
    import plotly.express as px
    import plotly.io as pio

    x_c = 'soc_discharge' if 'soc_discharge' in df.columns else 'charge_trigger_soc'
    y_c = 'charge_trigger_soc' if 'charge_trigger_soc' in df.columns else 'charge_gain'
    z_c = 'anxiety_score'

    if all(c in df.columns for c in [x_c, y_c, z_c]):
        sub_3d = df[[x_c, y_c, z_c, 'vehicle_type']].dropna().sample(
            min(3000, len(df)), random_state=42)

        fig_3d = px.scatter_3d(
            sub_3d, x=x_c, y=y_c, z=z_c,
            color='vehicle_type',
            opacity=0.6, size_max=4,
            title='3D Scatter: Discharge – Trigger SOC – Anxiety Score',
            labels={
                x_c: x_c.replace('_', ' ').title(),
                y_c: y_c.replace('_', ' ').title(),
                z_c: 'Anxiety Score',
                'vehicle_type': 'Vehicle Type',
            }
        )
        html_path = os.path.join(FIGURE_DIR, 'fig10_3d_scatter.html')
        fig_3d.write_html(html_path)
        fig_count['ok'] += 1
        print(f"   [OK]  fig10_3d_scatter.html")

        # Also save static PNG if kaleido available
        try:
            png_path = os.path.join(FIGURE_DIR, 'fig10_3d_scatter.png')
            fig_3d.write_image(png_path, width=900, height=700)
            print(f"   [OK]  fig10_3d_scatter.png")
        except Exception:
            print("   [INFO] Static PNG for Fig 10 not saved (kaleido not available)")
    else:
        print(f"   [SKIP] Fig 10: missing columns ({x_c}, {y_c}, {z_c})")
        fig_count['fail'] += 1
except ImportError:
    print("   [SKIP] Fig 10: plotly not installed")
    fig_count['fail'] += 1
except Exception as e:
    print(f"   [SKIP] Fig 10: {e}")
    fig_count['fail'] += 1

# ── 11: Sankey diagram – Vehicle type → Anxiety level ───────────────────────
print("\n  Fig 11: Sankey diagram – Vehicle type → Anxiety level")
try:
    import plotly.graph_objects as go

    if 'anxiety_level' in df.columns and 'vehicle_type' in df.columns:
        sankey_df = df[['vehicle_type', 'anxiety_level']].dropna()
        vtypes_s = sorted(sankey_df['vehicle_type'].unique())
        alevels = ['Low', 'Medium', 'High']

        # Build node list: vehicle types first, then anxiety levels
        nodes = vtypes_s + alevels
        node_idx = {n: i for i, n in enumerate(nodes)}

        sources, targets, values_s = [], [], []
        for vt in vtypes_s:
            for al in alevels:
                cnt = ((sankey_df['vehicle_type'] == vt) &
                       (sankey_df['anxiety_level'] == al)).sum()
                if cnt > 0:
                    sources.append(node_idx[vt])
                    targets.append(node_idx[al])
                    values_s.append(int(cnt))

        if sources:
            node_colors = (
                [COLORS[i % len(COLORS)] for i in range(len(vtypes_s))] +
                ['#2ECC71', '#F39C12', '#E74C3C']
            )
            fig_sk = go.Figure(data=[go.Sankey(
                node=dict(pad=15, thickness=20, label=nodes, color=node_colors),
                link=dict(source=sources, target=targets, value=values_s)
            )])
            fig_sk.update_layout(
                title_text='Sankey: Vehicle Type → Anxiety Level',
                font_size=12
            )
            html_path = os.path.join(FIGURE_DIR, 'fig11_sankey_vehicle_anxiety.html')
            fig_sk.write_html(html_path)
            fig_count['ok'] += 1
            print(f"   [OK]  fig11_sankey_vehicle_anxiety.html")

            try:
                png_path = os.path.join(FIGURE_DIR, 'fig11_sankey_vehicle_anxiety.png')
                fig_sk.write_image(png_path, width=900, height=600)
                print(f"   [OK]  fig11_sankey_vehicle_anxiety.png")
            except Exception:
                print("   [INFO] Static PNG for Fig 11 not saved (kaleido not available)")
        else:
            print("   [SKIP] Fig 11: no valid flows")
            fig_count['fail'] += 1
    else:
        print("   [SKIP] Fig 11: missing vehicle_type or anxiety_level")
        fig_count['fail'] += 1
except ImportError:
    print("   [SKIP] Fig 11: plotly not installed")
    fig_count['fail'] += 1
except Exception as e:
    print(f"   [SKIP] Fig 11: {e}")
    fig_count['fail'] += 1

# ── 12: Waterfall chart – Anxiety score components ───────────────────────────
print("\n  Fig 12: Waterfall chart – Anxiety components")
try:
    # Compute per-vehicle-type mean contribution of each component
    comp_defs = {
        'Discharge\n(40%)': ('soc_discharge', 0.40, True),
        'Trigger SOC\n(30%)': ('charge_trigger_soc', 0.30, False),   # inverted
        'Aggression\n(20%)': ('aggression', 0.20, True),
        'Speed\n(10%)': (None, 0.10, True),  # may be absent
    }
    speed_col = next((c for c in ['trip_avg_speed', 'avg_speed', 'speed_mean']
                      if c in df.columns), None)
    if speed_col:
        comp_defs['Speed\n(10%)'] = (speed_col, 0.10, True)

    # Use dataset-wide means for demonstration
    contribution_labels = []
    contribution_vals = []
    total_val = 0.0

    def normalise_series_mean(col, invert):
        s = df[col].dropna()
        if s.empty or s.std() == 0:
            return 0.5
        n = (s - s.min()) / (s.max() - s.min())
        return (1 - n.mean()) if invert else n.mean()

    for label, (col, weight, invert) in comp_defs.items():
        if col and col in df.columns:
            contrib = weight * normalise_series_mean(col, invert)
            contribution_labels.append(label)
            contribution_vals.append(contrib)
            total_val += contrib

    if contribution_labels:
        # Running total for waterfall
        running = [0.0]
        for v in contribution_vals:
            running.append(running[-1] + v)
        bottoms = running[:-1]

        fig, ax = plt.subplots(figsize=(10, 6))
        bar_colors = COLORS[:len(contribution_labels)]
        bars = ax.bar(contribution_labels, contribution_vals, bottom=bottoms,
                      color=bar_colors, edgecolor='black', alpha=0.85)
        # Total bar
        ax.bar(['Total\nAnxiety'], [total_val], color='#2C3E50',
               edgecolor='black', alpha=0.9)

        for bar, val, bot in zip(bars, contribution_vals, bottoms):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bot + val / 2,
                    f'{val:.3f}', ha='center', va='center',
                    fontsize=10, fontweight='bold', color='white')

        ax.set_ylabel('Anxiety Score Contribution', fontweight='bold')
        ax.set_title('Waterfall Chart: Anxiety Score Component Breakdown',
                     fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        savefig(fig, 'fig12_waterfall_anxiety_components.png')
    else:
        print("   [SKIP] Fig 12: no valid components")
        fig_count['fail'] += 1
except Exception as e:
    print(f"   [SKIP] Fig 12: {e}")
    fig_count['fail'] += 1

# ── 13: Network graph – Feature correlations ────────────────────────────────
print("\n  Fig 13: Network graph – Feature correlations")
try:
    import networkx as nx

    num_cols_net = [c for c in
                    ['anxiety_score', 'soc_discharge', 'charge_trigger_soc',
                     'aggression', 'charge_gain', 'charge_duration']
                    if c in df.columns and df[c].notna().sum() > 30]

    if len(num_cols_net) >= 3:
        corr_net = df[num_cols_net].corr()
        G = nx.Graph()
        for col in num_cols_net:
            G.add_node(col)

        threshold = 0.15
        for i, c1 in enumerate(num_cols_net):
            for j, c2 in enumerate(num_cols_net):
                if j <= i:
                    continue
                r = corr_net.loc[c1, c2]
                if abs(r) >= threshold:
                    G.add_edge(c1, c2, weight=abs(r), sign=np.sign(r))

        if G.number_of_edges() == 0:
            # Lower threshold if no edges
            threshold = 0.05
            for i, c1 in enumerate(num_cols_net):
                for j, c2 in enumerate(num_cols_net):
                    if j <= i:
                        continue
                    r = corr_net.loc[c1, c2]
                    if abs(r) >= threshold:
                        G.add_edge(c1, c2, weight=abs(r), sign=np.sign(r))

        fig, ax = plt.subplots(figsize=(10, 8))
        pos = nx.spring_layout(G, seed=42, k=2.5)
        edge_weights = [G[u][v]['weight'] * 4 for u, v in G.edges()]
        edge_colors = ['#E74C3C' if G[u][v]['sign'] > 0 else '#3498DB'
                       for u, v in G.edges()]

        nx.draw_networkx_nodes(G, pos, node_size=1200,
                               node_color='#ECF0F1', edgecolors='#2C3E50',
                               linewidths=2, ax=ax)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_weight='bold',
                                labels={n: n.replace('_', '\n') for n in G.nodes()})
        nx.draw_networkx_edges(G, pos, width=edge_weights, edge_color=edge_colors,
                               alpha=0.7, ax=ax)
        # Edge weight labels
        edge_labels = {(u, v): f'{G[u][v]["weight"]:.2f}' for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                     font_size=8, ax=ax)

        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='#E74C3C', lw=3, label='Positive correlation'),
            Line2D([0], [0], color='#3498DB', lw=3, label='Negative correlation'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        ax.set_title(f'Feature Correlation Network (|r| ≥ {threshold})', fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        savefig(fig, 'fig13_correlation_network.png')
    else:
        print(f"   [SKIP] Fig 13: only {len(num_cols_net)} numeric columns")
        fig_count['fail'] += 1
except ImportError:
    # Fallback: simple correlation bar chart
    print("   [INFO] networkx not available, using correlation bar chart fallback")
    try:
        if not df_corr.empty:
            fig, ax = plt.subplots(figsize=(9, 5))
            df_s = df_corr.sort_values('r')
            colors_bar = ['#E74C3C' if r > 0 else '#3498DB' for r in df_s['r']]
            ax.barh([f.replace('_', ' ').title() for f in df_s['feature']],
                    df_s['r'], color=colors_bar, edgecolor='black', alpha=0.8)
            ax.axvline(0, color='black', lw=0.8)
            ax.set_xlabel('Pearson r with Anxiety Score', fontweight='bold')
            ax.set_title('Feature Correlations with Anxiety Score\n'
                         '(Network graph fallback)', fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            savefig(fig, 'fig13_correlation_network.png')
        else:
            print("   [SKIP] Fig 13: fallback also unavailable")
            fig_count['fail'] += 1
    except Exception as e2:
        print(f"   [SKIP] Fig 13 (fallback): {e2}")
        fig_count['fail'] += 1
except Exception as e:
    print(f"   [SKIP] Fig 13: {e}")
    fig_count['fail'] += 1

# ── 14: Detailed boxplots – Anxiety components detail ───────────────────────
print("\n  Fig 14: Detailed anxiety components boxplots")
try:
    detail_cols = [c for c in
                   ['soc_discharge', 'charge_trigger_soc', 'aggression',
                    'charge_gain', 'charge_duration', 'trip_duration']
                   if c in df.columns and df[c].notna().sum() > 30]
    if detail_cols and 'anxiety_level' in df.columns:
        levels = ['Low', 'Medium', 'High']
        nc = len(detail_cols)
        ncols_g = min(3, nc)
        nrows_g = (nc + ncols_g - 1) // ncols_g

        fig, axes = plt.subplots(nrows_g, ncols_g,
                                  figsize=(5 * ncols_g, 4 * nrows_g))
        axes_flat = axes.flatten() if nc > 1 else [axes]

        for i, col in enumerate(detail_cols):
            ax = axes_flat[i]
            data = [df[df['anxiety_level'] == lvl][col].dropna().values
                    for lvl in levels]
            bp = ax.boxplot(data, labels=levels, patch_artist=True, widths=0.5,
                            notch=False)
            level_colors = ['#2ECC71', '#F39C12', '#E74C3C']
            for patch, color in zip(bp['boxes'], level_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax.set_title(col.replace('_', ' ').title(), fontweight='bold')
            ax.set_xlabel('Anxiety Level')
            ax.set_ylabel('Value')
            ax.grid(axis='y', alpha=0.3)

        for j in range(nc, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle('Anxiety Components by Anxiety Level', fontweight='bold', fontsize=14)
        plt.tight_layout()
        savefig(fig, 'fig14_anxiety_components_detail.png')
    else:
        print("   [SKIP] Fig 14: insufficient data")
        fig_count['fail'] += 1
except Exception as e:
    print(f"   [SKIP] Fig 14: {e}")
    fig_count['fail'] += 1

# ── 15: Parallel coordinates – Multi-dimensional analysis ───────────────────
print("\n  Fig 15: Parallel coordinates – Multi-dimensional analysis")
try:
    import plotly.express as px

    pc_cols = [c for c in
               ['anxiety_score', 'soc_discharge', 'charge_trigger_soc',
                'aggression', 'charge_gain', 'charge_duration']
               if c in df.columns and df[c].notna().sum() > 30]

    if len(pc_cols) >= 3 and 'vehicle_type' in df.columns:
        sub_pc = df[pc_cols + ['vehicle_type']].dropna().sample(
            min(3000, len(df)), random_state=42)

        # Encode vehicle_type as integer for color
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        sub_pc = sub_pc.copy()
        sub_pc['vtype_code'] = le.fit_transform(sub_pc['vehicle_type'])

        fig_pc = px.parallel_coordinates(
            sub_pc,
            color='vtype_code',
            dimensions=pc_cols,
            color_continuous_scale=px.colors.qualitative.Safe[:len(le.classes_)],
            title='Parallel Coordinates: Multi-dimensional Anxiety Analysis',
            labels={c: c.replace('_', ' ').title() for c in pc_cols},
        )
        html_path = os.path.join(FIGURE_DIR, 'fig15_parallel_coordinates.html')
        fig_pc.write_html(html_path)
        fig_count['ok'] += 1
        print(f"   [OK]  fig15_parallel_coordinates.html")

        try:
            png_path = os.path.join(FIGURE_DIR, 'fig15_parallel_coordinates.png')
            fig_pc.write_image(png_path, width=1000, height=600)
            print(f"   [OK]  fig15_parallel_coordinates.png")
        except Exception:
            print("   [INFO] Static PNG for Fig 15 not saved (kaleido not available)")

    elif len(pc_cols) >= 3:
        # matplotlib fallback
        from pandas.plotting import parallel_coordinates

        sub_pc = df[pc_cols + ['anxiety_level']].dropna().sample(
            min(500, len(df)), random_state=42)
        sub_pc = sub_pc.copy()
        sub_pc['anxiety_level'] = sub_pc['anxiety_level'].astype(str)

        fig, ax = plt.subplots(figsize=(12, 6))
        parallel_coordinates(sub_pc, 'anxiety_level', color=['#2ECC71', '#F39C12', '#E74C3C'],
                              ax=ax, alpha=0.3)
        ax.set_title('Parallel Coordinates: Multi-dimensional Anxiety Analysis',
                     fontweight='bold')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        savefig(fig, 'fig15_parallel_coordinates.png')
    else:
        print(f"   [SKIP] Fig 15: only {len(pc_cols)} parallel coord columns")
        fig_count['fail'] += 1
except ImportError:
    # matplotlib fallback
    try:
        from pandas.plotting import parallel_coordinates
        pc_cols_2 = [c for c in
                     ['anxiety_score', 'soc_discharge', 'charge_trigger_soc',
                      'aggression', 'charge_gain']
                     if c in df.columns and df[c].notna().sum() > 30]
        if len(pc_cols_2) >= 3 and 'anxiety_level' in df.columns:
            sub_pc = df[pc_cols_2 + ['anxiety_level']].dropna().sample(
                min(500, len(df)), random_state=42)
            sub_pc = sub_pc.copy()
            sub_pc['anxiety_level'] = sub_pc['anxiety_level'].astype(str)
            fig, ax = plt.subplots(figsize=(12, 6))
            parallel_coordinates(sub_pc, 'anxiety_level',
                                  color=['#2ECC71', '#F39C12', '#E74C3C'],
                                  ax=ax, alpha=0.3)
            ax.set_title('Parallel Coordinates: Multi-dimensional Anxiety Analysis',
                         fontweight='bold')
            ax.grid(alpha=0.3)
            plt.tight_layout()
            savefig(fig, 'fig15_parallel_coordinates.png')
        else:
            print("   [SKIP] Fig 15: fallback also insufficient data")
            fig_count['fail'] += 1
    except Exception as e2:
        print(f"   [SKIP] Fig 15 (fallback): {e2}")
        fig_count['fail'] += 1
except Exception as e:
    print(f"   [SKIP] Fig 15: {e}")
    fig_count['fail'] += 1

# ============================================================
# SUMMARY
# ============================================================
total_figs = fig_count['ok'] + fig_count['fail']
print("\n" + "=" * 70)
print(f"Step 15 Complete: {fig_count['ok']}/{total_figs} figures generated successfully")
print(f"Figures saved to: {FIGURE_DIR}")
print(f"Report saved to: {report_path}")
print("=" * 70)
