"""
Step 15: Complete Discharge-Vehicle-Anxiety-Charging Causality Analysis

分析放电片段特性、车辆特性、焦虑程度对充电行为（充电时长、快慢充选择）的影响。

输入:
  - coupling_analysis/results/inter_charge_trips_v2.csv   (放电行程 + 充电触发信息)
  - coupling_analysis/results/charging_events_stationary_only.csv  (充电事件明细)

输出:
  - coupling_analysis/results/step15_vehicle_anxiety_features.csv
  - coupling_analysis/results/step15_archetype_comparison.csv
  - coupling_analysis/results/step15_causality_report.txt
  - coupling_analysis/results/figures_step15/  (若干可视化图片)
"""

import os
import warnings
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['font.size'] = 11

print("=" * 70)
print("🔋 Step 15: Discharge–Vehicle–Anxiety–Charging Causality Analysis")
print("=" * 70)

# ============================================================
# 0. 路径配置
# ============================================================
INPUT_DIR  = "./coupling_analysis/results/"
FIGURE_DIR = os.path.join(INPUT_DIR, "figures_step15")
os.makedirs(FIGURE_DIR, exist_ok=True)

TRIPS_PATH    = os.path.join(INPUT_DIR, "inter_charge_trips_v2.csv")
CHARGING_PATH = os.path.join(INPUT_DIR, "charging_events_stationary_only.csv")

# ============================================================
# 1. 加载数据
# ============================================================
print(f"\n{'='*70}")
print("【STEP 1】Loading Data")
print(f"{'='*70}")

trips_df    = pd.read_csv(TRIPS_PATH)
charging_df = pd.read_csv(CHARGING_PATH)

print(f"   trips_df    : {len(trips_df):,} rows, {trips_df['vehicle_id'].nunique():,} vehicles")
print(f"   charging_df : {len(charging_df):,} rows, {charging_df['vehicle_id'].nunique():,} vehicles")

# 确保时间列为 datetime（可选，用于日期运算）
for col in ['start_time', 'end_time']:
    if col in trips_df.columns:
        trips_df[col] = pd.to_datetime(trips_df[col], errors='coerce')
    if col in charging_df.columns:
        charging_df[col] = pd.to_datetime(charging_df[col], errors='coerce')

# 快慢充标签
if 'charge_type' in charging_df.columns:
    charging_df['is_fast_charge'] = (charging_df['charge_type'] == 'fast').astype(int)
    print("   ✓ Fast/slow charge label: using existing 'charge_type' column")
else:
    # 备选：基于充电速率自动定义
    charging_df['charge_rate'] = charging_df['soc_gain'] / charging_df['duration_minutes'].replace(0, np.nan)
    charging_df['is_fast_charge'] = (charging_df['charge_rate'] > 1.0).astype(int)
    print("   ⚠ 'charge_type' not found – derived from charge_rate (>1 %/min = fast)")

# ============================================================
# 2. 特征工程：构建每辆车的综合特征矩阵
# ============================================================
print(f"\n{'='*70}")
print("【STEP 2】Feature Engineering (per-vehicle)")
print(f"{'='*70}")


def engineer_features(trips_df: pd.DataFrame, charging_df: pd.DataFrame) -> pd.DataFrame:
    """
    为每辆车计算放电特性、焦虑指标和充电行为特征。

    返回: DataFrame，每行一辆车，vehicle_id 作为普通列。
    """
    features_dict: dict = {}

    all_vehicles = trips_df['vehicle_id'].unique()

    for vehicle_id in all_vehicles:
        v_trips    = trips_df[trips_df['vehicle_id'] == vehicle_id]
        v_charging = charging_df[charging_df['vehicle_id'] == vehicle_id]

        feat: dict = {}

        # ── 1. 放电片段特性 ────────────────────────────────────────
        feat['avg_soc_drop']        = v_trips['trip_total_soc_drop'].mean()      if 'trip_total_soc_drop' in v_trips.columns else np.nan
        feat['std_soc_drop']        = v_trips['trip_total_soc_drop'].std()       if 'trip_total_soc_drop' in v_trips.columns else np.nan
        feat['avg_discharge_power'] = v_trips['trip_avg_power'].mean()           if 'trip_avg_power'      in v_trips.columns else np.nan
        feat['avg_trip_duration']   = v_trips['trip_duration_hrs'].mean()        if 'trip_duration_hrs'   in v_trips.columns else np.nan
        feat['total_distance']      = v_trips['trip_distance'].sum()             if 'trip_distance'       in v_trips.columns else np.nan
        feat['n_trips']             = len(v_trips)

        # ── 2. 驾驶模式占比 ────────────────────────────────────────
        for ratio_col in ['ratio_aggressive', 'ratio_conservative',
                          'ratio_moderate', 'ratio_highway']:
            if ratio_col in v_trips.columns:
                feat[ratio_col] = v_trips[ratio_col].mean()
            else:
                feat[ratio_col] = np.nan

        # ── 3. 车辆类别 ────────────────────────────────────────────
        if 'vehicle_type' in v_trips.columns:
            feat['vehicle_type'] = v_trips['vehicle_type'].mode().iloc[0]

        # ── 4. 焦虑指标 ────────────────────────────────────────────
        n_charging = len(v_charging)

        # 4a. 低SOC频率（充电触发SOC < 20%）
        if 'charge_trigger_soc' in v_trips.columns:
            feat['low_soc_freq']      = (v_trips['charge_trigger_soc'] < 20).mean()
            feat['critical_soc_freq'] = (v_trips['charge_trigger_soc'] < 10).mean()
            feat['min_soc']           = v_trips['charge_trigger_soc'].min()
        elif 'soc_start' in v_charging.columns:
            feat['low_soc_freq']      = (v_charging['soc_start'] < 20).mean()
            feat['critical_soc_freq'] = (v_charging['soc_start'] < 10).mean()
            feat['min_soc']           = v_charging['soc_start'].min()
        else:
            feat['low_soc_freq']      = np.nan
            feat['critical_soc_freq'] = np.nan
            feat['min_soc']           = np.nan

        # 4b. 充电开始时机（充电时的SOC越低 → 越焦虑）
        if 'soc_start' in v_charging.columns:
            feat['charge_start_soc_avg'] = v_charging['soc_start'].mean()
        elif 'charge_trigger_soc' in v_trips.columns:
            feat['charge_start_soc_avg'] = v_trips['charge_trigger_soc'].mean()
        else:
            feat['charge_start_soc_avg'] = np.nan

        # 4c. 日均充电频率
        if 'start_time' in v_trips.columns and v_trips['start_time'].notna().any():
            date_range = (v_trips['start_time'].max() - v_trips['start_time'].min()).days
            num_days   = max(date_range, 1)
            feat['daily_charge_freq'] = n_charging / num_days
        else:
            feat['daily_charge_freq'] = np.nan

        # 4d. 百公里充电频率
        if feat.get('total_distance', 0) and not np.isnan(feat.get('total_distance', np.nan)):
            total_km = feat['total_distance']
            feat['per_100km_charge_freq'] = (n_charging / total_km * 100) if total_km > 0 else np.nan
        else:
            feat['per_100km_charge_freq'] = np.nan

        # 4e. 充电间隔规律性（CV = std/mean，越小越规律）
        if 'start_time' in v_charging.columns and n_charging >= 3:
            intervals = v_charging['start_time'].sort_values().diff().dt.total_seconds().dropna()
            if len(intervals) > 0 and intervals.mean() > 0:
                feat['charge_interval_cv'] = intervals.std() / intervals.mean()
            else:
                feat['charge_interval_cv'] = np.nan
        else:
            feat['charge_interval_cv'] = np.nan

        # 4f. 充电完成率（soc_gain / 100）
        if 'soc_gain' in v_charging.columns:
            feat['charge_completion_rate'] = v_charging['soc_gain'].mean() / 100.0
        elif 'charge_gain_soc' in v_trips.columns:
            feat['charge_completion_rate'] = v_trips['charge_gain_soc'].mean() / 100.0
        else:
            feat['charge_completion_rate'] = np.nan

        # 4g. 快充依赖度（快充能量 / 总充电能量）
        if 'is_fast_charge' in v_charging.columns and 'soc_gain' in v_charging.columns:
            total_gain = v_charging['soc_gain'].sum()
            fast_gain  = v_charging.loc[v_charging['is_fast_charge'] == 1, 'soc_gain'].sum()
            feat['fast_charge_dependency'] = fast_gain / total_gain if total_gain > 0 else np.nan
        else:
            feat['fast_charge_dependency'] = np.nan

        # ── 5. 充电行为结果变量 ────────────────────────────────────
        if 'duration_minutes' in v_charging.columns:
            feat['avg_charge_duration']  = v_charging['duration_minutes'].mean()
        elif 'duration_seconds' in v_charging.columns:
            feat['avg_charge_duration']  = v_charging['duration_seconds'].mean() / 60.0
        else:
            feat['avg_charge_duration']  = np.nan

        if 'is_fast_charge' in v_charging.columns:
            feat['fast_charge_rate']     = v_charging['is_fast_charge'].mean()

        if 'soc_gain' in v_charging.columns:
            feat['avg_soc_gain']         = v_charging['soc_gain'].mean()

        # 开始充电时的SOC值（反映用户在多低的电量时才选择充电）
        if 'soc_start' in v_charging.columns:
            feat['avg_charge_start_soc']    = v_charging['soc_start'].mean()
            feat['median_charge_start_soc'] = v_charging['soc_start'].median()
        elif 'charge_trigger_soc' in v_trips.columns:
            feat['avg_charge_start_soc']    = v_trips['charge_trigger_soc'].mean()
            feat['median_charge_start_soc'] = v_trips['charge_trigger_soc'].median()
        else:
            feat['avg_charge_start_soc']    = np.nan
            feat['median_charge_start_soc'] = np.nan

        if 'start_hour' in v_charging.columns:
            feat['avg_charge_start_hour'] = v_charging['start_hour'].mean()

        features_dict[vehicle_id] = feat

    features_df = pd.DataFrame.from_dict(features_dict, orient='index')
    features_df = features_df.reset_index().rename(columns={'index': 'vehicle_id'})

    return features_df


features_df = engineer_features(trips_df, charging_df)
print(f"   ✓ Built feature matrix: {features_df.shape[0]:,} vehicles × {features_df.shape[1]} features")

# ============================================================
# 3. 计算综合焦虑评分
# ============================================================
print(f"\n{'='*70}")
print("【STEP 3】Computing Composite Anxiety Score")
print(f"{'='*70}")

ANXIETY_COMPONENTS = {
    # (column, weight, higher_is_more_anxious)
    'low_soc_freq':            (0.15, True),
    'critical_soc_freq':       (0.20, True),
    'min_soc':                 (0.10, False),   # 低值 → 高焦虑
    'charge_start_soc_avg':    (0.15, False),   # 低值 → 高焦虑
    'daily_charge_freq':       (0.12, True),
    'per_100km_charge_freq':   (0.12, True),
    'charge_interval_cv':      (0.08, True),
    'charge_completion_rate':  (0.04, False),   # 充越满 → 焦虑越低
    'fast_charge_dependency':  (0.04, True),
}

scaler = MinMaxScaler()
anxiety_cols_present = [c for c in ANXIETY_COMPONENTS if c in features_df.columns]

anxiety_matrix = features_df[anxiety_cols_present].copy()

# 对每列归一化到 [0, 1]，并处理"低值=高焦虑"的反向列
for col, (weight, higher_is_more) in ANXIETY_COMPONENTS.items():
    if col not in anxiety_matrix.columns:
        continue
    col_vals = anxiety_matrix[col].fillna(anxiety_matrix[col].median())
    col_min, col_max = col_vals.min(), col_vals.max()
    if col_max > col_min:
        normalized = (col_vals - col_min) / (col_max - col_min)
    else:
        normalized = pd.Series(0.5, index=col_vals.index)
    if not higher_is_more:
        normalized = 1.0 - normalized
    anxiety_matrix[col] = normalized

features_df['anxiety_score'] = sum(
    anxiety_matrix[col] * ANXIETY_COMPONENTS[col][0]
    for col in anxiety_cols_present
)

print(f"   Components used : {anxiety_cols_present}")
print(f"   Anxiety score   : mean={features_df['anxiety_score'].mean():.3f}, "
      f"std={features_df['anxiety_score'].std():.3f}")

anxiety_bins   = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
anxiety_labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
features_df['anxiety_level'] = pd.cut(
    features_df['anxiety_score'], bins=anxiety_bins, labels=anxiety_labels, include_lowest=True
)

# ============================================================
# 4. 描述性统计（按车辆类别）
# ============================================================
print(f"\n{'='*70}")
print("【STEP 4】Descriptive Statistics by Vehicle Type")
print(f"{'='*70}")

if 'vehicle_type' in features_df.columns:
    archetype_cols = ['avg_soc_drop', 'avg_discharge_power', 'ratio_aggressive',
                      'anxiety_score', 'avg_charge_duration', 'fast_charge_rate',
                      'avg_charge_start_soc', 'charge_completion_rate']
    archetype_cols = [c for c in archetype_cols if c in features_df.columns]

    archetype_stats = features_df.groupby('vehicle_type')[archetype_cols].agg(['mean', 'std']).round(3)
    print(archetype_stats.to_string())

    archetype_stats.to_csv(os.path.join(INPUT_DIR, "step15_archetype_comparison.csv"))
    print(f"\n   💾 Saved: step15_archetype_comparison.csv")

# ============================================================
# 5. 保存特征矩阵
# ============================================================
save_path = os.path.join(INPUT_DIR, "step15_vehicle_anxiety_features.csv")
features_df.to_csv(save_path, index=False)
print(f"\n   💾 Saved feature matrix: {save_path}")

# ============================================================
# 6. 统计分析
# ============================================================
print(f"\n{'='*70}")
print("【STEP 5】Statistical Analysis")
print(f"{'='*70}")

report_lines = ["=" * 70,
                "Step 15 Causality Analysis Report",
                "=" * 70, ""]

# 5.1 放电 → 充电时长 相关性
if 'avg_soc_drop' in features_df.columns and 'avg_charge_duration' in features_df.columns:
    valid = features_df[['avg_soc_drop', 'avg_charge_duration']].dropna()
    if len(valid) > 5:
        r, p = stats.pearsonr(valid['avg_soc_drop'], valid['avg_charge_duration'])
        line = f"Discharge SOC Drop vs Charge Duration      : r={r:.3f}, p={p:.4e}"
        print(f"   {line}")
        report_lines.append(line)

# 5.2 焦虑评分 → 快充选择
if 'anxiety_score' in features_df.columns and 'fast_charge_rate' in features_df.columns:
    valid = features_df[['anxiety_score', 'fast_charge_rate']].dropna()
    if len(valid) > 5:
        r, p = stats.pearsonr(valid['anxiety_score'], valid['fast_charge_rate'])
        line = f"Anxiety Score vs Fast Charge Rate           : r={r:.3f}, p={p:.4e}"
        print(f"   {line}")
        report_lines.append(line)

# 5.3 焦虑评分 → 充电时长
if 'anxiety_score' in features_df.columns and 'avg_charge_duration' in features_df.columns:
    valid = features_df[['anxiety_score', 'avg_charge_duration']].dropna()
    if len(valid) > 5:
        r, p = stats.pearsonr(valid['anxiety_score'], valid['avg_charge_duration'])
        line = f"Anxiety Score vs Charge Duration            : r={r:.3f}, p={p:.4e}"
        print(f"   {line}")
        report_lines.append(line)

# 5.4 开始充电SOC → 充电时长 / 快充率
if 'avg_charge_start_soc' in features_df.columns:
    if 'avg_charge_duration' in features_df.columns:
        valid = features_df[['avg_charge_start_soc', 'avg_charge_duration']].dropna()
        if len(valid) > 5:
            r, p = stats.pearsonr(valid['avg_charge_start_soc'], valid['avg_charge_duration'])
            line = f"Charge Start SOC vs Charge Duration         : r={r:.3f}, p={p:.4e}"
            print(f"   {line}")
            report_lines.append(line)
    if 'fast_charge_rate' in features_df.columns:
        valid = features_df[['avg_charge_start_soc', 'fast_charge_rate']].dropna()
        if len(valid) > 5:
            r, p = stats.pearsonr(valid['avg_charge_start_soc'], valid['fast_charge_rate'])
            line = f"Charge Start SOC vs Fast Charge Rate        : r={r:.3f}, p={p:.4e}"
            print(f"   {line}")
            report_lines.append(line)

# 5.5 车辆类别间焦虑差异 ANOVA
if 'vehicle_type' in features_df.columns and 'anxiety_score' in features_df.columns:
    groups_for_anova = [
        grp['anxiety_score'].dropna().values
        for _, grp in features_df.groupby('vehicle_type')
        if grp['anxiety_score'].dropna().shape[0] > 0
    ]
    if len(groups_for_anova) >= 2:
        f_stat, p_anova = stats.f_oneway(*groups_for_anova)
        line = f"ANOVA (Anxiety across vehicle types)        : F={f_stat:.4f}, p={p_anova:.4e}"
        print(f"   {line}")
        report_lines.append(line)

report_lines += ["", "Anxiety Score Distribution by Level:"]
if 'anxiety_level' in features_df.columns:
    for level, cnt in features_df['anxiety_level'].value_counts().sort_index().items():
        pct  = cnt / len(features_df) * 100
        line = f"   {level:<12}: {cnt:>5,} vehicles ({pct:5.1f}%)"
        print(line)
        report_lines.append(line)

report_path = os.path.join(INPUT_DIR, "step15_causality_report.txt")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("\n".join(report_lines))
print(f"\n   💾 Saved report: {report_path}")

# ============================================================
# 7. 可视化
# ============================================================
print(f"\n{'='*70}")
print("【STEP 6】Visualization")
print(f"{'='*70}")

PALETTE = ['#4878CF', '#6ACC65', '#D65F5F', '#B47CC7']

# ── Figure 1: 焦虑评分分布（按车辆类别）──────────────────────
if 'vehicle_type' in features_df.columns:
    fig, ax = plt.subplots(figsize=(10, 5))
    vtypes = sorted(features_df['vehicle_type'].dropna().unique())
    data_for_plot = [features_df.loc[features_df['vehicle_type'] == vt, 'anxiety_score'].dropna()
                     for vt in vtypes]
    bp = ax.boxplot(data_for_plot, labels=vtypes, patch_artist=True,
                    medianprops=dict(color='black', linewidth=2))
    for patch, color in zip(bp['boxes'], PALETTE[:len(vtypes)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_title('Anxiety Score Distribution by Vehicle Archetype', fontweight='bold')
    ax.set_ylabel('Composite Anxiety Score (0–1)')
    ax.set_xlabel('Vehicle Archetype')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(FIGURE_DIR, "01_anxiety_by_archetype.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: 01_anxiety_by_archetype.png")

# ── Figure 2: 放电量 vs 充电时长 ────────────────────────────
if 'avg_soc_drop' in features_df.columns and 'avg_charge_duration' in features_df.columns:
    valid = features_df[['avg_soc_drop', 'avg_charge_duration', 'vehicle_type']].dropna(
        subset=['avg_soc_drop', 'avg_charge_duration'])
    if len(valid) > 5:
        fig, ax = plt.subplots(figsize=(8, 6))
        if 'vehicle_type' in valid.columns:
            vtypes = sorted(valid['vehicle_type'].dropna().unique())
            for i, vt in enumerate(vtypes):
                sub = valid[valid['vehicle_type'] == vt]
                ax.scatter(sub['avg_soc_drop'], sub['avg_charge_duration'],
                           label=str(vt), alpha=0.6, s=40, color=PALETTE[i % len(PALETTE)])
        else:
            ax.scatter(valid['avg_soc_drop'], valid['avg_charge_duration'], alpha=0.5, s=40)

        x_vals = valid['avg_soc_drop'].values
        y_vals = valid['avg_charge_duration'].values
        slope, intercept, r_val, p_val, _ = stats.linregress(x_vals, y_vals)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
        ax.plot(x_line, slope * x_line + intercept, 'k--', linewidth=1.5,
                label=f'Linear fit (r={r_val:.2f}, p={p_val:.3f})')
        ax.set_xlabel('Avg SOC Drop per Discharge Trip (%)', fontweight='bold')
        ax.set_ylabel('Avg Charging Duration (min)', fontweight='bold')
        ax.set_title('Discharge SOC Drop vs Charging Duration', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig_path = os.path.join(FIGURE_DIR, "02_discharge_vs_charge_duration.png")
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: 02_discharge_vs_charge_duration.png")

# ── Figure 3: 焦虑评分 vs 快充选择 ──────────────────────────
if 'anxiety_score' in features_df.columns and 'fast_charge_rate' in features_df.columns:
    valid = features_df[['anxiety_score', 'fast_charge_rate', 'vehicle_type']].dropna(
        subset=['anxiety_score', 'fast_charge_rate'])
    if len(valid) > 5:
        fig, ax = plt.subplots(figsize=(8, 6))
        if 'vehicle_type' in valid.columns:
            vtypes = sorted(valid['vehicle_type'].dropna().unique())
            for i, vt in enumerate(vtypes):
                sub = valid[valid['vehicle_type'] == vt]
                ax.scatter(sub['anxiety_score'], sub['fast_charge_rate'],
                           label=str(vt), alpha=0.6, s=40, color=PALETTE[i % len(PALETTE)])
        else:
            ax.scatter(valid['anxiety_score'], valid['fast_charge_rate'], alpha=0.5, s=40)

        x_vals = valid['anxiety_score'].values
        y_vals = valid['fast_charge_rate'].values
        slope, intercept, r_val, p_val, _ = stats.linregress(x_vals, y_vals)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
        ax.plot(x_line, slope * x_line + intercept, 'k--', linewidth=1.5,
                label=f'Linear fit (r={r_val:.2f})')
        ax.set_xlabel('Composite Anxiety Score', fontweight='bold')
        ax.set_ylabel('Fast Charge Usage Rate', fontweight='bold')
        ax.set_title('Anxiety Level vs Fast Charging Usage', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig_path = os.path.join(FIGURE_DIR, "03_anxiety_vs_fast_charge.png")
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: 03_anxiety_vs_fast_charge.png")

# ── Figure 4: 驾驶模式占比 vs 焦虑评分 热力图 ────────────────
ratio_cols = [c for c in ['ratio_aggressive', 'ratio_conservative',
                           'ratio_moderate', 'ratio_highway']
              if c in features_df.columns]
if ratio_cols and 'anxiety_score' in features_df.columns:
    corr_cols = ratio_cols + ['avg_soc_drop', 'avg_discharge_power',
                               'anxiety_score', 'avg_charge_duration',
                               'avg_charge_start_soc', 'fast_charge_rate']
    corr_cols = [c for c in corr_cols if c in features_df.columns]
    corr_matrix = features_df[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, ax=ax, mask=mask,
                linewidths=0.5, linecolor='white', cbar_kws={'shrink': 0.8})
    ax.set_title('Correlation Heatmap: Discharge / Mode / Anxiety / Charging', fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(FIGURE_DIR, "04_correlation_heatmap.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: 04_correlation_heatmap.png")

# ── Figure 5: 各焦虑指标对比（按车辆类别）────────────────────
if 'vehicle_type' in features_df.columns:
    anxiety_component_cols = [c for c in ANXIETY_COMPONENTS if c in features_df.columns]
    if anxiety_component_cols:
        n_cols = 3
        n_rows = int(np.ceil(len(anxiety_component_cols) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = np.array(axes).flatten()

        vtypes = sorted(features_df['vehicle_type'].dropna().unique())
        for idx, comp in enumerate(anxiety_component_cols):
            ax = axes[idx]
            data_for_box = [
                features_df.loc[features_df['vehicle_type'] == vt, comp].dropna()
                for vt in vtypes
            ]
            bp = ax.boxplot(data_for_box, labels=vtypes, patch_artist=True,
                            medianprops=dict(color='black', linewidth=1.5))
            for patch, color in zip(bp['boxes'], PALETTE[:len(vtypes)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.65)
            ax.set_title(comp.replace('_', ' ').title(), fontsize=10)
            ax.grid(axis='y', alpha=0.3)

        for idx in range(len(anxiety_component_cols), len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle('Anxiety Component Comparison by Vehicle Archetype',
                     fontweight='bold', fontsize=13)
        plt.tight_layout()
        fig_path = os.path.join(FIGURE_DIR, "05_anxiety_components_by_archetype.png")
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: 05_anxiety_components_by_archetype.png")

# ── Figure 6: 焦虑等级 vs 充电时长 & 快充比例 & 开始充电SOC ──
if ('anxiety_level' in features_df.columns
        and 'avg_charge_duration' in features_df.columns):

    has_start_soc = 'avg_charge_start_soc' in features_df.columns
    n_panels = 3 if has_start_soc else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))

    level_order = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    level_order = [l for l in level_order if l in features_df['anxiety_level'].unique()]
    colors_level = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(level_order)))

    # Panel 0: 充电时长
    ax = axes[0]
    data_dur = [features_df.loc[features_df['anxiety_level'] == lv,
                                'avg_charge_duration'].dropna()
                for lv in level_order]
    bp = ax.boxplot(data_dur, labels=level_order, patch_artist=True,
                    medianprops=dict(color='black', linewidth=2))
    for patch, color in zip(bp['boxes'], colors_level):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_title('Anxiety Level vs Avg Charging Duration', fontweight='bold')
    ax.set_ylabel('Avg Charge Duration (min)')
    ax.set_xlabel('Anxiety Level')
    ax.grid(axis='y', alpha=0.3)

    # Panel 1: 快充比例
    if 'fast_charge_rate' in features_df.columns:
        ax = axes[1]
        mean_fcr = [features_df.loc[features_df['anxiety_level'] == lv,
                                    'fast_charge_rate'].mean()
                    for lv in level_order]
        ax.bar(level_order, mean_fcr, color=colors_level, edgecolor='black', linewidth=1)
        ax.set_title('Anxiety Level vs Fast Charge Usage Rate', fontweight='bold')
        ax.set_ylabel('Fast Charge Rate (mean)')
        ax.set_xlabel('Anxiety Level')
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)

    # Panel 2: 开始充电时的SOC（新增）
    if has_start_soc:
        ax = axes[2]
        data_soc = [features_df.loc[features_df['anxiety_level'] == lv,
                                    'avg_charge_start_soc'].dropna()
                    for lv in level_order]
        bp2 = ax.boxplot(data_soc, labels=level_order, patch_artist=True,
                         medianprops=dict(color='black', linewidth=2))
        for patch, color in zip(bp2['boxes'], colors_level):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax.set_title('Anxiety Level vs Charge Start SOC', fontweight='bold')
        ax.set_ylabel('Avg Charge Start SOC (%)')
        ax.set_xlabel('Anxiety Level')
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Impact of Anxiety Level on Charging Behavior', fontweight='bold', fontsize=13)
    plt.tight_layout()
    fig_path = os.path.join(FIGURE_DIR, "06_anxiety_level_vs_charging.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: 06_anxiety_level_vs_charging.png")

# ── Figure 7: 开始充电SOC 分布（按车辆类别）─────────────────
if 'avg_charge_start_soc' in features_df.columns and 'vehicle_type' in features_df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    vtypes = sorted(features_df['vehicle_type'].dropna().unique())

    # 左: 箱线图
    ax = axes[0]
    data_soc_vt = [features_df.loc[features_df['vehicle_type'] == vt,
                                   'avg_charge_start_soc'].dropna()
                   for vt in vtypes]
    bp = ax.boxplot(data_soc_vt, labels=vtypes, patch_artist=True,
                    medianprops=dict(color='black', linewidth=2))
    for patch, color in zip(bp['boxes'], PALETTE[:len(vtypes)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_title('Charge Start SOC by Vehicle Archetype', fontweight='bold')
    ax.set_ylabel('Avg Charge Start SOC (%)')
    ax.set_xlabel('Vehicle Archetype')
    ax.grid(axis='y', alpha=0.3)

    # 右: 充电开始SOC vs 充电时长散点图
    ax = axes[1]
    if 'avg_charge_duration' in features_df.columns:
        valid = features_df[['avg_charge_start_soc', 'avg_charge_duration',
                              'vehicle_type']].dropna(
            subset=['avg_charge_start_soc', 'avg_charge_duration'])
        if len(valid) > 5:
            for i, vt in enumerate(vtypes):
                sub = valid[valid['vehicle_type'] == vt]
                ax.scatter(sub['avg_charge_start_soc'], sub['avg_charge_duration'],
                           label=str(vt), alpha=0.6, s=40, color=PALETTE[i % len(PALETTE)])
            x_vals = valid['avg_charge_start_soc'].values
            y_vals = valid['avg_charge_duration'].values
            slope, intercept, r_val, p_val, _ = stats.linregress(x_vals, y_vals)
            x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
            ax.plot(x_line, slope * x_line + intercept, 'k--', linewidth=1.5,
                    label=f'Linear fit (r={r_val:.2f}, p={p_val:.3f})')
            ax.legend(fontsize=9)
    ax.set_title('Charge Start SOC vs Charging Duration', fontweight='bold')
    ax.set_xlabel('Avg Charge Start SOC (%)', fontweight='bold')
    ax.set_ylabel('Avg Charging Duration (min)', fontweight='bold')
    ax.grid(alpha=0.3)

    plt.suptitle('Charge Start SOC Analysis', fontweight='bold', fontsize=13)
    plt.tight_layout()
    fig_path = os.path.join(FIGURE_DIR, "07_charge_start_soc_analysis.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: 07_charge_start_soc_analysis.png")

# ============================================================
# 8. 完成
# ============================================================
print(f"\n{'='*70}")
print("✅ Step 15 Complete!")
print(f"   Output directory : {INPUT_DIR}")
print(f"   Figures          : {FIGURE_DIR}")
print(f"{'='*70}")
