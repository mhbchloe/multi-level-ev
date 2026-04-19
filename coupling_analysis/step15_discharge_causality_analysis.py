"""
Step 15: Discharge Pattern Analysis + Causality Charging Analysis
放电规律分析 + 因果充电分析

逻辑链：
  车辆类型 (Archetype)
      ↓
  驾驶特性 (Driving Pattern)
      ↓
  放电规律 (Discharge Pattern)
      ↓
  充电需求 (Charging Demand)

数据源:
  - segments_integrated_complete.csv  (段级数据, soc_start/soc_end/cluster_id/...)
  - vehicle_clustering_kmeans_final_k4.csv (或 vehicle_clustering_gmm_k4.csv)
  - charging_events_stationary_only.csv   (充电事件)

输出:
  - discharge_patterns_by_archetype.csv
  - causality_charging_analysis.csv
  - 6 张对比图表
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['font.size'] = 12

print("=" * 80)
print("⚡ Step 15: Discharge Pattern Analysis + Causality Charging Analysis")
print("=" * 80)

# ============================================================
# 0. 配置
# ============================================================
RESULTS_DIR = "./coupling_analysis/results/"
OUTPUT_DIR = "./coupling_analysis/results/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

VEHICLE_NAMES = {
    0: 'Long-Distance\nHighway (LDH)',
    1: 'Stationary/\nOccasional (SOC)',
    2: 'Urban\nCommuter (UCO)',
    3: 'Multi-purpose\nMixed (MUM)',
}
VEHICLE_NAMES_SHORT = {
    0: 'LDH',
    1: 'SOC',
    2: 'UCO',
    3: 'MUM',
}

CLUSTER_NAMES = {
    0: 'Idle',
    1: 'Urban',
    2: 'Highway',
    3: 'Short',
}

ARCHETYPE_COLORS = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12']
CLUSTER_COLORS   = ['#95a5a6', '#2ecc71', '#e74c3c', '#f39c12']

# Causality analysis parameters
LOOKBACK_HOURS       = 24   # how many hours before a charging event to search for discharge segs
MAX_PRECEDING_SEGS   = 50   # cap on number of preceding segments to aggregate

# Scatter plot sample sizes (avoids over-plotting on large datasets)
SCATTER_SAMPLE_LARGE = 2000
SCATTER_SAMPLE_SMALL = 1000

# ============================================================
# 1. 加载数据
# ============================================================
print(f"\n{'='*80}")
print("【STEP 1】Loading Data")
print("=" * 80)

# 1.1 段级数据
segments_path = os.path.join(RESULTS_DIR, 'segments_integrated_complete.csv')
print(f"\n   Loading segments from: {segments_path}")
segments = pd.read_csv(segments_path)
print(f"   ✓ Segments: {len(segments):,} rows, {segments['vehicle_id'].nunique():,} vehicles")
print(f"   Columns: {list(segments.columns)}")

# 统一 vehicle_id 为字符串
segments['vehicle_id'] = segments['vehicle_id'].astype(str)

# 时间列：尝试 start_dt 或 start_time
if 'start_dt' in segments.columns:
    segments['start_time'] = pd.to_datetime(segments['start_dt'], errors='coerce')
elif 'start_time' in segments.columns:
    segments['start_time'] = pd.to_datetime(segments['start_time'], errors='coerce')
else:
    raise ValueError("segments CSV must have 'start_dt' or 'start_time' column")

# end_time
if 'end_dt' in segments.columns:
    segments['end_time'] = pd.to_datetime(segments['end_dt'], errors='coerce')
elif 'end_time' in segments.columns:
    segments['end_time'] = pd.to_datetime(segments['end_time'], errors='coerce')
elif 'duration_seconds' in segments.columns:
    segments['end_time'] = segments['start_time'] + pd.to_timedelta(
        segments['duration_seconds'], unit='s')
else:
    segments['end_time'] = segments['start_time']

# 计算 soc_drop（如果没有直接字段）
if 'soc_drop' not in segments.columns:
    if 'soc_start' in segments.columns and 'soc_end' in segments.columns:
        segments['soc_drop'] = segments['soc_start'] - segments['soc_end']
    else:
        raise ValueError("segments CSV must have 'soc_drop' or 'soc_start'+'soc_end' columns")

# 添加时间特征
segments['start_hour']    = segments['start_time'].dt.hour
segments['start_weekday'] = segments['start_time'].dt.dayofweek   # 0=Mon
segments['date']          = segments['start_time'].dt.date

# duration_seconds（确保存在）
if 'duration_seconds' not in segments.columns:
    segments['duration_seconds'] = (
        segments['end_time'] - segments['start_time']
    ).dt.total_seconds().fillna(0)

# 放电速率 (%/min)
dur_min = segments['duration_seconds'] / 60.0
dur_min = dur_min.replace(0, np.nan)
segments['discharge_rate'] = segments['soc_drop'] / dur_min

print(f"\n   SOC drop range: [{segments['soc_drop'].min():.2f}, {segments['soc_drop'].max():.2f}]")

# 1.2 车辆聚类
vehicle_cluster_candidates = [
    os.path.join('./vehicle_clustering/results/', 'vehicle_clustering_kmeans_final_k4.csv'),
    os.path.join('./vehicle_clustering/results/', 'vehicle_clustering_gmm_k4.csv'),
    os.path.join(RESULTS_DIR, 'vehicle_clustering_kmeans_final_k4.csv'),
    os.path.join(RESULTS_DIR, 'vehicle_clustering_gmm_k4.csv'),
]

vehicle_cluster_df = None
for path in vehicle_cluster_candidates:
    if os.path.exists(path):
        vehicle_cluster_df = pd.read_csv(path)
        print(f"\n   ✓ Vehicle clustering from: {path}")
        break

if vehicle_cluster_df is None:
    raise FileNotFoundError(
        "Vehicle clustering file not found. Expected one of:\n" +
        "\n".join(f"  {p}" for p in vehicle_cluster_candidates)
    )

vehicle_cluster_df['vehicle_id'] = vehicle_cluster_df['vehicle_id'].astype(str)
print(f"   ✓ Vehicle clusters: {len(vehicle_cluster_df):,} vehicles, "
      f"columns: {list(vehicle_cluster_df.columns)}")

# 1.3 充电事件
charging_candidates = [
    os.path.join(RESULTS_DIR, 'charging_events_stationary_only.csv'),
    os.path.join(RESULTS_DIR, 'charging_events_meaningful.csv'),
    os.path.join(RESULTS_DIR, 'charging_events_clean.csv'),
    os.path.join(RESULTS_DIR, 'charging_events_stationary_meaningful.csv'),
    os.path.join(RESULTS_DIR, 'charging_events_raw_extracted.csv'),
]

charging_df = None
for path in charging_candidates:
    if os.path.exists(path):
        charging_df = pd.read_csv(path)
        print(f"\n   ✓ Charging events from: {path}")
        break

if charging_df is None:
    print("\n   ⚠️  No charging events file found. Part 2 will be skipped.")
    print(f"      Tried: {charging_candidates}")
else:
    charging_df['vehicle_id'] = charging_df['vehicle_id'].astype(str)
    charging_df['start_time'] = pd.to_datetime(charging_df['start_time'], errors='coerce')
    if 'end_time' in charging_df.columns:
        charging_df['end_time'] = pd.to_datetime(charging_df['end_time'], errors='coerce')
    if 'soc_gain' not in charging_df.columns:
        if 'soc_start' in charging_df.columns and 'soc_end' in charging_df.columns:
            charging_df['soc_gain'] = charging_df['soc_end'] - charging_df['soc_start']
    if 'start_hour' not in charging_df.columns:
        charging_df['start_hour'] = charging_df['start_time'].dt.hour
    print(f"   ✓ Charging events: {len(charging_df):,}")

# ============================================================
# 2. 合并车辆聚类到段级数据
# ============================================================
print(f"\n{'='*80}")
print("【STEP 2】Merging Vehicle Cluster Labels")
print("=" * 80)

# 确保 vehicle_cluster 列名正确
if 'vehicle_cluster' not in vehicle_cluster_df.columns:
    # 尝试其他可能的列名
    for alt in ['cluster', 'cluster_id', 'label', 'vehicle_type']:
        if alt in vehicle_cluster_df.columns:
            vehicle_cluster_df = vehicle_cluster_df.rename(columns={alt: 'vehicle_cluster'})
            break

merge_cols = ['vehicle_id', 'vehicle_cluster']
if 'cluster_label' in vehicle_cluster_df.columns:
    merge_cols.append('cluster_label')

segments = segments.merge(
    vehicle_cluster_df[merge_cols],
    on='vehicle_id', how='left'
)

n_matched = segments['vehicle_cluster'].notna().sum()
print(f"   ✓ Matched {n_matched:,} / {len(segments):,} segments "
      f"({n_matched/len(segments)*100:.1f}%)")

# 只保留有聚类标签的段
segments = segments.dropna(subset=['vehicle_cluster'])
segments['vehicle_cluster'] = segments['vehicle_cluster'].astype(int)
print(f"   ✓ Retained {len(segments):,} segments with cluster labels")

for arch in sorted(segments['vehicle_cluster'].unique()):
    n = (segments['vehicle_cluster'] == arch).sum()
    print(f"      Archetype {arch} ({VEHICLE_NAMES_SHORT[arch]}): {n:,} segments")

# ============================================================
# 3. Part 1: 放电规律分析
# ============================================================
print(f"\n{'='*80}")
print("【PART 1】Discharge Pattern Analysis")
print("=" * 80)

# 只取放电段（soc_drop > 0）
discharge_segs = segments[segments['soc_drop'] > 0].copy()
print(f"\n   Discharge segments: {len(discharge_segs):,} / {len(segments):,} "
      f"({len(discharge_segs)/len(segments)*100:.1f}%)")

# ---- 3.1 按车型统计放电分布 ----
print(f"\n   3.1 Discharge Distribution Statistics by Archetype:")

discharge_stats = []

for arch in sorted(segments['vehicle_cluster'].unique()):
    arch_segs = discharge_segs[discharge_segs['vehicle_cluster'] == arch]

    # 放电速率（每辆车的均值）
    rate_vals = arch_segs['discharge_rate'].replace([np.inf, -np.inf], np.nan).dropna()

    # 按驾驶模式的放电分布
    mode_counts = {}
    mode_discharge = {}
    cluster_id_col = 'cluster_id' if 'cluster_id' in arch_segs.columns else 'cluster'
    if cluster_id_col in arch_segs.columns:
        total_discharge = arch_segs['soc_drop'].sum()
        for mode in [0, 1, 2, 3]:
            mode_mask = arch_segs[cluster_id_col] == mode
            mode_counts[f'n_segs_mode_{mode}'] = int(mode_mask.sum())
            mode_discharge[f'discharge_mode_{mode}'] = float(
                arch_segs.loc[mode_mask, 'soc_drop'].sum()
            )
            mode_discharge[f'discharge_ratio_mode_{mode}'] = float(
                arch_segs.loc[mode_mask, 'soc_drop'].sum() / max(total_discharge, 1e-9)
            )

    # 每辆车的日均放电
    daily_per_vehicle = (
        arch_segs.groupby(['vehicle_id', 'date'])['soc_drop']
        .sum()
        .reset_index()
        .groupby('vehicle_id')['soc_drop']
        .mean()
    )

    row = {
        'archetype': arch,
        'archetype_name': VEHICLE_NAMES_SHORT[arch],
        'n_segments': len(arch_segs),
        'n_vehicles': arch_segs['vehicle_id'].nunique(),
        'avg_soc_drop': float(arch_segs['soc_drop'].mean()),
        'std_soc_drop': float(arch_segs['soc_drop'].std()),
        'p25_soc_drop': float(arch_segs['soc_drop'].quantile(0.25)),
        'p75_soc_drop': float(arch_segs['soc_drop'].quantile(0.75)),
        'max_soc_drop': float(arch_segs['soc_drop'].max()),
        'avg_discharge_rate_pct_per_min': float(rate_vals.mean()) if len(rate_vals) > 0 else np.nan,
        'std_discharge_rate': float(rate_vals.std()) if len(rate_vals) > 0 else np.nan,
        'avg_daily_discharge_per_vehicle': float(daily_per_vehicle.mean()),
        'std_daily_discharge_per_vehicle': float(daily_per_vehicle.std()),
    }
    row.update(mode_counts)
    row.update(mode_discharge)
    discharge_stats.append(row)

    print(f"\n   Archetype {arch} ({VEHICLE_NAMES_SHORT[arch]}):")
    print(f"      Discharge segments: {row['n_segments']:,}, vehicles: {row['n_vehicles']:,}")
    print(f"      Avg SOC drop: {row['avg_soc_drop']:.2f}% (±{row['std_soc_drop']:.2f}%)")
    print(f"      Avg discharge rate: {row['avg_discharge_rate_pct_per_min']:.4f} %/min")
    print(f"      Avg daily discharge/vehicle: {row['avg_daily_discharge_per_vehicle']:.2f}%")

discharge_patterns_df = pd.DataFrame(discharge_stats)

# ---- 3.2 按小时统计放电量 ----
print(f"\n   3.2 Hourly Discharge Distribution...")

hourly_discharge = (
    discharge_segs.groupby(['vehicle_cluster', 'start_hour'])['soc_drop']
    .agg(['sum', 'mean', 'count'])
    .reset_index()
)
hourly_discharge.columns = ['vehicle_cluster', 'start_hour', 'total_discharge', 'mean_discharge', 'count']

# ---- 3.3 每辆车的日均放电 ----
print(f"\n   3.3 Daily Average Discharge per Vehicle...")

daily_discharge_per_vehicle = (
    discharge_segs.groupby(['vehicle_id', 'date'])['soc_drop']
    .sum()
    .reset_index()
)
daily_discharge_per_vehicle.columns = ['vehicle_id', 'date', 'daily_soc_drop']

# 添加车辆聚类
daily_discharge_per_vehicle = daily_discharge_per_vehicle.merge(
    vehicle_cluster_df[['vehicle_id', 'vehicle_cluster']],
    on='vehicle_id', how='left'
)

# ---- 3.4 驾驶模式对放电的贡献度 ----
print(f"\n   3.4 Driving Mode Discharge Contribution...")

cluster_id_col = 'cluster_id' if 'cluster_id' in discharge_segs.columns else 'cluster'
if cluster_id_col in discharge_segs.columns:
    mode_contribution = (
        discharge_segs.groupby(['vehicle_cluster', cluster_id_col])['soc_drop']
        .sum()
        .reset_index()
    )
    mode_contribution.columns = ['vehicle_cluster', 'driving_mode', 'total_discharge']

    # 计算每类车中各模式的占比
    total_by_archetype = mode_contribution.groupby('vehicle_cluster')['total_discharge'].sum()
    mode_contribution['discharge_ratio'] = (
        mode_contribution.apply(
            lambda r: r['total_discharge'] / total_by_archetype[r['vehicle_cluster']],
            axis=1
        )
    )

    # 透视表：行=车型，列=驾驶模式
    mode_pivot = mode_contribution.pivot(
        index='vehicle_cluster', columns='driving_mode', values='discharge_ratio'
    ).fillna(0)
    mode_pivot.columns = [CLUSTER_NAMES.get(c, f'Mode{c}') for c in mode_pivot.columns]
else:
    mode_pivot = None
    print("   ⚠️  cluster_id column not found, skipping mode contribution analysis")

# ============================================================
# 4. Part 2: 因果充电分析
# ============================================================
do_causality = charging_df is not None

if do_causality:
    print(f"\n{'='*80}")
    print("【PART 2】Causality Charging Analysis")
    print("=" * 80)

    # 合并充电事件与车辆聚类
    charge_with_arch = charging_df.merge(
        vehicle_cluster_df[['vehicle_id', 'vehicle_cluster']],
        on='vehicle_id', how='inner'
    )
    print(f"\n   Charging events with archetype: {len(charge_with_arch):,}")

    # 将 segments 排序，便于后续 lookup
    segments_sorted = segments.sort_values(['vehicle_id', 'start_time']).reset_index(drop=True)

    # ---- 4.1 为每个充电事件找前序放电段 ----
    print(f"\n   4.1 Finding preceding discharge segments for each charging event...")
    print(f"       (using {LOOKBACK_HOURS}h lookback window)")

    causality_records = []

    vehicle_groups = {
        vid: grp.reset_index(drop=True)
        for vid, grp in segments_sorted.groupby('vehicle_id')
    }

    for idx, charge_event in charge_with_arch.iterrows():
        vid = charge_event['vehicle_id']
        charge_start = charge_event['start_time']
        arch = int(charge_event['vehicle_cluster'])

        if vid not in vehicle_groups:
            continue

        v_segs = vehicle_groups[vid]

        # 取充电开始前 LOOKBACK_HOURS 小时内的所有放电段
        lookback_start = charge_start - pd.Timedelta(hours=LOOKBACK_HOURS)
        mask = (
            (v_segs['end_time'] <= charge_start) &
            (v_segs['end_time'] >= lookback_start) &
            (v_segs['soc_drop'] > 0)
        )
        preceding = v_segs[mask].tail(MAX_PRECEDING_SEGS)

        total_preceding_discharge = float(preceding['soc_drop'].sum())
        total_preceding_duration_h = float(preceding['duration_seconds'].sum()) / 3600.0
        n_preceding_segs = int(len(preceding))

        # 从充电事件提取所需字段
        soc_at_charge = float(charge_event.get('soc_start', np.nan))
        soc_gain = float(charge_event.get('soc_gain', np.nan))

        # 放电到充电的时间延迟
        if len(preceding) > 0 and not preceding['end_time'].isna().all():
            last_discharge_end = preceding['end_time'].max()
            if pd.notna(last_discharge_end):
                delay_hours = (charge_start - last_discharge_end).total_seconds() / 3600.0
            else:
                delay_hours = np.nan
        else:
            delay_hours = np.nan

        causality_records.append({
            'vehicle_id': vid,
            'archetype': arch,
            'archetype_name': VEHICLE_NAMES_SHORT[arch],
            'charge_start_time': charge_start,
            'charge_start_hour': int(charge_event.get('start_hour', charge_start.hour)),
            'soc_at_charge_start': soc_at_charge,
            'soc_gain': soc_gain,
            'preceding_discharge': total_preceding_discharge,
            'preceding_duration_hours': total_preceding_duration_h,
            'preceding_segments_count': n_preceding_segs,
            'delay_hours': delay_hours,
            'discharge_to_charge_ratio': (
                total_preceding_discharge / soc_gain
                if (pd.notna(soc_gain) and soc_gain > 0) else np.nan
            ),
        })

    causality_df = pd.DataFrame(causality_records)
    print(f"   ✓ Causality records: {len(causality_df):,}")

    # ---- 4.2 按车型统计因果关系 ----
    print(f"\n   4.2 Charging-Discharge Causality by Archetype:")

    for arch in sorted(causality_df['archetype'].unique()):
        arch_df = causality_df[causality_df['archetype'] == arch]

        corr_val = arch_df['soc_gain'].corr(arch_df['preceding_discharge'])
        avg_preceding = arch_df['preceding_discharge'].mean()
        avg_gain = arch_df['soc_gain'].mean()
        avg_delay = arch_df['delay_hours'].mean()

        print(f"\n   Archetype {arch} ({VEHICLE_NAMES_SHORT[arch]}):")
        print(f"      Events: {len(arch_df):,}")
        print(f"      Avg preceding discharge (24h): {avg_preceding:.2f}%")
        print(f"      Avg soc_gain per charge: {avg_gain:.2f}%")
        print(f"      Correlation (soc_gain ~ preceding_discharge): {corr_val:.3f}")
        print(f"      Avg delay (discharge→charge): {avg_delay:.2f} h")

else:
    causality_df = pd.DataFrame()

# ============================================================
# 5. 可视化
# ============================================================
print(f"\n{'='*80}")
print("【STEP 5】Generating Visualizations")
print("=" * 80)

# ---- 图1: 放电分布对比 (boxplot + discharge rate) ----
print(f"\n   Generating Figure 1: Discharge Distribution...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

arch_labels = [VEHICLE_NAMES_SHORT[a] for a in sorted(discharge_segs['vehicle_cluster'].unique())]

# 1a: SOC drop 箱线图
ax = axes[0]
data_for_box = [
    discharge_segs[discharge_segs['vehicle_cluster'] == arch]['soc_drop'].values
    for arch in sorted(discharge_segs['vehicle_cluster'].unique())
]
bp = ax.boxplot(
    data_for_box,
    labels=arch_labels,
    patch_artist=True,
    medianprops=dict(color='black', linewidth=2),
    whiskerprops=dict(linewidth=1.5),
    capprops=dict(linewidth=1.5),
    flierprops=dict(marker='o', markersize=2, alpha=0.3),
)
for patch, color in zip(bp['boxes'], ARCHETYPE_COLORS[:len(arch_labels)]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_xlabel('Vehicle Archetype', fontweight='bold')
ax.set_ylabel('SOC Drop per Segment (%)', fontweight='bold')
ax.set_title('(a) SOC Drop Distribution by Archetype\n(Discharge per segment)',
             fontweight='bold', fontsize=12)
ax.grid(alpha=0.3, axis='y')

# 统计标注
for i, arch in enumerate(sorted(discharge_segs['vehicle_cluster'].unique())):
    arch_data = discharge_segs[discharge_segs['vehicle_cluster'] == arch]['soc_drop']
    ax.text(
        i + 1, arch_data.quantile(0.75) + 0.5,
        f'μ={arch_data.mean():.2f}%',
        ha='center', fontsize=9, color=ARCHETYPE_COLORS[i]
    )

# 1b: 放电速率分布
ax = axes[1]
for arch in sorted(discharge_segs['vehicle_cluster'].unique()):
    arch_data = (
        discharge_segs[discharge_segs['vehicle_cluster'] == arch]['discharge_rate']
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    # 截断到 p99 以去除极端值
    p99 = arch_data.quantile(0.99)
    arch_data_clipped = arch_data[arch_data <= p99]

    ax.hist(
        arch_data_clipped,
        bins=50,
        density=True,
        alpha=0.5,
        color=ARCHETYPE_COLORS[arch],
        label=f'{VEHICLE_NAMES_SHORT[arch]} (μ={arch_data.mean():.3f})',
        linewidth=0.8,
        edgecolor=ARCHETYPE_COLORS[arch],
    )

ax.set_xlabel('Discharge Rate (%/min)', fontweight='bold')
ax.set_ylabel('Density', fontweight='bold')
ax.set_title('(b) Discharge Rate Distribution by Archetype\n(SOC drop per minute)',
             fontweight='bold', fontsize=12)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.suptitle('Figure 1: Discharge Distribution Comparison by Vehicle Archetype',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()

fig1_path = os.path.join(OUTPUT_DIR, 'fig1_discharge_distribution.png')
plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✅ Saved: fig1_discharge_distribution.png")

# ---- 图2: 24小时放电时间分布 ----
print(f"\n   Generating Figure 2: 24-hour Discharge Distribution...")

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# 2a: 每小时累积放电量
ax = axes[0]
for arch in sorted(discharge_segs['vehicle_cluster'].unique()):
    arch_hourly = (
        hourly_discharge[hourly_discharge['vehicle_cluster'] == arch]
        .set_index('start_hour')['mean_discharge']
        .reindex(range(24), fill_value=0)
    )
    ax.plot(
        range(24), arch_hourly.values,
        color=ARCHETYPE_COLORS[arch],
        linewidth=2.5,
        marker='o', markersize=5,
        label=VEHICLE_NAMES_SHORT[arch],
    )

ax.set_xlabel('Hour of Day', fontweight='bold')
ax.set_ylabel('Avg SOC Drop per Segment (%)', fontweight='bold')
ax.set_title('(a) Mean Discharge by Hour of Day\n(by vehicle archetype)',
             fontweight='bold', fontsize=12)
ax.set_xticks(range(0, 24, 2))
ax.set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 2)], rotation=45, ha='right')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# 2b: 每小时放电频次
ax = axes[1]
for arch in sorted(discharge_segs['vehicle_cluster'].unique()):
    arch_hourly_count = (
        hourly_discharge[hourly_discharge['vehicle_cluster'] == arch]
        .set_index('start_hour')['count']
        .reindex(range(24), fill_value=0)
    )
    # 归一化为比例
    total_count = arch_hourly_count.sum()
    arch_hourly_ratio = arch_hourly_count / max(total_count, 1)
    ax.plot(
        range(24), arch_hourly_ratio.values * 100,
        color=ARCHETYPE_COLORS[arch],
        linewidth=2.5,
        marker='s', markersize=4,
        label=VEHICLE_NAMES_SHORT[arch],
    )

ax.set_xlabel('Hour of Day', fontweight='bold')
ax.set_ylabel('Proportion of Discharge Events (%)', fontweight='bold')
ax.set_title('(b) Discharge Event Frequency by Hour\n(% of daily discharge events)',
             fontweight='bold', fontsize=12)
ax.set_xticks(range(0, 24, 2))
ax.set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 2)], rotation=45, ha='right')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.suptitle('Figure 2: 24-Hour Discharge Time Distribution by Vehicle Archetype',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()

fig2_path = os.path.join(OUTPUT_DIR, 'fig2_hourly_discharge_distribution.png')
plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✅ Saved: fig2_hourly_discharge_distribution.png")

# ---- 图3: 驾驶模式放电贡献度 ----
print(f"\n   Generating Figure 3: Driving Mode Discharge Contribution...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

if mode_pivot is not None:
    # 3a: 热力图
    ax = axes[0]
    mode_data = mode_pivot.copy()
    mode_data.index = [VEHICLE_NAMES_SHORT[i] for i in mode_data.index]

    sns.heatmap(
        mode_data * 100,
        ax=ax,
        annot=True,
        fmt='.1f',
        cmap='YlOrRd',
        linewidths=0.5,
        cbar_kws={'label': 'Discharge Contribution (%)'},
        annot_kws={'fontsize': 12, 'fontweight': 'bold'},
    )
    ax.set_xlabel('Driving Mode', fontweight='bold')
    ax.set_ylabel('Vehicle Archetype', fontweight='bold')
    ax.set_title('(a) Discharge Contribution by Driving Mode\n(% of total discharge per archetype)',
                 fontweight='bold', fontsize=12)

    # 3b: 堆积柱状图
    ax = axes[1]
    n_archetypes = len(mode_data)
    x = np.arange(n_archetypes)
    bar_width = 0.6

    bottom = np.zeros(n_archetypes)
    mode_cols = list(mode_data.columns)
    for mi, mode_name in enumerate(mode_cols):
        vals = mode_data[mode_name].values * 100
        bars = ax.bar(
            x, vals, bar_width,
            bottom=bottom,
            color=CLUSTER_COLORS[mi % len(CLUSTER_COLORS)],
            label=mode_name,
            edgecolor='white',
            linewidth=0.8,
        )
        # 标注比例
        for bi, (b, v) in enumerate(zip(bottom, vals)):
            if v > 3:  # 只标注较大的部分
                ax.text(
                    x[bi], b + v / 2,
                    f'{v:.1f}%',
                    ha='center', va='center',
                    fontsize=9, fontweight='bold', color='white'
                )
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(list(mode_data.index), fontweight='bold')
    ax.set_xlabel('Vehicle Archetype', fontweight='bold')
    ax.set_ylabel('Discharge Contribution (%)', fontweight='bold')
    ax.set_title('(b) Stacked Discharge Contribution\nby Driving Mode',
                 fontweight='bold', fontsize=12)
    ax.legend(title='Driving Mode', loc='upper right', fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3, axis='y')

else:
    for ax in axes:
        ax.text(0.5, 0.5, 'cluster_id column not available',
                ha='center', va='center', transform=ax.transAxes, fontsize=14)

plt.suptitle('Figure 3: Driving Mode Discharge Contribution by Vehicle Archetype',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()

fig3_path = os.path.join(OUTPUT_DIR, 'fig3_driving_mode_discharge_contribution.png')
plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✅ Saved: fig3_driving_mode_discharge_contribution.png")

# ---- 图4: 因果充电分析散点图 ----
print(f"\n   Generating Figure 4: Causal Charging Analysis...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

if do_causality and len(causality_df) > 0:
    valid = causality_df.dropna(subset=['preceding_discharge', 'soc_gain'])
    valid = valid[(valid['soc_gain'] > 0) & (valid['preceding_discharge'] > 0)]

    # 4a: 散点图 (preceding discharge vs soc_gain)
    ax = axes[0]
    for arch in sorted(valid['archetype'].unique()):
        arch_data = valid[valid['archetype'] == arch]
        # 随机采样避免过度绘制
        sample = arch_data.sample(min(len(arch_data), SCATTER_SAMPLE_LARGE), random_state=42)
        ax.scatter(
            sample['preceding_discharge'],
            sample['soc_gain'],
            color=ARCHETYPE_COLORS[arch],
            alpha=0.4, s=15,
            label=f"{VEHICLE_NAMES_SHORT[arch]} (n={len(arch_data):,})",
        )

        # 添加趋势线
        if len(arch_data) > 10:
            x_vals = arch_data['preceding_discharge'].clip(0, 100)
            y_vals = arch_data['soc_gain'].clip(0, 100)
            try:
                z = np.polyfit(x_vals, y_vals, 1)
                p = np.poly1d(z)
                x_line = np.linspace(x_vals.quantile(0.05), x_vals.quantile(0.95), 50)
                ax.plot(x_line, p(x_line),
                        color=ARCHETYPE_COLORS[arch],
                        linewidth=2, linestyle='--', alpha=0.9)
            except Exception:
                pass

    ax.set_xlabel('Preceding Discharge (% SOC, 24h window)', fontweight='bold')
    ax.set_ylabel('SOC Gain per Charging Event (%)', fontweight='bold')
    ax.set_title('(a) Preceding Discharge vs Charging Amount\n(Scatter + Trend)',
                 fontweight='bold', fontsize=12)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(alpha=0.3)

    # 4b: 相关性对比柱状图
    ax = axes[1]
    correlations = []
    for arch in sorted(valid['archetype'].unique()):
        arch_data = valid[valid['archetype'] == arch]
        corr = arch_data['soc_gain'].corr(arch_data['preceding_discharge'])
        correlations.append(corr)

    arch_names = [VEHICLE_NAMES_SHORT[a] for a in sorted(valid['archetype'].unique())]
    colors_used = [ARCHETYPE_COLORS[a] for a in sorted(valid['archetype'].unique())]
    bars = ax.bar(arch_names, correlations, color=colors_used, edgecolor='black',
                  linewidth=1.5, alpha=0.8)
    for bar, corr in zip(bars, correlations):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            corr + 0.01 if corr >= 0 else corr - 0.03,
            f'{corr:.3f}',
            ha='center', va='bottom' if corr >= 0 else 'top',
            fontsize=11, fontweight='bold'
        )

    ax.axhline(0, color='black', linewidth=1)
    ax.set_xlabel('Vehicle Archetype', fontweight='bold')
    ax.set_ylabel('Pearson Correlation\n(soc_gain ~ preceding_discharge)', fontweight='bold')
    ax.set_title('(b) Discharge→Charging Correlation\nby Vehicle Archetype',
                 fontweight='bold', fontsize=12)
    ax.set_ylim(min(correlations) - 0.15, max(correlations) + 0.15)
    ax.grid(alpha=0.3, axis='y')

else:
    for ax in axes:
        ax.text(0.5, 0.5, 'Charging events data not available',
                ha='center', va='center', transform=ax.transAxes, fontsize=14)

plt.suptitle('Figure 4: Causality Charging Analysis — Preceding Discharge → Charging Demand',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()

fig4_path = os.path.join(OUTPUT_DIR, 'fig4_causality_charging_analysis.png')
plt.savefig(fig4_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✅ Saved: fig4_causality_charging_analysis.png")

# ---- 图5: 每日放电累积时间序列 ----
print(f"\n   Generating Figure 5: Daily Cumulative Discharge...")

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# 5a: 每天全体放电量折线图（按车型）
ax = axes[0]
daily_all = (
    discharge_segs.groupby(['vehicle_cluster', 'date'])['soc_drop']
    .sum()
    .reset_index()
)
daily_all.columns = ['vehicle_cluster', 'date', 'daily_discharge']
daily_all['date'] = pd.to_datetime(daily_all['date'])

for arch in sorted(daily_all['vehicle_cluster'].unique()):
    arch_daily = daily_all[daily_all['vehicle_cluster'] == arch].sort_values('date')
    ax.plot(
        arch_daily['date'],
        arch_daily['daily_discharge'],
        color=ARCHETYPE_COLORS[arch],
        alpha=0.7,
        linewidth=1.5,
        label=VEHICLE_NAMES_SHORT[arch],
    )

ax.set_xlabel('Date', fontweight='bold')
ax.set_ylabel('Total SOC Drop (%, sum all vehicles)', fontweight='bold')
ax.set_title('(a) Daily Total Discharge by Archetype\n(all vehicles combined)',
             fontweight='bold', fontsize=12)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')

# 5b: 每辆车的日均放电箱线图（按车型）
ax = axes[1]
if 'vehicle_cluster' in daily_discharge_per_vehicle.columns:
    daily_discharge_per_vehicle['vehicle_cluster'] = (
        daily_discharge_per_vehicle['vehicle_cluster']
        .dropna()
        .astype(int)
    )
    valid_daily = daily_discharge_per_vehicle.dropna(subset=['vehicle_cluster'])

    data_for_box = [
        valid_daily[valid_daily['vehicle_cluster'] == arch]['daily_soc_drop'].values
        for arch in sorted(valid_daily['vehicle_cluster'].unique())
    ]
    arch_labels_box = [
        VEHICLE_NAMES_SHORT[a]
        for a in sorted(valid_daily['vehicle_cluster'].unique())
    ]

    bp = ax.boxplot(
        data_for_box,
        labels=arch_labels_box,
        patch_artist=True,
        medianprops=dict(color='black', linewidth=2),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker='o', markersize=2, alpha=0.3),
    )
    for patch, color in zip(bp['boxes'], ARCHETYPE_COLORS[:len(arch_labels_box)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel('Vehicle Archetype', fontweight='bold')
    ax.set_ylabel('Daily SOC Drop per Vehicle (%)', fontweight='bold')
    ax.set_title('(b) Daily Discharge per Vehicle\n(boxplot by archetype)',
                 fontweight='bold', fontsize=12)
    ax.grid(alpha=0.3, axis='y')

    for i, arch in enumerate(sorted(valid_daily['vehicle_cluster'].unique())):
        arch_data = valid_daily[valid_daily['vehicle_cluster'] == arch]['daily_soc_drop']
        ax.text(
            i + 1, arch_data.quantile(0.75) + 0.5,
            f'μ={arch_data.mean():.1f}%',
            ha='center', fontsize=9, color=ARCHETYPE_COLORS[i]
        )

plt.suptitle('Figure 5: Daily Discharge Accumulation by Vehicle Archetype',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()

fig5_path = os.path.join(OUTPUT_DIR, 'fig5_daily_discharge_cumulative.png')
plt.savefig(fig5_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✅ Saved: fig5_daily_discharge_cumulative.png")

# ---- 图6: 放电-充电延迟分析 ----
print(f"\n   Generating Figure 6: Discharge-Charging Delay Analysis...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

if do_causality and len(causality_df) > 0:
    valid_delay = causality_df.dropna(subset=['delay_hours'])
    valid_delay = valid_delay[valid_delay['delay_hours'] >= 0]

    # 6a: 延迟时间分布（箱线图）
    ax = axes[0]
    delay_data = [
        valid_delay[valid_delay['archetype'] == arch]['delay_hours'].clip(0, 48).values
        for arch in sorted(valid_delay['archetype'].unique())
    ]
    delay_labels = [
        VEHICLE_NAMES_SHORT[a]
        for a in sorted(valid_delay['archetype'].unique())
    ]

    bp = ax.boxplot(
        delay_data,
        labels=delay_labels,
        patch_artist=True,
        medianprops=dict(color='black', linewidth=2),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker='o', markersize=2, alpha=0.3),
    )
    for patch, color in zip(bp['boxes'],
                             ARCHETYPE_COLORS[:len(delay_labels)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for i, arch in enumerate(sorted(valid_delay['archetype'].unique())):
        arch_data = valid_delay[valid_delay['archetype'] == arch]['delay_hours'].clip(0, 48)
        ax.text(
            i + 1, arch_data.quantile(0.75) + 0.5,
            f'μ={arch_data.mean():.1f}h',
            ha='center', fontsize=9, color=ARCHETYPE_COLORS[i]
        )

    ax.set_xlabel('Vehicle Archetype', fontweight='bold')
    ax.set_ylabel('Delay: Last Discharge End → Charge Start (hours)', fontweight='bold')
    ax.set_title('(a) Discharge-to-Charge Delay\nby Vehicle Archetype',
                 fontweight='bold', fontsize=12)
    ax.grid(alpha=0.3, axis='y')

    # 6b: 充电触发SOC vs 延迟时间（散点图）
    ax = axes[1]
    valid_soc = valid_delay.dropna(subset=['soc_at_charge_start'])
    for arch in sorted(valid_soc['archetype'].unique()):
        arch_data = valid_soc[valid_soc['archetype'] == arch]
        sample = arch_data.sample(min(len(arch_data), SCATTER_SAMPLE_SMALL), random_state=42)
        ax.scatter(
            sample['delay_hours'].clip(0, 48),
            sample['soc_at_charge_start'].clip(0, 100),
            color=ARCHETYPE_COLORS[arch],
            alpha=0.4, s=15,
            label=f"{VEHICLE_NAMES_SHORT[arch]} (n={len(arch_data):,})",
        )

    ax.set_xlabel('Delay: Last Discharge → Charge Start (hours)', fontweight='bold')
    ax.set_ylabel('SOC at Charge Start (%)', fontweight='bold')
    ax.set_title('(b) Delay vs Trigger SOC\n(When does low SOC drive charging?)',
                 fontweight='bold', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

else:
    for ax in axes:
        ax.text(0.5, 0.5, 'Charging events data not available',
                ha='center', va='center', transform=ax.transAxes, fontsize=14)

plt.suptitle('Figure 6: Discharge-to-Charging Delay Analysis by Vehicle Archetype',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()

fig6_path = os.path.join(OUTPUT_DIR, 'fig6_discharge_charging_delay.png')
plt.savefig(fig6_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✅ Saved: fig6_discharge_charging_delay.png")

# ============================================================
# 6. 保存输出文件
# ============================================================
print(f"\n{'='*80}")
print("【STEP 6】Saving Output Files")
print("=" * 80)

# 6.1 放电规律数据表
out1_path = os.path.join(OUTPUT_DIR, 'discharge_patterns_by_archetype.csv')
discharge_patterns_df.to_csv(out1_path, index=False)
print(f"\n   ✅ Saved: discharge_patterns_by_archetype.csv")
print(f"      Rows: {len(discharge_patterns_df):,}  Columns: {len(discharge_patterns_df.columns):,}")

# 6.2 因果关系分析表
if len(causality_df) > 0:
    out2_path = os.path.join(OUTPUT_DIR, 'causality_charging_analysis.csv')
    causality_df.to_csv(out2_path, index=False)
    print(f"\n   ✅ Saved: causality_charging_analysis.csv")
    print(f"      Rows: {len(causality_df):,}  Columns: {len(causality_df.columns):,}")
else:
    print(f"\n   ⚠️  Skipped: causality_charging_analysis.csv (no data)")

# ============================================================
# 7. 详细报告
# ============================================================
print(f"\n{'='*80}")
print("【STEP 7】Detailed Report")
print("=" * 80)

report_lines = []
report_lines.append("=" * 80)
report_lines.append("DISCHARGE PATTERN ANALYSIS + CAUSALITY CHARGING ANALYSIS")
report_lines.append("Step 15 Summary Report")
report_lines.append("=" * 80)
report_lines.append("")
report_lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
report_lines.append("PART 1: DISCHARGE PATTERN FINDINGS")
report_lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
report_lines.append("")

for _, row in discharge_patterns_df.iterrows():
    arch = int(row['archetype'])
    report_lines.append(f"  Archetype {arch} ({VEHICLE_NAMES_SHORT[arch]}):")
    report_lines.append(f"    - Avg SOC drop/segment: {row['avg_soc_drop']:.2f}% "
                        f"(±{row['std_soc_drop']:.2f}%)")
    report_lines.append(f"    - Discharge rate: {row['avg_discharge_rate_pct_per_min']:.4f} %/min")
    report_lines.append(f"    - Avg daily discharge: {row['avg_daily_discharge_per_vehicle']:.2f}% "
                        f"(±{row['std_daily_discharge_per_vehicle']:.2f}%) per vehicle")
    report_lines.append(f"    - Vehicles analyzed: {int(row['n_vehicles'])}")
    report_lines.append("")

if mode_pivot is not None:
    report_lines.append("  Driving Mode Contribution to Discharge:")
    header = f"  {'Archetype':>10}  " + "  ".join([f"{m:>8}" for m in mode_pivot.columns])
    report_lines.append(header)
    for arch_idx in mode_pivot.index:
        row_str = f"  {VEHICLE_NAMES_SHORT[arch_idx]:>10}  " + \
                  "  ".join([f"{v*100:>7.1f}%" for v in mode_pivot.loc[arch_idx].values])
        report_lines.append(row_str)
    report_lines.append("")

if do_causality and len(causality_df) > 0:
    report_lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    report_lines.append("PART 2: CAUSALITY CHARGING FINDINGS")
    report_lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    report_lines.append("")

    for arch in sorted(causality_df['archetype'].unique()):
        arch_df = causality_df[causality_df['archetype'] == arch]
        valid = arch_df.dropna(subset=['preceding_discharge', 'soc_gain'])
        valid = valid[(valid['soc_gain'] > 0) & (valid['preceding_discharge'] > 0)]

        corr_val = valid['soc_gain'].corr(valid['preceding_discharge']) if len(valid) > 5 else np.nan
        avg_delay = arch_df['delay_hours'].mean()

        report_lines.append(f"  Archetype {arch} ({VEHICLE_NAMES_SHORT[arch]}):")
        report_lines.append(f"    - Avg preceding discharge (24h): "
                            f"{arch_df['preceding_discharge'].mean():.2f}%")
        report_lines.append(f"    - Avg charge gain: {arch_df['soc_gain'].mean():.2f}%")
        corr_str = f"{corr_val:.3f}" if pd.notna(corr_val) else "N/A (insufficient data)"
        report_lines.append(f"    - Discharge→charging correlation: {corr_str}")
        report_lines.append(f"    - Avg delay discharge→charging: {avg_delay:.2f} h")
        report_lines.append("")

report_lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
report_lines.append("INFRASTRUCTURE PLANNING IMPLICATIONS")
report_lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
report_lines.append("")
report_lines.append("  1. Vehicles with high daily discharge → need more frequent")
report_lines.append("     charging opportunities (higher charger density in their areas)")
report_lines.append("  2. Highway-dominant discharge → chargers needed at highway nodes")
report_lines.append("  3. Urban-dominant discharge → destination charging suitable")
report_lines.append("  4. High delay between discharge and charging → flexible scheduling")
report_lines.append("     is acceptable (less time-critical)")
report_lines.append("  5. Low delay → fast-charging or emergency-charging infrastructure")
report_lines.append("     priority for that archetype")
report_lines.append("")
report_lines.append("=" * 80)
report_lines.append("Files generated:")
report_lines.append(f"  discharge_patterns_by_archetype.csv")
if len(causality_df) > 0:
    report_lines.append(f"  causality_charging_analysis.csv")
report_lines.append(f"  fig1_discharge_distribution.png")
report_lines.append(f"  fig2_hourly_discharge_distribution.png")
report_lines.append(f"  fig3_driving_mode_discharge_contribution.png")
report_lines.append(f"  fig4_causality_charging_analysis.png")
report_lines.append(f"  fig5_daily_discharge_cumulative.png")
report_lines.append(f"  fig6_discharge_charging_delay.png")
report_lines.append("=" * 80)

report_text = "\n".join(report_lines)
print("\n" + report_text)

report_path = os.path.join(OUTPUT_DIR, 'step15_discharge_causality_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_text)
print(f"\n   ✅ Saved: step15_discharge_causality_report.txt")

print(f"\n{'='*80}")
print("✅ Step 15 Complete!")
print("=" * 80)
