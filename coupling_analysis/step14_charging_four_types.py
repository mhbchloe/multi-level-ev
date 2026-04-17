"""
Step 14: Four-Type Vehicle Charging Pattern Analysis
=====================================================
分析四类车辆的充电行为模式对比

数据加载优先级:
  1. coupling_analysis/results/charging_events_meaningful.csv   (filter_charging_events.py 输出)
  2. coupling_analysis/results/charging_events_raw_extracted.csv (extract_charging_from_raw.py 输出)
  3. coupling_analysis/results/charging_events_stationary_meaningful.csv
  4. coupling_analysis/results/segments_integrated_complete.csv  (回退: 从 segments 检测)

车辆类型标签来源 (按优先级):
  1. vehicle_clustering/results/vehicle_clustering_gmm_k4.csv
  2. vehicle_clustering/results/vehicle_clustering_improved_3d.csv
  3. vehicle_clustering/results/vehicle_clustering_optimal.csv
  4. coupling_analysis/results/segments_with_cluster_labels.csv (driving_pattern_name)

输出:
  - step14_fig1_soc_gain_by_type.png        图1: SOC增益分布
  - step14_fig2_trigger_soc_by_type.png     图2: 充电触发SOC
  - step14_fig3_duration_speed_by_type.png  图3: 充电时长与快/慢充比例
  - step14_fig4_frequency_timing_by_type.png图4: 充电频率与时段分布
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
import json
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['font.size'] = 11

# ============================================================
# Config
# ============================================================
RESULTS_DIR   = "./coupling_analysis/results/"
CLUSTER_DIR   = "./vehicle_clustering/results/"
FIGURE_DIR    = "./coupling_analysis/results/"

os.makedirs(FIGURE_DIR, exist_ok=True)

# 充电事件检测参数（与 extract_charging_from_raw.py 一致）
IS_CHARGING_CODES    = [1, 2]
MIN_SOC_GAIN         = 0.5     # %
MIN_DURATION_SEC     = 60      # 秒
MIN_RECORDS          = 3       # 最少记录数
TIME_GAP_THRESHOLD   = 600     # 10 分钟→新事件

print("=" * 70)
print("🚗 Step 14: Four-Type Vehicle Charging Pattern Analysis")
print("=" * 70)

# ============================================================
# 1. 加载充电事件数据
# ============================================================
print("\n【STEP 1】Loading Charging Events Data")
print("=" * 70)

CHARGING_CANDIDATES = [
    os.path.join(RESULTS_DIR, 'charging_events_meaningful.csv'),
    os.path.join(RESULTS_DIR, 'charging_events_raw_extracted.csv'),
    os.path.join(RESULTS_DIR, 'charging_events_stationary_meaningful.csv'),
    os.path.join(RESULTS_DIR, 'charging_events_clean.csv'),
]

df_charging = None
charging_source = None

for path in CHARGING_CANDIDATES:
    if os.path.exists(path):
        print(f"   ✅ Found: {os.path.basename(path)}")
        df_charging = pd.read_csv(path)
        charging_source = path
        break

if df_charging is None:
    # 回退: 从 segments_integrated_complete.csv 检测充电事件
    segments_path = os.path.join(RESULTS_DIR, 'segments_integrated_complete.csv')
    if not os.path.exists(segments_path):
        print("   ❌ No charging data found. Checked:")
        for p in CHARGING_CANDIDATES + [segments_path]:
            print(f"      {p}")
        raise FileNotFoundError(
            "No charging events data available. "
            "Please run extract_charging_from_raw.py or filter_charging_events.py first."
        )

    print(f"   ⚠️  No pre-processed charging file found.")
    print(f"   🔄 Falling back to segments-based detection: {segments_path}")
    df_segs = pd.read_csv(segments_path)
    time_col = 'time' if 'time' in df_segs.columns else 'datetime'
    df_segs['datetime'] = pd.to_datetime(df_segs[time_col], errors='coerce')

    # 筛选充电行
    ch_col = 'ch_s' if 'ch_s' in df_segs.columns else None
    if ch_col is None:
        raise ValueError("segments file has no 'ch_s' column for charging detection")

    df_segs = df_segs.dropna(subset=[ch_col, 'datetime'])
    df_segs[ch_col] = df_segs[ch_col].astype(int)
    df_segs = df_segs.sort_values(['vehicle_id', 'datetime']).reset_index(drop=True)

    df_segs['is_ch'] = df_segs[ch_col].isin(IS_CHARGING_CODES).astype(int)

    # 检测事件边界（换车 或 时间间隔 > 10 分钟）
    df_segs['vid_prev'] = df_segs['vehicle_id'].shift(1)
    df_segs['dt_prev']  = df_segs['datetime'].shift(1)
    dt_diff_sec = (df_segs['datetime'] - df_segs['dt_prev']).dt.total_seconds()
    vid_change  = df_segs['vehicle_id'] != df_segs['vid_prev']
    new_event   = (vid_change | (dt_diff_sec > TIME_GAP_THRESHOLD)) & (df_segs['is_ch'] == 1)
    df_segs['event_id_raw'] = new_event.cumsum()

    df_ch = df_segs[df_segs['is_ch'] == 1].copy()

    soc_col   = 'soc'
    pow_col   = 'power' if 'power' in df_ch.columns else None
    spd_col   = 'spd'   if 'spd'   in df_ch.columns else None
    v_col     = 'v'     if 'v'     in df_ch.columns else None
    i_col     = 'i'     if 'i'     in df_ch.columns else None

    agg_dict = {
        'vehicle_id': 'first',
        soc_col: ['first', 'last', 'min', 'max'],
        'datetime': ['first', 'last', 'count'],
    }
    if pow_col: agg_dict[pow_col] = 'mean'
    if spd_col: agg_dict[spd_col] = 'mean'
    if v_col:   agg_dict[v_col]   = 'mean'
    if i_col:   agg_dict[i_col]   = 'mean'

    events = df_ch.groupby('event_id_raw').agg(agg_dict)

    # 展平多级列名
    new_cols = ['vehicle_id', 'soc_start', 'soc_end', 'soc_min', 'soc_max',
                'start_time', 'end_time', 'num_records']
    extra = []
    if pow_col: extra.append('power_mean')
    if spd_col: extra.append('speed_mean')
    if v_col:   extra.append('voltage_mean')
    if i_col:   extra.append('current_mean')
    events.columns = new_cols + extra
    events = events.reset_index(drop=True)

    events['soc_gain'] = events['soc_end'] - events['soc_start']
    events['duration_seconds'] = (
        pd.to_datetime(events['end_time']) - pd.to_datetime(events['start_time'])
    ).dt.total_seconds()
    events['duration_minutes'] = events['duration_seconds'] / 60.0

    # 过滤 (与 extract_charging_from_raw.py 一致)
    mask = (
        (events['soc_gain'] >= MIN_SOC_GAIN) &
        (events['duration_seconds'] >= MIN_DURATION_SEC) &
        (events['num_records'] >= MIN_RECORDS) &
        (events['soc_start'] >= 0) &
        (events['soc_end'] <= 100)
    )
    events = events[mask].reset_index(drop=True)

    # avg_soc_rate & charge_type
    events['avg_soc_rate'] = events['soc_gain'] / events['duration_minutes'].replace(0, np.nan)
    events['charge_type']  = np.where(events['avg_soc_rate'] > 1.0, 'fast', 'slow')
    # Generate per-vehicle sequential event IDs
    vid_counter: dict = {}
    event_ids = []
    for vid in events['vehicle_id']:
        vid_counter[vid] = vid_counter.get(vid, 0)
        event_ids.append(f"{vid}_ch_{vid_counter[vid]:05d}")
        vid_counter[vid] += 1
    events['charging_event_id'] = event_ids

    df_charging = events
    charging_source = segments_path
    print(f"   ✅ Detected {len(df_charging):,} charging events from segments")

# 标准化时间列
df_charging['start_time'] = pd.to_datetime(df_charging['start_time'], errors='coerce')
df_charging['end_time']   = pd.to_datetime(df_charging['end_time'],   errors='coerce')

# 补全派生字段
if 'duration_seconds' not in df_charging.columns:
    df_charging['duration_seconds'] = (
        df_charging['end_time'] - df_charging['start_time']
    ).dt.total_seconds()
if 'duration_minutes' not in df_charging.columns:
    df_charging['duration_minutes'] = df_charging['duration_seconds'] / 60.0
if 'avg_soc_rate' not in df_charging.columns:
    df_charging['avg_soc_rate'] = (
        df_charging['soc_gain'] / df_charging['duration_minutes'].replace(0, np.nan)
    )
if 'charge_type' not in df_charging.columns:
    df_charging['charge_type'] = np.where(df_charging['avg_soc_rate'] > 1.0, 'fast', 'slow')

df_charging['start_hour']    = df_charging['start_time'].dt.hour
df_charging['start_weekday'] = df_charging['start_time'].dt.dayofweek
df_charging['is_night']      = df_charging['start_hour'].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)

print(f"\n   📊 Charging events loaded: {len(df_charging):,}")
print(f"   📊 Unique vehicles:        {df_charging['vehicle_id'].nunique():,}")
print(f"   📊 Source: {os.path.basename(charging_source)}")

# ============================================================
# 2. 加载车辆类型标签
# ============================================================
print("\n【STEP 2】Loading Vehicle Type Labels")
print("=" * 70)

CLUSTER_CANDIDATES = [
    (os.path.join(CLUSTER_DIR, 'vehicle_clustering_gmm_k4.csv'),
     'vehicle_id', 'cluster_label'),
    (os.path.join(CLUSTER_DIR, 'vehicle_clustering_improved_3d.csv'),
     'vehicle_id', 'cluster_label'),
    (os.path.join(CLUSTER_DIR, 'vehicle_clustering_optimal.csv'),
     'vehicle_id', 'cluster_label'),
    (os.path.join(RESULTS_DIR,  'segments_with_cluster_labels.csv'),
     'vehicle_id', 'driving_pattern_name'),
    (os.path.join(RESULTS_DIR,  'coupling_analysis_dataset.csv'),
     'vehicle_id', 'driving_pattern_name'),
]

df_labels = None
label_col = None

for path, vid_col, lbl_col in CLUSTER_CANDIDATES:
    if os.path.exists(path):
        tmp = pd.read_csv(path)
        if vid_col in tmp.columns and lbl_col in tmp.columns:
            df_labels = (tmp[[vid_col, lbl_col]]
                         .rename(columns={vid_col: 'vehicle_id', lbl_col: 'vehicle_type'})
                         .drop_duplicates(subset=['vehicle_id']))
            label_col = lbl_col
            print(f"   ✅ Found: {os.path.basename(path)}  (label col: '{lbl_col}')")
            break

if df_labels is None:
    print("   ⚠️  No vehicle type labels found. Using 'All Vehicles' as single group.")
    df_labels = pd.DataFrame({
        'vehicle_id': df_charging['vehicle_id'].unique(),
        'vehicle_type': 'All Vehicles',
    })

n_types = df_labels['vehicle_type'].nunique()
print(f"   📊 Vehicle types: {n_types}")
for vt in sorted(df_labels['vehicle_type'].unique()):
    n = (df_labels['vehicle_type'] == vt).sum()
    print(f"      {vt}: {n:,} vehicles")

# ============================================================
# 3. 合并数据
# ============================================================
print("\n【STEP 3】Merging Charging Events with Vehicle Labels")
print("=" * 70)

df_merged = df_charging.merge(df_labels, on='vehicle_id', how='left')
df_merged['vehicle_type'] = df_merged['vehicle_type'].fillna('Unknown')

# 去掉 Unknown（无法归类）
n_before = len(df_merged)
df_merged = df_merged[df_merged['vehicle_type'] != 'Unknown'].copy()
n_after = len(df_merged)

print(f"   Events after merge:  {n_after:,} / {n_before:,}")
print(f"   Vehicle types found: {sorted(df_merged['vehicle_type'].unique())}")

if len(df_merged) == 0:
    raise ValueError(
        "No charging events matched vehicle type labels. "
        "Check that vehicle_id formats are consistent."
    )

# 确定绘图顺序（按 soc_start 均值从高到低）
type_order = (
    df_merged.groupby('vehicle_type')['soc_start']
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)

n_types = len(type_order)
PALETTE = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C'][:n_types]
color_map = {t: PALETTE[i] for i, t in enumerate(type_order)}

print(f"\n   Type order (by avg soc_start, high→low): {type_order}")

# ============================================================
# 4. 基础统计
# ============================================================
print("\n【STEP 4】Descriptive Statistics by Vehicle Type")
print("=" * 70)

stats_rows = []
for vt in type_order:
    sub = df_merged[df_merged['vehicle_type'] == vt]
    n_events = len(sub)
    n_veh    = sub['vehicle_id'].nunique()
    ev_per_v = n_events / max(n_veh, 1)

    fast_pct = (sub['charge_type'] == 'fast').mean() * 100 if 'charge_type' in sub.columns else np.nan
    night_pct = sub['is_night'].mean() * 100

    stats_rows.append({
        'Vehicle Type':     vt,
        'Events':           n_events,
        'Vehicles':         n_veh,
        'Events/Vehicle':   round(ev_per_v, 1),
        'Avg SOC_start (%)':round(sub['soc_start'].mean(), 1),
        'Avg SOC_gain (%)': round(sub['soc_gain'].mean(), 1),
        'Avg Duration (h)': round(sub['duration_minutes'].mean() / 60, 2),
        'Fast Charge (%)':  round(fast_pct, 1),
        'Night Charge (%)': round(night_pct, 1),
    })

df_stats = pd.DataFrame(stats_rows)
print("\n" + df_stats.to_string(index=False))

# 保存统计表
stats_path = os.path.join(RESULTS_DIR, 'step14_charging_stats_by_type.csv')
df_stats.to_csv(stats_path, index=False)
print(f"\n   💾 Saved: {os.path.basename(stats_path)}")

# ============================================================
# 5. 图 1: SOC 增益分布
# ============================================================
print("\n【STEP 5】Figure 1 — SOC Gain Distribution by Vehicle Type")
print("=" * 70)

fig1, axes = plt.subplots(1, 2, figsize=(16, 7))

# ── (a) 箱线图 ──
ax = axes[0]
data_gain = [df_merged.loc[df_merged['vehicle_type'] == vt, 'soc_gain'].values
             for vt in type_order]
bp = ax.boxplot(data_gain, patch_artist=True, showfliers=False, widths=0.55,
                medianprops=dict(color='black', linewidth=2.0))
for patch, color in zip(bp['boxes'], PALETTE[:n_types]):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

ax.set_xticks(range(1, n_types + 1))
ax.set_xticklabels(type_order, rotation=20, ha='right', fontsize=10)
ax.set_ylabel('SOC Gain per Event (%)', fontweight='bold')
ax.set_title('(a) SOC Gain Distribution by Vehicle Type', fontweight='bold', fontsize=12)
ax.grid(alpha=0.3, axis='y')

# 在箱线图上标注中位数
gain_y_offset = (df_merged['soc_gain'].quantile(0.75) - df_merged['soc_gain'].quantile(0.25)) * 0.04
for i, vt in enumerate(type_order):
    med = np.median(data_gain[i])
    ax.text(i + 1, med + gain_y_offset, f'{med:.1f}%', ha='center', va='bottom',
            fontsize=9, fontweight='bold', color='#333333')

# ── (b) 密度图 ──
ax = axes[1]
for vt in type_order:
    sub = df_merged.loc[df_merged['vehicle_type'] == vt, 'soc_gain']
    sub = sub[(sub > 0) & (sub <= 100)]
    if len(sub) > 30:
        sub.plot.kde(ax=ax, label=f'{vt} (med={sub.median():.1f}%)',
                     color=color_map[vt], linewidth=2.2)
ax.set_xlabel('SOC Gain (%)', fontweight='bold')
ax.set_ylabel('Density', fontweight='bold')
ax.set_xlim(0, 80)
ax.set_title('(b) SOC Gain Density by Vehicle Type', fontweight='bold', fontsize=12)
ax.legend(fontsize=9, loc='upper right')
ax.grid(alpha=0.3)

plt.suptitle('Figure 1: Charging Amount — SOC Gain by Vehicle Type',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
fig1_path = os.path.join(FIGURE_DIR, 'step14_fig1_soc_gain_by_type.png')
fig1.savefig(fig1_path, dpi=300, bbox_inches='tight')
plt.close(fig1)
print(f"   ✅ Saved: step14_fig1_soc_gain_by_type.png")

# ============================================================
# 6. 图 2: 充电触发 SOC (soc_start)
# ============================================================
print("\n【STEP 6】Figure 2 — Charging Trigger SOC by Vehicle Type")
print("=" * 70)

fig2, axes = plt.subplots(1, 2, figsize=(16, 7))

# ── (a) 箱线图 ──
ax = axes[0]
data_soc = [df_merged.loc[df_merged['vehicle_type'] == vt, 'soc_start'].values
            for vt in type_order]
bp = ax.boxplot(data_soc, patch_artist=True, showfliers=False, widths=0.55,
                medianprops=dict(color='black', linewidth=2.0))
for patch, color in zip(bp['boxes'], PALETTE[:n_types]):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

ax.set_xticks(range(1, n_types + 1))
ax.set_xticklabels(type_order, rotation=20, ha='right', fontsize=10)
ax.set_ylabel('SOC at Charging Start (%)', fontweight='bold')
ax.set_title('(a) Charge Trigger SOC by Vehicle Type\n(Lower = More risk-tolerant)',
             fontweight='bold', fontsize=12)
ax.grid(alpha=0.3, axis='y')

soc_y_offset = (df_merged['soc_start'].quantile(0.75) - df_merged['soc_start'].quantile(0.25)) * 0.04
for i, vt in enumerate(type_order):
    med = np.median(data_soc[i])
    ax.text(i + 1, med + soc_y_offset, f'{med:.1f}%', ha='center', va='bottom',
            fontsize=9, fontweight='bold', color='#333333')

# ── (b) 均值 ± std 条形图 ──
ax = axes[1]
means  = [np.mean(d) for d in data_soc]
stds   = [np.std(d)  for d in data_soc]
x_pos  = range(n_types)
bars = ax.bar(x_pos, means, yerr=stds, capsize=6,
              color=PALETTE[:n_types], alpha=0.8, edgecolor='black', linewidth=1.2)
ax.set_xticks(x_pos)
ax.set_xticklabels(type_order, rotation=20, ha='right', fontsize=10)
ax.set_ylabel('Mean Trigger SOC (%)', fontweight='bold')
ax.set_title('(b) Mean ± Std Trigger SOC by Vehicle Type', fontweight='bold', fontsize=12)
ax.grid(alpha=0.3, axis='y')

for bar, m in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(stds) * 0.05,
            f'{m:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# ANOVA 检验
groups_soc = [np.array(d) for d in data_soc if len(d) > 1]
if len(groups_soc) >= 2:
    f_stat, p_val = stats.f_oneway(*groups_soc)
    sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'ns'))
    ax.set_title(
        f'(b) Mean ± Std Trigger SOC\nANOVA F={f_stat:.2f}, p={p_val:.2e} {sig}',
        fontweight='bold', fontsize=11)

plt.suptitle('Figure 2: Charging Trigger SOC by Vehicle Type',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
fig2_path = os.path.join(FIGURE_DIR, 'step14_fig2_trigger_soc_by_type.png')
fig2.savefig(fig2_path, dpi=300, bbox_inches='tight')
plt.close(fig2)
print(f"   ✅ Saved: step14_fig2_trigger_soc_by_type.png")

# ============================================================
# 7. 图 3: 充电时长与快/慢充比例
# ============================================================
print("\n【STEP 7】Figure 3 — Duration and Charge Speed by Vehicle Type")
print("=" * 70)

fig3, axes = plt.subplots(1, 2, figsize=(16, 7))

# ── (a) 充电时长箱线图 ──
ax = axes[0]
data_dur = [df_merged.loc[df_merged['vehicle_type'] == vt, 'duration_minutes'].values
            for vt in type_order]
bp = ax.boxplot(data_dur, patch_artist=True, showfliers=False, widths=0.55,
                medianprops=dict(color='black', linewidth=2.0))
for patch, color in zip(bp['boxes'], PALETTE[:n_types]):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

ax.set_xticks(range(1, n_types + 1))
ax.set_xticklabels(type_order, rotation=20, ha='right', fontsize=10)
ax.set_ylabel('Charging Duration (minutes)', fontweight='bold')
ax.set_title('(a) Charging Duration by Vehicle Type', fontweight='bold', fontsize=12)
ax.grid(alpha=0.3, axis='y')

max_dur_median = max(np.median(d) for d in data_dur)
for i, vt in enumerate(type_order):
    med = np.median(data_dur[i])
    ax.text(i + 1, med + max_dur_median * 0.04,
            f'{med:.0f} min', ha='center', va='bottom',
            fontsize=9, fontweight='bold', color='#333333')

# ── (b) 快/慢充比例分组条形图 ──
ax = axes[1]
x_pos   = np.arange(n_types)
bar_w   = 0.35
fast_rates = []
slow_rates = []

for vt in type_order:
    sub = df_merged[df_merged['vehicle_type'] == vt]
    if 'charge_type' in sub.columns and len(sub) > 0:
        fast_rates.append((sub['charge_type'] == 'fast').mean() * 100)
        slow_rates.append((sub['charge_type'] == 'slow').mean() * 100)
    else:
        fast_rates.append(0.0)
        slow_rates.append(100.0)

bars_f = ax.bar(x_pos - bar_w / 2, fast_rates, bar_w,
                label='Fast (>1%/min)', color='#E74C3C', alpha=0.85, edgecolor='black')
bars_s = ax.bar(x_pos + bar_w / 2, slow_rates, bar_w,
                label='Slow (≤1%/min)', color='#3498DB', alpha=0.85, edgecolor='black')

for bar, v in zip(bars_f, fast_rates):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 1, f'{v:.1f}%',
            ha='center', va='bottom', fontsize=9, fontweight='bold')
for bar, v in zip(bars_s, slow_rates):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 1, f'{v:.1f}%',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xticks(x_pos)
ax.set_xticklabels(type_order, rotation=20, ha='right', fontsize=10)
ax.set_ylabel('Proportion (%)', fontweight='bold')
ax.set_title('(b) Fast vs. Slow Charge Ratio by Vehicle Type', fontweight='bold', fontsize=12)
ax.legend(fontsize=10, loc='upper right')
ax.set_ylim(0, 115)
ax.grid(alpha=0.3, axis='y')

plt.suptitle('Figure 3: Charging Duration and Speed by Vehicle Type',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
fig3_path = os.path.join(FIGURE_DIR, 'step14_fig3_duration_speed_by_type.png')
fig3.savefig(fig3_path, dpi=300, bbox_inches='tight')
plt.close(fig3)
print(f"   ✅ Saved: step14_fig3_duration_speed_by_type.png")

# ============================================================
# 8. 图 4: 充电频率与时段分布
# ============================================================
print("\n【STEP 8】Figure 4 — Charging Frequency and Time-of-Day by Vehicle Type")
print("=" * 70)

fig4, axes = plt.subplots(1, 2, figsize=(16, 7))

# ── (a) 每辆车平均充电次数 ──
ax = axes[0]
events_per_vehicle = []
for vt in type_order:
    sub = df_merged[df_merged['vehicle_type'] == vt]
    epv = len(sub) / max(sub['vehicle_id'].nunique(), 1)
    events_per_vehicle.append(epv)

bars = ax.bar(range(n_types), events_per_vehicle,
              color=PALETTE[:n_types], alpha=0.85, edgecolor='black', linewidth=1.2)
ax.set_xticks(range(n_types))
ax.set_xticklabels(type_order, rotation=20, ha='right', fontsize=10)
ax.set_ylabel('Avg Charging Events per Vehicle', fontweight='bold')
ax.set_title('(a) Charging Frequency by Vehicle Type', fontweight='bold', fontsize=12)
ax.grid(alpha=0.3, axis='y')
for bar, v in zip(bars, events_per_vehicle):
    ax.text(bar.get_x() + bar.get_width() / 2,
            v + max(events_per_vehicle) * 0.02,
            f'{v:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# ── (b) 一天24小时充电时段热力图 ──
ax = axes[1]
hour_matrix = np.zeros((n_types, 24))
for i, vt in enumerate(type_order):
    sub = df_merged[df_merged['vehicle_type'] == vt]
    for h in range(24):
        hour_matrix[i, h] = (sub['start_hour'] == h).sum()
    row_sum = hour_matrix[i].sum()
    if row_sum > 0:
        hour_matrix[i] /= row_sum   # 归一化为比例

im = ax.imshow(hour_matrix, aspect='auto', cmap='YlOrRd', vmin=0)
ax.set_yticks(range(n_types))
ax.set_yticklabels(type_order, fontsize=10)
ax.set_xticks(range(0, 24, 3))
ax.set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 3)], rotation=45, fontsize=9)
ax.set_xlabel('Hour of Day', fontweight='bold')
ax.set_title('(b) Charging Time-of-Day Distribution\n(row-normalized)',
             fontweight='bold', fontsize=12)
plt.colorbar(im, ax=ax, label='Proportion', shrink=0.85)

# 标注峰值小时
for i in range(n_types):
    peak_h = np.argmax(hour_matrix[i])
    ax.text(peak_h, i, f'{peak_h:02d}h', ha='center', va='center',
            fontsize=8, color='navy', fontweight='bold')

plt.suptitle('Figure 4: Charging Frequency and Timing by Vehicle Type',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
fig4_path = os.path.join(FIGURE_DIR, 'step14_fig4_frequency_timing_by_type.png')
fig4.savefig(fig4_path, dpi=300, bbox_inches='tight')
plt.close(fig4)
print(f"   ✅ Saved: step14_fig4_frequency_timing_by_type.png")

# ============================================================
# 9. 统计检验报告
# ============================================================
print("\n【STEP 9】Statistical Tests")
print("=" * 70)

test_metrics = {
    'soc_gain':    'SOC Gain (%)',
    'soc_start':   'Trigger SOC (%)',
    'duration_minutes': 'Duration (min)',
}

report_lines = [
    "Step 14: Four-Type Vehicle Charging Pattern — Statistical Tests",
    "=" * 70,
    f"Charging data source: {os.path.basename(charging_source)}",
    f"Vehicle label source: label col '{label_col}'",
    f"Total events: {len(df_merged):,}",
    f"Vehicle types: {type_order}",
    "",
]

for col, label in test_metrics.items():
    if col not in df_merged.columns:
        continue
    groups_vals = [df_merged.loc[df_merged['vehicle_type'] == vt, col].dropna().values
                   for vt in type_order]
    groups_vals = [g for g in groups_vals if len(g) > 1]
    if len(groups_vals) < 2:
        continue

    f_stat, p_val = stats.f_oneway(*groups_vals)
    sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'ns'))
    line = f"ANOVA [{label}]: F={f_stat:.3f}, p={p_val:.3e} {sig}"
    print(f"   {line}")
    report_lines.append(line)

report_lines.append("")
report_lines.append("Pairwise Kruskal-Wallis (non-parametric):")
for col, label in test_metrics.items():
    if col not in df_merged.columns:
        continue
    groups_vals = [df_merged.loc[df_merged['vehicle_type'] == vt, col].dropna().values
                   for vt in type_order]
    groups_vals = [g for g in groups_vals if len(g) > 1]
    if len(groups_vals) < 2:
        continue
    h_stat, p_val = stats.kruskal(*groups_vals)
    sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'ns'))
    line = f"Kruskal [{label}]: H={h_stat:.3f}, p={p_val:.3e} {sig}"
    print(f"   {line}")
    report_lines.append(line)

report_path = os.path.join(RESULTS_DIR, 'step14_statistical_tests.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))
print(f"\n   💾 Saved: {os.path.basename(report_path)}")

# ============================================================
# 完成
# ============================================================
print("\n" + "=" * 70)
print("✅ Step 14 Complete!")
print("=" * 70)
print(f"""
Output files:
  📊 step14_fig1_soc_gain_by_type.png        — SOC gain distribution
  📊 step14_fig2_trigger_soc_by_type.png     — Charging trigger SOC
  📊 step14_fig3_duration_speed_by_type.png  — Duration & fast/slow ratio
  📊 step14_fig4_frequency_timing_by_type.png— Frequency & time-of-day
  📋 step14_charging_stats_by_type.csv       — Summary statistics
  📋 step14_statistical_tests.txt            — ANOVA / Kruskal-Wallis tests
""")
print(f"   Charging events: {len(df_merged):,} | Vehicle types: {n_types}")
for vt in type_order:
    n = (df_merged['vehicle_type'] == vt).sum()
    print(f"   {vt}: {n:,} events ({n/len(df_merged)*100:.1f}%)")
