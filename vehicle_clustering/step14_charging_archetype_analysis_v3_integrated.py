"""
Step 14 (v3 Integrated): Charging Archetype Analysis
充电行为焦虑程度分析与四类车型对比

分析流程:
1. 验证数据标记（charge_type / speed_label / charge_type_name）
2. 计算每辆车的完整焦虑指标（9项）
3. 与KMeans K=4车辆聚类标签合并
4. 四类车型焦虑程度横向对比
5. 生成8-10张可视化图表
6. 输出4张数据表

输入:
  - coupling_analysis/results/charging_events_stationary_only.csv
  - coupling_analysis/results/segments_integrated_complete.csv
  - vehicle_clustering/results/vehicle_clustering_kmeans_final_k4.csv
    (fallback: vehicle_clustering_gmm_k4.csv)

输出:
  - vehicle_clustering/results/discharge_segments_analysis.csv
  - vehicle_clustering/results/vehicle_anxiety_metrics.csv
  - vehicle_clustering/results/archetype_anxiety_comparison.csv
  - vehicle_clustering/results/causality_discharge_to_charging.csv
  - vehicle_clustering/results/figures_step14/*.png (8-10 figures)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

# ── 字体与风格 ─────────────────────────────────────────────────────────────
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 150

print("=" * 80)
print("🔋 Step 14 (v3): Charging Archetype Analysis – Anxiety Metrics & Causal Chain")
print("=" * 80)

# ── 路径配置 ───────────────────────────────────────────────────────────────
BASE_RESULTS = "./coupling_analysis/results/"
VC_RESULTS   = "./vehicle_clustering/results/"
FIG_DIR      = os.path.join(VC_RESULTS, "figures_step14")
os.makedirs(FIG_DIR, exist_ok=True)

# ── 归一化常数（用于焦虑指数计算） ────────────────────────────────────────
MAX_EXPECTED_DOD          = 80.0   # 最大预期放电深度 (%)
MAX_EXPECTED_CV           = 2.0    # 最大预期充电间隔变异系数
MAX_CHARGE_DURATION_MIN   = 300.0  # 散点图颜色轴上限（分钟）
MAX_CHARGES_PER_DAY       = 3.0    # 每日充电次数归一化上限
EPSILON                   = 1e-9   # 防止除以零的小量

# ============================================================
# 1. 加载数据
# ============================================================
print("\n【STEP 1】Loading Data")
print("=" * 80)

# 1.1 停车充电事件
charging_path = os.path.join(BASE_RESULTS, "charging_events_stationary_only.csv")
df_ch = pd.read_csv(charging_path)
df_ch['start_time'] = pd.to_datetime(df_ch['start_time'])
df_ch['end_time']   = pd.to_datetime(df_ch['end_time'])
print(f"   ✓ Charging events (stationary): {len(df_ch):,} rows, "
      f"{df_ch['vehicle_id'].nunique():,} vehicles")

# 1.2 验证标记列
print("\n   📋 Checking label columns:")
for col in ['charge_type', 'speed_label', 'charge_type_name']:
    if col in df_ch.columns:
        vals = df_ch[col].value_counts().to_dict()
        print(f"      ✅ {col}: {vals}")
    else:
        print(f"      ⚠️  {col} NOT found – will derive from avg_soc_rate")

# 如果缺少 charge_type，基于 avg_soc_rate 自动定义
if 'charge_type' not in df_ch.columns:
    df_ch['charge_type'] = np.where(df_ch['avg_soc_rate'] > 1.0, 'fast', 'slow')
    print("      → charge_type derived from avg_soc_rate > 1.0%/min")

if 'charge_type_name' not in df_ch.columns:
    df_ch['charge_type_name'] = np.where(
        df_ch['charge_type'] == 'fast', '快充 (>1%/min)', '慢充 (≤1%/min)'
    )

# 1.3 放电/行驶片段
segments_path = os.path.join(BASE_RESULTS, "segments_integrated_complete.csv")
df_seg = pd.read_csv(segments_path)
df_seg['start_dt'] = pd.to_datetime(df_seg['start_dt'])
df_seg['end_dt']   = pd.to_datetime(df_seg['end_dt'])
print(f"   ✓ Segments: {len(df_seg):,} rows, "
      f"{df_seg['vehicle_id'].nunique():,} vehicles")

# 只保留放电（行驶）片段 seg_type == 0
df_discharge = df_seg[df_seg['seg_type'] == 0].copy()
print(f"   ✓ Discharge segments (seg_type=0): {len(df_discharge):,}")

# 1.4 车辆聚类标签 (KMeans K=4 优先, fallback GMM K=4)
cluster_candidates = [
    os.path.join(VC_RESULTS, "vehicle_clustering_kmeans_final_k4.csv"),
    os.path.join(VC_RESULTS, "vehicle_clustering_gmm_k4.csv"),
]
df_vc = None
for cpath in cluster_candidates:
    if os.path.exists(cpath):
        df_vc = pd.read_csv(cpath)
        print(f"   ✓ Vehicle clusters: {cpath} ({len(df_vc):,} vehicles)")
        break

if df_vc is None:
    raise FileNotFoundError(
        "Vehicle clustering file not found. "
        "Run step13_kmeans_final_publication.py or step8_vehicle_clustering.py first."
    )

# 确定聚类列名
cluster_col = None
for c in ['vehicle_cluster', 'cluster', 'vehicle_archetype_id']:
    if c in df_vc.columns:
        cluster_col = c
        break
if cluster_col is None:
    raise ValueError(f"No cluster column found in {df_vc.columns.tolist()}")

label_col = None
for c in ['cluster_label', 'vehicle_type', 'archetype_label']:
    if c in df_vc.columns:
        label_col = c
        break

print(f"   ✓ Using cluster column: '{cluster_col}', label column: '{label_col}'")

# ============================================================
# 2. 合并车辆聚类标签到充电事件
# ============================================================
print("\n【STEP 2】Merging vehicle cluster labels")
print("=" * 80)

vc_merge = df_vc[['vehicle_id', cluster_col]].copy()
if label_col:
    vc_merge[label_col] = df_vc[label_col]
vc_merge = vc_merge.drop_duplicates('vehicle_id')

df_ch = df_ch.merge(vc_merge, on='vehicle_id', how='left')
df_discharge = df_discharge.merge(vc_merge, on='vehicle_id', how='left')

# 如果没有 label，从 cluster id 生成
if label_col is None or label_col not in df_ch.columns:
    label_col = 'archetype_label'
    archetype_names = {
        0: 'Archetype-0', 1: 'Archetype-1',
        2: 'Archetype-2', 3: 'Archetype-3',
    }
    df_ch[label_col] = df_ch[cluster_col].map(archetype_names).fillna('Unknown')
    df_discharge[label_col] = df_discharge[cluster_col].map(archetype_names).fillna('Unknown')

matched = df_ch[cluster_col].notna().sum()
print(f"   ✓ Matched {matched:,} / {len(df_ch):,} charging events to vehicle clusters")

# ============================================================
# 3. 放电片段特征分析 → discharge_segments_analysis.csv
# ============================================================
print("\n【STEP 3】Discharge Segment Analysis")
print("=" * 80)

seg_features = df_discharge[[
    'segment_id', 'vehicle_id', 'date', 'start_dt', 'end_dt',
    'duration_min', 'soc_start', 'soc_end', 'soc_drop',
    'phys_avg_speed', 'phys_seg_length', 'phys_soc_rate',
    'cluster_name', cluster_col, label_col,
]].copy()

seg_features.columns = [
    'segment_id', 'vehicle_id', 'date', 'start_dt', 'end_dt',
    'duration_min', 'soc_start', 'soc_end', 'soc_drop',
    'avg_speed_kmh', 'seg_length_km', 'soc_rate_pct_per_min',
    'driving_mode', 'vehicle_cluster', 'archetype_label',
]

seg_out = os.path.join(VC_RESULTS, "discharge_segments_analysis.csv")
seg_features.to_csv(seg_out, index=False)
print(f"   ✓ Saved: {seg_out} ({len(seg_features):,} rows)")

# ============================================================
# 4. 计算每辆车焦虑指标
# ============================================================
print("\n【STEP 4】Computing Per-Vehicle Anxiety Metrics")
print("=" * 80)

# 数据集时间跨度（用于频率计算）
date_range_days = (df_ch['start_time'].max() - df_ch['start_time'].min()).days + 1
total_distance_by_vehicle = (
    df_discharge.groupby('vehicle_id')['phys_seg_length'].sum().rename('total_dist_km')
)

anxiety_records = []

for vid, g_ch in df_ch.groupby('vehicle_id'):
    g_ch = g_ch.sort_values('start_time').reset_index(drop=True)
    n_charges = len(g_ch)
    if n_charges < 2:
        continue

    # 车辆日期范围
    v_days = (g_ch['start_time'].max() - g_ch['start_time'].min()).days + 1
    v_days = max(v_days, 1)

    # ── 原有焦虑指标 ───────────────────────────────────────────
    # 1. 低SOC频率 (soc_start < 20%)
    low_soc_freq = (g_ch['soc_start'] < 20).mean()

    # 2. 极低SOC频率 (soc_start < 10%)
    very_low_soc_freq = (g_ch['soc_start'] < 10).mean()

    # 3. 平均最小SOC（充电开始时的均值）
    avg_min_soc = g_ch['soc_start'].mean()

    # ── 新增焦虑指标 ───────────────────────────────────────────
    # 4. 充电开始时机焦虑（SOC越低时才充 = 越焦虑）
    anxiety_from_timing = (100 - g_ch['soc_start'].mean()) / 100.0

    # 5. 单位时间充电频率（每天充电次数）
    charging_freq_per_day = n_charges / v_days

    # 6. 单位里程充电频率（每100km充电次数）
    total_dist = total_distance_by_vehicle.get(vid, np.nan)
    if pd.notna(total_dist) and total_dist > 0:
        charging_freq_per_100km = (n_charges / total_dist) * 100.0
    else:
        charging_freq_per_100km = np.nan

    # 7. 充电完成率（平均充到多少%）
    avg_soc_end = g_ch['soc_end'].mean()
    # 高完成率 (>95%) = 保守, 低完成率 (<60%) = 定量充电
    charge_completion_rate = avg_soc_end / 100.0

    # 8. 充电深度 DoD（充电前SOC与上次充后SOC的差值均值）
    dod_values = []
    for i in range(1, n_charges):
        prev_end = g_ch['soc_end'].iloc[i - 1]
        curr_start = g_ch['soc_start'].iloc[i]
        dod_values.append(prev_end - curr_start)
    avg_dod = np.mean(dod_values) if dod_values else np.nan

    # 9. 充电间隔变异系数（Coefficient of Variation）
    if n_charges >= 3:
        intervals_h = []
        for i in range(1, n_charges):
            delta = (g_ch['start_time'].iloc[i] - g_ch['end_time'].iloc[i - 1]).total_seconds() / 3600.0
            if delta >= 0:
                intervals_h.append(delta)
        if len(intervals_h) >= 2:
            m_int = np.mean(intervals_h)
            s_int = np.std(intervals_h, ddof=1)
            cv_interval = s_int / m_int if m_int > 0 else np.nan
        else:
            cv_interval = np.nan
    else:
        cv_interval = np.nan

    # 10. 快充依赖度
    fast_mask = g_ch['charge_type'] == 'fast'
    fast_ratio = fast_mask.mean()

    # 11. 平均充电时段集中度（夜间充电比例）
    night_charge_ratio = g_ch['is_night_charge'].mean() if 'is_night_charge' in g_ch.columns else np.nan

    # ── 综合焦虑指数（加权平均） ──────────────────────────────
    # 指标已归一化到 [0,1]；高值 = 高焦虑
    components = {
        'low_soc_freq':          (low_soc_freq,          0.20),
        'very_low_soc_freq':     (very_low_soc_freq,     0.10),
        'anxiety_from_timing':   (anxiety_from_timing,   0.20),
        'freq_per_day_norm':     (min(charging_freq_per_day / MAX_CHARGES_PER_DAY, 1.0), 0.15),
        'fast_ratio':            (fast_ratio,             0.15),
        'dod_norm':              (min(avg_dod / MAX_EXPECTED_DOD, 1.0) if pd.notna(avg_dod) else 0, 0.10),
        'cv_interval_norm':      (min(cv_interval / MAX_EXPECTED_CV, 1.0) if pd.notna(cv_interval) else 0, 0.10),
    }
    total_w = sum(w for _, w in components.values())
    anxiety_index = sum(v * w for v, w in components.values()) / total_w

    # 获取聚类信息
    v_cluster = g_ch[cluster_col].iloc[0] if cluster_col in g_ch.columns else np.nan
    v_label   = g_ch[label_col].iloc[0]   if label_col in g_ch.columns else 'Unknown'

    anxiety_records.append({
        'vehicle_id':               vid,
        'vehicle_cluster':          v_cluster,
        'archetype_label':          v_label,
        'n_charging_events':        n_charges,
        'observation_days':         v_days,
        # 焦虑指标
        'low_soc_freq':             round(low_soc_freq, 4),
        'very_low_soc_freq':        round(very_low_soc_freq, 4),
        'avg_charge_start_soc':     round(avg_min_soc, 2),
        'anxiety_from_timing':      round(anxiety_from_timing, 4),
        'charging_freq_per_day':    round(charging_freq_per_day, 4),
        'charging_freq_per_100km':  round(charging_freq_per_100km, 4) if pd.notna(charging_freq_per_100km) else np.nan,
        'avg_charge_completion_soc':round(avg_soc_end, 2),
        'avg_depth_of_discharge':   round(avg_dod, 2) if pd.notna(avg_dod) else np.nan,
        'cv_charging_interval':     round(cv_interval, 4) if pd.notna(cv_interval) else np.nan,
        'fast_charge_ratio':        round(fast_ratio, 4),
        'night_charge_ratio':       round(night_charge_ratio, 4) if pd.notna(night_charge_ratio) else np.nan,
        'anxiety_index':            round(anxiety_index, 4),
        # 辅助信息
        'total_distance_km':        round(total_dist, 2) if pd.notna(total_dist) else np.nan,
    })

df_anxiety = pd.DataFrame(anxiety_records)
print(f"   ✓ Computed anxiety metrics for {len(df_anxiety):,} vehicles")

anxiety_out = os.path.join(VC_RESULTS, "vehicle_anxiety_metrics.csv")
df_anxiety.to_csv(anxiety_out, index=False)
print(f"   ✓ Saved: {anxiety_out}")

# ============================================================
# 5. 四类车焦虑对比 → archetype_anxiety_comparison.csv
# ============================================================
print("\n【STEP 5】Archetype Anxiety Comparison")
print("=" * 80)

anxiety_metric_cols = [
    'low_soc_freq', 'very_low_soc_freq', 'avg_charge_start_soc',
    'anxiety_from_timing', 'charging_freq_per_day', 'charging_freq_per_100km',
    'avg_charge_completion_soc', 'avg_depth_of_discharge',
    'cv_charging_interval', 'fast_charge_ratio', 'night_charge_ratio',
    'anxiety_index',
]

comparison_rows = []
for lbl, grp in df_anxiety.groupby('archetype_label'):
    row = {'archetype_label': lbl, 'n_vehicles': len(grp)}
    for col in anxiety_metric_cols:
        if col in grp.columns:
            row[f'{col}_mean'] = round(grp[col].mean(), 4)
            row[f'{col}_std']  = round(grp[col].std(), 4)
            row[f'{col}_median'] = round(grp[col].median(), 4)
    comparison_rows.append(row)

df_comparison = pd.DataFrame(comparison_rows).sort_values('archetype_label')
comp_out = os.path.join(VC_RESULTS, "archetype_anxiety_comparison.csv")
df_comparison.to_csv(comp_out, index=False)
print(f"   ✓ Saved: {comp_out}")

for _, row in df_comparison.iterrows():
    print(f"   {row['archetype_label']} (n={row['n_vehicles']}): "
          f"anxiety_index={row.get('anxiety_index_mean', np.nan):.3f}  "
          f"fast_charge_ratio={row.get('fast_charge_ratio_mean', np.nan):.3f}")

# ============================================================
# 6. 因果关系数据 → causality_discharge_to_charging.csv
# ============================================================
print("\n【STEP 6】Causal Chain: Discharge → Charging")
print("=" * 80)

# 基于行程级别构建因果链
# 对每个充电事件，找它之前的最后一段放电片段
df_ch_sorted   = df_ch.sort_values(['vehicle_id', 'start_time'])
df_dis_sorted  = df_discharge.sort_values(['vehicle_id', 'start_dt'])

causal_records = []
for vid, g_ch in df_ch_sorted.groupby('vehicle_id'):
    g_dis = df_dis_sorted[df_dis_sorted['vehicle_id'] == vid]
    if g_dis.empty:
        continue
    for _, ch_row in g_ch.iterrows():
        # 找充电前的放电片段
        prev_dis = g_dis[g_dis['end_dt'] <= ch_row['start_time']]
        if prev_dis.empty:
            continue
        last_dis = prev_dis.iloc[-1]

        causal_records.append({
            'vehicle_id':           vid,
            'archetype_label':      ch_row.get(label_col, 'Unknown'),
            # 前序放电段特征
            'pre_seg_duration_min': last_dis['duration_min'],
            'pre_seg_soc_drop':     last_dis['soc_drop'],
            'pre_seg_avg_speed':    last_dis['phys_avg_speed'],
            'pre_seg_length_km':    last_dis['phys_seg_length'],
            'pre_seg_soc_rate':     last_dis['phys_soc_rate'],
            'pre_seg_driving_mode': last_dis.get('cluster_name', 'Unknown'),
            # 充电事件特征
            'charge_start_soc':     ch_row['soc_start'],
            'charge_end_soc':       ch_row['soc_end'],
            'charge_gain':          ch_row['soc_gain'],
            'charge_type':          ch_row['charge_type'],
            'charge_duration_min':  ch_row['duration_minutes'],
            'charge_start_hour':    ch_row['start_hour'] if 'start_hour' in ch_row else ch_row['start_time'].hour,
        })

df_causal = pd.DataFrame(causal_records)
causal_out = os.path.join(VC_RESULTS, "causality_discharge_to_charging.csv")
df_causal.to_csv(causal_out, index=False)
print(f"   ✓ Saved: {causal_out} ({len(df_causal):,} causal pairs)")

# ============================================================
# 7. 可视化
# ============================================================
print("\n【STEP 7】Generating Visualizations (8-10 figures)")
print("=" * 80)

archetypes = sorted(df_anxiety['archetype_label'].dropna().unique())
n_arch = len(archetypes)
palette = sns.color_palette("Set2", n_arch)
arch_colors = {a: c for a, c in zip(archetypes, palette)}

# ── Figure 1: 四类车焦虑程度分布（综合指数 violin + strip） ──────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
sns.violinplot(data=df_anxiety, x='archetype_label', y='anxiety_index',
               palette=palette, ax=ax, inner='quartile', linewidth=1.5)
ax.set_title('Charging Anxiety Index Distribution\nby Vehicle Archetype',
             fontweight='bold', fontsize=12)
ax.set_xlabel('Vehicle Archetype', fontweight='bold')
ax.set_ylabel('Anxiety Index (0–1, higher = more anxious)', fontweight='bold')
ax.tick_params(axis='x', rotation=15)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 1)

ax = axes[1]
mean_vals = df_anxiety.groupby('archetype_label')['anxiety_index'].mean().reindex(archetypes)
std_vals  = df_anxiety.groupby('archetype_label')['anxiety_index'].std().reindex(archetypes)
bars = ax.bar(archetypes, mean_vals, yerr=std_vals, capsize=5,
              color=palette, edgecolor='black', linewidth=1.2, error_kw={'linewidth': 1.5})
for bar, v in zip(bars, mean_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
            f'{v:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_title('Mean Anxiety Index by Vehicle Archetype',
             fontweight='bold', fontsize=12)
ax.set_xlabel('Vehicle Archetype', fontweight='bold')
ax.set_ylabel('Mean Anxiety Index', fontweight='bold')
ax.tick_params(axis='x', rotation=15)
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig01_anxiety_index_distribution.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ fig01_anxiety_index_distribution.png")

# ── Figure 2: 低SOC频率 + 充电开始时机 ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
sns.boxplot(data=df_anxiety, x='archetype_label', y='low_soc_freq',
            palette=palette, ax=ax, width=0.6)
ax.set_title('Low-SOC Frequency (SOC < 20%)\nby Vehicle Archetype',
             fontweight='bold', fontsize=12)
ax.set_xlabel('Vehicle Archetype', fontweight='bold')
ax.set_ylabel('Fraction of charges started at SOC < 20%', fontweight='bold')
ax.tick_params(axis='x', rotation=15)
ax.grid(True, alpha=0.3, axis='y')

ax = axes[1]
sns.boxplot(data=df_anxiety, x='archetype_label', y='avg_charge_start_soc',
            palette=palette, ax=ax, width=0.6)
ax.set_title('Average Charge-Start SOC\nby Vehicle Archetype (Lower = More Anxious)',
             fontweight='bold', fontsize=12)
ax.set_xlabel('Vehicle Archetype', fontweight='bold')
ax.set_ylabel('Mean SOC when charging starts (%)', fontweight='bold')
ax.tick_params(axis='x', rotation=15)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig02_low_soc_and_charge_timing.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ fig02_low_soc_and_charge_timing.png")

# ── Figure 3: 单位时间/里程充电频率 ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
df_plot = df_anxiety[df_anxiety['charging_freq_per_day'].notna()]
sns.boxplot(data=df_plot, x='archetype_label', y='charging_freq_per_day',
            palette=palette, ax=ax, width=0.6)
ax.set_title('Charging Frequency per Day\nby Vehicle Archetype (Higher = More Anxious)',
             fontweight='bold', fontsize=12)
ax.set_xlabel('Vehicle Archetype', fontweight='bold')
ax.set_ylabel('Charging events per day', fontweight='bold')
ax.tick_params(axis='x', rotation=15)
ax.grid(True, alpha=0.3, axis='y')

ax = axes[1]
df_plot2 = df_anxiety[df_anxiety['charging_freq_per_100km'].notna()]
if len(df_plot2) > 0:
    sns.boxplot(data=df_plot2, x='archetype_label', y='charging_freq_per_100km',
                palette=palette, ax=ax, width=0.6)
    ax.set_title('Charging Frequency per 100 km\nby Vehicle Archetype',
                 fontweight='bold', fontsize=12)
    ax.set_xlabel('Vehicle Archetype', fontweight='bold')
    ax.set_ylabel('Charging events per 100 km', fontweight='bold')
    ax.tick_params(axis='x', rotation=15)
    ax.grid(True, alpha=0.3, axis='y')
else:
    ax.text(0.5, 0.5, 'Distance data not available', ha='center', va='center',
            transform=ax.transAxes, fontsize=12)
    ax.set_title('Charging Frequency per 100 km\n(Data unavailable)', fontweight='bold')

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig03_charging_frequency.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ fig03_charging_frequency.png")

# ── Figure 4: 充电深度 DoD 与充电完成率 ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
df_dod = df_anxiety[df_anxiety['avg_depth_of_discharge'].notna()]
if len(df_dod) > 0:
    sns.boxplot(data=df_dod, x='archetype_label', y='avg_depth_of_discharge',
                palette=palette, ax=ax, width=0.6)
    ax.set_title('Average Depth of Discharge (DoD)\nby Vehicle Archetype',
                 fontweight='bold', fontsize=12)
    ax.set_xlabel('Vehicle Archetype', fontweight='bold')
    ax.set_ylabel('SOC drop between charges (%)', fontweight='bold')
    ax.tick_params(axis='x', rotation=15)
    ax.grid(True, alpha=0.3, axis='y')

ax = axes[1]
sns.boxplot(data=df_anxiety, x='archetype_label', y='avg_charge_completion_soc',
            palette=palette, ax=ax, width=0.6)
ax.set_title('Average Charge Completion SOC\nby Vehicle Archetype (Higher = More Conservative)',
             fontweight='bold', fontsize=12)
ax.set_xlabel('Vehicle Archetype', fontweight='bold')
ax.set_ylabel('Mean SOC when charging stops (%)', fontweight='bold')
ax.tick_params(axis='x', rotation=15)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig04_dod_and_completion.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ fig04_dod_and_completion.png")

# ── Figure 5: 快充依赖度分布 ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
sns.boxplot(data=df_anxiety, x='archetype_label', y='fast_charge_ratio',
            palette=palette, ax=ax, width=0.6)
ax.set_title('Fast Charge Ratio by Vehicle Archetype\n(Higher = More Dependent on Fast Charging)',
             fontweight='bold', fontsize=12)
ax.set_xlabel('Vehicle Archetype', fontweight='bold')
ax.set_ylabel('Fraction of fast-charging events', fontweight='bold')
ax.tick_params(axis='x', rotation=15)
ax.grid(True, alpha=0.3, axis='y')

ax = axes[1]
# 快充依赖度散点 vs 焦虑指数
for i, arch in enumerate(archetypes):
    mask = df_anxiety['archetype_label'] == arch
    ax.scatter(df_anxiety.loc[mask, 'fast_charge_ratio'],
               df_anxiety.loc[mask, 'anxiety_index'],
               color=arch_colors[arch], alpha=0.5, s=30, label=arch)
ax.set_title('Fast Charge Ratio vs Anxiety Index\n(Scatter by Archetype)',
             fontweight='bold', fontsize=12)
ax.set_xlabel('Fast Charge Ratio', fontweight='bold')
ax.set_ylabel('Anxiety Index', fontweight='bold')
ax.legend(fontsize=9, title='Archetype')
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig05_fast_charge_dependency.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ fig05_fast_charge_dependency.png")

# ── Figure 6: 充电间隔变异系数（规律性） ─────────────────────────────────
df_cv = df_anxiety[df_anxiety['cv_charging_interval'].notna()]
if len(df_cv) > 0:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(data=df_cv, x='archetype_label', y='cv_charging_interval',
                   palette=palette, ax=ax, inner='quartile', linewidth=1.5)
    ax.set_title('Charging Interval Variability (CV)\nby Vehicle Archetype\n'
                 '(Low CV = Regular charging schedule = Proactive management)',
                 fontweight='bold', fontsize=12)
    ax.set_xlabel('Vehicle Archetype', fontweight='bold')
    ax.set_ylabel('Coefficient of Variation of charging intervals', fontweight='bold')
    ax.tick_params(axis='x', rotation=15)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig06_charging_interval_cv.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ fig06_charging_interval_cv.png")

# ── Figure 7: 充电开始时段分布（小时热图） ───────────────────────────────
if 'start_hour' in df_ch.columns:
    fig, axes = plt.subplots(1, n_arch, figsize=(4 * n_arch, 5), sharey=False)
    if n_arch == 1:
        axes = [axes]
    for i, arch in enumerate(archetypes):
        mask = df_ch[label_col] == arch
        hour_counts = df_ch.loc[mask, 'start_hour'].value_counts().sort_index()
        # 填充缺失的小时
        all_hours = pd.Series(0, index=range(24))
        all_hours.update(hour_counts)
        axes[i].bar(all_hours.index, all_hours.values,
                    color=arch_colors[arch], edgecolor='black', linewidth=0.8)
        axes[i].set_title(f'{arch}', fontweight='bold', fontsize=10)
        axes[i].set_xlabel('Hour of Day', fontweight='bold')
        if i == 0:
            axes[i].set_ylabel('Number of Charging Events', fontweight='bold')
        axes[i].grid(True, alpha=0.3, axis='y')
        axes[i].set_xticks([0, 6, 12, 18, 23])

    plt.suptitle('Charging Start Hour Distribution by Vehicle Archetype\n'
                 '(Concentrated = Proactive planning; Spread = Reactive charging)',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig07_charge_hour_distribution.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ fig07_charge_hour_distribution.png")

# ── Figure 8: 前序放电特性 → 充电选择（因果流） ──────────────────────────
if len(df_causal) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax = axes[0, 0]
    for i, arch in enumerate(archetypes):
        mask = df_causal['archetype_label'] == arch
        ax.scatter(df_causal.loc[mask, 'pre_seg_soc_drop'],
                   df_causal.loc[mask, 'charge_start_soc'],
                   color=arch_colors[arch], alpha=0.3, s=15, label=arch)
    ax.set_title('Pre-Discharge SOC Drop → Charge Start SOC\n(Causal Link)',
                 fontweight='bold', fontsize=11)
    ax.set_xlabel('SOC drop during pre-discharge segment (%)', fontweight='bold')
    ax.set_ylabel('Charge start SOC (%)', fontweight='bold')
    ax.legend(fontsize=9, title='Archetype')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for i, arch in enumerate(archetypes):
        mask = df_causal['archetype_label'] == arch
        ax.scatter(df_causal.loc[mask, 'pre_seg_avg_speed'],
                   df_causal.loc[mask, 'charge_duration_min'],
                   color=arch_colors[arch], alpha=0.3, s=15, label=arch)
    ax.set_title('Pre-Discharge Speed → Charging Duration\n(Higher speed → Faster depletion → Longer charge?)',
                 fontweight='bold', fontsize=11)
    ax.set_xlabel('Pre-discharge avg speed (km/h)', fontweight='bold')
    ax.set_ylabel('Charging duration (min)', fontweight='bold')
    ax.legend(fontsize=9, title='Archetype')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    fast_count = df_causal.groupby(['archetype_label', 'charge_type']).size().unstack(fill_value=0)
    fast_count_norm = fast_count.div(fast_count.sum(axis=1), axis=0)
    if 'fast' in fast_count_norm.columns and 'slow' in fast_count_norm.columns:
        x_pos = np.arange(len(fast_count_norm))
        width = 0.35
        ax.bar(x_pos - width/2, fast_count_norm['slow'],
               width, label='Slow Charge', color='#4ECDC4', edgecolor='black')
        ax.bar(x_pos + width/2, fast_count_norm['fast'],
               width, label='Fast Charge', color='#FF6B6B', edgecolor='black')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(fast_count_norm.index, rotation=15, fontsize=9)
        ax.set_title('Charge Type Distribution\nby Vehicle Archetype',
                     fontweight='bold', fontsize=11)
        ax.set_xlabel('Vehicle Archetype', fontweight='bold')
        ax.set_ylabel('Proportion', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1, 1]
    # 充电开始SOC分布（ECDF风格的KDE）
    for i, arch in enumerate(archetypes):
        mask = df_causal['archetype_label'] == arch
        vals = df_causal.loc[mask, 'charge_start_soc'].dropna()
        if len(vals) > 10:
            vals.plot.kde(ax=ax, label=arch, color=arch_colors[arch], linewidth=2)
    ax.set_title('Charge-Start SOC Distribution\nby Vehicle Archetype\n(Left = Earlier charging = Lower anxiety)',
                 fontweight='bold', fontsize=11)
    ax.set_xlabel('Charge Start SOC (%)', fontweight='bold')
    ax.set_ylabel('Density', fontweight='bold')
    ax.legend(fontsize=9, title='Archetype')
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Causal Chain: Discharge Characteristics → Charging Behavior',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig08_causal_discharge_to_charging.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ fig08_causal_discharge_to_charging.png")

# ── Figure 9: 焦虑热力图（车型 × 指标） ─────────────────────────────────
heatmap_cols = [
    'low_soc_freq', 'very_low_soc_freq', 'anxiety_from_timing',
    'charging_freq_per_day', 'fast_charge_ratio',
    'avg_depth_of_discharge', 'cv_charging_interval', 'anxiety_index',
]
heatmap_labels = [
    'Low-SOC Freq\n(<20%)', 'Very Low-SOC\n(<10%)', 'Anxiety\nTiming',
    'Charge Freq\n(/day)', 'Fast Charge\nRatio',
    'Depth of\nDischarge', 'Interval CV\n(Irregularity)', 'Anxiety\nIndex',
]

df_hm = (df_anxiety.groupby('archetype_label')[heatmap_cols]
         .mean()
         .reindex(archetypes))

# z-score normalize per column for visual contrast
df_hm_norm = (df_hm - df_hm.mean()) / (df_hm.std() + EPSILON)

fig, ax = plt.subplots(figsize=(14, 5))
sns.heatmap(df_hm_norm.T, annot=df_hm.T.round(3), fmt='.3f',
            cmap='RdYlGn_r', ax=ax,
            xticklabels=archetypes,
            yticklabels=heatmap_labels,
            linewidths=0.5, linecolor='white',
            cbar_kws={'label': 'Z-score (relative anxiety level)'})
ax.set_title('Anxiety Metrics Heatmap by Vehicle Archetype\n'
             '(Red = Higher anxiety; Green = Lower anxiety; Values = Actual means)',
             fontweight='bold', fontsize=13, pad=12)
ax.set_xlabel('Vehicle Archetype', fontweight='bold')
ax.set_ylabel('Anxiety Metric', fontweight='bold')
ax.tick_params(axis='x', rotation=20)
ax.tick_params(axis='y', rotation=0)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig09_anxiety_heatmap.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ fig09_anxiety_heatmap.png")

# ── Figure 10: 放电-充电三维关系 ─────────────────────────────────────────
if len(df_causal) > 0:
    fig = plt.figure(figsize=(12, 8))

    ax_main = fig.add_subplot(111)
    for i, arch in enumerate(archetypes):
        mask = df_causal['archetype_label'] == arch
        sub = df_causal[mask].dropna(subset=['pre_seg_soc_drop', 'charge_start_soc', 'charge_duration_min'])
        if len(sub) == 0:
            continue
        scatter = ax_main.scatter(
            sub['pre_seg_soc_drop'],
            sub['charge_start_soc'],
            c=sub['charge_duration_min'].clip(0, 300),
            cmap='plasma', alpha=0.4, s=20,
            vmin=0, vmax=MAX_CHARGE_DURATION_MIN,
        )
    # Add annotation
    plt.colorbar(scatter, ax=ax_main, label='Charging Duration (min)', pad=0.01)
    ax_main.set_title('Discharge Characteristics → Charging Behavior\n'
                      '(X=pre-discharge SOC drop, Y=charge-start SOC, Color=charge duration)',
                      fontweight='bold', fontsize=12)
    ax_main.set_xlabel('Pre-discharge SOC Drop (%)', fontweight='bold')
    ax_main.set_ylabel('Charge Start SOC (%)', fontweight='bold')
    ax_main.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig10_discharge_charging_3d_relationship.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ fig10_discharge_charging_3d_relationship.png")

# ============================================================
# 8. 统计检验
# ============================================================
print("\n【STEP 8】Statistical Tests (ANOVA)")
print("=" * 80)

test_cols = ['anxiety_index', 'avg_charge_start_soc', 'fast_charge_ratio', 'low_soc_freq']
for col in test_cols:
    groups = [df_anxiety.loc[df_anxiety['archetype_label'] == a, col].dropna().values
              for a in archetypes]
    groups = [g for g in groups if len(g) >= 3]
    if len(groups) >= 2:
        f_stat, p_val = stats.f_oneway(*groups)
        sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'ns'))
        print(f"   {col:<35}: F={f_stat:.2f}, p={p_val:.4e} {sig}")

# ============================================================
# 9. 报告摘要
# ============================================================
print("\n【STEP 9】Summary Report")
print("=" * 80)

# 按焦虑指数排序
if len(df_comparison) > 0:
    idx_col = 'anxiety_index_mean'
    if idx_col in df_comparison.columns:
        ranked = df_comparison.sort_values(idx_col, ascending=False)
        print("\n🏆 Vehicle Archetype Anxiety Ranking (High → Low):")
        for rank, (_, row) in enumerate(ranked.iterrows(), 1):
            print(f"   {rank}. {row['archetype_label']:<30} "
                  f"anxiety_index={row[idx_col]:.3f}  "
                  f"fast_charge_ratio={row.get('fast_charge_ratio_mean', np.nan):.3f}  "
                  f"n={row['n_vehicles']}")

print(f"""
📦 Output Files:
   1. {seg_out}
   2. {anxiety_out}
   3. {comp_out}
   4. {causal_out}
   5. {FIG_DIR}/fig01_anxiety_index_distribution.png
   6. {FIG_DIR}/fig02_low_soc_and_charge_timing.png
   7. {FIG_DIR}/fig03_charging_frequency.png
   8. {FIG_DIR}/fig04_dod_and_completion.png
   9. {FIG_DIR}/fig05_fast_charge_dependency.png
  10. {FIG_DIR}/fig06_charging_interval_cv.png
  11. {FIG_DIR}/fig07_charge_hour_distribution.png
  12. {FIG_DIR}/fig08_causal_discharge_to_charging.png
  13. {FIG_DIR}/fig09_anxiety_heatmap.png
  14. {FIG_DIR}/fig10_discharge_charging_3d_relationship.png
""")

print("=" * 80)
print("✅ Step 14 (v3 Integrated) Complete!")
print("=" * 80)
