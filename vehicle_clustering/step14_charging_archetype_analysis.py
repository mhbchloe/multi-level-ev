"""
Step 14: Four Vehicle Archetypes Charging Pattern Analysis (Integrated)
四类车充电模式对比分析 - 集成版本

直接加载已处理的充电事件数据，进行四类车充电模式对比分析。

Data Sources:
  - coupling_analysis/results/charging_events_stationary_only.csv
  - vehicle_clustering/results/vehicle_clustering_kmeans_final_k4.csv
  - coupling_analysis/results/segments_integrated_complete.csv  (optional supplement)

Outputs:
  vehicle_clustering/results/
  ├── charging_archetype_analysis_integrated/
  │   ├── charging_characteristics_by_archetype.csv
  │   ├── archetype_summary.json
  │   └── charging_archetype_integrated_report.txt
  └── figures_charging_archetypes_integrated/
      ├── 01_four_archetypes_overview.png
      ├── 02_charging_demand_profile.png
      ├── 03_grid_impact_comparison.png
      └── 04_charging_infrastructure_requirement.png
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib import rcParams
from scipy.stats import entropy as scipy_entropy

warnings.filterwarnings('ignore')

rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 11
rcParams['figure.dpi'] = 150

print("=" * 80)
print("🚀 STEP 14: FOUR VEHICLE ARCHETYPES CHARGING PATTERN ANALYSIS (INTEGRATED)")
print("=" * 80)

# ============================================================
# 配置
# ============================================================
CONFIG = {
    'charging_events_path': './coupling_analysis/results/charging_events_stationary_only.csv',
    'clustering_path': './vehicle_clustering/results/vehicle_clustering_kmeans_final_k4.csv',
    'segments_path': './coupling_analysis/results/segments_integrated_complete.csv',
    'output_dir': './vehicle_clustering/results/charging_archetype_analysis_integrated/',
    'figures_dir': './vehicle_clustering/results/figures_charging_archetypes_integrated/',
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)
os.makedirs(CONFIG['figures_dir'], exist_ok=True)

# 四类车颜色与标签
ARCHETYPE_COLORS = {
    0: '#E74C3C',   # C0: Mixed-Pattern  – 红
    1: '#3498DB',   # C1: Highway-Intensive – 蓝
    2: '#2ECC71',   # C2: Idle-Stable   – 绿
    3: '#F39C12',   # C3: Urban-Dynamic – 橙
}
ARCHETYPE_LABELS = {
    0: 'C0: Mixed-Pattern',
    1: 'C1: Highway-Intensive',
    2: 'C2: Idle-Stable',
    3: 'C3: Urban-Dynamic',
}
ARCHETYPE_SHORT = {
    0: 'C0\nMixed',
    1: 'C1\nHighway',
    2: 'C2\nIdle',
    3: 'C3\nUrban',
}

# ============================================================
# 1. 加载数据
# ============================================================
print(f"\n【STEP 1】Loading Data")
print("=" * 80)

# 1.1 充电事件
print(f"   Loading charging events …")
df_charging = pd.read_csv(CONFIG['charging_events_path'])
df_charging['start_time'] = pd.to_datetime(df_charging['start_time'], errors='coerce')
df_charging['end_time'] = pd.to_datetime(df_charging['end_time'], errors='coerce')
print(f"   ✓ Charging events: {len(df_charging):,}  |  vehicles: {df_charging['vehicle_id'].nunique():,}")

# 1.2 聚类结果
print(f"   Loading vehicle clustering labels …")
df_cluster = pd.read_csv(CONFIG['clustering_path'])
print(f"   ✓ Clustering results: {len(df_cluster):,} vehicles")
print(f"   ✓ Columns: {list(df_cluster.columns)}")

# ============================================================
# 2. 数据预处理 & 合并
# ============================================================
print(f"\n【STEP 2】Data Preprocessing & Merging")
print("=" * 80)

# 确保必要字段存在
if 'duration_seconds' not in df_charging.columns:
    if 'duration_minutes' in df_charging.columns:
        df_charging['duration_seconds'] = df_charging['duration_minutes'] * 60
    else:
        df_charging['duration_seconds'] = (
            df_charging['end_time'] - df_charging['start_time']
        ).dt.total_seconds()

if 'duration_minutes' not in df_charging.columns:
    df_charging['duration_minutes'] = df_charging['duration_seconds'] / 60.0

if 'start_hour' not in df_charging.columns:
    df_charging['start_hour'] = df_charging['start_time'].dt.hour

if 'start_weekday' not in df_charging.columns:
    df_charging['start_weekday'] = df_charging['start_time'].dt.dayofweek

if 'is_night_charge' not in df_charging.columns:
    df_charging['is_night_charge'] = df_charging['start_hour'].isin(
        [22, 23, 0, 1, 2, 3, 4, 5]
    ).astype(int)

# charge_type: fast / slow
if 'charge_type' not in df_charging.columns:
    # 根据 ch_s_type 或 SOC 速率推断
    if 'ch_s_type' in df_charging.columns:
        df_charging['charge_type'] = df_charging['ch_s_type'].apply(
            lambda x: 'fast' if str(x).lower() in ('fast', '2', '快充') else 'slow'
        )
    else:
        # 利用 soc_gain / duration_hours 估算功率，>= 20%/h 视为快充
        dur_h = df_charging['duration_seconds'] / 3600
        dur_h = dur_h.replace(0, np.nan)
        soc_rate = df_charging['soc_gain'] / dur_h
        df_charging['charge_type'] = np.where(soc_rate >= 20, 'fast', 'slow')

df_charging['is_fast_charge'] = (df_charging['charge_type'] == 'fast').astype(int)

# 合并聚类标签
cluster_col = 'vehicle_cluster' if 'vehicle_cluster' in df_cluster.columns else df_cluster.columns[1]
merge_cols = ['vehicle_id', cluster_col]
if 'cluster_label' in df_cluster.columns:
    merge_cols.append('cluster_label')

df = df_charging.merge(df_cluster[merge_cols], on='vehicle_id', how='inner')
df.rename(columns={cluster_col: 'vehicle_cluster'}, inplace=True)

print(f"   ✓ Merged dataset: {len(df):,} events from {df['vehicle_id'].nunique():,} vehicles")
print(f"\n   Archetype distribution:")
for vc in sorted(df['vehicle_cluster'].unique()):
    n_v = df[df['vehicle_cluster'] == vc]['vehicle_id'].nunique()
    n_e = (df['vehicle_cluster'] == vc).sum()
    print(f"      {ARCHETYPE_LABELS.get(vc, f'C{vc}')}: {n_v:,} vehicles, {n_e:,} events")

# ============================================================
# 3. 提取每辆车的充电特征，再按聚类汇总
# ============================================================
print(f"\n【STEP 3】Extracting Charging Features per Vehicle")
print("=" * 80)

def compute_vehicle_features(grp):
    """计算单辆车的充电特征"""
    n = len(grp)
    if n == 0:
        return None

    # 时间跨度（天）
    t_min = grp['start_time'].min()
    t_max = grp['end_time'].max()
    days = max((t_max - t_min).total_seconds() / 86400, 1)

    # SOC 速率 (%/h)
    dur_h = grp['duration_seconds'] / 3600
    dur_h = dur_h.replace(0, np.nan)
    soc_rate = grp['soc_gain'] / dur_h

    # 充电时段的信息熵（0=完全规律）
    hour_counts = grp['start_hour'].value_counts(normalize=True)
    timing_entropy = float(scipy_entropy(hour_counts.values))

    return pd.Series({
        'charging_frequency': n / days,                      # 次/天
        'avg_soc_gain': grp['soc_gain'].mean(),              # %
        'avg_duration': grp['duration_minutes'].mean(),      # 分钟
        'fast_charge_ratio': grp['is_fast_charge'].mean(),   # 0-1
        'night_charge_ratio': grp['is_night_charge'].mean(), # 0-1
        'charge_timing_regularity': timing_entropy,          # 熵
        'peak_soc_rate': soc_rate.quantile(0.95),            # %/h (95th pct)
        'avg_power_demand': soc_rate.mean(),                  # %/h
        'n_events': n,
        'vehicle_cluster': grp['vehicle_cluster'].iloc[0],
    })

print(f"   Computing per-vehicle features for {df['vehicle_id'].nunique():,} vehicles …")
# apply() used intentionally here: each vehicle group needs entropy + quantile
# computations that are not easily vectorised without a per-group loop.
vehicle_features = df.groupby('vehicle_id', group_keys=False).apply(compute_vehicle_features)
vehicle_features = vehicle_features.dropna(subset=['vehicle_cluster'])
print(f"   ✓ Features computed for {len(vehicle_features):,} vehicles")

# ============================================================
# 4. 按聚类汇总统计
# ============================================================
print(f"\n【STEP 4】Aggregating Statistics by Archetype")
print("=" * 80)

feature_cols = [
    'charging_frequency', 'avg_soc_gain', 'avg_duration',
    'fast_charge_ratio', 'night_charge_ratio', 'charge_timing_regularity',
    'peak_soc_rate', 'avg_power_demand',
]

archetype_stats = {}
for vc in sorted(vehicle_features['vehicle_cluster'].unique()):
    grp = vehicle_features[vehicle_features['vehicle_cluster'] == vc]
    stats = {}
    for col in feature_cols:
        stats[col + '_mean'] = grp[col].mean()
        stats[col + '_std'] = grp[col].std()
        stats[col + '_median'] = grp[col].median()
    stats['n_vehicles'] = len(grp)
    archetype_stats[int(vc)] = stats

# 打印汇总
print(f"\n   Archetype Charging Feature Summary:")
header = f"{'Feature':<30}" + "".join(
    f"{ARCHETYPE_LABELS.get(int(vc), f'C{vc}'):>20}"
    for vc in sorted(archetype_stats.keys())
)
print(f"   {header}")
for col in feature_cols:
    row = f"   {col:<30}"
    for vc in sorted(archetype_stats.keys()):
        row += f"{archetype_stats[vc][col + '_mean']:>20.3f}"
    print(row)

# ============================================================
# 5. 生成对比图表
# ============================================================
print(f"\n【STEP 5】Generating Comparison Charts")
print("=" * 80)

clusters = sorted(archetype_stats.keys())
colors = [ARCHETYPE_COLORS[c] for c in clusters]
x = np.arange(len(clusters))
bar_width = 0.6

def bar_with_std(ax, values, stds, ylabel, title, cluster_labels=None):
    """通用条形图 + 误差棒"""
    if cluster_labels is None:
        cluster_labels = [ARCHETYPE_SHORT.get(c, f'C{c}') for c in clusters]
    bars = ax.bar(x, values, bar_width, color=colors, edgecolor='white',
                  linewidth=0.8, zorder=3)
    ax.errorbar(x, values, yerr=stds, fmt='none', color='#444', linewidth=1.5,
                capsize=4, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(cluster_labels, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold', pad=6)
    ax.grid(axis='y', alpha=0.3, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # 标注数值
    for i, (bar, v) in enumerate(zip(bars, values)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + stds[i] * 0.05,
                f'{v:.2f}', ha='center', va='bottom', fontsize=8)
    return bars

# ------------------------------------------------------------------
# 图1: 01_four_archetypes_overview.png
# ------------------------------------------------------------------
print("   Generating Figure 1: Four Archetypes Overview …")

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle('Four Vehicle Archetypes – Charging Behaviour Overview',
             fontsize=14, fontweight='bold', y=0.98)

plot_specs = [
    ('charging_frequency', '次/天', 'Charging Frequency (times/day)'),
    ('avg_soc_gain', '%', 'Average SOC Gain (%)'),
    ('charge_timing_regularity', 'Entropy', 'Charge Timing Regularity (Entropy)'),
    ('fast_charge_ratio', 'Ratio', 'Fast Charge Ratio'),
]

for ax, (feat, unit, title) in zip(axes.flat, plot_specs):
    vals = [archetype_stats[c][feat + '_mean'] for c in clusters]
    stds = [archetype_stats[c][feat + '_std'] for c in clusters]
    bar_with_std(ax, vals, stds, unit, title)

# 添加图例
legend_patches = [
    mpatches.Patch(color=ARCHETYPE_COLORS[c], label=ARCHETYPE_LABELS[c])
    for c in clusters
]
fig.legend(handles=legend_patches, loc='lower center', ncol=4,
           fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, 0.01))

plt.tight_layout(rect=[0, 0.06, 1, 0.97])
fig1_path = os.path.join(CONFIG['figures_dir'], '01_four_archetypes_overview.png')
plt.savefig(fig1_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved → {fig1_path}")

# ------------------------------------------------------------------
# 图2: 02_charging_demand_profile.png – 24小时充电需求分布
# ------------------------------------------------------------------
print("   Generating Figure 2: 24-hour Charging Demand Profile …")

fig, ax = plt.subplots(figsize=(12, 5))

hours = np.arange(24)
for vc in clusters:
    grp_events = df[df['vehicle_cluster'] == vc]
    hour_counts = grp_events.groupby('start_hour').size().reindex(hours, fill_value=0)
    # 归一化为密度
    total = hour_counts.sum()
    density = hour_counts / total if total > 0 else hour_counts
    # 平滑（滚动平均）
    density_smooth = pd.Series(density).rolling(window=3, center=True,
                                                 min_periods=1).mean().values
    ax.plot(hours, density_smooth, color=ARCHETYPE_COLORS[vc],
            linewidth=2.2, label=ARCHETYPE_LABELS[vc], marker='o',
            markersize=4, alpha=0.9)

ax.set_xlabel('Hour of Day', fontsize=11)
ax.set_ylabel('Charging Demand Density', fontsize=11)
ax.set_title('24-Hour Charging Demand Distribution by Archetype',
             fontsize=13, fontweight='bold')
ax.set_xticks(hours)
ax.set_xlim(-0.5, 23.5)
ax.legend(fontsize=10, loc='upper left')
ax.grid(alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 标注时段
ax.axvspan(0, 6, alpha=0.05, color='navy', label='_Night')
ax.axvspan(7, 9, alpha=0.05, color='orange', label='_Morning Peak')
ax.axvspan(17, 20, alpha=0.05, color='red', label='_Evening Peak')
ax.text(2, ax.get_ylim()[1] * 0.92, 'Night', fontsize=8, color='navy', ha='center')
ax.text(8, ax.get_ylim()[1] * 0.92, 'AM Peak', fontsize=8, color='darkorange', ha='center')
ax.text(18.5, ax.get_ylim()[1] * 0.92, 'PM Peak', fontsize=8, color='red', ha='center')

plt.tight_layout()
fig2_path = os.path.join(CONFIG['figures_dir'], '02_charging_demand_profile.png')
plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved → {fig2_path}")

# ------------------------------------------------------------------
# 图3: 03_grid_impact_comparison.png – 对电网的影响对比
# ------------------------------------------------------------------
print("   Generating Figure 3: Grid Impact Comparison …")

# 可控性指数：夜间充电占比高 + 充电规律性好 → 可控
for vc in clusters:
    night = archetype_stats[vc]['night_charge_ratio_mean']
    reg = archetype_stats[vc]['charge_timing_regularity_mean']
    # 可控性 = night_ratio * (1 / (1 + reg))  归一化到 0-1
    archetype_stats[vc]['controllability_mean'] = night * (1 / (1 + reg))
    archetype_stats[vc]['controllability_std'] = archetype_stats[vc]['night_charge_ratio_std']

# 充电功率标准差（跨车辆的 avg_power_demand 标准差）
for vc in clusters:
    grp = vehicle_features[vehicle_features['vehicle_cluster'] == vc]
    archetype_stats[vc]['power_std_mean'] = grp['avg_power_demand'].std()
    archetype_stats[vc]['power_std_std'] = 0.0

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle('Grid Impact Comparison by Archetype',
             fontsize=14, fontweight='bold', y=0.98)

grid_specs = [
    ('peak_soc_rate', '%/h', 'Peak SOC Rate (%/h) [95th pct]'),
    ('night_charge_ratio', 'Ratio', 'Night Charge Ratio (22:00–06:00)'),
    ('power_std', '%/h', 'Power Demand Std Dev (%/h)'),
    ('controllability', 'Index', 'Controllability Index'),
]

for ax, (feat, unit, title) in zip(axes.flat, grid_specs):
    vals = [archetype_stats[c][feat + '_mean'] for c in clusters]
    stds = [archetype_stats[c][feat + '_std'] for c in clusters]
    bar_with_std(ax, vals, stds, unit, title)

legend_patches = [
    mpatches.Patch(color=ARCHETYPE_COLORS[c], label=ARCHETYPE_LABELS[c])
    for c in clusters
]
fig.legend(handles=legend_patches, loc='lower center', ncol=4,
           fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, 0.01))

plt.tight_layout(rect=[0, 0.06, 1, 0.97])
fig3_path = os.path.join(CONFIG['figures_dir'], '03_grid_impact_comparison.png')
plt.savefig(fig3_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved → {fig3_path}")

# ------------------------------------------------------------------
# 图4: 04_charging_infrastructure_requirement.png – 雷达图
# ------------------------------------------------------------------
print("   Generating Figure 4: Charging Infrastructure Requirement Radar …")

# 雷达维度
radar_dims = [
    '快充需求\nFast-Charge\nDemand',
    '慢充需求\nSlow-Charge\nDemand',
    '时间规律\nTiming\nRegularity',
    '夜充偏好\nNight\nPreference',
    '高功率需求\nHigh-Power\nDemand',
]
N = len(radar_dims)

def get_radar_values(vc):
    st = archetype_stats[vc]
    # 归一化到 0-1（相对于所有聚类的最大值）
    return [
        st['fast_charge_ratio_mean'],
        1 - st['fast_charge_ratio_mean'],                         # slow = 1 - fast
        max(0, 1 - st['charge_timing_regularity_mean'] / 3.178),  # 低熵 → 规律; ln(24)≈3.178 is max entropy for 24-hour distribution
        st['night_charge_ratio_mean'],
        min(1, st['peak_soc_rate_mean'] / 100),
    ]

# 全局归一化
all_vals = np.array([get_radar_values(c) for c in clusters])
col_max = all_vals.max(axis=0)
col_max[col_max == 0] = 1

fig, axes = plt.subplots(2, 2, figsize=(12, 12),
                          subplot_kw=dict(polar=True))
fig.suptitle('Charging Infrastructure Requirement by Archetype',
             fontsize=14, fontweight='bold', y=0.99)

angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles_closed = angles + angles[:1]

for ax, vc in zip(axes.flat, clusters):
    raw = get_radar_values(vc)
    vals_norm = (np.array(raw) / col_max).tolist()
    vals_closed = vals_norm + vals_norm[:1]

    ax.plot(angles_closed, vals_closed, color=ARCHETYPE_COLORS[vc],
            linewidth=2, linestyle='solid')
    ax.fill(angles_closed, vals_closed, color=ARCHETYPE_COLORS[vc], alpha=0.25)

    ax.set_xticks(angles)
    ax.set_xticklabels(radar_dims, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.50, 0.75, 1.0])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=7)
    ax.set_title(ARCHETYPE_LABELS.get(vc, f'C{vc}'),
                 pad=20, fontsize=11, fontweight='bold',
                 color=ARCHETYPE_COLORS[vc])
    ax.grid(True, alpha=0.4)

plt.tight_layout()
fig4_path = os.path.join(CONFIG['figures_dir'], '04_charging_infrastructure_requirement.png')
plt.savefig(fig4_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved → {fig4_path}")

# ============================================================
# 6. 输出数据文件
# ============================================================
print(f"\n【STEP 6】Saving Output Files")
print("=" * 80)

# 6.1 charging_characteristics_by_archetype.csv
rows = []
for vc in clusters:
    st = archetype_stats[vc]
    row = {'vehicle_cluster': vc, 'cluster_label': ARCHETYPE_LABELS[vc],
           'n_vehicles': st['n_vehicles']}
    for col in feature_cols:
        row[col + '_mean'] = st[col + '_mean']
        row[col + '_std'] = st[col + '_std']
        row[col + '_median'] = st[col + '_median']
    rows.append(row)

df_stats = pd.DataFrame(rows)
csv_path = os.path.join(CONFIG['output_dir'], 'charging_characteristics_by_archetype.csv')
df_stats.to_csv(csv_path, index=False)
print(f"   ✓ CSV saved → {csv_path}")

# 6.2 archetype_summary.json
summary = {}
for vc in clusters:
    st = archetype_stats[vc]
    summary[ARCHETYPE_LABELS[vc]] = {
        'n_vehicles': int(st['n_vehicles']),
        'charging_frequency_per_day': round(st['charging_frequency_mean'], 3),
        'avg_soc_gain_pct': round(st['avg_soc_gain_mean'], 2),
        'avg_duration_min': round(st['avg_duration_mean'], 2),
        'fast_charge_ratio': round(st['fast_charge_ratio_mean'], 3),
        'night_charge_ratio': round(st['night_charge_ratio_mean'], 3),
        'charge_timing_entropy': round(st['charge_timing_regularity_mean'], 3),
        'peak_soc_rate_pct_per_h': round(st['peak_soc_rate_mean'], 2),
        'avg_power_demand_pct_per_h': round(st['avg_power_demand_mean'], 2),
        'controllability_index': round(st['controllability_mean'], 3),
    }

json_path = os.path.join(CONFIG['output_dir'], 'archetype_summary.json')
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"   ✓ JSON saved → {json_path}")

# 6.3 charging_archetype_integrated_report.txt
def pct(v):
    return f"{v * 100:.1f}%"

report_lines = [
    "=" * 80,
    "FOUR VEHICLE ARCHETYPES CHARGING PATTERN ANALYSIS – INTEGRATED REPORT",
    "Step 14 | Multi-Level EV Analysis",
    "=" * 80,
    "",
    "1. DATA OVERVIEW",
    "-" * 50,
    f"   Total charging events analysed: {len(df):,}",
    f"   Total vehicles analysed:         {df['vehicle_id'].nunique():,}",
    "",
]

report_lines += ["2. CHARGING CHARACTERISTICS BY ARCHETYPE", "-" * 50]
for vc in clusters:
    st = archetype_stats[vc]
    label = ARCHETYPE_LABELS[vc]
    report_lines += [
        f"\n   {label} (n={int(st['n_vehicles']):,} vehicles)",
        f"   {'Charging frequency':35s}: {st['charging_frequency_mean']:.3f} times/day  (±{st['charging_frequency_std']:.3f})",
        f"   {'Avg SOC gain':35s}: {st['avg_soc_gain_mean']:.2f}%  (±{st['avg_soc_gain_std']:.2f}%)",
        f"   {'Avg charge duration':35s}: {st['avg_duration_mean']:.1f} min  (±{st['avg_duration_std']:.1f} min)",
        f"   {'Fast-charge ratio':35s}: {pct(st['fast_charge_ratio_mean'])}",
        f"   {'Night-charge ratio (22-06)':35s}: {pct(st['night_charge_ratio_mean'])}",
        f"   {'Charge timing entropy':35s}: {st['charge_timing_regularity_mean']:.3f}",
        f"   {'Peak SOC rate (95th pct)':35s}: {st['peak_soc_rate_mean']:.2f} %/h",
        f"   {'Avg power demand':35s}: {st['avg_power_demand_mean']:.2f} %/h",
        f"   {'Controllability index':35s}: {st['controllability_mean']:.3f}",
    ]

report_lines += [
    "",
    "3. CHARGING ROLE COMPARISON",
    "-" * 50,
    "",
    "   C0 Mixed-Pattern:      Daily mixed charging; moderate frequency; spread across",
    "                          all hours; medium fast-charge use; moderate grid impact.",
    "",
    "   C1 Highway-Intensive:  Long-distance energy replenishment; high peak SOC rate;",
    "                          fast-charge preference; irregular timing (high entropy);",
    "                          significant grid peak stress.",
    "",
    "   C2 Idle-Stable:        Stationary, long-duration slow charging; high night-charge",
    "                          ratio; low timing entropy (predictable); most controllable",
    "                          and grid-friendly archetype.",
    "",
    "   C3 Urban-Dynamic:      Short, frequent top-ups; scattered timing; moderate fast-",
    "                          charge ratio; high variability; demands dense urban infra.",
    "",
    "4. GRID IMPACT ASSESSMENT",
    "-" * 50,
]

for vc in clusters:
    st = archetype_stats[vc]
    ctrl = st['controllability_mean']
    level = "HIGH" if ctrl > 0.15 else ("MEDIUM" if ctrl > 0.08 else "LOW")
    report_lines.append(
        f"   {ARCHETYPE_LABELS[vc]:<30}: Controllability = {ctrl:.3f}  → {level}"
    )

report_lines += [
    "",
    "5. INFRASTRUCTURE PLANNING RECOMMENDATIONS",
    "-" * 50,
    "",
    "   C0 Mixed-Pattern:",
    "      – Mixed AC/DC charging at workplaces and residential areas.",
    "      – Time-of-Use pricing to shift demand away from peaks.",
    "",
    "   C1 Highway-Intensive:",
    "      – High-power DC fast chargers along major corridors.",
    "      – Battery swap stations for ultra-fast turnaround.",
    "      – Dynamic pricing to smooth highway charging peaks.",
    "",
    "   C2 Idle-Stable:",
    "      – Smart residential AC slow chargers with V2G/V2H capability.",
    "      – Schedulable overnight charging programmes.",
    "      – Priority target for demand-response aggregation.",
    "",
    "   C3 Urban-Dynamic:",
    "      – Dense urban fast-charging network.",
    "      – Mobile charging & reservation systems to reduce queuing.",
    "      – Real-time availability information integration.",
    "",
    "6. ELECTRICITY PRICING RECOMMENDATIONS",
    "-" * 50,
    "",
    "   C0 & C2: Incentivise off-peak (night) charging via TOU tariffs.",
    "   C1:      Corridor pricing tiers; flat rate for highway stations.",
    "   C3:      Real-time dynamic pricing tied to grid load.",
    "",
    "=" * 80,
    "END OF REPORT",
    "=" * 80,
]

report_text = "\n".join(report_lines)
report_path = os.path.join(CONFIG['output_dir'], 'charging_archetype_integrated_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_text)
print(f"   ✓ Report saved → {report_path}")

# ============================================================
# 7. 完成
# ============================================================
print(f"\n{'=' * 80}")
print("✅ STEP 14 COMPLETED SUCCESSFULLY")
print(f"{'=' * 80}")
print(f"""
Output Files:
  vehicle_clustering/results/
  ├── charging_archetype_analysis_integrated/
  │   ├── charging_characteristics_by_archetype.csv
  │   ├── archetype_summary.json
  │   └── charging_archetype_integrated_report.txt
  └── figures_charging_archetypes_integrated/
      ├── 01_four_archetypes_overview.png
      ├── 02_charging_demand_profile.png
      ├── 03_grid_impact_comparison.png
      └── 04_charging_infrastructure_requirement.png
""")
print(report_text)
