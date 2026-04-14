"""
对充电事件进行智能分层过滤
创建多个分析用数据集
适配字段: charging_events_raw_extracted.csv
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['font.size'] = 11

print("=" * 70)
print("🔧 Intelligent Charging Event Filtering")
print("=" * 70)

output_dir = "./coupling_analysis/results/"
df_charging = pd.read_csv(os.path.join(output_dir, 'charging_events_raw_extracted.csv'))

print(f"\n📂 Loaded: {len(df_charging):,} events, {df_charging['vehicle_id'].nunique():,} vehicles")
print(f"   Columns: {list(df_charging.columns)}")

# ============================================================
# 0. 基础字段检查与补全
# ============================================================

# 确保时间列是 datetime
df_charging['start_time'] = pd.to_datetime(df_charging['start_time'], errors='coerce')
df_charging['end_time'] = pd.to_datetime(df_charging['end_time'], errors='coerce')

# 如果 duration_seconds 缺失，从时间差计算
if 'duration_seconds' not in df_charging.columns:
    df_charging['duration_seconds'] = (
        df_charging['end_time'] - df_charging['start_time']
    ).dt.total_seconds()

# 如果 duration_minutes 缺失，从秒数计算
if 'duration_minutes' not in df_charging.columns:
    df_charging['duration_minutes'] = df_charging['duration_seconds'] / 60.0

# 提取充电开始的时间特征（用于后续分析）
df_charging['start_hour'] = df_charging['start_time'].dt.hour
df_charging['start_weekday'] = df_charging['start_time'].dt.dayofweek  # 0=Mon
df_charging['is_night_charge'] = df_charging['start_hour'].isin(
    [22, 23, 0, 1, 2, 3, 4, 5]
).astype(int)

# ============================================================
# 1. 分析充电事件的特征
# ============================================================
print("\n📊 Analyzing charging event characteristics...")

# 按 SOC 增益分类
df_charging['gain_category'] = pd.cut(
    df_charging['soc_gain'],
    bins=[0, 1, 3, 5, 10, 100],
    labels=['<1% (Impulse)', '1-3% (Micro)', '3-5% (Small)',
            '5-10% (Medium)', '10%+ (Full)'],
    include_lowest=True
)

# 按时长分类（使用已有的 duration_minutes）
df_charging['duration_category'] = pd.cut(
    df_charging['duration_minutes'],
    bins=[0, 1, 5, 15, 60, 6000],
    labels=['<1min', '1-5min', '5-15min', '15-60min', '>1hour'],
    include_lowest=True
)

# 按充电类型分类（使用 ch_s_mode 而非 ch_s_type）
CH_S_MAP = {1: '停车充电', 2: '行驶充电'}
df_charging['charge_type_name'] = df_charging['ch_s_mode'].map(CH_S_MAP).fillna('未知')

# 按已有的 fast/slow 分类
if 'charge_type' in df_charging.columns:
    df_charging['speed_label'] = df_charging['charge_type'].map(
        {'fast': '快充 (>1%/min)', 'slow': '慢充 (≤1%/min)'}
    ).fillna('未知')

print(f"\n1️⃣ SOC Gain Distribution:")
print(df_charging['gain_category'].value_counts().sort_index().to_string())

print(f"\n2️⃣ Duration Distribution:")
print(df_charging['duration_category'].value_counts().sort_index().to_string())

print(f"\n3️⃣ Charging Mode (ch_s_mode):")
print(df_charging['charge_type_name'].value_counts().to_string())

if 'charge_type' in df_charging.columns:
    print(f"\n4️⃣ Charging Speed (fast/slow):")
    print(df_charging['speed_label'].value_counts().to_string())

print(f"\n5️⃣ Night Charging (22:00-06:00):")
night_n = df_charging['is_night_charge'].sum()
print(f"   Night: {night_n:,} ({night_n/len(df_charging)*100:.1f}%)")
print(f"   Day:   {len(df_charging)-night_n:,} ({(len(df_charging)-night_n)/len(df_charging)*100:.1f}%)")

# ============================================================
# 2. 异常值检测
# ============================================================
print(f"\n{'=' * 70}")
print("🔍 Anomaly Detection")
print(f"{'=' * 70}")

# 功率异常（充电时功率应为负，绝对值不应超过 250kW）
power_anomaly = df_charging['power_mean'].abs() > 250000
print(f"   |power| > 250kW:     {power_anomaly.sum():,} ({power_anomaly.mean()*100:.2f}%)")

# SOC 增益异常（单次充电 >100% 或 duration_seconds=0 时 gain 很大）
soc_anomaly = df_charging['soc_gain'] > 100
print(f"   SOC gain > 100%:     {soc_anomaly.sum():,}")

# 超长充电（>48小时）
long_anomaly = df_charging['duration_seconds'] > 48 * 3600
print(f"   Duration > 48h:      {long_anomaly.sum():,}")

# 充电速率异常（>5%/min 即 0-100% 在 20 分钟内，不太可能）
if 'avg_soc_rate' in df_charging.columns:
    rate_anomaly = df_charging['avg_soc_rate'] > 5.0
    print(f"   SOC rate > 5%/min:   {rate_anomaly.sum():,}")

# ============================================================
# 3. 创建多个版本的数据集
# ============================================================
print(f"\n{'=' * 70}")
print("📋 Creating filtered datasets for different analyses...")
print(f"{'=' * 70}")

datasets = {}

# 版本 0: 全部数据
datasets['all'] = df_charging.copy()

# 版本 1: 去掉极短充电（<1分钟）和异常值
datasets['no_instant'] = df_charging[
    (df_charging['duration_seconds'] >= 60) &
    (df_charging['duration_seconds'] <= 48 * 3600) &  # 去掉超长
    (~power_anomaly)                                     # 去掉功率异常
].copy()

# 版本 2: 有意义的充电（≥3% SOC + ≥1分钟 + 无异常）
datasets['meaningful'] = df_charging[
    (df_charging['soc_gain'] >= 3) &
    (df_charging['duration_seconds'] >= 60) &
    (df_charging['duration_seconds'] <= 48 * 3600) &
    (~power_anomaly)
].copy()

# 版本 3: 只保留停车充电（ch_s_mode == 1）
datasets['stationary_only'] = df_charging[
    df_charging['ch_s_mode'] == 1
].copy()

# 版本 4: 停车充电 + 有意义
datasets['stationary_meaningful'] = df_charging[
    (df_charging['ch_s_mode'] == 1) &
    (df_charging['soc_gain'] >= 3) &
    (df_charging['duration_seconds'] >= 60) &
    (df_charging['duration_seconds'] <= 48 * 3600)
].copy()

# 版本 5: 行驶充电
datasets['driving_only'] = df_charging[
    df_charging['ch_s_mode'] == 2
].copy()

# 版本 6: 快充事件
if 'charge_type' in df_charging.columns:
    datasets['fast_charge'] = df_charging[
        (df_charging['charge_type'] == 'fast') &
        (df_charging['duration_seconds'] >= 60)
    ].copy()

    datasets['slow_charge'] = df_charging[
        (df_charging['charge_type'] == 'slow') &
        (df_charging['duration_seconds'] >= 60)
    ].copy()

# 保存
print(f"\n💾 Saving datasets...")
for version_name, df in datasets.items():
    filename = f'charging_events_{version_name}.csv'
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    n_v = df['vehicle_id'].nunique() if len(df) > 0 else 0
    print(f"   ✅ {filename:<50} {len(df):>8,} events  {n_v:>6,} vehicles")

# ============================================================
# 4. 统计对比
# ============================================================
print(f"\n{'=' * 70}")
print("📊 Statistical Comparison")
print(f"{'=' * 70}")

rows = []
for name, df in datasets.items():
    if len(df) == 0:
        continue
    rows.append({
        'Dataset': name,
        'Events': f"{len(df):,}",
        'Vehicles': f"{df['vehicle_id'].nunique():,}",
        'Avg Gain': f"{df['soc_gain'].mean():.1f}%",
        'Med Gain': f"{df['soc_gain'].median():.1f}%",
        'Avg Dur(h)': f"{df['duration_minutes'].mean()/60:.2f}",
        'Avg SOC₀': f"{df['soc_start'].mean():.1f}%",
        'Night%': f"{df['is_night_charge'].mean()*100:.1f}%",
    })

df_cmp = pd.DataFrame(rows)
print("\n" + df_cmp.to_string(index=False))

# ============================================================
# 5. 可视化
# ============================================================
print(f"\n📈 Generating visualizations...")

versions = [k for k in datasets.keys() if len(datasets[k]) > 0]
n_versions = len(versions)
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
          '#F7DC6F', '#BB8FCE', '#85C1E9'][:n_versions]

fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# (a) 事件数量
ax = axes[0, 0]
counts = [len(datasets[v]) for v in versions]
bars = ax.bar(range(n_versions), counts, color=colors)
ax.set_xticks(range(n_versions))
ax.set_xticklabels(versions, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Number of Events')
ax.set_title('(a) Event Count by Dataset', fontweight='bold')
for i, (bar, c) in enumerate(zip(bars, counts)):
    ax.text(bar.get_x() + bar.get_width()/2, c, f'{c:,}',
            ha='center', va='bottom', fontsize=8, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

# (b) 车辆覆盖
ax = axes[0, 1]
vcounts = [datasets[v]['vehicle_id'].nunique() for v in versions]
bars = ax.bar(range(n_versions), vcounts, color=colors)
ax.set_xticks(range(n_versions))
ax.set_xticklabels(versions, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Unique Vehicles')
ax.set_title('(b) Vehicle Coverage', fontweight='bold')
for i, (bar, c) in enumerate(zip(bars, vcounts)):
    ax.text(bar.get_x() + bar.get_width()/2, c, f'{c:,}',
            ha='center', va='bottom', fontsize=8, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

# (c) 平均 SOC 增益
ax = axes[0, 2]
gains = [datasets[v]['soc_gain'].mean() for v in versions]
bars = ax.bar(range(n_versions), gains, color=colors)
ax.set_xticks(range(n_versions))
ax.set_xticklabels(versions, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Avg SOC Gain (%)')
ax.set_title('(c) Average Charging Amount', fontweight='bold')
for i, (bar, g) in enumerate(zip(bars, gains)):
    ax.text(bar.get_x() + bar.get_width()/2, g, f'{g:.1f}%',
            ha='center', va='bottom', fontsize=8, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

# (d) SOC 增益分布箱线图
ax = axes[1, 0]
box_data = [datasets[v]['soc_gain'].values for v in versions]
bp = ax.boxplot(box_data, patch_artist=True, showfliers=False, widths=0.6)
for patch, c in zip(bp['boxes'], colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.7)
ax.set_xticklabels(versions, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('SOC Gain (%)')
ax.set_title('(d) SOC Gain Distribution', fontweight='bold')
ax.grid(alpha=0.3, axis='y')

# (e) 充电时长分布（对数）
ax = axes[1, 1]
dur_data = [datasets[v]['duration_minutes'].values / 60 for v in versions]
bp = ax.boxplot(dur_data, patch_artist=True, showfliers=False, widths=0.6)
for patch, c in zip(bp['boxes'], colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.7)
ax.set_xticklabels(versions, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Duration (hours)')
ax.set_title('(e) Duration Distribution', fontweight='bold')
ax.set_yscale('log')
ax.grid(alpha=0.3, axis='y')

# (f) 起始 SOC
ax = axes[1, 2]
socs = [datasets[v]['soc_start'].mean() for v in versions]
bars = ax.bar(range(n_versions), socs, color=colors)
ax.set_xticks(range(n_versions))
ax.set_xticklabels(versions, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Avg Starting SOC (%)')
ax.set_title('(f) Average Trigger SOC', fontweight='bold')
for i, (bar, s) in enumerate(zip(bars, socs)):
    ax.text(bar.get_x() + bar.get_width()/2, s, f'{s:.1f}%',
            ha='center', va='bottom', fontsize=8, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

plt.suptitle('Charging Event Datasets Comparison',
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
fig.savefig(os.path.join(output_dir, 'charging_datasets_comparison.png'),
            dpi=300, bbox_inches='tight')
plt.close(fig)
print("   ✅ Saved: charging_datasets_comparison.png")

# ============================================================
# 6. 建议
# ============================================================
n_meaningful = len(datasets['meaningful'])
v_meaningful = datasets['meaningful']['vehicle_id'].nunique()
n_stat_mean = len(datasets['stationary_meaningful'])
v_stat_mean = datasets['stationary_meaningful']['vehicle_id'].nunique()

print(f"\n{'=' * 70}")
print("📋 RECOMMENDATION FOR COUPLING ANALYSIS")
print(f"{'=' * 70}")
print(f"""
🎯 For your Coupling Analysis, recommended datasets:

  PRIMARY: 'meaningful'
    - Filter: SOC gain ≥ 3% AND duration ≥ 1min AND no anomalies
    - Events: {n_meaningful:,}
    - Vehicles: {v_meaningful:,}
    - Rationale: Removes noise while preserving real charging decisions

  ALTERNATIVE: 'stationary_meaningful'
    - Filter: Stationary charging (ch_s_mode=1) + meaningful
    - Events: {n_stat_mean:,}
    - Vehicles: {v_stat_mean:,}
    - Rationale: Focus on deliberate "stop and charge" behavior

  FAST/SLOW SPLIT: 'fast_charge' / 'slow_charge'
    - For analyzing different charging strategies separately
""")

print("✅ Filtering complete!")