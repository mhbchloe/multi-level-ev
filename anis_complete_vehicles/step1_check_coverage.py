"""
Step 1: Check Vehicle Coverage Across 31 Days
检查每辆车在31天数据中的覆盖情况
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm
import os
import glob

# 创建结果目录
os.makedirs('./analysis_complete_vehicles/results', exist_ok=True)

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

print("="*70)
print("🔍 Vehicle Coverage Check (31 Days)")
print("="*70)

# ============ 自动扫描所有processed文件 ============
print("\n📂 Scanning for processed CSV files...")

csv_files = sorted(glob.glob('./data_*_processed.csv'))

if not csv_files:
    print("❌ No processed CSV files found!")
    print("   Expected: ./data_YYYYMMDD_processed.csv")
    exit()

print(f"✅ Found {len(csv_files)} files")

# 提取日期和创建日期列表
dates = []
for csv_file in csv_files:
    # 从文件名提取日期 data_20250701_processed.csv -> 20250701
    date_str = os.path.basename(csv_file).split('_')[1]
    dates.append(date_str)

dates = sorted(dates)
print(f"   Date range: {dates[0]} to {dates[-1]}")

# 创建日期标签
day_labels = [f"Day{i+1}\n({date})" for i, date in enumerate(dates)]

# ============ 扫描每天的车辆 ============
print(f"\n📊 Scanning vehicle presence in each day...\n")

vehicle_presence = defaultdict(lambda: defaultdict(int))

for file_idx, csv_file in enumerate(tqdm(csv_files, desc="Processing files")):
    date_str = dates[file_idx]
    day_label = f"Day{file_idx+1}"
    
    try:
        # 用chunksize读取，避免内存溢出
        reader = pd.read_csv(
            csv_file,
            chunksize=500_000,
            usecols=['vehicle_id'],
            on_bad_lines='skip'
        )
        
        day_vehicles = set()
        for chunk in reader:
            day_vehicles.update(chunk['vehicle_id'].dropna().unique())
        
        for vehicle_id in day_vehicles:
            vehicle_presence[vehicle_id][day_label] = 1
        
    except Exception as e:
        print(f"   ⚠️  Error in {csv_file}: {str(e)[:40]}")

print(f"\n✅ Scanned {len(csv_files)} days")

# ============ 构建覆盖矩阵 ============
print("\n📊 Building coverage matrix...")

coverage_data = []
for vehicle_id, day_dict in tqdm(vehicle_presence.items(), desc="Building matrix"):
    row = {'vehicle_id': vehicle_id}
    
    # 按顺序添加每一天的数据
    day_count = 0
    for i, day_label in enumerate([f"Day{i+1}" for i in range(len(dates))]):
        presence = day_dict.get(day_label, 0)
        row[day_label] = presence
        day_count += presence
    
    row['total_days'] = day_count
    coverage_data.append(row)

df_coverage = pd.DataFrame(coverage_data)

# ============ 统计 ============
print(f"\n{'='*70}")
print(f"📊 Coverage Statistics (31 Days)")
print(f"{'='*70}")

print(f"\nTotal unique vehicles: {len(df_coverage):,}")

days_dist = df_coverage['total_days'].value_counts().sort_index(ascending=False)
print(f"\n车辆按出现天数分布:")
for n_days, count in days_dist.items():
    pct = count / len(df_coverage) * 100
    bar_width = int(pct / 2)
    bar = '█' * bar_width
    print(f"   {n_days:2d} days: {count:6,} ({pct:5.2f}%) {bar}")

# 统计完整度等级
complete_vehicles = df_coverage[df_coverage['total_days'] == 31]
high_coverage = df_coverage[(df_coverage['total_days'] >= 28) & (df_coverage['total_days'] < 31)]
medium_coverage = df_coverage[(df_coverage['total_days'] >= 20) & (df_coverage['total_days'] < 28)]
low_coverage = df_coverage[df_coverage['total_days'] < 20]

print(f"\n完整度等级:")
print(f"   🟢 完全覆盖 (31 days):     {len(complete_vehicles):,}  ({len(complete_vehicles)/len(df_coverage)*100:5.2f}%)")
print(f"   🟡 高覆盖   (28-30 days): {len(high_coverage):,}  ({len(high_coverage)/len(df_coverage)*100:5.2f}%)")
print(f"   🟠 中覆盖   (20-27 days): {len(medium_coverage):,} ({len(medium_coverage)/len(df_coverage)*100:5.2f}%)")
print(f"   🔴 低覆盖   (<20 days):   {len(low_coverage):,}  ({len(low_coverage)/len(df_coverage)*100:5.2f}%)")

# ============ 保存 ============
print(f"\n💾 Saving results...\n")

df_coverage.to_csv('./analysis_complete_vehicles/results/vehicle_coverage_31days.csv', index=False)
print(f"   ✅ vehicle_coverage_31days.csv - {len(df_coverage):,} vehicles")

complete_vehicles[['vehicle_id']].to_csv('./analysis_complete_vehicles/results/complete_vehicles_31days.csv', index=False)
print(f"   ✅ complete_vehicles_31days.csv - {len(complete_vehicles):,} vehicles (31 days)")

high_coverage_vehicles = pd.concat([complete_vehicles, high_coverage])
high_coverage_vehicles[['vehicle_id']].to_csv('./analysis_complete_vehicles/results/high_coverage_vehicles.csv', index=False)
print(f"   ✅ high_coverage_vehicles.csv - {len(high_coverage_vehicles):,} vehicles (≥28 days)")

medium_plus_vehicles = pd.concat([complete_vehicles, high_coverage, medium_coverage])
medium_plus_vehicles[['vehicle_id']].to_csv('./analysis_complete_vehicles/results/medium_plus_vehicles.csv', index=False)
print(f"   ✅ medium_plus_vehicles.csv - {len(medium_plus_vehicles):,} vehicles (≥20 days)")

# ============ 可视化 ============
print(f"\n📈 Generating visualizations...")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# 图1: 天数分布柱状图
ax1 = fig.add_subplot(gs[0, :])
days_values = sorted(df_coverage['total_days'].unique(), reverse=True)
days_counts = [len(df_coverage[df_coverage['total_days'] == d]) for d in days_values]

# 颜色渐变
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(days_values)))
bars = ax1.bar(range(len(days_values)), days_counts, color=colors, edgecolor='black', alpha=0.8)

ax1.set_xlabel('Number of Days Present', fontsize=11, fontweight='bold')
ax1.set_ylabel('Number of Vehicles', fontsize=11, fontweight='bold')
ax1.set_title('Vehicle Distribution by Coverage Days (31 Days Total)', fontsize=13, fontweight='bold')
ax1.set_xticks(range(len(days_values)))
ax1.set_xticklabels([f"{d}" for d in days_values], rotation=45)
ax1.grid(alpha=0.3, axis='y')

for i, bar in enumerate(bars):
    height = bar.get_height()
    if height > 0:
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\n({height/len(df_coverage)*100:.1f}%)',
                ha='center', va='bottom', fontsize=8, fontweight='bold')

# 图2: 完整性分布饼图
ax2 = fig.add_subplot(gs[1, 0])
categories = ['Complete\n(31 days)', 'High\n(28-30)', 'Medium\n(20-27)', 'Low\n(<20)']
counts_pie = [
    len(complete_vehicles),
    len(high_coverage),
    len(medium_coverage),
    len(low_coverage)
]
colors_pie = ['#27ae60', '#f39c12', '#e67e22', '#e74c3c']

wedges, texts, autotexts = ax2.pie(counts_pie, labels=categories, autopct='%1.1f%%',
                                     colors=colors_pie, startangle=90,
                                     textprops={'fontsize': 9, 'fontweight': 'bold'})
ax2.set_title('Data Completeness Distribution', fontsize=11, fontweight='bold')

# 图3: 累积百分比曲线
ax3 = fig.add_subplot(gs[1, 1])
days_sorted = sorted(df_coverage['total_days'].values, reverse=True)
cumulative_pct = [len([d for d in days_sorted if d >= threshold]) / len(df_coverage) * 100 
                  for threshold in range(1, 32)]

ax3.plot(range(1, 32), cumulative_pct, marker='o', linewidth=2, markersize=6, color='#3498db')
ax3.fill_between(range(1, 32), cumulative_pct, alpha=0.3, color='#3498db')
ax3.set_xlabel('Minimum Days Required', fontsize=11, fontweight='bold')
ax3.set_ylabel('Vehicle Coverage (%)', fontsize=11, fontweight='bold')
ax3.set_title('Cumulative Coverage Curve', fontsize=11, fontweight='bold')
ax3.grid(alpha=0.3)
ax3.set_xlim(0, 31)
ax3.set_ylim(0, 105)

# 添加参考线
for threshold in [20, 28, 31]:
    pct = len([d for d in days_sorted if d >= threshold]) / len(df_coverage) * 100
    ax3.axvline(threshold, color='red', linestyle='--', alpha=0.5)
    ax3.text(threshold, 102, f'{threshold}d\n{pct:.0f}%', ha='center', fontsize=8, fontweight='bold')

# 图4: 覆盖热力图（样本）- 前50辆车
ax4 = fig.add_subplot(gs[2, :])
sample_n = min(50, len(df_coverage))
df_sample = df_coverage.iloc[:sample_n].copy()
df_sample = df_sample.set_index('vehicle_id')

# 只显示第1到30列（Day列）
day_cols = [col for col in df_sample.columns if col.startswith('Day')][:30]
heatmap_data = df_sample[day_cols].values

im = ax4.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', interpolation='nearest')

ax4.set_xlabel('Day', fontsize=11, fontweight='bold')
ax4.set_ylabel(f'Vehicle (First {sample_n})', fontsize=11, fontweight='bold')
ax4.set_title('Coverage Heatmap - Vehicle Presence by Day (Sample)', fontsize=11, fontweight='bold')
ax4.set_xticks(range(0, len(day_cols), 5))
ax4.set_xticklabels([f'D{i+1}' for i in range(0, len(day_cols), 5)])

# 添加colorbar
cbar = plt.colorbar(im, ax=ax4, label='Present (1) / Absent (0)')

plt.savefig('./analysis_complete_vehicles/results/vehicle_coverage_31days_detailed.png', dpi=300, bbox_inches='tight')
print(f"   ✅ vehicle_coverage_31days_detailed.png")

# 简化版本 - 只看统计
fig2, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：等级分布
ax = axes[0]
level_names = ['Complete\n(31)', 'High\n(28-30)', 'Medium\n(20-27)', 'Low\n(<20)']
level_counts = [len(complete_vehicles), len(high_coverage), len(medium_coverage), len(low_coverage)]
colors_bar = ['#27ae60', '#f39c12', '#e67e22', '#e74c3c']

bars = ax.barh(level_names, level_counts, color=colors_bar, edgecolor='black', alpha=0.8)
ax.set_xlabel('Number of Vehicles', fontsize=11, fontweight='bold')
ax.set_title('Vehicle Distribution by Coverage Level', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3, axis='x')

for i, (bar, count) in enumerate(zip(bars, level_counts)):
    ax.text(count, bar.get_y() + bar.get_height()/2, 
            f' {count:,} ({count/len(df_coverage)*100:.1f}%)',
            va='center', fontsize=10, fontweight='bold')

# 右图：推荐方案
ax = axes[1]
ax.axis('off')

recommendations = f"""
ANALYSIS RECOMMENDATIONS

📊 Data Overview:
   • Total unique vehicles: {len(df_coverage):,}
   • Data period: 31 days
   • Complete coverage vehicles: {len(complete_vehicles):,}

🎯 Recommended Approach:

   Option 1 (STRICT):
   ✓ Use: {len(complete_vehicles):,} vehicles (31 days)
   ✗ Pros: Perfect data quality
   ✗ Cons: Smaller sample size

   Option 2 (RECOMMENDED):
   ✓ Use: {len(high_coverage_vehicles):,} vehicles (≥28 days)
   ✗ Pros: Good quality + larger sample
   ✗ Cons: Minor data gaps

   Option 3 (INCLUSIVE):
   ✓ Use: {len(medium_plus_vehicles):,} vehicles (≥20 days)
   ✗ Pros: Largest sample size
   ✗ Cons: Data gaps may affect analysis

📁 Output Files:
   • complete_vehicles_31days.csv
   • high_coverage_vehicles.csv
   • medium_plus_vehicles.csv
"""

ax.text(0.05, 0.95, recommendations, transform=ax.transAxes,
        fontsize=10, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('./analysis_complete_vehicles/results/vehicle_coverage_summary.png', dpi=300, bbox_inches='tight')
print(f"   ✅ vehicle_coverage_summary.png")

plt.close('all')

# ============ 最终建议 ============
print(f"\n{'='*70}")
print(f"💡 ANALYSIS RECOMMENDATION")
print(f"{'='*70}")

high_coverage_pct = len(high_coverage_vehicles) / len(df_coverage) * 100

if len(complete_vehicles) / len(df_coverage) >= 0.5:
    recommendation = "STRICT"
    rec_vehicles = len(complete_vehicles)
    rec_file = "complete_vehicles_31days.csv"
elif high_coverage_pct >= 0.6:
    recommendation = "RECOMMENDED"
    rec_vehicles = len(high_coverage_vehicles)
    rec_file = "high_coverage_vehicles.csv"
else:
    recommendation = "INCLUSIVE"
    rec_vehicles = len(medium_plus_vehicles)
    rec_file = "medium_plus_vehicles.csv"

print(f"\n✅ RECOMMENDED: {recommendation} Approach")
print(f"   Use: {rec_vehicles:,} vehicles")
print(f"   File: {rec_file}")
print(f"   Coverage: {rec_vehicles/len(df_coverage)*100:.1f}% of all vehicles")

print(f"\n{'='*70}")
print(f"✅ Step 1 Complete!")
print(f"{'='*70}")
print(f"\n🚀 Next Steps:")
print(f"   1. Review the generated CSV files")
print(f"   2. Choose your vehicle list (complete/high/medium)")
print(f"   3. Proceed to filtering and analysis")