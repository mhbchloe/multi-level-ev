"""
Check Vehicle Data Coverage Across 7 Days
检查每辆车在7天数据中的覆盖情况
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

print("="*70)
print("🔍 Vehicle Data Coverage Check (7 Days)")
print("="*70)

# ============ 1. 定义数据文件 ============
csv_files = [
    '20250701.csv',
    '20250702.csv',
    '20250703.csv',
    '20250704.csv',
    '20250705.csv',
    '20250706.csv',
    '20250707.csv'
]

days = ['Day1', 'Day2', 'Day3', 'Day4', 'Day5', 'Day6', 'Day7']

# ============ 2. 统计每天的车辆 ============
print("\n📊 Scanning vehicle presence in each day...")

vehicle_presence = defaultdict(lambda: defaultdict(int))  # {vehicle_id: {day: count}}
daily_stats = {}

for day_idx, csv_file in enumerate(csv_files):
    day_name = days[day_idx]
    print(f"\n[{day_idx+1}/7] Processing: {csv_file}")
    
    try:
        # 只读取vehicle_id列（快速）
        chunk_iter = pd.read_csv(
            csv_file,
            chunksize=1_000_000,
            usecols=['vehicle_id'],
            on_bad_lines='skip',
            low_memory=False
        )
        
        day_vehicles = set()
        total_records = 0
        
        for chunk in chunk_iter:
            total_records += len(chunk)
            
            # 统计该chunk中的车辆
            for vehicle_id in chunk['vehicle_id'].unique():
                vehicle_presence[vehicle_id][day_name] = 1
                day_vehicles.add(vehicle_id)
        
        daily_stats[day_name] = {
            'file': csv_file,
            'vehicles': len(day_vehicles),
            'records': total_records
        }
        
        print(f"   ✅ Unique vehicles: {len(day_vehicles):,}")
        print(f"   ✅ Total records: {total_records:,}")
        
    except Exception as e:
        print(f"   ⚠️  Error: {str(e)}")
        daily_stats[day_name] = {'file': csv_file, 'vehicles': 0, 'records': 0}

# ============ 3. 构建车辆覆盖矩阵 ============
print(f"\n{'='*70}")
print(f"📊 Building Vehicle Coverage Matrix")
print(f"{'='*70}")

# 转换为DataFrame
coverage_data = []

for vehicle_id, day_dict in tqdm(vehicle_presence.items(), desc="Building matrix"):
    row = {'vehicle_id': vehicle_id}
    for day in days:
        row[day] = day_dict.get(day, 0)
    row['total_days'] = sum(row[day] for day in days)
    coverage_data.append(row)

df_coverage = pd.DataFrame(coverage_data)

print(f"\n✅ Coverage matrix built")
print(f"   Total unique vehicles: {len(df_coverage):,}")

# ============ 4. 统计分析 ============
print(f"\n{'='*70}")
print(f"📊 Coverage Statistics")
print(f"{'='*70}")

# 按天数统计
days_distribution = df_coverage['total_days'].value_counts().sort_index()

print(f"\n车辆按出现天数分布:")
for n_days, count in days_distribution.items():
    percentage = count / len(df_coverage) * 100
    print(f"   {n_days} days: {count:,} vehicles ({percentage:.2f}%)")

# 完整数据的车辆
complete_vehicles = df_coverage[df_coverage['total_days'] == 7]
print(f"\n✅ Vehicles with complete 7-day data: {len(complete_vehicles):,} ({len(complete_vehicles)/len(df_coverage)*100:.2f}%)")

# 部分数据的车辆
partial_vehicles = df_coverage[df_coverage['total_days'] < 7]
print(f"⚠️  Vehicles with partial data: {len(partial_vehicles):,} ({len(partial_vehicles)/len(df_coverage)*100:.2f}%)")

# 单天数据的车辆
single_day_vehicles = df_coverage[df_coverage['total_days'] == 1]
print(f"⚠️  Vehicles with only 1 day: {len(single_day_vehicles):,} ({len(single_day_vehicles)/len(df_coverage)*100:.2f}%)")

# ============ 5. 每日车辆数变化 ============
print(f"\n{'='*70}")
print(f"📊 Daily Vehicle Count")
print(f"{'='*70}")

print(f"\n每日车辆数:")
for day in days:
    count = (df_coverage[day] == 1).sum()
    print(f"   {day}: {count:,} vehicles")

# ============ 6. 缺失模式分析 ============
print(f"\n{'='*70}")
print(f"📊 Missing Pattern Analysis")
print(f"{'='*70}")

# 统计最常见的缺失模式
missing_patterns = defaultdict(int)

for _, row in df_coverage.iterrows():
    pattern = tuple([1 if row[day] == 1 else 0 for day in days])
    missing_patterns[pattern] += 1

# 排序
sorted_patterns = sorted(missing_patterns.items(), key=lambda x: x[1], reverse=True)

print(f"\n最常见的10种数据模式 (1=有数据, 0=缺失):")
print(f"   Pattern (D1 D2 D3 D4 D5 D6 D7) | Count")
print(f"   " + "-"*50)

for pattern, count in sorted_patterns[:10]:
    pattern_str = ' '.join([str(p) for p in pattern])
    percentage = count / len(df_coverage) * 100
    print(f"   {pattern_str}              | {count:,} ({percentage:.2f}%)")

# ============ 7. 保存结果 ============
print(f"\n💾 Saving results...")

# 保存完整覆盖矩阵
df_coverage.to_csv('./results/vehicle_coverage_7days.csv', index=False, encoding='utf-8-sig')
print(f"✅ Saved: vehicle_coverage_7days.csv")

# 保存完整数据的车辆列表
complete_vehicles[['vehicle_id']].to_csv('./results/vehicles_complete_7days.csv', 
                                          index=False, encoding='utf-8-sig')
print(f"✅ Saved: vehicles_complete_7days.csv ({len(complete_vehicles):,} vehicles)")

# 保存每日统计
df_daily_stats = pd.DataFrame(daily_stats).T
df_daily_stats.to_csv('./results/daily_vehicle_stats.csv', encoding='utf-8-sig')
print(f"✅ Saved: daily_vehicle_stats.csv")

# ============ 8. 可视化 ============
print(f"\n📈 Generating visualizations...")

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# === Plot 1: 车辆按天数分布 ===
ax = fig.add_subplot(gs[0, 0])

colors = ['#e74c3c', '#e67e22', '#f39c12', '#f1c40f', '#2ecc71', '#3498db', '#9b59b6']
bars = ax.bar(days_distribution.index, days_distribution.values, 
              color=colors[:len(days_distribution)], alpha=0.8, edgecolor='black', linewidth=1.5)

ax.set_xlabel('Number of Days Present', fontsize=11, fontweight='bold')
ax.set_ylabel('Number of Vehicles', fontsize=11, fontweight='bold')
ax.set_title('Vehicle Distribution by Days Present', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

# 添加数值标签
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height):,}\n({height/len(df_coverage)*100:.1f}%)',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

# === Plot 2: 每日车辆数变化 ===
ax = fig.add_subplot(gs[0, 1])

daily_counts = [daily_stats[day]['vehicles'] for day in days]

ax.plot(days, daily_counts, marker='o', linewidth=3, markersize=10, color='#3498db')
ax.fill_between(range(len(days)), daily_counts, alpha=0.3, color='#3498db')

ax.set_xlabel('Day', fontsize=11, fontweight='bold')
ax.set_ylabel('Number of Vehicles', fontsize=11, fontweight='bold')
ax.set_title('Daily Vehicle Count', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)

# 添加数值标签
for i, (day, count) in enumerate(zip(days, daily_counts)):
    ax.text(i, count + max(daily_counts)*0.02, f'{count:,}', 
            ha='center', va='bottom', fontsize=9, fontweight='bold')

# === Plot 3: 完整vs部分数据 ===
ax = fig.add_subplot(gs[0, 2])

categories = ['Complete\n(7 days)', 'Partial\n(<7 days)', 'Single\nDay']
counts = [len(complete_vehicles), len(partial_vehicles), len(single_day_vehicles)]
colors_pie = ['#2ecc71', '#f39c12', '#e74c3c']

wedges, texts, autotexts = ax.pie(counts, labels=categories, autopct='%1.1f%%',
                                    colors=colors_pie, startangle=90,
                                    textprops={'fontsize': 10, 'fontweight': 'bold'})

ax.set_title('Data Completeness Distribution', fontsize=12, fontweight='bold')

# === Plot 4: 覆盖热力图（随机抽样1000辆车） ===
ax = fig.add_subplot(gs[1, :])

# 随机抽样展示
sample_size = min(1000, len(df_coverage))
df_sample = df_coverage.sample(n=sample_size, random_state=42)

# 构建热力图数据
heatmap_data = df_sample[days].values

im = ax.imshow(heatmap_data, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)

ax.set_xticks(range(len(days)))
ax.set_xticklabels(days, fontsize=10)
ax.set_ylabel(f'Vehicles (sampled {sample_size:,})', fontsize=11, fontweight='bold')
ax.set_title('Vehicle Data Presence Heatmap (Green=Present, Red=Missing)', 
             fontsize=12, fontweight='bold')

plt.colorbar(im, ax=ax, label='Present (1) / Missing (0)')

# === Plot 5: 每日记录数 ===
ax = fig.add_subplot(gs[2, 0])

daily_records = [daily_stats[day]['records'] for day in days]

ax.bar(days, daily_records, color='#9b59b6', alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_xlabel('Day', fontsize=11, fontweight='bold')
ax.set_ylabel('Number of Records', fontsize=11, fontweight='bold')
ax.set_title('Daily GPS Record Count', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3, axis='y')
ax.tick_params(axis='x', rotation=45)

# 添加数值
for i, (day, count) in enumerate(zip(days, daily_records)):
    ax.text(i, count + max(daily_records)*0.02, f'{count/1e6:.2f}M', 
            ha='center', va='bottom', fontsize=8, fontweight='bold')

# === Plot 6: 缺失模式Top 10 ===
ax = fig.add_subplot(gs[2, 1:])

top_patterns = sorted_patterns[:10]
pattern_labels = [' '.join([str(p) for p in pattern]) for pattern, _ in top_patterns]
pattern_counts = [count for _, count in top_patterns]

ax.barh(range(len(pattern_labels)), pattern_counts, color='#34495e', alpha=0.8)
ax.set_yticks(range(len(pattern_labels)))
ax.set_yticklabels(pattern_labels, fontsize=8, family='monospace')
ax.set_xlabel('Number of Vehicles', fontsize=11, fontweight='bold')
ax.set_ylabel('Pattern (D1 D2 D3 D4 D5 D6 D7)', fontsize=11, fontweight='bold')
ax.set_title('Top 10 Data Presence Patterns\n(1=Present, 0=Missing)', 
             fontsize=12, fontweight='bold')
ax.grid(alpha=0.3, axis='x')
ax.invert_yaxis()

# 添加数值
for i, count in enumerate(pattern_counts):
    ax.text(count + max(pattern_counts)*0.01, i, f'{count:,}', 
            va='center', fontsize=8, fontweight='bold')

plt.suptitle('Vehicle Data Coverage Analysis (7 Days)', fontsize=16, fontweight='bold', y=0.995)

plt.savefig('./results/vehicle_coverage_analysis.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: vehicle_coverage_analysis.png")

# ============ 9. 生成文本报告 ============
print(f"\n{'='*70}")
print(f"📄 Generating Text Report")
print(f"{'='*70}")

report = []
report.append("="*70)
report.append("Vehicle Data Coverage Report (7 Days)")
report.append("="*70)
report.append("")

report.append("1. Overall Statistics")
report.append("-"*70)
report.append(f"Total unique vehicles: {len(df_coverage):,}")
report.append(f"Vehicles with complete 7-day data: {len(complete_vehicles):,} ({len(complete_vehicles)/len(df_coverage)*100:.2f}%)")
report.append(f"Vehicles with partial data: {len(partial_vehicles):,} ({len(partial_vehicles)/len(df_coverage)*100:.2f}%)")
report.append("")

report.append("2. Distribution by Days Present")
report.append("-"*70)
for n_days, count in days_distribution.items():
    percentage = count / len(df_coverage) * 100
    report.append(f"{n_days} days: {count:,} vehicles ({percentage:.2f}%)")
report.append("")

report.append("3. Daily Vehicle Count")
report.append("-"*70)
for day in days:
    count = daily_stats[day]['vehicles']
    records = daily_stats[day]['records']
    report.append(f"{day} ({daily_stats[day]['file']}): {count:,} vehicles, {records:,} records")
report.append("")

report.append("4. Top 10 Data Presence Patterns")
report.append("-"*70)
report.append("Pattern (D1 D2 D3 D4 D5 D6 D7) | Count | Percentage")
report.append("-"*70)
for pattern, count in sorted_patterns[:10]:
    pattern_str = ' '.join([str(p) for p in pattern])
    percentage = count / len(df_coverage) * 100
    report.append(f"{pattern_str}              | {count:,} | {percentage:.2f}%")
report.append("")

report.append("="*70)
report.append("Recommendation:")
report.append("="*70)
if len(complete_vehicles) / len(df_coverage) >= 0.7:
    report.append("✅ Good data quality: >70% vehicles have complete 7-day data")
    report.append("   Recommendation: Use complete vehicles for primary analysis")
elif len(complete_vehicles) / len(df_coverage) >= 0.5:
    report.append("⚠️  Moderate data quality: 50-70% vehicles have complete data")
    report.append("   Recommendation: Use complete vehicles, supplement with partial data")
else:
    report.append("⚠️  Poor data quality: <50% vehicles have complete data")
    report.append("   Recommendation: Consider using all available data with weighting")

report_text = "\n".join(report)
print(report_text)

with open('./results/vehicle_coverage_report.txt', 'w', encoding='utf-8') as f:
    f.write(report_text)

print(f"\n💾 Saved: vehicle_coverage_report.txt")

print(f"\n{'='*70}")
print(f"✅ Coverage Check Complete!")
print(f"{'='*70}")
print(f"\n📁 Generated files:")
print(f"   1. vehicle_coverage_7days.csv - 完整覆盖矩阵")
print(f"   2. vehicles_complete_7days.csv - 完整7天数据的车辆列表")
print(f"   3. daily_vehicle_stats.csv - 每日统计")
print(f"   4. vehicle_coverage_analysis.png - 可视化分析")
print(f"   5. vehicle_coverage_report.txt - 文本报告")
print(f"{'='*70}")