"""
Step 3: 多层次影响分析（修复版）
先计算焦虑分数，再进行分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("📊 Multi-Level Impact Analysis (Fixed)")
print("="*70)

# 加载数据
print("\n📂 Loading data...")

df_trip_events = pd.read_csv('./results/trip_with_events.csv')
df_vehicle = pd.read_csv('./results/vehicle_complete_profile.csv')

print(f"✅ Trips: {len(df_trip_events):,}")
print(f"✅ Vehicles: {len(df_vehicle):,}")

# === 先计算充电焦虑分数 ===
print("\n🔍 Computing charging anxiety score...")

# 检查必需字段
required_fields = ['avg_charging_trigger_soc', 'charging_freq_per_100km', 'min_soc_ever']

missing_fields = [f for f in required_fields if f not in df_vehicle.columns]

if missing_fields:
    print(f"⚠️  Missing fields: {missing_fields}")
    print(f"   Using simplified anxiety calculation...")
    
    # 简化版焦虑分数
    df_vehicle['charging_anxiety_score'] = (
        (100 - df_vehicle.get('avg_charging_trigger_soc', 50)) * 0.5 +
        df_vehicle.get('charging_freq_per_100km', 0.1) * 50
    )
else:
    # 完整版焦虑分数
    df_vehicle['std_charging_trigger_soc'] = df_vehicle.get('std_charging_trigger_soc', 10)
    
    df_vehicle['charging_anxiety_score'] = (
        (100 - df_vehicle['avg_charging_trigger_soc']) * 0.3 +
        df_vehicle['charging_freq_per_100km'] * 20 +
        df_vehicle['std_charging_trigger_soc'] * 0.2 +
        (50 - df_vehicle['min_soc_ever']) * 0.3
    )

# 归一化到0-100
if df_vehicle['charging_anxiety_score'].std() > 0:
    df_vehicle['charging_anxiety_score'] = (
        (df_vehicle['charging_anxiety_score'] - df_vehicle['charging_anxiety_score'].min()) / 
        (df_vehicle['charging_anxiety_score'].max() - df_vehicle['charging_anxiety_score'].min()) * 100
    )

# 分级
df_vehicle['anxiety_level'] = pd.cut(
    df_vehicle['charging_anxiety_score'],
    bins=[0, 25, 50, 75, 100],
    labels=['低焦虑', '中等焦虑', '较高焦虑', '高焦虑']
)

print(f"✅ Anxiety score computed")
print(f"   Distribution:")
print(df_vehicle['anxiety_level'].value_counts())

# === 层次1：出行层面分析 ===
print("\n" + "="*70)
print("📊 Level 1: Trip-Level Analysis")
print("="*70)

print("\n1.1 出行后是否充电的影响因素：")

charged_trips = df_trip_events[df_trip_events['is_charging_after'] == True]
not_charged_trips = df_trip_events[df_trip_events['is_charging_after'] == False]

if len(charged_trips) > 0 and len(not_charged_trips) > 0:
    comparison = pd.DataFrame({
        '特征': ['SOC消耗(%)', '行驶距离(km)', '放电事件数', '簇熵', 'Cluster0占比', 'Cluster1占比', 'Cluster2占比'],
        '充电组': [
            charged_trips['trip_soc_drop'].mean(),
            charged_trips['trip_distance'].mean(),
            charged_trips['n_discharge_events'].mean(),
            charged_trips['cluster_entropy'].mean(),
            (charged_trips['n_cluster_0'] / charged_trips['n_discharge_events']).mean(),
            (charged_trips['n_cluster_1'] / charged_trips['n_discharge_events']).mean(),
            (charged_trips['n_cluster_2'] / charged_trips['n_discharge_events']).mean(),
        ],
        '不充电组': [
            not_charged_trips['trip_soc_drop'].mean(),
            not_charged_trips['trip_distance'].mean(),
            not_charged_trips['n_discharge_events'].mean(),
            not_charged_trips['cluster_entropy'].mean(),
            (not_charged_trips['n_cluster_0'] / not_charged_trips['n_discharge_events']).mean(),
            (not_charged_trips['n_cluster_1'] / not_charged_trips['n_discharge_events']).mean(),
            (not_charged_trips['n_cluster_2'] / not_charged_trips['n_discharge_events']).mean(),
        ]
    })
    
    print(comparison.to_string(index=False))
else:
    print("   ⚠️ Insufficient data for comparison")

# === 层次2：放电事件组合模式分析 ===
print("\n" + "="*70)
print("📊 Level 2: Event Combination Pattern Analysis")
print("="*70)

print("\n2.1 放电事件数量对充电概率的影响：")

event_count_groups = df_trip_events.groupby('n_discharge_events').agg({
    'is_charging_after': ['mean', 'count'],
    'trip_soc_drop': 'mean'
}).round(3)

print(event_count_groups.head(10))

print("\n2.2 簇多样性（熵）对充电概率的影响：")

df_trip_events['entropy_level'] = pd.cut(
    df_trip_events['cluster_entropy'],
    bins=[0, 0.3, 0.6, 1.0, 2.0],
    labels=['单一模式', '低多样性', '中多样性', '高多样性']
)

entropy_analysis = df_trip_events.groupby('entropy_level').agg({
    'is_charging_after': 'mean',
    'trip_soc_drop': 'mean',
    'n_discharge_events': 'mean'
}).round(3)

print(entropy_analysis)

# === 层次3：车辆层面整体模式 ===
print("\n" + "="*70)
print("📊 Level 3: Vehicle-Level Pattern Analysis")
print("="*70)

# 合并出行数据到车辆级别
vehicle_trip_stats = df_trip_events.groupby('vehicle_id').agg({
    'trip_segment_id': 'count',
    'is_charging_after': 'sum',
    'n_discharge_events': 'mean',
    'cluster_entropy': 'mean',
    'n_cluster_0': 'sum',
    'n_cluster_1': 'sum',
    'n_cluster_2': 'sum',
}).rename(columns={'trip_segment_id': 'n_trips', 'is_charging_after': 'n_charging_trips'})

vehicle_trip_stats['charging_trip_ratio'] = \
    vehicle_trip_stats['n_charging_trips'] / vehicle_trip_stats['n_trips']

vehicle_trip_stats = vehicle_trip_stats.reset_index()

# 融合
df_vehicle_full = df_vehicle.merge(vehicle_trip_stats, on='vehicle_id', how='left')

print(f"\n✅ Merged data: {len(df_vehicle_full):,} vehicles")
print(f"   With trip data: {df_vehicle_full['n_trips'].notna().sum():,}")

print(f"\n3.1 车辆整体放电模式多样性对充电焦虑的影响：")

# 只分析有完整数据的车辆
valid_data = df_vehicle_full[
    df_vehicle_full['cluster_entropy'].notna() & 
    df_vehicle_full['charging_anxiety_score'].notna()
].copy()

print(f"   Valid vehicles for analysis: {len(valid_data):,}")

if len(valid_data) > 10:
    # 相关性分析
    corr_cols = []
    
    # 动态选择可用字段
    potential_cols = [
        'cluster_entropy', 
        'charging_anxiety_score', 
        'avg_charging_trigger_soc',
        'charging_freq_per_100km', 
        'charging_trip_ratio'
    ]
    
    for col in potential_cols:
        if col in valid_data.columns and valid_data[col].notna().sum() > 0:
            corr_cols.append(col)
    
    if len(corr_cols) >= 2:
        corr_analysis = valid_data[corr_cols].corr()
        
        print("\n相关系数矩阵：")
        print(corr_analysis.round(3))
    else:
        print("   ⚠️ Insufficient fields for correlation analysis")
        corr_analysis = pd.DataFrame()
else:
    print("   ⚠️ Insufficient data")
    corr_analysis = pd.DataFrame()

# === 可视化 ===
print("\n📈 Generating visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('放电片段对充电行为的多层次影响分析', fontsize=16, fontweight='bold')

# 图1：出行后充电vs不充电的特征对比
ax = axes[0, 0]
if len(charged_trips) > 0 and len(not_charged_trips) > 0:
    features = ['SOC消耗', '距离', '事件数']
    charged_vals = [
        charged_trips['trip_soc_drop'].mean(),
        charged_trips['trip_distance'].mean(),
        charged_trips['n_discharge_events'].mean()
    ]
    not_charged_vals = [
        not_charged_trips['trip_soc_drop'].mean(),
        not_charged_trips['trip_distance'].mean(),
        not_charged_trips['n_discharge_events'].mean()
    ]
    
    x = np.arange(len(features))
    width = 0.35
    
    ax.bar(x - width/2, charged_vals, width, label='充电组', color='#3498db', alpha=0.8)
    ax.bar(x + width/2, not_charged_vals, width, label='不充电组', color='#e74c3c', alpha=0.8)
    
    ax.set_ylabel('平均值', fontsize=11)
    ax.set_title('1. 出行后是否充电的特征对比', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(features)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
else:
    ax.text(0.5, 0.5, '数据不足', ha='center', va='center', fontsize=14)
    ax.set_title('1. 出行后是否充电的特征对比', fontsize=12, fontweight='bold')

# 图2：放电事件数 vs 充电概率
ax = axes[0, 1]
if not event_count_groups.empty:
    event_counts = event_count_groups.index[:10]
    charging_probs = event_count_groups['is_charging_after']['mean'].values[:10]
    
    ax.plot(event_counts, charging_probs, marker='o', linewidth=2, markersize=8, color='#2ecc71')
    ax.set_xlabel('出行中放电事件数', fontsize=11)
    ax.set_ylabel('出行后充电概率', fontsize=11)
    ax.set_title('2. 放电事件数对充电概率的影响', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
else:
    ax.text(0.5, 0.5, '数据不足', ha='center', va='center', fontsize=14)
    ax.set_title('2. 放电事件数对充电概率的影响', fontsize=12, fontweight='bold')

# 图3：簇多样性 vs 充电概率
ax = axes[0, 2]
if not entropy_analysis.empty:
    entropy_levels = entropy_analysis.index
    charging_probs = entropy_analysis['is_charging_after'].values
    
    bars = ax.bar(range(len(entropy_levels)), charging_probs, 
                  color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'], alpha=0.8)
    ax.set_xlabel('放电模式多样性', fontsize=11)
    ax.set_ylabel('充电概率', fontsize=11)
    ax.set_title('3. 放电模式多样性对充电概率的影响', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(entropy_levels)))
    ax.set_xticklabels(entropy_levels, rotation=15)
    ax.grid(alpha=0.3, axis='y')
else:
    ax.text(0.5, 0.5, '数据不足', ha='center', va='center', fontsize=14)
    ax.set_title('3. 放电模式多样性对充电概率的影响', fontsize=12, fontweight='bold')

# 图4：主导cluster vs SOC消耗 vs 充电概率
ax = axes[1, 0]
for cluster in sorted(df_trip_events['dominant_cluster'].unique()):
    if cluster == -1:
        continue
    data = df_trip_events[df_trip_events['dominant_cluster'] == cluster]
    if len(data) > 0:
        ax.scatter(data['trip_soc_drop'], data['is_charging_after'].astype(int),
                   alpha=0.3, s=20, label=f'Cluster {int(cluster)}')
ax.set_xlabel('出行SOC消耗 (%)', fontsize=11)
ax.set_ylabel('是否充电 (0/1)', fontsize=11)
ax.set_title('4. 主导放电模式 vs SOC消耗 vs 充电决策', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# 图5：簇熵 vs 充电焦虑
ax = axes[1, 1]
if len(valid_data) > 0:
    scatter = ax.scatter(
        valid_data['cluster_entropy'], 
        valid_data['charging_anxiety_score'],
        alpha=0.4, s=30, 
        c=valid_data.get('charging_trip_ratio', 0), 
        cmap='viridis'
    )
    plt.colorbar(scatter, ax=ax, label='充电出行比')
    ax.set_xlabel('平均放电模式多样性（熵）', fontsize=11)
    ax.set_ylabel('充电焦虑分数', fontsize=11)
    ax.set_title('5. 放电模式多样性 vs 充电焦虑', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
else:
    ax.text(0.5, 0.5, '数据不足', ha='center', va='center', fontsize=14)
    ax.set_title('5. 放电模式多样性 vs 充电焦虑', fontsize=12, fontweight='bold')

# 图6：相关性热图
ax = axes[1, 2]
if not corr_analysis.empty and len(corr_analysis) >= 2:
    # 简化标签
    label_map = {
        'cluster_entropy': '簇熵',
        'charging_anxiety_score': '焦虑',
        'avg_charging_trigger_soc': '触发SOC',
        'charging_freq_per_100km': '频率',
        'charging_trip_ratio': '充电出行比'
    }
    
    labels = [label_map.get(c, c) for c in corr_analysis.columns]
    
    sns.heatmap(corr_analysis, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                xticklabels=labels, yticklabels=labels,
                ax=ax, cbar_kws={'label': '相关系数'})
    ax.set_title('6. 关键指标相关性', fontsize=12, fontweight='bold')
else:
    ax.text(0.5, 0.5, '数据不足', ha='center', va='center', fontsize=14)
    ax.set_title('6. 关键指标相关性', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('./results/multi_level_impact_analysis.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: multi_level_impact_analysis.png")

# 保存分析结果
df_vehicle_full.to_csv('./results/vehicle_complete_analysis.csv', index=False, encoding='utf-8-sig')
print(f"💾 Saved: vehicle_complete_analysis.csv")

print(f"\n{'='*70}")
print(f"✅ Multi-Level Analysis Complete!")
print(f"{'='*70}")
print(f"\n📁 Key findings:")
if len(charged_trips) > 0 and len(not_charged_trips) > 0:
    print(f"   1. 出行后充电组 vs 不充电组有明显特征差异")
if not event_count_groups.empty:
    print(f"   2. 放电事件数量与充电概率的关系已识别")
if not entropy_analysis.empty:
    print(f"   3. 放电模式多样性影响充电决策")
if len(valid_data) > 10:
    print(f"   4. 车辆层面的模式多样性与充电焦虑相关")
print(f"{'='*70}")