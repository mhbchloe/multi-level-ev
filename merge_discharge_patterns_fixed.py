"""
Phase 2: 融合放电模式（修复版）
事件表只包含放电事件，没有event_type字段
"""

import pandas as pd
import numpy as np

print("="*70)
print("🔗 Phase 2: Merge Discharge Patterns with Charging Behavior")
print("="*70)

# 加载数据
print("\n📂 Loading data...")

df_vehicle = pd.read_csv('./results/vehicle_behavior_features.csv')
df_events = pd.read_csv('./results/event_table.csv')

print(f"✅ Vehicle features: {len(df_vehicle):,}")
print(f"✅ Events (all discharge): {len(df_events):,}")

# 检查列名
print(f"\n📋 Event table columns:")
print(f"   {list(df_events.columns)}")

# 计算每辆车的放电模式占比
print("\n📊 Computing discharge pattern distribution...")

vehicle_patterns = df_events.groupby(['vehicle_id', 'cluster']).size().unstack(fill_value=0)
vehicle_patterns.columns = [f'cluster_{int(c)}_count' for c in vehicle_patterns.columns]
vehicle_patterns['total_discharge_events'] = vehicle_patterns.sum(axis=1)

print(f"   Pattern columns: {list(vehicle_patterns.columns)}")

# 计算占比
for cluster_id in [0, 1, 2]:
    col_name = f'cluster_{cluster_id}_count'
    if col_name in vehicle_patterns.columns:
        vehicle_patterns[f'cluster_{cluster_id}_ratio'] = \
            vehicle_patterns[col_name] / vehicle_patterns['total_discharge_events']
    else:
        vehicle_patterns[f'cluster_{cluster_id}_ratio'] = 0

# 主导模式（占比最高的cluster）
ratio_cols = ['cluster_0_ratio', 'cluster_1_ratio', 'cluster_2_ratio']
vehicle_patterns['dominant_cluster'] = vehicle_patterns[ratio_cols].idxmax(axis=1)
vehicle_patterns['dominant_cluster'] = vehicle_patterns['dominant_cluster'].str.extract(r'cluster_(\d+)_ratio')[0].astype(float)

# 主导模式的占比
vehicle_patterns['dominant_cluster_ratio'] = vehicle_patterns[ratio_cols].max(axis=1)

vehicle_patterns = vehicle_patterns.reset_index()

print(f"✅ Computed patterns for {len(vehicle_patterns):,} vehicles")

# 统计放电模式分布
print(f"\n📊 Discharge pattern distribution:")
for cluster_id in [0, 1, 2]:
    count = (vehicle_patterns['dominant_cluster'] == cluster_id).sum()
    avg_ratio = vehicle_patterns[vehicle_patterns['dominant_cluster'] == cluster_id]['dominant_cluster_ratio'].mean()
    print(f"   Dominant Cluster {cluster_id}: {count:,} vehicles (平均占比 {avg_ratio*100:.1f}%)")

# 融合到车���特征
print(f"\n🔗 Merging with vehicle features...")

df_merged = df_vehicle.merge(vehicle_patterns, on='vehicle_id', how='left')

# 填充缺失值（Phase 1中有但Phase 2没有的车辆）
fill_cols = ['cluster_0_ratio', 'cluster_1_ratio', 'cluster_2_ratio', 
             'cluster_0_count', 'cluster_1_count', 'cluster_2_count',
             'total_discharge_events', 'dominant_cluster_ratio']

for col in fill_cols:
    if col in df_merged.columns:
        df_merged[col] = df_merged[col].fillna(0)

print(f"✅ Merged dataset: {len(df_merged):,} vehicles")

# 只保留有放电事件的车辆（用于分析）
df_merged_valid = df_merged[df_merged['total_discharge_events'] > 0].copy()

print(f"   Valid vehicles (with discharge events): {len(df_merged_valid):,}")

# 保存完整数据
df_merged.to_csv('./results/vehicle_complete_profile_all.csv', index=False, encoding='utf-8-sig')
print(f"💾 Saved: vehicle_complete_profile_all.csv (all vehicles)")

# 保存有效数据（用于分析）
df_merged_valid.to_csv('./results/vehicle_complete_profile.csv', index=False, encoding='utf-8-sig')
print(f"💾 Saved: vehicle_complete_profile.csv (valid vehicles only)")

# 详细统计
print(f"\n📊 Detailed Statistics:")

print(f"\n车辆分类：")
print(f"   总车辆数: {len(df_merged):,}")
print(f"   有充电事件: {(df_merged['n_charging_events'] > 0).sum():,}")
print(f"   有放电事件: {(df_merged['total_discharge_events'] > 0).sum():,}")
print(f"   充放电都有: {((df_merged['n_charging_events'] > 0) & (df_merged['total_discharge_events'] > 0)).sum():,}")

print(f"\n按主导放电模式分组：")
for cluster_id in sorted(df_merged_valid['dominant_cluster'].dropna().unique()):
    cluster_data = df_merged_valid[df_merged_valid['dominant_cluster'] == cluster_id]
    
    print(f"\n   Cluster {int(cluster_id)} (n={len(cluster_data):,}):")
    print(f"      平均放电事件数: {cluster_data['total_discharge_events'].mean():.1f}")
    print(f"      平均充电事件数: {cluster_data['n_charging_events'].mean():.1f}")
    print(f"      平均充电触发SOC: {cluster_data['avg_charging_trigger_soc'].mean():.1f}%")
    print(f"      平均充电频率(次/100km): {cluster_data['charging_freq_per_100km'].mean():.2f}")
    print(f"      主导模式平均占比: {cluster_data['dominant_cluster_ratio'].mean()*100:.1f}%")

# 预览数据
print(f"\n📋 Sample data:")
display_cols = [
    'vehicle_id', 'n_charging_events', 'avg_charging_trigger_soc',
    'charging_freq_per_100km', 'total_discharge_events',
    'cluster_0_ratio', 'cluster_1_ratio', 'cluster_2_ratio', 'dominant_cluster'
]

print(df_merged_valid[display_cols].head(10).to_string())

print(f"\n{'='*70}")
print(f"✅ Phase 2 Complete!")
print(f"{'='*70}")
print(f"\n📁 Generated files:")
print(f"   1. vehicle_complete_profile_all.csv - 所有车辆（包括缺失数据）")
print(f"   2. vehicle_complete_profile.csv - 有效车辆（用于Phase 3分析）")
print(f"\n💡 Next step:")
print(f"   python charging_anxiety_analysis.py")
print(f"{'='*70}")