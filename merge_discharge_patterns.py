"""
Phase 2: 融合放电模式（来自阉割事件表）与完整充电行为
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
print(f"✅ Discharge events: {len(df_events):,}")

# 只保留放电事件
df_discharge = df_events[df_events['event_type'] == 'discharge']

print(f"   Discharge events: {len(df_discharge):,}")

# 计算每辆车的放电模式占比
print("\n📊 Computing discharge pattern distribution...")

vehicle_patterns = df_discharge.groupby(['vehicle_id', 'cluster']).size().unstack(fill_value=0)
vehicle_patterns['total_discharge_events'] = vehicle_patterns.sum(axis=1)

# 计算占比
for col in [0, 1, 2]:
    if col in vehicle_patterns.columns:
        vehicle_patterns[f'cluster_{col}_ratio'] = vehicle_patterns[col] / vehicle_patterns['total_discharge_events']
    else:
        vehicle_patterns[f'cluster_{col}_ratio'] = 0

# 主导模式
vehicle_patterns['dominant_cluster'] = vehicle_patterns[[0, 1, 2]].idxmax(axis=1)

vehicle_patterns = vehicle_patterns.reset_index()

print(f"✅ Computed patterns for {len(vehicle_patterns):,} vehicles")

# 融合
df_merged = df_vehicle.merge(vehicle_patterns, on='vehicle_id', how='left')

# 填充缺失值（没有放电事件的车辆）
df_merged[['cluster_0_ratio', 'cluster_1_ratio', 'cluster_2_ratio']] = \
    df_merged[['cluster_0_ratio', 'cluster_1_ratio', 'cluster_2_ratio']].fillna(0)

print(f"✅ Merged dataset: {len(df_merged):,} vehicles")

# 保存
df_merged.to_csv('./results/vehicle_complete_profile.csv', index=False, encoding='utf-8-sig')
print(f"💾 Saved: vehicle_complete_profile.csv")

# 预览
print(f"\n📋 Sample data:")
print(df_merged[[
    'vehicle_id', 'n_charging_events', 'avg_charging_trigger_soc',
    'charging_freq_per_100km', 'cluster_0_ratio', 'cluster_1_ratio', 'cluster_2_ratio',
    'dominant_cluster'
]].head(10))

print(f"\n{'='*70}")
print(f"✅ Phase 2 Complete!")
print(f"{'='*70}")