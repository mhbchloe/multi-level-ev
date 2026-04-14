"""
诊断脚本：检查数据对齐问题
"""

import pandas as pd
import numpy as np
import os

print("="*70)
print("🔍 Data Diagnostic Report")
print("="*70)

input_dir = "./analysis_complete_vehicles/results/"

# 加载数据
df_segments = pd.read_csv(os.path.join(input_dir, 'segments_with_clusters_labeled.csv'))
df_charging = pd.read_csv(os.path.join(input_dir, 'charging_events_rebuilt.csv'))

print("\n📊 SEGMENTS TABLE:")
print(f"   Shape: {df_segments.shape}")
print(f"   Columns: {df_segments.columns.tolist()}")
print(f"   Vehicle IDs: {df_segments['vehicle_id'].nunique():,}")
print(f"   Sample vehicle IDs: {df_segments['vehicle_id'].unique()[:5]}")
print(f"\n   Time column info:")
print(f"   - start_time dtype: {df_segments['start_time'].dtype}")
print(f"   - start_time sample: {df_segments['start_time'].head()}")
print(f"   - start_time range: {df_segments['start_time'].min()} to {df_segments['start_time'].max()}")

print("\n📊 CHARGING TABLE:")
print(f"   Shape: {df_charging.shape}")
print(f"   Columns: {df_charging.columns.tolist()}")
print(f"   Vehicle IDs: {df_charging['vehicle_id'].nunique():,}")
print(f"   Sample vehicle IDs: {df_charging['vehicle_id'].unique()[:5]}")
print(f"\n   Time column info:")
print(f"   - start_time dtype: {df_charging['start_time'].dtype}")
print(f"   - start_time sample: {df_charging['start_time'].head()}")
print(f"   - start_time range: {df_charging['start_time'].min()} to {df_charging['start_time'].max()}")

# 检查车辆ID的交集
seg_vehicles = set(df_segments['vehicle_id'].unique())
charge_vehicles = set(df_charging['vehicle_id'].unique())
common_vehicles = seg_vehicles & charge_vehicles

print(f"\n🔗 Vehicle ID Overlap:")
print(f"   Segments has: {len(seg_vehicles):,} unique vehicles")
print(f"   Charging has: {len(charge_vehicles):,} unique vehicles")
print(f"   Common: {len(common_vehicles):,} vehicles")

if len(common_vehicles) == 0:
    print("   ❌ NO OVERLAP! This is the problem!")
else:
    print(f"   ✅ Good overlap!")

# 检查一个具体的共同车辆
if len(common_vehicles) > 0:
    test_vid = list(common_vehicles)[0]
    print(f"\n🔬 Sample vehicle: {test_vid}")
    
    v_segs = df_segments[df_segments['vehicle_id'] == test_vid]
    v_charges = df_charging[df_charging['vehicle_id'] == test_vid]
    
    print(f"   Segments: {len(v_segs)}")
    print(f"   Charging Events: {len(v_charges)}")
    
    print(f"\n   Segment time sample:")
    print(f"   {v_segs['start_time'].head().values}")
    
    print(f"\n   Charging time sample:")
    print(f"   {v_charges['start_time'].head().values}")