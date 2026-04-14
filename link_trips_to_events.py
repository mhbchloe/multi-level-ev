"""
Step 2: 关联完整出行片段与人为划分的放电事件
分析：一次出行包含多少个放电事件、什么类型
"""

import pandas as pd
import numpy as np

print("="*70)
print("🔗 Link Trip Segments to Discharge Events")
print("="*70)

# 加载数据
print("\n📂 Loading data...")

df_trips = pd.read_csv('./results/trip_segments.csv')
df_events = pd.read_csv('./results/event_table.csv')

print(f"✅ Trip segments: {len(df_trips):,}")
print(f"✅ Discharge events: {len(df_events):,}")

# 关联：找到每次出行包含哪些放电事件
print("\n🔗 Linking trips to events...")

trip_event_mapping = []

for _, trip in df_trips.iterrows():
    # 找到该车辆在该时间段内的放电事件
    vehicle_events = df_events[
        (df_events['vehicle_id'] == trip['vehicle_id']) &
        (df_events['start_time'] >= trip['start_time']) &
        (df_events['end_time'] <= trip['end_time'])
    ]
    
    if len(vehicle_events) == 0:
        continue
    
    # 统计该出行的放电事件特征
    trip_event_mapping.append({
        'trip_segment_id': trip['trip_segment_id'],
        'vehicle_id': trip['vehicle_id'],
        'trip_start_time': trip['start_time'],
        'trip_duration': trip['duration_seconds'],
        'trip_distance': trip['distance_km'],
        'trip_soc_drop': trip['soc_drop'],
        'is_charging_after': trip['is_charging_after'],
        
        # 放电事件统计
        'n_discharge_events': len(vehicle_events),
        'n_cluster_0': (vehicle_events['cluster'] == 0).sum(),
        'n_cluster_1': (vehicle_events['cluster'] == 1).sum(),
        'n_cluster_2': (vehicle_events['cluster'] == 2).sum(),
        
        # 主导cluster
        'dominant_cluster': vehicle_events['cluster'].mode()[0] if len(vehicle_events) > 0 else -1,
        
        # 放电事件多样性（熵）
        'cluster_entropy': -sum([
            (vehicle_events['cluster'] == c).sum() / len(vehicle_events) * 
            np.log((vehicle_events['cluster'] == c).sum() / len(vehicle_events) + 1e-10)
            for c in [0, 1, 2]
        ]) if len(vehicle_events) > 0 else 0,
        
        # 平均事件特征
        'avg_event_distance': vehicle_events['distance_km'].mean(),
        'avg_event_speed': vehicle_events['speed_mean'].mean(),
        'total_harsh_accel': vehicle_events['harsh_accel_count'].sum(),
        'total_harsh_brake': vehicle_events['harsh_brake_count'].sum(),
    })

df_trip_events = pd.DataFrame(trip_event_mapping)

print(f"✅ Linked {len(df_trip_events):,} trips to events")

# 保存
df_trip_events.to_csv('./results/trip_with_events.csv', index=False, encoding='utf-8-sig')
print(f"💾 Saved: trip_with_events.csv")

# 统计
print(f"\n📊 Statistics:")
print(f"   Avg discharge events per trip: {df_trip_events['n_discharge_events'].mean():.1f}")
print(f"   Avg cluster entropy: {df_trip_events['cluster_entropy'].mean():.3f}")
print(f"   Trips with charging after: {df_trip_events['is_charging_after'].sum():,}")

print(f"\n按主导cluster分组：")
for cluster in sorted(df_trip_events['dominant_cluster'].unique()):
    if cluster == -1:
        continue
    cluster_data = df_trip_events[df_trip_events['dominant_cluster'] == cluster]
    print(f"   Cluster {int(cluster)} (n={len(cluster_data):,}):")
    print(f"      充电比例: {cluster_data['is_charging_after'].mean()*100:.1f}%")
    print(f"      平均SOC消耗: {cluster_data['trip_soc_drop'].mean():.1f}%")

print(f"\n{'='*70}")
print(f"✅ Step 2 Complete!")
print(f"{'='*70}")