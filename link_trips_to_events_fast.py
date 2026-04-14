"""
Step 2: 关联出行与事件 - 超速优化版
使用预索引 + 向量化
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict

print("="*70)
print("🔗 Link Trip Segments to Discharge Events (FAST)")
print("="*70)

# 加载数据
print("\n📂 Loading data...")

df_trips = pd.read_csv('./results/trip_segments.csv')
df_events = pd.read_csv('./results/event_table.csv')

print(f"✅ Trip segments: {len(df_trips):,}")
print(f"✅ Discharge events: {len(df_events):,}")

# === 关键优化1：为每个vehicle_id预建事件索引 ===
print("\n🔧 Building event index by vehicle...")

events_by_vehicle = defaultdict(list)

for _, event in tqdm(df_events.iterrows(), total=len(df_events), desc="Indexing"):
    events_by_vehicle[event['vehicle_id']].append({
        'start_time': event['start_time'],
        'end_time': event['end_time'],
        'cluster': event['cluster'],
        'distance_km': event.get('distance_km', 0),
        'speed_mean': event.get('speed_mean', 0),
        'harsh_accel_count': event.get('harsh_accel_count', 0),
        'harsh_brake_count': event.get('harsh_brake_count', 0),
    })

print(f"✅ Indexed {len(events_by_vehicle):,} vehicles")

# === 关键优化2：向量化处理 ===
print("\n🔗 Linking trips to events (optimized)...")

trip_event_mapping = []

for _, trip in tqdm(df_trips.iterrows(), total=len(df_trips), desc="Processing trips"):
    vehicle_id = trip['vehicle_id']
    
    # 快速查找：只查该车辆的事件
    if vehicle_id not in events_by_vehicle:
        continue
    
    vehicle_events = events_by_vehicle[vehicle_id]
    
    # 筛选时间范围内的事件
    matched_events = [
        e for e in vehicle_events
        if e['start_time'] >= trip['start_time'] and e['end_time'] <= trip['end_time']
    ]
    
    if len(matched_events) == 0:
        continue
    
    # 统计
    clusters = [e['cluster'] for e in matched_events]
    cluster_counts = {0: 0, 1: 0, 2: 0}
    for c in clusters:
        if c in cluster_counts:
            cluster_counts[c] += 1
    
    # 主导cluster
    dominant_cluster = max(cluster_counts, key=cluster_counts.get) if matched_events else -1
    
    # 簇熵（多样性）
    total = len(matched_events)
    entropy = -sum([
        (cluster_counts[c] / total) * np.log(cluster_counts[c] / total + 1e-10)
        for c in [0, 1, 2] if cluster_counts[c] > 0
    ]) if total > 0 else 0
    
    trip_event_mapping.append({
        'trip_segment_id': trip['trip_segment_id'],
        'vehicle_id': vehicle_id,
        'trip_start_time': trip['start_time'],
        'trip_duration': trip['duration_seconds'],
        'trip_distance': trip['distance_km'],
        'trip_soc_drop': trip['soc_drop'],
        'is_charging_after': trip['is_charging_after'],
        
        # 放电事件统计
        'n_discharge_events': len(matched_events),
        'n_cluster_0': cluster_counts[0],
        'n_cluster_1': cluster_counts[1],
        'n_cluster_2': cluster_counts[2],
        
        'dominant_cluster': dominant_cluster,
        'cluster_entropy': entropy,
        
        # 平均特征
        'avg_event_distance': np.mean([e['distance_km'] for e in matched_events]) if matched_events else 0,
        'avg_event_speed': np.mean([e['speed_mean'] for e in matched_events]) if matched_events else 0,
        'total_harsh_accel': sum([e['harsh_accel_count'] for e in matched_events]),
        'total_harsh_brake': sum([e['harsh_brake_count'] for e in matched_events]),
    })

df_trip_events = pd.DataFrame(trip_event_mapping)

print(f"\n✅ Linked {len(df_trip_events):,} trips to events")
print(f"   Coverage: {len(df_trip_events)/len(df_trips)*100:.1f}% of trips have discharge events")

# 保存
output_file = './results/trip_with_events.csv'
df_trip_events.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"💾 Saved: {output_file}")

# 统计
print(f"\n📊 Quick Statistics:")
print(f"   Avg discharge events per trip: {df_trip_events['n_discharge_events'].mean():.1f}")
print(f"   Avg cluster entropy: {df_trip_events['cluster_entropy'].mean():.3f}")
print(f"   Trips with charging after: {df_trip_events['is_charging_after'].sum():,} ({df_trip_events['is_charging_after'].mean()*100:.1f}%)")

print(f"\n按主导cluster分组：")
for cluster in sorted(df_trip_events['dominant_cluster'].unique()):
    if cluster == -1:
        continue
    cluster_data = df_trip_events[df_trip_events['dominant_cluster'] == cluster]
    print(f"   Cluster {int(cluster)}: {len(cluster_data):,} trips")
    print(f"      充电比例: {cluster_data['is_charging_after'].mean()*100:.1f}%")
    print(f"      平均SOC消耗: {cluster_data['trip_soc_drop'].mean():.1f}%")
    print(f"      平均事件数: {cluster_data['n_discharge_events'].mean():.1f}")

print(f"\n{'='*70}")
print(f"✅ Step 2 Complete!")
print(f"{'='*70}")