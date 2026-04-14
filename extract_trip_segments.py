"""
Step 1: 从原始数据提取完整出行片段
定义：从一个停车/充电点到下一个停车/充电点
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta

print("="*70)
print("🚗 Extract Complete Trip Segments")
print("="*70)

# 配置
CONFIG = {
    'csv_files': ['20250701.csv'],  # 先测试1天
    'time_gap_threshold': 1800,  # 30分钟无活动=停车
    'location_change_threshold': 0.001,  # 约100米
    'output_file': './results/trip_segments.csv'
}

print("\n📂 Processing GPS data...")

all_trips = []
trip_id_counter = 0

for file_idx, csv_file in enumerate(CONFIG['csv_files'], 1):
    print(f"\n[{file_idx}] {csv_file}")
    
    # 分块读取
    reader = pd.read_csv(
        csv_file,
        chunksize=1000000,
        usecols=['vehicle_id', 'time', 'datetime', 'soc', 'is_charging', 
                 'lat', 'lon', 'spd', 'distance_km'],
        encoding='utf-8',
        on_bad_lines='skip'
    )
    
    vehicle_data = {}
    
    # 收集数据
    for chunk in tqdm(reader, desc="  Reading"):
        for vehicle_id, group in chunk.groupby('vehicle_id'):
            if vehicle_id not in vehicle_data:
                vehicle_data[vehicle_id] = []
            vehicle_data[vehicle_id].append(group)
    
    # 识别出��片段
    print(f"  🔍 Identifying trip segments...")
    
    for vehicle_id, groups in tqdm(vehicle_data.items(), desc="  Vehicles"):
        # 合并所有chunk
        df = pd.concat(groups).sort_values('time')
        
        if len(df) < 10:
            continue
        
        # 计算时间间隔
        df['time_diff'] = df['time'].diff()
        
        # 计算位置变化
        df['lat_diff'] = df['lat'].diff().abs()
        df['lon_diff'] = df['lon'].diff().abs()
        df['location_change'] = np.sqrt(df['lat_diff']**2 + df['lon_diff']**2)
        
        # 识别停车点（时间间隔大 + 位置不变）
        df['is_stop'] = (
            (df['time_diff'] > CONFIG['time_gap_threshold']) |  # 时间间隔>30分钟
            ((df['spd'] == 0) & (df['location_change'] < CONFIG['location_change_threshold']))  # 停车+位置不变
        )
        
        # 识别充电点
        df['is_charging_start'] = (df['is_charging'] == 1) & (df['is_charging'].shift(1) == 0)
        
        # 出行片段 = 停车点之间的轨迹
        df['trip_id'] = (df['is_stop'] | df['is_charging_start']).cumsum()
        
        # 统计每个出行片段
        for trip_id, trip_group in df.groupby('trip_id'):
            if len(trip_group) < 5:  # 至少5个GPS点
                continue
            
            # 判断是否是真实出行（有移动）
            if trip_group['distance_km'].sum() < 0.1:  # 总距离<100米
                continue
            
            # 统计
            trip = {
                'trip_segment_id': trip_id_counter,
                'vehicle_id': vehicle_id,
                'start_time': trip_group['time'].iloc[0],
                'end_time': trip_group['time'].iloc[-1],
                'duration_seconds': trip_group['time'].iloc[-1] - trip_group['time'].iloc[0],
                
                # 位置
                'start_lat': trip_group['lat'].iloc[0],
                'start_lon': trip_group['lon'].iloc[0],
                'end_lat': trip_group['lat'].iloc[-1],
                'end_lon': trip_group['lon'].iloc[-1],
                
                # SOC
                'soc_start': trip_group['soc'].iloc[0],
                'soc_end': trip_group['soc'].iloc[-1],
                'soc_drop': trip_group['soc'].iloc[0] - trip_group['soc'].iloc[-1],
                
                # 行驶特征
                'distance_km': trip_group['distance_km'].sum(),
                'avg_speed': trip_group['spd'].mean(),
                'max_speed': trip_group['spd'].max(),
                
                # 充电状态
                'is_charging_before': (trip_group['is_charging'].iloc[0] == 1),
                'is_charging_after': (trip_group['is_charging'].iloc[-1] == 1),
                'has_charging': (trip_group['is_charging'] == 1).any(),
                
                # GPS点数
                'n_points': len(trip_group)
            }
            
            all_trips.append(trip)
            trip_id_counter += 1

# 转为DataFrame
df_trips = pd.DataFrame(all_trips)

print(f"\n✅ Identified {len(df_trips):,} trip segments")
print(f"   Vehicles: {df_trips['vehicle_id'].nunique():,}")

# 保存
df_trips.to_csv(CONFIG['output_file'], index=False, encoding='utf-8-sig')
print(f"💾 Saved: {CONFIG['output_file']}")

# 统计
print(f"\n📊 Trip Statistics:")
print(f"   Avg duration: {df_trips['duration_seconds'].mean()/60:.1f} minutes")
print(f"   Avg distance: {df_trips['distance_km'].mean():.2f} km")
print(f"   Avg SOC drop: {df_trips['soc_drop'].mean():.1f}%")
print(f"   Trips ending with charging: {df_trips['is_charging_after'].sum():,} ({df_trips['is_charging_after'].mean()*100:.1f}%)")

print(f"\n{'='*70}")
print(f"✅ Step 1 Complete!")
print(f"{'='*70}")