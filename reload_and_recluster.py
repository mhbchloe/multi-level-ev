"""
从原始7天CSV重新加载和聚类
使用更合理的事件分割策略
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import glob
import os

print("="*70)
print("🔄 Reload from Original 7-Day Data")
print("="*70)

# ==================== 1. 加载所有7天数据 ====================
print("\n📂 Loading 7-day data...")

data_files = sorted(glob.glob('./20250*_processed.csv'))
print(f"Found {len(data_files)} files:")
for f in data_files:
    print(f"  - {os.path.basename(f)}")

all_data = []
for file in data_files:
    df = pd.read_csv(file)
    all_data.append(df)
    print(f"  {os.path.basename(file)}: {len(df)} records")

combined_df = pd.concat(all_data, ignore_index=True)
print(f"\n✅ Total records: {len(combined_df)}")
print(f"   Date range: {combined_df['datetime'].min()} to {combined_df['datetime'].max()}")
print(f"   Unique vehicles: {combined_df['vehicle_id'].nunique()}")

# ==================== 2. 重新分割驾驶事件 ====================
print(f"\n{'='*70}")
print("🚗 Re-segmenting Driving Events")
print("="*70)

# 转换时间
combined_df['datetime'] = pd.to_datetime(combined_df['datetime'])
combined_df = combined_df.sort_values(['vehicle_id', 'datetime'])

def segment_trips_improved(df, time_gap_minutes=30, min_distance_km=0.3, min_duration_minutes=2):
    """
    改进的行程分割策略：
    1. 时间间隔 > 30分钟 → 新行程
    2. 位置变化 > 0.3km → 新行程
    3. 行程至少2分钟
    """
    trips = []
    
    for vehicle_id, vehicle_data in df.groupby('vehicle_id'):
        vehicle_data = vehicle_data.sort_values('datetime').reset_index(drop=True)
        
        trip_id = 0
        current_trip_start = 0
        
        for i in range(1, len(vehicle_data)):
            # 计算时间间隔
            time_gap = (vehicle_data.loc[i, 'datetime'] - 
                       vehicle_data.loc[i-1, 'datetime']).total_seconds() / 60
            
            # 计算位置变化（如果有经纬度）
            if 'distance_km' in vehicle_data.columns:
                distance_change = vehicle_data.loc[i, 'distance_km']
            else:
                distance_change = 0
            
            # 判断是否新行程
            if time_gap > time_gap_minutes or (distance_change > 0 and time_gap > 5):
                # 保存当前行程
                trip_data = vehicle_data.iloc[current_trip_start:i].copy()
                trip_duration = (trip_data['datetime'].max() - 
                               trip_data['datetime'].min()).total_seconds() / 60
                
                if trip_duration >= min_duration_minutes:
                    trip_data['trip_id'] = f"{vehicle_id}_{trip_id}"
                    trips.append(trip_data)
                    trip_id += 1
                
                current_trip_start = i
        
        # 保存最后一个行程
        trip_data = vehicle_data.iloc[current_trip_start:].copy()
        trip_duration = (trip_data['datetime'].max() - 
                        trip_data['datetime'].min()).total_seconds() / 60
        
        if trip_duration >= min_duration_minutes:
            trip_data['trip_id'] = f"{vehicle_id}_{trip_id}"
            trips.append(trip_data)
    
    return trips

# 分割行程
print("\n🔍 Segmenting trips (time_gap=30min, min_duration=2min)...")
trips = segment_trips_improved(combined_df, 
                               time_gap_minutes=30,
                               min_distance_km=0.3,
                               min_duration_minutes=2)

print(f"✅ Segmented into {len(trips)} trips")

# ==================== 3. 提取行程特征 ====================
print(f"\n{'='*70}")
print("📊 Extracting Trip Features")
print("="*70)

trip_features = []

for trip_data in trips:
    if len(trip_data) < 5:  # 至少5个记录点
        continue
    
    trip_id = trip_data['trip_id'].iloc[0]
    vehicle_id = trip_data['vehicle_id'].iloc[0]
    
    # 时间特征
    duration_minutes = (trip_data['datetime'].max() - 
                       trip_data['datetime'].min()).total_seconds() / 60
    start_hour = trip_data['datetime'].iloc[0].hour
    is_weekend = trip_data['is_weekend'].iloc[0] if 'is_weekend' in trip_data.columns else 0
    
    # 速度特征
    speed_mean = trip_data['spd'].mean()
    speed_max = trip_data['spd'].max()
    speed_std = trip_data['spd'].std()
    speed_median = trip_data['spd'].median()
    
    # 加速度特征
    if 'acc' in trip_data.columns:
        acc_mean = trip_data['acc'].mean()
        acc_std = trip_data['acc'].std()
        harsh_accel = (trip_data['acc'] > 2).sum() if 'acc' in trip_data.columns else 0
        harsh_decel = (trip_data['acc'] < -2).sum() if 'acc' in trip_data.columns else 0
    else:
        acc_mean = acc_std = harsh_accel = harsh_decel = 0
    
    # 距离特征
    if 'distance_km' in trip_data.columns:
        total_distance = trip_data['distance_km'].sum()
    else:
        total_distance = speed_mean * duration_minutes / 60
    
    # 能量特征
    soc_start = trip_data['soc'].iloc[0]
    soc_end = trip_data['soc'].iloc[-1]
    soc_drop = soc_start - soc_end
    
    if 'power' in trip_data.columns:
        power_mean = trip_data['power'].mean()
        power_max = trip_data['power'].max()
        power_std = trip_data['power'].std()
    else:
        power_mean = power_max = power_std = 0
    
    if 'energy_consumption' in trip_data.columns:
        energy_total = trip_data['energy_consumption'].sum()
    else:
        energy_total = 0
    
    # 移动状态
    if 'is_moving' in trip_data.columns:
        moving_ratio = trip_data['is_moving'].mean()
    else:
        moving_ratio = (trip_data['spd'] > 1).mean()
    
    # 速度分布
    low_speed_ratio = (trip_data['spd'] < 20).mean()
    medium_speed_ratio = ((trip_data['spd'] >= 20) & (trip_data['spd'] < 60)).mean()
    high_speed_ratio = (trip_data['spd'] >= 60).mean()
    
    # 充电/再生制动
    if 'is_charging' in trip_data.columns:
        charging_ratio = trip_data['is_charging'].mean()
    else:
        charging_ratio = 0
    
    if 'is_regenerative_braking' in trip_data.columns:
        regen_ratio = trip_data['is_regenerative_braking'].mean()
    else:
        regen_ratio = 0
    
    # 效率
    if total_distance > 0 and soc_drop > 0:
        efficiency = total_distance / soc_drop
    else:
        efficiency = 0
    
    trip_features.append({
        'trip_id': trip_id,
        'vehicle_id': vehicle_id,
        
        # 时间特征
        'duration_minutes': duration_minutes,
        'start_hour': start_hour,
        'is_weekend': is_weekend,
        
        # 速度特征
        'speed_mean': speed_mean,
        'speed_max': speed_max,
        'speed_std': speed_std,
        'speed_median': speed_median,
        
        # 加速度特征
        'acc_mean': acc_mean,
        'acc_std': acc_std,
        'harsh_accel': harsh_accel,
        'harsh_decel': harsh_decel,
        
        # 距离特征
        'distance_total': total_distance,
        'moving_ratio': moving_ratio,
        
        # 能量特征
        'soc_drop_total': soc_drop,
        'soc_mean': trip_data['soc'].mean(),
        'soc_std': trip_data['soc'].std(),
        'power_mean': power_mean,
        'power_max': power_max,
        'power_std': power_std,
        'energy_consumption_total': energy_total,
        'efficiency_mean': efficiency,
        
        # 分布特征
        'low_speed_ratio': low_speed_ratio,
        'medium_speed_ratio': medium_speed_ratio,
        'high_speed_ratio': high_speed_ratio,
        'charging_ratio': charging_ratio,
        'regen_braking_ratio': regen_ratio
    })

trip_features_df = pd.DataFrame(trip_features)
print(f"✅ Extracted features for {len(trip_features_df)} trips")

# 保存
os.makedirs('./results/reloaded', exist_ok=True)
trip_features_df.to_csv('./results/reloaded/trip_features.csv', index=False)
print(f"💾 Saved: ./results/reloaded/trip_features.csv")

# ==================== 4. 数据质量分析 ====================
print(f"\n{'='*70}")
print("📊 Data Quality Analysis")
print("="*70)

print(f"\n✅ Trip Statistics:")
print(f"   Total trips: {len(trip_features_df)}")
print(f"   Unique vehicles: {trip_features_df['vehicle_id'].nunique()}")
print(f"   Trips per vehicle: {len(trip_features_df) / trip_features_df['vehicle_id'].nunique():.1f}")

print(f"\n⏱️  Duration:")
print(f"   Mean: {trip_features_df['duration_minutes'].mean():.1f} min")
print(f"   Median: {trip_features_df['duration_minutes'].median():.1f} min")
print(f"   Range: {trip_features_df['duration_minutes'].min():.1f} - {trip_features_df['duration_minutes'].max():.1f} min")

print(f"\n🚗 Speed:")
print(f"   Mean: {trip_features_df['speed_mean'].mean():.1f} km/h")
print(f"   Median: {trip_features_df['speed_mean'].median():.1f} km/h")
print(f"   Range: {trip_features_df['speed_mean'].min():.1f} - {trip_features_df['speed_mean'].max():.1f} km/h")

print(f"\n📏 Distance:")
print(f"   Mean: {trip_features_df['distance_total'].mean():.1f} km")
print(f"   Median: {trip_features_df['distance_total'].median():.1f} km")
print(f"   Range: {trip_features_df['distance_total'].min():.1f} - {trip_features_df['distance_total'].max():.1f} km")

# 速度分布分析
print(f"\n📊 Speed Distribution:")
print(f"   Low speed (<20 km/h): {(trip_features_df['speed_mean'] < 20).sum()} ({(trip_features_df['speed_mean'] < 20).mean()*100:.1f}%)")
print(f"   Medium speed (20-60): {((trip_features_df['speed_mean'] >= 20) & (trip_features_df['speed_mean'] < 60)).sum()} ({((trip_features_df['speed_mean'] >= 20) & (trip_features_df['speed_mean'] < 60)).mean()*100:.1f}%)")
print(f"   High speed (≥60 km/h): {(trip_features_df['speed_mean'] >= 60).sum()} ({(trip_features_df['speed_mean'] >= 60).mean()*100:.1f}%)")

print(f"\n💡 Comparison with previous segmentation:")
print(f"   Previous: 3005 events (SOC-based)")
print(f"   Current: {len(trip_features_df)} trips (time & location-based)")
print(f"   Difference: {len(trip_features_df) - 3005} ({(len(trip_features_df) - 3005)/3005*100:+.1f}%)")

print("\n" + "="*70)
print("✅ Data Reload Complete!")
print("="*70)
print("\n💡 Next step: Run dual-channel clustering on these new trip features")
print("   python dual_channel_resampling_complete.py  # but use ./results/reloaded/trip_features.csv")