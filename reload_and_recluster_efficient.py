"""
内存高效版本：逐文件、逐车辆处理
处理9800万条记录
"""
import pandas as pd
import numpy as np
from datetime import datetime
import glob
import os
from tqdm import tqdm

print("="*70)
print("🔄 Memory-Efficient Data Reload")
print("="*70)

# ==================== 配置 ====================
CHUNK_SIZE = 100000  # 每次读取10万条
SAMPLE_VEHICLES = None  # None=全部车辆，可设置如100只处理100辆车
OUTPUT_DIR = './results/reloaded'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 1. 快速统计 ====================
print("\n📊 Quick Statistics...")

data_files = sorted(glob.glob('./20250*_processed.csv'))
print(f"Found {len(data_files)} files")

# 只读第一个文件的前几行看列��
sample_df = pd.read_csv(data_files[0], nrows=1000)
print(f"\n✅ Columns ({len(sample_df.columns)}):")
print(sample_df.columns.tolist())

# 统计每个文件的车辆数
print(f"\n🚗 Scanning vehicles...")
all_vehicle_ids = set()

for file in data_files:
    print(f"  {os.path.basename(file)}...", end=' ')
    # 只读vehicle_id列
    vehicles_in_file = pd.read_csv(file, usecols=['vehicle_id'])['vehicle_id'].unique()
    all_vehicle_ids.update(vehicles_in_file)
    print(f"{len(vehicles_in_file)} vehicles")

print(f"\n✅ Total unique vehicles: {len(all_vehicle_ids)}")

# 是否采样
if SAMPLE_VEHICLES and SAMPLE_VEHICLES < len(all_vehicle_ids):
    selected_vehicles = np.random.choice(list(all_vehicle_ids), SAMPLE_VEHICLES, replace=False)
    print(f"⚠️  Sampling {SAMPLE_VEHICLES} vehicles for analysis")
else:
    selected_vehicles = list(all_vehicle_ids)
    print(f"✅ Processing all {len(selected_vehicles)} vehicles")

# ==================== 2. 逐车辆处理 ====================
print(f"\n{'='*70}")
print("🚗 Processing Vehicles")
print("="*70)

def segment_trips_for_vehicle(vehicle_data, vehicle_id, 
                              time_gap_minutes=30, 
                              min_duration_minutes=2):
    """为单个车辆分割行程"""
    vehicle_data = vehicle_data.sort_values('datetime').reset_index(drop=True)
    
    trips = []
    trip_id = 0
    current_trip_start = 0
    
    for i in range(1, len(vehicle_data)):
        time_gap = (vehicle_data.loc[i, 'datetime'] - 
                   vehicle_data.loc[i-1, 'datetime']).total_seconds() / 60
        
        # 判断是否新行程（时间间隔>30分钟）
        if time_gap > time_gap_minutes:
            # 保存当前行程
            trip_data = vehicle_data.iloc[current_trip_start:i].copy()
            trip_duration = (trip_data['datetime'].max() - 
                           trip_data['datetime'].min()).total_seconds() / 60
            
            if trip_duration >= min_duration_minutes and len(trip_data) >= 5:
                trips.append({
                    'trip_id': f"{vehicle_id}_{trip_id}",
                    'vehicle_id': vehicle_id,
                    'data': trip_data
                })
                trip_id += 1
            
            current_trip_start = i
    
    # 最后一个行程
    trip_data = vehicle_data.iloc[current_trip_start:].copy()
    trip_duration = (trip_data['datetime'].max() - 
                    trip_data['datetime'].min()).total_seconds() / 60
    
    if trip_duration >= min_duration_minutes and len(trip_data) >= 5:
        trips.append({
            'trip_id': f"{vehicle_id}_{trip_id}",
            'vehicle_id': vehicle_id,
            'data': trip_data
        })
    
    return trips

def extract_trip_features(trip_data, trip_id, vehicle_id):
    """提取单个行程的特征"""
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
        harsh_accel = (trip_data['acc'] > 2).sum()
        harsh_decel = (trip_data['acc'] < -2).sum()
    else:
        acc_mean = acc_std = harsh_accel = harsh_decel = 0
    
    # 距离特征
    if 'distance_km' in trip_data.columns:
        total_distance = trip_data['distance_km'].sum()
    else:
        total_distance = speed_mean * duration_minutes / 60
    
    # 能量特征
    soc_drop = trip_data['soc'].iloc[0] - trip_data['soc'].iloc[-1]
    
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
    
    # 电压电流
    if 'v' in trip_data.columns:
        voltage_mean = trip_data['v'].mean()
        voltage_std = trip_data['v'].std()
    else:
        voltage_mean = voltage_std = 0
    
    if 'i' in trip_data.columns:
        current_mean = trip_data['i'].mean()
        current_max = trip_data['i'].max()
    else:
        current_mean = current_max = 0
    
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
    
    return {
        'trip_id': trip_id,
        'vehicle_id': vehicle_id,
        'duration_minutes': duration_minutes,
        'start_hour': start_hour,
        'is_weekend': is_weekend,
        'speed_mean': speed_mean,
        'speed_max': speed_max,
        'speed_std': speed_std,
        'speed_median': speed_median,
        'acc_mean': acc_mean,
        'acc_std': acc_std,
        'harsh_accel': harsh_accel,
        'harsh_decel': harsh_decel,
        'distance_total': total_distance,
        'moving_ratio': moving_ratio,
        'soc_drop_total': soc_drop,
        'soc_mean': trip_data['soc'].mean(),
        'soc_std': trip_data['soc'].std(),
        'voltage_mean': voltage_mean,
        'voltage_std': voltage_std,
        'current_mean': current_mean,
        'current_max': current_max,
        'power_mean': power_mean,
        'power_max': power_max,
        'power_std': power_std,
        'energy_consumption_total': energy_total,
        'efficiency_mean': efficiency,
        'low_speed_ratio': low_speed_ratio,
        'medium_speed_ratio': medium_speed_ratio,
        'high_speed_ratio': high_speed_ratio,
        'charging_ratio': charging_ratio,
        'regen_braking_ratio': regen_ratio
    }

# 逐车辆处理
all_trip_features = []
processed_vehicles = 0

print(f"\nProcessing {len(selected_vehicles)} vehicles...")
print("This may take 10-30 minutes...")

for vehicle_id in tqdm(selected_vehicles, desc="Vehicles"):
    vehicle_data_list = []
    
    # 从所有文件中读取该车辆的数据
    for file in data_files:
        try:
            # 分块读取，只保留该车辆的数据
            for chunk in pd.read_csv(file, chunksize=CHUNK_SIZE):
                vehicle_chunk = chunk[chunk['vehicle_id'] == vehicle_id]
                if len(vehicle_chunk) > 0:
                    vehicle_data_list.append(vehicle_chunk)
        except Exception as e:
            print(f"\n⚠️  Error reading {file} for vehicle {vehicle_id}: {e}")
            continue
    
    if len(vehicle_data_list) == 0:
        continue
    
    # 合并该车辆的所有数据
    vehicle_data = pd.concat(vehicle_data_list, ignore_index=True)
    vehicle_data['datetime'] = pd.to_datetime(vehicle_data['datetime'])
    vehicle_data = vehicle_data.sort_values('datetime')
    
    # 分割行程
    trips = segment_trips_for_vehicle(vehicle_data, vehicle_id)
    
    # 提取特征
    for trip in trips:
        try:
            features = extract_trip_features(trip['data'], trip['trip_id'], vehicle_id)
            all_trip_features.append(features)
        except Exception as e:
            print(f"\n⚠️  Error extracting features for {trip['trip_id']}: {e}")
            continue
    
    processed_vehicles += 1
    
    # 每处理100辆车保存一次（防止崩溃）
    if processed_vehicles % 100 == 0:
        temp_df = pd.DataFrame(all_trip_features)
        temp_df.to_csv(f'{OUTPUT_DIR}/trip_features_temp_{processed_vehicles}.csv', index=False)
        print(f"\n💾 Checkpoint: Saved {len(all_trip_features)} trips from {processed_vehicles} vehicles")

# ==================== 3. 保存最终结果 ====================
print(f"\n{'='*70}")
print("💾 Saving Results")
print("="*70)

trip_features_df = pd.DataFrame(all_trip_features)
trip_features_df.to_csv(f'{OUTPUT_DIR}/trip_features_final.csv', index=False)

print(f"\n✅ Saved {len(trip_features_df)} trips")
print(f"   From {trip_features_df['vehicle_id'].nunique()} vehicles")
print(f"   File: {OUTPUT_DIR}/trip_features_final.csv")

# ==================== 4. 统计分析 ====================
print(f"\n{'='*70}")
print("📊 Trip Statistics")
print("="*70)

print(f"\n✅ Trip Summary:")
print(f"   Total trips: {len(trip_features_df)}")
print(f"   Unique vehicles: {trip_features_df['vehicle_id'].nunique()}")
print(f"   Trips per vehicle: {len(trip_features_df) / trip_features_df['vehicle_id'].nunique():.1f}")

print(f"\n⏱️  Duration:")
print(trip_features_df['duration_minutes'].describe())

print(f"\n🚗 Speed:")
print(trip_features_df['speed_mean'].describe())

print(f"\n📊 Speed Distribution:")
print(f"   Low (<20 km/h): {(trip_features_df['speed_mean'] < 20).sum()} ({(trip_features_df['speed_mean'] < 20).mean()*100:.1f}%)")
print(f"   Medium (20-60): {((trip_features_df['speed_mean'] >= 20) & (trip_features_df['speed_mean'] < 60)).sum()} ({((trip_features_df['speed_mean'] >= 20) & (trip_features_df['speed_mean'] < 60)).mean()*100:.1f}%)")
print(f"   High (≥60): {(trip_features_df['speed_mean'] >= 60).sum()} ({(trip_features_df['speed_mean'] >= 60).mean()*100:.1f}%)")

print("\n" + "="*70)
print("✅ Processing Complete!")
print("="*70)