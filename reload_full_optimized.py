"""
优化版：使用4-8个进程，避免I/O争抢
按文件分批处理，而不是按车辆
"""
import pandas as pd
import numpy as np
import glob
import os
from multiprocessing import Pool
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🚀 Optimized Full Data Processing")
print("="*70)

# ==================== 配置 ====================
OUTPUT_DIR = './results/reloaded_full'
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_PROCESSES = 6  # ⭐ 固定用6个进程，避免I/O争抢
CHUNK_SIZE = 100000

print(f"\n⚙️  Configuration:")
print(f"   Using: {N_PROCESSES} processes (optimized for I/O)")
print(f"   Chunk size: {CHUNK_SIZE}")

# ==================== 查找数据文件 ====================
data_files = sorted(glob.glob('./*processed.csv'))
if not data_files:
    data_files = sorted(glob.glob('./data/*processed.csv'))

print(f"\n✅ Found {len(data_files)} files")

# ==================== 策略1：逐文件处理（单进程）====================
print(f"\n{'='*70}")
print("📂 Strategy: Process File by File")
print("="*70)

def segment_trips(vehicle_data, vehicle_id, time_gap_minutes=30):
    """分割行程"""
    vehicle_data = vehicle_data.sort_values('datetime').reset_index(drop=True)
    trips = []
    trip_id = 0
    start = 0
    
    for i in range(1, len(vehicle_data)):
        time_gap = (vehicle_data.loc[i, 'datetime'] - 
                   vehicle_data.loc[i-1, 'datetime']).total_seconds() / 60
        
        if time_gap > time_gap_minutes:
            trip_data = vehicle_data.iloc[start:i]
            duration = (trip_data['datetime'].max() - 
                       trip_data['datetime'].min()).total_seconds() / 60
            
            if duration >= 2 and len(trip_data) >= 5:
                trips.append({
                    'trip_id': f"{vehicle_id}_{trip_id}",
                    'vehicle_id': vehicle_id,
                    'data': trip_data
                })
                trip_id += 1
            start = i
    
    # 最后一个
    trip_data = vehicle_data.iloc[start:]
    duration = (trip_data['datetime'].max() - 
               trip_data['datetime'].min()).total_seconds() / 60
    if duration >= 2 and len(trip_data) >= 5:
        trips.append({
            'trip_id': f"{vehicle_id}_{trip_id}",
            'vehicle_id': vehicle_id,
            'data': trip_data
        })
    
    return trips

def extract_features(trip):
    """提取特征"""
    try:
        td = trip['data']
        
        return {
            'trip_id': trip['trip_id'],
            'vehicle_id': trip['vehicle_id'],
            'duration_minutes': (td['datetime'].max() - td['datetime'].min()).total_seconds() / 60,
            'start_hour': td['datetime'].iloc[0].hour,
            'speed_mean': td['spd'].mean(),
            'speed_max': td['spd'].max(),
            'speed_std': td['spd'].std(),
            'speed_median': td['spd'].median(),
            'acc_mean': td.get('acc', pd.Series([0])).mean(),
            'acc_std': td.get('acc', pd.Series([0])).std(),
            'harsh_accel': (td.get('acc', pd.Series([0])) > 2).sum(),
            'harsh_decel': (td.get('acc', pd.Series([0])) < -2).sum(),
            'distance_total': td.get('distance_km', pd.Series([0])).sum(),
            'moving_ratio': td.get('is_moving', (td['spd'] > 1)).mean(),
            'soc_drop_total': td['soc'].iloc[0] - td['soc'].iloc[-1],
            'soc_mean': td['soc'].mean(),
            'soc_std': td['soc'].std(),
            'voltage_mean': td.get('v', pd.Series([0])).mean(),
            'voltage_std': td.get('v', pd.Series([0])).std(),
            'current_mean': td.get('i', pd.Series([0])).mean(),
            'current_max': td.get('i', pd.Series([0])).max(),
            'power_mean': td.get('power', pd.Series([0])).mean(),
            'power_max': td.get('power', pd.Series([0])).max(),
            'power_std': td.get('power', pd.Series([0])).std(),
            'energy_consumption_total': td.get('energy_consumption', pd.Series([0])).sum(),
            'efficiency_mean': 0,
            'low_speed_ratio': (td['spd'] < 20).mean(),
            'medium_speed_ratio': ((td['spd'] >= 20) & (td['spd'] < 60)).mean(),
            'high_speed_ratio': (td['spd'] >= 60).mean(),
            'charging_ratio': td.get('is_charging', pd.Series([0])).mean(),
            'regen_braking_ratio': td.get('is_regenerative_braking', pd.Series([0])).mean(),
            'is_weekend': td.get('is_weekend', pd.Series([0])).iloc[0]
        }
    except:
        return None

def process_one_file(file_path):
    """处理单个文件"""
    print(f"\n📄 Processing {os.path.basename(file_path)}...")
    
    all_vehicles = {}
    chunk_count = 0
    
    # 分块读取
    for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE):
        chunk['datetime'] = pd.to_datetime(chunk['datetime'])
        
        # 按车辆分组
        for vehicle_id, vehicle_data in chunk.groupby('vehicle_id'):
            if vehicle_id not in all_vehicles:
                all_vehicles[vehicle_id] = []
            all_vehicles[vehicle_id].append(vehicle_data)
        
        chunk_count += 1
        if chunk_count % 10 == 0:
            print(f"   Processed {chunk_count * CHUNK_SIZE / 1e6:.1f}M records, {len(all_vehicles)} vehicles", end='\r')
    
    print(f"   ✓ Loaded {len(all_vehicles)} vehicles")
    
    # 处理每辆车
    all_features = []
    for vid, data_list in all_vehicles.items():
        vehicle_data = pd.concat(data_list).sort_values('datetime')
        trips = segment_trips(vehicle_data, vid)
        
        for trip in trips:
            feat = extract_features(trip)
            if feat:
                all_features.append(feat)
    
    print(f"   ✓ Extracted {len(all_features)} trips")
    
    return all_features

# ==================== 并行处理7个文件 ====================
print(f"\n🔄 Processing {len(data_files)} files in parallel...")

with Pool(processes=min(N_PROCESSES, len(data_files))) as pool:
    results = pool.map(process_one_file, data_files)

# 合并结果
all_features = []
for file_features in results:
    all_features.extend(file_features)

# ==================== 合并跨天行程 ====================
print(f"\n🔗 Merging cross-day trips...")

features_df = pd.DataFrame(all_features)

# 按vehicle_id和时间排序
features_df = features_df.sort_values(['vehicle_id', 'trip_id'])

print(f"   Before merge: {len(features_df)} trips")

# 简单去重（相同vehicle + 相近时间）
features_df = features_df.drop_duplicates(subset=['vehicle_id', 'start_hour', 'duration_minutes', 'speed_mean'], keep='first')

print(f"   After merge: {len(features_df)} trips")

# ==================== 保存 ====================
print(f"\n{'='*70}")
print("💾 Saving Results")
print("="*70)

features_df.to_csv(f'{OUTPUT_DIR}/trip_features_full.csv', index=False)

print(f"\n✅ Saved {len(features_df)} trips")
print(f"   From {features_df['vehicle_id'].nunique()} vehicles")
print(f"   Average: {len(features_df) / features_df['vehicle_id'].nunique():.1f} trips/vehicle")

# ==================== 统计 ====================
print(f"\n📊 Statistics:")
print(f"\n⏱️  Duration: {features_df['duration_minutes'].mean():.1f} min (median: {features_df['duration_minutes'].median():.1f})")
print(f"🚗 Speed: {features_df['speed_mean'].mean():.1f} km/h (median: {features_df['speed_mean'].median():.1f})")
print(f"📏 Distance: {features_df['distance_total'].mean():.1f} km (median: {features_df['distance_total'].median():.1f})")

print(f"\n📊 Speed Distribution:")
low = (features_df['speed_mean'] < 20).sum()
med = ((features_df['speed_mean'] >= 20) & (features_df['speed_mean'] < 60)).sum()
high = (features_df['speed_mean'] >= 60).sum()

print(f"   Low (<20): {low} ({low/len(features_df)*100:.1f}%)")
print(f"   Med (20-60): {med} ({med/len(features_df)*100:.1f}%)")
print(f"   High (≥60): {high} ({high/len(features_df)*100:.1f}%)")

print("\n" + "="*70)
print("✅ Complete!")
print("="*70)