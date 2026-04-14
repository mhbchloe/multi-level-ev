"""
Step 1: Build Discharge-Charging Pairs (7 days, all vehicles)
使用7天完整数据，处理所有车辆的充放电事件
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict

print("="*70)
print("📊 Step 1: Building Discharge-Charging Pairs (7 Days, All Vehicles)")
print("="*70)

# ============ 1. Load Discharge Events ============
print("\n📂 Loading discharge events...")

df_events = pd.read_csv('./results/event_table.csv')
print(f"✅ Discharge events: {len(df_events):,}")
print(f"   Vehicles: {df_events['vehicle_id'].nunique():,}")
print(f"   Clusters: {sorted(df_events['cluster'].unique())}")

# ============ 2. Extract Charging Events from 7 Days ============
print("\n🔋 Extracting charging events from 7 days of GPS data...")

csv_files = [
    '20250701.csv',
    '20250702.csv',
    '20250703.csv',
    '20250704.csv',
    '20250705.csv',
    '20250706.csv',
    '20250707.csv'
]

all_charging_events = []
event_id_counter = 0

total_files = len(csv_files)

for file_idx, csv_file in enumerate(csv_files, 1):
    print(f"\n[{file_idx}/{total_files}] Processing: {csv_file}")
    
    try:
        # 分块读取
        chunk_iter = pd.read_csv(
            csv_file,
            chunksize=1_000_000,
            usecols=['vehicle_id', 'time', 'datetime', 'soc', 'is_charging', 'lat', 'lon'],
            on_bad_lines='skip',
            low_memory=False
        )
        
        # 收集每辆车的数据
        vehicle_data = defaultdict(list)
        
        for chunk_id, chunk in enumerate(chunk_iter):
            print(f"  Reading chunk {chunk_id+1}... ({len(chunk):,} rows)")
            
            for vehicle_id, group in chunk.groupby('vehicle_id'):
                vehicle_data[vehicle_id].append(group)
        
        print(f"  ✅ Loaded data for {len(vehicle_data):,} vehicles")
        
        # 识别充电事件
        print(f"  🔍 Identifying charging events...")
        
        file_charging_count = 0
        
        for vehicle_id, groups in tqdm(vehicle_data.items(), desc=f"  {csv_file}"):
            # 合并该车所有chunk
            df_vehicle = pd.concat(groups, ignore_index=True).sort_values('time')
            
            if len(df_vehicle) < 5:
                continue
            
            # 计算差值
            df_vehicle['soc_diff'] = df_vehicle['soc'].diff()
            df_vehicle['time_diff'] = df_vehicle['time'].diff()
            df_vehicle['lat_diff'] = df_vehicle['lat'].diff().abs()
            df_vehicle['lon_diff'] = df_vehicle['lon'].diff().abs()
            df_vehicle['location_change'] = np.sqrt(
                df_vehicle['lat_diff']**2 + df_vehicle['lon_diff']**2
            )
            
            # 充电条件（放宽标准）
            charging_mask = (
                (df_vehicle['is_charging'] == 1) & 
                (df_vehicle['soc_diff'] > 0) &
                (df_vehicle['location_change'] < 0.002)  # 放宽位置稳定条件
            )
            
            if not charging_mask.any():
                continue
            
            charging_segments = df_vehicle[charging_mask].copy()
            charging_segments['time_gap'] = charging_segments['time'].diff()
            
            # 识别连续充电片段
            current_event = []
            
            for idx, row in charging_segments.iterrows():
                if len(current_event) == 0 or (row['time_gap'] < 1200):  # 20分钟内
                    current_event.append(row)
                else:
                    # 保存上一个充电事件
                    if len(current_event) >= 3:  # 至少3个点
                        all_charging_events.append({
                            'charging_event_id': event_id_counter,
                            'vehicle_id': vehicle_id,
                            'start_time': current_event[0]['time'],
                            'end_time': current_event[-1]['time'],
                            'soc_start': current_event[0]['soc'],
                            'soc_end': current_event[-1]['soc'],
                            'soc_gain': current_event[-1]['soc'] - current_event[0]['soc'],
                            'duration_seconds': current_event[-1]['time'] - current_event[0]['time'],
                            'n_points': len(current_event),
                            'lat': np.mean([p['lat'] for p in current_event]),
                            'lon': np.mean([p['lon'] for p in current_event]),
                        })
                        event_id_counter += 1
                        file_charging_count += 1
                    
                    current_event = [row]
            
            # 处理最后一个事件
            if len(current_event) >= 3:
                all_charging_events.append({
                    'charging_event_id': event_id_counter,
                    'vehicle_id': vehicle_id,
                    'start_time': current_event[0]['time'],
                    'end_time': current_event[-1]['time'],
                    'soc_start': current_event[0]['soc'],
                    'soc_end': current_event[-1]['soc'],
                    'soc_gain': current_event[-1]['soc'] - current_event[0]['soc'],
                    'duration_seconds': current_event[-1]['time'] - current_event[0]['time'],
                    'n_points': len(current_event),
                    'lat': np.mean([p['lat'] for p in current_event]),
                    'lon': np.mean([p['lon'] for p in current_event]),
                })
                event_id_counter += 1
                file_charging_count += 1
        
        print(f"  ✅ Found {file_charging_count:,} charging events in {csv_file}")
        
    except Exception as e:
        print(f"  ⚠️  Error processing {csv_file}: {str(e)}")
        continue

# 转为DataFrame
df_charging = pd.DataFrame(all_charging_events)

print(f"\n{'='*70}")
print(f"✅ Charging Event Extraction Complete!")
print(f"{'='*70}")
print(f"   Total charging events: {len(df_charging):,}")
print(f"   Unique vehicles: {df_charging['vehicle_id'].nunique():,}")
print(f"   Avg SOC gain: {df_charging['soc_gain'].mean():.2f}%")
print(f"   Avg duration: {df_charging['duration_seconds'].mean()/60:.1f} minutes")

# 保存充电事件
df_charging.to_csv('./results/charging_events_7days.csv', index=False, encoding='utf-8-sig')
print(f"\n💾 Saved: charging_events_7days.csv")

# ============ 3. Build Discharge-Charging Pairs ============
print(f"\n{'='*70}")
print(f"🔗 Building Discharge-Charging Pairs")
print(f"{'='*70}")

pairs = []

# 获取有充电事件的车辆
vehicles_with_charging = set(df_charging['vehicle_id'].unique())
vehicles_with_discharge = set(df_events['vehicle_id'].unique())
common_vehicles = vehicles_with_charging & vehicles_with_discharge

print(f"\n   Vehicles with discharge events: {len(vehicles_with_discharge):,}")
print(f"   Vehicles with charging events: {len(vehicles_with_charging):,}")
print(f"   Common vehicles: {len(common_vehicles):,}")

for vehicle_id in tqdm(common_vehicles, desc="Matching pairs"):
    # 该车的放电和充电事件
    vehicle_discharges = df_events[df_events['vehicle_id'] == vehicle_id].sort_values('start_time')
    vehicle_chargings = df_charging[df_charging['vehicle_id'] == vehicle_id].sort_values('start_time')
    
    # 为每个放电事件找下一次充电
    for _, discharge in vehicle_discharges.iterrows():
        # 找到放电结束后的充电事件
        next_charging = vehicle_chargings[
            vehicle_chargings['start_time'] > discharge['end_time']
        ]
        
        if len(next_charging) == 0:
            continue
        
        next_charging = next_charging.iloc[0]
        
        # 时间间隔
        time_to_next_charging = next_charging['start_time'] - discharge['end_time']
        
        # 限制在3天内
        if time_to_next_charging > 86400 * 3:
            continue
        
        # 构建配对
        pair = {
            # IDs
            'discharge_event_id': discharge['event_id'],
            'charging_event_id': next_charging['charging_event_id'],
            'vehicle_id': vehicle_id,
            
            # 时间
            'discharge_start_time': discharge['start_time'],
            'discharge_end_time': discharge['end_time'],
            'charging_start_time': next_charging['start_time'],
            'time_to_next_charging': time_to_next_charging,
            
            # 放电特征
            'discharge_cluster': discharge['cluster'],
            'discharge_duration': discharge['duration_seconds'],
            'discharge_distance': discharge.get('distance_km', 0),
            'discharge_speed_mean': discharge.get('speed_mean', 0),
            'discharge_speed_std': discharge.get('speed_std', 0),
            'discharge_harsh_accel': discharge.get('harsh_accel_count', 0),
            'discharge_harsh_brake': discharge.get('harsh_brake_count', 0),
            'discharge_idle_ratio': discharge.get('idle_ratio', 0),
            'discharge_soc_start': discharge['soc_start'],
            'discharge_soc_end': discharge['soc_end'],
            'discharge_soc_drop': discharge['soc_drop'],
            'discharge_energy_kwh': discharge.get('energy_consumption_kwh', 0),
            'discharge_efficiency_kwh_per_km': discharge.get('efficiency_kwh_per_km', 0),
            
            # 充电行为（因变量）
            'charging_soc_start': next_charging['soc_start'],
            'charging_soc_end': next_charging['soc_end'],
            'charging_soc_gain': next_charging['soc_gain'],
            'charging_duration': next_charging['duration_seconds'],
        }
        
        pairs.append(pair)

df_pairs = pd.DataFrame(pairs)

print(f"\n{'='*70}")
print(f"✅ Pair Matching Complete!")
print(f"{'='*70}")
print(f"   Total pairs: {len(df_pairs):,}")
print(f"   Unique vehicles: {df_pairs['vehicle_id'].nunique():,}")

# ============ 4. 统计摘要 ============
print(f"\n📊 Discharge Cluster Distribution:")
cluster_dist = df_pairs['discharge_cluster'].value_counts().sort_index()
for cluster, count in cluster_dist.items():
    print(f"   Cluster {cluster}: {count:,} pairs ({count/len(df_pairs)*100:.1f}%)")

print(f"\n📊 Charging Behavior Summary:")
print(f"   Avg charging trigger SOC: {df_pairs['charging_soc_start'].mean():.1f}%")
print(f"   Avg charging gain: {df_pairs['charging_soc_gain'].mean():.1f}%")
print(f"   Avg charging duration: {df_pairs['charging_duration'].mean()/60:.1f} minutes")
print(f"   Avg time to next charging: {df_pairs['time_to_next_charging'].mean()/3600:.1f} hours")

print(f"\n📊 By Discharge Cluster:")
for cluster in sorted(df_pairs['discharge_cluster'].unique()):
    cluster_data = df_pairs[df_pairs['discharge_cluster'] == cluster]
    print(f"\n   Cluster {cluster} (n={len(cluster_data):,}):")
    print(f"      Avg trigger SOC: {cluster_data['charging_soc_start'].mean():.1f}%")
    print(f"      Avg charging gain: {cluster_data['charging_soc_gain'].mean():.1f}%")
    print(f"      Avg time to charging: {cluster_data['time_to_next_charging'].mean()/3600:.1f} hours")

# ============ 5. 保存结果 ============
df_pairs.to_csv('./results/discharge_charging_pairs_7days_all_vehicles.csv', 
                index=False, encoding='utf-8-sig')

print(f"\n{'='*70}")
print(f"💾 Saved: discharge_charging_pairs_7days_all_vehicles.csv")
print(f"{'='*70}")

# ============ 6. 数据质量检查 ============
print(f"\n{'='*70}")
print(f"🔍 Data Quality Check")
print(f"{'='*70}")

print(f"\n缺失值检查:")
missing_counts = df_pairs.isnull().sum()
if missing_counts.sum() > 0:
    print(missing_counts[missing_counts > 0])
else:
    print("   ✅ No missing values")

print(f"\n异常值检查:")
print(f"   Charging SOC start range: [{df_pairs['charging_soc_start'].min():.1f}, {df_pairs['charging_soc_start'].max():.1f}]")
print(f"   Charging SOC gain range: [{df_pairs['charging_soc_gain'].min():.1f}, {df_pairs['charging_soc_gain'].max():.1f}]")
print(f"   Time to charging range: [{df_pairs['time_to_next_charging'].min()/3600:.1f}h, {df_pairs['time_to_next_charging'].max()/3600:.1f}h]")

# 检查负值
if (df_pairs['charging_soc_gain'] < 0).any():
    print(f"   ⚠️  Warning: {(df_pairs['charging_soc_gain'] < 0).sum()} negative charging gains")

print(f"\n{'='*70}")
print(f"✅ Step 1 Complete!")
print(f"{'='*70}")
print(f"\n📁 Generated files:")
print(f"   1. charging_events_7days.csv ({len(df_charging):,} events)")
print(f"   2. discharge_charging_pairs_7days_all_vehicles.csv ({len(df_pairs):,} pairs)")
print(f"\n🚀 Next step:")
print(f"   python step2_statistical_analysis_fixed.py")
print(f"{'='*70}")