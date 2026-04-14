"""
Phase 1: 从原始GPS数据提取车辆完整行为特征
不依赖阉割的事件表，重新计算
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime

print("="*70)
print("🚗 Phase 1: Extract Complete Vehicle Behavior")
print("="*70)

# Step 1: 加载原始数据（逐日处理，避免内存爆炸）
print("\n📂 Processing raw GPS data...")

csv_files = [
    '20250701.csv', '20250702.csv', '20250703.csv',
    '20250704.csv', '20250705.csv', '20250706.csv', '20250707.csv'
]

vehicle_stats = {}

for file_idx, csv_file in enumerate(csv_files, 1):
    print(f"\n[{file_idx}/7] {csv_file}")
    
    # 分块读取
    reader = pd.read_csv(
        csv_file,
        chunksize=1000000,
        usecols=['vehicle_id', 'datetime', 'soc', 'is_charging', 
                 'distance_km', 'spd', 'lat', 'lon'],
        encoding='utf-8',
        on_bad_lines='skip'
    )
    
    for chunk in tqdm(reader, desc=f"  {csv_file}"):
        # 按车辆分组统计
        for vehicle_id, group in chunk.groupby('vehicle_id'):
            if vehicle_id not in vehicle_stats:
                vehicle_stats[vehicle_id] = {
                    'total_points': 0,
                    'charging_points': 0,
                    'driving_points': 0,
                    'total_distance': 0,
                    'soc_samples': [],
                    'charging_events': [],
                    'daily_data': {}
                }
            
            stats = vehicle_stats[vehicle_id]
            
            # 基础统计
            stats['total_points'] += len(group)
            stats['charging_points'] += (group['is_charging'] == 1).sum()
            stats['driving_points'] += (group['spd'] > 0).sum()
            stats['total_distance'] += group['distance_km'].sum()
            
            # SOC采样（用于分析充电触发点）
            stats['soc_samples'].extend(group['soc'].dropna().tolist())
            
            # 识别充电事件（简单版：连续充电片段）
            group = group.sort_values('datetime')
            group['soc_diff'] = group['soc'].diff()
            
            charging_mask = (group['is_charging'] == 1) & (group['soc_diff'] > 0)
            
            if charging_mask.any():
                charging_segments = group[charging_mask].copy()
                
                # 识别连续充电片段
                charging_segments['time_gap'] = pd.to_datetime(charging_segments['datetime']).diff().dt.total_seconds()
                
                current_event = []
                for idx, row in charging_segments.iterrows():
                    if len(current_event) == 0 or row['time_gap'] < 600:  # 10分钟内算同一事件
                        current_event.append(row['soc'])
                    else:
                        if len(current_event) >= 3:  # 至少3个点
                            stats['charging_events'].append({
                                'soc_start': current_event[0],
                                'soc_end': current_event[-1],
                                'soc_gain': current_event[-1] - current_event[0]
                            })
                        current_event = [row['soc']]
                
                # 处理最后一个事件
                if len(current_event) >= 3:
                    stats['charging_events'].append({
                        'soc_start': current_event[0],
                        'soc_end': current_event[-1],
                        'soc_gain': current_event[-1] - current_event[0]
                    })
            
            # 按日统计
            group['date'] = pd.to_datetime(group['datetime']).dt.date
            for date, day_group in group.groupby('date'):
                date_str = str(date)
                if date_str not in stats['daily_data']:
                    stats['daily_data'][date_str] = {
                        'distance': 0,
                        'charging_count': 0,
                        'min_soc': 100,
                        'max_soc': 0
                    }
                
                daily = stats['daily_data'][date_str]
                daily['distance'] += day_group['distance_km'].sum()
                daily['charging_count'] += len([e for e in stats['charging_events'] if True])  # 简化
                
                soc_values = day_group['soc'].dropna()
                if len(soc_values) > 0:
                    daily['min_soc'] = min(daily['min_soc'], soc_values.min())
                    daily['max_soc'] = max(daily['max_soc'], soc_values.max())

print(f"\n✅ Processed {len(vehicle_stats):,} vehicles")

# Step 2: 计算车辆级别指标
print("\n📊 Computing vehicle-level metrics...")

vehicle_features = []

for vehicle_id, stats in tqdm(vehicle_stats.items(), desc="Computing"):
    n_charging_events = len(stats['charging_events'])
    
    if n_charging_events == 0:
        continue
    
    # 充电触发SOC
    charging_trigger_socs = [e['soc_start'] for e in stats['charging_events']]
    
    # 日均指标
    n_days = len(stats['daily_data'])
    total_daily_distance = sum(d['distance'] for d in stats['daily_data'].values())
    
    # SOC分布
    soc_samples = stats['soc_samples']
    
    feature = {
        'vehicle_id': vehicle_id,
        
        # 基础统计
        'total_points': stats['total_points'],
        'total_distance_km': stats['total_distance'],
        'n_days': n_days,
        
        # 充电行为
        'n_charging_events': n_charging_events,
        'charging_freq_per_day': n_charging_events / n_days if n_days > 0 else 0,
        'charging_freq_per_100km': n_charging_events / (stats['total_distance'] / 100) if stats['total_distance'] > 0 else 0,
        
        # 充电触发点
        'avg_charging_trigger_soc': np.mean(charging_trigger_socs),
        'median_charging_trigger_soc': np.median(charging_trigger_socs),
        'min_charging_trigger_soc': np.min(charging_trigger_socs),
        'std_charging_trigger_soc': np.std(charging_trigger_socs),
        
        # 充电量
        'avg_charging_gain': np.mean([e['soc_gain'] for e in stats['charging_events']]),
        'median_charging_gain': np.median([e['soc_gain'] for e in stats['charging_events']]),
        
        # SOC统计
        'avg_soc': np.mean(soc_samples) if soc_samples else np.nan,
        'min_soc_ever': np.min(soc_samples) if soc_samples else np.nan,
        'soc_25percentile': np.percentile(soc_samples, 25) if soc_samples else np.nan,
        
        # 日均行为
        'avg_daily_distance': total_daily_distance / n_days if n_days > 0 else 0,
        'avg_daily_charging': n_charging_events / n_days if n_days > 0 else 0,
    }
    
    vehicle_features.append(feature)

df_vehicle_features = pd.DataFrame(vehicle_features)

print(f"✅ Computed features for {len(df_vehicle_features):,} vehicles")

# 保存
df_vehicle_features.to_csv('./results/vehicle_behavior_features.csv', index=False, encoding='utf-8-sig')
print(f"💾 Saved: vehicle_behavior_features.csv")

print(f"\n📊 Sample statistics:")
print(df_vehicle_features[['avg_charging_trigger_soc', 'charging_freq_per_day', 
                           'charging_freq_per_100km', 'avg_charging_gain']].describe())

print(f"\n{'='*70}")
print(f"✅ Phase 1 Complete!")
print(f"{'='*70}")