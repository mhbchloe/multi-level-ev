"""
灵活选择覆盖天数的车辆进行分析
可以用28天、25天、20天等的车，平衡样本量和数据质量
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import os
import glob

os.makedirs('./analysis_complete_vehicles/results', exist_ok=True)

print("="*70)
print("🔄 Flexible Vehicle Selection & Analysis")
print("="*70)

# ============ 1. 加载覆盖数据 ============
print("\n📂 Loading vehicle coverage data...")

df_coverage = pd.read_csv('./analysis_complete_vehicles/results/vehicle_coverage_31days.csv')

# 显示可选方案
print(f"\n{'='*70}")
print(f"📊 Available Vehicle Selection Options:")
print(f"{'='*70}\n")

options = {
    '31': 31,
    '28': 28,
    '25': 25,
    '20': 20,
    '15': 15,
}

for key, days in options.items():
    count = len(df_coverage[df_coverage['total_days'] >= days])
    pct = count / len(df_coverage) * 100
    print(f"   Option {key}: >= {days} days -> {count:,} vehicles ({pct:.1f}%)")

# ============ 选择方案 ============
print(f"\n{'─'*70}")
print(f"🎯 RECOMMENDATION: Use >= 28 days (Option 28)")
print(f"   Reason: Balances data quality (~75% coverage) with large sample size")
print(f"{'─'*70}\n")

# 默认选择28天
selected_days = 28
complete_vehicle_ids = set(df_coverage[df_coverage['total_days'] >= selected_days]['vehicle_id'].tolist())

print(f"✅ Selected: {len(complete_vehicle_ids):,} vehicles (>= {selected_days} days)")

if len(complete_vehicle_ids) == 0:
    print("❌ No vehicles found!")
    exit()

# ============ 2. 找processed文件 ============
print(f"\n📂 Scanning for processed CSV files...")

processed_files = sorted(glob.glob('./data_20250*_processed.csv'))
if not processed_files:
    processed_files = sorted(glob.glob('./data_processed_one_month/data_*_processed.csv'))
if not processed_files:
    processed_files = sorted(glob.glob('./**/data_*_processed.csv', recursive=True))

print(f"Found {len(processed_files)} processed files")

if not processed_files:
    print("❌ No processed files found!")
    exit()

# ============ 3. 提取充电事件 ============
print(f"\n{'='*70}")
print(f"⚡ Extracting Charging Events")
print(f"{'='*70}\n")

all_charging_events = []
event_id = 0
vehicles_with_data = set()

for csv_file in tqdm(processed_files, desc="Processing files"):
    date_str = os.path.basename(csv_file).split('_')[1]
    
    try:
        reader = pd.read_csv(
            csv_file,
            chunksize=100_000,
            usecols=['vehicle_id', 'time', 'soc', 'is_charging', 'lat', 'lon'],
            on_bad_lines='skip',
            dtype={'vehicle_id': 'str', 'soc': 'float', 'is_charging': 'int'}
        )
        
        vehicle_data = defaultdict(list)
        
        for chunk in reader:
            chunk = chunk.dropna(subset=['vehicle_id', 'time', 'soc'])
            chunk['time'] = pd.to_datetime(chunk['time'], format='mixed', errors='coerce')
            chunk = chunk.dropna(subset=['time'])
            chunk = chunk[chunk['vehicle_id'].isin(complete_vehicle_ids)]
            
            if len(chunk) == 0:
                continue
            
            for vehicle_id, group in chunk.groupby('vehicle_id'):
                vehicle_data[vehicle_id].append(group)
                vehicles_with_data.add(vehicle_id)
        
        # 识别充电事件
        for vehicle_id, groups in vehicle_data.items():
            if not groups:
                continue
                
            df_vehicle = pd.concat(groups, ignore_index=True)
            df_vehicle = df_vehicle.sort_values('time').reset_index(drop=True)
            
            if len(df_vehicle) < 3:
                continue
            
            df_vehicle['soc_diff'] = df_vehicle['soc'].diff()
            df_vehicle['lat_diff'] = df_vehicle['lat'].diff().abs()
            df_vehicle['lon_diff'] = df_vehicle['lon'].diff().abs()
            df_vehicle['location_change'] = np.sqrt(df_vehicle['lat_diff']**2 + df_vehicle['lon_diff']**2)
            
            charging_mask = (
                (df_vehicle['is_charging'] == 1) & 
                (df_vehicle['soc_diff'] > 0) &
                (df_vehicle['location_change'] < 0.002)
            )
            
            if not charging_mask.any():
                continue
            
            charging_segments = df_vehicle[charging_mask].copy()
            charging_segments['time_gap'] = charging_segments['time'].diff().dt.total_seconds()
            charging_segments = charging_segments.dropna(subset=['time_gap'])
            
            if len(charging_segments) == 0:
                continue
            
            current_event = []
            
            for _, row in charging_segments.iterrows():
                time_gap = row['time_gap']
                if len(current_event) == 0 or (pd.notna(time_gap) and time_gap < 1200):
                    current_event.append(row)
                else:
                    if len(current_event) >= 3:
                        event_times = [r['time'] for r in current_event]
                        event_socs = [r['soc'] for r in current_event]
                        
                        try:
                            all_charging_events.append({
                                'charging_event_id': event_id,
                                'vehicle_id': str(vehicle_id),
                                'date': date_str,
                                'start_time': event_times[0],
                                'end_time': event_times[-1],
                                'soc_start': float(event_socs[0]),
                                'soc_end': float(event_socs[-1]),
                                'soc_gain': float(event_socs[-1]) - float(event_socs[0]),
                                'duration_seconds': (event_times[-1] - event_times[0]).total_seconds(),
                                'num_records': len(current_event),
                            })
                            event_id += 1
                        except:
                            pass
                    
                    current_event = [row]
            
            if len(current_event) >= 3:
                event_times = [r['time'] for r in current_event]
                event_socs = [r['soc'] for r in current_event]
                
                try:
                    all_charging_events.append({
                        'charging_event_id': event_id,
                        'vehicle_id': str(vehicle_id),
                        'date': date_str,
                        'start_time': event_times[0],
                        'end_time': event_times[-1],
                        'soc_start': float(event_socs[0]),
                        'soc_end': float(event_socs[-1]),
                        'soc_gain': float(event_socs[-1]) - float(event_socs[0]),
                        'duration_seconds': (event_times[-1] - event_times[0]).total_seconds(),
                        'num_records': len(current_event),
                    })
                    event_id += 1
                except:
                    pass
    
    except Exception as e:
        print(f"   ⚠️  Error in {os.path.basename(csv_file)}: {str(e)[:60]}")
        continue

if len(all_charging_events) == 0:
    print("❌ No charging events extracted!")
    exit()

df_charging = pd.DataFrame(all_charging_events)

print(f"\n✅ Extracted {len(df_charging):,} charging events")
print(f"   From {df_charging['vehicle_id'].nunique():,} vehicles")
print(f"   Vehicles with charging data: {len(vehicles_with_data):,} / {len(complete_vehicle_ids):,}")

print(f"\n📊 Charging Event Statistics:")
print(f"   Avg SOC gain: {df_charging['soc_gain'].mean():.2f}%")
print(f"   Median SOC gain: {df_charging['soc_gain'].median():.2f}%")
print(f"   Avg duration: {df_charging['duration_seconds'].mean()/60:.1f} minutes")
print(f"   Min SOC gain: {df_charging['soc_gain'].min():.2f}%")
print(f"   Max SOC gain: {df_charging['soc_gain'].max():.2f}%")

df_charging.to_csv(f'./analysis_complete_vehicles/results/charging_events_{selected_days}days.csv', index=False)
print(f"\n💾 Saved: charging_events_{selected_days}days.csv")

# ============ 4. 构建配对 ============
print(f"\n{'='*70}")
print(f"🔗 Building Charging Event Pairs")
print(f"{'='*70}\n")

pairs = []

for vehicle_id in tqdm(vehicles_with_data, desc="Building pairs"):
    vehicle_id_str = str(vehicle_id)
    
    vehicle_chargings = df_charging[df_charging['vehicle_id'] == vehicle_id_str].sort_values('start_time').reset_index(drop=True)
    
    if len(vehicle_chargings) < 2:
        continue
    
    for idx in range(len(vehicle_chargings) - 1):
        current = vehicle_chargings.iloc[idx]
        next_charging = vehicle_chargings.iloc[idx + 1]
        
        try:
            time_to_next = (next_charging['start_time'] - current['end_time']).total_seconds()
            
            if time_to_next > 86400 * 3 or time_to_next < 0:
                continue
            
            pairs.append({
                'pair_id': len(pairs),
                'vehicle_id': vehicle_id_str,
                'charging_event_1_id': int(current['charging_event_id']),
                'charging_event_2_id': int(next_charging['charging_event_id']),
                'charge_1_date': current['date'],
                'charge_2_date': next_charging['date'],
                'charge_1_soc_start': float(current['soc_start']),
                'charge_1_soc_end': float(current['soc_end']),
                'charge_1_gain': float(current['soc_gain']),
                'charge_2_soc_start': float(next_charging['soc_start']),
                'charge_2_soc_end': float(next_charging['soc_end']),
                'charge_2_gain': float(next_charging['soc_gain']),
                'time_between_charges': time_to_next,
                'charge_1_duration': float(current['duration_seconds']),
                'charge_2_duration': float(next_charging['duration_seconds']),
            })
        except:
            continue

if len(pairs) > 0:
    df_pairs = pd.DataFrame(pairs)
    
    print(f"\n✅ Built {len(df_pairs):,} charge-to-charge pairs")
    print(f"   From {df_pairs['vehicle_id'].nunique():,} vehicles")
    print(f"   Avg time between charges: {df_pairs['time_between_charges'].mean()/3600:.1f} hours")
    print(f"   Median time between charges: {df_pairs['time_between_charges'].median()/3600:.1f} hours")
    
    df_pairs.to_csv(f'./analysis_complete_vehicles/results/charging_pairs_{selected_days}days.csv', index=False)
    print(f"\n💾 Saved: charging_pairs_{selected_days}days.csv")

# ============ 汇总 ============
print(f"\n{'='*70}")
print(f"✅ Analysis Complete")
print(f"{'='*70}")

print(f"\n📊 FINAL SUMMARY:")
print(f"   Selection criteria: >= {selected_days} days of data")
print(f"   Total vehicles selected: {len(complete_vehicle_ids):,}")
print(f"   Vehicles with charging events: {len(vehicles_with_data):,}")
print(f"   Total charging events: {len(df_charging):,}")
print(f"   Total charge pairs: {len(pairs):,}")

print(f"\n📁 Output files:")
print(f"   ✅ charging_events_{selected_days}days.csv")
print(f"   ✅ charging_pairs_{selected_days}days.csv")
print(f"\n{'='*70}")