"""
Step 12: Extract Inter-Charge Trips (Robust Version)
放宽时间边界，增加容错机制，确保最大化提取行程
"""

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🔗 Step 12: Building Inter-Charge Trip Dataset (Robust)")
print("="*70)

input_dir = "./analysis_complete_vehicles/results/"
macro_dir = "./vehicle_clustering/results/"
output_dir = "./coupling_analysis/results/"
os.makedirs(output_dir, exist_ok=True)

# 1. 加载数据
print("📂 Loading data...")
df_segments = pd.read_csv(os.path.join(input_dir, 'segments_with_clusters_labeled.csv'))
df_charging = pd.read_csv(os.path.join(input_dir, 'charging_events_rebuilt.csv'))
df_vehicles = pd.read_csv(os.path.join(macro_dir, 'vehicle_clustering_results_v3.csv'))

# 2. 强健的时间格式转换函数
def parse_time_safe(series):
    if pd.api.types.is_numeric_dtype(series):
        if series.max() > 1e11:
            return pd.to_datetime(series, unit='ms', errors='coerce')
        else:
            return pd.to_datetime(series, unit='s', errors='coerce')
    else:
        return pd.to_datetime(series, errors='coerce')

print("⏳ Standardizing time formats...")
df_segments['start_dt'] = parse_time_safe(df_segments['start_time'])
df_charging['start_dt'] = parse_time_safe(df_charging['start_time'])

# 剔除转换失败的数据
df_segments = df_segments.dropna(subset=['start_dt'])
df_charging = df_charging.dropna(subset=['start_dt'])

trips = []
vehicles = df_charging['vehicle_id'].unique()

print(f"🚗 Processing {len(vehicles):,} vehicles that have charging events...")

for vid in tqdm(vehicles, desc="Extracting Trips"):
    v_charges = df_charging[df_charging['vehicle_id'] == vid].sort_values('start_dt').reset_index(drop=True)
    v_segs = df_segments[df_segments['vehicle_id'] == vid].sort_values('start_dt')
    
    if len(v_charges) == 0 or len(v_segs) == 0:
        continue
        
    earliest_seg_time = v_segs['start_dt'].min()
    
    # 遍历充电事件
    for i in range(len(v_charges)):
        current_charge = v_charges.iloc[i]
        
        # 充电开始时间（往后退1小时作为容差，防止日志时间漂移）
        charge_start_time = current_charge['start_dt'] + pd.Timedelta(hours=1)
        
        # 行程起点（第一次充电从车辆最早数据开始算，往前推1小时作为容差）
        if i == 0:
            trip_start_time = earliest_seg_time - pd.Timedelta(hours=1)
        else:
            # 取上一次充电的开始时间作为起点即可
            trip_start_time = v_charges.iloc[i-1]['start_dt']
        
        # 只要放电片段的开始时间落在这个大区间内，就算作本次行程的片段！
        inter_segs = v_segs[
            (v_segs['start_dt'] > trip_start_time) & 
            (v_segs['start_dt'] < charge_start_time)
        ]
        
        # 放宽条件：只要有 1 个 3%的片段，就算一次有效行程
        if len(inter_segs) >= 1:
            cluster_counts = inter_segs['cluster'].value_counts(normalize=True).to_dict()
            end_segs = inter_segs.tail(2)
            end_power = end_segs['power_mean'].mean()
            
            trip_data = {
                'trip_id': f"{vid}_trip_{i}",
                'vehicle_id': vid,
                'num_3pct_segments': len(inter_segs),
                
                'ratio_moderate': cluster_counts.get(0, 0.0),
                'ratio_conservative': cluster_counts.get(1, 0.0),
                'ratio_aggressive': cluster_counts.get(2, 0.0),
                'ratio_highway': cluster_counts.get(3, 0.0),
                
                'trip_avg_power': inter_segs['power_mean'].mean(),
                'trip_avg_speed': inter_segs['speed_mean'].mean(),
                'trip_acc_std': inter_segs['acc_std'].mean(),
                'trip_total_soc_drop': inter_segs['soc_drop'].sum(),
                'end_stage_power': end_power,
                
                'charge_trigger_soc': current_charge['soc_start'],
                'charge_gain_soc': current_charge['soc_gain']
            }
            trips.append(trip_data)

df_trips = pd.DataFrame(trips)

if len(df_trips) == 0:
    print("❌ 仍然没有找到行程！我们打印一下第一个车辆的时间看看为什么：")
    vid = vehicles[0]
    c_time = df_charging[df_charging['vehicle_id']==vid]['start_dt'].iloc[0]
    s_time = df_segments[df_segments['vehicle_id']==vid]['start_dt'].iloc[0]
    print(f"Vehicle: {vid}")
    print(f"First Charge Time: {c_time}")
    print(f"First Segment Time: {s_time}")
else:
    df_trips = df_trips.merge(
        df_vehicles[['vehicle_id', 'vehicle_type', 'cluster']], 
        on='vehicle_id', 
        how='inner'
    ).rename(columns={'cluster': 'vehicle_archetype_id'})

    df_trips.to_csv(os.path.join(output_dir, 'inter_charge_trips.csv'), index=False)
    print(f"\n✅ 成功提取！ Created Trip Dataset: {len(df_trips):,} trips extracted.")
    print(f"✅ Average segments per trip: {df_trips['num_3pct_segments'].mean():.1f}")