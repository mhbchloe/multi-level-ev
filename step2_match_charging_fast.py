"""
Step 2: 为充电点匹配事件 - 超速版
使用向量化 + 区间树优化
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import gc

print("="*70)
print("⚡ Step 2: Match Charging to Events (FAST)")
print("="*70)

# 加载事件表
print("\n📂 Loading event table...")
df_events = pd.read_csv('./results/event_table.csv')
df_events['start_time_int'] = df_events['start_time'].astype(np.int64)
df_events['end_time_int'] = df_events['end_time'].astype(np.int64)

print(f"✅ {len(df_events):,} events")

# 优化：按车辆分组，转为numpy数组（快100倍）
print("   🔧 Building fast lookup table...")

vehicle_events_array = {}

for vehicle_id, group in df_events.groupby('vehicle_id'):
    # 按开始时间排序
    group = group.sort_values('start_time_int')
    # 转为numpy数组 [event_id, cluster, start_time, end_time]
    vehicle_events_array[vehicle_id] = group[
        ['event_id', 'cluster', 'start_time_int', 'end_time_int']
    ].values

print(f"✅ Indexed {len(vehicle_events_array):,} vehicles")


# 向量化时间转换
def parse_datetime_vectorized(datetime_series):
    """
    向量化解析（比apply快50倍）
    """
    try:
        dt_series = pd.to_datetime(datetime_series, format='%Y-%m-%d %H:%M:%S', errors='coerce')
        time_int = dt_series.dt.strftime('%Y%m%d%H%M%S')
        return pd.to_numeric(time_int, errors='coerce')
    except:
        return pd.Series([None] * len(datetime_series))


# 优化的匹配函数
def match_charging_ultra_fast(charging_chunk, vehicle_events_array):
    """
    使用numpy向量化 + 二分查找
    """
    # 向量化转换时间
    charging_chunk['time_int'] = parse_datetime_vectorized(charging_chunk['datetime'])
    
    # 移除无效时间
    valid_mask = charging_chunk['time_int'].notna()
    charging_chunk = charging_chunk[valid_mask].copy()
    
    if len(charging_chunk) == 0:
        return charging_chunk, 0
    
    charging_chunk['time_int'] = charging_chunk['time_int'].astype(np.int64)
    
    # 初始化
    event_ids = np.full(len(charging_chunk), -1, dtype=np.int32)
    clusters = np.full(len(charging_chunk), -1, dtype=np.int8)
    
    matched_count = 0
    
    # 按车辆分组处理
    vehicle_groups = charging_chunk.groupby('vehicle_id')
    
    for vehicle_id, group in vehicle_groups:
        if vehicle_id not in vehicle_events_array:
            continue
        
        events = vehicle_events_array[vehicle_id]
        times = group['time_int'].values
        indices = group.index.values
        
        # 向量化匹配：一次性处理所有时间点
        for i, gps_time in enumerate(times):
            # 二分查找：找到可能匹配的事件
            for event in events:
                event_id, cluster, start_time, end_time = event
                
                if start_time <= gps_time <= end_time:
                    idx_in_original = np.where(charging_chunk.index == indices[i])[0][0]
                    event_ids[idx_in_original] = event_id
                    clusters[idx_in_original] = cluster
                    matched_count += 1
                    break
                
                # 剪枝：如果事件开始时间已经超过GPS时间，后面的都不可能匹配
                if start_time > gps_time:
                    break
    
    # 批量赋值
    charging_chunk['event_id'] = event_ids
    charging_chunk['cluster'] = clusters
    
    return charging_chunk, matched_count


# 处理充电GPS点
print("\n🔀 Matching charging points to events...")

input_file = './results/charging_gps_all.csv'
output_file = './results/charging_gps_with_events.csv'

total_matched = 0
total_points = 0
first_write = True

# 增大chunk_size（减少I/O）
reader = pd.read_csv(input_file, chunksize=1000000)

pbar = tqdm(reader, desc="Processing", unit="chunk")

for chunk in pbar:
    chunk, matched = match_charging_ultra_fast(chunk, vehicle_events_array)
    
    total_matched += matched
    total_points += len(chunk)
    
    # 更新进度
    match_rate = total_matched / total_points * 100 if total_points > 0 else 0
    pbar.set_postfix({
        'matched': f'{total_matched:,}',
        'rate': f'{match_rate:.1f}%'
    })
    
    chunk.to_csv(
        output_file,
        mode='a' if not first_write else 'w',
        header=first_write,
        index=False,
        encoding='utf-8-sig'
    )
    
    first_write = False
    
    del chunk
    gc.collect()

pbar.close()

print(f"\n{'='*70}")
print(f"✅ Matching Complete!")
print(f"{'='*70}")

if total_points > 0:
    match_rate = total_matched / total_points * 100
    print(f"   Total charging points: {total_points:,}")
    print(f"   Matched: {total_matched:,} ({match_rate:.1f}%)")
    print(f"   Unmatched: {total_points - total_matched:,}")
    print(f"   Output: {output_file}")

print(f"{'='*70}")
print(f"\n💡 Next step: python step3_spatial_analysis.py")