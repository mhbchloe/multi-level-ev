"""
事件表与GPS数据融合 - 调试版
增加详细的调试信息
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import gc
import os
import time
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🔗 Event-GPS Data Fusion (Debug Mode)")
print("="*70)


# ==================== 配置 ====================
CONFIG = {
    'csv_files': ['20250701.csv'],  # 先测试1个文件
    'event_table': './results/event_table.csv',
    'output_full': './results/gps_with_events.csv',
    'chunk_size': 1000000,
}


# ==================== Step 1: 检查CSV文件结构 ====================
print("\n📋 Step 1: Checking CSV structure...")

test_file = CONFIG['csv_files'][0]

if not Path(test_file).exists():
    print(f"❌ File not found: {test_file}")
    exit(1)

# 读取前5行看看结构
print(f"\n   Reading first 5 rows of {test_file}...")
df_sample = pd.read_csv(test_file, nrows=5)

print(f"\n   Columns ({len(df_sample.columns)}):")
for i, col in enumerate(df_sample.columns, 1):
    print(f"      {i:2d}. {col}")

print(f"\n   Sample data (first 2 rows):")
print(df_sample.head(2).to_string())

# 检查必需列
required_cols = ['vehicle_id', 'datetime']
missing_cols = [col for col in required_cols if col not in df_sample.columns]

if missing_cols:
    print(f"\n❌ Missing required columns: {missing_cols}")
    print(f"   Available columns: {list(df_sample.columns)}")
    exit(1)

print(f"\n✅ All required columns present")


# ==================== Step 2: 加载事件表 ====================
print("\n📂 Step 2: Loading event table...")

df_events = pd.read_csv(CONFIG['event_table'])
df_events['start_time_int'] = df_events['start_time'].astype(np.int64)
df_events['end_time_int'] = df_events['end_time'].astype(np.int64)

print(f"✅ {len(df_events):,} events, {df_events['vehicle_id'].nunique():,} vehicles")

# 构建车辆索引
vehicle_events_dict = {}
for vehicle_id in df_events['vehicle_id'].unique():
    vehicle_events_dict[vehicle_id] = df_events[df_events['vehicle_id'] == vehicle_id]

print(f"✅ Indexed {len(vehicle_events_dict):,} vehicles")


# ==================== Step 3: 时间转换测试 ====================
print("\n🕐 Step 3: Testing time conversion...")

sample_datetimes = df_sample['datetime'].head(5)
print(f"\n   Sample datetime values:")
for i, dt in enumerate(sample_datetimes, 1):
    print(f"      {i}. {dt} (type: {type(dt)})")

def parse_datetime(dt_str):
    """解析datetime"""
    if pd.isna(dt_str):
        return None
    
    try:
        # 尝试多种格式
        for fmt in ['%Y/%m/%d %H:%M', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M']:
            try:
                dt = pd.to_datetime(dt_str, format=fmt)
                result = int(dt.strftime('%Y%m%d%H%M%S'))
                return result
            except:
                continue
        
        # 最后尝试自动解析
        dt = pd.to_datetime(dt_str, errors='coerce')
        if pd.notna(dt):
            return int(dt.strftime('%Y%m%d%H%M%S'))
    except Exception as e:
        print(f"      ⚠️  Parse error for '{dt_str}': {e}")
    
    return None

print(f"\n   Testing conversion:")
for i, dt in enumerate(sample_datetimes, 1):
    result = parse_datetime(dt)
    print(f"      {i}. '{dt}' → {result}")

if all(parse_datetime(dt) is None for dt in sample_datetimes):
    print(f"\n❌ Time conversion failed for all samples!")
    print(f"   Please check the datetime format.")
    exit(1)

print(f"\n✅ Time conversion working")


# ==================== Step 4: 匹配函数 ====================
def match_gps_simple(gps_chunk, vehicle_events_dict):
    """
    简单但稳定的匹配函数
    """
    print(f"\n   📊 Processing chunk: {len(gps_chunk):,} rows")
    
    # 初始化
    gps_chunk['event_id'] = -1
    gps_chunk['cluster'] = -1
    
    # 转换时间
    print(f"      🕐 Converting datetime...")
    gps_chunk['time_int'] = gps_chunk['datetime'].apply(parse_datetime)
    
    # 统计转换成功率
    valid_count = gps_chunk['time_int'].notna().sum()
    print(f"      ✅ Valid time: {valid_count:,}/{len(gps_chunk):,} ({valid_count/len(gps_chunk)*100:.1f}%)")
    
    if valid_count == 0:
        print(f"      ⚠️  No valid timestamps in this chunk, skipping...")
        return gps_chunk, 0
    
    # 移除无效时间
    gps_chunk = gps_chunk.dropna(subset=['time_int']).copy()
    gps_chunk['time_int'] = gps_chunk['time_int'].astype(np.int64)
    
    # 统计车辆
    unique_vehicles = gps_chunk['vehicle_id'].unique()
    print(f"      🚗 Unique vehicles: {len(unique_vehicles):,}")
    
    # 匹配
    print(f"      🔀 Matching events...")
    matched_count = 0
    
    for vehicle_id in tqdm(unique_vehicles, desc="      Vehicles", leave=False):
        if vehicle_id not in vehicle_events_dict:
            continue
        
        vehicle_mask = gps_chunk['vehicle_id'] == vehicle_id
        vehicle_gps = gps_chunk[vehicle_mask]
        vehicle_events = vehicle_events_dict[vehicle_id]
        
        for _, event in vehicle_events.iterrows():
            time_mask = (
                (vehicle_gps['time_int'] >= event['start_time_int']) &
                (vehicle_gps['time_int'] <= event['end_time_int'])
            )
            
            indices = vehicle_gps[time_mask].index
            
            if len(indices) > 0:
                gps_chunk.loc[indices, 'event_id'] = event['event_id']
                gps_chunk.loc[indices, 'cluster'] = event['cluster']
                matched_count += len(indices)
    
    match_rate = matched_count / len(gps_chunk) * 100 if len(gps_chunk) > 0 else 0
    print(f"      ✅ Matched: {matched_count:,}/{len(gps_chunk):,} ({match_rate:.1f}%)")
    
    return gps_chunk, matched_count


# ==================== Step 5: 处理测试文件 ====================
print("\n" + "="*70)
print("🔀 Step 5: Processing Test File")
print("="*70)

test_file = CONFIG['csv_files'][0]
print(f"\nProcessing: {test_file}")

total_points = 0
total_matched = 0
first_write = True

try:
    # 读取文件
    print(f"\n📖 Reading CSV in chunks...")
    
    reader = pd.read_csv(
        test_file,
        chunksize=CONFIG['chunk_size'],
        encoding='utf-8',
        on_bad_lines='skip',
        low_memory=False
    )
    
    chunk_count = 0
    
    for chunk_idx, chunk in enumerate(reader, 1):
        print(f"\n{'='*70}")
        print(f"Chunk {chunk_idx}")
        print(f"{'='*70}")
        
        # 清洗
        print(f"   Original size: {len(chunk):,}")
        chunk = chunk.dropna(subset=['vehicle_id'])
        print(f"   After dropna: {len(chunk):,}")
        
        if len(chunk) == 0:
            print(f"   ⚠️  Empty chunk, skipping...")
            continue
        
        # 匹配
        chunk, matched = match_gps_simple(chunk, vehicle_events_dict)
        
        # 统计
        total_matched += matched
        total_points += len(chunk)
        
        # 写入
        print(f"   💾 Writing to output...")
        chunk.to_csv(
            CONFIG['output_full'],
            mode='a' if not first_write else 'w',
            header=first_write,
            index=False,
            encoding='utf-8-sig'
        )
        
        first_write = False
        chunk_count += 1
        
        print(f"\n   ✅ Chunk {chunk_idx} complete:")
        print(f"      Points: {len(chunk):,}")
        print(f"      Matched: {matched:,}")
        print(f"      Cumulative: {total_points:,} points, {total_matched:,} matched")
        
        del chunk
        gc.collect()
        
        # 测试模式：只处理前3个chunk
        if chunk_count >= 3:
            print(f"\n   🛑 Test mode: stopping after 3 chunks")
            break
    
    print(f"\n{'='*70}")
    print("✅ Processing Complete")
    print(f"{'='*70}")
    
    if total_points > 0:
        match_rate = total_matched / total_points * 100
        print(f"\n📊 Final Statistics:")
        print(f"   Total points: {total_points:,}")
        print(f"   Matched: {total_matched:,} ({match_rate:.1f}%)")
        print(f"   Unmatched: {total_points - total_matched:,}")
        
        if os.path.exists(CONFIG['output_full']):
            size_mb = os.path.getsize(CONFIG['output_full']) / (1024**2)
            print(f"   Output file: {CONFIG['output_full']} ({size_mb:.1f} MB)")
    else:
        print(f"\n⚠️  No data was processed!")
        print(f"   This might indicate:")
        print(f"   1. All chunks were empty after dropna")
        print(f"   2. Time conversion failed")
        print(f"   3. File format issues")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print(f"\n{'='*70}")
print("🔍 Debug session complete")
print(f"{'='*70}")