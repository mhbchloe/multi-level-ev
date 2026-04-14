"""
事件表与GPS数据融合（修复版）
支持多文件夹、流式处理，不需要合并大表
"""

import numpy as np
import pandas as pd
from pathlib import Path  # 添加这个导入！
from tqdm import tqdm
import gc
import os

print("="*70)
print("🔗 Merging Events with GPS (Multi-folder Support)")
print("="*70)


# ==================== Step 1: 加载事件表 ====================
print("\n📂 Loading event table...")

df_events = pd.read_csv('./results/event_table.csv')
print(f"✅ Events: {len(df_events):,}")

# 时间格式统一
df_events['start_time_int'] = df_events['start_time'].astype(int)
df_events['end_time_int'] = df_events['end_time'].astype(int)

print(f"   Time range: {df_events['start_time_int'].min()} - {df_events['end_time_int'].max()}")


# ==================== Step 2: 扫描文件夹 ====================
print("\n📁 Scanning data folders...")

# 识别日期文件夹（20250701, 20250702, ...）
base_dir = Path('.')
date_folders = sorted([f for f in base_dir.iterdir() 
                      if f.is_dir() and f.name.startswith('202507')])

print(f"✅ Found {len(date_folders)} date folders:")
for folder in date_folders:
    print(f"   - {folder.name}")

# 在每个文件夹里找CSV文件
all_csv_files = []
for folder in date_folders:
    csv_files = list(folder.glob('*.csv'))
    all_csv_files.extend(csv_files)
    print(f"      {folder.name}: {len(csv_files)} CSV files")

print(f"\n   Total CSV files: {len(all_csv_files)}")


# ==================== Step 3: 为GPS点匹配事件（优化版） ====================
def match_gps_to_events_optimized(gps_chunk, df_events):
    """
    优化的匹配算法：使用向量化操作
    """
    # 初始化列
    gps_chunk['event_id'] = -1
    gps_chunk['cluster'] = -1
    
    # 按车辆ID分组（关键优化：避免跨车匹配）
    for vehicle_id, vehicle_gps in gps_chunk.groupby('vehicle_id'):
        # 获取该车的所有事件
        vehicle_events = df_events[df_events['vehicle_id'] == vehicle_id]
        
        if len(vehicle_events) == 0:
            continue
        
        # 向��化匹配：为每个GPS点找到对应事件
        for _, event in vehicle_events.iterrows():
            # 找到时间在事件范围内的GPS点
            mask = (
                (vehicle_gps['time'] >= event['start_time_int']) &
                (vehicle_gps['time'] <= event['end_time_int'])
            )
            
            # 批量赋值
            gps_chunk.loc[vehicle_gps[mask].index, 'event_id'] = event['event_id']
            gps_chunk.loc[vehicle_gps[mask].index, 'cluster'] = event['cluster']
    
    return gps_chunk


# ==================== Step 4: 流式处理所有文件 ====================
def process_all_gps_files(csv_files, df_events, output_file='./results/gps_with_events.csv'):
    """
    流式处理所有GPS文件，逐个写入结果
    """
    print("\n" + "="*70)
    print("🔀 Processing GPS Files (Streaming Mode)")
    print("="*70)
    
    # 统计
    total_points = 0
    total_matched = 0
    first_write = True
    
    # 逐个处理文件
    for file_idx, csv_file in enumerate(tqdm(csv_files, desc="Processing files"), 1):
        print(f"\n📄 File {file_idx}/{len(csv_files)}: {csv_file.name}")
        
        try:
            # 分块读取（每次100万行）
            chunk_iter = pd.read_csv(
                csv_file,
                chunksize=1000000,
                encoding='utf-8',
                on_bad_lines='skip'  # 跳过坏行
            )
            
            for chunk_idx, chunk in enumerate(chunk_iter, 1):
                # 检查必需列
                required_cols = ['vehicle_id', 'time']
                if not all(col in chunk.columns for col in required_cols):
                    print(f"   ⚠️  Chunk {chunk_idx}: Missing required columns, skipping")
                    continue
                
                # 清洗数据
                chunk = chunk.dropna(subset=['vehicle_id', 'time'])
                
                # 匹配事件
                chunk = match_gps_to_events_optimized(chunk, df_events)
                
                # 统计
                matched = (chunk['event_id'] != -1).sum()
                total_matched += matched
                total_points += len(chunk)
                
                print(f"   Chunk {chunk_idx}: {len(chunk):,} points, {matched:,} matched ({matched/len(chunk)*100:.1f}%)")
                
                # 写入文件（追加模式）
                chunk.to_csv(
                    output_file,
                    mode='a' if not first_write else 'w',
                    header=first_write,
                    index=False,
                    encoding='utf-8-sig'
                )
                
                first_write = False
                
                # 释放内存
                del chunk
                gc.collect()
        
        except Exception as e:
            print(f"   ❌ Error processing {csv_file.name}: {e}")
            continue
    
    # 最终统计
    match_rate = total_matched / total_points * 100 if total_points > 0 else 0
    
    print("\n" + "="*70)
    print("✅ Processing Complete!")
    print("="*70)
    print(f"   Total GPS points: {total_points:,}")
    print(f"   Matched points: {total_matched:,} ({match_rate:.1f}%)")
    print(f"   Unmatched: {total_points - total_matched:,} ({100-match_rate:.1f}%)")
    print(f"   Output file: {output_file}")
    print(f"   File size: {os.path.getsize(output_file) / (1024**3):.2f} GB")
    print("="*70)
    
    return output_file


# ==================== Step 5: 快速验证 ====================
def quick_validate(output_file, sample_size=100000):
    """
    快速验证融合结果
    """
    print("\n" + "="*70)
    print("✅ Quick Validation")
    print("="*70)
    
    try:
        # 只读取前N行
        df_sample = pd.read_csv(output_file, nrows=sample_size)
        
        print(f"\n📊 Sample statistics (first {len(df_sample):,} points):")
        print(f"   Matched rate: {(df_sample['event_id'] != -1).mean()*100:.1f}%")
        
        print(f"\n   Distribution by cluster:")
        for cluster_id in range(4):
            count = (df_sample['cluster'] == cluster_id).sum()
            pct = count / len(df_sample) * 100
            print(f"      Cluster {cluster_id}: {count:,} ({pct:.1f}%)")
        
        print(f"\n   Vehicles in sample: {df_sample['vehicle_id'].nunique():,}")
        
        # GPS质量检查
        if 'lat' in df_sample.columns and 'lon' in df_sample.columns:
            print(f"\n   GPS coordinates:")
            print(f"      Lat: [{df_sample['lat'].min():.6f}, {df_sample['lat'].max():.6f}]")
            print(f"      Lon: [{df_sample['lon'].min():.6f}, {df_sample['lon'].max():.6f}]")
        
        print("\n✅ Validation passed!")
        
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")


# ==================== Step 6: 提取充电相关GPS点（可选） ====================
def extract_charging_gps_only(input_file, output_file='./results/gps_charging_only.csv'):
    """
    只提取充电相关的GPS点（大幅减小数据量）
    逻辑：SOC上升的GPS点 = 充电中
    """
    print("\n" + "="*70)
    print("🔋 Extracting Charging GPS Points Only")
    print("="*70)
    
    charging_points = 0
    first_write = True
    
    # 分块处理
    chunk_iter = pd.read_csv(input_file, chunksize=1000000)
    
    for chunk in tqdm(chunk_iter, desc="Extracting charging points"):
        # 只保留已匹配的点
        chunk = chunk[chunk['event_id'] != -1]
        
        if len(chunk) == 0:
            continue
        
        # 计算SOC变化
        chunk = chunk.sort_values(['vehicle_id', 'time'])
        chunk['soc_diff'] = chunk.groupby('vehicle_id')['soc'].diff()
        
        # 只保留SOC上升的点（充电中）
        charging_mask = chunk['soc_diff'] > 0
        chunk_charging = chunk[charging_mask]
        
        charging_points += len(chunk_charging)
        
        # 写入
        if len(chunk_charging) > 0:
            chunk_charging.to_csv(
                output_file,
                mode='a' if not first_write else 'w',
                header=first_write,
                index=False,
                encoding='utf-8-sig'
            )
            first_write = False
        
        del chunk, chunk_charging
        gc.collect()
    
    print(f"\n✅ Extracted {charging_points:,} charging GPS points")
    print(f"   Saved to: {output_file}")
    
    return output_file


# ==================== Main ====================
def main():
    # 处理所有GPS文件
    output_file = process_all_gps_files(all_csv_files, df_events)
    
    # 快速验证
    quick_validate(output_file)
    
    # 可选：只提取充电点（减小数据量）
    print("\n💡 Do you want to extract charging GPS points only? (Reduce data size)")
    print("   This will create a much smaller file with only charging-related GPS points.")
    
    # 自动提取充电点
    charging_file = extract_charging_gps_only(output_file)
    
    print("\n" + "="*70)
    print("✅ All Done!")
    print("="*70)
    print("\n📁 Generated files:")
    print(f"   1. {output_file} - 完整GPS+事件标签")
    print(f"   2. {charging_file} - 仅充电相关GPS点（推荐用于空间分析）")
    print("\n💡 Next step:")
    print("   python spatial_charging_analysis_full.py")
    print("="*70)


if __name__ == "__main__":
    main()