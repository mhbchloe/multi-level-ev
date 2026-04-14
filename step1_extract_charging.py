"""
Step 1: 提取所有充电GPS点
从7个CSV文件中提取 is_charging==1 的GPS点
"""

import pandas as pd
from tqdm import tqdm
import gc
from pathlib import Path
import os

print("="*70)
print("🔋 Step 1: Extract Charging GPS Points")
print("="*70)

files = [
    '20250701.csv', '20250702.csv', '20250703.csv', 
    '20250704.csv', '20250705.csv', '20250706.csv', '20250707.csv'
]

output_file = './results/charging_gps_all.csv'

total_charging = 0
first_write = True

for file_idx, csv_file in enumerate(files, 1):
    if not Path(csv_file).exists():
        print(f"⚠️  {csv_file} not found, skipping...")
        continue
    
    print(f"\n[{file_idx}/7] {csv_file}")
    
    file_charging = 0
    
    try:
        reader = pd.read_csv(
            csv_file,
            chunksize=1000000,
            usecols=['vehicle_id', 'datetime', 'time', 'soc', 'lat', 'lon', 'is_charging', 'power'],
            encoding='utf-8',
            on_bad_lines='skip',
            low_memory=False
        )
        
        pbar = tqdm(reader, desc=f"  Processing", unit="chunk")
        
        for chunk in pbar:
            # 只保留充电中的GPS点
            charging = chunk[chunk['is_charging'] == 1]
            file_charging += len(charging)
            
            pbar.set_postfix({'charging': f'{file_charging:,}'})
            
            if len(charging) > 0:
                charging.to_csv(
                    output_file,
                    mode='a' if not first_write else 'w',
                    header=first_write,
                    index=False,
                    encoding='utf-8-sig'
                )
                first_write = False
            
            del chunk, charging
            gc.collect()
        
        print(f"  ✅ {file_charging:,} charging points")
        total_charging += file_charging
    
    except Exception as e:
        print(f"  ❌ Error: {e}")
        continue

print(f"\n{'='*70}")
print(f"✅ Extraction Complete!")
print(f"{'='*70}")
print(f"   Total charging GPS points: {total_charging:,}")
print(f"   Output: {output_file}")

if Path(output_file).exists():
    size_mb = os.path.getsize(output_file) / (1024**2)
    print(f"   File size: {size_mb:.1f} MB")
    
    # 估算压缩比
    total_points = 14000000 * 7  # 约9800万总GPS点
    compression = (1 - total_charging / total_points) * 100
    print(f"   Data reduction: {compression:.1f}%")

print(f"{'='*70}")
print(f"\n💡 Next step: Match charging points to events")
print(f"   python step2_match_charging_to_events.py")