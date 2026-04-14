"""
======================================================================
🔥 全量数据准备：仅用SOC 3%分割，不限制序列长度
======================================================================
- 使用所有车辆（不采样）
- 只用SOC累计下降≥3%分割
- 不截断超长序列
- 保存序列数据供编码器对比使用
======================================================================
"""

import numpy as np
import pandas as pd
import glob
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings("ignore")

print("="*70)
print("🔥 准备全量数据 - 仅SOC 3%分割")
print("="*70)

# ==================== 配置 ====================
CONFIG = {
    'soc_threshold': 3.0,           # SOC下降3%
    'min_seq_length': 5,            # 最小5个点（避免噪声）
    'max_seq_length': None,         # ← 不限制最大长度！
    'driving_features': ['spd', 'acc'],
    'energy_features': ['soc', 'v', 'i'],
    'output_dir': './results/temporal_soc_full'
}

Path(CONFIG['output_dir']).mkdir(parents=True, exist_ok=True)

print("\n⚙️  配置:")
print(f"   SOC阈值: {CONFIG['soc_threshold']}%")
print(f"   最小长度: {CONFIG['min_seq_length']}")
print(f"   最大长度: 无限制")
print(f"   使用全量数据（不采样）")

# ==================== SOC分割函数（无长度上限）====================
def segment_by_soc_cumulative(vehicle_data, soc_threshold):
    """
    累计SOC下降分割，不限制最大长度
    """
    trips = []
    if len(vehicle_data) < 2:
        return trips
    
    soc = vehicle_data['soc'].to_numpy(dtype=float)
    delta = np.diff(soc)
    drop = np.maximum(-delta, 0.0)  # 只累计下降
    
    start_idx = 0
    acc_drop = 0.0
    
    for i in range(1, len(vehicle_data)):
        acc_drop += drop[i - 1]
        
        # 累计下降达到阈值，切段
        if acc_drop >= soc_threshold:
            trip = vehicle_data.iloc[start_idx : i + 1].copy()
            
            # 只检查最小长度
            if len(trip) >= CONFIG['min_seq_length']:
                trips.append(trip)
            
            start_idx = i
            acc_drop = 0.0
    
    # 尾段
    tail = vehicle_data.iloc[start_idx:].copy()
    if len(tail) >= CONFIG['min_seq_length']:
        trips.append(tail)
    
    return trips


# ==================== 数据加载 ====================
print("\n" + "="*70)
print("📂 加载全量数据")
print("="*70)

data_files = sorted(glob.glob('./*_processed.csv'))
if not data_files:
    data_files = sorted(glob.glob('./data/*_processed.csv'))

print(f"✅ 找到 {len(data_files)} 个文件\n")

all_trips = []

# 列名映射（适配你的数据）
col_map = {
    'vehicle_id': 'vid',
    'datetime': 'datetime',
    'soc': 'soc',
    'spd': 'spd',
    'acc': 'acc',
    'v': 'v',
    'i': 'i'
}

for file_idx, file in enumerate(data_files, 1):
    print(f"📄 文件 {file_idx}/{len(data_files)}: {Path(file).name}")
    
    # 分块读取（避免内存溢出）
    chunk_size = 100000
    file_trip_count = 0
    
    try:
        for chunk_idx, chunk in enumerate(pd.read_csv(file, chunksize=chunk_size)):
            # 重命名列
            chunk = chunk.rename(columns=col_map)
            
            # 按车辆分组
            for vid, vehicle_data in chunk.groupby('vid'):
                # 排序
                vehicle_data = vehicle_data.sort_values('datetime').reset_index(drop=True)
                
                # 转换datetime
                try:
                    vehicle_data['datetime'] = pd.to_datetime(vehicle_data['datetime'])
                except:
                    pass
                
                # SOC分割
                trips = segment_by_soc_cumulative(vehicle_data, CONFIG['soc_threshold'])
                
                all_trips.extend(trips)
                file_trip_count += len(trips)
        
        print(f"   本文件行程数: {file_trip_count:,}")
        print(f"   累计总行程数: {len(all_trips):,}\n")
        
    except Exception as e:
        print(f"   ⚠️  文件处理出错: {e}\n")
        continue

print(f"✅ 共收集 {len(all_trips):,} 个行程")
print(f"   来自全量数据（未采样）")

# 统计长度分布
lengths = [len(trip) for trip in all_trips]
print(f"\n📊 序列长度统计:")
print(f"   最小: {min(lengths)}")
print(f"   最大: {max(lengths)}")
print(f"   平均: {np.mean(lengths):.1f}")
print(f"   中位数: {np.median(lengths):.0f}")
print(f"   P95: {np.percentile(lengths, 95):.0f}")
print(f"   P99: {np.percentile(lengths, 99):.0f}")

# 超长序列警告
ultra_long = sum(1 for l in lengths if l > 1000)
if ultra_long > 0:
    print(f"   ⚠️  超过1000点的行程: {ultra_long} ({ultra_long/len(lengths)*100:.1f}%)")


# ==================== 提取序列特征（变长）====================
print("\n" + "="*70)
print("🔧 提取变长序列")
print("="*70)

driving_sequences = []
energy_sequences = []
seq_lengths = []
trip_ids = []

print("处理行程...")
for idx, trip in enumerate(tqdm(all_trips)):
    try:
        # 提取特征
        driving_seq = trip[CONFIG['driving_features']].values
        energy_seq = trip[CONFIG['energy_features']].values
        
        # 检查有效性
        if np.isnan(driving_seq).any() or np.isnan(energy_seq).any():
            continue
        if np.isinf(driving_seq).any() or np.isinf(energy_seq).any():
            continue
        
        driving_sequences.append(driving_seq.astype(np.float32))
        energy_sequences.append(energy_seq.astype(np.float32))
        seq_lengths.append(len(driving_seq))
        trip_ids.append(f"trip_{idx}")
        
    except Exception as e:
        continue

print(f"\n✅ 有效序列: {len(driving_sequences):,}")
print(f"   平均长度: {np.mean(seq_lengths):.1f}")

# ==================== 归一化（逐特征）====================
print("\n📊 归一化特征...")

# 合并所有序列用于拟合scaler
all_driving = np.vstack(driving_sequences)
all_energy = np.vstack(energy_sequences)

# 驾驶特征归一化
driving_scalers = []
for i in range(all_driving.shape[1]):
    scaler = RobustScaler()
    scaler.fit(all_driving[:, i:i+1])
    driving_scalers.append(scaler)

# 能量特征归一化
energy_scalers = []
for i in range(all_energy.shape[1]):
    scaler = RobustScaler()
    scaler.fit(all_energy[:, i:i+1])
    energy_scalers.append(scaler)

# 应用归一化
print("应用归一化...")
for idx in tqdm(range(len(driving_sequences))):
    for i, scaler in enumerate(driving_scalers):
        driving_sequences[idx][:, i] = scaler.transform(
            driving_sequences[idx][:, i:i+1]
        ).flatten()
    
    for i, scaler in enumerate(energy_scalers):
        energy_sequences[idx][:, i] = scaler.transform(
            energy_sequences[idx][:, i:i+1]
        ).flatten()
    
    # 裁剪极值
    driving_sequences[idx] = np.clip(driving_sequences[idx], -5, 5)
    energy_sequences[idx] = np.clip(energy_sequences[idx], -5, 5)

print("✅ 归一化完成")

# ==================== 保存数据 ====================
print("\n💾 保存数据...")

# 变长序列保存为list格式
np.save(f"{CONFIG['output_dir']}/driving_sequences.npy", 
        np.array(driving_sequences, dtype=object), allow_pickle=True)
np.save(f"{CONFIG['output_dir']}/energy_sequences.npy", 
        np.array(energy_sequences, dtype=object), allow_pickle=True)
np.save(f"{CONFIG['output_dir']}/seq_lengths.npy", 
        np.array(seq_lengths, dtype=np.int32))

# 保存元信息
metadata = {
    'n_samples': len(driving_sequences),
    'driving_features': CONFIG['driving_features'],
    'energy_features': CONFIG['energy_features'],
    'soc_threshold': CONFIG['soc_threshold'],
    'min_seq_length': CONFIG['min_seq_length'],
    'max_seq_length': None,
    'length_stats': {
        'min': int(min(seq_lengths)),
        'max': int(max(seq_lengths)),
        'mean': float(np.mean(seq_lengths)),
        'median': float(np.median(seq_lengths)),
        'p95': float(np.percentile(seq_lengths, 95)),
        'p99': float(np.percentile(seq_lengths, 99))
    }
}

import json
with open(f"{CONFIG['output_dir']}/metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✅ 数据已保存至: {CONFIG['output_dir']}")
print("\n" + "="*70)
print("✅ 全量数据准备完成！")
print("="*70)
print(f"\n📊 最终统计:")
print(f"   样本数: {len(driving_sequences):,}")
print(f"   平均长度: {np.mean(seq_lengths):.1f}")
print(f"   最长序列: {max(seq_lengths)} 点")