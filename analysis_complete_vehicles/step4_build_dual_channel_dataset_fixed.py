"""
Step 4: Build Packed HDF5 Dataset (无 padding 版本)
按 SOC 下降 3% 划分的变长序列，直接拼接存储
"""

import numpy as np
import h5py
import pandas as pd
import os
from tqdm import tqdm

print("=" * 70)
print("🏗️ Building Packed HDF5 Dataset for GRU (No Padding)")
print("=" * 70)

# ============ 1. 配置路径 ============
input_file  = './analysis_complete_vehicles/results/discharge_segments_28days.pkl'
output_file = './analysis_complete_vehicles/results/dual_channel_dataset.h5'

print(f"\n📂 Input : {input_file}")
print(f"   Output: {output_file}")

# ============ 2. 加载 ============
print(f"\n📖 Loading...")
if not os.path.exists(input_file):
    print(f"❌ {input_file} not found!"); exit(1)

df = pd.read_pickle(input_file)
print(f"✅ Loaded {len(df):,} segments")

# ============ 3. 分类 + 过滤静止 ============
print(f"\n🏷️  Classifying...")

def classify_segment(row):
    if row['speed_mean'] > 1.0:
        return 0   # 行驶
    elif row['power_mean'] > 50:
        return 1   # 怠速
    else:
        return 2   # 静止 → 过滤

df['segment_type'] = df.apply(classify_segment, axis=1)
df = df[df['segment_type'] != 2].reset_index(drop=True)

n_driving = (df['segment_type'] == 0).sum()
n_idle    = (df['segment_type'] == 1).sum()
print(f"✅ 行驶={n_driving:,}  怠速={n_idle:,}  共={len(df):,}")

# ============ 4. 定义特征 ============
DRIVING_FEATURES = ['speed', 'acc', 'heading']
ENERGY_FEATURES  = ['soc', 'voltage', 'current', 'power']

seq_cols = {
    'speed':   'seq_speed',
    'acc':     'seq_acc',
    'heading': 'seq_heading',
    'soc':     'seq_soc',
    'voltage': 'seq_v',
    'current': 'seq_i',
    'power':   'seq_power',
}

def to_array(val):
    if isinstance(val, list):       return np.array(val, dtype=np.float32)
    elif isinstance(val, np.ndarray): return val.astype(np.float32)
    else:                           return np.array([float(val)], dtype=np.float32)

# ============ 5. 解析序列 ============
print(f"\n🔍 Parsing sequences...")

parsed_data  = []
total_steps  = 0

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Parsing"):
    d_seqs = [to_array(row[seq_cols[f]]) for f in DRIVING_FEATURES]
    e_seqs = [to_array(row[seq_cols[f]]) for f in ENERGY_FEATURES]

    # 以最短公共长度为准（防止各特征长度不一致）
    seq_len = min(len(s) for s in d_seqs + e_seqs)
    if seq_len < 2:
        continue

    parsed_data.append({
        'segment_id':   str(row['segment_id']),
        'segment_type': int(row['segment_type']),
        'd_seqs':       [s[:seq_len] for s in d_seqs],
        'e_seqs':       [s[:seq_len] for s in e_seqs],
        'seq_len':      seq_len,
    })
    total_steps += seq_len

print(f"✅ {len(parsed_data):,} segments, {total_steps:,} total timesteps")

# ============ 6. 分配拼接矩阵 ============
print(f"\n🔨 Building packed arrays...")

n_samples     = len(parsed_data)
n_drv_feats   = len(DRIVING_FEATURES)
n_eng_feats   = len(ENERGY_FEATURES)

# 核心存储：所有时间步直接拼接，无 padding
driving_packed = np.zeros((total_steps, n_drv_feats), dtype=np.float32)
energy_packed  = np.zeros((total_steps, n_eng_feats), dtype=np.float32)
lengths        = np.zeros(n_samples, dtype=np.int32)
offsets        = np.zeros(n_samples + 1, dtype=np.int64)  # offsets[i]~offsets[i+1]
segment_types  = np.zeros(n_samples, dtype=np.int8)
segment_ids    = []

ptr = 0
for i, data in enumerate(tqdm(parsed_data, desc="Packing")):
    L = data['seq_len']
    for fi, seq in enumerate(data['d_seqs']):
        driving_packed[ptr:ptr+L, fi] = seq
    for fi, seq in enumerate(data['e_seqs']):
        energy_packed[ptr:ptr+L, fi] = seq

    offsets[i]       = ptr
    lengths[i]       = L
    segment_types[i] = data['segment_type']
    segment_ids.append(data['segment_id'])
    ptr += L

offsets[n_samples] = ptr  # 末尾哨兵

print(f"\n✅ Packed shapes:")
print(f"   driving_packed : {driving_packed.shape}")
print(f"   energy_packed  : {energy_packed.shape}")
print(f"   offsets        : {offsets.shape}")
print(f"   lengths        : {lengths.shape}")

# 长度统计
print(f"\n📏 Sequence length stats:")
print(f"   min={lengths.min()}, max={lengths.max()}, "
      f"mean={lengths.mean():.1f}, median={np.median(lengths):.1f}")

# ============ 7. 归一化（全量有效数据，无 padding 污染） ============
print(f"\n📊 Normalizing (all data is valid, no padding)...")

def compute_stats(arr):
    """arr: (total_steps, F)，直接计算分位数"""
    arr_min = np.percentile(arr, 1,  axis=0).astype(np.float32)
    arr_max = np.percentile(arr, 99, axis=0).astype(np.float32)
    return arr_min, arr_max

def normalize(arr, arr_min, arr_max):
    mn   = arr_min[np.newaxis, :]
    mx   = arr_max[np.newaxis, :]
    norm = (arr - mn) / (mx - mn + 1e-8)
    return np.clip(norm, 0.0, 1.0).astype(np.float32)

driving_min, driving_max = compute_stats(driving_packed)
energy_min,  energy_max  = compute_stats(energy_packed)

driving_norm = normalize(driving_packed, driving_min, driving_max)
energy_norm  = normalize(energy_packed,  energy_min,  energy_max)

print(f"✅ Driving: [{driving_norm.min():.4f}, {driving_norm.max():.4f}]")
print(f"✅ Energy : [{energy_norm.min():.4f},  {energy_norm.max():.4f}]")

# ============ 8. 保存 HDF5 ============
print(f"\n💾 Saving to {output_file}...")

str_dtype = h5py.string_dtype(encoding='utf-8')

with h5py.File(output_file, 'w') as f:
    # 核心序列数据
    f.create_dataset('driving_packed',  data=driving_norm,
                     compression='gzip', compression_opts=4)
    f.create_dataset('energy_packed',   data=energy_norm,
                     compression='gzip', compression_opts=4)
    # 索引信息
    f.create_dataset('offsets',         data=offsets)
    f.create_dataset('lengths',         data=lengths)
    f.create_dataset('segment_types',   data=segment_types)
    f.create_dataset('segment_ids',     data=segment_ids, dtype=str_dtype)
    # 归一化参数（推理时反归一化用）
    f.create_dataset('driving_min',     data=driving_min)
    f.create_dataset('driving_max',     data=driving_max)
    f.create_dataset('energy_min',      data=energy_min)
    f.create_dataset('energy_max',      data=energy_max)

    # 元数据
    f.attrs['n_samples']             = n_samples
    f.attrs['total_timesteps']       = int(total_steps)
    f.attrs['n_driving_features']    = n_drv_feats
    f.attrs['n_energy_features']     = n_eng_feats
    f.attrs['driving_feature_names'] = str(DRIVING_FEATURES)
    f.attrs['energy_feature_names']  = str(ENERGY_FEATURES)
    f.attrs['segment_type_mapping']  = '0=driving, 1=idle'
    f.attrs['n_driving']             = int(n_driving)
    f.attrs['n_idle']                = int(n_idle)
    f.attrs['storage_format']        = 'packed (no padding)'

print(f"✅ Saved!")

# ============ 9. 验证 ============
print(f"\n✅ Verification:")
print(f"   NaN in driving : {np.isnan(driving_norm).sum()}")
print(f"   NaN in energy  : {np.isnan(energy_norm).sum()}")
print(f"   offsets[-1]    : {offsets[-1]} == total_steps {total_steps} "
      f"{'✅' if offsets[-1]==total_steps else '❌'}")

# 随机抽样验证索引正确性
test_i = np.random.randint(0, n_samples)
s, e   = offsets[test_i], offsets[test_i+1]
print(f"   Sample {test_i}: offset=[{s},{e}), len={e-s} == {lengths[test_i]} "
      f"{'✅' if e-s==lengths[test_i] else '❌'}")

# 文件大小
size_mb = os.path.getsize(output_file) / 1024 / 1024
print(f"   File size      : {size_mb:.1f} MB")

print(f"\n{'='*70}")
print(f"✅ Step 4 Complete! (Packed format, 0% padding)")
print(f"{'='*70}")
print(f"\n📝 Output : {output_file}")
print(f"   Format  : packed sequences via offsets[]")
print(f"   Usage   : seq = driving_packed[offsets[i]:offsets[i+1]]")
print(f"\n🚀 Next: step5_dual_channel_gru_model.py")
print(f"{'='*70}\n")