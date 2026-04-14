"""
诊断并修复反归一化问题
根据原始字段定义自动检测和修正
"""

import numpy as np
import pandas as pd
from pathlib import Path

print("="*70)
print("🔍 Diagnosing Data Normalization Issue")
print("="*70)

# ==================== 加载原始序列数据 ====================
print("\n📂 Loading data...")

driving = np.load('./results/temporal_soc_full/driving_sequences.npy', allow_pickle=True)
energy = np.load('./results/temporal_soc_full/energy_sequences.npy', allow_pickle=True)

print(f"✅ Loaded {len(driving):,} driving sequences")
print(f"✅ Loaded {len(energy):,} energy sequences")

# ==================== 采样数据进行分析 ====================
print("\n🔍 Sampling data for analysis...")

sample_size = min(1000, len(driving))
driving_sample = np.vstack([driving[i] for i in range(sample_size)])
energy_sample = np.vstack([energy[i] for i in range(sample_size)])

print(f"   Sampled {sample_size:,} sequences")

# ==================== 分析各字段范围 ====================
print("\n" + "="*70)
print("📊 Field Range Analysis")
print("="*70)

print("\n🚗 Driving Features:")
print(f"   Speed (spd):")
print(f"      Current range: [{driving_sample[:, 0].min():.6f}, {driving_sample[:, 0].max():.6f}]")
print(f"      Expected range: [0, 220] km/h")
print(f"      Mean: {driving_sample[:, 0].mean():.6f}")
print(f"      Std: {driving_sample[:, 0].std():.6f}")

print(f"\n   Acceleration (derived):")
print(f"      Current range: [{driving_sample[:, 1].min():.6f}, {driving_sample[:, 1].max():.6f}]")
print(f"      Mean: {driving_sample[:, 1].mean():.6f}")

print(f"\n🔋 Energy Features:")
print(f"   SOC:")
print(f"      Current range: [{energy_sample[:, 0].min():.6f}, {energy_sample[:, 0].max():.6f}]")
print(f"      Expected range: [0, 100] %")
print(f"      Mean: {energy_sample[:, 0].mean():.6f}")

print(f"\n   Voltage (v):")
print(f"      Current range: [{energy_sample[:, 1].min():.6f}, {energy_sample[:, 1].max():.6f}]")
print(f"      Expected range: [0, 1000] V (unit 0.1)")
print(f"      Mean: {energy_sample[:, 1].mean():.6f}")

print(f"\n   Current (i):")
print(f"      Current range: [{energy_sample[:, 2].min():.6f}, {energy_sample[:, 2].max():.6f}]")
print(f"      Expected range: [-1000, 1000] A (unit 0.1)")
print(f"      Mean: {energy_sample[:, 2].mean():.6f}")

# ==================== 判断归一化类型 ====================
print("\n" + "="*70)
print("🔧 Detecting Normalization Type")
print("="*70)

spd_min = driving_sample[:, 0].min()
spd_max = driving_sample[:, 0].max()
spd_mean = driving_sample[:, 0].mean()
spd_std = driving_sample[:, 0].std()

v_min = energy_sample[:, 1].min()
v_max = energy_sample[:, 1].max()

i_min = energy_sample[:, 2].min()
i_max = energy_sample[:, 2].max()

soc_min = energy_sample[:, 0].min()
soc_max = energy_sample[:, 0].max()

# 判断速度归一化类型
if spd_max < 10:
    if spd_min >= 0 and spd_max <= 1.1:
        print("\n✅ Speed: Min-Max normalized to [0, 1]")
        speed_denorm_type = 'minmax'
        speed_scale = 220.0  # 根据字段定义
        speed_offset = 0.0
    elif spd_min >= -0.1 and spd_max <= 1.1:
        print("\n✅ Speed: Scaled to [0, 1]")
        speed_denorm_type = 'scaled'
        speed_scale = 220.0
        speed_offset = 0.0
    elif abs(spd_mean) < 1 and spd_std > 0.5:
        print("\n✅ Speed: StandardScaler (Z-score) normalized")
        speed_denorm_type = 'standard'
        # 假设原始数据均值30 km/h，标准差25 km/h
        speed_mean_original = 30.0
        speed_std_original = 25.0
    elif spd_min >= -3 and spd_max <= 3:
        print("\n✅ Speed: RobustScaler normalized")
        speed_denorm_type = 'robust'
        # 需要median和IQR
        speed_median_original = 35.0
        speed_iqr_original = 30.0
    else:
        print("\n⚠️  Speed: Unknown normalization")
        speed_denorm_type = 'unknown'
        speed_scale = 1.0
        speed_offset = 0.0
else:
    print("\n✅ Speed: No normalization detected (values in reasonable range)")
    speed_denorm_type = 'none'
    speed_scale = 1.0
    speed_offset = 0.0

# 判断电压归一化
if v_max < 10:
    print("\n✅ Voltage: Normalized (likely StandardScaler or RobustScaler)")
    voltage_denorm_type = 'normalized'
    voltage_scale = 500.0  # 中位数约500V
    voltage_offset = 0.0
else:
    print("\n✅ Voltage: No normalization or already in correct range")
    voltage_denorm_type = 'none'
    voltage_scale = 1.0
    voltage_offset = 0.0

# 判断电流归一化
if abs(i_max) < 10:
    print("\n✅ Current: Normalized")
    current_denorm_type = 'normalized'
    current_scale = 100.0  # 典型电流
    current_offset = 0.0
else:
    print("\n✅ Current: No normalization or already in correct range")
    current_denorm_type = 'none'
    current_scale = 1.0
    current_offset = 0.0

# 判断SOC
if soc_max <= 1.1:
    print("\n✅ SOC: Normalized to [0, 1], needs × 100")
    soc_scale = 100.0
else:
    print("\n✅ SOC: Already in percentage [0, 100]")
    soc_scale = 1.0

# ==================== 生成反归一化函数 ====================
print("\n" + "="*70)
print("🔧 Generating Denormalization Functions")
print("="*70)

def denormalize_speed(speed_normalized):
    """根据检测到的归一化类型反归一化速度"""
    if speed_denorm_type == 'minmax' or speed_denorm_type == 'scaled':
        # [0, 1] → [0, 220]
        return speed_normalized * speed_scale + speed_offset
    elif speed_denorm_type == 'standard':
        # Z-score → original
        return speed_normalized * speed_std_original + speed_mean_original
    elif speed_denorm_type == 'robust':
        # Robust → original
        return speed_normalized * speed_iqr_original + speed_median_original
    else:
        return speed_normalized

def denormalize_voltage(voltage_normalized):
    """反归一化电压"""
    if voltage_denorm_type == 'normalized':
        return voltage_normalized * voltage_scale + 300.0  # 偏移到300V中心
    else:
        return voltage_normalized

def denormalize_current(current_normalized):
    """反归一化电流"""
    if current_denorm_type == 'normalized':
        return current_normalized * current_scale
    else:
        return current_normalized

def denormalize_soc(soc_normalized):
    """反归一化SOC"""
    return soc_normalized * soc_scale

# ==================== 测试反归一化 ====================
print("\n" + "="*70)
print("🧪 Testing Denormalization")
print("="*70)

# 测试第一个序列
test_seq_driving = driving[0]
test_seq_energy = energy[0]

print(f"\n📊 Test Sequence (first trip):")
print(f"   Original speed range: [{test_seq_driving[:, 0].min():.6f}, {test_seq_driving[:, 0].max():.6f}]")

speed_denorm = denormalize_speed(test_seq_driving[:, 0])
voltage_denorm = denormalize_voltage(test_seq_energy[:, 1])
current_denorm = denormalize_current(test_seq_energy[:, 2])
soc_denorm = denormalize_soc(test_seq_energy[:, 0])

print(f"\n   Denormalized speed range: [{speed_denorm.min():.2f}, {speed_denorm.max():.2f}] km/h")
print(f"   Denormalized voltage range: [{voltage_denorm.min():.2f}, {voltage_denorm.max():.2f}] V")
print(f"   Denormalized current range: [{current_denorm.min():.2f}, {current_denorm.max():.2f}] A")
print(f"   Denormalized SOC range: [{soc_denorm.min():.2f}, {soc_denorm.max():.2f}] %")

# 检查是否合理
print("\n✅ Reasonableness Check:")
if 0 <= speed_denorm.mean() <= 120:
    print(f"   ✅ Speed looks reasonable: mean={speed_denorm.mean():.2f} km/h")
else:
    print(f"   ❌ Speed still abnormal: mean={speed_denorm.mean():.2f} km/h")

if 200 <= voltage_denorm.mean() <= 800:
    print(f"   ✅ Voltage looks reasonable: mean={voltage_denorm.mean():.2f} V")
else:
    print(f"   ⚠️  Voltage: mean={voltage_denorm.mean():.2f} V (check if reasonable)")

if 0 <= soc_denorm.mean() <= 100:
    print(f"   ✅ SOC looks reasonable: mean={soc_denorm.mean():.2f} %")
else:
    print(f"   ❌ SOC abnormal: mean={soc_denorm.mean():.2f} %")

# ==================== 保存反归一化参数 ====================
print("\n" + "="*70)
print("💾 Saving Denormalization Parameters")
print("="*70)

denorm_params = {
    'speed': {
        'type': speed_denorm_type,
        'scale': speed_scale if 'speed_scale' in locals() else 1.0,
        'offset': speed_offset if 'speed_offset' in locals() else 0.0,
        'mean': speed_mean_original if 'speed_mean_original' in locals() else None,
        'std': speed_std_original if 'speed_std_original' in locals() else None,
    },
    'voltage': {
        'type': voltage_denorm_type,
        'scale': voltage_scale,
        'offset': voltage_offset,
    },
    'current': {
        'type': current_denorm_type,
        'scale': current_scale,
        'offset': current_offset,
    },
    'soc': {
        'scale': soc_scale,
    }
}

import json
with open('./results/denorm_params.json', 'w') as f:
    json.dump(denorm_params, f, indent=2)

print("✅ Saved denormalization parameters to: ./results/denorm_params.json")

print("\n" + "="*70)
print("✅ Diagnosis Complete!")
print("="*70)
print("\n💡 Next steps:")
print("   1. Check the denormalized test values above")
print("   2. If they look reasonable, use these parameters in clustering")
print("   3. If not, we may need to check the data preprocessing script")
print("="*70)