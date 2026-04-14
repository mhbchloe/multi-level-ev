"""
Step 12: Build Inter-Charge Trip Dataset (纯 CPU 版本)
不依赖 cudf/cupy，用 pandas 处理
对 65GB 数据只读目标车辆 + 放电行，内存可控
"""

import pandas as pd
import numpy as np
import os
import glob
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("🔗 Step 12: Building Inter-Charge Trip Dataset")
print("=" * 70)

# ============================================================
# Config
# ============================================================
RAW_DATA_DIR = "./"
OUTPUT_DIR = "./coupling_analysis/results/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

USECOLS = ['vehicle_id', 'time', 'ch_s', 'soc', 'spd', 'power']

t_start = time.time()

# ============================================================
# 1. 加载充电事件
# ============================================================
print("\n📂 Loading charging events...")

charging_path = os.path.join(OUTPUT_DIR, 'charging_events_meaningful.csv')
if not os.path.exists(charging_path):
    for alt in ['charging_events_stationary_meaningful.csv',
                'charging_events_raw_extracted.csv']:
        alt_path = os.path.join(OUTPUT_DIR, alt)
        if os.path.exists(alt_path):
            charging_path = alt_path
            break

df_charging = pd.read_csv(charging_path)
df_charging['start_time'] = pd.to_datetime(df_charging['start_time'])
df_charging['end_time'] = pd.to_datetime(df_charging['end_time'])
df_charging = df_charging.sort_values(['vehicle_id', 'start_time']).reset_index(drop=True)

target_vehicles = set(df_charging['vehicle_id'].unique())

print(f"   ✅ Charging events: {len(df_charging):,}")
print(f"   ✅ Target vehicles: {len(target_vehicles):,}")

# ============================================================
# 2. 从原始 CSV 逐文件提取目标车辆的放电数据
#    只读需要的列 + 只保留目标车辆 → 内存可控
# ============================================================
print(f"\n📂 Loading driving data (CPU, filtered)...")

csv_files = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "data_2025*_processed.csv")))
print(f"   Found {len(csv_files)} files")

driving_chunks = []

for f in tqdm(csv_files, desc="Reading files", ncols=80):
    # 只读需要的列
    df = pd.read_csv(f, usecols=USECOLS)

    # 只保留目标车辆
    df = df[df['vehicle_id'].isin(target_vehicles)]

    # 只保留放电状态 (ch_s == 3)
    df = df[df['ch_s'] == 3]

    if len(df) > 0:
        # 解析时间
        df['datetime'] = pd.to_datetime(df['time'], errors='coerce')
        df = df.dropna(subset=['datetime'])
        driving_chunks.append(df[['vehicle_id', 'datetime', 'soc', 'spd', 'power']])

    del df

print("   Concatenating...")
df_driving = pd.concat(driving_chunks, ignore_index=True)
del driving_chunks

df_driving = df_driving.sort_values(['vehicle_id', 'datetime']).reset_index(drop=True)
print(f"   ✅ Driving rows: {len(df_driving):,}")
print(f"   ✅ Memory: {df_driving.memory_usage(deep=True).sum()/1024**3:.2f} GB")

# ============================================================
# 3. 构建 Inter-Charge Trips
# ============================================================
print(f"\n{'=' * 70}")
print("🚗 Building Inter-Charge Trips...")
print(f"{'=' * 70}")

# 预分组，避免循环内重复过滤
charging_groups = {vid: grp.reset_index(drop=True)
                   for vid, grp in df_charging.groupby('vehicle_id')}
driving_groups = {vid: grp for vid, grp in df_driving.groupby('vehicle_id')}

# 释放大表
del df_driving, df_charging

trips = []

for vid in tqdm(target_vehicles, desc="Processing vehicles", ncols=80):
    if vid not in charging_groups or vid not in driving_groups:
        continue

    v_charges = charging_groups[vid]
    v_driving = driving_groups[vid]

    if len(v_charges) == 0 or len(v_driving) == 0:
        continue

    # 用 numpy 数组加速时间比较
    driving_times = v_driving['datetime'].values  # datetime64
    driving_soc = v_driving['soc'].values
    driving_spd = v_driving['spd'].values
    driving_pwr = v_driving['power'].values

    for i in range(len(v_charges)):
        current_charge = v_charges.iloc[i]
        charge_start = current_charge['start_time']

        # 行程起点：上次充电结束 或 数据最早时间
        if i == 0:
            trip_start = driving_times[0]
        else:
            trip_start = v_charges.iloc[i - 1]['end_time']

        # 转为 numpy datetime64 方便比较
        ts_start = np.datetime64(trip_start)
        ts_end = np.datetime64(charge_start)

        # 布尔索引 (numpy 向量化，比 pandas 快 5-10x)
        mask = (driving_times > ts_start) & (driving_times < ts_end)
        n_records = mask.sum()

        if n_records < 3:
            continue

        trip_soc = driving_soc[mask]
        trip_spd = driving_spd[mask]
        trip_pwr = driving_pwr[mask]
        trip_times = driving_times[mask]

        # ===== 基础指标 =====
        duration_sec = (trip_times[-1] - trip_times[0]) / np.timedelta64(1, 's')
        if duration_sec <= 0:
            continue

        duration_hrs = duration_sec / 3600.0
        soc_start_trip = trip_soc[0]
        soc_end_trip = trip_soc[-1]
        soc_drop = soc_start_trip - soc_end_trip

        # ===== 速度特征 =====
        spd_valid = trip_spd[trip_spd > 0.5]
        if len(spd_valid) > 1:
            speed_mean = float(np.mean(spd_valid))
            speed_std = float(np.std(spd_valid))
            speed_max = float(np.max(spd_valid))
            speed_cv = speed_std / speed_mean if speed_mean > 0 else 0
        else:
            speed_mean = float(np.mean(trip_spd))
            speed_std = 0.0
            speed_max = float(np.max(trip_spd))
            speed_cv = 0.0

        idle_ratio = float(np.mean(trip_spd <= 0.5))

        # ===== 能耗特征 =====
        soc_rate = soc_drop / duration_hrs if duration_hrs > 0 else 0

        # ===== 末端特征 (最后 20%) =====
        n_end = max(1, n_records // 5)
        end_speed_mean = float(np.mean(trip_spd[-n_end:]))
        end_power_mean = float(np.mean(trip_pwr[-n_end:]))

        trips.append({
            'trip_id': f"{vid}_trip_{i:04d}",
            'vehicle_id': vid,
            'trip_start': str(trip_start),
            'trip_end': str(charge_start),

            # X1: 行程信息
            'num_records': int(n_records),
            'trip_duration_hrs': round(duration_hrs, 4),

            # X2: 驾驶行为
            'speed_mean': round(speed_mean, 2),
            'speed_std': round(speed_std, 2),
            'speed_max': round(speed_max, 2),
            'speed_cv': round(speed_cv, 4),
            'idle_ratio': round(idle_ratio, 4),

            # X3: 能耗
            'power_mean': round(float(np.mean(trip_pwr)), 2),
            'power_std': round(float(np.std(trip_pwr)), 2),
            'soc_start_trip': round(float(soc_start_trip), 2),
            'soc_end_trip': round(float(soc_end_trip), 2),
            'soc_drop': round(float(soc_drop), 2),
            'soc_rate_per_hr': round(soc_rate, 4),

            # X4: 末端特征
            'end_speed_mean': round(end_speed_mean, 2),
            'end_power_mean': round(end_power_mean, 2),

            # Y: 充电决策
            'charge_trigger_soc': current_charge['soc_start'],
            'charge_gain_soc': current_charge['soc_gain'],
            'charge_duration_min': current_charge.get('duration_minutes', np.nan),
            'charge_type': current_charge.get('charge_type', 'unknown'),
            'charge_soc_rate': current_charge.get('avg_soc_rate', np.nan),
        })

df_trips = pd.DataFrame(trips)

# ============================================================
# 4. 质量过滤
# ============================================================
print(f"\n🔧 Post-processing...")

n_raw = len(df_trips)

if n_raw > 0:
    df_trips = df_trips[
        (df_trips['trip_duration_hrs'] > 0.01) &
        (df_trips['trip_duration_hrs'] < 168) &
        (df_trips['soc_drop'] > 0) &
        (df_trips['soc_drop'] < 100) &
        (df_trips['speed_mean'] < 200) &
        (df_trips['num_records'] >= 3)
    ].reset_index(drop=True)

    print(f"   Raw: {n_raw:,} → Filtered: {len(df_trips):,}")

    output_path = os.path.join(OUTPUT_DIR, 'inter_charge_trips.csv')
    df_trips.to_csv(output_path, index=False)

    t_total = time.time() - t_start

    # ============================================================
    # 5. 报告
    # ============================================================
    print(f"\n{'=' * 70}")
    print(f"✅ Inter-Charge Trip Dataset Created!")
    print(f"{'=' * 70}")

    print(f"\n⏱️  Total time: {t_total:.1f}s ({t_total/60:.1f} min)")

    print(f"\n📊 Dataset:")
    print(f"   Trips:          {len(df_trips):,}")
    print(f"   Vehicles:       {df_trips['vehicle_id'].nunique():,}")
    print(f"   Trips/vehicle:  {len(df_trips)/df_trips['vehicle_id'].nunique():.1f}")

    print(f"\n🚗 Driving (X):")
    print(f"   Avg speed:      {df_trips['speed_mean'].mean():.1f} km/h")
    print(f"   Avg speed CV:   {df_trips['speed_cv'].mean():.3f}")
    print(f"   Avg idle ratio: {df_trips['idle_ratio'].mean():.2%}")
    print(f"   Avg duration:   {df_trips['trip_duration_hrs'].mean():.2f} hrs")
    print(f"   Avg SOC drop:   {df_trips['soc_drop'].mean():.1f}%")

    print(f"\n⚡ Charging (Y):")
    print(f"   Avg trigger SOC:  {df_trips['charge_trigger_soc'].mean():.1f}%")
    print(f"   Avg charge gain:  {df_trips['charge_gain_soc'].mean():.1f}%")
    print(f"   Avg charge time:  {df_trips['charge_duration_min'].mean():.1f} min")

    if 'charge_type' in df_trips.columns:
        print(f"\n   Charge types:")
        for ct, cnt in df_trips['charge_type'].value_counts().items():
            print(f"     {ct}: {cnt:,} ({cnt/len(df_trips)*100:.1f}%)")

    print(f"\n💾 Saved: {output_path}")
    print(f"   Size: {os.path.getsize(output_path)/1024/1024:.1f} MB")
else:
    print("\n❌ No trips extracted!")

print(f"\n✅ Done!")