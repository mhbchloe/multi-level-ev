"""Step 3: Extract Discharge Segments (3% SOC Drop) — Fast + Derived Features"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import os, glob, warnings
warnings.filterwarnings('ignore')

os.makedirs('./analysis_complete_vehicles/results', exist_ok=True)

print("="*70)
print("🔋 Extract Discharge Segments (3% SOC Drop) — Fast + Derived")
print("="*70)

# ============ 1. 加载28天覆盖的车辆列表 ============
print("\n📂 Loading vehicle list (>= 28 days)...")
df_coverage = pd.read_csv('./analysis_complete_vehicles/results/vehicle_coverage_31days.csv')
complete_vehicle_ids = set(df_coverage[df_coverage['total_days'] >= 28]['vehicle_id'].tolist())
print(f"✅ Vehicles with >= 28 days data: {len(complete_vehicle_ids):,}")

# ============ 2. 找 processed 文件 ============
print("\n📂 Scanning for processed CSV files...")
processed_files = sorted(glob.glob('./data_20250*_processed.csv'))
if not processed_files:
    processed_files = sorted(glob.glob('./data_processed_one_month/data_*_processed.csv'))
if not processed_files:
    processed_files = sorted(glob.glob('./**/data_*_processed.csv', recursive=True))
print(f"Found {len(processed_files)} processed files")
if not processed_files:
    print("❌ No processed files found!")
    exit()

# ============ 3. 提取放电片段 ============
print(f"\n{'='*70}\n📊 Extracting Discharge Segments\n{'='*70}\n")

all_segments = []
segment_id = 0

for csv_file in tqdm(processed_files, desc="Processing files"):
    date_str = os.path.basename(csv_file).split('_')[1]

    try:
        reader = pd.read_csv(
            csv_file,
            chunksize=50_000,
            usecols=['vehicle_id', 'time', 'soc', 'v', 'i', 'power', 'spd', 'lat', 'lon', 'is_charging'],
            on_bad_lines='skip',
            dtype={'vehicle_id': 'str', 'soc': 'float', 'is_charging': 'int'}
        )

        vehicle_data = defaultdict(list)

        for chunk in reader:
            chunk = chunk.dropna(subset=['vehicle_id', 'time', 'soc'])
            chunk['time'] = pd.to_datetime(chunk['time'], errors='coerce')
            chunk = chunk.dropna(subset=['time'])
            chunk = chunk[chunk['vehicle_id'].isin(complete_vehicle_ids)]
            if len(chunk) == 0:
                continue
            for vid, g in chunk.groupby('vehicle_id'):
                vehicle_data[vid].append(g)

        for vehicle_id, groups in vehicle_data.items():
            df_vehicle = pd.concat(groups, ignore_index=True).sort_values('time').reset_index(drop=True)
            if len(df_vehicle) < 10:
                continue

            # 过滤充电段
            df_vehicle = df_vehicle[df_vehicle['is_charging'] == 0]
            if len(df_vehicle) < 10:
                continue

            # heading
            df_vehicle['lat_diff'] = df_vehicle['lat'].diff()
            df_vehicle['lon_diff'] = df_vehicle['lon'].diff()
            df_vehicle['heading'] = np.degrees(np.arctan2(df_vehicle['lon_diff'], df_vehicle['lat_diff'])) % 360
            df_vehicle['heading'] = df_vehicle['heading'].fillna(method='bfill').fillna(0)

            # 质量过滤
            df_vehicle = df_vehicle[
                (df_vehicle['soc'].between(0, 100)) &
                (df_vehicle['spd'].between(0, 250)) &
                (df_vehicle['v'].between(0, 500)) &
                (np.abs(df_vehicle['power']) < 400)
            ]
            if len(df_vehicle) < 10:
                continue

            # ======== 极速版识别放电片段（含加速度等派生特征） ========
            times = df_vehicle['time'].values                 # datetime64[ns]
            socs = df_vehicle['soc'].values
            spds = df_vehicle['spd'].values                   # km/h
            vs = df_vehicle['v'].values
            iss = df_vehicle['i'].values
            powers_raw = df_vehicle['power'].values
            headings = df_vehicle['heading'].values

            # 若 power 缺失，用 v*i 补算
            powers = np.where(np.isfinite(powers_raw), powers_raw, vs * iss)

            # 计算 dt（秒）和加速度 acc = Δspeed / dt
            # speed 单位 km/h，则加速度单位为 (km/h)/s
            dt_sec = np.diff(times).astype('timedelta64[ns]').astype(np.float64) / 1e9
            # 防零/异常
            dt_sec = np.where(dt_sec <= 0, np.nan, dt_sec)
            acc = np.zeros_like(spds, dtype=np.float64)
            if len(spds) > 1:
                delta_spd = spds[1:] - spds[:-1]
                acc[1:] = np.divide(delta_spd, dt_sec, out=np.zeros_like(delta_spd), where=np.isfinite(dt_sec))
                if len(acc) > 1:
                    acc[0] = acc[1]  # 首个点用第二个点填充
            acc = np.nan_to_num(acc)

            current_idx = []
            soc_start = None

            for i in range(len(socs)):
                if soc_start is None:
                    soc_start = socs[i]
                    current_idx = [i]
                else:
                    soc_drop = soc_start - socs[i]
                    if soc_drop >= 3.0:
                        current_idx.append(i)
                        if len(current_idx) >= 10:
                            idx_arr = current_idx
                            try:
                                t_start = pd.Timestamp(times[idx_arr[0]])
                                t_end = pd.Timestamp(times[idx_arr[-1]])
                                duration_sec = (t_end - t_start).total_seconds()

                                seg_spd = spds[idx_arr]
                                seg_soc = socs[idx_arr]
                                seg_v = vs[idx_arr]
                                seg_i = iss[idx_arr]
                                seg_power = powers[idx_arr]
                                seg_heading = headings[idx_arr]
                                seg_acc = acc[idx_arr]

                                all_segments.append({
                                    'segment_id': segment_id,
                                    'vehicle_id': str(vehicle_id),
                                    'date': date_str,
                                    'start_time': t_start,
                                    'end_time': t_end,
                                    'duration_seconds': duration_sec,
                                    'n_points': len(idx_arr),

                                    # 统计特征
                                    'soc_start': float(seg_soc[0]),
                                    'soc_end': float(seg_soc[-1]),
                                    'soc_drop': float(seg_soc[0] - seg_soc[-1]),
                                    'speed_mean': float(seg_spd.mean()),
                                    'speed_std': float(seg_spd.std()),
                                    'speed_max': float(seg_spd.max()),
                                    'speed_min': float(seg_spd.min()),
                                    'heading_mean': float(seg_heading.mean()),
                                    'heading_std': float(seg_heading.std()),
                                    'voltage_mean': float(seg_v.mean()),
                                    'current_mean': float(seg_i.mean()),
                                    'power_mean': float(seg_power.mean()),
                                    'power_std': float(seg_power.std()),
                                    'power_max': float(seg_power.max()),
                                    'power_min': float(seg_power.min()),
                                    # 新增加速度统计
                                    'acc_mean': float(seg_acc.mean()),
                                    'acc_std': float(seg_acc.std()),
                                    'acc_max': float(seg_acc.max()),
                                    'acc_min': float(seg_acc.min()),

                                    # 序列特征（供 GRU 使用）
                                    'seq_speed': seg_spd.tolist(),
                                    'seq_soc': seg_soc.tolist(),
                                    'seq_v': seg_v.tolist(),
                                    'seq_i': seg_i.tolist(),
                                    'seq_power': seg_power.tolist(),
                                    'seq_heading': seg_heading.tolist(),
                                    'seq_acc': seg_acc.tolist(),
                                })
                                segment_id += 1
                            except Exception:
                                pass
                        # 重新开始
                        soc_start = socs[i]
                        current_idx = [i]
                    else:
                        current_idx.append(i)

    except Exception as e:
        print(f"   ⚠️ Error in {os.path.basename(csv_file)}: {str(e)[:80]}")
        continue

# ======== 收尾与保存 ========
if len(all_segments) == 0:
    print("❌ No segments extracted!")
    exit()

df_segments = pd.DataFrame(all_segments)
print(f"\n✅ Extracted {len(df_segments):,} discharge segments")
print(f"   From {df_segments['vehicle_id'].nunique():,} vehicles")

# 保存
pkl_path = './analysis_complete_vehicles/results/discharge_segments_28days.pkl'
csv_path = './analysis_complete_vehicles/results/discharge_segments_28days.csv'

df_segments.to_pickle(pkl_path)  # 含完整序列
cols_to_drop = [c for c in df_segments.columns if c.startswith('seq_')]
df_segments.drop(columns=cols_to_drop).to_csv(csv_path, index=False)

print(f"\n💾 Saved full data with sequences: {pkl_path}")
print(f"💾 Saved lightweight CSV (no sequences): {csv_path}")

# 简要统计
print(f"\nTotal segments: {len(df_segments):,}")
print(f"Unique vehicles: {df_segments['vehicle_id'].nunique():,}")
print(f"Avg segments per vehicle: {len(df_segments)/df_segments['vehicle_id'].nunique():.1f}")
print(f"Data Points Per Segment: mean={df_segments['n_points'].mean():.1f}, "
      f"median={df_segments['n_points'].median():.1f}, "
      f"range={df_segments['n_points'].min():.0f}-{df_segments['n_points'].max():.0f}")
print(f"\n✅ Step 3 Complete!\n")