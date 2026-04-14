"""
充电事件提取 - RTX 4090 GPU 全速版 (兼容性修复版)
修复: cudf.to_datetime 不支持 errors='coerce'
修复: cudf groupby agg 列名兼容
"""

import cudf
import cupy as cp
import pandas as pd
import numpy as np
import os
import glob
import time
import warnings
warnings.filterwarnings('ignore')

# GPU 信息
print("=" * 70)
print("⚡ Charging Event Extraction - RTX 4090 GPU Accelerated")
print("=" * 70)

gpu = cp.cuda.Device(0) if hasattr(cp, 'cuda') else None
mem_info = cp.cuda.Device(0).mem_info
print(f"   GPU: NVIDIA GeForce RTX 4090")
print(f"   VRAM: {mem_info[1]/1024**3:.1f} GB total, {mem_info[0]/1024**3:.1f} GB free")

# ============================================================
# Config
# ============================================================
RAW_DATA_DIR = "./"
OUTPUT_DIR = "./coupling_analysis/results/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IS_CHARGING_CODES = [1, 2]
MIN_SOC_GAIN = 0.5
MIN_DURATION_SEC = 60
MIN_RECORDS = 3
TIME_GAP_THRESHOLD = 600  # 10分钟，判定为两次独立充电
BATCH_SIZE = 5

USECOLS = ['vehicle_id', 'time', 'ch_s', 'soc', 'v', 'i', 'power', 'spd']

csv_files = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "data_2025*_processed.csv")))
print(f"\n📂 Found {len(csv_files)} files (~{len(csv_files)*2:.0f} GB total)")

t_start = time.time()
all_charging_chunks = []

print(f"\n🚀 Processing in batches of {BATCH_SIZE} files...")

for batch_idx in range(0, len(csv_files), BATCH_SIZE):
    batch_files = csv_files[batch_idx : batch_idx + BATCH_SIZE]
    batch_num = batch_idx // BATCH_SIZE + 1
    total_batches = (len(csv_files) - 1) // BATCH_SIZE + 1
    batch_label = f"Batch {batch_num}/{total_batches}"

    print(f"\n   📦 {batch_label}: loading {len(batch_files)} files to GPU...")
    t_batch = time.time()

    # ---- 加载到 GPU ----
    gpu_chunks = []
    for f in batch_files:
        gdf = cudf.read_csv(f, usecols=USECOLS)
        gpu_chunks.append(gdf)

    gdf_batch = cudf.concat(gpu_chunks, ignore_index=True)
    del gpu_chunks
    cp.get_default_memory_pool().free_all_blocks()

    n_rows = len(gdf_batch)
    print(f"      Loaded: {n_rows:,} rows ({time.time()-t_batch:.1f}s)")

    # ---- 时间解析 (cudf 兼容方式) ----
    # cudf.to_datetime 不支持 errors='coerce'
    # 方案: 先 astype 转换，无效值会变成 NaT
    try:
        gdf_batch['datetime'] = cudf.to_datetime(gdf_batch['time'])
    except Exception:
        # 如果格式不统一，用 str 列直接 astype
        try:
            gdf_batch['datetime'] = gdf_batch['time'].astype('datetime64[ns]')
        except Exception:
            # 最后兜底：落回 CPU 处理这一列，再传回 GPU
            print("      ⚠️ Falling back to CPU for datetime parsing...")
            dt_series = pd.to_datetime(gdf_batch['time'].to_pandas(), errors='coerce')
            gdf_batch['datetime'] = cudf.Series(dt_series)

    # 删除时间解析失败的行
    null_mask = gdf_batch['datetime'].isna()
    n_null = int(null_mask.sum())
    if n_null > 0:
        gdf_batch = gdf_batch[~null_mask]
        print(f"      Dropped {n_null:,} rows with invalid datetime")

    # 删除 ch_s 或 soc 为空的行
    gdf_batch = gdf_batch.dropna(subset=['ch_s', 'soc'])
    gdf_batch['ch_s'] = gdf_batch['ch_s'].astype('int32')

    # ---- 排序 (GPU 并行排序) ----
    gdf_batch = gdf_batch.sort_values(['vehicle_id', 'datetime']).reset_index(drop=True)

    # ---- 标记充电状态 ----
    gdf_batch['is_ch'] = gdf_batch['ch_s'].isin(IS_CHARGING_CODES).astype('int8')

    # 检测充电段边界
    gdf_batch['vid_prev'] = gdf_batch['vehicle_id'].shift(1)
    gdf_batch['is_ch_prev'] = gdf_batch['is_ch'].shift(1, fill_value=0)

    vid_change = (gdf_batch['vehicle_id'] != gdf_batch['vid_prev'])
    ch_start = (gdf_batch['is_ch'] == 1) & ((gdf_batch['is_ch_prev'] == 0) | vid_change)
    gdf_batch['event_group'] = ch_start.astype('int32').cumsum()

    # 只保留充电中的行
    gdf_ch = gdf_batch[gdf_batch['is_ch'] == 1].copy()

    n_ch = len(gdf_ch)
    print(f"      Charging rows: {n_ch:,} / {n_rows:,} ({n_ch/max(n_rows,1)*100:.1f}%)")

    if n_ch > 0:
        keep_cols = ['vehicle_id', 'datetime', 'ch_s', 'soc', 'v', 'i', 'power', 'spd', 'event_group']
        all_charging_chunks.append(gdf_ch[keep_cols])

    del gdf_batch, gdf_ch
    cp.get_default_memory_pool().free_all_blocks()
    print(f"      Done ({time.time()-t_batch:.1f}s)")

# ============================================================
# 2. 合并所有充电行，重新检测跨天事件边界
# ============================================================
print(f"\n📊 Merging all charging records...")
t_merge = time.time()

if len(all_charging_chunks) == 0:
    print("   ⚠️ No charging records found in any file!")
    exit()

gdf_all_ch = cudf.concat(all_charging_chunks, ignore_index=True)
del all_charging_chunks
cp.get_default_memory_pool().free_all_blocks()

print(f"   Total charging rows: {len(gdf_all_ch):,}")

# 全局排序
print("   Re-sorting globally...")
gdf_all_ch = gdf_all_ch.sort_values(['vehicle_id', 'datetime']).reset_index(drop=True)

# 重新划分事件边界 (换车 或 时间间隔 > 10分钟)
gdf_all_ch['vid_prev'] = gdf_all_ch['vehicle_id'].shift(1)
gdf_all_ch['dt_prev'] = gdf_all_ch['datetime'].shift(1)

vid_change = (gdf_all_ch['vehicle_id'] != gdf_all_ch['vid_prev'])

# cudf 时间差 → 秒数
dt_diff = (gdf_all_ch['datetime'] - gdf_all_ch['dt_prev'])
# cudf timedelta → 纳秒 → 秒
dt_diff_sec = dt_diff.astype('int64') / 1_000_000_000

gdf_all_ch['new_event'] = (vid_change | (dt_diff_sec > TIME_GAP_THRESHOLD)).astype('int32')
gdf_all_ch['event_group'] = gdf_all_ch['new_event'].cumsum()

n_events_raw = int(gdf_all_ch['event_group'].max()) + 1
print(f"   Raw events detected: {n_events_raw:,} ({time.time()-t_merge:.1f}s)")

# ============================================================
# 3. GPU 聚合
# ============================================================
print(f"\n⚡ Aggregating events on GPU...")
t_agg = time.time()

# cudf groupby agg 用字典格式
agg_dict = {
    'vehicle_id': 'first',
    'soc':        ['first', 'last', 'min', 'max'],
    'ch_s':       'max',
    'v':          'mean',
    'i':          'mean',
    'power':      'mean',
    'spd':        'mean',
    'datetime':   ['first', 'last', 'count'],
}

events = gdf_all_ch.groupby('event_group').agg(agg_dict)

# 展平多级列名
events.columns = [
    'vehicle_id',
    'soc_start', 'soc_end', 'soc_min', 'soc_max',
    'ch_s_mode',
    'voltage_mean', 'current_mean', 'power_mean', 'speed_mean',
    'start_time', 'end_time', 'num_records',
]
events = events.reset_index(drop=True)

print(f"   Aggregated: {len(events):,} events ({time.time()-t_agg:.1f}s)")

# ============================================================
# 4. 派生字段 + 过滤
# ============================================================
print(f"\n🔧 Computing derived fields & filtering...")

events['soc_gain'] = events['soc_end'] - events['soc_start']

# 时长计算 (纳秒 → 秒)
duration_td = events['end_time'] - events['start_time']
events['duration_seconds'] = duration_td.astype('int64') / 1_000_000_000
events['duration_minutes'] = events['duration_seconds'] / 60.0

n_before = len(events)

# GPU 布尔过滤
mask = (
    (events['soc_gain'] >= MIN_SOC_GAIN) &
    (events['duration_seconds'] >= MIN_DURATION_SEC) &
    (events['num_records'] >= MIN_RECORDS) &
    (events['soc_start'] >= 0) &
    (events['soc_end'] <= 100)
)
events = events[mask].reset_index(drop=True)

print(f"   Before: {n_before:,} → After: {len(events):,} ({len(events)/max(n_before,1)*100:.1f}%)")

# 快充/慢充
events['avg_soc_rate'] = events['soc_gain'] / events['duration_minutes']

# ============================================================
# 5. 转回 CPU 保存
# ============================================================
print(f"\n💾 Transferring to CPU & saving...")
t_save = time.time()

df_events = events.to_pandas()

# charge_type 在 CPU 上赋值 (字符串操作)
df_events['charge_type'] = np.where(df_events['avg_soc_rate'] > 1.0, 'fast', 'slow')
df_events['charge_type_cn'] = np.where(df_events['avg_soc_rate'] > 1.0, '快充', '慢充')

# 生成 event_id
df_events['charging_event_id'] = [
    f"{vid}_ch_{i:05d}" for i, vid in enumerate(df_events['vehicle_id'])
]

t_total = time.time() - t_start

# ============================================================
# 6. 输出报告
# ============================================================
print(f"\n{'=' * 70}")
print(f"✅ RESULTS: {len(df_events):,} charging events")
print(f"{'=' * 70}")

print(f"\n⏱️ Total Time: {t_total:.1f}s ({t_total/60:.1f} min)")
total_bytes = sum(os.path.getsize(f) for f in csv_files)
print(f"   Throughput: {total_bytes/1024**3/t_total:.1f} GB/s")

print(f"\n📊 Summary:")
n_vehicles = df_events['vehicle_id'].nunique()
print(f"   Total events:     {len(df_events):,}")
print(f"   Unique vehicles:  {n_vehicles:,}")
print(f"   Events/vehicle:   {len(df_events) / max(n_vehicles, 1):.2f}")

fast_n = (df_events['charge_type'] == 'fast').sum()
slow_n = (df_events['charge_type'] == 'slow').sum()
print(f"\n⚡ Charging Type:")
print(f"   快充 (>1%/min): {fast_n:,} ({fast_n/max(len(df_events),1)*100:.1f}%)")
print(f"   慢充 (≤1%/min): {slow_n:,} ({slow_n/max(len(df_events),1)*100:.1f}%)")

for label, col, unit in [
    ('📈 SOC Gain', 'soc_gain', '%'),
    ('🔋 Starting SOC', 'soc_start', '%'),
    ('⚡ Charging Rate', 'avg_soc_rate', '%/min'),
]:
    print(f"\n{label}:")
    print(f"   Min:    {df_events[col].min():.2f} {unit}")
    print(f"   Median: {df_events[col].median():.2f} {unit}")
    print(f"   Mean:   {df_events[col].mean():.2f} {unit}")
    print(f"   Max:    {df_events[col].max():.2f} {unit}")

print(f"\n⏱️ Duration:")
print(f"   Min:    {df_events['duration_minutes'].min():.1f} min")
print(f"   Median: {df_events['duration_minutes'].median():.1f} min")
print(f"   Mean:   {df_events['duration_minutes'].mean():.1f} min")
print(f"   Max:    {df_events['duration_minutes'].max()/60:.1f} hours")

# 保存
output_cols = [
    'charging_event_id', 'vehicle_id', 'start_time', 'end_time',
    'soc_start', 'soc_end', 'soc_gain', 'soc_min', 'soc_max',
    'duration_seconds', 'duration_minutes',
    'ch_s_mode', 'charge_type', 'charge_type_cn',
    'avg_soc_rate', 'voltage_mean', 'current_mean', 'power_mean',
    'speed_mean', 'num_records',
]
output_path = os.path.join(OUTPUT_DIR, 'charging_events_raw_extracted.csv')
df_events[output_cols].to_csv(output_path, index=False)
print(f"\n💾 Saved: {output_path}")
print(f"   Size: {os.path.getsize(output_path)/1024/1024:.1f} MB")

# 对比旧数据
old_path = "./analysis_complete_vehicles/results/charging_events_rebuilt.csv"
if os.path.exists(old_path):
    df_old = pd.read_csv(old_path)
    print(f"\n{'=' * 70}")
    print(f"📊 New vs Old")
    print(f"{'=' * 70}")
    print(f"   {'Metric':<25} {'Old':>12} {'New':>12}")
    print(f"   {'─'*25} {'─'*12} {'─'*12}")
    print(f"   {'Total events':<25} {len(df_old):>12,} {len(df_events):>12,}")
    print(f"   {'Unique vehicles':<25} {df_old['vehicle_id'].nunique():>12,} {df_events['vehicle_id'].nunique():>12,}")
    print(f"   {'Avg SOC gain':<25} {df_old['soc_gain'].mean():>12.1f} {df_events['soc_gain'].mean():>12.1f}")

cp.get_default_memory_pool().free_all_blocks()
print(f"\n✅ Done! Total: {t_total:.1f}s")