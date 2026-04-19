"""
完整集成脚本：
1. 为 segments 添加 trip_id
2. 整合片段聚类标签
3. 聚合到行程和车辆级别
4. 生成车辆聚类的准备数据
"""

import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
from tqdm import tqdm
from scipy.stats import entropy as scipy_entropy

print("=" * 80)
print("🔗 COMPLETE DATA INTEGRATION (Segments → Trips → Vehicles)")
print("=" * 80)

# ============================================================
# 0. 加载所有数据
# ============================================================
print("\n【STEP 0】Loading raw data...")

# 0.1 NPZ 聚类结果
seg_result = np.load('./analysis_complete_vehicles/results/clustering_v3/clustering_v3_results.npz')
labels = seg_result['labels']
seg_types = seg_result['seg_types']

phys_keys = ['avg_speed', 'avg_speed_mov', 'speed_std', 'speed_max',
             'acc_std_mov', 'heading_change', 'idle_ratio', 'soc_rate', 'power_mean', 'seg_length']
seg_phys = {k: seg_result[k] for k in phys_keys if k in seg_result}

print(f"   ✓ NPZ: {len(labels):,} segments, 4 clusters")

# 0.2 Segment 数据（已有聚类标签）
seg_df = pd.read_csv('./coupling_analysis/results/segments_with_cluster_labels.csv')
print(f"   ✓ Segment CSV: {len(seg_df):,} rows, {len(seg_df.columns)} columns")

# 0.3 行程数据
trips_df = pd.read_csv('./coupling_analysis/results/inter_charge_trips.csv')
print(f"   ✓ Trips CSV: {len(trips_df):,} trips, {trips_df['vehicle_id'].nunique():,} vehicles")

# 0.4 摘要
with open('./analysis_complete_vehicles/results/clustering_v3/clustering_v3_summary.json', 'r') as f:
    summary = json.load(f)

cluster_names = {}
for c_str, stats in summary['cluster_stats'].items():
    cluster_names[int(c_str)] = stats['label']

print(f"   ✓ Cluster names: {cluster_names}")

# ============================================================
# 1. 为 segments 添加 trip_id
# ============================================================
print(f"\n【STEP 1】Adding trip_id to segments...")

# 确保长度一致
if len(seg_df) != len(labels):
    print(f"   ⚠️  Length mismatch: {len(seg_df)} vs {len(labels)}")
    n = min(len(seg_df), len(labels))
    seg_df = seg_df.iloc[:n].reset_index(drop=True)
    labels = labels[:n]
    seg_types = seg_types[:n]
    for k in seg_phys:
        seg_phys[k] = seg_phys[k][:n]

# 验证 vehicle_id 一致性
if 'vehicle_id' in seg_df.columns:
    print(f"   ✓ Segment data has vehicle_id")
    n_vehicles_seg = seg_df['vehicle_id'].nunique()
    print(f"     Vehicles in segments: {n_vehicles_seg:,}")

# 核心逻辑：根据 vehicle_id 和 时间戳，为每个 segment 分配 trip_id
print(f"\n   Assigning trip_id based on time gaps...")

seg_df['start_dt'] = pd.to_datetime(seg_df['start_dt'])
seg_df['end_dt'] = pd.to_datetime(seg_df['end_dt'])
seg_df['duration_min'] = seg_df['duration_seconds'] / 60.0

# 按车辆排序
seg_df_sorted = seg_df.sort_values(['vehicle_id', 'start_dt']).reset_index(drop=True)

# 定义行程：同一车辆，时间间隔 < 30 分钟的连续片段
TRIP_GAP_MINUTES = 30  # 定义行程边界的时间间隔

trip_ids = []
current_trip_id = 0
last_vehicle = None
last_end_time = None

for idx, row in tqdm(seg_df_sorted.iterrows(), total=len(seg_df_sorted), 
                     desc="   🔄 Assigning trips", ncols=80):
    current_vehicle = row['vehicle_id']
    current_start = row['start_dt']
    
    # 检查是否是新行程
    if current_vehicle != last_vehicle:
        # 新车辆
        current_trip_id += 1
        new_trip = True
    elif last_end_time is None:
        # 第一个片段
        new_trip = True
    else:
        # 检查时间间隔
        time_gap = (current_start - last_end_time).total_seconds() / 60.0
        new_trip = time_gap > TRIP_GAP_MINUTES
        if new_trip:
            current_trip_id += 1
    
    # 为这个片段分配 trip_id
    trip_ids.append(current_trip_id)
    
    # 更新状态
    last_vehicle = current_vehicle
    last_end_time = row['end_dt']

seg_df_sorted['trip_id'] = trip_ids

# 恢复原始顺序
seg_df_sorted['original_index'] = seg_df.index
seg_df_final = seg_df_sorted.sort_values('original_index').reset_index(drop=True)
seg_df_final = seg_df_final.drop(columns=['original_index'])

print(f"   ✓ Generated {current_trip_id:,} trips")
print(f"   Sample trip assignments:")
for vid in seg_df_final['vehicle_id'].unique()[:3]:
    v_trips = seg_df_final[seg_df_final['vehicle_id'] == vid]['trip_id'].nunique()
    print(f"     {vid}: {v_trips} trips")

# ============================================================
# 2. 整合聚类标签和物理特征
# ============================================================
print(f"\n【STEP 2】Integrating cluster labels and physical features...")

seg_df_final['cluster_id'] = labels
seg_df_final['cluster_name'] = seg_df_final['cluster_id'].map(cluster_names)
seg_df_final['seg_type'] = seg_types

# 添加物理特征
for pk in phys_keys:
    if pk in seg_phys:
        seg_df_final[f'phys_{pk}'] = seg_phys[pk]

print(f"   ✓ Added: cluster_id, cluster_name, seg_type")
print(f"   ✓ Added {len(phys_keys)} physical features")

# 验证
print(f"\n   Cluster distribution:")
for c in sorted(seg_df_final['cluster_id'].unique()):
    count = (seg_df_final['cluster_id'] == c).sum()
    name = seg_df_final[seg_df_final['cluster_id'] == c]['cluster_name'].iloc[0]
    print(f"      C{c} ({name}): {count:>10,} ({count/len(seg_df_final)*100:>5.2f}%)")

# ============================================================
# 3. 聚合到行程级
# ============================================================
print(f"\n【STEP 3】Aggregating to trip level...")

trip_features = []

for (vehicle_id, trip_id), group in tqdm(seg_df_final.groupby(['vehicle_id', 'trip_id']),
                                         desc="   🔄 Trip aggregation", ncols=80):
    n_segs = len(group)
    
    # 基本信息
    feat = {
        'vehicle_id': vehicle_id,
        'trip_id': trip_id,
        'n_segments': n_segs,
        'trip_start': group['start_dt'].min(),
        'trip_end': group['end_dt'].max(),
    }
    
    # 计算行程时长
    trip_duration = (group['end_dt'].max() - group['start_dt'].min()).total_seconds() / 3600.0
    feat['trip_duration_hrs'] = trip_duration
    
    # SOC 变化
    soc_drops = group['soc_start'].values - group['soc_end'].values
    feat['soc_drop_total'] = soc_drops.sum()
    feat['soc_drop_mean'] = soc_drops.mean()
    
    # 聚类组成（占比）
    cluster_dist = group['cluster_id'].value_counts(normalize=True).to_dict()
    for c in range(4):
        feat[f'cluster_{c}_ratio'] = cluster_dist.get(c, 0.0)
    
    # 物理特征均值
    for pk in phys_keys:
        col = f'phys_{pk}'
        if col in group.columns:
            feat[f'trip_avg_{pk}'] = group[col].mean()
    
    trip_features.append(feat)

trip_agg_df = pd.DataFrame(trip_features)
print(f"   ✓ Aggregated {len(trip_agg_df):,} trips")

# ============================================================
# 4. 聚合到车辆级
# ============================================================
print(f"\n【STEP 4】Aggregating to vehicle level...")

vehicle_features = []

for vehicle_id, v_group in tqdm(seg_df_final.groupby('vehicle_id'),
                                desc="   🔄 Vehicle aggregation", ncols=80):
    n_segs = len(v_group)
    n_trips = v_group['trip_id'].nunique()
    
    # 基本信息
    feat = {
        'vehicle_id': vehicle_id,
        'n_segments': n_segs,
        'n_trips': n_trips,
    }
    
    # ① 分布特征 (Distribution)
    cluster_dist = v_group['cluster_id'].value_counts(normalize=True).to_dict()
    ratios = []
    for c in range(4):
        r = cluster_dist.get(c, 0.0)
        feat[f'cluster_{c}_ratio'] = r
        ratios.append(r)
    
    # 模式多样性 (Shannon entropy of cluster distribution)
    ratios_arr = np.array(ratios)
    ratios_pos = ratios_arr[ratios_arr > 0]
    feat['mode_diversity'] = float(scipy_entropy(ratios_pos, base=2)) if len(ratios_pos) > 0 else 0.0
    
    # ② 转移特征 (Transition)
    clusters = v_group['cluster_id'].values
    
    # 完整 4×4 转移矩阵 (always 16 features, no sparsity)
    trans_matrix = np.zeros((4, 4), dtype=np.float64)
    if len(clusters) > 1:
        for i in range(len(clusters) - 1):
            from_c = int(clusters[i])
            to_c = int(clusters[i + 1])
            trans_matrix[from_c, to_c] += 1
        trans_total = trans_matrix.sum()
        if trans_total > 0:
            trans_matrix /= trans_total
    
    for fc in range(4):
        for tc in range(4):
            feat[f'trans_{fc}_to_{tc}'] = float(trans_matrix[fc, tc])
    
    # 模式切换率 (mode switch rate)
    if len(clusters) > 1:
        n_switches = sum(1 for i in range(len(clusters) - 1) if clusters[i] != clusters[i + 1])
        feat['mode_switch_rate'] = n_switches / (len(clusters) - 1)
    else:
        feat['mode_switch_rate'] = 0.0
    
    # 转移熵 (transition entropy)
    trans_flat = trans_matrix.flatten()
    trans_pos = trans_flat[trans_flat > 0]
    feat['transition_entropy'] = float(scipy_entropy(trans_pos, base=2)) if len(trans_pos) > 0 else 0.0
    
    # 自环比例 (self-loop ratio: staying in same mode)
    self_loop = np.trace(trans_matrix)
    feat['self_loop_ratio'] = float(self_loop)
    
    # ③ 演化特征 (Evolution)
    # 平均连续运行长度 (average run length in same mode)
    if len(clusters) > 1:
        run_lengths = []
        current_run = 1
        for i in range(1, len(clusters)):
            if clusters[i] == clusters[i - 1]:
                current_run += 1
            else:
                run_lengths.append(current_run)
                current_run = 1
        run_lengths.append(current_run)
        feat['avg_run_length'] = float(np.mean(run_lengths))
    else:
        feat['avg_run_length'] = float(n_segs)
    
    # 物理特征均值和标准差
    for pk in phys_keys:
        col = f'phys_{pk}'
        if col in v_group.columns:
            feat[f'avg_{pk}'] = v_group[col].mean()
            feat[f'std_{pk}'] = v_group[col].std() if n_segs > 1 else 0.0
    
    # SOC 和能耗
    soc_drops = v_group['soc_start'].values - v_group['soc_end'].values
    feat['avg_soc_drop_per_segment'] = soc_drops.mean()
    feat['max_soc_drop'] = soc_drops.max() if len(soc_drops) > 0 else 0.0
    feat['soc_consumption_rate'] = soc_drops.sum() / max(v_group['duration_seconds'].sum() / 3600.0, 1e-6)
    feat['total_duration_hrs'] = v_group['duration_seconds'].sum() / 3600.0
    
    vehicle_features.append(feat)

vehicle_agg_df = pd.DataFrame(vehicle_features)
print(f"   ✓ Aggregated {len(vehicle_agg_df):,} vehicles")

# ============================================================
# 5. 与现有行程数据对齐
# ============================================================
print(f"\n【STEP 5】Aligning with existing trip data...")

# 如果 inter_charge_trips.csv 有对应的行程，添加聚类特征
if len(trip_agg_df) > 0:
    # 尝试匹配 trip_id 和 vehicle_id
    trips_df_with_clusters = trips_df.copy()
    
    # 创建匹配键
    trip_agg_df['merge_key'] = trip_agg_df['vehicle_id'] + '_' + trip_agg_df['trip_id'].astype(str)
    trips_df_with_clusters['merge_key'] = trips_df_with_clusters['vehicle_id'] + '_' + \
                                           trips_df_with_clusters['trip_id'].astype(str)
    
    # 尝试合并
    merged = trips_df_with_clusters.merge(
        trip_agg_df[['merge_key', 'cluster_0_ratio', 'cluster_1_ratio', 
                     'cluster_2_ratio', 'cluster_3_ratio', 'n_segments']],
        on='merge_key', how='left'
    )
    
    match_rate = (~merged['cluster_0_ratio'].isna()).sum() / len(merged) * 100
    print(f"   ✓ Matched {match_rate:.1f}% of trips with cluster info")
    
    trips_df_with_clusters = merged.drop(columns=['merge_key'])
    
    output_path = './coupling_analysis/results/inter_charge_trips_with_clusters.csv'
    trips_df_with_clusters.to_csv(output_path, index=False)
    print(f"   ✓ Saved: {output_path}")

# ============================================================
# 6. 保存所有文件
# ============================================================
print(f"\n【STEP 6】Saving integrated data...")

output_dir = './coupling_analysis/results/'
os.makedirs(output_dir, exist_ok=True)

# 6.1 段级别（最详细）
seg_output_path = os.path.join(output_dir, 'segments_integrated_complete.csv')
seg_df_final.to_csv(seg_output_path, index=False)
size_mb = os.path.getsize(seg_output_path) / 1024 / 1024
print(f"   ✓ segments_integrated_complete.csv ({size_mb:.1f} MB)")
print(f"     Columns: {len(seg_df_final.columns)}")

# 6.2 行程级别
trip_output_path = os.path.join(output_dir, 'trips_aggregated_with_clusters.csv')
trip_agg_df.to_csv(trip_output_path, index=False)
print(f"   ✓ trips_aggregated_with_clusters.csv")
print(f"     Columns: {len(trip_agg_df.columns)}")

# 6.3 车辆级别
vehicle_output_path = os.path.join(output_dir, 'vehicles_aggregated_features.csv')
vehicle_agg_df.to_csv(vehicle_output_path, index=False)
print(f"   ✓ vehicles_aggregated_features.csv")
print(f"     Columns: {len(vehicle_agg_df.columns)}")

# 6.4 元数据
metadata = {
    'timestamp': datetime.now().isoformat(),
    'n_segments': len(seg_df_final),
    'n_trips': len(trip_agg_df),
    'n_vehicles': len(vehicle_agg_df),
    'cluster_names': cluster_names,
    'cluster_distribution': {
        str(c): int((seg_df_final['cluster_id'] == c).sum()) 
        for c in range(4)
    },
    'integration_info': {
        'trip_gap_threshold_minutes': TRIP_GAP_MINUTES,
        'segment_source': 'clustering_v3/clustering_v3_results.npz',
        'physical_features': phys_keys,
    }
}

metadata_path = os.path.join(output_dir, 'integration_metadata.json')
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2, default=str)
print(f"   ✓ integration_metadata.json")

# ============================================================
# 7. 验证数据质量
# ============================================================
print(f"\n【STEP 7】Data Quality Verification...")

print(f"\n   Segment Level:")
print(f"      Total rows: {len(seg_df_final):,}")
print(f"      Columns: {len(seg_df_final.columns)}")
print(f"      Missing values: {seg_df_final.isna().sum().sum():,}")
print(f"      Vehicles: {seg_df_final['vehicle_id'].nunique():,}")

print(f"\n   Trip Level:")
print(f"      Total rows: {len(trip_agg_df):,}")
print(f"      Vehicles: {trip_agg_df['vehicle_id'].nunique():,}")
print(f"      Avg segments/trip: {trip_agg_df['n_segments'].mean():.1f}")

print(f"\n   Vehicle Level:")
print(f"      Total rows: {len(vehicle_agg_df):,}")
print(f"      Avg segments/vehicle: {vehicle_agg_df['n_segments'].mean():.1f}")
print(f"      Avg trips/vehicle: {vehicle_agg_df['n_trips'].mean():.1f}")

print(f"\n   Cluster Distribution:")
for c in range(4):
    ratio_col = f'cluster_{c}_ratio'
    if ratio_col in vehicle_agg_df.columns:
        mean_ratio = vehicle_agg_df[ratio_col].mean()
        print(f"      C{c} ({cluster_names[c]:>15}): {mean_ratio:>6.2%} vehicles on average")

# ============================================================
# 完成
# ============================================================
print("\n" + "=" * 80)
print("✅ INTEGRATION COMPLETE!")
print("=" * 80)

print(f"""
Generated Files:
  1. segments_integrated_complete.csv       ({len(seg_df_final):,} rows)
     - segment_id, vehicle_id, trip_id
     - cluster_id, cluster_name
     - Physical features (avg_speed, power_mean, etc.)
     
  2. trips_aggregated_with_clusters.csv    ({len(trip_agg_df):,} rows)
     - trip_id, vehicle_id
     - cluster_0_ratio ~ cluster_3_ratio
     - soc_drop, duration
     
  3. vehicles_aggregated_features.csv      ({len(vehicle_agg_df):,} rows)
     - vehicle_id
     - cluster_0_ratio ~ cluster_3_ratio
     - Driving behavior indices
     
  4. inter_charge_trips_with_clusters.csv
     - Original trips + cluster features

Next Steps:
  1. Use vehicles_aggregated_features.csv for vehicle clustering
  2. Run: python vehicle_clustering/step8_vehicle_clustering_fixed.py
  3. Update the config to use segments_integrated_complete.csv
""")