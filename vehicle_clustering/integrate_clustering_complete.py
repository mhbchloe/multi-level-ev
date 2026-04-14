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

print("=" * 80)
print("🔗 COMPLETE DATA INTEGRATION (Segments → Trips → Vehicles)")
print("=" * 80)

# ============================================================
# 0. 加载所有数据
# ============================================================
print("\n【STEP 0】Loading raw data...")

# 0.1 NPZ 聚类结果
seg_result = np.load('./anis_complete_vehicles/results/clustering_v3/clustering_v3_results.npz')
labels = seg_result['labels']
seg_types = seg_result['seg_types']

phys_keys = ['avg_speed', 'avg_speed_mov', 'speed_std', 'speed_max',
             'acc_std_mov', 'heading_change', 'idle_ratio', 'soc_rate', 'power_mean', 'seg_length']
seg_phys = {k: seg_result[k] for k in phys_keys if k in seg_result}

print(f"   ✓ NPZ: {len(labels):,} segments, 4 clusters")

# 0.2 Segment 数据（已有聚类标签）
seg_df = pd.read_csv('./anis_complete_vehicles/results/segments_with_cluster_labels.csv')
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
# 4. 聚合到车辆级 — 三维特征框架
#    ① Distribution (4)  ② Transition (16+)  ③ Evolution (~)
# ============================================================
print(f"\n【STEP 4】Aggregating to vehicle level (3-Dimension Framework)")

from scipy.stats import entropy as scipy_entropy

K = 4  # 片段聚类数

vehicle_features = []

for vehicle_id, v_group in tqdm(seg_df_final.groupby('vehicle_id'),
                                desc="   🔄 Vehicle aggregation", ncols=80):
    v_sorted = v_group.sort_values('start_dt')
    n_segs = len(v_sorted)
    n_trips = v_sorted['trip_id'].nunique()
    clusters = v_sorted['cluster_id'].astype(int).values

    feat = {
        'vehicle_id': vehicle_id,
        'n_segments': n_segs,
        'n_trips': n_trips,
    }

    # =============================================
    # ① Distribution Features (4 维)
    # =============================================
    cluster_dist = v_sorted['cluster_id'].value_counts(normalize=True).to_dict()
    for c in range(K):
        feat[f'cluster_{c}_ratio'] = cluster_dist.get(c, 0.0)

    # =============================================
    # ② Transition Dynamics (K×K=16 + 辅助指标)
    # =============================================
    T = np.zeros((K, K))
    if n_segs > 1:
        for t in range(n_segs - 1):
            T[clusters[t], clusters[t + 1]] += 1
        # 行归一化 → 转移概率矩阵
        row_sums = T.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        T_prob = T / row_sums
    else:
        T_prob = np.zeros((K, K))

    # flatten 为 16 个特征
    for i in range(K):
        for j in range(K):
            feat[f'trans_{i}_to_{j}'] = float(T_prob[i, j])

    # 转移熵 (transition entropy): 越低越规律
    if n_segs > 1:
        trans_entropies = []
        for i in range(K):
            row = T_prob[i]
            if row.sum() > 0:
                trans_entropies.append(scipy_entropy(row + 1e-12))
        feat['transition_entropy'] = float(np.mean(trans_entropies)) if trans_entropies else 0.0
    else:
        feat['transition_entropy'] = 0.0

    # 模式切换率
    if n_segs > 1:
        switches = sum(1 for t in range(n_segs - 1) if clusters[t] != clusters[t + 1])
        feat['mode_switch_rate'] = switches / (n_segs - 1)
    else:
        feat['mode_switch_rate'] = 0.0

    # 自循环比率
    if n_segs > 1:
        self_loops = sum(1 for t in range(n_segs - 1) if clusters[t] == clusters[t + 1])
        feat['self_loop_ratio'] = self_loops / (n_segs - 1)
    else:
        feat['self_loop_ratio'] = 0.0

    # =============================================
    # ③ Evolution / Rhythm / Stability Features
    # =============================================
    # --- 状态熵 (state entropy): 模式多样性 ---
    p_dist = np.array([cluster_dist.get(c, 0.0) for c in range(K)])
    feat['state_entropy'] = float(scipy_entropy(p_dist + 1e-12))

    # --- 能量累积 (Energy accumulation) ---
    soc_drops = v_sorted['soc_start'].values - v_sorted['soc_end'].values
    durations_min = v_sorted['duration_seconds'].values / 60.0
    total_duration_min = durations_min.sum()

    # SOC 消耗速率 (%/min)
    feat['soc_consumption_rate'] = float(soc_drops.sum() / total_duration_min) if total_duration_min > 0 else 0.0
    # 最大 SOC drop（极端能耗事件）
    feat['max_soc_drop'] = float(soc_drops.max()) if len(soc_drops) > 0 else 0.0
    feat['avg_soc_drop_per_segment'] = float(soc_drops.mean()) if len(soc_drops) > 0 else 0.0
    feat['total_duration_hrs'] = float(total_duration_min / 60.0)

    # 连续高能耗段（cluster 2 & 3）最长长度和次数
    high_energy_mask = np.isin(clusters, [2, 3])
    max_consecutive_high = 0
    count_consecutive_high = 0
    current_run = 0
    for flag in high_energy_mask:
        if flag:
            current_run += 1
        else:
            if current_run > 0:
                max_consecutive_high = max(max_consecutive_high, current_run)
                count_consecutive_high += 1
            current_run = 0
    if current_run > 0:
        max_consecutive_high = max(max_consecutive_high, current_run)
        count_consecutive_high += 1
    feat['max_consecutive_high_energy'] = max_consecutive_high
    feat['count_consecutive_high_energy'] = count_consecutive_high

    # --- 节奏 (Temporal rhythm) ---
    # 各模式平均持续时间（秒）
    for c in range(K):
        mode_mask = clusters == c
        mode_durations = v_sorted['duration_seconds'].values[mode_mask]
        feat[f'mode_{c}_avg_duration'] = float(mode_durations.mean()) if len(mode_durations) > 0 else 0.0

    # 平均 run length（连续相同模式的平均长度）
    if n_segs > 0:
        run_lengths = []
        current_run = 1
        for t in range(1, n_segs):
            if clusters[t] == clusters[t - 1]:
                current_run += 1
            else:
                run_lengths.append(current_run)
                current_run = 1
        run_lengths.append(current_run)
        feat['avg_run_length'] = float(np.mean(run_lengths))
    else:
        feat['avg_run_length'] = 0.0

    # --- 驾驶行为指标（保留兼容） ---
    feat['high_energy_ratio'] = (v_sorted['cluster_id'].isin([2, 3])).sum() / n_segs
    feat['idle_dominant_ratio'] = (v_sorted['cluster_id'] == 0).sum() / n_segs

    # --- 物理特征均值 ---
    for pk in phys_keys:
        col = f'phys_{pk}'
        if col in v_sorted.columns:
            feat[f'avg_{pk}'] = v_sorted[col].mean()

    vehicle_features.append(feat)

vehicle_agg_df = pd.DataFrame(vehicle_features)

# 确保所有转移矩阵列存在（填充为 0）
for i in range(K):
    for j in range(K):
        col = f'trans_{i}_to_{j}'
        if col not in vehicle_agg_df.columns:
            vehicle_agg_df[col] = 0.0
vehicle_agg_df = vehicle_agg_df.fillna(0.0)

print(f"   ✓ Aggregated {len(vehicle_agg_df):,} vehicles")
print(f"   Feature dimensions:")
dist_cols = [f'cluster_{c}_ratio' for c in range(K)]
trans_cols = [f'trans_{i}_to_{j}' for i in range(K) for j in range(K)]
evol_cols = ['transition_entropy', 'mode_switch_rate', 'self_loop_ratio',
             'state_entropy', 'soc_consumption_rate', 'max_soc_drop',
             'avg_soc_drop_per_segment', 'max_consecutive_high_energy',
             'count_consecutive_high_energy', 'avg_run_length',
             'total_duration_hrs'] + [f'mode_{c}_avg_duration' for c in range(K)]
print(f"      ① Distribution: {len(dist_cols)} features")
print(f"      ② Transition:   {len(trans_cols)} features (+3 auxiliary)")
print(f"      ③ Evolution:    {len(evol_cols)} features")

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
    },
    'vehicle_feature_framework': {
        'distribution_features': [f'cluster_{c}_ratio' for c in range(K)],
        'transition_features': [f'trans_{i}_to_{j}' for i in range(K) for j in range(K)]
                                + ['transition_entropy', 'mode_switch_rate', 'self_loop_ratio'],
        'evolution_features': ['state_entropy', 'soc_consumption_rate', 'max_soc_drop',
                               'avg_soc_drop_per_segment', 'max_consecutive_high_energy',
                               'count_consecutive_high_energy', 'avg_run_length',
                               'total_duration_hrs']
                              + [f'mode_{c}_avg_duration' for c in range(K)],
    },
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
     - ① Distribution: cluster_0_ratio ~ cluster_3_ratio  (4 features)
     - ② Transition:   trans_i_to_j (16) + entropy/switch_rate/self_loop (3)
     - ③ Evolution:    energy, rhythm, stability metrics ({len(evol_cols)} features)
     
  4. inter_charge_trips_with_clusters.csv
     - Original trips + cluster features

Next Steps:
  1. Use vehicles_aggregated_features.csv for vehicle clustering
  2. Run: python vehicle_clustering/step8_vehicle_clustering.py
  3. Run: python coupling_analysis/step_3_3_coupling_analysis.py
""")