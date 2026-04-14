"""
诊断脚本：检查片段聚类结果 + 验证数据集成
确保后续车辆聚类能正确使用
"""

import numpy as np
import pandas as pd
import json
import os
from collections import Counter

print("=" * 80)
print("🔍 CLUSTERING INTEGRITY CHECK & DATA INTEGRATION")
print("=" * 80)

# ============================================================
# 1. 检查片段聚类结果的字段
# ============================================================
print("\n" + "=" * 80)
print("1️⃣  SEGMENT CLUSTERING RESULTS (Step 7)")
print("=" * 80)

seg_result_path = './analysis_complete_vehicles/results/clustering_v3/clustering_v3_results.npz'
if os.path.exists(seg_result_path):
    data = np.load(seg_result_path)
    print(f"\n   ✓ Found: {seg_result_path}")
    print(f"\n   NPZ Keys:")
    for key in sorted(data.files):
        arr = data[key]
        print(f"      {key:<25} Shape: {str(arr.shape):<20} dtype: {arr.dtype}")
    
    # 关键检查
    labels = data['labels']
    seg_types = data['seg_types']
    
    print(f"\n   📊 Clustering Statistics:")
    print(f"      Total segments: {len(labels):,}")
    print(f"      Unique clusters: {np.unique(labels)}")
    print(f"      Cluster distribution:")
    for c in sorted(np.unique(labels)):
        count = (labels == c).sum()
        print(f"         C{c}: {count:>10,} ({count/len(labels)*100:>6.2f}%)")
    
    print(f"\n   📊 Segment Type Statistics:")
    print(f"      Unique types: {np.unique(seg_types)}")
    for t in sorted(np.unique(seg_types)):
        count = (seg_types == t).sum()
        print(f"         Type {t}: {count:>10,} ({count/len(seg_types)*100:>6.2f}%)")
    
    # 物理特征
    print(f"\n   📊 Physical Features Available:")
    phys_keys = ['avg_speed', 'avg_speed_mov', 'speed_std', 'speed_max',
                 'acc_std_mov', 'heading_change', 'idle_ratio',
                 'soc_rate', 'power_mean', 'seg_length']
    for pk in phys_keys:
        if pk in data:
            arr = data[pk]
            print(f"      {pk:<20} min={arr.min():.2f}, max={arr.max():.2f}, mean={arr.mean():.2f}")
        else:
            print(f"      {pk:<20} ❌ MISSING")
    
    # Z_pca
    if 'z_pca' in data:
        z_pca = data['z_pca']
        print(f"\n   📊 Latent Features (z_pca):")
        print(f"      Shape: {z_pca.shape}")
        print(f"      Dimensions: {z_pca.shape[1]}")

else:
    print(f"   ❌ NOT FOUND: {seg_result_path}")

# ============================================================
# 2. 检查聚类摘要文件
# ============================================================
print("\n" + "=" * 80)
print("2️⃣  CLUSTERING SUMMARY")
print("=" * 80)

summary_path = './analysis_complete_vehicles/results/clustering_v3/clustering_v3_summary.json'
if os.path.exists(summary_path):
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    print(f"\n   ✓ Found: {summary_path}")
    print(f"\n   Summary Keys: {list(summary.keys())}")
    
    print(f"\n   Best Method: {summary.get('best_method', 'N/A')}")
    print(f"   N Clusters: {summary.get('n_clusters', 'N/A')}")
    print(f"   PCA Dims: {summary.get('pca_dims', 'N/A')}")
    print(f"   PCA Variance Retained: {summary.get('pca_variance_retained', 'N/A'):.2%}")
    
    # 聚类统计
    if 'cluster_stats' in summary:
        print(f"\n   Cluster Statistics:")
        for cluster_id, stats in summary['cluster_stats'].items():
            print(f"      Cluster {cluster_id}:")
            print(f"         Label: {stats.get('label', 'N/A')}")
            print(f"         Size: {stats.get('size', 'N/A')}")
            print(f"         Pct: {stats.get('pct', 'N/A'):.2f}%")

else:
    print(f"   ❌ NOT FOUND: {summary_path}")

# ============================================================
# 3. 检查 segments_with_cluster_labels.csv
# ============================================================
print("\n" + "=" * 80)
print("3️⃣  SEGMENT-TRIP MAPPING")
print("=" * 80)

seg_trip_path = './coupling_analysis/results/segments_with_cluster_labels.csv'
if os.path.exists(seg_trip_path):
    seg_trip_df = pd.read_csv(seg_trip_path)
    print(f"\n   ✓ Found: {seg_trip_path}")
    print(f"   Shape: {seg_trip_df.shape}")
    print(f"\n   Columns: {list(seg_trip_df.columns)}")
    
    print(f"\n   Sample rows:")
    print(seg_trip_df.head(3))
    
    print(f"\n   Data Types:")
    print(seg_trip_df.dtypes)
    
    # 检查关键列
    print(f"\n   ✅ Key Columns Check:")
    required = ['vehicle_id', 'trip_id', 'segment_id']
    for col in required:
        if col in seg_trip_df.columns:
            n_unique = seg_trip_df[col].nunique()
            print(f"      {col:<20} ✓ Unique: {n_unique:>10,}")
        else:
            print(f"      {col:<20} ❌ MISSING")
    
    # 检查聚类信息
    cluster_cols = [c for c in seg_trip_df.columns if 'cluster' in c.lower()]
    if cluster_cols:
        print(f"\n   Cluster-related columns: {cluster_cols}")
        for col in cluster_cols:
            if seg_trip_df[col].dtype in ['int64', 'float64']:
                print(f"      {col}: unique values = {seg_trip_df[col].nunique()}")
    else:
        print(f"   ⚠️  No cluster columns found in CSV!")

else:
    print(f"   ❌ NOT FOUND: {seg_trip_path}")

# ============================================================
# 4. 检查行程数据
# ============================================================
print("\n" + "=" * 80)
print("4️⃣  INTER-CHARGE TRIPS")
print("=" * 80)

trips_path = './coupling_analysis/results/inter_charge_trips.csv'
if os.path.exists(trips_path):
    trips_df = pd.read_csv(trips_path)
    print(f"\n   ✓ Found: {trips_path}")
    print(f"   Shape: {trips_df.shape}")
    print(f"   Columns: {list(trips_df.columns)}")
    
    print(f"\n   Sample rows:")
    print(trips_df[['trip_id', 'vehicle_id', 'trip_duration_hrs', 'soc_drop']].head(3))
    
    print(f"\n   Unique values:")
    print(f"      Vehicles: {trips_df['vehicle_id'].nunique():,}")
    print(f"      Trips: {trips_df['trip_id'].nunique():,}")

else:
    print(f"   ❌ NOT FOUND: {trips_path}")

# ============================================================
# 5. 核心问题：检查数据一致性
# ============================================================
print("\n" + "=" * 80)
print("5️⃣  DATA CONSISTENCY CHECK")
print("=" * 80)

try:
    seg_result = np.load(seg_result_path)
    seg_trip_df = pd.read_csv(seg_trip_path)
    trips_df = pd.read_csv(trips_path)
    
    labels = seg_result['labels']
    
    print(f"\n   长度匹配检查:")
    print(f"      NPZ labels: {len(labels):,}")
    print(f"      CSV rows: {len(seg_trip_df):,}")
    if len(labels) == len(seg_trip_df):
        print(f"      ✓ 完美匹配！")
    else:
        print(f"      ⚠️  差异: {abs(len(labels) - len(seg_trip_df)):,}")
    
    # 尝试对齐
    if len(labels) == len(seg_trip_df):
        seg_trip_df['cluster_from_npz'] = labels
        
        print(f"\n   已将 NPZ 聚类标签添加到 CSV")
        print(f"   Cluster distribution (from NPZ):")
        for c in sorted(np.unique(labels)):
            count = (labels == c).sum()
            print(f"      C{c}: {count:>10,}")
        
        # 检查是否有现存的聚类标签
        if 'cluster' in seg_trip_df.columns:
            print(f"\n   ⚠️  CSV 中已有 'cluster' 列，对比:")
            existing = seg_trip_df['cluster'].values
            match_rate = (existing == labels).sum() / len(labels) * 100
            print(f"      匹配率: {match_rate:.1f}%")
            if match_rate < 99:
                print(f"      可能需要更新！")
        
        # 检查行程-车辆映射
        print(f"\n   行程-车辆映射检查:")
        seg_trip_df['trip_id'] = seg_trip_df['trip_id'].fillna('UNKNOWN')
        trips_in_seg = seg_trip_df['trip_id'].nunique()
        trips_in_trips_df = trips_df['trip_id'].nunique()
        print(f"      段中的行程: {trips_in_seg:,}")
        print(f"      trips CSV 中的行程: {trips_in_trips_df:,}")
        
        # 检查车辆映射
        print(f"\n   车辆映射检查:")
        vehicles_in_seg = seg_trip_df['vehicle_id'].nunique()
        vehicles_in_trips_df = trips_df['vehicle_id'].nunique()
        print(f"      段中的车辆: {vehicles_in_seg:,}")
        print(f"      trips CSV 中的车辆: {vehicles_in_trips_df:,}")

except Exception as e:
    print(f"   ❌ 对齐失败: {e}")

# ============================================================
# 6. 建议
# ============================================================
print("\n" + "=" * 80)
print("💡 RECOMMENDATIONS")
print("=" * 80)

print(f"""
根据诊断结果，你需要：

1️⃣  【必做】确保片段聚类标签集成到原始数据中：
   - 如果 segments_with_cluster_labels.csv 没有聚类标签列
   - 需要从 NPZ 中读取 labels 并添加到 CSV
   - 这样后续车辆聚类才能正确使用

2️⃣  【必做】验证三层数据对齐：
   - segment_id ↔ trip_id (1:many)
   - trip_id ↔ vehicle_id (1:1)
   - 长度必须完全匹配或有明确的映射规则

3️⃣  【推荐】生成集成数据文件：
   运行下一个脚本：integrate_clustering_results.py
   - 自动添加聚类标签到 segment 数据
   - 生成规范化的 vehicle_segment_features.csv
   - 为车辆聚类做好准备

4️⃣  【检查】聚类质量：
   - 查看 clustering_v3_summary.json 中的分数
   - 确保 Silhouette > 0.3 (好)，> 0.5 (优)
   - 检查聚类大小分布是否均衡

""")

print("=" * 80)
