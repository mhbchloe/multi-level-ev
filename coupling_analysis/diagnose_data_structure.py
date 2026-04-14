"""
诊断脚本：检查所有聚类输出文件的字段结构
"""

import pandas as pd
import numpy as np
import os
import json

print("=" * 80)
print("🔍 DATA STRUCTURE DIAGNOSIS")
print("=" * 80)

# ============================================================
# 1. 片段聚类数据
# ============================================================
print("\n" + "=" * 80)
print("1️⃣  SEGMENT CLUSTERING (Step 7)")
print("=" * 80)

seg_cluster_path = './analysis_complete_vehicles/results/clustering_v3/clustering_v3_results.npz'
if os.path.exists(seg_cluster_path):
    data = np.load(seg_cluster_path)
    print(f"\n   File: {seg_cluster_path}")
    print(f"   Keys: {list(data.files)}")
    print(f"   Shapes:")
    for key in data.files:
        print(f"      {key:<20}: {data[key].shape}")
else:
    print(f"   ❌ File not found: {seg_cluster_path}")

# 尝试找 summary 文件
summary_path = './analysis_complete_vehicles/results/clustering_v3/clustering_v3_summary.json'
if os.path.exists(summary_path):
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    print(f"\n   Summary file found:")
    print(f"      Keys: {list(summary.keys())}")
    if 'cluster_stats' in summary:
        print(f"      Cluster stats: {list(summary['cluster_stats'].keys())}")

# ============================================================
# 2. 车辆聚类数据
# ============================================================
print("\n" + "=" * 80)
print("2️⃣  VEHICLE CLUSTERING")
print("=" * 80)

vehicle_cluster_paths = [
    './vehicle_clustering/results/vehicle_clustering_results_v3.csv',
    './vehicle_clustering/results/vehicle_clustering_results.csv',
    './vehicle_clustering/results/clustering_results_v3.csv',
]

found_vehicle = False
for path in vehicle_cluster_paths:
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"\n   ✓ Found: {path}")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        print(f"\n   First few rows:")
        print(df.head())
        found_vehicle = True
        break

if not found_vehicle:
    print(f"   ❌ Vehicle clustering CSV not found")
    print(f"   Searched paths:")
    for path in vehicle_cluster_paths:
        print(f"      {path}")

# ============================================================
# 3. 行程数据
# ============================================================
print("\n" + "=" * 80)
print("3️⃣  INTER-CHARGE TRIPS")
print("=" * 80)

trip_paths = [
    './coupling_analysis/results/inter_charge_trips.csv',
    './coupling_analysis/inter_charge_trips.csv',
]

found_trips = False
for path in trip_paths:
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"\n   ✓ Found: {path}")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        print(f"\n   Sample data:")
        print(df.head())
        print(f"\n   Data types:")
        print(df.dtypes)
        found_trips = True
        break

if not found_trips:
    print(f"   ❌ Inter-charge trips CSV not found")

# ============================================================
# 4. 段数据
# ============================================================
print("\n" + "=" * 80)
print("4️⃣  SEGMENT DATA")
print("=" * 80)

segment_paths = [
    './analysis_complete_vehicles/results/segments_with_clustering.csv',
    './analysis_complete_vehicles/results/segments.csv',
    './analysis_complete_vehicles/results/discharge_segments.csv',
]

found_segments = False
for path in segment_paths:
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"\n   ✓ Found: {path}")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        print(f"\n   Sample data:")
        print(df.head(3))
        found_segments = True
        break

if not found_segments:
    print(f"   ❌ Segment data CSV not found")

# ============================================================
# 5. 列出所有相关目录
# ============================================================
print("\n" + "=" * 80)
print("📁 FILE LISTING")
print("=" * 80)

dirs_to_check = [
    './analysis_complete_vehicles/results/',
    './vehicle_clustering/results/',
    './coupling_analysis/results/',
]

for dir_path in dirs_to_check:
    if os.path.exists(dir_path):
        print(f"\n{dir_path}")
        files = sorted(os.listdir(dir_path))
        for f in files:
            full_path = os.path.join(dir_path, f)
            if os.path.isfile(full_path):
                size = os.path.getsize(full_path) / 1024
                print(f"   {f:<50} {size:>10.1f} KB")
            elif os.path.isdir(full_path):
                print(f"   [DIR] {f}")

print("\n" + "=" * 80)
print("✅ Diagnosis Complete!")
print("=" * 80)