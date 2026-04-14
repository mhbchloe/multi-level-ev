import pandas as pd
import numpy as np
import json

# ============ 读取已处理好的点数据 ============
result = pd.read_csv('hainan_trajectory_points.csv')
print(f"原始点数: {len(result)}")

# ============ 每10个点取1个（422MB ÷ 10 ≈ 40MB左右）============
sampled = result.groupby('vehicle_id').apply(
    lambda g: g.iloc[::10]
).reset_index(drop=True)
print(f"降采样后: {len(sampled)} 条")

# ============ 导出轻量GeoJSON ============
features = []
for vid, group in sampled.groupby('vehicle_id'):
    if len(group) < 2:
        continue
    lons = group['lon'].values
    lats = group['lat'].values
    coords = np.column_stack([lons, lats]).tolist()
    features.append({
        "type": "Feature",
        "properties": {"vehicle_id": vid},
        "geometry": {"type": "LineString", "coordinates": coords}
    })

geojson = {"type": "FeatureCollection", "features": features}
with open('hainan_trajectory_light.geojson', 'w') as f:
    json.dump(geojson, f)

print(f"📁 hainan_trajectory_light.geojson ({len(features)} 条轨迹线)")