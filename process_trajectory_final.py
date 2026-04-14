import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
import json

# ============ 1. 读取数据 ============
df = pd.read_csv('data_20250701_processed.csv')
print(f"原始数据: {len(df)} 条")

# ============ 2. 提取所需字段 ============
traj = df[['vehicle_id', 'lat', 'lon', 'datetime']].copy()

# ============ 3. 删除缺失和无效坐标 ============
traj = traj.dropna(subset=['lat', 'lon'])
traj = traj[(traj['lat'] > 3) & (traj['lat'] < 54) &
            (traj['lon'] > 73) & (traj['lon'] < 136)]
print(f"清洗后: {len(traj)} 条, {traj['vehicle_id'].nunique()} 辆车")

# ============ 4. 排序 ============
traj = traj.sort_values(['vehicle_id', 'datetime']).reset_index(drop=True)

# ============ 5. 高斯平滑 ============
sigma = 2

smoothed = []
for vid, group in traj.groupby('vehicle_id'):
    group = group.copy()
    if len(group) >= 5:
        group['lat_smooth'] = gaussian_filter1d(group['lat'].values, sigma=sigma)
        group['lon_smooth'] = gaussian_filter1d(group['lon'].values, sigma=sigma)
    else:
        group['lat_smooth'] = group['lat']
        group['lon_smooth'] = group['lon']
    smoothed.append(group)

result = pd.concat(smoothed, ignore_index=True)
print(f"✅ 平滑完成: {len(result)} 条, {result['vehicle_id'].nunique()} 辆车")

# ============ 6. 导出CSV ============
result[['vehicle_id', 'lat_smooth', 'lon_smooth']].to_csv(
    'trajectory_points.csv', index=False,
    header=['vehicle_id', 'lat', 'lon']
)
print("📁 trajectory_points.csv 已导出")

# ============ 7. 导出GeoJSON线文件 ============
features = []
for vid, group in result.groupby('vehicle_id'):
    if len(group) < 2:
        continue
    coords = [[row['lon_smooth'], row['lat_smooth']] for _, row in group.iterrows()]
    features.append({
        "type": "Feature",
        "properties": {"vehicle_id": vid},
        "geometry": {"type": "LineString", "coordinates": coords}
    })

geojson = {"type": "FeatureCollection", "features": features}
with open('trajectory_lines.geojson', 'w') as f:
    json.dump(geojson, f)

print(f"📁 trajectory_lines.geojson 已导出 ({len(features)} 条轨迹线)")