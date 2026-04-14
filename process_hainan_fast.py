import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
import json

# ============ 1. 读取数据 ============
df = pd.read_csv('data_20250701_processed.csv')
print(f"原始数据: {len(df)} 条")

# ============ 2. 提取 + 清洗 + 只保留海南 ============
traj = df[['vehicle_id', 'lat', 'lon', 'datetime']].copy()
traj = traj.dropna(subset=['lat', 'lon'])
traj = traj[(traj['lat'] >= 18.1) & (traj['lat'] <= 20.2) &
            (traj['lon'] >= 108.6) & (traj['lon'] <= 111.4)]
traj = traj.sort_values(['vehicle_id', 'datetime']).reset_index(drop=True)
print(f"海南范围: {len(traj)} 条, {traj['vehicle_id'].nunique()} 辆车")

# ============ 3. 高斯平滑 ============
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

# ============ 4. 导出CSV ============
result[['vehicle_id', 'lat_smooth', 'lon_smooth']].to_csv(
    'hainan_trajectory_points.csv', index=False,
    header=['vehicle_id', 'lat', 'lon']
)
print("📁 hainan_trajectory_points.csv 已导出")

# ============ 5. 快速导出GeoJSON（核心优化）============
print("正在生成GeoJSON...")

features = []
# 用 numpy 数组代替 iterrows，速度快几十倍
for vid, group in result.groupby('vehicle_id'):
    if len(group) < 2:
        continue
    # 直接取 numpy 数组，避免逐行遍历
    lons = group['lon_smooth'].values
    lats = group['lat_smooth'].values
    coords = np.column_stack([lons, lats]).tolist()

    features.append({
        "type": "Feature",
        "properties": {"vehicle_id": vid},
        "geometry": {"type": "LineString", "coordinates": coords}
    })

    if len(features) % 1000 == 0:
        print(f"  已处理 {len(features)} 条轨迹...")

geojson = {"type": "FeatureCollection", "features": features}

# 用较快的方式写入
with open('hainan_trajectory_lines.geojson', 'w') as f:
    json.dump(geojson, f)

print(f"📁 hainan_trajectory_lines.geojson 已导出 ({len(features)} 条轨迹线)")