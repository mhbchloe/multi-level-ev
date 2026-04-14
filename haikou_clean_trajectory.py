import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
import json

# ============ 1. 读取数据 ============
df = pd.read_csv('data_20250701_processed.csv')
print(f"原始数据: {len(df)} 条")

# ============ 2. 提取 + 只保留海口市 ============
traj = df[['vehicle_id', 'lat', 'lon', 'datetime']].copy()
traj = traj.dropna(subset=['lat', 'lon'])
traj = traj[(traj['lat'] >= 19.75) & (traj['lat'] <= 20.10) &
            (traj['lon'] >= 110.10) & (traj['lon'] <= 110.70)]
traj = traj.sort_values(['vehicle_id', 'datetime']).reset_index(drop=True)
print(f"海口范围: {len(traj)} 条, {traj['vehicle_id'].nunique()} 辆车")

# ============ 3. 计算相邻点距离，过滤GPS漂移 ============
def haversine_np(lat1, lon1, lat2, lon2):
    """批量计算两组坐标间的距离（米）"""
    R = 6371000  # 地球半径（米）
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

print("正在过滤GPS漂移...")
cleaned = []
count = 0

for vid, group in traj.groupby('vehicle_id'):
    group = group.copy().reset_index(drop=True)
    
    if len(group) < 2:
        continue
    
    # 计算相邻点之间的距离（米）
    lat_arr = group['lat'].values
    lon_arr = group['lon'].values
    
    dist = np.zeros(len(group))
    dist[1:] = haversine_np(lat_arr[:-1], lon_arr[:-1], lat_arr[1:], lon_arr[1:])
    
    # ===== 核心：过滤漂移点 =====
    # 规则1: 相邻点距离 > 5000米（5km）视为GPS跳变，删除
    # 规则2: 相邻点距离 > 2000米 且 速度隐含 > 200km/h 也删除
    #        （假设采样间隔10秒，2000m/10s = 200m/s = 720km/h 明显异常）
    
    mask = dist <= 5000  # 保留距离 <= 5km 的点
    group = group[mask].reset_index(drop=True)
    
    if len(group) < 2:
        continue
    
    # 第二轮：再算一次，去除残留漂移
    lat_arr = group['lat'].values
    lon_arr = group['lon'].values
    dist2 = np.zeros(len(group))
    dist2[1:] = haversine_np(lat_arr[:-1], lon_arr[:-1], lat_arr[1:], lon_arr[1:])
    mask2 = dist2 <= 2000  # 第二轮更严格：2km
    group = group[mask2].reset_index(drop=True)
    
    if len(group) >= 2:
        cleaned.append(group)
    
    count += 1
    if count % 1000 == 0:
        print(f"  已处理 {count} 辆车...")

result = pd.concat(cleaned, ignore_index=True)
print(f"✅ 漂移过滤完成: {len(result)} 条, {result['vehicle_id'].nunique()} 辆车")

# ============ 4. 高斯平滑 ============
print("正在平滑轨迹...")
sigma = 3  # 加大平滑力度

smoothed = []
for vid, group in result.groupby('vehicle_id'):
    group = group.copy()
    if len(group) >= 5:
        group['lat_smooth'] = gaussian_filter1d(group['lat'].values, sigma=sigma)
        group['lon_smooth'] = gaussian_filter1d(group['lon'].values, sigma=sigma)
    else:
        group['lat_smooth'] = group['lat']
        group['lon_smooth'] = group['lon']
    smoothed.append(group)

result = pd.concat(smoothed, ignore_index=True)

# ============ 5. 降采样（每5个点取1个）============
sampled = result.groupby('vehicle_id').apply(
    lambda g: g.iloc[::5]
).reset_index(drop=True)
print(f"降采样后: {len(sampled)} 条")

# ============ 6. 导出GeoJSON ============
print("正在导出GeoJSON...")
features = []
for vid, group in sampled.groupby('vehicle_id'):
    if len(group) < 2:
        continue
    lons = group['lon_smooth'].values
    lats = group['lat_smooth'].values
    coords = np.column_stack([lons, lats]).tolist()
    features.append({
        "type": "Feature",
        "properties": {"vehicle_id": vid},
        "geometry": {"type": "LineString", "coordinates": coords}
    })

geojson = {"type": "FeatureCollection", "features": features}
with open('haikou_clean_trajectory.geojson', 'w') as f:
    json.dump(geojson, f)

print(f"📁 haikou_clean_trajectory.geojson ({len(features)} 条轨迹线)")