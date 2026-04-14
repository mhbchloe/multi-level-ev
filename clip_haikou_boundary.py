import geopandas as gpd
import json
import requests
import numpy as np

# ============ 1. 获取海口市行政边界 ============
# 使用 GeoJSON.cn 提供的中国行政区划数据（海口市编码：460100）
print("正在下载海口市边界...")
url = "https://geo.datav.aliyun.com/areas_v3/bound/460100_full.json"
resp = requests.get(url)
haikou_boundary = gpd.GeoDataFrame.from_features(resp.json()["features"], crs="EPSG:4326")

# 合并为一个完整的海口市边界
haikou_boundary = haikou_boundary.dissolve()
print("✅ 海口市边界下载完成")

# 保存边界文件（可以在QGIS里单独查看）
haikou_boundary.to_file("haikou_boundary.geojson", driver="GeoJSON")
print("📁 haikou_boundary.geojson 已导出")

# ============ 2. 读取轨迹数据 ============
print("正在读取轨迹数据...")
tracks = gpd.read_file("haikou_clean_trajectory.geojson")
print(f"轨迹数量: {len(tracks)} 条")

# ============ 3. 用海口市边界裁剪轨迹 ============
print("正在裁剪...")
clipped = gpd.clip(tracks, haikou_boundary)
print(f"✅ 裁剪完成: {len(clipped)} 条轨迹")

# ============ 4. 导出 ============
clipped.to_file("haikou_city_trajectory.geojson", driver="GeoJSON")
print("📁 haikou_city_trajectory.geojson 已导出")