"""
跳过事件匹配，直接进行空间分析
充电站识别只需要GPS坐标 + 车辆ID
"""

import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap, MarkerCluster
from sklearn.cluster import DBSCAN
from tqdm import tqdm

print("="*70)
print("🗺️ Direct Spatial Analysis (Skip Event Matching)")
print("="*70)

# Step 1: 采样读取充电GPS（减少内存占用）
print("\n📂 Loading charging GPS (sampled)...")

# 读取10%数据（约700万行）
sample_rate = 0.1
df_charging = pd.read_csv(
    './results/charging_gps_all.csv',
    skiprows=lambda i: i > 0 and np.random.rand() > sample_rate,
    usecols=['vehicle_id', 'datetime', 'soc', 'lat', 'lon']
)

print(f"✅ Sampled {len(df_charging):,} charging points ({sample_rate*100:.0f}%)")

# Step 2: 清洗GPS数据
print("\n🧹 Cleaning GPS data...")

original_count = len(df_charging)

df_charging = df_charging[
    (df_charging['lat'].notna()) &
    (df_charging['lon'].notna()) &
    (df_charging['lat'] > 0) &
    (df_charging['lon'] > 0) &
    (df_charging['lat'] < 90) &
    (df_charging['lon'] < 180)
]

print(f"✅ Valid GPS: {len(df_charging):,}/{original_count:,} ({len(df_charging)/original_count*100:.1f}%)")

# Step 3: 聚类识别充电站
print("\n🔍 Clustering charging stations (DBSCAN)...")

coords = np.radians(df_charging[['lat', 'lon']].values)
eps_km = 0.1  # 100米
eps_radians = eps_km / 6371

clustering = DBSCAN(eps=eps_radians, min_samples=10, metric='haversine')
df_charging['station_id'] = clustering.fit_predict(coords)

n_stations = (df_charging['station_id'] != -1).sum()
print(f"✅ Identified {df_charging['station_id'].nunique() - 1} charging stations")
print(f"   Clustered points: {n_stations:,}/{len(df_charging):,} ({n_stations/len(df_charging)*100:.1f}%)")

# Step 4: 统计充电站
print("\n📊 Computing station statistics...")

stations = []

for station_id in tqdm(df_charging['station_id'].unique(), desc="Stations"):
    if station_id == -1:
        continue
    
    station_data = df_charging[df_charging['station_id'] == station_id]
    
    station = {
        'station_id': station_id,
        'lat': station_data['lat'].mean(),
        'lon': station_data['lon'].mean(),
        'charging_count': len(station_data),
        'vehicle_count': station_data['vehicle_id'].nunique(),
        'avg_soc': station_data['soc'].mean(),
        'std_soc': station_data['soc'].std(),
    }
    stations.append(station)

df_stations = pd.DataFrame(stations).sort_values('charging_count', ascending=False)

print(f"\n✅ Station statistics:")
print(f"   Total stations: {len(df_stations)}")
print(f"   Avg charging count: {df_stations['charging_count'].mean():.0f}")
print(f"   Avg vehicles per station: {df_stations['vehicle_count'].mean():.0f}")

print(f"\n   Top 10 busiest stations:")
for i, row in df_stations.head(10).iterrows():
    print(f"      #{i+1}. Station {row['station_id']:3d}: "
          f"{row['charging_count']:,} charges, {row['vehicle_count']:,} vehicles, "
          f"SOC {row['avg_soc']:.1f}%")

df_stations.to_csv('./results/charging_stations_direct.csv', index=False, encoding='utf-8-sig')

# Step 5: 创建交互式地图
print("\n🗺️ Creating interactive map...")

center_lat = df_charging['lat'].mean()
center_lon = df_charging['lon'].mean()

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=11,
    tiles='OpenStreetMap'
)

# 图层1: 充电站标记
station_layer = folium.FeatureGroup(name='⚡ Charging Stations', show=True)

for _, station in df_stations.iterrows():
    radius = min(station['charging_count'] / 20, 30)
    
    # 根据使用频率着色
    if station['charging_count'] > 1000:
        color = 'red'
    elif station['charging_count'] > 500:
        color = 'orange'
    else:
        color = 'green'
    
    folium.CircleMarker(
        location=[station['lat'], station['lon']],
        radius=radius,
        popup=folium.Popup(f"""
            <b>Station {station['station_id']}</b><br>
            <hr>
            <b>Charging count:</b> {station['charging_count']:,}<br>
            <b>Vehicles:</b> {station['vehicle_count']:,}<br>
            <b>Avg SOC:</b> {station['avg_soc']:.1f}%<br>
            <b>SOC std:</b> {station['std_soc']:.1f}%
        """, max_width=300),
        color=color,
        fill=True,
        fillColor=color,
        fillOpacity=0.7,
        weight=2
    ).add_to(station_layer)

station_layer.add_to(m)

# 图层2: 充电热力图
heat_layer = folium.FeatureGroup(name='🔥 Charging Heatmap', show=True)

# 采样用于热力图（最多5万点）
if len(df_charging) > 50000:
    df_heatmap = df_charging.sample(50000)
else:
    df_heatmap = df_charging

heat_data = [[row['lat'], row['lon']] for _, row in df_heatmap.iterrows()]

HeatMap(
    heat_data,
    min_opacity=0.3,
    radius=15,
    blur=20,
    gradient={0.4: 'blue', 0.65: 'lime', 0.8: 'orange', 1.0: 'red'}
).add_to(heat_layer)

heat_layer.add_to(m)

# 图层控制
folium.LayerControl(collapsed=False).add_to(m)

# 保存
map_file = './results/charging_map_direct.html'
m.save(map_file)

print(f"✅ Saved: {map_file}")

# Step 6: 空间统计
print(f"\n📊 Spatial Statistics:")

lat_range = df_stations['lat'].max() - df_stations['lat'].min()
lon_range = df_stations['lon'].max() - df_stations['lon'].min()
area_km2 = lat_range * lon_range * 111 * 111

station_density = len(df_stations) / area_km2 if area_km2 > 0 else 0

print(f"   Coverage area: ~{area_km2:.1f} km²")
print(f"   Station density: {station_density:.2f} stations/km²")
print(f"   Avg distance between stations: ~{np.sqrt(1/station_density) if station_density > 0 else 0:.2f} km")

print(f"\n{'='*70}")
print(f"✅ Spatial Analysis Complete!")
print(f"{'='*70}")
print(f"\n📁 Generated files:")
print(f"   1. charging_stations_direct.csv - 充电站统计")
print(f"   2. {map_file} - 交互式地图")
print(f"\n💡 Open the map in browser:")
print(f"   {map_file}")
print(f"{'='*70}")