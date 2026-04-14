"""
Step 3: 充电空间分析
生成交互式地图
"""

import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap, MarkerCluster
from sklearn.cluster import DBSCAN
from tqdm import tqdm

print("="*70)
print("🗺️ Step 3: Spatial Charging Analysis")
print("="*70)

# 加载数据
print("\n📂 Loading charging GPS with events...")
df_charging = pd.read_csv('./results/charging_gps_with_events.csv')

print(f"✅ Loaded {len(df_charging):,} charging GPS points")
print(f"   Vehicles: {df_charging['vehicle_id'].nunique():,}")
print(f"   With cluster info: {(df_charging['cluster'] != -1).sum():,}")

# 移除无效GPS坐标
df_charging = df_charging[
    (df_charging['lat'].notna()) &
    (df_charging['lon'].notna()) &
    (df_charging['lat'] > 0) &
    (df_charging['lon'] > 0)
]

print(f"   Valid GPS: {len(df_charging):,}")

# 识别充电站（聚类）
print("\n🔍 Identifying charging stations...")

coords = np.radians(df_charging[['lat', 'lon']].values)
eps_km = 0.1  # 100米内认为是同一个充电站
eps_radians = eps_km / 6371

clustering = DBSCAN(eps=eps_radians, min_samples=5, metric='haversine')
df_charging['station_id'] = clustering.fit_predict(coords)

# 统计充电站
stations = []
for station_id in df_charging['station_id'].unique():
    if station_id == -1:
        continue
    
    station_data = df_charging[df_charging['station_id'] == station_id]
    
    station = {
        'station_id': station_id,
        'lat': station_data['lat'].mean(),
        'lon': station_data['lon'].mean(),
        'charging_count': len(station_data),
        'vehicle_count': station_data['vehicle_id'].nunique(),
        'avg_soc_start': station_data['soc'].mean(),
        'cluster_0': (station_data['cluster'] == 0).sum(),
        'cluster_1': (station_data['cluster'] == 1).sum(),
        'cluster_2': (station_data['cluster'] == 2).sum(),
    }
    stations.append(station)

df_stations = pd.DataFrame(stations).sort_values('charging_count', ascending=False)

print(f"✅ Identified {len(df_stations)} charging stations")
print(f"\n   Top 5 stations:")
for _, s in df_stations.head(5).iterrows():
    print(f"      Station {s['station_id']:3d}: {s['charging_count']:,} charges, {s['vehicle_count']:,} vehicles")

df_stations.to_csv('./results/charging_stations.csv', index=False, encoding='utf-8-sig')

# 创建交互式地图
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

colors_map = {0: 'blue', 1: 'green', 2: 'orange'}

for _, station in df_stations.iterrows():
    radius = min(station['charging_count'] / 10, 30)
    
    # 主导簇
    dominant_cluster = max(
        [(0, station['cluster_0']), 
         (1, station['cluster_1']), 
         (2, station['cluster_2'])],
        key=lambda x: x[1]
    )[0]
    
    folium.CircleMarker(
        location=[station['lat'], station['lon']],
        radius=radius,
        popup=folium.Popup(f"""
            <b>Station {station['station_id']}</b><br>
            <hr>
            <b>Usage:</b> {station['charging_count']} times<br>
            <b>Vehicles:</b> {station['vehicle_count']}<br>
            <b>Avg SOC:</b> {station['avg_soc_start']:.1f}%<br>
            <hr>
            <b>User clusters:</b><br>
            C0: {station['cluster_0']}<br>
            C1: {station['cluster_1']}<br>
            C2: {station['cluster_2']}<br>
            <b>Dominant:</b> C{dominant_cluster}
        """, max_width=300),
        color=colors_map[dominant_cluster],
        fill=True,
        fillColor=colors_map[dominant_cluster],
        fillOpacity=0.7,
        weight=2
    ).add_to(station_layer)

station_layer.add_to(m)

# 图层2: 充电热力图
heat_layer = folium.FeatureGroup(name='🔥 Charging Heatmap', show=True)

# 采样数据（如果点太多）
if len(df_charging) > 50000:
    df_sample = df_charging.sample(50000)
else:
    df_sample = df_charging

heat_data = [[row['lat'], row['lon']] for _, row in df_sample.iterrows()]

HeatMap(
    heat_data,
    min_opacity=0.3,
    radius=15,
    blur=20,
    gradient={0.4: 'blue', 0.65: 'lime', 0.8: 'orange', 1.0: 'red'}
).add_to(heat_layer)

heat_layer.add_to(m)

# 图层3: 按簇分布
for cluster_id in range(3):
    cluster_layer = folium.FeatureGroup(name=f'🚗 Cluster {cluster_id}', show=False)
    
    cluster_data = df_charging[df_charging['cluster'] == cluster_id]
    
    if len(cluster_data) > 10000:
        cluster_data = cluster_data.sample(10000)
    
    for _, row in cluster_data.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=3,
            color=colors_map[cluster_id],
            fill=True,
            fillOpacity=0.6,
            popup=f"C{cluster_id}: SOC {row['soc']:.0f}%"
        ).add_to(cluster_layer)
    
    cluster_layer.add_to(m)

# 图层控制
folium.LayerControl(collapsed=False).add_to(m)

# 保存地图
map_file = './results/charging_spatial_map.html'
m.save(map_file)

print(f"✅ Saved: {map_file}")

# 空间统计
print(f"\n📊 Spatial Statistics:")

lat_range = df_stations['lat'].max() - df_stations['lat'].min()
lon_range = df_stations['lon'].max() - df_stations['lon'].min()
area_km2 = lat_range * lon_range * 111 * 111

station_density = len(df_stations) / area_km2 if area_km2 > 0 else 0

print(f"   Coverage area: ~{area_km2:.1f} km²")
print(f"   Station density: {station_density:.2f} stations/km²")
print(f"   Avg distance: ~{np.sqrt(1/station_density) if station_density > 0 else 0:.2f} km")

print(f"\n   Charging by cluster:")
for cluster_id in range(3):
    count = (df_charging['cluster'] == cluster_id).sum()
    pct = count / len(df_charging) * 100 if len(df_charging) > 0 else 0
    print(f"      Cluster {cluster_id}: {count:,} ({pct:.1f}%)")

print(f"\n{'='*70}")
print(f"✅ Spatial Analysis Complete!")
print(f"{'='*70}")
print(f"\n📁 Generated files:")
print(f"   1. charging_stations.csv - 充电站统计")
print(f"   2. charging_spatial_map.html - 交互式地图")
print(f"\n💡 Open {map_file} in browser to explore!")
print(f"{'='*70}")