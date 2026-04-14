"""
超轻量空间分析
用网格聚类替代DBSCAN（内存占用极低）
"""

import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from tqdm import tqdm
from collections import defaultdict

print("="*70)
print("🗺️ Lightweight Spatial Analysis")
print("="*70)

# Step 1: 采样读取（降到2%）
print("\n📂 Loading charging GPS (2% sample)...")

sample_rate = 0.02  # 降到2%，约140万点
df_charging = pd.read_csv(
    './results/charging_gps_all.csv',
    skiprows=lambda i: i > 0 and np.random.rand() > sample_rate,
    usecols=['vehicle_id', 'soc', 'lat', 'lon']
)

print(f"✅ Sampled {len(df_charging):,} charging points")

# Step 2: 清洗
print("\n🧹 Cleaning GPS data...")

df_charging = df_charging[
    (df_charging['lat'].notna()) &
    (df_charging['lon'].notna()) &
    (df_charging['lat'] > 0) &
    (df_charging['lon'] > 0)
]

print(f"✅ Valid GPS: {len(df_charging):,}")

# Step 3: 网格聚类（替代DBSCAN，超快！）
print("\n🔍 Grid-based clustering...")

# 网格大小：约100米 = 0.001度
grid_size = 0.001

df_charging['grid_lat'] = (df_charging['lat'] / grid_size).round().astype(int)
df_charging['grid_lon'] = (df_charging['lon'] / grid_size).round().astype(int)
df_charging['grid_id'] = df_charging['grid_lat'].astype(str) + '_' + df_charging['grid_lon'].astype(str)

# 统计每个网格
print("   Computing grid statistics...")

grid_stats = df_charging.groupby('grid_id').agg({
    'lat': 'mean',
    'lon': 'mean',
    'vehicle_id': 'nunique',
    'soc': ['mean', 'std', 'count']
}).reset_index()

grid_stats.columns = ['grid_id', 'lat', 'lon', 'vehicle_count', 'avg_soc', 'std_soc', 'charging_count']

# 只保留充电次数>=10的网格（过滤噪声）
grid_stats = grid_stats[grid_stats['charging_count'] >= 10].sort_values('charging_count', ascending=False)

print(f"✅ Identified {len(grid_stats)} charging locations")
print(f"   (Grids with ≥10 charging events)")

# Step 4: 充电站统计
print(f"\n📊 Top 15 busiest charging locations:")

for i, row in grid_stats.head(15).iterrows():
    print(f"   #{i+1}. Grid {row['grid_id'][:10]}...: "
          f"{row['charging_count']:,} charges, "
          f"{row['vehicle_count']:,} vehicles, "
          f"SOC {row['avg_soc']:.1f}%")

grid_stats.to_csv('./results/charging_locations.csv', index=False, encoding='utf-8-sig')

# Step 5: 创建地图
print("\n🗺️ Creating map...")

center_lat = df_charging['lat'].mean()
center_lon = df_charging['lon'].mean()

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=11,
    tiles='OpenStreetMap'
)

# 图层1: 充电位置标记（只显示前100个）
print("   Adding charging location markers...")

location_layer = folium.FeatureGroup(name='⚡ Charging Locations (Top 100)', show=True)

for _, loc in grid_stats.head(100).iterrows():
    radius = min(loc['charging_count'] / 50, 25)
    
    if loc['charging_count'] > 1000:
        color = 'red'
    elif loc['charging_count'] > 500:
        color = 'orange'
    elif loc['charging_count'] > 100:
        color = 'yellow'
    else:
        color = 'green'
    
    folium.CircleMarker(
        location=[loc['lat'], loc['lon']],
        radius=radius,
        popup=f"""
            <b>Charging Location</b><br>
            Count: {loc['charging_count']:,}<br>
            Vehicles: {loc['vehicle_count']:,}<br>
            Avg SOC: {loc['avg_soc']:.1f}%
        """,
        color=color,
        fill=True,
        fillColor=color,
        fillOpacity=0.7
    ).add_to(location_layer)

location_layer.add_to(m)

# 图层2: 热力图
print("   Adding heatmap...")

heat_layer = folium.FeatureGroup(name='🔥 Heatmap', show=True)

# 采样用于热力图
if len(df_charging) > 20000:
    df_heat = df_charging.sample(20000)
else:
    df_heat = df_charging

heat_data = [[row['lat'], row['lon']] for _, row in df_heat.iterrows()]

HeatMap(
    heat_data,
    min_opacity=0.3,
    radius=12,
    blur=15
).add_to(heat_layer)

heat_layer.add_to(m)

folium.LayerControl().add_to(m)

# 保存
map_file = './results/charging_map.html'
m.save(map_file)

print(f"✅ Saved: {map_file}")

# Step 6: 空间统计
print(f"\n📊 Spatial Statistics:")

total_charging = df_charging['charging_count'].sum() if 'charging_count' in df_charging.columns else len(df_charging)

print(f"   Total charging locations: {len(grid_stats):,}")
print(f"   Total charging events (sampled): {len(df_charging):,}")
print(f"   Unique vehicles: {df_charging['vehicle_id'].nunique():,}")
print(f"   Avg SOC at charging: {df_charging['soc'].mean():.1f}%")

lat_range = grid_stats['lat'].max() - grid_stats['lat'].min()
lon_range = grid_stats['lon'].max() - grid_stats['lon'].min()
area_km2 = lat_range * lon_range * 111 * 111

if area_km2 > 0:
    density = len(grid_stats) / area_km2
    print(f"   Coverage area: ~{area_km2:.1f} km²")
    print(f"   Location density: {density:.2f} locations/km²")

print(f"\n{'='*70}")
print(f"✅ Analysis Complete!")
print(f"{'='*70}")
print(f"\n📁 Generated files:")
print(f"   1. charging_locations.csv")
print(f"   2. {map_file}")
print(f"\n💡 Open map: {map_file}")
print(f"{'='*70}")