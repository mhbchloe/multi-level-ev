"""
充电站识别 - 只输出CSV（用于QGIS）
"""

import pandas as pd
import numpy as np
from tqdm import tqdm

print("="*70)
print("🔋 Charging Station Identification (CSV Only)")
print("="*70)

# Step 1: 读取充电GPS数据
print("\n📂 Loading charging GPS data...")

df = pd.read_csv(
    './results/charging_gps_all.csv',
    skiprows=lambda i: i > 0 and np.random.rand() > 0.05,  # 采样5%
    usecols=['vehicle_id', 'time', 'lat', 'lon', 'soc', 'is_charging']
)

print(f"✅ Loaded {len(df):,} GPS points")

# 清洗
df = df[
    (df['lat'].notna()) & (df['lon'].notna()) &
    (df['lat'] > 0) & (df['lon'] > 0) &
    (df['is_charging'] == 1)
]

df = df.sort_values(['vehicle_id', 'time'])

print(f"   Valid charging points: {len(df):,}")

# Step 2: 识别充电事件
print("\n🔍 Identifying charging events...")

charging_events = []

for vehicle_id, group in tqdm(df.groupby('vehicle_id'), desc="Vehicles"):
    if len(group) < 5:
        continue
    
    group = group.copy()
    group['lat_diff'] = group['lat'].diff().abs()
    group['lon_diff'] = group['lon'].diff().abs()
    group['location_change'] = np.sqrt(group['lat_diff']**2 + group['lon_diff']**2)
    
    current_event = []
    
    for idx, row in group.iterrows():
        if len(current_event) == 0 or row['location_change'] < 0.001:
            current_event.append({
                'lat': row['lat'],
                'lon': row['lon'],
                'soc': row['soc']
            })
        else:
            if len(current_event) >= 5:
                event_lats = [p['lat'] for p in current_event]
                event_lons = [p['lon'] for p in current_event]
                event_socs = [p['soc'] for p in current_event]
                
                charging_events.append({
                    'vehicle_id': vehicle_id,
                    'lat': np.mean(event_lats),
                    'lon': np.mean(event_lons),
                    'soc_start': event_socs[0],
                    'soc_end': event_socs[-1],
                    'duration_points': len(current_event),
                    'lat_std': np.std(event_lats),
                    'lon_std': np.std(event_lons),
                })
            
            current_event = [{
                'lat': row['lat'],
                'lon': row['lon'],
                'soc': row['soc']
            }]
    
    if len(current_event) >= 5:
        event_lats = [p['lat'] for p in current_event]
        event_lons = [p['lon'] for p in current_event]
        event_socs = [p['soc'] for p in current_event]
        
        charging_events.append({
            'vehicle_id': vehicle_id,
            'lat': np.mean(event_lats),
            'lon': np.mean(event_lons),
            'soc_start': event_socs[0],
            'soc_end': event_socs[-1],
            'duration_points': len(current_event),
            'lat_std': np.std(event_lats),
            'lon_std': np.std(event_lons),
        })

df_events = pd.DataFrame(charging_events)

print(f"✅ Identified {len(df_events):,} charging events")

# Step 3: 过滤稳定充电事件
print("\n🔍 Filtering stable events...")

df_events = df_events[
    (df_events['lat_std'] < 0.0005) &
    (df_events['lon_std'] < 0.0005) &
    (df_events['duration_points'] >= 10)
]

print(f"✅ Stable events: {len(df_events):,}")

# Step 4: 聚类识别充电站
print("\n🗺️ Clustering stations...")

grid_size = 0.0005

df_events['grid_lat'] = (df_events['lat'] / grid_size).round().astype(int)
df_events['grid_lon'] = (df_events['lon'] / grid_size).round().astype(int)
df_events['station_id'] = df_events['grid_lat'].astype(str) + '_' + df_events['grid_lon'].astype(str)

# 统计充电站
stations = df_events.groupby('station_id').agg({
    'lat': 'mean',
    'lon': 'mean',
    'vehicle_id': 'nunique',
    'soc_start': 'mean',
    'soc_end': 'mean',
    'duration_points': ['sum', 'mean'],
    'station_id': 'count'
}).reset_index()

stations.columns = ['station_id', 'lat', 'lon', 'vehicle_count', 
                    'avg_soc_start', 'avg_soc_end', 'total_duration', 
                    'avg_duration', 'charging_count']

stations = stations[stations['charging_count'] >= 3].sort_values('charging_count', ascending=False)

# 添加有用的列
stations['soc_gain'] = stations['avg_soc_end'] - stations['avg_soc_start']
stations['usage_level'] = pd.cut(
    stations['charging_count'], 
    bins=[0, 5, 10, 20, float('inf')],
    labels=['Low', 'Medium', 'High', 'Very High']
)

print(f"✅ Identified {len(stations)} charging stations (≥3 uses)")

# Step 5: 输出CSV
print("\n💾 Saving CSV files...")

# 主文件：充电站统计
output_file = './results/charging_stations.csv'
stations.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"   ✅ {output_file}")

# 额外：充电事件明细（用于验证）
events_file = './results/charging_events.csv'
df_events.to_csv(events_file, index=False, encoding='utf-8-sig')
print(f"   ✅ {events_file}")

# Step 6: 统计报告
print(f"\n{'='*70}")
print(f"📊 Summary Statistics")
print(f"{'='*70}")

print(f"\n总体统计：")
print(f"   充电站数量: {len(stations)}")
print(f"   总充电次数: {stations['charging_count'].sum():,}")
print(f"   涉及车辆: {df_events['vehicle_id'].nunique():,}")
print(f"   平均SOC增益: {stations['soc_gain'].mean():.1f}%")

print(f"\n充电站使用等级分布：")
for level in ['Low', 'Medium', 'High', 'Very High']:
    count = (stations['usage_level'] == level).sum()
    print(f"   {level}: {count} 个站点")

print(f"\nTop 10 最繁忙充电站：")
for i, row in stations.head(10).iterrows():
    print(f"   {i+1}. ({row['lat']:.4f}, {row['lon']:.4f})")
    print(f"      充电{row['charging_count']}次, {row['vehicle_count']}辆车, SOC +{row['soc_gain']:.1f}%")

print(f"\n{'='*70}")
print(f"✅ Complete!")
print(f"{'='*70}")

print(f"\n📁 Generated files:")
print(f"   1. {output_file}")
print(f"      → 充电站位置和统计（用于QGIS）")
print(f"   2. {events_file}")
print(f"      → 充电事件明细（可选）")

print(f"\n💡 QGIS导入步骤：")
print(f"   1. Layer → Add Layer → Add Delimited Text Layer")
print(f"   2. 选择 charging_stations.csv")
print(f"   3. X field = lon, Y field = lat")
print(f"   4. Geometry CRS = EPSG:4326 (WGS 84)")
print(f"   5. 按 'charging_count' 或 'usage_level' 设置样式")

print(f"{'='*70}")