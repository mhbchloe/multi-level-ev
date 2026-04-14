import pandas as pd

# ============ 读取数据 ============
df = pd.read_csv('data_20250701_processed.csv')

# ============ 基本信息 ============
print("=" * 60)
print(f"数据形状: {df.shape[0]} 行 × {df.shape[1]} 列")
print("=" * 60)

# 所有列名
print("\n📋 列名:")
print(list(df.columns))

# 数据类型
print("\n📊 各列数据类型:")
print(df.dtypes)

# 前10行预览
print("\n👀 前10行数据:")
print(df.head(10))

# 基本统计
print("\n📈 数值列统计摘要:")
print(df.describe())

# 缺失值
print("\n❓ 各列缺失值数量:")
print(df.isnull().sum())

# 如果有 vehicle_id 列，看看有多少辆车
if 'vehicle_id' in df.columns:
    print(f"\n🚗 车辆数量: {df['vehicle_id'].nunique()}")
    print(f"车辆ID列表: {df['vehicle_id'].unique()[:20]}")  # 最多显示20个

# 如果有 lat/lon 列，看看坐标范围
for lat_col in ['lat', 'latitude', 'LAT']:
    if lat_col in df.columns:
        print(f"\n📍 纬度范围: {df[lat_col].min()} ~ {df[lat_col].max()}")
        break
for lon_col in ['lon', 'longitude', 'LON']:
    if lon_col in df.columns:
        print(f"📍 经度范围: {df[lon_col].min()} ~ {df[lon_col].max()}")
        break
    