import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d

# ============ 1. 读取数据 ============
df = pd.read_csv('data_20250701_processed.csv')
print(f"原始数据: {len(df)} 条")

# ============ 2. 提取所需字段 ============
traj = df[['vehicle_id', 'lat', 'lon', 'datetime']].copy()

# ============ 3. 删除经纬度缺失和无效值 ============
traj = traj.dropna(subset=['lat', 'lon'])
traj = traj[(traj['lat'] > 0) & (traj['lon'] > 0)]
print(f"删除缺失/零值后: {len(traj)} 条")

# ============ 4. 探查坐标真实范围，判断缩放系数 ============
print(f"\n🔍 当前坐标范围:")
print(f"  lat: {traj['lat'].min()} ~ {traj['lat'].max()}")
print(f"  lon: {traj['lon'].min()} ~ {traj['lon'].max()}")
print(f"  lat 中位数: {traj['lat'].median()}")
print(f"  lon 中位数: {traj['lon'].median()}")

# 查看典型值来判断需要除以多少
print(f"\n📋 随机抽样10条坐标:")
print(traj[['lat', 'lon']].sample(10))

