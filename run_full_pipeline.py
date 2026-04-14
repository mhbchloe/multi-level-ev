"""
完整端到端流程：从数据加载到模型训练
"""
import os
import sys
import time
import pandas as pd
import numpy as np
import glob
from datetime import datetime
from tqdm import tqdm
import gc
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🚗 电动车驾驶行为聚类 - 完整端到端流程")
print("="*70)
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

start_time = time.time()

# ==================== 配置参数 ====================
DATA_DIR = './'
TEST_MODE = True  # 改为False处理全部数据
SOC_THRESHOLD = 3.0
MIN_EVENT_LENGTH = 5
SAMPLE_RATIO = 0.1 if TEST_MODE else None

print("📋 配置参数:")
print(f"   数据目录: {DATA_DIR}")
print(f"   模式: {'测试模式 (10%数据)' if TEST_MODE else '生产模式 (全部数据)'}")
print(f"   SOC阈值: {SOC_THRESHOLD}%")
print(f"   最小事件长度: {MIN_EVENT_LENGTH}")

# 创建结果目录
os.makedirs('./results/features', exist_ok=True)
os.makedirs('./results/events', exist_ok=True)

# ==================== 步骤1: 数据加载 ====================
print("\n" + "="*70)
print("📥 步骤1: 数据加载")
print("="*70)

csv_files = sorted(glob.glob(os.path.join(DATA_DIR, '*_processed.csv')))
print(f"找到 {len(csv_files)} 个CSV文件")

if len(csv_files) == 0:
    print("❌ 错误: 未找到数据文件！")
    print("请确保当前目录有 *_processed.csv 文件")
    sys.exit(1)

# 必需列
essential_cols = [
    'vehicle_id', 'datetime', 'time', 'soc', 'v', 'i', 'power',
    'is_charging', 'is_discharging', 'spd', 'acc',
    'kinematic_state', 'is_moving', 'driving_mode',
    'lat', 'lon', 'distance_km', 'heading_change',
    'is_regenerative_braking', 'energy_consumption',
    'efficiency_wh_per_km', 'time_diff', 'soc_change',
    'soc_rate', 'acc_smooth', 'spd_change_rate'
]

# 数据类型优化
dtype_dict = {
    'vehicle_id': 'category',
    'kinematic_state': 'category',
    'driving_mode': 'category',
    'spd': 'float32', 'v': 'float32', 'i': 'float32',
    'soc': 'float32', 'lat': 'float32', 'lon': 'float32',
    'acc': 'float32', 'distance_km': 'float32', 'power': 'float32',
    'is_moving': 'uint8', 'is_charging': 'uint8', 'is_discharging': 'uint8'
}

# 分块读取
all_chunks = []
for file in csv_files:
    print(f"\n📖 读取: {os.path.basename(file)}")
    
    # 检查文件列
    sample = pd.read_csv(file, nrows=1)
    available_cols = [col for col in essential_cols if col in sample.columns]
    
    print(f"   可用列数: {len(available_cols)}/{len(essential_cols)}")
    
    try:
        reader = pd.read_csv(
            file,
            chunksize=100000,
            usecols=available_cols,
            dtype={k: v for k, v in dtype_dict.items() if k in available_cols},
            low_memory=False
        )
        
        for chunk in reader:
            if SAMPLE_RATIO is not None:
                chunk = chunk.sample(frac=SAMPLE_RATIO, random_state=42)
            all_chunks.append(chunk)
        
        gc.collect()
        
    except Exception as e:
        print(f"   ⚠️  读取失败: {e}")
        continue

print("\n🔗 合并数据...")
df = pd.concat(all_chunks, ignore_index=True)
del all_chunks
gc.collect()

print(f"✅ 加载完成: {len(df):,} 行 x {len(df.columns)} 列")
mem_mb = df.memory_usage(deep=True).sum() / (1024**2)
print(f"   内存占用: {mem_mb:.2f} MB")

# ==================== 步骤2: 事件切分 ====================
print("\n" + "="*70)
print("📌 步骤2: 事件切分")
print("="*70)

df['datetime'] = pd.to_datetime(df['datetime'])
events = []

vehicle_ids = df['vehicle_id'].unique()
print(f"处理 {len(vehicle_ids)} 辆车的数据...")

for vehicle_id in tqdm(vehicle_ids, desc="切分事件"):
    vehicle_data = df[df['vehicle_id'] == vehicle_id].sort_values('datetime').reset_index(drop=True)
    
    if len(vehicle_data) < MIN_EVENT_LENGTH:
        continue
    
    start_idx = 0
    start_soc = vehicle_data.iloc[0]['soc']
    start_charging = vehicle_data.iloc[0].get('is_charging', 0)
    
    for i in range(1, len(vehicle_data)):
        current_soc = vehicle_data.iloc[i]['soc']
        current_charging = vehicle_data.iloc[i].get('is_charging', 0)
        
        # 切分条件
        soc_dropped = start_soc - current_soc >= SOC_THRESHOLD
        charging_changed = current_charging != start_charging
        
        if soc_dropped or charging_changed:
            event_data = vehicle_data.iloc[start_idx:i]
            
            if len(event_data) >= MIN_EVENT_LENGTH:
                # 排除充电事件
                if event_data['is_charging'].mean() < 0.5:
                    events.append({
                        'event_id': len(events),
                        'vehicle_id': vehicle_id,
                        'data': event_data,
                        'soc_drop': start_soc - current_soc,
                        'distance_km': event_data['distance_km'].sum() if 'distance_km' in event_data.columns else 0,
                        'duration_minutes': (event_data.iloc[-1]['datetime'] - event_data.iloc[0]['datetime']).total_seconds() / 60
                    })
            
            start_idx = i
            start_soc = current_soc
            start_charging = current_charging

print(f"✅ 提取事件: {len(events)} 个驾驶事件")

# 保存事件
with open('./results/events/events.pkl', 'wb') as f:
    pickle.dump(events, f)
print(f"💾 事件已保存至: ./results/events/events.pkl")

# ==================== 步骤3: 特征提取 ====================
print("\n" + "="*70)
print("📌 步骤3: 特征提取")
print("="*70)

energy_features_list = []
driving_features_list = []
event_ids = []
vehicle_ids = []

for event in tqdm(events, desc="提取特征"):
    ed = event['data']
    
    # 电量特征
    energy_feat = {
        'soc_drop_total': ed['soc'].iloc[0] - ed['soc'].iloc[-1],
        'soc_mean': ed['soc'].mean(),
        'soc_std': ed['soc'].std(),
        'voltage_mean': ed['v'].mean(),
        'voltage_std': ed['v'].std(),
        'current_mean': ed['i'].mean(),
        'current_max': ed['i'].max(),
        'power_mean': ed['power'].mean() if 'power' in ed.columns else 0,
        'power_max': ed['power'].max() if 'power' in ed.columns else 0,
        'power_std': ed['power'].std() if 'power' in ed.columns else 0,
    }
    
    if 'energy_consumption' in ed.columns:
        energy_feat['energy_consumption_total'] = ed['energy_consumption'].sum()
    else:
        energy_feat['energy_consumption_total'] = 0
    
    if 'efficiency_wh_per_km' in ed.columns:
        eff_clean = ed['efficiency_wh_per_km'].replace([np.inf, -np.inf], np.nan).dropna()
        energy_feat['efficiency_mean'] = eff_clean.mean() if len(eff_clean) > 0 else 0
    else:
        energy_feat['efficiency_mean'] = 0
    
    if 'is_charging' in ed.columns:
        energy_feat['charging_ratio'] = (ed['is_charging'] == 1).sum() / len(ed)
    else:
        energy_feat['charging_ratio'] = 0
    
    if 'is_regenerative_braking' in ed.columns:
        energy_feat['regen_braking_ratio'] = (ed['is_regenerative_braking'] == 1).sum() / len(ed)
    else:
        energy_feat['regen_braking_ratio'] = 0
    
    # 驾驶特征
    driving_feat = {
        'speed_mean': ed['spd'].mean(),
        'speed_max': ed['spd'].max(),
        'speed_std': ed['spd'].std(),
        'speed_median': ed['spd'].median(),
        'speed_cv': ed['spd'].std() / (ed['spd'].mean() + 1e-6),
        'low_speed_ratio': ((ed['spd'] > 0) & (ed['spd'] <= 40)).sum() / len(ed),
        'medium_speed_ratio': ((ed['spd'] > 40) & (ed['spd'] <= 80)).sum() / len(ed),
        'high_speed_ratio': (ed['spd'] > 80).sum() / len(ed),
    }
    
    if 'acc' in ed.columns:
        driving_feat['acc_mean'] = ed['acc'].mean()
        driving_feat['acc_std'] = ed['acc'].std()
        driving_feat['acc_max'] = ed['acc'].max()
        driving_feat['acc_min'] = ed['acc'].min()
        driving_feat['harsh_accel'] = (ed['acc'] > 2).sum()
        driving_feat['harsh_decel'] = (ed['acc'] < -2).sum()
    else:
        for key in ['acc_mean', 'acc_std', 'acc_max', 'acc_min', 'harsh_accel', 'harsh_decel']:
            driving_feat[key] = 0
    
    if 'is_moving' in ed.columns:
        driving_feat['moving_ratio'] = (ed['is_moving'] == 1).sum() / len(ed)
    else:
        driving_feat['moving_ratio'] = (ed['spd'] > 0).sum() / len(ed)
    
    if 'kinematic_state' in ed.columns:
        driving_feat['idle_ratio'] = (ed['kinematic_state'] == '静止').sum() / len(ed)
    else:
        driving_feat['idle_ratio'] = (ed['spd'] == 0).sum() / len(ed)
    
    driving_feat['distance_total'] = event['distance_km']
    driving_feat['duration_minutes'] = event['duration_minutes']
    
    if 'heading_change' in ed.columns:
        heading_clean = ed['heading_change'].replace([np.inf, -np.inf], np.nan).dropna()
        driving_feat['heading_change_mean'] = heading_clean.abs().mean() if len(heading_clean) > 0 else 0
        driving_feat['sharp_turn_count'] = (np.abs(heading_clean) > 45).sum() if len(heading_clean) > 0 else 0
    else:
        driving_feat['heading_change_mean'] = 0
        driving_feat['sharp_turn_count'] = 0
    
    if len(ed) > 1:
        driving_feat['stop_count'] = ((ed['spd'] == 0).astype(int).diff() == 1).sum()
    else:
        driving_feat['stop_count'] = 0
    
    energy_features_list.append(energy_feat)
    driving_features_list.append(driving_feat)
    event_ids.append(event['event_id'])
    vehicle_ids.append(event['vehicle_id'])

# 转换为DataFrame
energy_df = pd.DataFrame(energy_features_list)
driving_df = pd.DataFrame(driving_features_list)

energy_df.insert(0, 'event_id', event_ids)
energy_df.insert(1, 'vehicle_id', vehicle_ids)
driving_df.insert(0, 'event_id', event_ids)
driving_df.insert(1, 'vehicle_id', vehicle_ids)

# 合并特征
combined_df = pd.concat([energy_df, driving_df.drop(['event_id', 'vehicle_id'], axis=1)], axis=1)

# 处理异常值
energy_df = energy_df.replace([np.inf, -np.inf], np.nan).fillna(0)
driving_df = driving_df.replace([np.inf, -np.inf], np.nan).fillna(0)
combined_df = combined_df.replace([np.inf, -np.inf], np.nan).fillna(0)

print(f"✅ 特征提取完成")
print(f"   电量特征: {energy_df.shape}")
print(f"   驾驶特征: {driving_df.shape}")
print(f"   合并特征: {combined_df.shape}")

# 保存特征
energy_df.to_csv('./results/features/energy_features.csv', index=False)
driving_df.to_csv('./results/features/driving_features.csv', index=False)
combined_df.to_csv('./results/features/combined_features.csv', index=False)

print(f"💾 特征已保存至: ./results/features/")

# ==================== 步骤4: 快速聚类测试 ====================
print("\n" + "="*70)
print("📌 步骤4: 快速聚类测试 (K-Means)")
print("="*70)

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

X = combined_df.drop(['event_id', 'vehicle_id'], axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

silhouette = silhouette_score(X_scaled, labels)

print(f"✅ K-Means聚类完成")
print(f"   Silhouette Score: {silhouette:.3f}")
print(f"   聚类分布: {np.bincount(labels)}")

# 保存快速测试结果
quick_results = combined_df[['event_id', 'vehicle_id']].copy()
quick_results['cluster'] = labels
quick_results.to_csv('./results/quick_clustering_results.csv', index=False)

# ==================== 总结 ====================
total_time = time.time() - start_time

print("\n" + "="*70)
print("✅ 数据预处理完成！")
print("="*70)
print(f"⏱️  总耗时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
print(f"📊 处理数据: {len(df):,} 行")
print(f"📦 提取事件: {len(events)} 个")
print(f"🔧 提取特征: {combined_df.shape[1]-2} 个")
print(f"📁 结果保存: ./results/")
print("\n💡 下一步: 运行 'python train_all_models.py' 训练所有模型")
print("="*70)