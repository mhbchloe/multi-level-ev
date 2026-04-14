"""
车辆聚类 - 高级特征版本（正确的时间戳格式）
时间戳格式：YYYYMMDDHHmmss
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from scipy.stats import entropy

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("🚗 Vehicle Clustering with Advanced Features")
print("="*70)


# ==================== 加载并解析时间 ====================
print("\n📂 Loading event table...")

df_events = pd.read_csv('./results/event_table.csv')

print(f"✅ Loaded {len(df_events):,} events from {df_events['vehicle_id'].nunique():,} vehicles")

# 解析时间戳（格式：YYYYMMDDHHmmss）
print(f"\n🔍 Parsing timestamps (format: YYYYMMDDHHmmss)...")
print(f"   Example: {df_events['start_time'].iloc[0]} → ", end="")

df_events['start_datetime'] = pd.to_datetime(df_events['start_time'].astype(str), format='%Y%m%d%H%M%S')
df_events['end_datetime'] = pd.to_datetime(df_events['end_time'].astype(str), format='%Y%m%d%H%M%S')

print(f"{df_events['start_datetime'].iloc[0]}")

# 提取时间特征
df_events['hour'] = df_events['start_datetime'].dt.hour
df_events['day_of_week'] = df_events['start_datetime'].dt.dayofweek
df_events['is_weekend'] = df_events['day_of_week'].isin([5, 6]).astype(int)

print(f"   ✅ Time range: {df_events['start_datetime'].min()} to {df_events['start_datetime'].max()}")
print(f"   Hour range: {df_events['hour'].min()}-{df_events['hour'].max()}")
print(f"   Weekend events: {df_events['is_weekend'].sum():,} ({df_events['is_weekend'].mean()*100:.1f}%)")


# ==================== 生成车辆级高级特征 ====================
def generate_advanced_vehicle_features(df_events):
    """
    生成包含状态极值、转移、时间偏好的车辆特征
    """
    print("\n" + "="*70)
    print("📊 Generating Advanced Vehicle Features")
    print("="*70)
    
    vehicle_features = []
    vehicle_ids = df_events['vehicle_id'].unique()
    
    print(f"Processing {len(vehicle_ids):,} vehicles...")
    
    from tqdm import tqdm
    
    for vehicle_id in tqdm(vehicle_ids, desc="Extracting features"):
        vehicle_events = df_events[df_events['vehicle_id'] == vehicle_id].sort_values('start_datetime')
        
        if len(vehicle_events) < 2:
            continue
        
        feat = {'vehicle_id': vehicle_id}
        
        # ==================== A. 状态极值特征 ====================
        
        # SOC边界
        feat['soc_start_mean'] = vehicle_events['soc_start'].mean()
        feat['soc_start_std'] = vehicle_events['soc_start'].std()
        feat['soc_end_mean'] = vehicle_events['soc_end'].mean()
        feat['soc_end_std'] = vehicle_events['soc_end'].std()
        feat['soc_end_min'] = vehicle_events['soc_end'].min()  # 续航焦虑边界
        
        # 续航焦虑指标
        feat['range_anxiety_threshold'] = feat['soc_end_min']
        feat['comfort_soc_buffer'] = feat['soc_start_mean'] - feat['soc_end_mean']
        feat['soc_usage_range'] = vehicle_events['soc_start'].max() - feat['soc_end_min']
        
        # 低SOC事件
        feat['critical_soc_events'] = (vehicle_events['soc_end'] < 20).sum()
        feat['critical_soc_ratio'] = feat['critical_soc_events'] / len(vehicle_events)
        
        # 充电触发SOC
        feat['charging_trigger_soc'] = np.percentile(vehicle_events['soc_end'], 25)
        
        # ==================== B. 转移特征 ====================
        
        # 驾驶模式切换
        event_clusters = vehicle_events['cluster'].values
        mode_switches = np.sum(event_clusters[:-1] != event_clusters[1:])
        feat['mode_switch_count'] = mode_switches
        feat['mode_switch_rate'] = mode_switches / (len(vehicle_events) - 1)
        
        # 模式多样性
        mode_dist = vehicle_events['cluster'].value_counts(normalize=True)
        feat['mode_diversity'] = entropy(mode_dist)
        feat['mode_stability'] = -entropy(mode_dist)
        
        feat['dominant_mode'] = vehicle_events['cluster'].mode()[0]
        feat['dominant_mode_ratio'] = (vehicle_events['cluster'] == feat['dominant_mode']).mean()
        
        # 速度模式切换（激进 vs 节能）
        aggressive_mask = (vehicle_events['speed_mean'] > 50) | (vehicle_events['harsh_accel_count'] > 0)
        aggressive_switches = np.sum(aggressive_mask.values[:-1] != aggressive_mask.values[1:])
        feat['aggressive_switch_count'] = aggressive_switches
        feat['aggressive_switch_rate'] = aggressive_switches / (len(vehicle_events) - 1)
        
        # 切换方向不对称性
        if aggressive_switches > 0:
            eco_to_agg = np.sum((~aggressive_mask.values[:-1]) & (aggressive_mask.values[1:]))
            agg_to_eco = np.sum((aggressive_mask.values[:-1]) & (~aggressive_mask.values[1:]))
            feat['mode_switch_asymmetry'] = (eco_to_agg - agg_to_eco) / aggressive_switches
        else:
            feat['mode_switch_asymmetry'] = 0
        
        # ==================== C. 时间偏好特征 ====================
        
        hours = vehicle_events['hour'].values
        
        # 时段分布
        feat['night_driving_ratio'] = np.mean((hours >= 22) | (hours < 6))
        feat['morning_peak_ratio'] = np.mean((hours >= 7) & (hours < 9))
        feat['evening_peak_ratio'] = np.mean((hours >= 17) & (hours < 19))
        feat['daytime_ratio'] = np.mean((hours >= 9) & (hours < 17))
        
        # 周末
        feat['weekend_ratio'] = vehicle_events['is_weekend'].mean()
        
        # 时间集中度
        hour_dist = pd.Series(hours).value_counts(normalize=True)
        feat['temporal_concentration'] = -entropy(hour_dist)
        
        feat['peak_activity_hour'] = hours.mean()
        feat['activity_hour_std'] = hours.std()
        
        # 时间规律性（相邻事件间隔）
        time_diffs = vehicle_events['start_datetime'].diff().dt.total_seconds().dropna() / 3600  # 小时
        if len(time_diffs) > 0:
            feat['avg_interval_hours'] = time_diffs.mean()
            feat['interval_regularity'] = 1 / (time_diffs.std() + 1)
        else:
            feat['avg_interval_hours'] = 0
            feat['interval_regularity'] = 0
        
        # ==================== D. 基础统计特征 ====================
        
        feat['total_events'] = len(vehicle_events)
        feat['total_duration_hours'] = vehicle_events['duration_seconds'].sum() / 3600
        feat['total_distance_km'] = vehicle_events['distance_km'].sum()
        feat['total_energy_kwh'] = vehicle_events['energy_consumption_kwh'].sum()
        
        feat['speed_mean'] = vehicle_events['speed_mean'].mean()
        feat['speed_std'] = vehicle_events['speed_mean'].std()
        feat['idle_ratio_mean'] = vehicle_events['idle_ratio'].mean()
        feat['accel_abs_mean'] = vehicle_events['accel_abs_mean'].mean()
        
        feat['power_mean'] = vehicle_events['power_mean'].mean()
        feat['efficiency_kwh_per_km'] = vehicle_events['efficiency_kwh_per_km'].mean()
        
        # 事件簇分布
        for cluster_id in range(4):
            feat[f'event_cluster_{cluster_id}_ratio'] = (vehicle_events['cluster'] == cluster_id).mean()
        
        vehicle_features.append(feat)
    
    df_vehicles = pd.DataFrame(vehicle_features)
    
    print(f"\n✅ Generated {len(df_vehicles.columns)-1} features for {len(df_vehicles):,} vehicles")
    
    return df_vehicles


# ==================== 特征准备 ====================
def prepare_features(df_vehicles):
    print("\n" + "="*70)
    print("🔧 Preparing Features")
    print("="*70)
    
    feature_groups = {
        '续航焦虑': [
            'range_anxiety_threshold',
            'charging_trigger_soc',
            'critical_soc_ratio',
            'comfort_soc_buffer',
        ],
        '模式切换': [
            'mode_switch_rate',
            'mode_diversity',
            'aggressive_switch_rate',
            'mode_switch_asymmetry',
        ],
        '时间偏好': [
            'night_driving_ratio',
            'morning_peak_ratio',
            'evening_peak_ratio',
            'weekend_ratio',
            'temporal_concentration',
            'interval_regularity',
        ],
        '驾驶行为': [
            'speed_mean',
            'idle_ratio_mean',
            'accel_abs_mean',
        ],
        '能量使用': [
            'power_mean',
            'efficiency_kwh_per_km',
            'total_energy_kwh',
        ],
        '使用强度': [
            'total_events',
            'total_distance_km',
        ],
        '事件簇分布': [
            'event_cluster_0_ratio',
            'event_cluster_1_ratio',
            'event_cluster_2_ratio',
            'event_cluster_3_ratio',
        ]
    }
    
    all_features = []
    for features in feature_groups.values():
        all_features.extend(features)
    
    available = [f for f in all_features if f in df_vehicles.columns]
    
    print(f"\n📋 {len(available)} features:")
    for group, features in feature_groups.items():
        avail_in_group = [f for f in features if f in available]
        print(f"   {group:15s}: {len(avail_in_group)}")
    
    X = df_vehicles[available].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"\n✅ Matrix: {X_scaled.shape}")
    
    return X_scaled, available, scaler, feature_groups


# ==================== K值选择 ====================
def find_optimal_k(X, k_range=range(2, 8)):
    print("\n" + "="*70)
    print("🔍 Finding Optimal K")
    print("="*70)
    
    metrics = {'inertias': [], 'silhouettes': [], 'db': []}
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = kmeans.fit_predict(X)
        
        metrics['inertias'].append(kmeans.inertia_)
        metrics['silhouettes'].append(silhouette_score(X, labels))
        metrics['db'].append(davies_bouldin_score(X, labels))
        
        print(f"   K={k}: Sil={metrics['silhouettes'][-1]:.3f}, DB={metrics['db'][-1]:.3f}")
    
    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].plot(k_range, metrics['inertias'], 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('K', fontweight='bold')
    axes[0].set_ylabel('Inertia', fontweight='bold')
    axes[0].set_title('Elbow Method', fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(k_range, metrics['silhouettes'], 'go-', linewidth=2, markersize=8)
    axes[1].set_xlabel('K', fontweight='bold')
    axes[1].set_ylabel('Silhouette', fontweight='bold')
    axes[1].set_title('Silhouette (Higher Better)', fontweight='bold')
    axes[1].axhline(y=0.5, color='r', linestyle='--')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(k_range, metrics['db'], 'ro-', linewidth=2, markersize=8)
    axes[2].set_xlabel('K', fontweight='bold')
    axes[2].set_ylabel('Davies-Bouldin', fontweight='bold')
    axes[2].set_title('DB (Lower Better)', fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./results/k_selection_advanced.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: k_selection_advanced.png")
    
    best_k = k_range[np.argmax(metrics['silhouettes'])]
    print(f"\n💡 Recommended K: {best_k}")
    
    return best_k


# ==================== 聚类 ====================
def perform_clustering(X, k):
    print(f"\n{'='*70}")
    print(f"🎯 Clustering (K={k})")
    print(f"{'='*70}")
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = kmeans.fit_predict(X)
    
    sil = silhouette_score(X, labels)
    
    print(f"\n✅ Silhouette: {sil:.3f}")
    
    unique, counts = np.unique(labels, return_counts=True)
    for cid, count in zip(unique, counts):
        print(f"   C{cid}: {count:,} ({count/len(labels)*100:.1f}%)")
    
    return labels, kmeans, sil


# ==================== 簇画像 ====================
def create_profiles(df_vehicles, labels, feature_groups):
    print(f"\n{'='*70}")
    print("💡 Cluster Profiles")
    print(f"{'='*70}")
    
    df_vehicles['cluster'] = labels
    k = len(np.unique(labels))
    
    profiles = []
    
    for cid in range(k):
        data = df_vehicles[df_vehicles['cluster'] == cid]
        
        profile = {'cluster': cid, 'count': len(data)}
        
        print(f"\n{'='*70}")
        print(f"🔷 Cluster {cid} (n={len(data):,})")
        print(f"{'='*70}")
        
        for group, features in feature_groups.items():
            print(f"\n   【{group}】")
            
            for feat in features:
                if feat in data.columns:
                    val = data[feat].mean()
                    global_mean = df_vehicles[feat].mean()
                    dev = (val - global_mean) / (df_vehicles[feat].std() + 1e-10)
                    
                    profile[feat] = val
                    
                    if abs(dev) > 0.5:
                        symbol = "🔺" if dev > 0 else "🔻"
                        print(f"      {symbol} {feat:30s}: {val:.3f} ({dev:+.2f}σ)")
        
        profiles.append(profile)
    
    df_profiles = pd.DataFrame(profiles)
    df_profiles.to_csv('./results/cluster_profiles_advanced.csv', index=False, encoding='utf-8-sig')
    print(f"\n💾 Saved: cluster_profiles_advanced.csv")
    
    return df_profiles


# ==================== Main ====================
def main():
    # 生成特征
    df_vehicles = generate_advanced_vehicle_features(df_events)
    df_vehicles.to_csv('./results/vehicle_features_advanced.csv', index=False, encoding='utf-8-sig')
    print(f"💾 Saved: vehicle_features_advanced.csv")
    
    # 准备特征
    X, features, scaler, feature_groups = prepare_features(df_vehicles)
    
    # 选择K
    optimal_k = find_optimal_k(X, k_range=range(2, 8))
    
    # 聚类
    k = 3  # 或 optimal_k
    labels, kmeans, sil = perform_clustering(X, k)
    
    # 簇画像
    profiles = create_profiles(df_vehicles, labels, feature_groups)
    
    # 保存
    df_vehicles['cluster'] = labels
    df_vehicles.to_csv('./results/vehicle_features_clustered_advanced.csv', index=False, encoding='utf-8-sig')
    
    print(f"\n{'='*70}")
    print("✅ Complete!")
    print(f"{'='*70}")
    print("\n📁 Files:")
    print("   1. vehicle_features_advanced.csv")
    print("   2. vehicle_features_clustered_advanced.csv")
    print("   3. cluster_profiles_advanced.csv")
    print("   4. k_selection_advanced.png")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()