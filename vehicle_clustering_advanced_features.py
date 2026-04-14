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
        
        event_clusters = vehicle_events['cluster'].values
        n_events = len(vehicle_events)
        
        # ==================== A. 分布特征 (Distribution) ====================
        # 四种模式各占多少比例
        
        for cluster_id in range(4):
            feat[f'event_cluster_{cluster_id}_ratio'] = (event_clusters == cluster_id).mean()
        
        # 模式多样性（Shannon熵）
        mode_dist = vehicle_events['cluster'].value_counts(normalize=True)
        feat['mode_diversity'] = entropy(mode_dist)
        
        mode_result = vehicle_events['cluster'].mode()
        feat['dominant_mode'] = mode_result.iloc[0] if len(mode_result) > 0 else 0
        feat['dominant_mode_ratio'] = (event_clusters == feat['dominant_mode']).mean()
        
        # ==================== B. 转移特征 (Transition) ====================
        # 4×4=16 转移概率矩阵：模式之间如何转换
        
        # 构建4×4转移计数矩阵
        trans_matrix = np.zeros((4, 4))
        for t in range(len(event_clusters) - 1):
            from_mode = int(event_clusters[t])
            to_mode = int(event_clusters[t + 1])
            if 0 <= from_mode < 4 and 0 <= to_mode < 4:
                trans_matrix[from_mode, to_mode] += 1
        
        # 行归一化得到转移概率
        row_sums = trans_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # 避免除零
        trans_prob = trans_matrix / row_sums
        
        # 提取16个转移概率特征
        for i in range(4):
            for j in range(4):
                feat[f'trans_{i}_to_{j}'] = trans_prob[i, j]
        
        # 总体切换率
        mode_switches = np.sum(event_clusters[:-1] != event_clusters[1:])
        feat['mode_switch_rate'] = mode_switches / (n_events - 1)
        
        # 转移矩阵熵（转移的不确定性/多样性）
        trans_entropies = []
        for i in range(4):
            if row_sums[i, 0] > 0:
                trans_entropies.append(entropy(trans_prob[i]))
        feat['transition_entropy'] = np.mean(trans_entropies) if trans_entropies else 0
        
        # 自环比例（停留在同一模式的概率）
        self_loop_sum = sum(trans_prob[i, i] for i in range(4) if row_sums[i, 0] > 0)
        n_active_modes = sum(1 for i in range(4) if row_sums[i, 0] > 0)
        feat['self_loop_ratio'] = self_loop_sum / n_active_modes if n_active_modes > 0 else 0
        
        # ==================== C. 演化特征 (Evolution) ====================
        # 时序累积、节奏、稳定性
        
        # --- C1. 时序累积 (Temporal Accumulation) ---
        # 前半段 vs 后半段的模式分布漂移
        half = n_events // 2
        if half > 0:
            first_half = event_clusters[:half]
            second_half = event_clusters[half:]
            dist_first = np.array([(first_half == c).mean() for c in range(4)])
            dist_second = np.array([(second_half == c).mean() for c in range(4)])
            # L1距离衡量模式分布漂移
            feat['mode_drift'] = np.sum(np.abs(dist_first - dist_second))
        else:
            feat['mode_drift'] = 0
        
        # SOC使用趋势（时间序列斜率）
        if n_events >= 3:
            soc_drops = vehicle_events['soc_start'].values - vehicle_events['soc_end'].values
            x_norm = np.arange(n_events, dtype=float)
            x_norm = (x_norm - x_norm.mean())
            denom = np.sum(x_norm ** 2)
            if denom > 0:
                feat['soc_trend'] = np.sum(x_norm * soc_drops) / denom
            else:
                feat['soc_trend'] = 0
        else:
            feat['soc_trend'] = 0
        
        # --- C2. 节奏 (Rhythm) ---
        # 模式序列自相关（lag-1）
        if n_events >= 3:
            mode_seq = event_clusters.astype(float)
            mode_mean = mode_seq.mean()
            mode_var = np.var(mode_seq)
            if mode_var > 0:
                autocorr = np.sum((mode_seq[:-1] - mode_mean) * (mode_seq[1:] - mode_mean)) / ((n_events - 1) * mode_var)
                feat['mode_autocorr_lag1'] = autocorr
            else:
                feat['mode_autocorr_lag1'] = 0
        else:
            feat['mode_autocorr_lag1'] = 0
        
        # 时间间隔规律性
        time_diffs = vehicle_events['start_datetime'].diff().dt.total_seconds().dropna() / 3600
        if len(time_diffs) > 1:
            feat['avg_interval_hours'] = time_diffs.mean()
            feat['interval_regularity'] = 1 / (time_diffs.std() + 1)
            td_mean = time_diffs.mean()
            feat['interval_cv'] = time_diffs.std() / td_mean if td_mean > 0 else 0
        else:
            feat['avg_interval_hours'] = 0
            feat['interval_regularity'] = 0
            feat['interval_cv'] = 0
        
        # 时段分布
        hours = vehicle_events['hour'].values
        feat['night_driving_ratio'] = np.mean((hours >= 22) | (hours < 6))
        feat['morning_peak_ratio'] = np.mean((hours >= 7) & (hours < 9))
        feat['evening_peak_ratio'] = np.mean((hours >= 17) & (hours < 19))
        feat['daytime_ratio'] = np.mean((hours >= 9) & (hours < 17))
        feat['weekend_ratio'] = vehicle_events['is_weekend'].mean()
        
        # 时间集中度
        hour_dist = pd.Series(hours).value_counts(normalize=True)
        feat['temporal_concentration'] = entropy(hour_dist)
        
        # --- C3. 稳定性 (Stability) ---
        # 将事件序列分成若干窗口，计算各窗口模式分布熵的标准差
        window_size = max(n_events // 3, 2)
        if n_events >= 6:
            window_entropies = []
            for w_start in range(0, n_events - window_size + 1, window_size):
                w_end = min(w_start + window_size, n_events)
                w_clusters = event_clusters[w_start:w_end]
                w_dist = np.array([(w_clusters == c).mean() for c in range(4)])
                w_dist = w_dist[w_dist > 0]
                if len(w_dist) > 0:
                    window_entropies.append(entropy(w_dist))
            if len(window_entropies) > 1:
                feat['mode_entropy_stability'] = np.std(window_entropies)
            else:
                feat['mode_entropy_stability'] = 0
        else:
            feat['mode_entropy_stability'] = 0
        
        # 主导模式持续长度（平均连续相同模式的run长度）
        run_lengths = []
        current_run = 1
        for t in range(1, n_events):
            if event_clusters[t] == event_clusters[t - 1]:
                current_run += 1
            else:
                run_lengths.append(current_run)
                current_run = 1
        run_lengths.append(current_run)
        feat['avg_run_length'] = np.mean(run_lengths)
        feat['max_run_length'] = np.max(run_lengths)
        
        # ==================== D. 辅助特征 ====================
        
        # SOC边界
        feat['soc_start_mean'] = vehicle_events['soc_start'].mean()
        feat['soc_end_mean'] = vehicle_events['soc_end'].mean()
        feat['soc_end_min'] = vehicle_events['soc_end'].min()
        feat['range_anxiety_threshold'] = feat['soc_end_min']
        feat['comfort_soc_buffer'] = feat['soc_start_mean'] - feat['soc_end_mean']
        feat['critical_soc_ratio'] = (vehicle_events['soc_end'] < 20).mean()
        feat['charging_trigger_soc'] = np.percentile(vehicle_events['soc_end'], 25)
        
        # 驾驶行为
        feat['total_events'] = n_events
        feat['total_distance_km'] = vehicle_events['distance_km'].sum()
        feat['total_energy_kwh'] = vehicle_events['energy_consumption_kwh'].sum()
        feat['speed_mean'] = vehicle_events['speed_mean'].mean()
        feat['idle_ratio_mean'] = vehicle_events['idle_ratio'].mean()
        feat['accel_abs_mean'] = vehicle_events['accel_abs_mean'].mean()
        feat['power_mean'] = vehicle_events['power_mean'].mean()
        feat['efficiency_kwh_per_km'] = vehicle_events['efficiency_kwh_per_km'].mean()
        
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
        # ① 分布：四种模式各占多少比例
        '分布_Distribution': [
            'event_cluster_0_ratio',
            'event_cluster_1_ratio',
            'event_cluster_2_ratio',
            'event_cluster_3_ratio',
            'mode_diversity',
        ],
        # ② 转移：4×4=16 转移概率 + 汇总指标
        '转移_Transition': [
            'trans_0_to_0', 'trans_0_to_1', 'trans_0_to_2', 'trans_0_to_3',
            'trans_1_to_0', 'trans_1_to_1', 'trans_1_to_2', 'trans_1_to_3',
            'trans_2_to_0', 'trans_2_to_1', 'trans_2_to_2', 'trans_2_to_3',
            'trans_3_to_0', 'trans_3_to_1', 'trans_3_to_2', 'trans_3_to_3',
            'mode_switch_rate',
            'transition_entropy',
            'self_loop_ratio',
        ],
        # ③ 演化：时序累积、节奏、稳定性
        '演化_Evolution': [
            # 时序累积
            'mode_drift',
            'soc_trend',
            # 节奏
            'mode_autocorr_lag1',
            'interval_regularity',
            'interval_cv',
            'temporal_concentration',
            # 稳定性
            'mode_entropy_stability',
            'avg_run_length',
        ],
        # 辅助特征
        '辅助_Auxiliary': [
            'range_anxiety_threshold',
            'charging_trigger_soc',
            'critical_soc_ratio',
            'comfort_soc_buffer',
            'speed_mean',
            'idle_ratio_mean',
            'power_mean',
            'efficiency_kwh_per_km',
        ],
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