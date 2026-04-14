"""
车辆级聚类分析
基于三维特征体系：
  ① 分布 (Distribution)：四种模式各占多少比例
  ② 转移 (Transition)：4×4=16 模式转移概率矩阵
  ③ 演化 (Evolution)：时序累积、节奏、稳定性
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from scipy.stats import entropy

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("🚗 Vehicle-Level Clustering Analysis")
print("="*70)


# ==================== 加载事件表 ====================
print("\n📂 Loading event table...")

df_events = pd.read_csv('./results/event_table.csv')

print(f"✅ Loaded:")
print(f"   Total events: {len(df_events):,}")
print(f"   Total vehicles: {df_events['vehicle_id'].nunique():,}")
print(f"   Event clusters: {df_events['cluster'].unique()}")


# ==================== 生成车辆级特征 ====================
def generate_vehicle_features(df_events):
    """
    从事件表聚合生成车辆级特征
    基于三维体系：分布、转移、演化
    """
    print("\n" + "="*70)
    print("📊 Generating Vehicle-Level Features (Three-Dimension Framework)")
    print("="*70)
    
    # 解析时间戳（如果尚未解析）
    if 'start_datetime' not in df_events.columns:
        df_events['start_datetime'] = pd.to_datetime(
            df_events['start_time'].astype(str), format='%Y%m%d%H%M%S')
    if 'hour' not in df_events.columns:
        df_events['hour'] = df_events['start_datetime'].dt.hour
    if 'is_weekend' not in df_events.columns:
        df_events['is_weekend'] = df_events['start_datetime'].dt.dayofweek.isin([5, 6]).astype(int)
    
    vehicle_features = []
    
    for vehicle_id in df_events['vehicle_id'].unique():
        vehicle_events = df_events[df_events['vehicle_id'] == vehicle_id].sort_values('start_datetime')
        
        if len(vehicle_events) < 2:
            continue
        
        feat = {'vehicle_id': vehicle_id}
        event_clusters = vehicle_events['cluster'].values
        n_events = len(vehicle_events)
        
        # ==================== ① 分布 (Distribution) ====================
        # 四种模式各占多少比例
        
        event_cluster_dist = vehicle_events['cluster'].value_counts(normalize=True)
        for cluster_id in range(4):
            feat[f'event_cluster_{cluster_id}_ratio'] = event_cluster_dist.get(cluster_id, 0)
        
        # 主导事件簇
        mode_result = vehicle_events['cluster'].mode()
        feat['dominant_event_cluster'] = mode_result.iloc[0] if len(mode_result) > 0 else -1
        
        # 事件簇多样性（Shannon熵）
        feat['event_cluster_diversity'] = entropy(event_cluster_dist.values)
        
        # ==================== ② 转移 (Transition) ====================
        # 4×4=16 转移概率矩阵
        
        trans_matrix = np.zeros((4, 4))
        for t in range(len(event_clusters) - 1):
            from_mode = int(event_clusters[t])
            to_mode = int(event_clusters[t + 1])
            if 0 <= from_mode < 4 and 0 <= to_mode < 4:
                trans_matrix[from_mode, to_mode] += 1
        
        # 行归一化得到转移概率
        row_sums = trans_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        trans_prob = trans_matrix / row_sums
        
        # 16个转移概率特征
        for i in range(4):
            for j in range(4):
                feat[f'trans_{i}_to_{j}'] = trans_prob[i, j]
        
        # 总体切换率
        mode_switches = np.sum(event_clusters[:-1] != event_clusters[1:])
        feat['mode_switch_rate'] = mode_switches / (n_events - 1)
        
        # 转移熵
        trans_entropies = []
        for i in range(4):
            if trans_matrix.sum(axis=1)[i] > 0:
                trans_entropies.append(entropy(trans_prob[i]))
        feat['transition_entropy'] = np.mean(trans_entropies) if trans_entropies else 0
        
        # 自环比例
        n_active = sum(1 for i in range(4) if trans_matrix.sum(axis=1)[i] > 0)
        feat['self_loop_ratio'] = (
            sum(trans_prob[i, i] for i in range(4) if trans_matrix.sum(axis=1)[i] > 0)
            / n_active if n_active > 0 else 0
        )
        
        # ==================== ③ 演化 (Evolution) ====================
        # 时序累积、节奏、稳定性
        
        # --- 时序累积 ---
        half = n_events // 2
        if half > 0:
            first_half = event_clusters[:half]
            second_half = event_clusters[half:]
            dist_first = np.array([(first_half == c).mean() for c in range(4)])
            dist_second = np.array([(second_half == c).mean() for c in range(4)])
            feat['mode_drift'] = np.sum(np.abs(dist_first - dist_second))
        else:
            feat['mode_drift'] = 0
        
        # SOC使用趋势
        if n_events >= 3:
            soc_drops = vehicle_events['soc_start'].values - vehicle_events['soc_end'].values
            x_norm = np.arange(n_events, dtype=float)
            x_norm = x_norm - x_norm.mean()
            denom = np.sum(x_norm ** 2)
            feat['soc_trend'] = np.sum(x_norm * soc_drops) / denom if denom > 0 else 0
        else:
            feat['soc_trend'] = 0
        
        # --- 节奏 ---
        # 模式序列自相关（lag-1）
        if n_events >= 3:
            mode_seq = event_clusters.astype(float)
            mode_mean = mode_seq.mean()
            mode_var = np.var(mode_seq)
            if mode_var > 0:
                feat['mode_autocorr_lag1'] = (
                    np.sum((mode_seq[:-1] - mode_mean) * (mode_seq[1:] - mode_mean))
                    / ((n_events - 1) * mode_var)
                )
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
        
        # 时间集中度
        hours = vehicle_events['hour'].values
        hour_dist = pd.Series(hours).value_counts(normalize=True)
        feat['temporal_concentration'] = entropy(hour_dist)
        
        # --- 稳定性 ---
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
            feat['mode_entropy_stability'] = np.std(window_entropies) if len(window_entropies) > 1 else 0
        else:
            feat['mode_entropy_stability'] = 0
        
        # 平均连续相同模式长度
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
        
        # ==================== 辅助特征 ====================
        feat['total_events'] = n_events
        feat['total_duration_hours'] = vehicle_events['duration_seconds'].sum() / 3600
        feat['total_distance_km'] = vehicle_events['distance_km'].sum()
        feat['total_energy_kwh'] = vehicle_events['energy_consumption_kwh'].sum()
        
        feat['speed_mean'] = vehicle_events['speed_mean'].mean()
        feat['speed_std'] = vehicle_events['speed_mean'].std()
        feat['idle_ratio_mean'] = vehicle_events['idle_ratio'].mean()
        feat['accel_abs_mean'] = vehicle_events['accel_abs_mean'].mean()
        feat['harsh_accel_rate'] = vehicle_events['harsh_accel_count'].sum() / n_events
        feat['harsh_brake_rate'] = vehicle_events['harsh_brake_count'].sum() / n_events
        
        feat['power_mean'] = vehicle_events['power_mean'].mean()
        feat['efficiency_kwh_per_km_mean'] = vehicle_events['efficiency_kwh_per_km'].mean()
        
        feat['soc_start_mean'] = vehicle_events['soc_start'].mean()
        feat['soc_end_mean'] = vehicle_events['soc_end'].mean()
        feat['soc_end_min'] = vehicle_events['soc_end'].min()
        feat['soc_drop_mean'] = vehicle_events['soc_drop'].mean()
        feat['soc_usage_range'] = vehicle_events['soc_start'].max() - vehicle_events['soc_end'].min()
        feat['low_soc_event_ratio'] = (vehicle_events['soc_end'] < 30).mean()
        
        vehicle_features.append(feat)
    
    df_vehicles = pd.DataFrame(vehicle_features)
    
    print(f"\n✅ Generated features for {len(df_vehicles):,} vehicles")
    print(f"   Total features: {len(df_vehicles.columns) - 1}")
    
    return df_vehicles


# ==================== 特征选择和标准化 ====================
def prepare_features_for_clustering(df_vehicles):
    """
    选择关键特征并标准化
    """
    print("\n" + "="*70)
    print("🔧 Preparing Features for Clustering")
    print("="*70)
    
    # 选择用于聚类的特征 - 三维体系
    clustering_features = [
        # ① 分布 (Distribution)
        'event_cluster_0_ratio', 'event_cluster_1_ratio',
        'event_cluster_2_ratio', 'event_cluster_3_ratio',
        'event_cluster_diversity',
        
        # ② 转移 (Transition) - 4×4=16 转移概率
        'trans_0_to_0', 'trans_0_to_1', 'trans_0_to_2', 'trans_0_to_3',
        'trans_1_to_0', 'trans_1_to_1', 'trans_1_to_2', 'trans_1_to_3',
        'trans_2_to_0', 'trans_2_to_1', 'trans_2_to_2', 'trans_2_to_3',
        'trans_3_to_0', 'trans_3_to_1', 'trans_3_to_2', 'trans_3_to_3',
        'mode_switch_rate',
        'transition_entropy',
        'self_loop_ratio',
        
        # ③ 演化 (Evolution) - 时序累积、节奏、稳定性
        'mode_drift',
        'soc_trend',
        'mode_autocorr_lag1',
        'interval_regularity',
        'interval_cv',
        'temporal_concentration',
        'mode_entropy_stability',
        'avg_run_length',
        
        # 辅助特征
        'speed_mean', 'idle_ratio_mean',
        'power_mean', 'efficiency_kwh_per_km_mean',
        'soc_end_min', 'soc_usage_range', 'low_soc_event_ratio',
        'total_events', 'total_distance_km', 'total_energy_kwh',
    ]
    
    # 检查是否所有特征都存在
    available_features = [f for f in clustering_features if f in df_vehicles.columns]
    missing_features = set(clustering_features) - set(available_features)
    
    if missing_features:
        print(f"⚠️  Missing features: {missing_features}")
    
    print(f"\n📋 Selected {len(available_features)} features:")
    for i, feat in enumerate(available_features, 1):
        print(f"   {i:2d}. {feat}")
    
    # 提取特征矩阵
    X = df_vehicles[available_features].values
    
    # 检查缺失值和无穷值
    if np.isnan(X).any() or np.isinf(X).any():
        print("\n⚠️  Found NaN/Inf, filling with 0...")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"\n✅ Feature matrix: {X_scaled.shape}")
    
    return X_scaled, available_features, scaler


# ==================== 确定最佳K值 ====================
def find_optimal_k(X, k_range=range(2, 8)):
    """
    使用多种指标确定最佳K值
    """
    print("\n" + "="*70)
    print("🔍 Finding Optimal Number of Clusters")
    print("="*70)
    
    inertias = []
    silhouettes = []
    davies_bouldins = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = kmeans.fit_predict(X)
        
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X, labels))
        davies_bouldins.append(davies_bouldin_score(X, labels))
        
        print(f"   K={k}: Silhouette={silhouettes[-1]:.3f}, Davies-Bouldin={davies_bouldins[-1]:.3f}")
    
    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 肘部法则
    axes[0].plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Number of Clusters (K)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Inertia', fontsize=12, fontweight='bold')
    axes[0].set_title('Elbow Method', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # 轮廓系数（越大越好）
    axes[1].plot(k_range, silhouettes, 'go-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Number of Clusters (K)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
    axes[1].set_title('Silhouette Score (Higher is Better)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0.5, color='r', linestyle='--', label='Good threshold')
    axes[1].legend()
    
    # Davies-Bouldin指数（越小越好）
    axes[2].plot(k_range, davies_bouldins, 'ro-', linewidth=2, markersize=8)
    axes[2].set_xlabel('Number of Clusters (K)', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Davies-Bouldin Index', fontsize=12, fontweight='bold')
    axes[2].set_title('Davies-Bouldin Index (Lower is Better)', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle('Clustering Quality Metrics for Different K', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('./results/vehicle_clustering_k_selection.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: vehicle_clustering_k_selection.png")
    
    # 推荐K值
    best_k_silhouette = k_range[np.argmax(silhouettes)]
    best_k_db = k_range[np.argmin(davies_bouldins)]
    
    print(f"\n💡 Recommendations:")
    print(f"   Best K by Silhouette: {best_k_silhouette}")
    print(f"   Best K by Davies-Bouldin: {best_k_db}")
    
    return best_k_silhouette


# ==================== 执行聚类 ====================
def perform_vehicle_clustering(X, k, feature_names):
    """
    执行车辆聚类
    """
    print("\n" + "="*70)
    print(f"🎯 Vehicle Clustering (K={k})")
    print("="*70)
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = kmeans.fit_predict(X)
    
    sil = silhouette_score(X, labels)
    db = davies_bouldin_score(X, labels)
    
    unique, counts = np.unique(labels, return_counts=True)
    
    print(f"\n✅ Clustering completed:")
    print(f"   Silhouette Score: {sil:.3f}")
    print(f"   Davies-Bouldin Index: {db:.3f}")
    print(f"\n   Distribution:")
    for cluster_id, count in zip(unique, counts):
        pct = count / len(labels) * 100
        print(f"      Cluster {cluster_id}: {count:,} ({pct:.1f}%)")
    
    # 分析每个簇的特征中心
    print(f"\n📊 Cluster Centroids (Top 5 distinctive features per cluster):")
    
    for cluster_id in range(k):
        cluster_mask = labels == cluster_id
        cluster_center = X[cluster_mask].mean(axis=0)
        global_mean = X.mean(axis=0)
        
        # 计算每个特征的偏离度
        deviations = (cluster_center - global_mean) / (X.std(axis=0) + 1e-10)
        
        # 找出最显著的特征（正向和负向）
        top_positive_idx = np.argsort(deviations)[-5:][::-1]
        top_negative_idx = np.argsort(deviations)[:5]
        
        print(f"\n   Cluster {cluster_id}:")
        print(f"      High features:")
        for idx in top_positive_idx:
            print(f"         {feature_names[idx]}: +{deviations[idx]:.2f} std")
        print(f"      Low features:")
        for idx in top_negative_idx:
            print(f"         {feature_names[idx]}: {deviations[idx]:.2f} std")
    
    return labels, kmeans, sil


# ==================== 可视化车辆聚类 ====================
def visualize_vehicle_clustering(df_vehicles, labels, X, feature_names):
    """
    可视化车辆聚类结果
    """
    print("\n🎨 Creating visualizations...")
    
    k = len(np.unique(labels))
    colors = plt.cm.Set3(np.linspace(0, 1, k))
    
    # ==================== 图1：PCA空间 ====================
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 3, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    
    for cluster_id in range(k):
        mask = labels == cluster_id
        count = np.sum(mask)
        ax1.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=[colors[cluster_id]], label=f'C{cluster_id} (n={count:,})',
                   alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontweight='bold')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontweight='bold')
    ax1.set_title('Vehicle Clustering (PCA Space)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ==================== 图2-6：关键特征对比 ====================
    key_features = [
        'speed_mean', 'idle_ratio_mean', 'power_mean', 
        'soc_end_min', 'total_events'
    ]
    
    for idx, feat in enumerate(key_features, start=1):
        if feat not in df_vehicles.columns:
            continue
        
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        
        data_by_cluster = [df_vehicles[labels == i][feat].values for i in range(k)]
        
        bp = ax.boxplot(data_by_cluster, labels=[f'C{i}' for i in range(k)],
                       patch_artist=True, widths=0.6)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_xlabel('Cluster', fontweight='bold')
        ax.set_ylabel(feat, fontweight='bold')
        ax.set_title(feat, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Vehicle Clustering Analysis (K={k})', fontsize=18, fontweight='bold')
    plt.savefig('./results/vehicle_clustering_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: vehicle_clustering_analysis.png")


# ==================== 为充电分析准备特征 ====================
def prepare_for_charging_analysis(df_vehicles, labels):
    """
    生成充电行为分析所需的特征和标签
    """
    print("\n" + "="*70)
    print("🔋 Preparing for Charging Behavior Analysis")
    print("="*70)
    
    df_vehicles['vehicle_cluster'] = labels
    
    # 按簇分析充电相关特征
    print(f"\n📊 Charging-Related Features by Vehicle Cluster:")
    
    for cluster_id in range(len(np.unique(labels))):
        cluster_data = df_vehicles[df_vehicles['vehicle_cluster'] == cluster_id]
        
        print(f"\n🔷 Cluster {cluster_id} (n={len(cluster_data):,}):")
        
        # SOC使用模式
        print(f"   SOC Usage Pattern:")
        print(f"      Start mean: {cluster_data['soc_start_mean'].mean():.1f}%")
        print(f"      End mean: {cluster_data['soc_end_mean'].mean():.1f}%")
        print(f"      End min: {cluster_data['soc_end_min'].mean():.1f}%")
        print(f"      Low SOC event ratio: {cluster_data['low_soc_event_ratio'].mean()*100:.1f}%")
        
        # 使用强度
        print(f"   Usage Intensity:")
        print(f"      Avg events per vehicle: {cluster_data['total_events'].mean():.1f}")
        print(f"      Avg distance per vehicle: {cluster_data['total_distance_km'].mean():.1f} km")
        print(f"      Avg energy per vehicle: {cluster_data['total_energy_kwh'].mean():.1f} kWh")
        
        # 驾驶模式
        print(f"   Driving Pattern:")
        print(f"      Avg speed: {cluster_data['speed_mean'].mean():.1f} km/h")
        print(f"      Idle ratio: {cluster_data['idle_ratio_mean'].mean()*100:.1f}%")
        print(f"      Dominant event cluster: {cluster_data['dominant_event_cluster'].mode()[0] if len(cluster_data) > 0 else 'N/A'}")
    
    # 充电需求预测特征
    charging_features = [
        'soc_end_min',  # 最低SOC → 充电紧迫性
        'low_soc_event_ratio',  # 低SOC事件比例 → 充电频率需求
        'total_events',  # 使用频率 → 充电频率
        'total_energy_kwh',  # 总能耗 → 充电量需求
        'soc_usage_range',  # SOC使用范围 → 充电习惯
        'vehicle_cluster'  # 车辆类型
    ]
    
    df_charging = df_vehicles[['vehicle_id'] + charging_features].copy()
    df_charging.to_csv('./results/vehicle_charging_features.csv', index=False, encoding='utf-8-sig')
    
    print(f"\n💾 Saved: vehicle_charging_features.csv")
    print(f"   Columns: {list(df_charging.columns)}")
    
    return df_charging


# ==================== 聚类解释和命名 ====================
def interpret_clusters(df_vehicles, labels):
    """
    为每个车辆簇生成业务解释和命名
    """
    print("\n" + "="*70)
    print("💡 Cluster Interpretation")
    print("="*70)
    
    df_vehicles['vehicle_cluster'] = labels
    k = len(np.unique(labels))
    
    cluster_profiles = []
    
    for cluster_id in range(k):
        cluster_data = df_vehicles[df_vehicles['vehicle_cluster'] == cluster_id]
        
        profile = {
            'cluster': cluster_id,
            'count': len(cluster_data),
            'avg_speed': cluster_data['speed_mean'].mean(),
            'avg_idle_ratio': cluster_data['idle_ratio_mean'].mean(),
            'avg_power': cluster_data['power_mean'].mean(),
            'avg_events': cluster_data['total_events'].mean(),
            'avg_distance': cluster_data['total_distance_km'].mean(),
            'soc_end_min': cluster_data['soc_end_min'].mean(),
            'low_soc_ratio': cluster_data['low_soc_event_ratio'].mean()
        }
        
        # 业务命名逻辑
        tags = []
        
        # 使用频率
        if profile['avg_events'] > df_vehicles['total_events'].quantile(0.75):
            tags.append('高频使用')
        elif profile['avg_events'] < df_vehicles['total_events'].quantile(0.25):
            tags.append('低频使用')
        else:
            tags.append('中频使用')
        
        # 驾驶风格
        if profile['avg_speed'] > df_vehicles['speed_mean'].quantile(0.75):
            tags.append('高速驾驶')
        elif profile['avg_idle_ratio'] > df_vehicles['idle_ratio_mean'].quantile(0.75):
            tags.append('拥堵/通勤')
        else:
            tags.append('混合路况')
        
        # 充电需求
        if profile['soc_end_min'] < 20:
            tags.append('深度放电')
        if profile['low_soc_ratio'] > 0.3:
            tags.append('充电需求高')
        
        profile['tags'] = ' + '.join(tags)
        
        cluster_profiles.append(profile)
        
        print(f"\n🔷 Cluster {cluster_id}: 【{profile['tags']}】")
        print(f"   Vehicles: {profile['count']:,}")
        print(f"   Avg speed: {profile['avg_speed']:.1f} km/h")
        print(f"   Idle ratio: {profile['avg_idle_ratio']*100:.1f}%")
        print(f"   Usage: {profile['avg_events']:.0f} events, {profile['avg_distance']:.0f} km")
        print(f"   SOC end min: {profile['soc_end_min']:.1f}%")
        print(f"   Low SOC events: {profile['low_soc_ratio']*100:.1f}%")
    
    df_profiles = pd.DataFrame(cluster_profiles)
    df_profiles.to_csv('./results/vehicle_cluster_profiles.csv', index=False, encoding='utf-8-sig')
    print(f"\n💾 Saved: vehicle_cluster_profiles.csv")
    
    return df_profiles


# ==================== Main ====================
def main():
    # Step 1: 生成车辆特征
    df_vehicles = generate_vehicle_features(df_events)
    df_vehicles.to_csv('./results/vehicle_features.csv', index=False, encoding='utf-8-sig')
    print(f"💾 Saved: vehicle_features.csv")
    
    # Step 2: 准备聚类特征
    X, feature_names, scaler = prepare_features_for_clustering(df_vehicles)
    
    # Step 3: 确定最佳K值
    optimal_k = find_optimal_k(X, k_range=range(2, 8))
    
    # Step 4: 执行聚类（可以手动指定K）
    # 建议：K=3或4最常见
    k = 3  # 你可以改成 optimal_k 或其他值
    labels, kmeans, sil = perform_vehicle_clustering(X, k, feature_names)
    
    # Step 5: 可视化
    visualize_vehicle_clustering(df_vehicles, labels, X, feature_names)
    
    # Step 6: 簇解释
    df_profiles = interpret_clusters(df_vehicles, labels)
    
    # Step 7: 为充电分析准备
    df_charging = prepare_for_charging_analysis(df_vehicles, labels)
    
    # Step 8: 保存最终结果
    df_vehicles['vehicle_cluster'] = labels
    df_vehicles.to_csv('./results/vehicle_features_with_clusters.csv', index=False, encoding='utf-8-sig')
    
    print("\n" + "="*70)
    print("✅ Vehicle Clustering Complete!")
    print("="*70)
    print("\n📁 Generated files:")
    print("   1. vehicle_features.csv - 车辆特征表")
    print("   2. vehicle_features_with_clusters.csv - 带聚类标签的车辆表")
    print("   3. vehicle_clustering_k_selection.png - K值选择图")
    print("   4. vehicle_clustering_analysis.png - 聚类分析图")
    print("   5. vehicle_cluster_profiles.csv - 簇画像")
    print("   6. vehicle_charging_features.csv - 充电分析特征")
    print("\n💡 Next steps for charging analysis:")
    print("   - 收集充电数据（充电时间、充电量、充电类型）")
    print("   - 分析每个车辆簇的充电习惯")
    print("   - 预测充电需求")
    print("="*70)


if __name__ == "__main__":
    main()