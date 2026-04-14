"""
车辆级聚类分析
1. 从事件表聚合车辆特征
2. 尝试不同K值聚类
3. 为充电行为分析做准备
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
    """
    print("\n" + "="*70)
    print("📊 Generating Vehicle-Level Features")
    print("="*70)
    
    vehicle_features = []
    
    for vehicle_id in df_events['vehicle_id'].unique():
        vehicle_events = df_events[df_events['vehicle_id'] == vehicle_id]
        
        feat = {'vehicle_id': vehicle_id}
        
        # ==================== 基本统计 ====================
        feat['total_events'] = len(vehicle_events)
        feat['total_duration_hours'] = vehicle_events['duration_seconds'].sum() / 3600
        feat['total_distance_km'] = vehicle_events['distance_km'].sum()
        feat['total_energy_kwh'] = vehicle_events['energy_consumption_kwh'].sum()
        
        # ==================== 事件簇分布（重要！体现驾驶模式多样性） ====================
        event_cluster_dist = vehicle_events['cluster'].value_counts(normalize=True)
        for cluster_id in range(4):
            feat[f'event_cluster_{cluster_id}_ratio'] = event_cluster_dist.get(cluster_id, 0)
        
        # 主导事件簇
        feat['dominant_event_cluster'] = vehicle_events['cluster'].mode()[0] if len(vehicle_events) > 0 else -1
        
        # 事件簇多样性（熵）
        from scipy.stats import entropy
        feat['event_cluster_diversity'] = entropy(event_cluster_dist.values)
        
        # ==================== 驾驶行为特征 ====================
        feat['speed_mean'] = vehicle_events['speed_mean'].mean()
        feat['speed_std'] = vehicle_events['speed_mean'].std()
        feat['speed_max'] = vehicle_events['speed_max'].max()
        
        feat['idle_ratio_mean'] = vehicle_events['idle_ratio'].mean()
        feat['high_speed_ratio_mean'] = vehicle_events['high_speed_ratio'].mean()
        
        feat['accel_abs_mean'] = vehicle_events['accel_abs_mean'].mean()
        feat['harsh_accel_rate'] = vehicle_events['harsh_accel_count'].sum() / len(vehicle_events)
        feat['harsh_brake_rate'] = vehicle_events['harsh_brake_count'].sum() / len(vehicle_events)
        
        # ==================== 能量特征 ====================
        feat['power_mean'] = vehicle_events['power_mean'].mean()
        feat['power_std'] = vehicle_events['power_mean'].std()
        feat['power_max'] = vehicle_events['power_max'].max()
        
        # 能效
        feat['efficiency_kwh_per_km_mean'] = vehicle_events['efficiency_kwh_per_km'].mean()
        feat['efficiency_soc_per_km_mean'] = vehicle_events['efficiency_soc_per_km'].mean()
        
        # ==================== SOC使用模式（充电分析的关键） ====================
        feat['soc_start_mean'] = vehicle_events['soc_start'].mean()
        feat['soc_start_std'] = vehicle_events['soc_start'].std()
        feat['soc_start_min'] = vehicle_events['soc_start'].min()  # 最低起始SOC
        feat['soc_start_max'] = vehicle_events['soc_start'].max()
        
        feat['soc_end_mean'] = vehicle_events['soc_end'].mean()
        feat['soc_end_std'] = vehicle_events['soc_end'].std()
        feat['soc_end_min'] = vehicle_events['soc_end'].min()  # 最低结束SOC（充电触发点）
        
        feat['soc_drop_mean'] = vehicle_events['soc_drop'].mean()
        feat['soc_drop_std'] = vehicle_events['soc_drop'].std()
        
        # SOC使用范围
        feat['soc_usage_range'] = feat['soc_start_max'] - feat['soc_end_min']
        
        # 低SOC事件比例（<30%）
        feat['low_soc_event_ratio'] = (vehicle_events['soc_end'] < 30).mean()
        
        # ==================== 行程特征 ====================
        feat['trip_length_mean'] = vehicle_events['num_points'].mean()
        feat['trip_length_std'] = vehicle_events['num_points'].std()
        
        feat['trip_duration_mean'] = vehicle_events['duration_seconds'].mean()
        feat['trip_duration_std'] = vehicle_events['duration_seconds'].std()
        
        # ==================== 时间模式（如果有时间戳） ====================
        # TODO: 提取小时、星期等特征
        
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
    
    # 选择用于聚类的特征
    clustering_features = [
        # 驾驶行为
        'speed_mean', 'speed_std', 'idle_ratio_mean', 'high_speed_ratio_mean',
        'accel_abs_mean', 'harsh_accel_rate', 'harsh_brake_rate',
        
        # 能量使用
        'power_mean', 'efficiency_kwh_per_km_mean',
        
        # SOC模式（充电相关）
        'soc_start_mean', 'soc_end_mean', 'soc_end_min', 'soc_usage_range', 'low_soc_event_ratio',
        
        # 事件簇分布（驾驶模式多样性）
        'event_cluster_0_ratio', 'event_cluster_1_ratio', 'event_cluster_2_ratio', 'event_cluster_3_ratio',
        'event_cluster_diversity',
        
        # 使用强度
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