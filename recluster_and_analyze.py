"""
重新聚类并分析 - 使用K-Means在特征空间
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("="*70)
print("🔄 重新聚类分析")
print("="*70)

# ==================== 1. 加载数据 ====================
print("\n📂 加载数据...")
features_df = pd.read_csv('./results/features/combined_features.csv')

print(f"原始数据: {len(features_df)} 个事件")
print(f"特征数: {len(features_df.columns)}")

# ==================== 2. 过滤真实驾驶事件 ====================
print("\n🔍 过滤驾驶事件...")

driving_events = features_df[
    (features_df['speed_mean'] > 5) &  # 平均速度 > 5 km/h
    (features_df['distance_total'] > 0.5) &  # 距离 > 0.5 km
    (features_df['moving_ratio'] > 0.3)  # 移动比例 > 30%
].copy()

print(f"驾驶事件: {len(driving_events)} ({len(driving_events)/len(features_df)*100:.1f}%)")

if len(driving_events) < 100:
    print("⚠️  驾驶事件太少！降低过滤条件...")
    driving_events = features_df[
        (features_df['speed_mean'] > 1) |  # 速度 > 1
        (features_df['distance_total'] > 0.1)  # 或距离 > 0.1
    ].copy()
    print(f"调整后: {len(driving_events)} 个事件")

# ==================== 3. 选择关键��征 ====================
print("\n🔧 选择聚类特征...")

key_features = [
    'speed_mean', 'speed_max', 'speed_std', 'speed_median',
    'acc_mean', 'acc_std', 'acc_max', 'acc_min',
    'harsh_accel', 'harsh_decel',
    'power_mean', 'power_max',
    'distance_total', 'duration_minutes',
    'moving_ratio', 'idle_ratio',
    'sharp_turn_count', 'stop_count'
]

# 只保留存在的特征
available_features = [f for f in key_features if f in driving_events.columns]
print(f"可用特征: {len(available_features)}")
print(f"  {available_features[:10]}...")

X = driving_events[available_features].copy()
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

# ==================== 4. 标准化 ====================
print("\n📊 标准化特征...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==================== 5. 确定最佳聚类数 ====================
print("\n🔍 确定最佳聚类数（肘部法则）...")

inertias = []
silhouettes = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))
    print(f"  K={k}: Silhouette={silhouettes[-1]:.3f}")

# 绘制肘部图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(k_range, inertias, 'bo-', linewidth=2)
axes[0].set_xlabel('Number of Clusters (K)')
axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow Method')
axes[0].grid(True, alpha=0.3)

axes[1].plot(k_range, silhouettes, 'ro-', linewidth=2)
axes[1].set_xlabel('Number of Clusters (K)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score vs K')
axes[1].grid(True, alpha=0.3)

os.makedirs('./results/recluster', exist_ok=True)
plt.tight_layout()
plt.savefig('./results/recluster/elbow_method.png', dpi=300, bbox_inches='tight')
print(f"\n📊 肘部图已保存")
plt.close()

# 选择最佳K（Silhouette最大）
best_k = k_range[np.argmax(silhouettes)]
print(f"\n✅ 最佳聚类数: K={best_k} (Silhouette={max(silhouettes):.3f})")

# ==================== 6. 使用最佳K重新聚类 ====================
print(f"\n🎯 使用K={best_k}进行聚类...")

kmeans_best = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels_best = kmeans_best.fit_predict(X_scaled)

# 评估
metrics = {
    'silhouette': silhouette_score(X_scaled, labels_best),
    'ch_score': calinski_harabasz_score(X_scaled, labels_best),
    'db_score': davies_bouldin_score(X_scaled, labels_best)
}

print(f"\n聚类效果:")
print(f"  Silhouette Score: {metrics['silhouette']:.3f}")
print(f"  CH Score: {metrics['ch_score']:.2f}")
print(f"  DB Score: {metrics['db_score']:.3f}")

print(f"\n簇分布:")
cluster_counts = pd.Series(labels_best).value_counts().sort_index()
for cluster_id, count in cluster_counts.items():
    print(f"  簇 {cluster_id}: {count} 事件 ({count/len(labels_best)*100:.1f}%)")

# ==================== 7. 保存结果 ====================
driving_events['cluster'] = labels_best

results_df = driving_events[['event_id', 'vehicle_id', 'cluster']].copy()
results_df.to_csv('./results/recluster/clustered_results.csv', index=False)

print(f"\n💾 聚类结果已保存: ./results/recluster/clustered_results.csv")

# ==================== 8. 可视化聚类结果 ====================
print("\n📊 生成可视化...")

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# PCA可视化
scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels_best, 
                           cmap='tab10', alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
axes[0].set_title(f'Clustering Result (K={best_k})')
axes[0].grid(True, alpha=0.3)
plt.colorbar(scatter1, ax=axes[0], label='Cluster')

# 簇大小
axes[1].bar(cluster_counts.index, cluster_counts.values, 
           color=sns.color_palette('Set2', len(cluster_counts)))
axes[1].set_xlabel('Cluster')
axes[1].set_ylabel('Number of Events')
axes[1].set_title('Cluster Size Distribution')
axes[1].grid(axis='y', alpha=0.3)

for i, (cluster_id, count) in enumerate(cluster_counts.items()):
    axes[1].text(i, count, str(count), ha='center', va='bottom')

plt.tight_layout()
plt.savefig('./results/recluster/clustering_visualization.png', dpi=300, bbox_inches='tight')
print(f"   ✅ 可视化已保存")
plt.close()

# ==================== 9. 簇特征分析 ====================
print("\n📋 簇特征分析...")

key_analysis_features = ['speed_mean', 'acc_std', 'harsh_accel', 'harsh_decel', 
                         'power_mean', 'distance_total']
available_analysis = [f for f in key_analysis_features if f in driving_events.columns]

cluster_profiles = driving_events.groupby('cluster')[available_analysis].mean()

print("\n各簇平均特征:")
print(cluster_profiles.to_string())

# 保存
cluster_profiles.to_csv('./results/recluster/cluster_profiles.csv')

# ==================== 10. 簇命名 ====================
print("\n🏷️  簇命名...")

cluster_names = {}
for cluster_id in cluster_profiles.index:
    profile = cluster_profiles.loc[cluster_id]
    
    speed = profile.get('speed_mean', 0)
    harsh_accel = profile.get('harsh_accel', 0)
    harsh_decel = profile.get('harsh_decel', 0)
    
    if harsh_accel > cluster_profiles['harsh_accel'].mean() * 1.5:
        name = "🔴 激进驾驶"
    elif speed < 20:
        name = "🟡 城市慢速"
    elif speed > 60:
        name = "🟢 高速巡航"
    elif harsh_decel > cluster_profiles['harsh_decel'].mean() * 1.5:
        name = "🟠 频繁制动"
    else:
        name = "⚪ 平稳驾驶"
    
    cluster_names[cluster_id] = name
    print(f"  簇 {cluster_id}: {name}")
    print(f"    - 平均速度: {speed:.1f} km/h")
    print(f"    - 急加速: {harsh_accel:.1f}")
    print(f"    - 急减速: {harsh_decel:.1f}")

print("\n" + "="*70)
print("✅ 重新聚类完成！")
print("="*70)
print("📁 结果保存在: ./results/recluster/")
print("   - clustered_results.csv (聚类结果)")
print("   - elbow_method.png (最佳K选择)")
print("   - clustering_visualization.png (聚类可视化)")
print("   - cluster_profiles.csv (簇特征)")