"""
为Transformer-AE的结果找到更好的簇数
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt

# 读取Transformer-AE的潜在特征
# 如果有保存的latent features，直接读取
# 否则使用combined features

features_df = pd.read_csv('./results/features/combined_features.csv')
X = features_df.drop(['event_id', 'vehicle_id'], axis=1, errors='ignore')
X = X.select_dtypes(include=[np.number])
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

# 过滤驾驶事件
driving_mask = (features_df['speed_mean'] > 5) & \
               (features_df['distance_total'] > 0.5) & \
               (features_df['moving_ratio'] > 0.3)
X_driving = X[driving_mask]

print(f"驾驶事件数: {len(X_driving)}")

# 测试不同K值
k_range = range(3, 11)
results = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_driving)
    
    # 计算簇大小分布的均匀度（标准差/平均值）
    cluster_sizes = pd.Series(labels).value_counts().values
    size_cv = cluster_sizes.std() / cluster_sizes.mean()
    
    metrics = {
        'k': k,
        'silhouette': silhouette_score(X_driving, labels),
        'ch_score': calinski_harabasz_score(X_driving, labels),
        'db_score': davies_bouldin_score(X_driving, labels),
        'size_cv': size_cv,  # 簇大小变异系数（越小越均匀）
        'min_size': cluster_sizes.min(),
        'max_size': cluster_sizes.max()
    }
    results.append(metrics)
    
    print(f"K={k}: Sil={metrics['silhouette']:.3f}, CV={size_cv:.2f}, "
          f"Size: {cluster_sizes.min()}-{cluster_sizes.max()}")

results_df = pd.DataFrame(results)

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(results_df['k'], results_df['silhouette'], 'o-', linewidth=2)
axes[0, 0].set_xlabel('K')
axes[0, 0].set_ylabel('Silhouette Score')
axes[0, 0].set_title('Silhouette Score vs K')
axes[0, 0].grid(alpha=0.3)

axes[0, 1].plot(results_df['k'], results_df['ch_score'], 'o-', linewidth=2, color='green')
axes[0, 1].set_xlabel('K')
axes[0, 1].set_ylabel('CH Score')
axes[0, 1].set_title('CH Score vs K')
axes[0, 1].grid(alpha=0.3)

axes[1, 0].plot(results_df['k'], results_df['size_cv'], 'o-', linewidth=2, color='red')
axes[1, 0].set_xlabel('K')
axes[1, 0].set_ylabel('Size CV (lower is better)')
axes[1, 0].set_title('Cluster Size Uniformity')
axes[1, 0].grid(alpha=0.3)

# 簇大小范围
for _, row in results_df.iterrows():
    axes[1, 1].plot([row['k'], row['k']], [row['min_size'], row['max_size']], 'o-', linewidth=2)
axes[1, 1].set_xlabel('K')
axes[1, 1].set_ylabel('Cluster Size Range')
axes[1, 1].set_title('Min-Max Cluster Sizes')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('./results/better_k_analysis.png', dpi=300)
print("\n✅ 分析图已保存: ./results/better_k_analysis.png")

# 推荐
best_k = results_df.loc[results_df['silhouette'].idxmax(), 'k']
most_uniform_k = results_df.loc[results_df['size_cv'].idxmin(), 'k']

print(f"\n💡 推荐:")
print(f"  最佳性能: K = {int(best_k)} (Silhouette最高)")
print(f"  最均匀: K = {int(most_uniform_k)} (簇大小最均匀)")