"""
智能寻找最佳K值（无硬性阈值）
综合考虑性能和分布均匀度
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import torch

# 设置
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("🔍 Smart K Selection for Dual-Channel Model")
print("="*70)

# ==================== 1. 加载双通道潜在特征 ====================
print("\n📂 Loading dual-channel latent features...")

try:
    from dual_channel_transformer_ae import DualChannelAutoencoder, DualChannelTrainer
    
    features_df = pd.read_csv('./results/features/combined_features.csv')
    trainer = DualChannelTrainer(n_clusters=3)
    
    X_driving, X_energy, _, _ = trainer.prepare_dual_channel_data(features_df)
    trainer.build_model(X_driving.shape[1], X_energy.shape[1], latent_dim=8)
    
    trainer.model.load_state_dict(torch.load('./results/dual_channel_best.pth'))
    latent_features = trainer.extract_features(X_driving, X_energy)
    
    print(f"✅ Latent features: {latent_features.shape}")

except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

# ==================== 2. 测试不同K值 ====================
print("\n🎯 Testing K=2 to K=10...")

k_range = range(2, 11)
results = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = kmeans.fit_predict(latent_features)
    
    # 计算指标
    sil = silhouette_score(latent_features, labels)
    ch = calinski_harabasz_score(latent_features, labels)
    db = davies_bouldin_score(latent_features, labels)
    
    # 簇大小统计
    cluster_sizes = pd.Series(labels).value_counts().sort_values(ascending=False).values
    size_std = cluster_sizes.std()
    size_cv = size_std / cluster_sizes.mean()
    min_size = cluster_sizes.min()
    max_size = cluster_sizes.max()
    min_pct = min_size / len(labels) * 100
    
    # 计算Gini系数（分布不均匀度，0=完全均匀，1=完全不均）
    sorted_sizes = np.sort(cluster_sizes)
    n = len(sorted_sizes)
    cumsum = np.cumsum(sorted_sizes)
    gini = (2 * np.sum((np.arange(1, n+1)) * sorted_sizes)) / (n * cumsum[-1]) - (n+1)/n
    
    results.append({
        'K': k,
        'Silhouette': sil,
        'CH_Score': ch,
        'DB_Score': db,
        'Size_CV': size_cv,
        'Gini': gini,
        'Min_Size': min_size,
        'Max_Size': max_size,
        'Min_Pct': min_pct,
        'Sizes': cluster_sizes.tolist()
    })
    
    print(f"\nK={k}:")
    print(f"  Silhouette: {sil:.3f}, CH: {ch:.1f}, DB: {db:.3f}")
    print(f"  Cluster sizes: {cluster_sizes.tolist()}")
    print(f"  Min: {min_size} ({min_pct:.2f}%), CV: {size_cv:.2f}, Gini: {gini:.3f}")

results_df = pd.DataFrame(results)

# ==================== 3. 智能打分系统 ====================
print("\n📊 Calculating composite scores...")

# 标准化各项指标到[0,1]
results_df['Sil_Norm'] = (results_df['Silhouette'] - results_df['Silhouette'].min()) / \
                         (results_df['Silhouette'].max() - results_df['Silhouette'].min())

results_df['CH_Norm'] = (results_df['CH_Score'] - results_df['CH_Score'].min()) / \
                        (results_df['CH_Score'].max() - results_df['CH_Score'].min())

results_df['DB_Norm'] = 1 - (results_df['DB_Score'] - results_df['DB_Score'].min()) / \
                            (results_df['DB_Score'].max() - results_df['DB_Score'].min())

results_df['CV_Norm'] = 1 - (results_df['Size_CV'] - results_df['Size_CV'].min()) / \
                            (results_df['Size_CV'].max() - results_df['Size_CV'].min())

results_df['Gini_Norm'] = 1 - results_df['Gini']  # 越低越好

results_df['MinPct_Norm'] = results_df['Min_Pct'] / results_df['Min_Pct'].max()

# 方案1: 性能优先（70%性能 + 30%均匀度）
results_df['Score_Performance'] = (
    results_df['Sil_Norm'] * 0.4 +
    results_df['CH_Norm'] * 0.2 +
    results_df['DB_Norm'] * 0.1 +
    results_df['CV_Norm'] * 0.2 +
    results_df['MinPct_Norm'] * 0.1
)

# 方案2: 均衡型（50%性能 + 50%均匀度）
results_df['Score_Balanced'] = (
    results_df['Sil_Norm'] * 0.3 +
    results_df['CH_Norm'] * 0.1 +
    results_df['DB_Norm'] * 0.1 +
    results_df['CV_Norm'] * 0.25 +
    results_df['Gini_Norm'] * 0.15 +
    results_df['MinPct_Norm'] * 0.1
)

# 方案3: 分布优先（30%性能 + 70%均匀度）
results_df['Score_Distribution'] = (
    results_df['Sil_Norm'] * 0.2 +
    results_df['CH_Norm'] * 0.1 +
    results_df['CV_Norm'] * 0.35 +
    results_df['Gini_Norm'] * 0.25 +
    results_df['MinPct_Norm'] * 0.1
)

# ==================== 4. 三种推荐方案 ====================
print("\n🏆 Recommendations:")

# 方案1: 性能优先
best_perf_idx = results_df['Score_Performance'].idxmax()
best_perf = results_df.loc[best_perf_idx]
print(f"\n1️⃣ Performance-Oriented (70% metrics, 30% balance):")
print(f"   Recommended K = {int(best_perf['K'])}")
print(f"   Silhouette: {best_perf['Silhouette']:.3f}")
print(f"   Cluster sizes: {best_perf['Sizes']}")
print(f"   Min cluster: {best_perf['Min_Pct']:.2f}%")

# 方案2: 均衡型
best_bal_idx = results_df['Score_Balanced'].idxmax()
best_bal = results_df.loc[best_bal_idx]
print(f"\n2️⃣ Balanced (50% metrics, 50% balance):")
print(f"   Recommended K = {int(best_bal['K'])}")
print(f"   Silhouette: {best_bal['Silhouette']:.3f}")
print(f"   Cluster sizes: {best_bal['Sizes']}")
print(f"   Min cluster: {best_bal['Min_Pct']:.2f}%")

# 方案3: 分布优先
best_dist_idx = results_df['Score_Distribution'].idxmax()
best_dist = results_df.loc[best_dist_idx]
print(f"\n3️⃣ Distribution-Oriented (30% metrics, 70% balance):")
print(f"   Recommended K = {int(best_dist['K'])}")
print(f"   Silhouette: {best_dist['Silhouette']:.3f}")
print(f"   Cluster sizes: {best_dist['Sizes']}")
print(f"   Min cluster: {best_dist['Min_Pct']:.2f}%")

# 最终推荐（使用均衡型）
recommended_k = int(best_bal['K'])
print(f"\n✨ FINAL RECOMMENDATION (Balanced): K={recommended_k}")

# ==================== 5. 可视化 ====================
print("\n📊 Generating visualizations...")

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 5.1 Silhouette
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(results_df['K'], results_df['Silhouette'], 'o-', linewidth=2, markersize=10, color='#2E86AB')
ax1.axvline(recommended_k, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Recommended K={recommended_k}')
ax1.set_xlabel('K', fontsize=11, fontweight='bold')
ax1.set_ylabel('Silhouette Score', fontsize=11, fontweight='bold')
ax1.set_title('Silhouette Score vs K', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# 5.2 CH Score (Elbow)
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(results_df['K'], results_df['CH_Score'], 'o-', linewidth=2, markersize=10, color='#A23B72')
ax2.axvline(recommended_k, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax2.set_xlabel('K', fontsize=11, fontweight='bold')
ax2.set_ylabel('CH Score', fontsize=11, fontweight='bold')
ax2.set_title('Elbow Method (CH Score)', fontsize=12, fontweight='bold')
ax2.grid(alpha=0.3)

# 5.3 DB Score
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(results_df['K'], results_df['DB_Score'], 'o-', linewidth=2, markersize=10, color='#F18F01')
ax3.axvline(recommended_k, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax3.set_xlabel('K', fontsize=11, fontweight='bold')
ax3.set_ylabel('DB Score (lower better)', fontsize=11, fontweight='bold')
ax3.set_title('Davies-Bouldin Score', fontsize=12, fontweight='bold')
ax3.grid(alpha=0.3)

# 5.4 Cluster Size CV
ax4 = fig.add_subplot(gs[1, 0])
ax4.plot(results_df['K'], results_df['Size_CV'], 'o-', linewidth=2, markersize=10, color='#6A994E')
ax4.axvline(recommended_k, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax4.set_xlabel('K', fontsize=11, fontweight='bold')
ax4.set_ylabel('Coefficient of Variation', fontsize=11, fontweight='bold')
ax4.set_title('Cluster Size Uniformity (lower better)', fontsize=12, fontweight='bold')
ax4.grid(alpha=0.3)

# 5.5 Gini系数
ax5 = fig.add_subplot(gs[1, 1])
ax5.plot(results_df['K'], results_df['Gini'], 'o-', linewidth=2, markersize=10, color='#BC4B51')
ax5.axvline(recommended_k, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax5.set_xlabel('K', fontsize=11, fontweight='bold')
ax5.set_ylabel('Gini Coefficient', fontsize=11, fontweight='bold')
ax5.set_title('Distribution Inequality (lower better)', fontsize=12, fontweight='bold')
ax5.grid(alpha=0.3)

# 5.6 最小簇百分比
ax6 = fig.add_subplot(gs[1, 2])
ax6.plot(results_df['K'], results_df['Min_Pct'], 'o-', linewidth=2, markersize=10, color='#8B5A3C')
ax6.axhline(1.0, color='gray', linestyle=':', alpha=0.5, label='1% threshold')
ax6.axvline(recommended_k, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax6.set_xlabel('K', fontsize=11, fontweight='bold')
ax6.set_ylabel('Smallest Cluster %', fontsize=11, fontweight='bold')
ax6.set_title('Minimum Cluster Percentage', fontsize=12, fontweight='bold')
ax6.legend()
ax6.grid(alpha=0.3)

# 5.7 综合得分对比
ax7 = fig.add_subplot(gs[2, :])
width = 0.25
x = results_df['K'].values
x_pos = np.arange(len(x))

bars1 = ax7.bar(x_pos - width, results_df['Score_Performance'], width, 
               label='Performance (70/30)', color='#2E86AB', alpha=0.8)
bars2 = ax7.bar(x_pos, results_df['Score_Balanced'], width,
               label='Balanced (50/50)', color='#A23B72', alpha=0.8)
bars3 = ax7.bar(x_pos + width, results_df['Score_Distribution'], width,
               label='Distribution (30/70)', color='#F18F01', alpha=0.8)

ax7.set_xlabel('K', fontsize=11, fontweight='bold')
ax7.set_ylabel('Composite Score', fontsize=11, fontweight='bold')
ax7.set_title('Composite Scores Comparison (Three Strategies)', fontsize=12, fontweight='bold')
ax7.set_xticks(x_pos)
ax7.set_xticklabels(x)
ax7.legend()
ax7.grid(axis='y', alpha=0.3)

# 标注推荐K
for bars in [bars1, bars2, bars3]:
    for i, bar in enumerate(bars):
        if x[i] == recommended_k:
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height,
                    '★', ha='center', va='bottom', fontsize=20, color='red')

plt.suptitle('Optimal K Selection Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('./results/dual_channel/optimal_k_analysis.png', dpi=300, bbox_inches='tight')
print("  ✅ optimal_k_analysis.png")
plt.close()

# ==================== 6. 使用推荐K重新聚类 ====================
print(f"\n🎯 Re-clustering with K={recommended_k}...")

kmeans_final = KMeans(n_clusters=recommended_k, random_state=42, n_init=20)
final_labels = kmeans_final.fit_predict(latent_features)

final_metrics = {
    'silhouette': silhouette_score(latent_features, final_labels),
    'ch_score': calinski_harabasz_score(latent_features, final_labels),
    'db_score': davies_bouldin_score(latent_features, final_labels)
}

print(f"\n✅ Final Results (K={recommended_k}):")
print(f"   Silhouette: {final_metrics['silhouette']:.3f}")
print(f"   CH Score: {final_metrics['ch_score']:.2f}")
print(f"   DB Score: {final_metrics['db_score']:.3f}")

print(f"\n📈 Final Cluster Distribution:")
final_counts = pd.Series(final_labels).value_counts().sort_index()
for cid, count in final_counts.items():
    print(f"   Cluster {cid}: {count:5d} ({count/len(final_labels)*100:5.1f}%)")

# 保存结果
results_df_final = features_df[['event_id', 'vehicle_id']].copy()
results_df_final['cluster'] = final_labels
results_df_final.to_csv(f'./results/dual_channel/clustered_results_k{recommended_k}_optimal.csv', index=False)

results_df.to_csv('./results/dual_channel/k_analysis_detailed.csv', index=False)

print("\n" + "="*70)
print(f"✅ Smart K Selection Complete!")
print("="*70)
print(f"\n🎯 FINAL RECOMMENDATION: K={recommended_k}")
print(f"📁 Results saved in: ./results/dual_channel/")
print(f"   - clustered_results_k{recommended_k}_optimal.csv")
print(f"   - optimal_k_analysis.png")
print(f"   - k_analysis_detailed.csv")