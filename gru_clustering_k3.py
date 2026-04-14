"""
======================================================================
🎯 GRU聚类分析 - K=3版本
======================================================================
使用已有的GRU特征，重新进行K=3聚类
生成雷达图和特征对比
======================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("🎯 GRU Clustering K=3 Analysis")
print("="*70)

# 加载已有的GRU特征
features_path = Path('./results/features_k4.npy')
if not features_path.exists():
    print("❌ Error: GRU features not found")
    print("   Please run the clustering analysis first to generate features")
    exit(1)

print("\n📂 Loading GRU features...")
gru_features = np.load(features_path)
print(f"✅ Loaded features: {gru_features.shape}")

# 重新进行K=3聚类
print("\n🎯 Performing K=3 clustering...")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=20)
labels_k3 = kmeans.fit_predict(gru_features)

# 评估
sil = silhouette_score(gru_features, labels_k3)
unique, counts = np.unique(labels_k3, return_counts=True)
cv = np.std(counts) / np.mean(counts)

print(f"\n✅ Clustering completed:")
print(f"   Silhouette: {sil:.3f}")
print(f"   CV: {cv:.3f}")
print(f"\n   Distribution:")
for cluster_id, count in zip(unique, counts):
    pct = count / len(labels_k3) * 100
    print(f"      Cluster {cluster_id}: {count:6,} ({pct:5.1f}%)")

# 保存K=3标签
np.save('./results/labels_k3.npy', labels_k3)
print(f"\n💾 K=3 labels saved: ./results/labels_k3.npy")

# 加载原始序列数据来提取特征
print("\n📊 Extracting statistical features for K=3 clusters...")

# 尝试加载序列数据
driving_file = Path('./results/temporal_soc_full/driving_sequences.npy')
energy_file = Path('./results/temporal_soc_full/energy_sequences.npy')

if not (driving_file.exists() and energy_file.exists()):
    print("⚠️  Original sequence data not found")
    print("   Using synthetic data for demonstration")
    # 这种情况下，我们需要用户提供数据或从其他地方加载
    exit(1)

driving_seqs = np.load(driving_file, allow_pickle=True)
energy_seqs = np.load(energy_file, allow_pickle=True)

# 确保数据长度匹配
if len(labels_k3) != len(driving_seqs):
    print(f"⚠️  Data mismatch: labels={len(labels_k3)}, sequences={len(driving_seqs)}")
    # 截取匹配的长度
    min_len = min(len(labels_k3), len(driving_seqs))
    labels_k3 = labels_k3[:min_len]
    driving_seqs = driving_seqs[:min_len]
    energy_seqs = energy_seqs[:min_len]
    print(f"   Adjusted to: {min_len}")

# 提取每个簇的统计特征
cluster_stats = []

for cluster_id in range(3):
    print(f"\n  Analyzing Cluster {cluster_id}...")
    
    cluster_mask = (labels_k3 == cluster_id)
    cluster_driving = driving_seqs[cluster_mask]
    cluster_energy = energy_seqs[cluster_mask]
    
    stats = {}
    
    # 驾驶特征
    all_spd = np.concatenate([seq[:, 0] for seq in cluster_driving])
    all_acc = np.concatenate([seq[:, 1] for seq in cluster_driving])
    
    stats['Avg Speed'] = np.mean(all_spd)
    stats['Max Speed'] = np.percentile(all_spd, 95)
    stats['Speed Std'] = np.std(all_spd)
    stats['Accel Std'] = np.std(all_acc)
    
    # 能量特征
    all_soc = np.concatenate([seq[:, 0] for seq in cluster_energy])
    all_v = np.concatenate([seq[:, 1] for seq in cluster_energy])
    all_i = np.concatenate([seq[:, 2] for seq in cluster_energy])
    
    stats['SOC Drop Rate'] = np.mean([seq[0, 0] - seq[-1, 0] for seq in cluster_energy if len(seq) > 1])
    stats['Avg Power'] = np.mean(np.abs(all_v * all_i))
    stats['Trip Length'] = np.mean([len(seq) for seq in cluster_driving])
    
    cluster_stats.append(stats)
    
    print(f"     Avg Speed: {stats['Avg Speed']:.2f}")
    print(f"     Max Speed: {stats['Max Speed']:.2f}")
    print(f"     Trip Length: {stats['Trip Length']:.0f}")

# 保存特征
df = pd.DataFrame(cluster_stats, index=['Cluster 0', 'Cluster 1', 'Cluster 2'])
df.to_csv('./results/cluster_features_k3.csv', encoding='utf-8-sig')
print(f"\n✅ K=3 features saved: ./results/cluster_features_k3.csv")

print("\n" + "="*70)
print("📊 K=3 Cluster Features")
print("="*70)
print(df.round(2))

# ==================== 绘制K=3雷达图 ====================
print("\n🎨 Drawing K=3 radar chart...")

# 只使用关键特征
features = ['Avg Speed', 'Max Speed', 'Avg Power', 'Trip Length']

# 提取数据
data_matrix = df[features].values

# 归一化
data_normalized = np.zeros_like(data_matrix)
for j in range(data_matrix.shape[1]):
    col = data_matrix[:, j]
    col_min = col.min()
    col_max = col.max()
    
    if col_max - col_min > 1e-6:
        normalized = (col - col_min) / (col_max - col_min)
        # 放大小差异
        if (col_max - col_min) / col.mean() < 0.05:
            normalized = 0.3 + normalized * 0.7
    else:
        normalized = np.ones(len(col)) * 0.5
    
    data_normalized[:, j] = normalized

# 绘制雷达图
N = len(features)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig = plt.figure(figsize=(14, 14))
ax = plt.subplot(111, projection='polar')

# 背景
ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('white')

# 配色（3种颜色）
colors = ['#E74C3C', '#3498DB', '#2ECC71']  # 红、蓝、绿
cluster_names = ['Cluster 0', 'Cluster 1', 'Cluster 2']
markers = ['o', 's', '^']
line_styles = ['-', '--', '-.']

# 绘制每个簇
for i in range(3):
    values = data_normalized[i].tolist()
    values += values[:1]
    
    ax.plot(angles, values, 
           linewidth=4.5, 
           linestyle=line_styles[i],
           color=colors[i], 
           marker=markers[i],
           markersize=16,
           markeredgecolor='white',
           markeredgewidth=3,
           label=cluster_names[i],
           zorder=10)
    
    ax.fill(angles, values, alpha=0.20, color=colors[i], zorder=5)

# 设置标签
ax.set_xticks(angles[:-1])
ax.set_xticklabels(features, fontsize=16, fontweight='bold', color='#2c3e50')

# 径向刻度
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], 
                   fontsize=13, color='#7f8c8d', fontweight='bold')

# 网格
ax.grid(True, linestyle='--', linewidth=1.8, alpha=0.5, color='#34495e')

# 背景圆环
for y in [0.2, 0.4, 0.6, 0.8, 1.0]:
    ax.plot(angles, [y] * len(angles), 'k-', linewidth=1, alpha=0.25)

# 标题
ax.set_title('Driving Behavior Clustering (K=3)\nFeature Comparison', 
            fontsize=24, fontweight='bold', pad=50, color='#2c3e50')

# 图例
legend = ax.legend(loc='upper right', 
                  bbox_to_anchor=(1.4, 1.15),
                  fontsize=16,
                  frameon=True,
                  fancybox=True,
                  shadow=True,
                  framealpha=0.95,
                  edgecolor='#34495e',
                  facecolor='white')

for text, color in zip(legend.get_texts(), colors):
    text.set_color(color)
    text.set_fontweight('bold')

plt.tight_layout()
output_radar = './results/cluster_radar_k3.png'
plt.savefig(output_radar, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"✅ K=3 radar chart saved: {output_radar}")

# ==================== 绘制特征柱状图 ====================
print("\n📊 Drawing feature comparison bars...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, feat in enumerate(features):
    ax = axes[idx]
    
    values = data_matrix[:, idx]
    x = np.arange(3)
    
    bars = ax.bar(x, values, color=colors, alpha=0.85, 
                  edgecolor='black', linewidth=2.5, width=0.65)
    
    ax.set_xlabel('Cluster', fontsize=13, fontweight='bold')
    ax.set_ylabel(feat, fontsize=13, fontweight='bold')
    ax.set_title(feat, fontsize=15, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(['C0', 'C1', 'C2'], fontweight='bold', fontsize=13)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1.2)
    ax.set_facecolor('#f8f9fa')
    
    # 数值标签
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.2f}', ha='center', va='bottom', 
               fontsize=12, fontweight='bold')
    
    # 变异度
    variation = (values.max() - values.min()) / values.mean() * 100
    ax.text(0.98, 0.98, f'Var: {variation:.1f}%', 
           transform=ax.transAxes, ha='right', va='top',
           fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6, 
                    edgecolor='blue', linewidth=2))

plt.suptitle('K=3 Cluster Feature Comparison', 
            fontsize=18, fontweight='bold', y=0.98)
plt.tight_layout()
output_bars = './results/cluster_features_bars_k3.png'
plt.savefig(output_bars, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ K=3 bar charts saved: {output_bars}")

# ==================== 对比K=3 vs K=4 ====================
print("\n" + "="*70)
print("📊 K=3 vs K=4 Comparison")
print("="*70)

# 加载K=4结果
labels_k4 = np.load('./results/labels_k4.npy')
sil_k4 = silhouette_score(gru_features, labels_k4)
cv_k4 = np.std(np.bincount(labels_k4)) / np.mean(np.bincount(labels_k4))

print(f"\nK=3:")
print(f"   Silhouette: {sil:.3f}")
print(f"   CV: {cv:.3f}")
print(f"   Clusters: {len(unique)}")

print(f"\nK=4:")
print(f"   Silhouette: {sil_k4:.3f}")
print(f"   CV: {cv_k4:.3f}")
print(f"   Clusters: 4")

if sil > sil_k4:
    print(f"\n✅ K=3 has better silhouette score (+{(sil-sil_k4)/sil_k4*100:.1f}%)")
else:
    print(f"\n⚠️  K=4 has better silhouette score (+{(sil_k4-sil)/sil*100:.1f}%)")

if cv < cv_k4:
    print(f"✅ K=3 has better balance (lower CV)")
else:
    print(f"⚠️  K=4 has better balance (lower CV)")

print("\n" + "="*70)
print("✅ K=3 Analysis Complete!")
print("="*70)
print(f"\n📁 Generated files:")
print(f"   1. {output_radar} - K=3 radar chart")
print(f"   2. {output_bars} - K=3 bar charts")
print(f"   3. ./results/cluster_features_k3.csv - K=3 features")
print(f"   4. ./results/labels_k3.npy - K=3 cluster labels")
print("="*70)