"""
Cross-Attention模型 - 4个高区分度特征雷达图
去掉Speed Std和Accel Std，使用更好的归一化方法拉开差异
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("🎨 Radar Chart - 4 High-Variance Features (Cross-Attention)")
print("="*70)

# ==================== 加载数据 ====================
print("\n📂 Loading data...")

labels = np.load('./results/labels_k4_crossattn.npy')
driving_seqs = np.load('./results/temporal_soc_full/driving_sequences.npy', allow_pickle=True)
energy_seqs = np.load('./results/temporal_soc_full/energy_sequences.npy', allow_pickle=True)

min_len = min(len(labels), len(driving_seqs), len(energy_seqs))
labels = labels[:min_len]
driving_seqs = driving_seqs[:min_len]
energy_seqs = energy_seqs[:min_len]

print(f"✅ Loaded {len(labels):,} samples")

# ==================== 只提取4个高区分度特征 ====================
print("\n📊 Extracting 4 high-variance features...")

# 只要这4个有明显差异的特征
features_to_use = [
    'Avg Speed',
    'Max Speed',
    'Avg Power',
    'Trip Length'
]

cluster_stats = []

for cluster_id in range(4):
    cluster_mask = (labels == cluster_id)
    cluster_driving = driving_seqs[cluster_mask]
    cluster_energy = energy_seqs[cluster_mask]
    
    stats = {}
    
    # 驾驶特征
    all_spd = np.concatenate([seq[:, 0] for seq in cluster_driving])
    stats['Avg Speed'] = np.mean(all_spd)
    stats['Max Speed'] = np.percentile(all_spd, 95)
    
    # 能量特征
    all_v = np.concatenate([seq[:, 1] for seq in cluster_energy])
    all_i = np.concatenate([seq[:, 2] for seq in cluster_energy])
    stats['Avg Power'] = np.mean(np.abs(all_v * all_i))
    
    # 行程特征
    stats['Trip Length'] = np.mean([len(seq) for seq in cluster_driving])
    
    cluster_stats.append(stats)

df = pd.DataFrame(cluster_stats, index=['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3'])

print("\n📊 Feature values:")
print(df[features_to_use].round(2))

# 诊断特征差异
print("\n🔍 Feature variation:")
for feat in features_to_use:
    values = df[feat].values
    cv = (values.std() / values.mean() * 100) if values.mean() > 0 else 0
    print(f"   {feat:15s}: {values.min():.2f} - {values.max():.2f} (CV={cv:.1f}%)")

# ==================== 改进的归一化方法 ====================
print("\n🔧 Using improved normalization...")

data = df[features_to_use].values

# 方法：Z-score标准化后映射到[0,1]，再拉伸到更宽范围
data_normalized = np.zeros_like(data, dtype=float)

for j in range(data.shape[1]):
    col = data[:, j]
    
    # Z-score标准化
    mean_val = col.mean()
    std_val = col.std()
    
    if std_val > 1e-6:
        z_scores = (col - mean_val) / std_val
        
        # 映射到[0, 1]
        z_min = z_scores.min()
        z_max = z_scores.max()
        if z_max - z_min > 1e-6:
            normalized = (z_scores - z_min) / (z_max - z_min)
        else:
            normalized = np.ones(len(col)) * 0.5
        
        # 拉伸到[0.2, 1.0]，让差异更明显
        normalized = 0.2 + normalized * 0.8
    else:
        normalized = np.ones(len(col)) * 0.6
    
    data_normalized[:, j] = normalized

print("\nNormalized values (spread range):")
for i in range(4):
    print(f"Cluster {i}: {data_normalized[i]}")

# ==================== 绘制雷达图 ====================
print("\n🎨 Drawing radar chart...")

N = len(features_to_use)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig = plt.figure(figsize=(14, 14))
ax = plt.subplot(111, projection='polar')

# 背景
ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('white')

# 颜色配置
colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
cluster_names = ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3']
markers = ['o', 's', '^', 'D']
line_styles = ['-', '--', '-.', ':']

# 绘制每个簇
for i in range(4):
    values = data_normalized[i].tolist()
    values += values[:1]
    
    ax.plot(angles, values, 
           marker=markers[i],
           linewidth=4.5, 
           linestyle=line_styles[i],
           color=colors[i], 
           markersize=16,
           markeredgecolor='white',
           markeredgewidth=3,
           label=cluster_names[i],
           zorder=10)
    
    ax.fill(angles, values, alpha=0.18, color=colors[i], zorder=5)

# 设置标签（更大字体）
ax.set_xticks(angles[:-1])
ax.set_xticklabels(features_to_use, fontsize=16, fontweight='bold', color='#2c3e50')

# 径向刻度（设置更宽范围）
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], 
                   fontsize=13, color='#7f8c8d', fontweight='bold')

# 网格
ax.grid(True, linestyle='--', linewidth=2, alpha=0.6, color='#34495e')

# 背景圆环
for y in [0.2, 0.4, 0.6, 0.8, 1.0]:
    ax.plot(angles, [y] * len(angles), 'k-', linewidth=1.2, alpha=0.3)

# 标题
ax.set_title('Driving Behavior Clustering (K=4)\nCross-Attention Model - Key Features', 
            fontsize=22, fontweight='bold', pad=50, color='#2c3e50')

# 图例
legend = ax.legend(loc='upper right', 
                  bbox_to_anchor=(1.45, 1.15),
                  fontsize=15,
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
output = './results/cluster_radar_k4_crossattn_spread.png'
plt.savefig(output, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"\n✅ Saved: {output}")

# ==================== 绘制柱状图对比 ====================
print("\n📊 Drawing bar charts...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, feat in enumerate(features_to_use):
    ax = axes[idx]
    
    values = df[feat].values
    x = np.arange(4)
    
    bars = ax.bar(x, values, color=colors, alpha=0.85, 
                  edgecolor='black', linewidth=2.5, width=0.65)
    
    ax.set_xlabel('Cluster', fontsize=13, fontweight='bold')
    ax.set_ylabel(feat, fontsize=13, fontweight='bold')
    ax.set_title(feat, fontsize=15, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(['C0', 'C1', 'C2', 'C3'], fontweight='bold', fontsize=13)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1.2)
    ax.set_facecolor('#f8f9fa')
    
    # 数值标签
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.2f}', ha='center', va='bottom', 
               fontsize=12, fontweight='bold')
    
    # 变异度
    cv = (values.std() / values.mean() * 100) if values.mean() > 0 else 0
    ax.text(0.98, 0.98, f'CV: {cv:.1f}%', 
           transform=ax.transAxes, ha='right', va='top',
           fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7, 
                    edgecolor='darkgreen', linewidth=2))

plt.suptitle('Feature Comparison (Cross-Attention, 4 Key Features)', 
            fontsize=18, fontweight='bold', y=0.98)
plt.tight_layout()
output_bars = './results/cluster_bars_crossattn_spread.png'
plt.savefig(output_bars, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Saved: {output_bars}")

# ==================== 聚类解释 ====================
print("\n" + "="*70)
print("💡 Cluster Interpretation (Based on 4 Key Features)")
print("="*70)

for i in range(4):
    print(f"\n🔷 Cluster {i}:")
    row = df.loc[f'Cluster {i}']
    
    # 计算每个特征的相对排名
    for feat in features_to_use:
        col = df[feat].values
        value = row[feat]
        rank = np.sum(col >= value)  # 从大到小的排名
        
        if rank == 1:
            emoji = "🥇"
        elif rank == 2:
            emoji = "🥈"
        elif rank == 3:
            emoji = "🥉"
        else:
            emoji = "4️⃣"
        
        # 计算相对百分位
        percentile = (len(col) - rank + 1) / len(col) * 100
        
        print(f"   {emoji} {feat}: {value:.2f} (rank #{rank}, top {percentile:.0f}%)")

# 保存CSV
df.to_csv('./results/cluster_features_crossattn_4features.csv', encoding='utf-8-sig')
print(f"\n💾 Saved: ./results/cluster_features_crossattn_4features.csv")

print("\n" + "="*70)
print("✅ Visualization Complete!")
print("="*70)
print("\n📁 Generated files:")
print(f"   1. {output} - Radar chart (spread out)")
print(f"   2. {output_bars} - Bar charts")
print(f"   3. cluster_features_crossattn_4features.csv")
print("\n💡 Key improvements:")
print("   ✅ Removed low-variance features (Speed Std, Accel Std)")
print("   ✅ Used Z-score normalization for better spread")
print("   ✅ Mapped to [0.2, 1.0] range to amplify differences")
print("="*70)