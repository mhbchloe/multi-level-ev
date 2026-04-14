"""
绘制K=4雷达图 - 只保留有区分度的特征
移除：Speed Std, Accel Std (差异太小)
保留：Avg Speed, Max Speed, Avg Power, Trip Length
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("🎨 Drawing Radar Chart - Selected Features Only")
print("="*70)

# 数据（根据你提供的）
data = {
    'Cluster 0': {
        'Avg Speed': 0.88,
        'Max Speed': 3.73,
        'Avg Power': 1.69,
        'Trip Length': 109.56
    },
    'Cluster 1': {
        'Avg Speed': 0.88,
        'Max Speed': 3.74,
        'Avg Power': 1.67,
        'Trip Length': 109.68
    },
    'Cluster 2': {
        'Avg Speed': 0.89,
        'Max Speed': 3.72,
        'Avg Power': 1.69,
        'Trip Length': 108.09
    },
    'Cluster 3': {
        'Avg Speed': 0.88,
        'Max Speed': 3.73,
        'Avg Power': 1.66,
        'Trip Length': 109.36
    }
}

# 只保留有区分度的特征
features = ['Avg Speed', 'Max Speed', 'Avg Power', 'Trip Length']

print(f"\n✅ Selected features: {features}")

# 构建数据矩阵
matrix = []
for cluster in ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3']:
    row = [data[cluster][f] for f in features]
    matrix.append(row)

matrix = np.array(matrix)

# 数据诊断
print("\n📊 Feature variation analysis:")
for j, feat in enumerate(features):
    col = matrix[:, j]
    variation = (col.max() - col.min()) / col.mean() * 100
    print(f"  {feat:15s}: range=[{col.min():.3f}, {col.max():.3f}], variation={variation:.2f}%")

# Min-Max归一化，保留相对差异
print("\n🔧 Normalizing data...")

data_normalized = np.zeros_like(matrix)
for j in range(matrix.shape[1]):
    col = matrix[:, j]
    col_min = col.min()
    col_max = col.max()
    
    if col_max - col_min > 1e-6:
        normalized = (col - col_min) / (col_max - col_min)
        # 如果变异<5%，将[0,1]映射到[0.3,1.0]放大差异
        if (col_max - col_min) / col.mean() < 0.05:
            normalized = 0.3 + normalized * 0.7
    else:
        normalized = np.ones(len(col)) * 0.5
    
    data_normalized[:, j] = normalized

print("\nNormalized values:")
for i in range(4):
    print(f"Cluster {i}: {data_normalized[i]}")

# 绘制雷达图
N = len(features)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig = plt.figure(figsize=(14, 14))
ax = plt.subplot(111, projection='polar')

# 背景设置
ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('white')

# 配色和样式
colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']  # 红、蓝、绿、橙
cluster_names = ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3']
markers = ['o', 's', '^', 'D']
line_styles = ['-', '--', '-.', ':']

# 绘制每个簇
for i in range(4):
    values = data_normalized[i].tolist()
    values += values[:1]
    
    ax.plot(angles, values, 
           linewidth=4, 
           linestyle=line_styles[i],
           color=colors[i], 
           marker=markers[i],
           markersize=14,
           markeredgecolor='white',
           markeredgewidth=2.5,
           label=cluster_names[i],
           zorder=10)
    
    ax.fill(angles, values, alpha=0.18, color=colors[i], zorder=5)

# 设置特征标签（加大字体，因为特征少了）
ax.set_xticks(angles[:-1])
ax.set_xticklabels(features, fontsize=15, fontweight='bold', color='#2c3e50')

# 设置径向刻度
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], 
                   fontsize=12, color='#7f8c8d', fontweight='bold')

# 网格
ax.grid(True, linestyle='--', linewidth=1.5, alpha=0.4, color='#34495e')

# 背景圆环
for y in [0.2, 0.4, 0.6, 0.8, 1.0]:
    ax.plot(angles, [y] * len(angles), 'k-', linewidth=0.8, alpha=0.2)

# 标题
ax.set_title('Driving Behavior Clustering (K=4)\nKey Feature Comparison', 
            fontsize=22, fontweight='bold', pad=45, color='#2c3e50')

# 图例
legend = ax.legend(loc='upper right', 
                  bbox_to_anchor=(1.4, 1.15),
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
output = './results/cluster_radar_k4_selected.png'
plt.savefig(output, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"\n✅ Radar chart saved: {output}")

# 生成对比柱状图（2x2布局，因为只有4个特征）
print("\n📊 Generating feature bars...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, feat in enumerate(features):
    ax = axes[idx]
    
    values = [data[f'Cluster {i}'][feat] for i in range(4)]
    x = np.arange(4)
    
    bars = ax.bar(x, values, color=colors, alpha=0.85, 
                  edgecolor='black', linewidth=2.5, width=0.65)
    
    ax.set_xlabel('Cluster', fontsize=13, fontweight='bold')
    ax.set_ylabel(feat, fontsize=13, fontweight='bold')
    ax.set_title(feat, fontsize=15, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(['C0', 'C1', 'C2', 'C3'], fontweight='bold', fontsize=12)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1.2)
    ax.set_facecolor('#f8f9fa')
    
    # 数值标签
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.2f}', ha='center', va='bottom', 
               fontsize=12, fontweight='bold')
    
    # 变异度标签
    variation = (max(values) - min(values)) / np.mean(values) * 100
    ax.text(0.98, 0.98, f'Var: {variation:.1f}%', 
           transform=ax.transAxes, ha='right', va='top',
           fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6, edgecolor='orange', linewidth=2))

plt.suptitle('Key Feature Comparison (Selected Features)', 
            fontsize=18, fontweight='bold', y=0.98)
plt.tight_layout()
output2 = './results/cluster_features_bars_selected.png'
plt.savefig(output2, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Bar charts saved: {output2}")

# 生成聚类特征解释
print("\n" + "="*70)
print("💡 Cluster Interpretation (Based on Key Features)")
print("="*70)

# 找出每个簇的突出特征
for i in range(4):
    print(f"\n🔷 Cluster {i}:")
    
    # 归一化值（相对于所有簇）
    cluster_profile = []
    for j, feat in enumerate(features):
        col = matrix[:, j]
        value = matrix[i, j]
        # 计算相对位置（0=最小，1=最大）
        if col.max() - col.min() > 1e-6:
            rel_pos = (value - col.min()) / (col.max() - col.min())
        else:
            rel_pos = 0.5
        cluster_profile.append((feat, value, rel_pos))
    
    # 按相对位置排序，找出最突出的特征
    cluster_profile.sort(key=lambda x: abs(x[2] - 0.5), reverse=True)
    
    # 打印top 2特征
    for feat, value, rel_pos in cluster_profile[:2]:
        if rel_pos > 0.7:
            print(f"   ✨ HIGH {feat}: {value:.2f} (top {rel_pos*100:.0f}%)")
        elif rel_pos < 0.3:
            print(f"   📉 LOW {feat}: {value:.2f} (bottom {(1-rel_pos)*100:.0f}%)")
        else:
            print(f"   ➡️  MEDIUM {feat}: {value:.2f}")

print("\n" + "="*70)
print("✅ Visualization complete!")
print("="*70)
print(f"\n📁 Output files:")
print(f"   1. {output} - Radar chart (4 features)")
print(f"   2. {output2} - Bar charts")
print(f"\n💡 Removed features with low variation:")
print(f"   ❌ Speed Std - All clusters nearly identical")
print(f"   ❌ Accel Std - All clusters identical")
print("="*70)