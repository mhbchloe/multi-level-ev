"""
快速修复：只重新绘制雷达图
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'DejaVu Sans'

# 加载已有结果
df = pd.read_csv('./results/cluster_features_k4.csv', index_col=0)

print("原始数据:")
print(df)

# 选择特征
features = ['Avg Speed', 'Speed Std', 'High Speed Ratio', 'Idle Ratio',
           'Accel Std', 'Harsh Accel %', 'Harsh Decel %',
           'SOC Drop Rate', 'Avg Power', 'Trip Length']

data = df[features].values

# 检查数据
print("\n数据范围:")
for i, f in enumerate(features):
    print(f"{f:20s}: {data[:, i].min():.2f} - {data[:, i].max():.2f}")

# 归一化（每个特征除以最大值）
data_norm = np.zeros_like(data)
for j in range(data.shape[1]):
    max_val = data[:, j].max()
    if max_val > 0:
        data_norm[:, j] = data[:, j] / max_val

print("\n归一化后:")
print(data_norm)

# 绘制
N = len(features)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(14, 14), subplot_kw=dict(projection='polar'))

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
labels = ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3']

for i in range(4):
    values = data_norm[i].tolist()
    values += values[:1]
    
    ax.plot(angles, values, 'o-', linewidth=3, 
           label=labels[i], color=colors[i], markersize=10)
    ax.fill(angles, values, alpha=0.2, color=colors[i])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(features, fontsize=12, fontweight='bold')
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.grid(True, linestyle='--', alpha=0.7)

ax.set_title('Driving Behavior Clustering (K=4) - Fixed', 
            fontsize=18, fontweight='bold', pad=35)
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=13)

plt.tight_layout()
plt.savefig('./results/cluster_radar_k4_FIXED.png', dpi=300, bbox_inches='tight')
print("\n✅ Fixed radar saved!")
plt.show()