"""
使用Cross-Attention模型结果绘制6指标雷达图
包含：Avg Speed, Max Speed, Speed Std, Accel Std, Avg Power, Trip Length
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("🎨 Drawing 6-Feature Radar Chart (Cross-Attention)")
print("="*70)

# ==================== 加载Cross-Attention的聚类结果 ====================
print("\n📂 Loading Cross-Attention clustering results...")

labels_path = Path('./results/labels_k4_crossattn.npy')
driving_path = Path('./results/temporal_soc_full/driving_sequences.npy')
energy_path = Path('./results/temporal_soc_full/energy_sequences.npy')

if not all([labels_path.exists(), driving_path.exists(), energy_path.exists()]):
    print("❌ Required files not found. Please run gru_clustering_k4_crossattn.py first")
    exit(1)

labels = np.load(labels_path)
driving_seqs = np.load(driving_path, allow_pickle=True)
energy_seqs = np.load(energy_path, allow_pickle=True)

# 确保长度匹配
min_len = min(len(labels), len(driving_seqs), len(energy_seqs))
labels = labels[:min_len]
driving_seqs = driving_seqs[:min_len]
energy_seqs = energy_seqs[:min_len]

print(f"✅ Data loaded:")
print(f"   Samples: {len(labels):,}")
print(f"   Clusters: {len(np.unique(labels))}")

# ==================== 提取6个关键特征 ====================
print("\n📊 Extracting 6 key features for each cluster...")

features_to_extract = [
    'Avg Speed',
    'Max Speed', 
    'Speed Std',
    'Accel Std',
    'Avg Power',
    'Trip Length'
]

cluster_stats = []

for cluster_id in range(4):
    print(f"\n  Analyzing Cluster {cluster_id}...")
    
    cluster_mask = (labels == cluster_id)
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
    all_v = np.concatenate([seq[:, 1] for seq in cluster_energy])
    all_i = np.concatenate([seq[:, 2] for seq in cluster_energy])
    
    stats['Avg Power'] = np.mean(np.abs(all_v * all_i))
    stats['Trip Length'] = np.mean([len(seq) for seq in cluster_driving])
    
    cluster_stats.append(stats)
    
    # 打印
    print(f"     Avg Speed: {stats['Avg Speed']:.2f}")
    print(f"     Max Speed: {stats['Max Speed']:.2f}")
    print(f"     Speed Std: {stats['Speed Std']:.2f}")
    print(f"     Accel Std: {stats['Accel Std']:.2f}")
    print(f"     Avg Power: {stats['Avg Power']:.2f}")
    print(f"     Trip Length: {stats['Trip Length']:.0f}")

# 转为DataFrame
df = pd.DataFrame(cluster_stats, index=['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3'])

print("\n" + "="*70)
print("📊 Cluster Features Summary")
print("="*70)
print(df.round(2))

# 保存
df.to_csv('./results/cluster_features_k4_crossattn_6features.csv', encoding='utf-8-sig')
print(f"\n💾 Saved: ./results/cluster_features_k4_crossattn_6features.csv")

# ==================== 数据诊断 ====================
print("\n" + "="*70)
print("🔍 Feature Variation Analysis")
print("="*70)

for feat in features_to_extract:
    values = df[feat].values
    mean_val = values.mean()
    std_val = values.std()
    min_val = values.min()
    max_val = values.max()
    cv = (std_val / mean_val * 100) if mean_val > 0 else 0
    
    print(f"\n{feat}:")
    print(f"   Range: [{min_val:.2f}, {max_val:.2f}]")
    print(f"   Mean: {mean_val:.2f}, Std: {std_val:.2f}")
    print(f"   CV: {cv:.1f}%")

# ==================== 归一化策略 ====================
print("\n" + "="*70)
print("🔧 Normalization Strategy")
print("="*70)

data = df[features_to_extract].values

# 方法1：Min-Max归一化（每个特征独立）
data_norm_minmax = np.zeros_like(data, dtype=float)
for j in range(data.shape[1]):
    col = data[:, j]
    col_min = col.min()
    col_max = col.max()
    
    if col_max - col_min > 1e-6:
        data_norm_minmax[:, j] = (col - col_min) / (col_max - col_min)
    else:
        data_norm_minmax[:, j] = 0.5

print("\nMethod 1: Min-Max Normalization")
for i in range(4):
    print(f"Cluster {i}: {data_norm_minmax[i]}")

# 方法2：除以最大值（放大小差异）
data_norm_max = np.zeros_like(data, dtype=float)
for j in range(data.shape[1]):
    col = data[:, j]
    max_val = col.max()
    
    if max_val > 1e-6:
        normalized = col / max_val
        # 如果变异<5%，映射到[0.3, 1.0]
        if (col.max() - col.min()) / col.mean() < 0.05:
            normalized = 0.3 + normalized * 0.7
    else:
        normalized = np.ones(len(col)) * 0.5
    
    data_norm_max[:, j] = normalized

print("\nMethod 2: Max Normalization (amplified)")
for i in range(4):
    print(f"Cluster {i}: {data_norm_max[i]}")

# ==================== 绘制雷达图 ====================
print("\n" + "="*70)
print("🎨 Plotting Radar Charts")
print("="*70)

N = len(features_to_extract)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
cluster_names = ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3']
markers = ['o', 's', '^', 'D']
line_styles = ['-', '--', '-.', ':']

# ==================== 图1：Min-Max归一化 ====================
fig = plt.figure(figsize=(15, 15))
ax = plt.subplot(111, projection='polar')

ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('white')

for i in range(4):
    values = data_norm_minmax[i].tolist()
    values += values[:1]
    
    ax.plot(angles, values, 
           marker=markers[i],
           linewidth=4, 
           linestyle=line_styles[i],
           color=colors[i], 
           markersize=14,
           markeredgecolor='white',
           markeredgewidth=2.5,
           label=cluster_names[i],
           zorder=10)
    
    ax.fill(angles, values, alpha=0.15, color=colors[i], zorder=5)

# 设置标签
ax.set_xticks(angles[:-1])
ax.set_xticklabels(features_to_extract, fontsize=14, fontweight='bold', color='#2c3e50')

# 径向刻度
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], 
                   fontsize=12, color='#7f8c8d', fontweight='bold')

# 网格
ax.grid(True, linestyle='--', linewidth=1.8, alpha=0.5, color='#34495e')

# 背景圆环
for y in [0.2, 0.4, 0.6, 0.8, 1.0]:
    ax.plot(angles, [y] * len(angles), 'k-', linewidth=1, alpha=0.25)

# 标题
ax.set_title('Driving Behavior Clustering (K=4)\nCross-Attention Model - 6 Key Features', 
            fontsize=20, fontweight='bold', pad=45, color='#2c3e50')

# 图例
legend = ax.legend(loc='upper right', 
                  bbox_to_anchor=(1.4, 1.15),
                  fontsize=14,
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
output1 = './results/cluster_radar_k4_crossattn_6features_v1.png'
plt.savefig(output1, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"✅ Saved: {output1}")

# ==================== 图2：放大差异版本 ====================
fig = plt.figure(figsize=(15, 15))
ax = plt.subplot(111, projection='polar')

ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('white')

for i in range(4):
    values = data_norm_max[i].tolist()
    values += values[:1]
    
    ax.plot(angles, values, 
           marker=markers[i],
           linewidth=4, 
           linestyle=line_styles[i],
           color=colors[i], 
           markersize=14,
           markeredgecolor='white',
           markeredgewidth=2.5,
           label=cluster_names[i],
           zorder=10)
    
    ax.fill(angles, values, alpha=0.15, color=colors[i], zorder=5)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(features_to_extract, fontsize=14, fontweight='bold', color='#2c3e50')

ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], 
                   fontsize=12, color='#7f8c8d', fontweight='bold')

ax.grid(True, linestyle='--', linewidth=1.8, alpha=0.5, color='#34495e')

for y in [0.2, 0.4, 0.6, 0.8, 1.0]:
    ax.plot(angles, [y] * len(angles), 'k-', linewidth=1, alpha=0.25)

ax.set_title('Driving Behavior Clustering (K=4)\nCross-Attention - Amplified Differences', 
            fontsize=20, fontweight='bold', pad=45, color='#2c3e50')

legend = ax.legend(loc='upper right', 
                  bbox_to_anchor=(1.4, 1.15),
                  fontsize=14,
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
output2 = './results/cluster_radar_k4_crossattn_6features_v2.png'
plt.savefig(output2, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"✅ Saved: {output2}")

# ==================== 绘制特征柱状图对比 ====================
print("\n📊 Plotting feature bar charts...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, feat in enumerate(features_to_extract):
    ax = axes[idx]
    
    values = df[feat].values
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
    
    # 变异度
    variation = (values.max() - values.min()) / values.mean() * 100 if values.mean() > 0 else 0
    ax.text(0.98, 0.98, f'Var: {variation:.1f}%', 
           transform=ax.transAxes, ha='right', va='top',
           fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6, 
                    edgecolor='blue', linewidth=2))

plt.suptitle('Cluster Feature Comparison (Cross-Attention, K=4)', 
            fontsize=18, fontweight='bold', y=0.98)
plt.tight_layout()
output3 = './results/cluster_features_bars_crossattn_6features.png'
plt.savefig(output3, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Saved: {output3}")

# ==================== 生成特征解释 ====================
print("\n" + "="*70)
print("💡 Cluster Interpretation (Cross-Attention, 6 Features)")
print("="*70)

for i in range(4):
    print(f"\n🔷 Cluster {i}:")
    
    # 找出突出特征
    row = df.loc[f'Cluster {i}']
    
    # 计算相对位置
    for feat in features_to_extract:
        col = df[feat].values
        value = row[feat]
        
        # 标准化到0-1
        if col.max() - col.min() > 1e-6:
            rel_pos = (value - col.min()) / (col.max() - col.min())
        else:
            rel_pos = 0.5
        
        if rel_pos > 0.75:
            print(f"   ✨ HIGH {feat}: {value:.2f} (rank #{np.sum(col >= value)})")
        elif rel_pos < 0.25:
            print(f"   📉 LOW {feat}: {value:.2f} (rank #{4 - np.sum(col <= value) + 1})")

print("\n" + "="*70)
print("✅ All visualizations completed!")
print("="*70)
print("\n📁 Generated files:")
print(f"   1. {output1}")
print(f"   2. {output2}")
print(f"   3. {output3}")
print(f"   4. cluster_features_k4_crossattn_6features.csv")
print("\n💡 Version 1: Standard min-max normalization")
print("💡 Version 2: Amplified differences for better visualization")
print("="*70)