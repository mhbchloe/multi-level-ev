"""
Cross-Attention模型 - 6特征雷达图（低变异特征差异放大版）
特殊处理Speed Std和Accel Std，使微小差异在视觉上明显
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("🎨 Radar Chart - 6 Features with Amplified Differences")
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

# ==================== 提取6个特征 ====================
print("\n📊 Extracting 6 features...")

features_to_use = [
    'Avg Speed',
    'Max Speed',
    'Speed Std',
    'Accel Std',
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
    all_acc = np.concatenate([seq[:, 1] for seq in cluster_driving])
    
    stats['Avg Speed'] = np.mean(all_spd)
    stats['Max Speed'] = np.percentile(all_spd, 95)
    stats['Speed Std'] = np.std(all_spd)
    stats['Accel Std'] = np.std(all_acc)
    
    # 能量特征
    all_v = np.concatenate([seq[:, 1] for seq in cluster_energy])
    all_i = np.concatenate([seq[:, 2] for seq in cluster_energy])
    stats['Avg Power'] = np.mean(np.abs(all_v * all_i))
    
    # 行程特征
    stats['Trip Length'] = np.mean([len(seq) for seq in cluster_driving])
    
    cluster_stats.append(stats)

df = pd.DataFrame(cluster_stats, index=['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3'])

print("\n📊 Raw feature values:")
print(df[features_to_use].round(4))

# ==================== 诊断特征变异 ====================
print("\n🔍 Feature variation analysis:")

low_variance_features = []
high_variance_features = []

for feat in features_to_use:
    values = df[feat].values
    mean_val = values.mean()
    std_val = values.std()
    cv = (std_val / mean_val * 100) if mean_val > 0 else 0
    
    print(f"\n{feat}:")
    print(f"   Range: [{values.min():.4f}, {values.max():.4f}]")
    print(f"   Mean: {mean_val:.4f}, Std: {std_val:.4f}")
    print(f"   CV: {cv:.2f}%")
    
    if cv < 5:
        print(f"   ⚠️  LOW VARIANCE - Will apply amplification")
        low_variance_features.append(feat)
    else:
        print(f"   ✅ NORMAL VARIANCE")
        high_variance_features.append(feat)

# ==================== 智能归一化策略 ====================
print("\n" + "="*70)
print("🔧 Applying Smart Normalization Strategy")
print("="*70)

data = df[features_to_use].values
data_normalized = np.zeros_like(data, dtype=float)

for j, feat in enumerate(features_to_use):
    col = data[:, j]
    col_min = col.min()
    col_max = col.max()
    mean_val = col.mean()
    
    # 计算变异系数
    cv = (col.std() / mean_val * 100) if mean_val > 0 else 0
    
    if cv < 5:  # 低变异特征 - 激进放大
        print(f"\n📌 {feat} (CV={cv:.2f}%) - LOW VARIANCE")
        
        if col_max - col_min > 1e-9:
            # 步骤1：基础归一化到[0, 1]
            normalized = (col - col_min) / (col_max - col_min)
            
            # 步骤2：指数放大（使用平方根的反函数）
            # 将[0, 1]映射到更宽的范围，放大差异
            # 使用分段线性映射：最小值→0.2，最大值→1.0
            normalized = 0.2 + normalized * 0.8
            
            # 步骤3：进一步放大差异（对于极低变异）
            if cv < 2:
                # 使用非线性变换进一步拉开差异
                # 中间值压缩，两端拉伸
                center = 0.6
                normalized = np.where(
                    normalized < center,
                    0.2 + (normalized - 0.2) * 0.5,  # 下半部分压缩
                    center + (normalized - center) * 1.5  # 上半部分拉伸
                )
                normalized = np.clip(normalized, 0.2, 1.0)
            
            print(f"   Original range: [{col_min:.6f}, {col_max:.6f}]")
            print(f"   Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")
            print(f"   Spread: {(normalized.max() - normalized.min()):.3f}")
        else:
            normalized = np.ones(len(col)) * 0.6
        
    else:  # 正常变异特征 - 标准处理
        print(f"\n📌 {feat} (CV={cv:.2f}%) - NORMAL VARIANCE")
        
        # Z-score标准化后映射
        mean_val = col.mean()
        std_val = col.std()
        
        if std_val > 1e-6:
            z_scores = (col - mean_val) / std_val
            z_min = z_scores.min()
            z_max = z_scores.max()
            
            if z_max - z_min > 1e-6:
                normalized = (z_scores - z_min) / (z_max - z_min)
                normalized = 0.2 + normalized * 0.8
            else:
                normalized = np.ones(len(col)) * 0.6
        else:
            normalized = np.ones(len(col)) * 0.6
        
        print(f"   Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")
        print(f"   Spread: {(normalized.max() - normalized.min()):.3f}")
    
    data_normalized[:, j] = normalized

print("\n" + "="*70)
print("📊 Final Normalized Values")
print("="*70)
for i in range(4):
    print(f"Cluster {i}: {data_normalized[i].round(3)}")

# ==================== 绘制雷达图 ====================
print("\n🎨 Drawing radar chart...")

N = len(features_to_use)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig = plt.figure(figsize=(16, 16))
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
    
    ax.fill(angles, values, alpha=0.15, color=colors[i], zorder=5)

# 设置标签
ax.set_xticks(angles[:-1])
ax.set_xticklabels(features_to_use, fontsize=15, fontweight='bold', color='#2c3e50')

# 径向刻度
ax.set_ylim(0, 1.1)  # 稍微扩大一点范围
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], 
                   fontsize=13, color='#7f8c8d', fontweight='bold')

# 网格
ax.grid(True, linestyle='--', linewidth=2, alpha=0.6, color='#34495e')

# 背景圆环
for y in [0.2, 0.4, 0.6, 0.8, 1.0]:
    ax.plot(angles, [y] * len(angles), 'k-', linewidth=1.2, alpha=0.3)

# 标题
title_text = 'Driving Behavior Clustering (K=4)\nCross-Attention - 6 Features (Amplified Differences)'
ax.set_title(title_text, fontsize=20, fontweight='bold', pad=50, color='#2c3e50')

# 添加说明文字
note_text = "Note: Speed Std & Accel Std differences amplified for visualization"
fig.text(0.5, 0.02, note_text, ha='center', fontsize=11, 
        style='italic', color='gray', bbox=dict(boxstyle='round', 
        facecolor='wheat', alpha=0.3))

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
output = './results/cluster_radar_k4_crossattn_6features_amplified.png'
plt.savefig(output, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"\n✅ Saved: {output}")

# ==================== 绘制对比图：原始值 vs 归一化值 ====================
print("\n📊 Drawing comparison charts...")

fig, axes = plt.subplots(3, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, feat in enumerate(features_to_use):
    ax = axes[idx]
    
    # 原始值
    values_raw = df[feat].values
    x = np.arange(4)
    
    # 归一化值（用于显示差异）
    values_norm = data_normalized[:, idx]
    
    # 双Y轴
    ax2 = ax.twinx()
    
    # 左轴：原始值（柱状图）
    bars = ax.bar(x - 0.2, values_raw, width=0.4, color=colors, alpha=0.7, 
                  edgecolor='black', linewidth=2, label='Raw Value')
    
    # 右轴：归一化值（折线图）
    line = ax2.plot(x + 0.2, values_norm, 'ro-', linewidth=3, 
                    markersize=10, markeredgecolor='white', 
                    markeredgewidth=2, label='Normalized', zorder=10)
    
    ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{feat} (Raw)', fontsize=12, fontweight='bold', color='blue')
    ax2.set_ylabel('Normalized [0-1]', fontsize=12, fontweight='bold', color='red')
    
    # 检查是否是低变异特征
    cv = (values_raw.std() / values_raw.mean() * 100) if values_raw.mean() > 0 else 0
    title_color = 'darkred' if cv < 5 else 'black'
    title_suffix = ' (AMPLIFIED)' if cv < 5 else ''
    
    ax.set_title(f'{feat}{title_suffix}', fontsize=14, fontweight='bold', 
                color=title_color, pad=12)
    
    ax.set_xticks(x)
    ax.set_xticklabels(['C0', 'C1', 'C2', 'C3'], fontweight='bold', fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_facecolor('#f8f9fa')
    
    # 设置Y轴范围
    ax2.set_ylim(0, 1.1)
    
    # 添加CV标签
    ax.text(0.02, 0.98, f'CV: {cv:.2f}%', 
           transform=ax.transAxes, ha='left', va='top',
           fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round', 
                    facecolor='yellow' if cv < 5 else 'lightblue', 
                    alpha=0.7, edgecolor='red' if cv < 5 else 'blue', 
                    linewidth=2))
    
    # 数值标签
    for i, (bar, val_raw, val_norm) in enumerate(zip(bars, values_raw, values_norm)):
        # 原始值标签
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val_raw:.3f}', ha='center', va='bottom', 
               fontsize=9, fontweight='bold', color='blue')

plt.suptitle('Feature Comparison: Raw vs Normalized (Amplified)', 
            fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
output_compare = './results/cluster_comparison_amplified.png'
plt.savefig(output_compare, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Saved: {output_compare}")

# ==================== 保存结果 ====================
df.to_csv('./results/cluster_features_k4_crossattn_6features_amplified.csv', 
         encoding='utf-8-sig')
print(f"✅ Saved: cluster_features_k4_crossattn_6features_amplified.csv")

# 保存归一化后的值
df_normalized = pd.DataFrame(data_normalized, 
                            columns=features_to_use,
                            index=['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3'])
df_normalized.to_csv('./results/cluster_features_normalized_amplified.csv', 
                    encoding='utf-8-sig')
print(f"✅ Saved: cluster_features_normalized_amplified.csv")

print("\n" + "="*70)
print("✅ Visualization Complete!")
print("="*70)
print("\n📁 Generated files:")
print(f"   1. {output}")
print(f"   2. {output_compare}")
print(f"   3. cluster_features_k4_crossattn_6features_amplified.csv")
print(f"   4. cluster_features_normalized_amplified.csv")
print("\n💡 Key features:")
print(f"   ✅ Low variance features (CV<5%) are amplified")
print(f"   ✅ Speed Std & Accel Std differences are now visible")
print(f"   ✅ Original values preserved in comparison charts")
print("="*70)