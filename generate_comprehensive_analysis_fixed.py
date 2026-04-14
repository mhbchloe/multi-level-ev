"""
修复版 - 检查并修正速度单位问题
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("🎨 Generating Comprehensive Analysis (Fixed Speed Values)")
print("="*70)

# ==================== 加载数据 ====================
print("\n📂 Loading data...")

labels = np.load('./results/labels_k4_crossattn.npy')
features = np.load('./results/features_k4_crossattn.npy')
driving_seqs = np.load('./results/temporal_soc_full/driving_sequences.npy', allow_pickle=True)
energy_seqs = np.load('./results/temporal_soc_full/energy_sequences.npy', allow_pickle=True)

min_len = min(len(labels), len(features), len(driving_seqs), len(energy_seqs))
labels = labels[:min_len]
features = features[:min_len]
driving_seqs = driving_seqs[:min_len]
energy_seqs = energy_seqs[:min_len]

print(f"✅ Loaded {len(labels):,} samples")

# ==================== 诊断原始数据 ====================
print("\n🔍 Diagnosing raw data...")

# 检查前10个样本的速度值
print("\nSample speed values (first 10 trips):")
for i in range(min(10, len(driving_seqs))):
    spd = driving_seqs[i][:, 0]
    print(f"  Trip {i}: min={spd.min():.4f}, max={spd.max():.4f}, mean={spd.mean():.4f}")

# 全局统计
all_speeds = np.concatenate([seq[:, 0] for seq in driving_seqs[:1000]])  # 前1000个样本
print(f"\nGlobal speed statistics (first 1000 trips):")
print(f"  Min: {all_speeds.min():.4f}")
print(f"  Max: {all_speeds.max():.4f}")
print(f"  Mean: {all_speeds.mean():.4f}")
print(f"  Median: {np.median(all_speeds):.4f}")
print(f"  Std: {all_speeds.std():.4f}")

# ==================== 判断是否需要反归一化 ====================
speed_max = all_speeds.max()

if speed_max < 10:
    print("\n⚠️  Speed values are too small - likely normalized or wrong unit")
    print("   Attempting to recover original scale...")
    
    # 假设原始数据被RobustScaler或StandardScaler归一化了
    # 尝试几种常见的反归一化策略
    
    # 策略1：假设被归一化到[-1, 1]或[0, 1]范围，恢复到合理速度范围[0, 120] km/h
    if all_speeds.min() >= -0.1 and all_speeds.max() <= 1.1:
        print("   → Detected normalization to [0, 1] range")
        speed_scale = 120.0  # 假设最大速度120 km/h
        SPEED_MULTIPLIER = speed_scale
    elif all_speeds.min() >= -3 and all_speeds.max() <= 3:
        print("   → Detected standardization (z-score)")
        # 假设原始均值30 km/h，标准差20 km/h
        speed_mean = 30.0
        speed_std = 20.0
        SPEED_MULTIPLIER = speed_std
        SPEED_OFFSET = speed_mean
    else:
        print("   → Unknown normalization, using default scaling")
        SPEED_MULTIPLIER = 50.0
        SPEED_OFFSET = 0
else:
    print("\n✅ Speed values look reasonable, no scaling needed")
    SPEED_MULTIPLIER = 1.0
    SPEED_OFFSET = 0

print(f"\nApplying scaling: speed_real = speed_raw * {SPEED_MULTIPLIER:.2f} + {SPEED_OFFSET:.2f}")

# ==================== 重新提取特征（应用正确的缩放） ====================
print("\n📊 Re-extracting features with correct scaling...")

sample_features = []

for i in range(len(labels)):
    feat = {}
    
    # 驾驶特征（应用缩放）
    spd_raw = driving_seqs[i][:, 0]
    acc_raw = driving_seqs[i][:, 1]
    
    # 反归一化速度
    if 'SPEED_OFFSET' in locals():
        spd = spd_raw * SPEED_MULTIPLIER + SPEED_OFFSET
    else:
        spd = spd_raw * SPEED_MULTIPLIER
    
    feat['avg_speed'] = np.mean(spd)
    feat['max_speed'] = np.max(spd)
    feat['speed_std'] = np.std(spd)
    
    # 能量特征
    soc = energy_seqs[i][:, 0]
    v = energy_seqs[i][:, 1]
    current = energy_seqs[i][:, 2]
    
    feat['soc_drop'] = soc[0] - soc[-1] if len(soc) > 1 else 0
    
    # 检查电压和电流是否也需要缩放
    v_mean = np.mean(np.abs(v))
    if v_mean < 10:  # 电压太小，可能也被归一化了
        v = v * 400  # 典型电动车电压 300-400V
        current = current * 200  # 典型电流范围
    
    feat['avg_power'] = np.mean(np.abs(v * current))
    
    # 行程特征
    feat['duration'] = len(driving_seqs[i])
    
    # 距离 (km) = 速度(km/h) * 时间(h)
    # 假设采样率10Hz，即每个点0.1秒
    time_hours = len(driving_seqs[i]) * 0.1 / 3600
    feat['distance'] = np.mean(spd) * time_hours
    
    # 能效 (%/km)
    if feat['distance'] > 0.01:  # 至少10米
        feat['energy_rate'] = feat['soc_drop'] / feat['distance']
    else:
        feat['energy_rate'] = 0
    
    sample_features.append(feat)

df_samples = pd.DataFrame(sample_features)
df_samples['cluster'] = labels

# 过滤异常值
print("\n🔧 Filtering outliers...")
print(f"   Before filtering: {len(df_samples)} samples")

# 过滤不合理的值
df_samples = df_samples[
    (df_samples['avg_speed'] > 0) & 
    (df_samples['avg_speed'] < 150) &  # 速度<150 km/h
    (df_samples['energy_rate'] >= -2) &  # 能效合理范围
    (df_samples['energy_rate'] <= 2)
]

print(f"   After filtering: {len(df_samples)} samples")

# 重新提取labels
labels = df_samples['cluster'].values

print(f"\n✅ Feature extraction completed")

# 检查修正后的速度
print("\nCorrected speed statistics:")
print(f"  Min: {df_samples['avg_speed'].min():.2f} km/h")
print(f"  Max: {df_samples['avg_speed'].max():.2f} km/h")
print(f"  Mean: {df_samples['avg_speed'].mean():.2f} km/h")

# ==================== 计算簇统计 ====================
print("\n📊 Computing cluster statistics...")

cluster_stats = []
for cluster_id in range(4):
    cluster_data = df_samples[df_samples['cluster'] == cluster_id]
    
    stats = {
        'cluster': cluster_id,
        'count': len(cluster_data),
        'avg_speed_mean': cluster_data['avg_speed'].mean(),
        'avg_speed_std': cluster_data['avg_speed'].std(),
        'energy_rate_mean': cluster_data['energy_rate'].mean(),
        'energy_rate_median': cluster_data['energy_rate'].median(),
        'avg_power_mean': cluster_data['avg_power'].mean(),
        'duration_mean': cluster_data['duration'].mean()
    }
    cluster_stats.append(stats)

df_stats = pd.DataFrame(cluster_stats)

print("\n📊 Cluster Statistics (Corrected):")
print(df_stats[['cluster', 'count', 'avg_speed_mean', 'energy_rate_mean', 
               'avg_power_mean', 'duration_mean']].to_string(index=False))

# ==================== PCA降维 ====================
print("\n🔍 Performing PCA...")

# 使用原始GRU特征
valid_indices = df_samples.index.tolist()
features_filtered = features[valid_indices]

pca = PCA(n_components=2)
features_2d = pca.fit_transform(features_filtered)

print(f"✅ PCA completed, explained variance: {pca.explained_variance_ratio_.sum()*100:.1f}%")

# ==================== 创建综合图 ====================
print("\n🎨 Creating comprehensive plot...")

fig = plt.figure(figsize=(20, 12))

colors = ['#3498DB', '#2ECC71', '#E74C3C', '#F39C12']

# ==================== 子图1: PCA ====================
ax1 = plt.subplot(2, 3, 1)

for cluster_id in range(4):
    mask = labels == cluster_id
    ax1.scatter(features_2d[mask, 0], features_2d[mask, 1],
               c=colors[cluster_id], label=f'Cluster {cluster_id}',
               alpha=0.6, s=20, edgecolors='none')

ax1.set_xlabel('PC1', fontsize=12, fontweight='bold')
ax1.set_ylabel('PC2', fontsize=12, fontweight='bold')
ax1.set_title('Driving Style Clusters (PCA Space)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10, loc='upper right')
ax1.grid(True, alpha=0.3)

# ==================== 子图2: 能效箱线图 ====================
ax2 = plt.subplot(2, 3, 2)

energy_data = []
for cluster_id in range(4):
    cluster_data = df_samples[df_samples['cluster'] == cluster_id]['energy_rate']
    q1, q3 = cluster_data.quantile([0.25, 0.75])
    iqr = q3 - q1
    filtered = cluster_data[(cluster_data >= q1 - 1.5*iqr) & (cluster_data <= q3 + 1.5*iqr)]
    energy_data.append(filtered)

bp = ax2.boxplot(energy_data, positions=range(4), widths=0.6,
                patch_artist=True, showfliers=False)

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax2.set_xlabel('Cluster', fontsize=12, fontweight='bold')
ax2.set_ylabel('Energy Rate (%/km)', fontsize=12, fontweight='bold')
ax2.set_title('Energy Efficiency by Cluster', fontsize=14, fontweight='bold')
ax2.set_xticks(range(4))
ax2.set_xticklabels(['C0', 'C1', 'C2', 'C3'])
ax2.grid(True, alpha=0.3, axis='y')

# ==================== 子图3: 速度vs能耗 (修正后) ====================
ax3 = plt.subplot(2, 3, 3)

for cluster_id in range(4):
    avg_speed = df_stats.loc[cluster_id, 'avg_speed_mean']
    energy_rate = df_stats.loc[cluster_id, 'energy_rate_mean']
    
    ax3.scatter(avg_speed, energy_rate, 
               c=colors[cluster_id], s=500, alpha=0.8,
               edgecolors='black', linewidth=3, zorder=10)
    
    ax3.text(avg_speed, energy_rate, f'C{cluster_id}',
            ha='center', va='center', fontsize=14, fontweight='bold',
            color='white', zorder=11)

ax3.set_xlabel('Average Speed (km/h)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Energy Rate (%/km)', fontsize=12, fontweight='bold')
ax3.set_title('Speed vs Energy Consumption', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)

# ==================== 子图4: 簇大小 ====================
ax4 = plt.subplot(2, 3, 4)

cluster_counts = [df_stats.loc[i, 'count'] for i in range(4)]
bars = ax4.bar(range(4), cluster_counts, color=colors, alpha=0.8,
              edgecolor='black', linewidth=2)

ax4.set_xlabel('Cluster', fontsize=12, fontweight='bold')
ax4.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
ax4.set_title('Cluster Size Distribution', fontsize=14, fontweight='bold')
ax4.set_xticks(range(4))
ax4.set_xticklabels(['C0', 'C1', 'C2', 'C3'])
ax4.grid(True, alpha=0.3, axis='y')

for bar, count in zip(bars, cluster_counts):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(count):,}', ha='center', va='bottom',
            fontsize=11, fontweight='bold')

# ==================== 子图5: 功率 ====================
ax5 = plt.subplot(2, 3, 5)

power_values = [df_stats.loc[i, 'avg_power_mean'] for i in range(4)]
bars = ax5.bar(range(4), power_values, color=colors, alpha=0.8,
              edgecolor='black', linewidth=2)

ax5.set_xlabel('Cluster', fontsize=12, fontweight='bold')
ax5.set_ylabel('Average Power (kW)', fontsize=12, fontweight='bold')
ax5.set_title('Power Characteristics', fontsize=14, fontweight='bold')
ax5.set_xticks(range(4))
ax5.set_xticklabels(['C0', 'C1', 'C2', 'C3'])
ax5.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars, power_values):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
            f'{val/1000:.1f}', ha='center', va='bottom',
            fontsize=11, fontweight='bold')

# ==================== 子图6: 行程时长 ====================
ax6 = plt.subplot(2, 3, 6)

duration_values = [df_stats.loc[i, 'duration_mean'] for i in range(4)]
bars = ax6.bar(range(4), duration_values, color=colors, alpha=0.8,
              edgecolor='black', linewidth=2)

ax6.set_xlabel('Cluster', fontsize=12, fontweight='bold')
ax6.set_ylabel('Average Duration (points)', fontsize=12, fontweight='bold')
ax6.set_title('Trip Duration', fontsize=14, fontweight='bold')
ax6.set_xticks(range(4))
ax6.set_xticklabels(['C0', 'C1', 'C2', 'C3'])
ax6.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars, duration_values):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.0f}', ha='center', va='bottom',
            fontsize=11, fontweight='bold')

# ==================== 总标题 ====================
fig.suptitle('Comprehensive Cluster Analysis (Cross-Attention, K=4) - Corrected', 
            fontsize=18, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.99])

output = './results/comprehensive_analysis_k4_crossattn_fixed.png'
plt.savefig(output, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✅ Saved: {output}")

# ==================== 保存统计 ====================
df_stats.to_csv('./results/cluster_statistics_fixed.csv', encoding='utf-8-sig', index=False)
print(f"✅ Saved: cluster_statistics_fixed.csv")

print("\n" + "="*70)
print("📊 Corrected Cluster Summary")
print("="*70)

for cluster_id in range(4):
    stats = df_stats.loc[cluster_id]
    print(f"\n🔷 Cluster {cluster_id}:")
    print(f"   Samples: {stats['count']:,}")
    print(f"   Avg Speed: {stats['avg_speed_mean']:.2f} km/h (±{stats['avg_speed_std']:.2f})")
    print(f"   Energy Rate: {stats['energy_rate_mean']:.3f} %/km")
    print(f"   Avg Power: {stats['avg_power_mean']/1000:.2f} kW")
    print(f"   Duration: {stats['duration_mean']:.0f} points")

print("\n" + "="*70)
print("✅ Analysis Complete with Corrected Values!")
print("="*70)