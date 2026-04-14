"""
GRU聚类 + 深度物理意义分析
1. 使用GRU提取时序特征聚类
2. 回到原始数据提取统计特征
3. 可视化典型行程轨迹
4. 给出物理解释
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import seaborn as sns

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("🔬 GRU Clustering with Physical Interpretation")
print("="*70)

# ==================== 加载GRU聚类结果 ====================
print("\n📂 Loading GRU clustering results...")

labels = np.load('./results/labels_k4_crossattn.npy')
driving_seqs = np.load('./results/temporal_soc_full/driving_sequences.npy', allow_pickle=True)
energy_seqs = np.load('./results/temporal_soc_full/energy_sequences.npy', allow_pickle=True)

min_len = min(len(labels), len(driving_seqs), len(energy_seqs))
labels = labels[:min_len]
driving_seqs = driving_seqs[:min_len]
energy_seqs = energy_seqs[:min_len]

print(f"✅ Loaded {len(labels):,} samples")
print(f"   Clusters: {np.unique(labels)}")

# ==================== 提取每个簇的统计特征（归一化尺度） ====================
print("\n📊 Extracting statistical features for each cluster...")

cluster_stats = []

for cluster_id in range(4):
    print(f"\n  Analyzing Cluster {cluster_id}...")
    
    cluster_mask = (labels == cluster_id)
    cluster_driving = driving_seqs[cluster_mask]
    cluster_energy = energy_seqs[cluster_mask]
    
    stats = {
        'cluster': cluster_id,
        'count': int(np.sum(cluster_mask))
    }
    
    # 速度特征
    all_spd = np.concatenate([seq[:, 0] for seq in cluster_driving])
    stats['speed_mean'] = np.mean(all_spd)
    stats['speed_max'] = np.max(all_spd)
    stats['speed_p95'] = np.percentile(all_spd, 95)
    stats['speed_std'] = np.std(all_spd)
    stats['speed_min'] = np.min(all_spd)
    
    # 加速度特征
    all_acc = np.concatenate([seq[:, 1] for seq in cluster_driving])
    stats['accel_mean_abs'] = np.mean(np.abs(all_acc))
    stats['accel_std'] = np.std(all_acc)
    stats['accel_positive_ratio'] = np.mean(all_acc > 0.05)  # 加速比例
    stats['accel_negative_ratio'] = np.mean(all_acc < -0.05)  # 减速比例
    
    # 怠速特征
    stats['idle_ratio'] = np.mean(all_spd < 0.1)  # 速度接近0的比例
    stats['low_speed_ratio'] = np.mean(all_spd < 0.5)  # 低速比例
    stats['high_speed_ratio'] = np.mean(all_spd > 2.0)  # 高速比例
    
    # 能量特征
    all_soc = np.concatenate([seq[:, 0] for seq in cluster_energy])
    all_v = np.concatenate([seq[:, 1] for seq in cluster_energy])
    all_i = np.concatenate([seq[:, 2] for seq in cluster_energy])
    
    stats['soc_mean'] = np.mean(all_soc)
    stats['soc_drop_mean'] = np.mean([seq[0, 0] - seq[-1, 0] for seq in cluster_energy if len(seq) > 1])
    stats['soc_drop_rate'] = stats['soc_drop_mean'] / np.mean([len(seq) for seq in cluster_energy])  # 每点下降
    
    stats['voltage_mean'] = np.mean(all_v)
    stats['current_mean'] = np.mean(all_i)
    stats['power_mean'] = np.mean(np.abs(all_v * all_i))
    
    # 行程特征
    trip_lengths = [len(seq) for seq in cluster_driving]
    stats['trip_length_mean'] = np.mean(trip_lengths)
    stats['trip_length_std'] = np.std(trip_lengths)
    
    # 速度变化特征（时序特征）
    speed_changes = []
    for seq in cluster_driving:
        if len(seq) > 1:
            speed_changes.append(np.mean(np.abs(np.diff(seq[:, 0]))))
    stats['speed_change_rate'] = np.mean(speed_changes)
    
    # 加速/减速切换频率
    acc_sign_changes = []
    for seq in cluster_driving:
        if len(seq) > 2:
            acc_signs = np.sign(seq[:, 1])
            changes = np.sum(np.abs(np.diff(acc_signs))) / len(seq)
            acc_sign_changes.append(changes)
    stats['accel_switch_freq'] = np.mean(acc_sign_changes) if acc_sign_changes else 0
    
    cluster_stats.append(stats)
    
    # 打印关键特征
    print(f"     Samples: {stats['count']:,}")
    print(f"     Speed mean: {stats['speed_mean']:.3f}")
    print(f"     Speed max: {stats['speed_max']:.3f}")
    print(f"     Idle ratio: {stats['idle_ratio']*100:.1f}%")
    print(f"     High speed ratio: {stats['high_speed_ratio']*100:.1f}%")
    print(f"     Trip length: {stats['trip_length_mean']:.1f}")

df_stats = pd.DataFrame(cluster_stats)

# 保存
df_stats.to_csv('./results/cluster_physical_features.csv', encoding='utf-8-sig', index=False)
print(f"\n💾 Saved: cluster_physical_features.csv")

# ==================== 颜色映射 ====================
# 根据你的描述：蓝、绿、红、黄
colors = ['#3498DB', '#2ECC71', '#E74C3C', '#F39C12']  # 蓝、绿、红、黄
color_names = ['Blue', 'Green', 'Red', 'Yellow']

# ==================== 综合对比图 ====================
print("\n🎨 Creating comprehensive comparison plot...")

fig = plt.figure(figsize=(20, 14))
gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

# 定义要对比的关键特征
comparison_features = [
    ('trip_length_mean', 'Trip Length (points)', 'Trip Duration'),
    ('speed_mean', 'Speed Mean (norm)', 'Average Speed'),
    ('speed_max', 'Speed Max (norm)', 'Maximum Speed'),
    ('speed_std', 'Speed Std (norm)', 'Speed Variability'),
    ('accel_std', 'Accel Std (norm)', 'Acceleration Variability'),
    ('power_mean', 'Power Mean (norm)', 'Energy Consumption'),
    ('idle_ratio', 'Idle Ratio', 'Idle Time Ratio'),
    ('high_speed_ratio', 'High Speed Ratio', 'High Speed Ratio'),
    ('soc_drop_mean', 'SOC Drop (norm)', 'SOC Drop per Trip'),
    ('speed_change_rate', 'Speed Change Rate', 'Speed Variation Rate'),
    ('accel_switch_freq', 'Accel Switch Freq', 'Driving Smoothness'),
]

for idx, (feat, ylabel, title) in enumerate(comparison_features):
    ax = fig.add_subplot(gs[idx // 4, idx % 4])
    
    values = df_stats[feat].values
    x = np.arange(4)
    
    bars = ax.bar(x, values, color=colors, alpha=0.85, 
                  edgecolor='black', linewidth=2.5, width=0.65)
    
    ax.set_xlabel('Cluster', fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(['C0\nBlue', 'C1\nGreen', 'C2\nRed', 'C3\nYellow'], fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 数值标签
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.3f}', ha='center', va='bottom',
               fontsize=9, fontweight='bold')
    
    # 标注最大/最小
    max_idx = np.argmax(values)
    min_idx = np.argmin(values)
    bars[max_idx].set_edgecolor('darkgreen')
    bars[max_idx].set_linewidth(4)
    bars[min_idx].set_edgecolor('darkred')
    bars[min_idx].set_linewidth(4)

plt.suptitle('Physical Feature Comparison Across Clusters (GRU-based Clustering)', 
            fontsize=18, fontweight='bold', y=0.98)
plt.savefig('./results/physical_features_comparison.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: physical_features_comparison.png")

# ==================== 可视化典型轨迹 ====================
print("\n🎨 Plotting typical trajectories for each cluster...")

fig, axes = plt.subplots(4, 3, figsize=(18, 16))

for cluster_id in range(4):
    cluster_mask = (labels == cluster_id)
    cluster_driving = driving_seqs[cluster_mask]
    cluster_energy = energy_seqs[cluster_mask]
    
    # 随机选3条典型轨迹
    sample_indices = np.random.choice(len(cluster_driving), min(3, len(cluster_driving)), replace=False)
    
    for i, idx in enumerate(sample_indices):
        ax = axes[cluster_id, i]
        
        driving_seq = cluster_driving[idx]
        energy_seq = cluster_energy[idx]
        
        time_steps = np.arange(len(driving_seq))
        
        # 速度曲线
        ax_speed = ax
        ax_speed.plot(time_steps, driving_seq[:, 0], 
                     color=colors[cluster_id], linewidth=2, label='Speed')
        ax_speed.set_ylabel('Speed (norm)', fontsize=10, fontweight='bold', color=colors[cluster_id])
        ax_speed.tick_params(axis='y', labelcolor=colors[cluster_id])
        ax_speed.grid(True, alpha=0.3)
        
        # SOC曲线（右轴）
        ax_soc = ax_speed.twinx()
        ax_soc.plot(time_steps, energy_seq[:, 0], 
                   color='purple', linewidth=2, linestyle='--', alpha=0.7, label='SOC')
        ax_soc.set_ylabel('SOC (norm)', fontsize=10, fontweight='bold', color='purple')
        ax_soc.tick_params(axis='y', labelcolor='purple')
        
        if i == 0:
            ax_speed.set_title(f'Cluster {cluster_id} ({color_names[cluster_id]})', 
                             fontsize=12, fontweight='bold', color=colors[cluster_id])
        
        if cluster_id == 3:
            ax_speed.set_xlabel('Time Step', fontsize=10, fontweight='bold')

plt.suptitle('Typical Trip Trajectories by Cluster', fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('./results/typical_trajectories.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: typical_trajectories.png")

# ==================== 生成物理解释 ====================
print("\n" + "="*70)
print("💡 Physical Interpretation of Each Cluster")
print("="*70)

interpretations = {}

for cluster_id in range(4):
    stats = df_stats.loc[cluster_id]
    
    print(f"\n{'='*70}")
    print(f"🔷 Cluster {cluster_id} ({color_names[cluster_id]})")
    print(f"{'='*70}")
    print(f"   Sample count: {stats['count']:,} trips")
    
    interpretation = []
    
    # 分析行程长度
    if stats['trip_length_mean'] > df_stats['trip_length_mean'].quantile(0.75):
        print(f"   ✅ LONG TRIPS: Avg {stats['trip_length_mean']:.0f} points")
        interpretation.append("Long trips")
    elif stats['trip_length_mean'] < df_stats['trip_length_mean'].quantile(0.25):
        print(f"   ✅ SHORT TRIPS: Avg {stats['trip_length_mean']:.0f} points")
        interpretation.append("Short trips")
    else:
        print(f"   ○ Medium trips: {stats['trip_length_mean']:.0f} points")
    
    # 分析速度特征
    if stats['speed_mean'] > df_stats['speed_mean'].quantile(0.75):
        print(f"   ✅ HIGH SPEED: Mean {stats['speed_mean']:.3f}")
        interpretation.append("High speed")
    elif stats['speed_mean'] < df_stats['speed_mean'].quantile(0.25):
        print(f"   ✅ LOW SPEED: Mean {stats['speed_mean']:.3f}")
        interpretation.append("Low speed")
    
    if stats['speed_max'] > df_stats['speed_max'].quantile(0.75):
        print(f"   ✅ HIGH MAX SPEED: {stats['speed_max']:.3f}")
        interpretation.append("High max speed")
    
    # 分析怠速
    if stats['idle_ratio'] > 0.3:
        print(f"   ✅ HIGH IDLE TIME: {stats['idle_ratio']*100:.1f}%")
        interpretation.append("Frequent stops")
    elif stats['idle_ratio'] < 0.05:
        print(f"   ✅ CONTINUOUS DRIVING: Idle {stats['idle_ratio']*100:.1f}%")
        interpretation.append("Continuous")
    
    # 分析速度变化
    if stats['speed_std'] > df_stats['speed_std'].quantile(0.75):
        print(f"   ✅ HIGH SPEED VARIABILITY: Std {stats['speed_std']:.3f}")
        interpretation.append("Variable speed")
    elif stats['speed_std'] < df_stats['speed_std'].quantile(0.25):
        print(f"   ✅ STABLE SPEED: Std {stats['speed_std']:.3f}")
        interpretation.append("Stable speed")
    
    # 分析加速度
    if stats['accel_std'] > df_stats['accel_std'].quantile(0.75):
        print(f"   ✅ AGGRESSIVE DRIVING: Accel std {stats['accel_std']:.3f}")
        interpretation.append("Aggressive")
    elif stats['accel_std'] < df_stats['accel_std'].quantile(0.25):
        print(f"   ✅ SMOOTH DRIVING: Accel std {stats['accel_std']:.3f}")
        interpretation.append("Smooth")
    
    # 分析能耗
    if stats['power_mean'] > df_stats['power_mean'].quantile(0.75):
        print(f"   ✅ HIGH ENERGY CONSUMPTION: Power {stats['power_mean']:.3f}")
        interpretation.append("High energy")
    elif stats['power_mean'] < df_stats['power_mean'].quantile(0.25):
        print(f"   ✅ LOW ENERGY CONSUMPTION: Power {stats['power_mean']:.3f}")
        interpretation.append("Eco-friendly")
    
    # 综合解释
    summary = " + ".join(interpretation)
    print(f"\n   🎯 SUMMARY: {summary}")
    
    # 场景识别
    if stats['idle_ratio'] > 0.5:
        scenario = "🚦 Urban congestion / Traffic jam"
    elif stats['speed_mean'] > 2.0 and stats['speed_std'] < 0.5:
        scenario = "🛣️  Highway cruising"
    elif stats['speed_mean'] > 1.5 and stats['accel_std'] > 0.8:
        scenario = "🏙️  Urban driving with frequent acceleration"
    elif stats['trip_length_mean'] < 50:
        scenario = "🅿️  Parking / Short distance movement"
    else:
        scenario = "🚗 Mixed driving conditions"
    
    print(f"   📍 LIKELY SCENARIO: {scenario}")
    
    interpretations[f'Cluster {cluster_id}'] = {
        'summary': summary,
        'scenario': scenario
    }

# 保存解释
import json
with open('./results/cluster_interpretations.json', 'w', encoding='utf-8') as f:
    json.dump(interpretations, f, indent=2, ensure_ascii=False)

print("\n" + "="*70)
print("✅ Physical Interpretation Complete!")
print("="*70)
print("\n📁 Generated files:")
print("   1. physical_features_comparison.png - 特征对比图")
print("   2. typical_trajectories.png - 典型轨迹图")
print("   3. cluster_physical_features.csv - 物理特征统计")
print("   4. cluster_interpretations.json - 簇解释")
print("="*70)