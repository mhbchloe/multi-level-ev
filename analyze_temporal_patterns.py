"""
分析GRU学到的时序模式差异
不看统计量，看序列形状、动态变化、转换模式
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.spatial.distance import euclidean
from scipy.stats import entropy
from pathlib import Path

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("🔬 Temporal Pattern Analysis (Not Statistics)")
print("="*70)

# ==================== 加载数据 ====================
labels = np.load('./results/labels_k4_crossattn.npy')
driving_seqs = np.load('./results/temporal_soc_full/driving_sequences.npy', allow_pickle=True)
energy_seqs = np.load('./results/temporal_soc_full/energy_sequences.npy', allow_pickle=True)

min_len = min(len(labels), len(driving_seqs), len(energy_seqs))
labels = labels[:min_len]
driving_seqs = driving_seqs[:min_len]
energy_seqs = energy_seqs[:min_len]

print(f"✅ Loaded {len(labels):,} samples")

colors = ['#3498DB', '#2ECC71', '#E74C3C', '#F39C12']

# ==================== 1. 可视化典型轨迹形状 ====================
print("\n🎨 Visualizing trajectory shapes (what GRU learns)...")

fig, axes = plt.subplots(4, 4, figsize=(20, 16))

for cluster_id in range(4):
    cluster_mask = (labels == cluster_id)
    cluster_driving = driving_seqs[cluster_mask]
    cluster_energy = energy_seqs[cluster_mask]
    
    # 选4条有代表性的轨迹
    # 策略：选择长度接近中位数、速度模式不同的
    lengths = [len(seq) for seq in cluster_driving]
    median_len = np.median(lengths)
    
    # 找接近中位数长度的样本
    close_to_median = []
    for i, seq in enumerate(cluster_driving):
        if abs(len(seq) - median_len) < 20:
            close_to_median.append((i, seq))
    
    # 随机选4个
    if len(close_to_median) >= 4:
        samples = np.random.choice(len(close_to_median), 4, replace=False)
        sample_seqs = [close_to_median[i] for i in samples]
    else:
        samples = np.random.choice(len(cluster_driving), min(4, len(cluster_driving)), replace=False)
        sample_seqs = [(i, cluster_driving[i]) for i in samples]
    
    for col, (idx, seq) in enumerate(sample_seqs[:4]):
        ax = axes[cluster_id, col]
        
        time_steps = np.arange(len(seq))
        speed = seq[:, 0]
        accel = seq[:, 1]
        
        # 速度曲线
        ax.plot(time_steps, speed, color=colors[cluster_id], 
               linewidth=2.5, label='Speed', alpha=0.8)
        ax.fill_between(time_steps, 0, speed, color=colors[cluster_id], alpha=0.2)
        
        # 加速度曲线（右轴）
        ax2 = ax.twinx()
        ax2.plot(time_steps, accel, color='red', 
                linewidth=1.5, linestyle='--', alpha=0.6, label='Accel')
        ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
        
        ax.set_ylim([0, 5])
        ax2.set_ylim([-2, 2])
        
        if col == 0:
            ax.set_ylabel(f'Cluster {cluster_id}\nSpeed (norm)', 
                         fontsize=11, fontweight='bold', color=colors[cluster_id])
        
        if cluster_id == 0 and col == 0:
            ax.legend(loc='upper left', fontsize=9)
            ax2.legend(loc='upper right', fontsize=9)
        
        if cluster_id == 3:
            ax.set_xlabel('Time Step', fontsize=10)
        
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f9f9f9')

plt.suptitle('Trajectory Shape Patterns by Cluster (What GRU Learns)', 
            fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('./results/trajectory_shapes_comparison.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: trajectory_shapes_comparison.png")

# ==================== 2. 分析速度转换模式 ====================
print("\n🔍 Analyzing speed transition patterns...")

transition_stats = []

for cluster_id in range(4):
    cluster_mask = (labels == cluster_id)
    cluster_driving = driving_seqs[cluster_mask]
    
    # 定义速度状态：Idle(0), Low(1), Medium(2), High(3)
    def speed_to_state(speed):
        if speed < 0.1:
            return 0  # Idle
        elif speed < 1.0:
            return 1  # Low
        elif speed < 2.0:
            return 2  # Medium
        else:
            return 3  # High
    
    # 构建转换矩阵
    transition_matrix = np.zeros((4, 4))
    
    for seq in cluster_driving:
        speeds = seq[:, 0]
        states = [speed_to_state(s) for s in speeds]
        
        for i in range(len(states) - 1):
            from_state = states[i]
            to_state = states[i + 1]
            transition_matrix[from_state, to_state] += 1
    
    # 归一化
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    transition_matrix = np.divide(transition_matrix, row_sums, 
                                  where=row_sums!=0, out=np.zeros_like(transition_matrix))
    
    # 计算转换熵（衡量变化的随机性）
    trans_entropy = np.mean([entropy(row) for row in transition_matrix if row.sum() > 0])
    
    # 计算状态停留时间
    state_durations = {0: [], 1: [], 2: [], 3: []}
    for seq in cluster_driving:
        speeds = seq[:, 0]
        states = [speed_to_state(s) for s in speeds]
        
        current_state = states[0]
        duration = 1
        
        for i in range(1, len(states)):
            if states[i] == current_state:
                duration += 1
            else:
                state_durations[current_state].append(duration)
                current_state = states[i]
                duration = 1
    
    avg_durations = {state: np.mean(durs) if durs else 0 
                     for state, durs in state_durations.items()}
    
    transition_stats.append({
        'cluster': cluster_id,
        'transition_matrix': transition_matrix,
        'transition_entropy': trans_entropy,
        'avg_idle_duration': avg_durations[0],
        'avg_low_speed_duration': avg_durations[1],
        'avg_medium_speed_duration': avg_durations[2],
        'avg_high_speed_duration': avg_durations[3],
    })
    
    print(f"\n  Cluster {cluster_id}:")
    print(f"     Transition entropy: {trans_entropy:.3f}")
    print(f"     Avg idle duration: {avg_durations[0]:.1f} steps")
    print(f"     Avg high-speed duration: {avg_durations[3]:.1f} steps")

# ==================== 3. 可视化转换矩阵 ====================
print("\n🎨 Visualizing state transition matrices...")

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

state_names = ['Idle', 'Low', 'Med', 'High']

for cluster_id in range(4):
    ax = axes[cluster_id]
    
    matrix = transition_stats[cluster_id]['transition_matrix']
    
    im = ax.imshow(matrix, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
    
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(state_names, fontsize=10)
    ax.set_yticklabels(state_names, fontsize=10)
    
    ax.set_xlabel('To State', fontsize=11, fontweight='bold')
    if cluster_id == 0:
        ax.set_ylabel('From State', fontsize=11, fontweight='bold')
    
    ax.set_title(f'Cluster {cluster_id}\nEntropy={transition_stats[cluster_id]["transition_entropy"]:.3f}', 
                fontsize=12, fontweight='bold', color=colors[cluster_id])
    
    # 添加数值标签
    for i in range(4):
        for j in range(4):
            text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                          ha="center", va="center", color="black" if matrix[i, j] < 0.5 else "white",
                          fontsize=9, fontweight='bold')
    
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle('Speed State Transition Patterns (Temporal Dynamics)', 
            fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('./results/state_transitions.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: state_transitions.png")

# ==================== 4. 加速度模式分析 ====================
print("\n🔍 Analyzing acceleration patterns...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for cluster_id in range(4):
    ax = axes[cluster_id]
    
    cluster_mask = (labels == cluster_id)
    cluster_driving = driving_seqs[cluster_mask]
    
    # 采样加速度数据
    all_accel = []
    for seq in cluster_driving[:1000]:  # 只取前1000个样本
        all_accel.extend(seq[:, 1])
    
    # 加速度分布
    ax.hist(all_accel, bins=50, color=colors[cluster_id], 
           alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
    ax.set_xlabel('Acceleration (norm)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title(f'Cluster {cluster_id} - Acceleration Distribution', 
                fontsize=12, fontweight='bold', color=colors[cluster_id])
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.suptitle('Acceleration Pattern Differences', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('./results/acceleration_patterns.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: acceleration_patterns.png")

# ==================== 5. 总结时序模式差异 ====================
print("\n" + "="*70)
print("💡 Temporal Pattern Summary (Not Statistical Differences)")
print("="*70)

for cluster_id in range(4):
    stats = transition_stats[cluster_id]
    
    print(f"\n🔷 Cluster {cluster_id}:")
    print(f"   Transition entropy: {stats['transition_entropy']:.3f}")
    print(f"      → {'High variability' if stats['transition_entropy'] > 0.8 else 'Stable patterns'}")
    
    print(f"   Idle duration: {stats['avg_idle_duration']:.1f} steps")
    print(f"   High-speed duration: {stats['avg_high_speed_duration']:.1f} steps")
    
    # 分析主导转换
    matrix = stats['transition_matrix']
    max_trans = []
    for i in range(4):
        max_to = np.argmax(matrix[i])
        if matrix[i, max_to] > 0.5:
            max_trans.append(f"{state_names[i]}→{state_names[max_to]}")
    
    if max_trans:
        print(f"   Dominant transitions: {', '.join(max_trans)}")

print("\n" + "="*70)
print("✅ Temporal Pattern Analysis Complete!")
print("="*70)
print("\n📌 Key insight:")
print("   GRU learns TEMPORAL PATTERNS (sequence shapes, transitions)")
print("   NOT statistical averages (mean speed, mean power)")
print("   → Clusters may have similar statistics but different dynamics!")
print("="*70)