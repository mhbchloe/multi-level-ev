"""
对比简单拼接 vs Cross-Attention 的聚类效果
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("📊 Comparing Fusion Methods")
print("="*70)

# 加载两个版本的结果
print("\n1️⃣ Loading Simple Concatenation results...")
try:
    labels_simple = np.load('./results/labels_k4.npy')
    features_simple = pd.read_csv('./results/cluster_features_k4.csv', index_col=0)
    print(f"   ✅ Simple version loaded")
except:
    print("   ❌ Simple version not found, please run gru_clustering_k4_english.py first")
    exit(1)

print("\n2️⃣ Loading Cross-Attention results...")
try:
    labels_crossattn = np.load('./results/labels_k4_crossattn.npy')
    features_crossattn = pd.read_csv('./results/cluster_features_k4_crossattn.csv', index_col=0)
    print(f"   ✅ Cross-Attention version loaded")
except:
    print("   ❌ Cross-Attention version not found")
    exit(1)

# 计算评估指标
from sklearn.metrics import silhouette_score

features_gru_simple = np.load('./results/features_k4.npy')
features_gru_crossattn = np.load('./results/features_k4_crossattn.npy')

sil_simple = silhouette_score(features_gru_simple, labels_simple)
sil_crossattn = silhouette_score(features_gru_crossattn, labels_crossattn)

cv_simple = np.std(np.bincount(labels_simple)) / np.mean(np.bincount(labels_simple))
cv_crossattn = np.std(np.bincount(labels_crossattn)) / np.mean(np.bincount(labels_crossattn))

print(f"\n📊 Metrics Comparison:")
print(f"   Simple Concatenation:")
print(f"      Silhouette: {sil_simple:.3f}")
print(f"      CV: {cv_simple:.3f}")
print(f"\n   Cross-Attention:")
print(f"      Silhouette: {sil_crossattn:.3f} ({(sil_crossattn-sil_simple)/sil_simple*100:+.1f}%)")
print(f"      CV: {cv_crossattn:.3f} ({(cv_crossattn-cv_simple)/cv_simple*100:+.1f}%)")

# ==================== 1. 对比雷达图 ====================
print("\n🎨 Generating radar chart comparison...")

fig = plt.figure(figsize=(20, 9))

# 选择关键特征
features_to_plot = ['Avg Speed', 'Max Speed', 'Avg Power', 'Trip Length']

N = len(features_to_plot)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']

# 简单拼接版本
ax1 = plt.subplot(121, projection='polar')

for i in range(4):
    values = features_simple.loc[f'Cluster {i}', features_to_plot].values
    # 归一化
    max_vals = features_simple[features_to_plot].max().values
    values_norm = values / max_vals
    values_norm = np.concatenate([values_norm, [values_norm[0]]])
    
    ax1.plot(angles, values_norm, 'o-', linewidth=3.5, 
            label=f'C{i}', color=colors[i], markersize=10,
            markeredgecolor='white', markeredgewidth=2)
    ax1.fill(angles, values_norm, alpha=0.15, color=colors[i])

ax1.set_xticks(angles[:-1])
ax1.set_xticklabels(features_to_plot, fontsize=13, fontweight='bold')
ax1.set_ylim(0, 1)
ax1.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax1.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=11, color='gray')
ax1.grid(True, linestyle='--', alpha=0.6, linewidth=1.5)

title1 = f'Simple Concatenation\nSil={sil_simple:.3f}, CV={cv_simple:.2f}'
ax1.set_title(title1, fontsize=15, fontweight='bold', pad=25)
ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12, framealpha=0.9)

# Cross-Attention版本
ax2 = plt.subplot(122, projection='polar')

for i in range(4):
    values = features_crossattn.loc[f'Cluster {i}', features_to_plot].values
    max_vals = features_crossattn[features_to_plot].max().values
    values_norm = values / max_vals
    values_norm = np.concatenate([values_norm, [values_norm[0]]])
    
    ax2.plot(angles, values_norm, 'o-', linewidth=3.5, 
            label=f'C{i}', color=colors[i], markersize=10,
            markeredgecolor='white', markeredgewidth=2)
    ax2.fill(angles, values_norm, alpha=0.15, color=colors[i])

ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(features_to_plot, fontsize=13, fontweight='bold')
ax2.set_ylim(0, 1)
ax2.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax2.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=11, color='gray')
ax2.grid(True, linestyle='--', alpha=0.6, linewidth=1.5)

title2 = f'Cross-Attention Fusion\nSil={sil_crossattn:.3f} (+{(sil_crossattn-sil_simple)/sil_simple*100:.1f}%), CV={cv_crossattn:.2f}'
ax2.set_title(title2, fontsize=15, fontweight='bold', pad=25)
ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12, framealpha=0.9)

plt.suptitle('Fusion Method Comparison: Driving Behavior Clustering', 
            fontsize=18, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('./results/fusion_comparison_radar.png', dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Radar comparison saved: ./results/fusion_comparison_radar.png")

# ==================== 2. 簇分布对比 ====================
print("\n📊 Generating cluster distribution comparison...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 简单拼接
ax = axes[0]
unique, counts = np.unique(labels_simple, return_counts=True)
percentages = counts / len(labels_simple) * 100

bars = ax.bar(unique, percentages, color=colors, alpha=0.8, edgecolor='black', linewidth=2, width=0.6)
ax.set_xlabel('Cluster', fontsize=13, fontweight='bold')
ax.set_ylabel('Percentage (%)', fontsize=13, fontweight='bold')
ax.set_title(f'Simple Concatenation\nCV={cv_simple:.3f}', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(unique)
ax.set_xticklabels([f'C{i}' for i in unique], fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, max(percentages) * 1.2)

for bar, pct in zip(bars, percentages):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{pct:.1f}%', ha='center', va='bottom', 
           fontsize=12, fontweight='bold')

# Cross-Attention
ax = axes[1]
unique, counts = np.unique(labels_crossattn, return_counts=True)
percentages = counts / len(labels_crossattn) * 100

bars = ax.bar(unique, percentages, color=colors, alpha=0.8, edgecolor='black', linewidth=2, width=0.6)
ax.set_xlabel('Cluster', fontsize=13, fontweight='bold')
ax.set_ylabel('Percentage (%)', fontsize=13, fontweight='bold')
ax.set_title(f'Cross-Attention Fusion\nCV={cv_crossattn:.3f}', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(unique)
ax.set_xticklabels([f'C{i}' for i in unique], fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, max(percentages) * 1.2)

for bar, pct in zip(bars, percentages):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{pct:.1f}%', ha='center', va='bottom', 
           fontsize=12, fontweight='bold')

plt.suptitle('Cluster Distribution Comparison', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('./results/fusion_comparison_distribution.png', dpi=300, bbox_inches='tight')
print(f"✅ Distribution comparison saved: ./results/fusion_comparison_distribution.png")

# ==================== 3. 指标对比柱状图 ====================
print("\n📊 Generating metrics comparison...")

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

metrics = ['Silhouette\nScore', 'CV\n(lower better)', 'Parameters\n(K)']
simple_vals = [sil_simple, cv_simple, 9.4]
crossattn_vals = [sil_crossattn, cv_crossattn, 11.9]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax.bar(x - width/2, simple_vals, width, label='Simple Concatenation',
              color='#95a5a6', alpha=0.8, edgecolor='black', linewidth=2)
bars2 = ax.bar(x + width/2, crossattn_vals, width, label='Cross-Attention',
              color='#3498db', alpha=0.8, edgecolor='black', linewidth=2)

ax.set_ylabel('Value', fontsize=13, fontweight='bold')
ax.set_title('Fusion Method Metrics Comparison', fontsize=15, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=12, fontweight='bold')
ax.legend(fontsize=12, framealpha=0.9, loc='upper left')
ax.grid(axis='y', alpha=0.3, linestyle='--')

# 添加数值标签
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}', ha='center', va='bottom', 
               fontsize=11, fontweight='bold')

# 添加改进百分比
improvements = [
    (sil_crossattn - sil_simple) / sil_simple * 100,
    (cv_crossattn - cv_simple) / cv_simple * 100,
    (11.9 - 9.4) / 9.4 * 100
]

for i, improvement in enumerate(improvements):
    color = 'green' if (i == 0 or (i == 1 and improvement < 0)) else 'red'
    ax.text(i, max(simple_vals[i], crossattn_vals[i]) * 1.15,
           f'{improvement:+.1f}%', ha='center', 
           fontsize=11, fontweight='bold', color=color)

plt.tight_layout()
plt.savefig('./results/fusion_comparison_metrics.png', dpi=300, bbox_inches='tight')
print(f"✅ Metrics comparison saved: ./results/fusion_comparison_metrics.png")

# ==================== 4. 生成对比报告 ====================
print("\n📄 Generating comparison report...")

report = {
    'fusion_methods': {
        'simple_concatenation': {
            'description': 'Direct concatenation of driving and energy features',
            'architecture': '[h_d; h_e] → MLP(32→64)',
            'parameters': 9400,
            'silhouette': float(sil_simple),
            'cv': float(cv_simple),
            'cluster_distribution': {
                f'cluster_{i}': int(c) for i, c in enumerate(np.bincount(labels_simple))
            }
        },
        'cross_attention': {
            'description': 'Bidirectional cross-channel attention mechanism',
            'architecture': '[h_d; Attn(h_d,h_e); h_e; Attn(h_e,h_d)] → MLP(64→48→32)',
            'parameters': 11877,
            'silhouette': float(sil_crossattn),
            'cv': float(cv_crossattn),
            'cluster_distribution': {
                f'cluster_{i}': int(c) for i, c in enumerate(np.bincount(labels_crossattn))
            }
        }
    },
    'improvements': {
        'silhouette': f'{(sil_crossattn-sil_simple)/sil_simple*100:+.2f}%',
        'cv': f'{(cv_crossattn-cv_simple)/cv_simple*100:+.2f}%',
        'parameters': f'{(11877-9400)/9400*100:+.2f}%'
    }
}

import json
with open('./results/fusion_comparison_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print(f"✅ Report saved: ./results/fusion_comparison_report.json")

print("\n" + "="*70)
print("✅ Comparison Complete!")
print("="*70)
print("\n📁 Generated files:")
print("   1. fusion_comparison_radar.png - Radar chart comparison")
print("   2. fusion_comparison_distribution.png - Cluster distribution")
print("   3. fusion_comparison_metrics.png - Metrics comparison")
print("   4. fusion_comparison_report.json - Detailed report")
print("\n" + "="*70)