"""
Step 8 补充：车辆聚类分析图表 (修复版)
生成论文所需的所有可视化：
  3.2.2 Vehicle-level Representation Construction
  3.2.3 Clustering Method & Model Selection
  4.3.4 结果展示
"""

import numpy as np
import pandas as pd
import os
import json
import warnings
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from scipy.stats import entropy
import seaborn as sns
from tqdm import tqdm

warnings.filterwarnings('ignore')

rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 10
rcParams['figure.dpi'] = 150

print("=" * 80)
print("📊 VEHICLE CLUSTERING ANALYSIS FIGURES")
print("=" * 80)

# ============================================================
# 配置
# ============================================================
CONFIG = {
    'vehicle_results': './vehicle_clustering/results/vehicle_clustering_gmm_k4.csv',
    'vehicle_summary': './vehicle_clustering/results/vehicle_clustering_gmm_k4_summary.json',
    'segments_path': './coupling_analysis/results/segments_integrated_complete.csv',
    'save_dir': './vehicle_clustering/results/paper_figures/',
    'seed': 42,
}

os.makedirs(CONFIG['save_dir'], exist_ok=True)

# ============================================================
# 加载数据
# ============================================================
print("\n【Loading Data】")

vehicle_results = pd.read_csv(CONFIG['vehicle_results'])
with open(CONFIG['vehicle_summary'], 'r') as f:
    summary = json.load(f)

segments_df = pd.read_csv(CONFIG['segments_path'])

print(f"   ✓ {len(vehicle_results):,} vehicles")
print(f"   ✓ K={summary['n_clusters']} clusters")

# ============================================================
# 图1：Composition Features 可视化
# ============================================================
print("\n【Figure 1】Composition Features (分布特征)")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

K = summary['n_clusters']
cluster_stats = summary['cluster_stats']
unique_clusters = sorted([int(k) for k in cluster_stats.keys()])

colors_segment = ['#5B9BD5', '#70AD47', '#C0504D', '#FFC000']

# 每个车辆聚类的组成分布
for vi, vc in enumerate(unique_clusters):
    ax = axes[vi]
    comp = cluster_stats[str(vc)]['composition']
    comp_vals = [comp[f'C{c}'] for c in range(4)]
    
    # 创建堆叠条形图（显示所有车辆的组成）
    vehicles_in_cluster = vehicle_results[vehicle_results['vehicle_cluster'] == vc]
    
    # 获取这个聚类中所有车的组成均值
    means = []
    stds = []
    for c in range(4):
        col = f'cluster_{c}_ratio'
        mean = vehicles_in_cluster[col].mean()
        std = vehicles_in_cluster[col].std()
        means.append(mean)
        stds.append(std)
    
    # 绘制带误差条的条形图
    x_pos = np.arange(4)
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, 
                 color=colors_segment, edgecolor='black', linewidth=1.5,
                 alpha=0.8, error_kw={'linewidth': 2, 'ecolor': 'gray'})
    
    # 添加数值标签
    for i, (bar, m) in enumerate(zip(bars, means)):
        ax.text(bar.get_x() + bar.get_width()/2, m + stds[i] + 0.03, 
               f'{m:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['C0\nLong Idle', 'C1\nUrban', 'C2\nHighway', 'C3\nShort Idle'],
                      fontsize=9)
    ax.set_ylabel('Mean Ratio', fontweight='bold', fontsize=10)
    ax.set_title(f'({"abcd"[vi]}) V{vc}: {cluster_stats[str(vc)]["label"]}\n(n={len(vehicles_in_cluster):,})',
                fontweight='bold', fontsize=11)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.2, axis='y')

plt.suptitle('3.2.2(1) Composition Features: Segment Distribution by Vehicle Cluster',
            fontweight='bold', fontsize=13)
plt.tight_layout()

for fmt, dpi in [('.png', 300), ('.pdf', None)]:
    path = os.path.join(CONFIG['save_dir'], f'fig_composition_features{fmt}')
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: fig_composition_features.png/pdf")

# ============================================================
# 图2：Transition Dynamics 热力图
# ============================================================
print("\n【Figure 2】Transition Dynamics (转移动力学)")
print("=" * 80)

fig, axes = plt.subplots(1, K, figsize=(4*K, 4.5))
if K == 1:
    axes = [axes]

for vi, vc in enumerate(unique_clusters):
    ax = axes[vi]
    
    # 构建平均转移矩阵
    vehicles_in_cluster = vehicle_results[vehicle_results['vehicle_cluster'] == vc]
    
    # 从段数据中计算每辆车的转移矩阵
    transition_matrices = []
    for vid in tqdm(vehicles_in_cluster['vehicle_id'].values[:100], 
                   desc=f"   V{vc}", leave=False):
        v_segs = segments_df[segments_df['vehicle_id'] == vid].sort_values('segment_id')
        if len(v_segs) > 1:
            clusters = v_segs['cluster_id'].values
            T = np.zeros((4, 4))
            for t in range(len(clusters)-1):
                T[int(clusters[t]), int(clusters[t+1])] += 1
            # 行归一化
            row_sums = T.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            T = T / row_sums
            transition_matrices.append(T)
    
    # 平均转移矩阵
    if len(transition_matrices) > 0:
        T_avg = np.mean(transition_matrices, axis=0)
    else:
        T_avg = np.eye(4) * 0.25
    
    # 绘制热力图
    sns.heatmap(T_avg, annot=True, fmt='.3f', cmap='YlOrRd', 
               ax=ax, cbar_kws={'label': 'Probability'},
               xticklabels=['C0', 'C1', 'C2', 'C3'],
               yticklabels=['C0', 'C1', 'C2', 'C3'],
               linewidths=1, linecolor='black')
    
    ax.set_title(f'({"abcd"[vi]}) V{vc}: {cluster_stats[str(vc)]["label"]}',
                fontweight='bold', fontsize=11)
    ax.set_xlabel('To →', fontweight='bold')
    ax.set_ylabel('From ↓', fontweight='bold')

plt.suptitle('3.2.2(2) Transition Dynamics: Mode Transition Probabilities',
            fontweight='bold', fontsize=13)
plt.tight_layout()

for fmt, dpi in [('.png', 300), ('.pdf', None)]:
    path = os.path.join(CONFIG['save_dir'], f'fig_transition_dynamics{fmt}')
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: fig_transition_dynamics.png/pdf")

# ============================================================
# 图3：Evolution Features 对比 (修复版)
# ============================================================
print("\n【Figure 3】Evolution & Rhythm Features")
print("=" * 80)

# 提取演化特征 - 使用实际存在的列
available_cols = list(vehicle_results.columns)

# 定义候选特征，检查哪些存在
evolution_candidates = [
    ('avg_soc_drop_per_segment', 'Avg SOC Drop', '%'),
    ('high_energy_ratio', 'High Energy Ratio', ''),
    ('idle_dominant_ratio', 'Idle Dominant Ratio', ''),
    ('avg_heading_change', 'Avg Heading Change', 'deg'),
    ('avg_power_mean', 'Avg Power', 'W'),
    ('avg_acc_std_mov', 'Avg Acceleration Std', 'm/s²'),
    ('avg_soc_rate', 'Avg SOC Rate', '%/min'),
    ('total_duration_hrs', 'Total Duration', 'hrs'),
]

# 过滤存在的列
evolution_features = [f for f in evolution_candidates if f[0] in available_cols][:6]

print(f"   Using features: {[f[0] for f in evolution_features]}")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx in range(6):
    ax = axes[idx]
    
    if idx < len(evolution_features):
        feat_col, feat_name, unit = evolution_features[idx]
        
        data_for_box = []
        labels_for_box = []
        
        for vc in unique_clusters:
            vehicles_in_cluster = vehicle_results[vehicle_results['vehicle_cluster'] == vc]
            if feat_col in vehicles_in_cluster.columns:
                data = vehicles_in_cluster[feat_col].dropna().values
                if len(data) > 0:
                    data_for_box.append(data)
                    labels_for_box.append(f"V{vc}")
        
        # 只有在有足够数据时才绘制
        if len(data_for_box) > 0 and all(len(d) > 0 for d in data_for_box):
            bp = ax.boxplot(data_for_box, labels=labels_for_box, patch_artist=True,
                           medianprops=dict(color='red', linewidth=2),
                           boxprops=dict(linewidth=1.5),
                           whiskerprops=dict(linewidth=1.2),
                           capprops=dict(linewidth=1.2))
            
            for patch, color in zip(bp['boxes'], ['#5B9BD5', '#70AD47', '#C0504D', '#FFC000'][:len(data_for_box)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_ylabel(f'{feat_name} {unit}'.strip(), fontweight='bold')
            ax.set_title(f'({chr(97+idx)}) {feat_name}', fontweight='bold', fontsize=11)
            ax.grid(True, alpha=0.2, axis='y')
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
    else:
        ax.axis('off')

plt.suptitle('3.2.2(3) Evolution & Rhythm Features: Driving Behavior Metrics',
            fontweight='bold', fontsize=13)
plt.tight_layout()

for fmt, dpi in [('.png', 300), ('.pdf', None)]:
    path = os.path.join(CONFIG['save_dir'], f'fig_evolution_features{fmt}')
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: fig_evolution_features.png/pdf")

# ============================================================
# 图4：Clustering Quality Metrics
# ============================================================
print("\n【Figure 4】Clustering Quality Metrics")
print("=" * 80)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

metrics = [
    ('silhouette_score', 'Silhouette Score\n(higher is better)', 0.5),
    ('calinski_harabasz_score', 'Calinski-Harabasz Index\n(higher is better)', 1.0),
    ('davies_bouldin_score', 'Davies-Bouldin Index\n(lower is better)', 0.0),
]

for idx, (metric_key, metric_name, baseline) in enumerate(metrics):
    ax = axes[idx]
    
    value = summary[metric_key]
    
    # 创建简单的度量显示
    ax.text(0.5, 0.6, f'{value:.4f}', ha='center', va='center',
           fontsize=60, fontweight='bold', color='#2ecc71')
    ax.text(0.5, 0.25, metric_name, ha='center', va='center',
           fontsize=12, fontweight='bold')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # 添加背景
    rect = plt.Rectangle((0.05, 0.05), 0.9, 0.9, linewidth=2,
                         edgecolor='black', facecolor='#ecf0f1')
    ax.add_patch(rect)

plt.suptitle(f'3.2.3 Clustering Method & Model Selection\n(GMM K={summary["n_clusters"]}, cov_type={summary["covariance_type"]})',
            fontweight='bold', fontsize=13)
plt.tight_layout()

for fmt, dpi in [('.png', 300), ('.pdf', None)]:
    path = os.path.join(CONFIG['save_dir'], f'fig_clustering_quality{fmt}')
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: fig_clustering_quality.png/pdf")

# ============================================================
# 图5：PCA 可视化（降维投影）- 修复版
# ============================================================
print("\n【Figure 5】PCA Visualization (降维投影)")
print("=" * 80)

# 重新计算 PCA - 只使用数值列
numeric_cols = vehicle_results.select_dtypes(include=[np.number]).columns.tolist()

# 排除不需要的列
exclude_cols = ['vehicle_cluster', 'n_segments', 'n_trips']
feature_cols = [c for c in numeric_cols if c not in exclude_cols]

print(f"   Using {len(feature_cols)} numeric features for PCA")

X = vehicle_results[feature_cols].fillna(0).astype(np.float32).values
X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=3, random_state=CONFIG['seed'])
X_pca = pca.fit_transform(X_scaled)

# 3D PCA 投影
fig = plt.figure(figsize=(16, 6))

# 2D PC1-PC2
ax1 = fig.add_subplot(121)
colors_map = ['#5B9BD5', '#70AD47', '#C0504D', '#FFC000']
labels_result = vehicle_results['vehicle_cluster'].values

for vc in unique_clusters:
    mask = labels_result == vc
    label = cluster_stats[str(vc)]['label']
    ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], 
               c=colors_map[vc], s=50, alpha=0.6,
               label=f'V{vc}: {label}',
               edgecolors='black', linewidth=0.5)

ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontweight='bold')
ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontweight='bold')
ax1.set_title('(a) PC1 vs PC2', fontweight='bold', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.2)

# 2D PC1-PC3
ax2 = fig.add_subplot(122)
for vc in unique_clusters:
    mask = labels_result == vc
    label = cluster_stats[str(vc)]['label']
    ax2.scatter(X_pca[mask, 0], X_pca[mask, 2], 
               c=colors_map[vc], s=50, alpha=0.6,
               label=f'V{vc}: {label}',
               edgecolors='black', linewidth=0.5)

ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontweight='bold')
ax2.set_ylabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})', fontweight='bold')
ax2.set_title('(b) PC1 vs PC3', fontweight='bold', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.2)

plt.suptitle('4.3.4 Vehicle Cluster Visualization (PCA Projection)',
            fontweight='bold', fontsize=13)
plt.tight_layout()

for fmt, dpi in [('.png', 300), ('.pdf', None)]:
    path = os.path.join(CONFIG['save_dir'], f'fig_pca_visualization{fmt}')
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: fig_pca_visualization.png/pdf")

# ============================================================
# 图6：聚类大小和特征对比总结表
# ============================================================
print("\n【Figure 6】Cluster Summary Table")
print("=" * 80)

fig, ax = plt.subplots(figsize=(14, 6))
ax.axis('tight')
ax.axis('off')

# 准备表格数据
table_data = []
headers = ['Cluster', 'Label', 'Size', 'C0\n(Idle)', 'C1\n(Urban)', 'C2\n(Highway)', 'C3\n(Short)', 
          'High E.', 'Idle D.']

for vc in unique_clusters:
    vehicles_in_cluster = vehicle_results[vehicle_results['vehicle_cluster'] == vc]
    label = cluster_stats[str(vc)]['label']
    size = len(vehicles_in_cluster)
    
    comp = cluster_stats[str(vc)]['composition']
    high_e = cluster_stats[str(vc)]['high_energy_ratio']
    idle_d = cluster_stats[str(vc)]['idle_dominant_ratio']
    
    row = [
        f'V{vc}',
        label,
        f'{size:,}',
        f'{comp["C0"]:.1%}',
        f'{comp["C1"]:.1%}',
        f'{comp["C2"]:.1%}',
        f'{comp["C3"]:.1%}',
        f'{high_e:.2f}',
        f'{idle_d:.2f}',
    ]
    table_data.append(row)

table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center',
                loc='center', colWidths=[0.10, 0.15, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# 美化表头
for i in range(len(headers)):
    table[(0, i)].set_facecolor('#3498db')
    table[(0, i)].set_text_props(weight='bold', color='white')

# 美化数据行
for i in range(1, len(table_data) + 1):
    for j in range(len(headers)):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#ecf0f1')
        else:
            table[(i, j)].set_facecolor('#ffffff')

ax.set_title('Table 1: Vehicle Cluster Summary Statistics',
            fontweight='bold', fontsize=12, pad=20)

plt.tight_layout()

for fmt, dpi in [('.png', 300), ('.pdf', None)]:
    path = os.path.join(CONFIG['save_dir'], f'fig_cluster_summary_table{fmt}')
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: fig_cluster_summary_table.png/pdf")

# ============================================================
# 完成
# ============================================================
print("\n" + "=" * 80)
print("✅ ALL FIGURES GENERATED!")
print("=" * 80)

print(f"""
Generated Figures:
  1. fig_composition_features.png/pdf
     → 3.2.2(1) Composition Features 分布特征
  
  2. fig_transition_dynamics.png/pdf
     → 3.2.2(2) Transition Dynamics 转移动力学
  
  3. fig_evolution_features.png/pdf
     → 3.2.2(3) Evolution & Rhythm Features 演化和节奏特征
  
  4. fig_clustering_quality.png/pdf
     → 3.2.3 Clustering Quality Metrics 聚类质量指标
  
  5. fig_pca_visualization.png/pdf
     → 4.3.4 Vehicle Cluster Visualization PCA 投影
  
  6. fig_cluster_summary_table.png/pdf
     → Table 1: Cluster Summary Statistics 聚类摘要表

All figures saved to: {CONFIG['save_dir']}
""")

print("=" * 80)