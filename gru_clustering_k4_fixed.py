"""
Fixed version with better data normalization and validation
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from pathlib import Path
import json
import warnings
from tqdm import tqdm

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler  # 改用StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("🎯 GRU Clustering K=4 - Fixed Version")
print("="*70)


# ==================== [前面的代码保持不变，从Dataset到extract_cluster_features] ====================
# [这里插入之前所有的类和函数定义...]
# [为了简洁，我只展示需要修改的部分]


# ==================== 修复后的雷达图绘制函数 ====================
def plot_radar_k4_fixed(cluster_stats):
    """Plot K=4 radar chart with improved normalization"""
    print("\n" + "="*70)
    print("🎨 Plotting K=4 Radar Chart (Fixed)")
    print("="*70)
    
    # 特征列表
    feature_keys = [
        'Avg Speed',
        'Speed Std',
        'High Speed Ratio',
        'Idle Ratio',
        'Accel Std',
        'Harsh Accel %',
        'Harsh Decel %',
        'SOC Drop Rate',
        'Avg Power',
        'Trip Length'
    ]
    
    # 提取数据
    data = []
    for stats in cluster_stats:
        data.append([stats[key] for key in feature_keys])
    
    data = np.array(data)
    
    # 打印原始数据（诊断用）
    print("\n原始数据:")
    for i, row in enumerate(data):
        print(f"Cluster {i}: {row}")
    
    # 检查数据有效性
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        print("⚠️  Warning: Data contains NaN or Inf values!")
        data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0)
    
    # 改进的归一化方法：使用相对比例而不是MinMaxScaler
    # 每个特征除以该特征的最大值（如果最大值>0）
    data_normalized = np.zeros_like(data)
    for j in range(data.shape[1]):
        col = data[:, j]
        max_val = np.max(col)
        if max_val > 1e-6:  # 避免除以0
            data_normalized[:, j] = col / max_val
        else:
            data_normalized[:, j] = 0.0
    
    print("\n归一化后数据:")
    for i, row in enumerate(data_normalized):
        print(f"Cluster {i}: {row}")
    
    # 绘制雷达图
    N = len(feature_keys)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(14, 14), subplot_kw=dict(projection='polar'))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    labels = ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3']
    markers = ['o', 's', '^', 'D']  # 不同形状的标记
    
    # 绘制每个簇
    for cluster_id in range(4):
        values = data_normalized[cluster_id].tolist()
        values += values[:1]
        
        # 检查这个簇的数据
        print(f"\nCluster {cluster_id} values for plot: {values[:3]}...")
        
        ax.plot(angles, values, marker=markers[cluster_id], linewidth=3, 
               label=labels[cluster_id], color=colors[cluster_id], 
               markersize=10, markeredgecolor='black', markeredgewidth=1.5)
        ax.fill(angles, values, alpha=0.15, color=colors[cluster_id])
    
    # 设置图表
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_keys, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=11, color='gray')
    ax.grid(True, linestyle='--', alpha=0.7, linewidth=1.5)
    
    ax.set_title('Driving Behavior Clustering (K=4) - Fixed', 
                fontsize=18, fontweight='bold', pad=35)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), 
             fontsize=13, framealpha=0.95, shadow=True)
    
    plt.tight_layout()
    plt.savefig('./results/cluster_radar_k4_fixed.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Fixed radar chart saved: ./results/cluster_radar_k4_fixed.png")
    
    # 额外：绘制对数尺度版本（如果数据跨度很大）
    plot_radar_log_scale(data, feature_keys, colors, labels)


def plot_radar_log_scale(data, feature_keys, colors, labels):
    """绘制对数尺度的雷达图（适用于数值跨度大的情况）"""
    print("\n📊 Plotting log-scale version...")
    
    # 对数变换（加1避免log(0)）
    data_log = np.log1p(data)
    
    # 归一化
    data_log_norm = np.zeros_like(data_log)
    for j in range(data_log.shape[1]):
        col = data_log[:, j]
        max_val = np.max(col)
        if max_val > 1e-6:
            data_log_norm[:, j] = col / max_val
        else:
            data_log_norm[:, j] = 0.0
    
    N = len(feature_keys)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(14, 14), subplot_kw=dict(projection='polar'))
    
    markers = ['o', 's', '^', 'D']
    
    for cluster_id in range(4):
        values = data_log_norm[cluster_id].tolist()
        values += values[:1]
        
        ax.plot(angles, values, marker=markers[cluster_id], linewidth=3, 
               label=labels[cluster_id], color=colors[cluster_id], 
               markersize=10, markeredgecolor='black', markeredgewidth=1.5)
        ax.fill(angles, values, alpha=0.15, color=colors[cluster_id])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_keys, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=11, color='gray')
    ax.grid(True, linestyle='--', alpha=0.7, linewidth=1.5)
    
    ax.set_title('Driving Behavior Clustering (K=4) - Log Scale', 
                fontsize=18, fontweight='bold', pad=35)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), 
             fontsize=13, framealpha=0.95, shadow=True)
    
    plt.tight_layout()
    plt.savefig('./results/cluster_radar_k4_logscale.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Log-scale radar saved: ./results/cluster_radar_k4_logscale.png")


# ==================== 新增：数据诊断函数 ====================
def diagnose_cluster_data(cluster_stats):
    """诊断聚类数据，找出为什么有些簇不显示"""
    print("\n" + "="*70)
    print("🔍 Cluster Data Diagnosis")
    print("="*70)
    
    feature_keys = [
        'Avg Speed', 'Speed Std', 'High Speed Ratio', 'Idle Ratio',
        'Accel Std', 'Harsh Accel %', 'Harsh Decel %',
        'SOC Drop Rate', 'Avg Power', 'Trip Length'
    ]
    
    # 创建DataFrame便于分析
    df = pd.DataFrame(cluster_stats, index=['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3'])
    
    print("\n📊 Feature Statistics Across Clusters:")
    print(df[feature_keys].to_string())
    
    print("\n📊 Feature Range (Min-Max):")
    for feature in feature_keys:
        values = df[feature].values
        print(f"  {feature:20s}: {values.min():8.2f} - {values.max():8.2f} (ratio: {values.max()/max(values.min(), 0.01):.1f}x)")
    
    print("\n📊 Per-Cluster Summary:")
    for i in range(4):
        values = df.iloc[i][feature_keys].values
        print(f"  Cluster {i}: mean={values.mean():.2f}, std={values.std():.2f}, max={values.max():.2f}")
    
    # 检查是否有异常大的值导致其他簇被压缩
    print("\n⚠️  Potential Issues:")
    for feature in feature_keys:
        values = df[feature].values
        if values.max() > 10 * values.mean():
            print(f"  - {feature}: Cluster {np.argmax(values)} has outlier value ({values.max():.2f})")


# ==================== 修改主函数 ====================
def main():
    # ... [前面的代码保持不变] ...
    
    # 在绘制雷达图之前，先诊断数据
    diagnose_cluster_data(cluster_stats)
    
    # 使用修复后的雷达图函数
    plot_radar_k4_fixed(cluster_stats)
    
    # ... [后续代码保持不变] ...


# ==================== 如果你只想快速修复当前结果 ====================
def quick_fix_existing_results():
    """如果已经有聚类结果，快速重新绘制雷达图"""
    print("="*70)
    print("🔧 Quick Fix: Replot radar from existing results")
    print("="*70)
    
    # 加载已有的聚类特征
    try:
        df = pd.read_csv('./results/cluster_features_k4.csv', index_col=0)
        print(f"\n✅ Loaded existing cluster features")
        
        # 转换为字典格式
        cluster_stats = []
        for i in range(4):
            stats = df.iloc[i].to_dict()
            cluster_stats.append(stats)
        
        # 诊断数据
        diagnose_cluster_data(cluster_stats)
        
        # 重新绘制
        plot_radar_k4_fixed(cluster_stats)
        
        print("\n✅ Fixed radar chart generated!")
        
    except FileNotFoundError:
        print("❌ Error: cluster_features_k4.csv not found")
        print("   Please run the full clustering analysis first")


if __name__ == "__main__":
    # 选择运行模式
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--quick-fix':
        # 快速修复模式：只重新绘制雷达图
        quick_fix_existing_results()
    else:
        # 完整运行模式
        main()