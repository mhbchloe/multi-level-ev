"""
聚类结果可视化
读取已保存的features.npy和labels.npy
重新读取原始数据计算真实物理特征并可视化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from tqdm import tqdm
from sklearn.decomposition import PCA
import pickle

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("🎨 Clustering Results Visualization")
print("="*70)


# ==================== 加载聚类结果 ====================
print("\n📂 Loading clustering results...")

features = np.load('./results/features.npy')
labels = np.load('./results/labels.npy')

print(f"✅ Loaded:")
print(f"   Features: {features.shape}")
print(f"   Labels: {labels.shape}")
print(f"   Clusters: {np.unique(labels)}")


# ==================== 重新分段并提取统计特征 ====================
def resegment_and_analyze(csv_files, labels, max_segments=None):
    """
    重新分段，并按照labels计算每个簇的统计特征
    """
    print("\n" + "="*70)
    print("📊 Re-segmenting for Physical Analysis")
    print("="*70)
    
    if max_segments is None:
        max_segments = len(labels)
    
    segments_by_cluster = {0: [], 1: [], 2: [], 3: []}
    segment_idx = 0
    
    required_cols = ['vehicle_id', 'time', 'soc', 'spd', 'v', 'i', 'acc']
    
    for file in tqdm(csv_files, desc="Re-processing CSVs"):
        chunk_iter = pd.read_csv(file, usecols=required_cols, chunksize=1000000)
        
        for chunk in chunk_iter:
            # 清洗
            chunk = chunk.dropna(subset=['soc', 'spd', 'v', 'i', 'acc'])
            chunk = chunk[
                (chunk['soc'] >= 0) & (chunk['soc'] <= 100) &
                (chunk['spd'] >= 0) & (chunk['spd'] <= 220) &
                (chunk['v'] > 0) & (chunk['v'] <= 1000) &
                (chunk['i'] >= -1000) & (chunk['i'] <= 1000) &
                (chunk['acc'] >= -10) & (chunk['acc'] <= 10)
            ]
            
            # 分段
            for vehicle_id in chunk['vehicle_id'].unique():
                vehicle_data = chunk[chunk['vehicle_id'] == vehicle_id].sort_values('time')
                
                if len(vehicle_data) < 10:
                    continue
                
                soc_values = vehicle_data['soc'].values
                start_idx = 0
                
                while start_idx < len(vehicle_data):
                    soc_start = soc_values[start_idx]
                    
                    for end_idx in range(start_idx + 1, len(vehicle_data)):
                        soc_current = soc_values[end_idx]
                        soc_drop = soc_start - soc_current
                        
                        if soc_current > soc_start:
                            start_idx = end_idx
                            break
                        
                        if soc_drop >= 3.0:
                            segment = vehicle_data.iloc[start_idx:end_idx+1][required_cols]
                            
                            if len(segment) >= 10 and segment_idx < len(labels):
                                # 根据labels分类
                                cluster_id = labels[segment_idx]
                                segments_by_cluster[cluster_id].append(segment)
                                segment_idx += 1
                                
                                if segment_idx >= max_segments:
                                    print(f"\n✅ Collected {segment_idx} segments")
                                    return segments_by_cluster
                            
                            start_idx = end_idx + 1
                            break
                    else:
                        start_idx += 1
    
    print(f"\n✅ Collected {segment_idx} segments")
    return segments_by_cluster


# ==================== 计算物理特征统计 ====================
def compute_cluster_statistics(segments_by_cluster):
    """
    计算每个簇的真实物理特征统计
    """
    print("\n" + "="*70)
    print("📊 Computing Physical Statistics")
    print("="*70)
    
    cluster_stats = []
    
    for cluster_id in range(4):
        print(f"\n  Cluster {cluster_id}...")
        
        cluster_segments = segments_by_cluster.get(cluster_id, [])
        
        if len(cluster_segments) == 0:
            print(f"     ⚠️  No segments")
            continue
        
        # 合并所有片段数据
        all_data = pd.concat(cluster_segments, ignore_index=True)
        
        stats = {
            'cluster': cluster_id,
            'count': len(cluster_segments)
        }
        
        # 速度特征 (km/h)
        stats['avg_speed'] = all_data['spd'].mean()
        stats['max_speed'] = all_data['spd'].quantile(0.95)
        stats['speed_std'] = all_data['spd'].std()
        stats['min_speed'] = all_data['spd'].min()
        
        # 怠速
        stats['idle_ratio'] = (all_data['spd'] < 1).mean() * 100
        
        # 加速度 (m/s²)
        stats['avg_accel'] = all_data['acc'].abs().mean()
        stats['accel_std'] = all_data['acc'].std()
        
        # 能量特征
        stats['avg_soc'] = all_data['soc'].mean()
        
        # 每个片段的SOC下降
        soc_drops = []
        for seg in cluster_segments:
            if len(seg) > 1:
                soc_drops.append(seg['soc'].iloc[0] - seg['soc'].iloc[-1])
        stats['soc_drop'] = np.mean(soc_drops) if soc_drops else 0
        
        # 功率 (kW)
        stats['avg_power'] = (all_data['v'] * all_data['i']).abs().mean() / 1000
        stats['max_power'] = (all_data['v'] * all_data['i']).abs().quantile(0.95) / 1000
        
        # 行程长度
        stats['avg_trip_length'] = np.mean([len(seg) for seg in cluster_segments])
        
        cluster_stats.append(stats)
        
        print(f"     Samples: {stats['count']:,}")
        print(f"     Avg Speed: {stats['avg_speed']:.1f} km/h")
        print(f"     Idle Ratio: {stats['idle_ratio']:.1f}%")
        print(f"     Avg Power: {stats['avg_power']:.1f} kW")
    
    df_stats = pd.DataFrame(cluster_stats)
    df_stats.to_csv('./results/cluster_statistics_visual.csv', index=False, encoding='utf-8-sig')
    print(f"\n💾 Saved: cluster_statistics_visual.csv")
    
    return df_stats


# ==================== 雷达图 ====================
def plot_radar_chart(df_stats):
    """
    绘制雷达图
    """
    print("\n🎨 Creating radar chart...")
    
    # 选择6个特征
    features_to_plot = ['avg_speed', 'max_speed', 'speed_std', 'accel_std', 'avg_power', 'avg_trip_length']
    feature_labels = ['Avg Speed\n(km/h)', 'Max Speed\n(km/h)', 'Speed Std\n(km/h)', 
                     'Accel Std\n(m/s²)', 'Avg Power\n(kW)', 'Trip Length\n(points)']
    
    N = len(features_to_plot)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    
    fig = plt.figure(figsize=(14, 14))
    ax = plt.subplot(111, projection='polar')
    
    colors = ['#3498DB', '#2ECC71', '#E74C3C', '#F39C12']
    markers = ['o', 's', '^', 'D']
    
    for cluster_id in range(4):
        if cluster_id not in df_stats['cluster'].values:
            continue
        
        # 提取数值并归一化
        values = df_stats[df_stats['cluster'] == cluster_id][features_to_plot].values[0]
        
        # 归一化到[0, 1]
        max_vals = df_stats[features_to_plot].max().values
        values_norm = values / max_vals
        values_norm = np.append(values_norm, values_norm[0])
        
        ax.plot(angles, values_norm, 
               marker=markers[cluster_id],
               linewidth=3.5, 
               color=colors[cluster_id], 
               markersize=12,
               markeredgecolor='white',
               markeredgewidth=2,
               label=f'Cluster {cluster_id}',
               zorder=10)
        
        ax.fill(angles, values_norm, alpha=0.15, color=colors[cluster_id])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_labels, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=11)
    ax.grid(True, linestyle='--', linewidth=1.5, alpha=0.6)
    
    ax.set_title('Driving Behavior Clustering - Physical Features\n(Real Physical Units)', 
                fontsize=16, fontweight='bold', pad=30)
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), 
             fontsize=13, frameon=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig('./results/radar_chart.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: radar_chart.png")


# ==================== 综合分析图 ====================
def plot_comprehensive_analysis(df_stats, features, labels):
    """
    6子图综合分析
    """
    print("\n🎨 Creating comprehensive analysis...")
    
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    colors = ['#3498DB', '#2ECC71', '#E74C3C', '#F39C12']
    
    # 1. PCA空间
    ax1 = fig.add_subplot(gs[0, 0])
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)
    
    for cluster_id in range(4):
        if cluster_id in labels:
            mask = labels == cluster_id
            ax1.scatter(features_2d[mask, 0], features_2d[mask, 1],
                       c=colors[cluster_id], label=f'C{cluster_id}',
                       alpha=0.6, s=20, edgecolors='none')
    
    ax1.set_xlabel('PC1', fontsize=11, fontweight='bold')
    ax1.set_ylabel('PC2', fontsize=11, fontweight='bold')
    ax1.set_title('PCA Space', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2-6. 特征对比柱状图
    feature_plots = [
        ('avg_speed', 'Average Speed (km/h)'),
        ('idle_ratio', 'Idle Ratio (%)'),
        ('avg_power', 'Average Power (kW)'),
        ('avg_trip_length', 'Trip Length (points)'),
        ('soc_drop', 'SOC Drop per Trip (%)'),
    ]
    
    for idx, (feat, ylabel) in enumerate(feature_plots, start=1):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        
        if feat not in df_stats.columns:
            continue
        
        x = df_stats['cluster'].values.astype(int)
        values = df_stats[feat].values
        
        bars = ax.bar(x, values, color=[colors[i] for i in x], 
                      alpha=0.85, edgecolor='black', linewidth=2.5, width=0.65)
        
        ax.set_xlabel('Cluster', fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=10, fontweight='bold')
        ax.set_title(ylabel, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'C{i}' for i in x], fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 数值标签
        for cid, bar, val in zip(x, bars, values):
            height = bar.get_height()
            ax.text(cid, height, f'{val:.1f}', 
                   ha='center', va='bottom',
                   fontsize=10, fontweight='bold')
    
    plt.suptitle('Comprehensive Cluster Analysis (Real Physical Units)', 
                fontsize=18, fontweight='bold')
    plt.savefig('./results/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: comprehensive_analysis.png")


# ==================== 特征对比详细图 ====================
def plot_detailed_comparison(df_stats):
    """
    详细特征对比（多子图）
    """
    print("\n🎨 Creating detailed feature comparison...")
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    axes = axes.flatten()
    
    colors = ['#3498DB', '#2ECC71', '#E74C3C', '#F39C12']
    
    features_to_compare = [
        ('avg_speed', 'Average Speed (km/h)'),
        ('max_speed', 'Max Speed (km/h)'),
        ('speed_std', 'Speed Std (km/h)'),
        ('idle_ratio', 'Idle Ratio (%)'),
        ('avg_accel', 'Avg |Accel| (m/s²)'),
        ('accel_std', 'Accel Std (m/s²)'),
        ('avg_power', 'Avg Power (kW)'),
        ('soc_drop', 'SOC Drop per Trip (%)'),
        ('avg_trip_length', 'Trip Length (points)'),
    ]
    
    for idx, (feat, ylabel) in enumerate(features_to_compare):
        ax = axes[idx]
        
        if feat not in df_stats.columns:
            ax.text(0.5, 0.5, f'{feat} N/A', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            continue
        
        x = df_stats['cluster'].values.astype(int)
        values = df_stats[feat].values
        
        bars = ax.bar(x, values, color=[colors[i] for i in x], 
                      alpha=0.85, edgecolor='black', linewidth=2)
        
        ax.set_xlabel('Cluster', fontsize=10, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=9, fontweight='bold')
        ax.set_title(ylabel, fontsize=11, fontweight='bold', pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels([f'C{i}' for i in x])
        ax.grid(True, alpha=0.3, axis='y')
        
        # 标注最大/最小
        max_idx = np.argmax(values)
        min_idx = np.argmin(values)
        bars[max_idx].set_edgecolor('darkgreen')
        bars[max_idx].set_linewidth(3)
        bars[min_idx].set_edgecolor('darkred')
        bars[min_idx].set_linewidth(3)
        
        # 数值标签
        for cid, bar, val in zip(x, bars, values):
            height = bar.get_height()
            ax.text(cid, height, f'{val:.1f}', 
                   ha='center', va='bottom',
                   fontsize=9, fontweight='bold')
    
    plt.suptitle('Detailed Physical Feature Comparison', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('./results/detailed_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: detailed_comparison.png")


# ==================== Main ====================
def main():
    # 查找CSV文件
    csv_files = sorted(Path('.').glob('*_processed.csv'))
    
    if len(csv_files) == 0:
        print("\n⚠️  No CSV files found. Using saved statistics only.")
        # 尝试加载已有的统计文件
        if Path('./results/cluster_statistics_visual.csv').exists():
            df_stats = pd.read_csv('./results/cluster_statistics_visual.csv')
        else:
            print("❌ No statistics file found. Cannot visualize.")
            return
    else:
        print(f"\nFound {len(csv_files)} CSV files")
        
        # 重新分段并计算统计
        segments_by_cluster = resegment_and_analyze(csv_files, labels, max_segments=len(labels))
        df_stats = compute_cluster_statistics(segments_by_cluster)
    
    # 可视化
    plot_radar_chart(df_stats)
    plot_comprehensive_analysis(df_stats, features, labels)
    plot_detailed_comparison(df_stats)
    
    # 打印总结
    print("\n" + "="*70)
    print("📊 Cluster Summary")
    print("="*70)
    
    for _, row in df_stats.iterrows():
        cid = int(row['cluster'])
        print(f"\n🔷 Cluster {cid}:")
        print(f"   Samples: {int(row['count']):,}")
        print(f"   Avg Speed: {row['avg_speed']:.1f} km/h")
        print(f"   Max Speed: {row['max_speed']:.1f} km/h")
        print(f"   Idle Ratio: {row['idle_ratio']:.1f}%")
        print(f"   Avg Power: {row['avg_power']:.1f} kW")
        print(f"   SOC Drop: {row['soc_drop']:.1f}%")
    
    print("\n" + "="*70)
    print("✅ Visualization Complete!")
    print("="*70)
    print("\n📁 Generated files:")
    print("   ./results/radar_chart.png")
    print("   ./results/comprehensive_analysis.png")
    print("   ./results/detailed_comparison.png")
    print("   ./results/cluster_statistics_visual.csv")
    print("="*70)


if __name__ == "__main__":
    main()