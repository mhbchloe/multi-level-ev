"""
精确映射版本 - 增强日志输出
显示：车辆数、分段数、进度等详细信息
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from tqdm import tqdm
from sklearn.decomposition import PCA
from collections import defaultdict

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("🎨 Exact Mapping Visualization (Enhanced Logging)")
print("="*70)


# ==================== 加载聚类结果 ====================
print("\n📂 Loading clustering results...")

features = np.load('./results/features.npy')
labels = np.load('./results/labels.npy')

print(f"✅ Loaded:")
print(f"   Features: {features.shape}")
print(f"   Labels: {labels.shape}")
print(f"   Expected segments: {len(labels):,}")
print(f"   Clusters: {np.unique(labels)}")

# 统计每个簇的样本数
for cluster_id in range(4):
    count = np.sum(labels == cluster_id)
    print(f"      Cluster {cluster_id}: {count:,} ({count/len(labels)*100:.1f}%)")


# ==================== 重新分段并打标签（增强日志） ====================
def resegment_with_labels_enhanced(csv_files, labels):
    """
    重新分段，并提供详细的统计信息
    """
    print("\n" + "="*70)
    print("📊 Re-segmenting with Exact Label Mapping")
    print("="*70)
    
    # 存储结构
    segments_by_cluster = {0: [], 1: [], 2: [], 3: []}
    
    # 统计信息
    segment_idx = 0
    total_segments = len(labels)
    
    # 车辆统计
    vehicle_segments = defaultdict(int)  # 每辆车的片段数
    vehicle_clusters = defaultdict(lambda: defaultdict(int))  # 每辆车每个簇的片段数
    
    # 文件统计
    file_stats = []
    
    required_cols = ['vehicle_id', 'time', 'soc', 'spd', 'v', 'i', 'acc']
    
    print(f"\n🚗 Processing {len(csv_files)} CSV files...\n")
    
    for file_idx, file in enumerate(csv_files, 1):
        if segment_idx >= total_segments:
            print(f"\n✅ Reached target: {total_segments:,} segments")
            break
        
        print(f"{'='*70}")
        print(f"📂 File {file_idx}/{len(csv_files)}: {file.name}")
        print(f"{'='*70}")
        
        file_start_idx = segment_idx
        file_vehicles = set()
        
        # 逐块读取
        chunk_iter = pd.read_csv(file, usecols=required_cols, chunksize=1000000)
        
        for chunk_idx, chunk in enumerate(chunk_iter, 1):
            if segment_idx >= total_segments:
                break
            
            chunk_start = segment_idx
            
            # 数据清洗
            original_len = len(chunk)
            chunk = chunk.dropna(subset=required_cols)
            chunk = chunk[
                (chunk['soc'] >= 0) & (chunk['soc'] <= 100) &
                (chunk['spd'] >= 0) & (chunk['spd'] <= 220) &
                (chunk['v'] > 0) & (chunk['v'] <= 1000) &
                (chunk['i'] >= -1000) & (chunk['i'] <= 1000) &
                (chunk['acc'] >= -10) & (chunk['acc'] <= 10)
            ]
            cleaned_len = len(chunk)
            
            chunk_vehicles = chunk['vehicle_id'].unique()
            file_vehicles.update(chunk_vehicles)
            
            print(f"   Chunk {chunk_idx}:")
            print(f"      Records: {original_len:,} → {cleaned_len:,} (cleaned)")
            print(f"      Vehicles in chunk: {len(chunk_vehicles)}")
            
            # 按车辆分段
            chunk_segments = 0
            
            for vehicle_id in chunk_vehicles:
                if segment_idx >= total_segments:
                    break
                
                vehicle_data = chunk[chunk['vehicle_id'] == vehicle_id].sort_values('time')
                
                if len(vehicle_data) < 10:
                    continue
                
                soc_values = vehicle_data['soc'].values
                start_idx = 0
                
                # 按SOC下降≥3%分段
                while start_idx < len(vehicle_data):
                    if segment_idx >= total_segments:
                        break
                    
                    soc_start = soc_values[start_idx]
                    
                    for end_idx in range(start_idx + 1, len(vehicle_data)):
                        soc_current = soc_values[end_idx]
                        soc_drop = soc_start - soc_current
                        
                        # SOC上升，重新开始
                        if soc_current > soc_start:
                            start_idx = end_idx
                            break
                        
                        # SOC下降≥3%，保存片段
                        if soc_drop >= 3.0:
                            segment = vehicle_data.iloc[start_idx:end_idx+1].copy()
                            
                            if len(segment) >= 10:
                                # 使用labels[segment_idx]作为标签
                                cluster_id = labels[segment_idx]
                                segments_by_cluster[cluster_id].append(segment)
                                
                                # 统计
                                vehicle_segments[vehicle_id] += 1
                                vehicle_clusters[vehicle_id][cluster_id] += 1
                                
                                segment_idx += 1
                                chunk_segments += 1
                            
                            start_idx = end_idx + 1
                            break
                    else:
                        start_idx += 1
            
            print(f"      Segments found: {chunk_segments}")
            print(f"      Progress: {segment_idx}/{total_segments} ({segment_idx/total_segments*100:.1f}%)")
        
        # 文件统计
        file_segments = segment_idx - file_start_idx
        file_stats.append({
            'file': file.name,
            'vehicles': len(file_vehicles),
            'segments': file_segments
        })
        
        print(f"\n   📊 File Summary:")
        print(f"      Total vehicles: {len(file_vehicles)}")
        print(f"      Total segments: {file_segments:,}")
        print(f"      Overall progress: {segment_idx}/{total_segments} ({segment_idx/total_segments*100:.1f}%)")
        print()
    
    # ==================== 最终统计 ====================
    print("\n" + "="*70)
    print("📊 Final Statistics")
    print("="*70)
    
    total_vehicles = len(vehicle_segments)
    total_collected = sum(len(segs) for segs in segments_by_cluster.values())
    
    print(f"\n🚗 Vehicle Statistics:")
    print(f"   Total vehicles: {total_vehicles:,}")
    print(f"   Avg segments per vehicle: {total_collected/total_vehicles:.1f}")
    
    # 车辆片段分布
    seg_counts = list(vehicle_segments.values())
    print(f"\n   Segment distribution per vehicle:")
    print(f"      Min: {min(seg_counts)} segments")
    print(f"      Max: {max(seg_counts)} segments")
    print(f"      Median: {np.median(seg_counts):.0f} segments")
    print(f"      Mean: {np.mean(seg_counts):.1f} segments")
    
    # Top 10 车辆
    top_vehicles = sorted(vehicle_segments.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\n   Top 10 vehicles by segment count:")
    for i, (vid, count) in enumerate(top_vehicles, 1):
        print(f"      {i:2d}. Vehicle {vid}: {count:,} segments")
    
    print(f"\n📦 Segment Statistics:")
    print(f"   Total segments collected: {total_collected:,}")
    print(f"   Expected: {total_segments:,}")
    
    if total_collected == total_segments:
        print(f"   ✅ Perfect match!")
    else:
        print(f"   ⚠️  Mismatch: {abs(total_collected - total_segments):,} difference")
    
    print(f"\n🎯 Cluster Distribution:")
    for cluster_id in range(4):
        count = len(segments_by_cluster[cluster_id])
        pct = count / total_collected * 100 if total_collected > 0 else 0
        
        # 统计有多少车辆有这个簇的片段
        vehicles_in_cluster = sum(1 for v_clusters in vehicle_clusters.values() 
                                  if cluster_id in v_clusters)
        
        print(f"   Cluster {cluster_id}:")
        print(f"      Segments: {count:,} ({pct:.1f}%)")
        print(f"      Vehicles: {vehicles_in_cluster:,} vehicles have segments in this cluster")
    
    # 每个文件的统计
    print(f"\n📂 Per-File Statistics:")
    for stat in file_stats:
        print(f"   {stat['file']}:")
        print(f"      Vehicles: {stat['vehicles']}")
        print(f"      Segments: {stat['segments']:,}")
    
    # 保存详细统计
    vehicle_stats_df = pd.DataFrame([
        {
            'vehicle_id': vid,
            'total_segments': vehicle_segments[vid],
            'cluster_0': vehicle_clusters[vid][0],
            'cluster_1': vehicle_clusters[vid][1],
            'cluster_2': vehicle_clusters[vid][2],
            'cluster_3': vehicle_clusters[vid][3],
        }
        for vid in vehicle_segments.keys()
    ])
    vehicle_stats_df.to_csv('./results/vehicle_statistics.csv', index=False, encoding='utf-8-sig')
    print(f"\n💾 Saved vehicle statistics: ./results/vehicle_statistics.csv")
    
    return segments_by_cluster, vehicle_segments


# ==================== 计算精确的物理统计 ====================
def compute_exact_statistics(segments_by_cluster):
    """
    计算每个簇的精确物理统计特征
    """
    print("\n" + "="*70)
    print("📊 Computing Exact Physical Statistics")
    print("="*70)
    
    cluster_stats = []
    
    for cluster_id in range(4):
        segs = segments_by_cluster.get(cluster_id, [])
        
        if len(segs) == 0:
            print(f"\n  Cluster {cluster_id}: ⚠️  No data")
            continue
        
        print(f"\n  Cluster {cluster_id}: {len(segs):,} segments")
        
        # 合并所有片段的数据
        all_data = pd.concat(segs, ignore_index=True)
        
        stats = {
            'cluster': cluster_id,
            'count': len(segs)
        }
        
        # 速度特征
        stats['avg_speed'] = all_data['spd'].mean()
        stats['max_speed'] = all_data['spd'].quantile(0.95)
        stats['min_speed'] = all_data['spd'].min()
        stats['speed_std'] = all_data['spd'].std()
        stats['speed_median'] = all_data['spd'].median()
        
        # 速度分布
        stats['idle_ratio'] = (all_data['spd'] < 1).mean() * 100
        stats['low_speed_ratio'] = ((all_data['spd'] >= 1) & (all_data['spd'] < 40)).mean() * 100
        stats['high_speed_ratio'] = (all_data['spd'] >= 60).mean() * 100
        
        # 加速度特征
        stats['avg_accel'] = all_data['acc'].abs().mean()
        stats['accel_std'] = all_data['acc'].std()
        stats['accel_positive_ratio'] = (all_data['acc'] > 0.1).mean() * 100
        stats['accel_negative_ratio'] = (all_data['acc'] < -0.1).mean() * 100
        
        # 能量特征
        stats['avg_soc'] = all_data['soc'].mean()
        stats['avg_voltage'] = all_data['v'].mean()
        stats['avg_current'] = all_data['i'].mean()
        
        # 每个片段的SOC下降
        soc_drops = []
        for seg in segs:
            if len(seg) > 1:
                soc_drops.append(seg['soc'].iloc[0] - seg['soc'].iloc[-1])
        stats['soc_drop_mean'] = np.mean(soc_drops) if soc_drops else 0
        stats['soc_drop_std'] = np.std(soc_drops) if soc_drops else 0
        
        # 功率 (kW)
        power = (all_data['v'] * all_data['i']).abs() / 1000
        stats['avg_power'] = power.mean()
        stats['max_power'] = power.quantile(0.95)
        stats['power_std'] = power.std()
        
        # 行程特征
        trip_lengths = [len(seg) for seg in segs]
        stats['avg_trip_length'] = np.mean(trip_lengths)
        stats['trip_length_std'] = np.std(trip_lengths)
        stats['min_trip_length'] = np.min(trip_lengths)
        stats['max_trip_length'] = np.max(trip_lengths)
        
        cluster_stats.append(stats)
        
        # 打印关键统计
        print(f"     Speed: {stats['avg_speed']:.1f} km/h (±{stats['speed_std']:.1f})")
        print(f"     Idle: {stats['idle_ratio']:.1f}%, High speed: {stats['high_speed_ratio']:.1f}%")
        print(f"     Power: {stats['avg_power']:.1f} kW (±{stats['power_std']:.1f})")
        print(f"     SOC Drop: {stats['soc_drop_mean']:.2f}% (±{stats['soc_drop_std']:.2f})")
        print(f"     Trip Length: {stats['avg_trip_length']:.1f} pts (±{stats['trip_length_std']:.1f})")
    
    df = pd.DataFrame(cluster_stats)
    df.to_csv('./results/cluster_stats_exact.csv', index=False, encoding='utf-8-sig')
    print(f"\n💾 Saved: cluster_stats_exact.csv")
    
    return df


# ==================== 雷达图 ====================
def plot_radar(df_stats):
    print("\n🎨 Creating radar chart...")
    
    features = ['avg_speed', 'max_speed', 'speed_std', 'accel_std', 'avg_power', 'avg_trip_length']
    labels_text = ['Avg Speed\n(km/h)', 'Max Speed\n(km/h)', 'Speed Std\n(km/h)', 
                   'Accel Std\n(m/s²)', 'Avg Power\n(kW)', 'Trip Length\n(points)']
    
    N = len(features)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    
    fig = plt.figure(figsize=(12, 12))
    ax = plt.subplot(111, projection='polar')
    
    colors = ['#3498DB', '#2ECC71', '#E74C3C', '#F39C12']
    markers = ['o', 's', '^', 'D']
    
    for cluster_id in range(4):
        if cluster_id not in df_stats['cluster'].values:
            continue
        
        values = df_stats[df_stats['cluster'] == cluster_id][features].values[0]
        max_vals = df_stats[features].max().values
        values_norm = values / max_vals
        values_norm = np.append(values_norm, values_norm[0])
        
        ax.plot(angles, values_norm, marker=markers[cluster_id],
               linewidth=3, color=colors[cluster_id], markersize=10,
               markeredgecolor='white', markeredgewidth=2,
               label=f'Cluster {cluster_id} (n={int(df_stats[df_stats["cluster"]==cluster_id]["count"].values[0]):,})')
        ax.fill(angles, values_norm, alpha=0.15, color=colors[cluster_id])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels_text, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_title('Driving Behavior Clustering - Exact Physical Features\n(All Segments, Real Physical Units)', 
                fontsize=15, fontweight='bold', pad=25)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    
    plt.tight_layout()
    plt.savefig('./results/radar_exact.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: radar_exact.png")


# ==================== 综合分析图 ====================
def plot_comprehensive(df_stats, features, labels):
    print("\n🎨 Creating comprehensive analysis...")
    
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 3, hspace=0.3, wspace=0.3)
    
    colors = ['#3498DB', '#2ECC71', '#E74C3C', '#F39C12']
    
    # PCA
    ax1 = fig.add_subplot(gs[0, 0])
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)
    
    for cid in range(4):
        mask = labels == cid
        count = np.sum(mask)
        ax1.scatter(features_2d[mask, 0], features_2d[mask, 1],
                   c=colors[cid], label=f'C{cid} (n={count:,})', alpha=0.5, s=15)
    
    ax1.set_xlabel('PC1', fontsize=11, fontweight='bold')
    ax1.set_ylabel('PC2', fontsize=11, fontweight='bold')
    ax1.set_title('PCA Space', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 5个特征柱状图
    feature_plots = [
        ('avg_speed', 'Avg Speed (km/h)'),
        ('idle_ratio', 'Idle Ratio (%)'),
        ('avg_power', 'Avg Power (kW)'),
        ('avg_trip_length', 'Trip Length (pts)'),
        ('soc_drop_mean', 'SOC Drop (%)'),
    ]
    
    for idx, (feat, ylabel) in enumerate(feature_plots, 1):
        ax = fig.add_subplot(gs[idx//3, idx%3])
        
        if feat not in df_stats.columns:
            continue
        
        x = df_stats['cluster'].values.astype(int)
        vals = df_stats[feat].values
        
        bars = ax.bar(x, vals, color=[colors[i] for i in x],
                      alpha=0.85, edgecolor='black', linewidth=2)
        
        ax.set_xlabel('Cluster', fontweight='bold', fontsize=10)
        ax.set_ylabel(ylabel, fontweight='bold', fontsize=9)
        ax.set_title(ylabel, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'C{i}' for i in x], fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 标注数值和样本数
        for cid, bar, v in zip(x, bars, vals):
            height = bar.get_height()
            count = int(df_stats[df_stats['cluster']==cid]['count'].values[0])
            ax.text(cid, height, f'{v:.1f}\n(n={count:,})',
                   ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.suptitle('Comprehensive Analysis - Exact Mapping', 
                fontsize=18, fontweight='bold')
    plt.savefig('./results/comprehensive_exact.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: comprehensive_exact.png")


# ==================== Main ====================
def main():
    csv_files = sorted(Path('.').glob('*_processed.csv'))
    
    if len(csv_files) == 0:
        print("❌ No CSV files found")
        return
    
    print(f"\n📂 Found {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"   - {f.name}")
    
    # 重新分段并精确映射labels（增强日志）
    segments, vehicle_stats = resegment_with_labels_enhanced(csv_files, labels)
    
    # 计算精确统计
    df_stats = compute_exact_statistics(segments)
    
    # 可视化
    plot_radar(df_stats)
    plot_comprehensive(df_stats, features, labels)
    
    print("\n" + "="*70)
    print("✅ Complete!")
    print("="*70)
    print("\n📁 Generated files:")
    print("   ./results/radar_exact.png")
    print("   ./results/comprehensive_exact.png")
    print("   ./results/cluster_stats_exact.csv")
    print("   ./results/vehicle_statistics.csv")
    print("="*70)


if __name__ == "__main__":
    main()