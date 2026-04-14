"""
生成事件表 - 每个放电事件一行
包含：vehicle_id, 时间范围, 聚类类型, 所有物理指标
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

print("="*70)
print("📋 Generating Event Table")
print("="*70)


# ==================== 加载聚类结果 ====================
print("\n📂 Loading clustering results...")

features = np.load('./results/features.npy')
labels = np.load('./results/labels.npy')

print(f"✅ Loaded {len(labels):,} labels")


# ==================== 生成事件表 ====================
def generate_event_table(csv_files, labels):
    """
    重新分段，为每个事件生成一行记录
    """
    print("\n" + "="*70)
    print("📊 Generating Event Table")
    print("="*70)
    
    events = []
    segment_idx = 0
    total_segments = len(labels)
    
    required_cols = ['vehicle_id', 'time', 'soc', 'spd', 'v', 'i', 'acc']
    
    print(f"\n🚗 Processing {len(csv_files)} files...\n")
    
    for file_idx, file in enumerate(csv_files, 1):
        if segment_idx >= total_segments:
            break
        
        print(f"📂 File {file_idx}/{len(csv_files)}: {file.name}")
        
        chunk_iter = pd.read_csv(file, usecols=required_cols, chunksize=1000000)
        
        for chunk_idx, chunk in enumerate(chunk_iter, 1):
            if segment_idx >= total_segments:
                break
            
            print(f"   Chunk {chunk_idx}: ", end="")
            
            # 数据清洗
            chunk = chunk.dropna(subset=required_cols)
            chunk = chunk[
                (chunk['soc'] >= 0) & (chunk['soc'] <= 100) &
                (chunk['spd'] >= 0) & (chunk['spd'] <= 220) &
                (chunk['v'] > 0) & (chunk['v'] <= 1000) &
                (chunk['i'] >= -1000) & (chunk['i'] <= 1000) &
                (chunk['acc'] >= -10) & (chunk['acc'] <= 10)
            ]
            
            chunk_events = 0
            
            # 按车辆分段
            for vehicle_id in chunk['vehicle_id'].unique():
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
                        
                        if soc_current > soc_start:
                            start_idx = end_idx
                            break
                        
                        if soc_drop >= 3.0:
                            segment = vehicle_data.iloc[start_idx:end_idx+1]
                            
                            if len(segment) >= 10:
                                # 提取事件信息
                                event = extract_event_features(segment, segment_idx, labels[segment_idx])
                                events.append(event)
                                
                                segment_idx += 1
                                chunk_events += 1
                            
                            start_idx = end_idx + 1
                            break
                    else:
                        start_idx += 1
            
            print(f"Events: {chunk_events}, Total: {segment_idx}/{total_segments}")
    
    print(f"\n✅ Generated {len(events):,} event records")
    
    # 转换为DataFrame
    df_events = pd.DataFrame(events)
    
    return df_events


# ==================== 提取单个事件的特征 ====================
def extract_event_features(segment, event_id, cluster_label):
    """
    从一个片段中提取所有特征
    """
    event = {}
    
    # ==================== 基本信息 ====================
    event['event_id'] = event_id
    event['vehicle_id'] = segment['vehicle_id'].iloc[0]
    event['cluster'] = int(cluster_label)
    
    # 时间信息
    event['start_time'] = segment['time'].iloc[0]
    event['end_time'] = segment['time'].iloc[-1]
    event['duration_seconds'] = (event['end_time'] - event['start_time']) / 1000  # 假设时间戳是毫秒
    event['num_points'] = len(segment)
    
    # ==================== 速度特征 ====================
    speeds = segment['spd'].values
    
    event['speed_mean'] = float(speeds.mean())
    event['speed_std'] = float(speeds.std())
    event['speed_min'] = float(speeds.min())
    event['speed_max'] = float(speeds.max())
    event['speed_median'] = float(np.median(speeds))
    event['speed_p95'] = float(np.percentile(speeds, 95))
    
    # 速度分类占比
    event['idle_ratio'] = float((speeds < 1).mean())
    event['low_speed_ratio'] = float(((speeds >= 1) & (speeds < 40)).mean())
    event['medium_speed_ratio'] = float(((speeds >= 40) & (speeds < 60)).mean())
    event['high_speed_ratio'] = float((speeds >= 60).mean())
    
    # ==================== 加速度特征 ====================
    accels = segment['acc'].values
    
    event['accel_mean'] = float(accels.mean())
    event['accel_abs_mean'] = float(np.abs(accels).mean())
    event['accel_std'] = float(accels.std())
    event['accel_min'] = float(accels.min())
    event['accel_max'] = float(accels.max())
    
    # 加速/减速/平稳
    event['accel_positive_ratio'] = float((accels > 0.1).mean())
    event['accel_negative_ratio'] = float((accels < -0.1).mean())
    event['accel_steady_ratio'] = float((np.abs(accels) <= 0.1).mean())
    
    # 急加速/急减速
    event['harsh_accel_count'] = int(np.sum(accels > 2.0))
    event['harsh_brake_count'] = int(np.sum(accels < -2.0))
    
    # ==================== 能量特征 ====================
    soc = segment['soc'].values
    voltage = segment['v'].values
    current = segment['i'].values
    
    # SOC
    event['soc_start'] = float(soc[0])
    event['soc_end'] = float(soc[-1])
    event['soc_drop'] = float(soc[0] - soc[-1])
    event['soc_mean'] = float(soc.mean())
    event['soc_std'] = float(soc.std())
    
    # 电压
    event['voltage_mean'] = float(voltage.mean())
    event['voltage_std'] = float(voltage.std())
    event['voltage_min'] = float(voltage.min())
    event['voltage_max'] = float(voltage.max())
    
    # 电流
    event['current_mean'] = float(current.mean())
    event['current_abs_mean'] = float(np.abs(current).mean())
    event['current_std'] = float(current.std())
    event['current_min'] = float(current.min())
    event['current_max'] = float(current.max())
    
    # 功率 (kW)
    power = np.abs(voltage * current) / 1000
    event['power_mean'] = float(power.mean())
    event['power_std'] = float(power.std())
    event['power_min'] = float(power.min())
    event['power_max'] = float(power.max())
    event['power_p95'] = float(np.percentile(power, 95))
    
    # 总能耗 (kWh) = 平均功率 × 时间
    event['energy_consumption_kwh'] = event['power_mean'] * event['duration_seconds'] / 3600
    
    # ==================== 驾驶行为特征 ====================
    # 速度变化率
    speed_changes = np.abs(np.diff(speeds))
    event['speed_change_rate'] = float(speed_changes.mean()) if len(speed_changes) > 0 else 0
    
    # 怠速次数和时长
    idle_mask = speeds < 1
    if idle_mask.any():
        # 连续怠速片段
        idle_segments = []
        in_idle = False
        idle_start = 0
        
        for i, is_idle in enumerate(idle_mask):
            if is_idle and not in_idle:
                idle_start = i
                in_idle = True
            elif not is_idle and in_idle:
                idle_segments.append(i - idle_start)
                in_idle = False
        
        if in_idle:
            idle_segments.append(len(idle_mask) - idle_start)
        
        event['idle_count'] = len(idle_segments)
        event['idle_max_duration'] = max(idle_segments) if idle_segments else 0
        event['idle_total_duration'] = sum(idle_segments)
    else:
        event['idle_count'] = 0
        event['idle_max_duration'] = 0
        event['idle_total_duration'] = 0
    
    # ==================== 效率指标 ====================
    # 估算距离 (km) = 平均速度 × 时间
    event['distance_km'] = event['speed_mean'] * event['duration_seconds'] / 3600
    
    # 能效
    if event['distance_km'] > 0:
        event['efficiency_kwh_per_km'] = event['energy_consumption_kwh'] / event['distance_km']
        event['efficiency_soc_per_km'] = event['soc_drop'] / event['distance_km']
    else:
        event['efficiency_kwh_per_km'] = 0
        event['efficiency_soc_per_km'] = 0
    
    return event


# ==================== 数据质量检查 ====================
def check_data_quality(df):
    """
    检查数据质量
    """
    print("\n" + "="*70)
    print("🔍 Data Quality Check")
    print("="*70)
    
    print(f"\n📊 Basic Info:")
    print(f"   Total events: {len(df):,}")
    print(f"   Total vehicles: {df['vehicle_id'].nunique():,}")
    print(f"   Date range: {df['start_time'].min()} - {df['start_time'].max()}")
    
    print(f"\n🎯 Cluster Distribution:")
    for cluster_id in df['cluster'].unique():
        count = len(df[df['cluster'] == cluster_id])
        print(f"   Cluster {cluster_id}: {count:,} ({count/len(df)*100:.1f}%)")
    
    print(f"\n📈 Feature Ranges:")
    numeric_cols = ['speed_mean', 'accel_abs_mean', 'power_mean', 'soc_drop', 'duration_seconds']
    
    for col in numeric_cols:
        if col in df.columns:
            print(f"   {col}:")
            print(f"      Min: {df[col].min():.2f}")
            print(f"      Max: {df[col].max():.2f}")
            print(f"      Mean: {df[col].mean():.2f}")
            print(f"      Median: {df[col].median():.2f}")
    
    # 检查异常值
    print(f"\n⚠️  Anomaly Check:")
    anomalies = []
    
    if (df['speed_mean'] < 0).any() or (df['speed_mean'] > 220).any():
        anomalies.append(f"Speed out of range: {(df['speed_mean'] > 220).sum()} events")
    
    if (df['soc_drop'] < 0).any():
        anomalies.append(f"Negative SOC drop: {(df['soc_drop'] < 0).sum()} events")
    
    if (df['duration_seconds'] < 0).any():
        anomalies.append(f"Negative duration: {(df['duration_seconds'] < 0).sum()} events")
    
    if anomalies:
        for anomaly in anomalies:
            print(f"   ⚠️  {anomaly}")
    else:
        print(f"   ✅ No anomalies detected")


# ==================== 保存事件表 ====================
def save_event_table(df, filename='./results/event_table.csv'):
    """
    保存事件表到CSV
    """
    print("\n" + "="*70)
    print("💾 Saving Event Table")
    print("="*70)
    
    # 保存完整表
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"✅ Saved full table: {filename}")
    print(f"   Size: {len(df):,} rows × {len(df.columns)} columns")
    
    # 保存简化表（只包含关键列）
    key_columns = [
        'event_id', 'vehicle_id', 'cluster',
        'start_time', 'end_time', 'duration_seconds', 'num_points',
        'speed_mean', 'speed_max', 'speed_std', 'idle_ratio',
        'accel_abs_mean', 'accel_std',
        'power_mean', 'power_max',
        'soc_start', 'soc_end', 'soc_drop',
        'distance_km', 'energy_consumption_kwh',
        'efficiency_kwh_per_km', 'efficiency_soc_per_km'
    ]
    
    df_simple = df[key_columns]
    simple_filename = filename.replace('.csv', '_simple.csv')
    df_simple.to_csv(simple_filename, index=False, encoding='utf-8-sig')
    print(f"✅ Saved simplified table: {simple_filename}")
    print(f"   Size: {len(df_simple):,} rows × {len(df_simple.columns)} columns")
    
    # 打印列名列表
    print(f"\n📋 All columns ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i:2d}. {col}")


# ==================== 生成汇总统计 ====================
def generate_summary_stats(df):
    """
    按簇生成汇总统计
    """
    print("\n" + "="*70)
    print("📊 Summary Statistics by Cluster")
    print("="*70)
    
    # 按簇分组统计
    summary_cols = [
        'speed_mean', 'speed_max', 'speed_std', 'idle_ratio',
        'accel_abs_mean', 'accel_std',
        'power_mean', 'power_max',
        'soc_drop', 'duration_seconds', 'num_points',
        'distance_km', 'energy_consumption_kwh',
        'efficiency_kwh_per_km', 'efficiency_soc_per_km',
        'harsh_accel_count', 'harsh_brake_count'
    ]
    
    summary = df.groupby('cluster')[summary_cols].agg(['mean', 'std', 'median', 'min', 'max'])
    summary.to_csv('./results/cluster_summary_stats.csv', encoding='utf-8-sig')
    print(f"✅ Saved: cluster_summary_stats.csv")
    
    # 打印简要统计
    for cluster_id in df['cluster'].unique():
        cluster_data = df[df['cluster'] == cluster_id]
        
        print(f"\n🔷 Cluster {cluster_id} (n={len(cluster_data):,}):")
        print(f"   Speed: {cluster_data['speed_mean'].mean():.1f} ± {cluster_data['speed_mean'].std():.1f} km/h")
        print(f"   Idle ratio: {cluster_data['idle_ratio'].mean()*100:.1f}%")
        print(f"   Power: {cluster_data['power_mean'].mean():.1f} ± {cluster_data['power_mean'].std():.1f} kW")
        print(f"   SOC drop: {cluster_data['soc_drop'].mean():.2f} ± {cluster_data['soc_drop'].std():.2f} %")
        print(f"   Efficiency: {cluster_data['efficiency_kwh_per_km'].mean():.3f} kWh/km")


# ==================== Main ====================
def main():
    csv_files = sorted(Path('.').glob('*_processed.csv'))
    
    if len(csv_files) == 0:
        print("❌ No CSV files found")
        return
    
    print(f"\n📂 Found {len(csv_files)} CSV files")
    
    # 生成事件表
    df_events = generate_event_table(csv_files, labels)
    
    # 数据质量检查
    check_data_quality(df_events)
    
    # 保存
    save_event_table(df_events)
    
    # 汇总统计
    generate_summary_stats(df_events)
    
    print("\n" + "="*70)
    print("✅ Event Table Generation Complete!")
    print("="*70)
    print("\n📁 Generated files:")
    print("   ./results/event_table.csv (完整表)")
    print("   ./results/event_table_simple.csv (简化表)")
    print("   ./results/cluster_summary_stats.csv (汇总统计)")
    print("\n💡 Usage:")
    print("   df = pd.read_csv('./results/event_table.csv')")
    print("   df[df['cluster'] == 0].describe()")
    print("="*70)


if __name__ == "__main__":
    main()