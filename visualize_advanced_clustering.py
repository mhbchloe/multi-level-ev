"""
车辆聚类高级可视化
包含：PCA、雷达图、箱线图、时间分布、续航焦虑等
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from sklearn.decomposition import PCA

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("🎨 Advanced Clustering Visualization")
print("="*70)

# ==================== 加载数据 ====================
print("\n📂 Loading data...")

df_vehicles = pd.read_csv('./results/vehicle_features_clustered_advanced.csv')
df_profiles = pd.read_csv('./results/cluster_profiles_advanced.csv')

print(f"✅ Loaded {len(df_vehicles):,} vehicles in {df_vehicles['cluster'].nunique()} clusters")

k = df_vehicles['cluster'].nunique()
colors = plt.cm.Set3(np.linspace(0, 1, k))


# ==================== 1. 综合分析图（6子图） ====================
def plot_comprehensive_analysis():
    print("\n🎨 Creating comprehensive analysis...")
    
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 3, hspace=0.3, wspace=0.3)
    
    # 准备特征用于PCA
    feature_cols = [col for col in df_vehicles.columns 
                   if col not in ['vehicle_id', 'cluster'] and df_vehicles[col].dtype in [np.float64, np.int64]]
    
    X = df_vehicles[feature_cols].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 1. PCA空间
    ax1 = fig.add_subplot(gs[0, 0])
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)
    
    for cid in range(k):
        mask = df_vehicles['cluster'] == cid
        ax1.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=[colors[cid]], label=f'C{cid} (n={np.sum(mask):,})',
                   alpha=0.6, s=40, edgecolors='black', linewidth=0.5)
    
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontweight='bold')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontweight='bold')
    ax1.set_title('PCA Space', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2-6. 关键特征箱线图
    key_features = [
        ('range_anxiety_threshold', 'Range Anxiety\nThreshold (%)'),
        ('charging_trigger_soc', 'Charging Trigger\nSOC (%)'),
        ('mode_switch_rate', 'Mode Switch\nRate'),
        ('night_driving_ratio', 'Night Driving\nRatio'),
        ('speed_mean', 'Avg Speed\n(km/h)'),
    ]
    
    for idx, (feat, ylabel) in enumerate(key_features, start=1):
        if feat not in df_vehicles.columns:
            continue
        
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        
        data = [df_vehicles[df_vehicles['cluster'] == i][feat].values for i in range(k)]
        
        bp = ax.boxplot(data, labels=[f'C{i}' for i in range(k)],
                       patch_artist=True, widths=0.6, showfliers=False)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_xlabel('Cluster', fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold', fontsize=9)
        ax.set_title(ylabel.replace('\n', ' '), fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Vehicle Clustering Analysis (K={k}, Advanced Features)', 
                fontsize=18, fontweight='bold')
    plt.savefig('./results/comprehensive_advanced.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: comprehensive_advanced.png")


# ==================== 2. 雷达图��续航焦虑+模式切换+时间偏好） ====================
def plot_radar_chart():
    print("\n🎨 Creating radar chart...")
    
    # 选择6个关键特征（每个维度2个）
    radar_features = [
        'range_anxiety_threshold',  # 续航焦虑
        'charging_trigger_soc',
        'mode_switch_rate',  # 模式切换
        'mode_diversity',
        'night_driving_ratio',  # 时间偏好
        'morning_peak_ratio',
    ]
    
    labels_text = [
        'Range Anxiety\nThreshold',
        'Charging\nTrigger SOC',
        'Mode Switch\nRate',
        'Mode\nDiversity',
        'Night\nDriving',
        'Morning\nPeak',
    ]
    
    # 检查特征是否存在
    available_features = [f for f in radar_features if f in df_profiles.columns]
    available_labels = [labels_text[i] for i, f in enumerate(radar_features) if f in df_profiles.columns]
    
    if len(available_features) == 0:
        print("⚠️  No features available for radar chart")
        return
    
    N = len(available_features)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    
    fig = plt.figure(figsize=(12, 12))
    ax = plt.subplot(111, projection='polar')
    
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
    
    for cid in range(k):
        if cid not in df_profiles['cluster'].values:
            continue
        
        values = df_profiles[df_profiles['cluster'] == cid][available_features].values[0]
        
        # 归一化到[0, 1]
        max_vals = df_profiles[available_features].max().values
        min_vals = df_profiles[available_features].min().values
        values_norm = (values - min_vals) / (max_vals - min_vals + 1e-10)
        values_norm = np.append(values_norm, values_norm[0])
        
        ax.plot(angles, values_norm, marker=markers[cid],
               linewidth=3, color=colors[cid], markersize=10,
               markeredgecolor='white', markeredgewidth=2,
               label=f'Cluster {cid} (n={int(df_profiles[df_profiles["cluster"]==cid]["count"].values[0]):,})')
        ax.fill(angles, values_norm, alpha=0.15, color=colors[cid])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(available_labels, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    ax.grid(True, linestyle='--', linewidth=1.5, alpha=0.6)
    
    ax.set_title('Vehicle Clustering - Advanced Features\n(Range Anxiety + Mode Switch + Temporal)', 
                fontsize=15, fontweight='bold', pad=25)
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    
    plt.tight_layout()
    plt.savefig('./results/radar_advanced.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: radar_advanced.png")


# ==================== 3. 详细特征对比（12子图） ====================
def plot_detailed_features():
    print("\n🎨 Creating detailed feature comparison...")
    
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    axes = axes.flatten()
    
    detailed_features = [
        ('range_anxiety_threshold', 'Range Anxiety Threshold (%)'),
        ('charging_trigger_soc', 'Charging Trigger SOC (%)'),
        ('critical_soc_ratio', 'Critical SOC Ratio'),
        ('mode_switch_rate', 'Mode Switch Rate'),
        ('mode_diversity', 'Mode Diversity (Entropy)'),
        ('aggressive_switch_rate', 'Aggressive Switch Rate'),
        ('night_driving_ratio', 'Night Driving Ratio'),
        ('morning_peak_ratio', 'Morning Peak Ratio'),
        ('evening_peak_ratio', 'Evening Peak Ratio'),
        ('temporal_concentration', 'Temporal Concentration'),
        ('speed_mean', 'Avg Speed (km/h)'),
        ('efficiency_kwh_per_km', 'Efficiency (kWh/km)'),
    ]
    
    for idx, (feat, ylabel) in enumerate(detailed_features):
        if feat not in df_vehicles.columns:
            axes[idx].text(0.5, 0.5, f'{feat}\nN/A', ha='center', va='center',
                          transform=axes[idx].transAxes)
            continue
        
        ax = axes[idx]
        
        data = [df_vehicles[df_vehicles['cluster'] == i][feat].values for i in range(k)]
        
        bp = ax.boxplot(data, labels=[f'C{i}' for i in range(k)],
                       patch_artist=True, widths=0.6, showfliers=False)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # 标注均值
        for i in range(k):
            mean_val = df_vehicles[df_vehicles['cluster'] == i][feat].mean()
            ax.text(i+1, mean_val, f'{mean_val:.2f}',
                   ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax.set_xlabel('Cluster', fontweight='bold', fontsize=9)
        ax.set_ylabel(ylabel, fontweight='bold', fontsize=8)
        ax.set_title(ylabel, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Detailed Feature Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('./results/detailed_features_advanced.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: detailed_features_advanced.png")


# ==================== 4. 时间偏好分布（堆叠柱状图） ====================
def plot_temporal_patterns():
    print("\n🎨 Creating temporal pattern analysis...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 时段分布
    ax1 = axes[0]
    
    time_features = ['night_driving_ratio', 'morning_peak_ratio', 'daytime_ratio', 'evening_peak_ratio']
    time_labels = ['Night\n(22-6)', 'Morning\n(7-9)', 'Day\n(9-17)', 'Evening\n(17-19)']
    
    available_time_features = [f for f in time_features if f in df_profiles.columns]
    available_time_labels = [time_labels[i] for i, f in enumerate(time_features) if f in df_profiles.columns]
    
    if len(available_time_features) > 0:
        x = np.arange(k)
        width = 0.2
        
        for i, (feat, label) in enumerate(zip(available_time_features, available_time_labels)):
            values = [df_profiles[df_profiles['cluster']==cid][feat].values[0] * 100 
                     if cid in df_profiles['cluster'].values else 0 
                     for cid in range(k)]
            ax1.bar(x + i*width, values, width, label=label, alpha=0.8)
        
        ax1.set_xlabel('Cluster', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Driving Ratio (%)', fontweight='bold', fontsize=12)
        ax1.set_title('Temporal Driving Patterns', fontsize=14, fontweight='bold')
        ax1.set_xticks(x + width * 1.5)
        ax1.set_xticklabels([f'C{i}' for i in range(k)])
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
    
    # 周末 vs 工作日
    ax2 = axes[1]
    
    if 'weekend_ratio' in df_profiles.columns:
        weekend_ratios = []
        weekday_ratios = []
        
        for cid in range(k):
            if cid in df_profiles['cluster'].values:
                weekend = df_profiles[df_profiles['cluster']==cid]['weekend_ratio'].values[0] * 100
                weekday = 100 - weekend
            else:
                weekend, weekday = 0, 0
            
            weekend_ratios.append(weekend)
            weekday_ratios.append(weekday)
        
        x = np.arange(k)
        width = 0.5
        
        ax2.bar(x, weekday_ratios, width, label='Weekday', color='steelblue', alpha=0.8)
        ax2.bar(x, weekend_ratios, width, bottom=weekday_ratios, 
               label='Weekend', color='coral', alpha=0.8)
        
        ax2.set_xlabel('Cluster', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Driving Ratio (%)', fontweight='bold', fontsize=12)
        ax2.set_title('Weekday vs Weekend Driving', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'C{i}' for i in range(k)])
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('./results/temporal_patterns.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: temporal_patterns.png")


# ==================== 5. 续航焦虑分析 ====================
def plot_range_anxiety():
    print("\n🎨 Creating range anxiety analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 续航焦虑阈值分布
    ax1 = axes[0, 0]
    
    if 'range_anxiety_threshold' in df_vehicles.columns:
        for cid in range(k):
            data = df_vehicles[df_vehicles['cluster'] == cid]['range_anxiety_threshold']
            ax1.hist(data, bins=30, alpha=0.6, color=colors[cid], 
                    label=f'C{cid}', edgecolor='black', linewidth=0.5)
        
        ax1.set_xlabel('Range Anxiety Threshold (%)', fontweight='bold')
        ax1.set_ylabel('Count', fontweight='bold')
        ax1.set_title('Range Anxiety Threshold Distribution', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. 充电触发SOC
    ax2 = axes[0, 1]
    
    if 'charging_trigger_soc' in df_vehicles.columns:
        x = np.arange(k)
        values = [df_vehicles[df_vehicles['cluster']==cid]['charging_trigger_soc'].mean() 
                 for cid in range(k)]
        std_vals = [df_vehicles[df_vehicles['cluster']==cid]['charging_trigger_soc'].std() 
                   for cid in range(k)]
        
        bars = ax2.bar(x, values, yerr=std_vals, color=colors, alpha=0.8,
                      edgecolor='black', linewidth=2, capsize=5)
        
        ax2.set_xlabel('Cluster', fontweight='bold')
        ax2.set_ylabel('Charging Trigger SOC (%)', fontweight='bold')
        ax2.set_title('Charging Trigger SOC', fontsize=13, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'C{i}' for i in range(k)])
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 3. 低SOC事件比例
    ax3 = axes[1, 0]
    
    if 'critical_soc_ratio' in df_vehicles.columns:
        data = [df_vehicles[df_vehicles['cluster']==cid]['critical_soc_ratio']*100 
               for cid in range(k)]
        
        bp = ax3.boxplot(data, labels=[f'C{i}' for i in range(k)],
                        patch_artist=True, widths=0.6)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax3.set_xlabel('Cluster', fontweight='bold')
        ax3.set_ylabel('Critical SOC Event Ratio (%)', fontweight='bold')
        ax3.set_title('Low SOC Risk Events', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. SOC使用范围
    ax4 = axes[1, 1]
    
    if 'soc_usage_range' in df_vehicles.columns:
        x = np.arange(k)
        values = [df_vehicles[df_vehicles['cluster']==cid]['soc_usage_range'].mean() 
                 for cid in range(k)]
        
        bars = ax4.bar(x, values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        
        ax4.set_xlabel('Cluster', fontweight='bold')
        ax4.set_ylabel('SOC Usage Range (%)', fontweight='bold')
        ax4.set_title('SOC Usage Range', fontsize=13, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels([f'C{i}' for i in range(k)])
        ax4.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Range Anxiety Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('./results/range_anxiety_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: range_anxiety_analysis.png")


# ==================== 6. 模式切换分析 ====================
def plot_mode_switching():
    print("\n🎨 Creating mode switching analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 模式切换率
    ax1 = axes[0, 0]
    
    if 'mode_switch_rate' in df_vehicles.columns:
        data = [df_vehicles[df_vehicles['cluster']==cid]['mode_switch_rate'] 
               for cid in range(k)]
        
        bp = ax1.boxplot(data, labels=[f'C{i}' for i in range(k)],
                        patch_artist=True, widths=0.6)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_xlabel('Cluster', fontweight='bold')
        ax1.set_ylabel('Mode Switch Rate', fontweight='bold')
        ax1.set_title('Driving Mode Switch Rate', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. 模式多样性
    ax2 = axes[0, 1]
    
    if 'mode_diversity' in df_vehicles.columns:
        x = np.arange(k)
        values = [df_vehicles[df_vehicles['cluster']==cid]['mode_diversity'].mean() 
                 for cid in range(k)]
        
        bars = ax2.bar(x, values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        
        ax2.set_xlabel('Cluster', fontweight='bold')
        ax2.set_ylabel('Mode Diversity (Entropy)', fontweight='bold')
        ax2.set_title('Driving Mode Diversity', fontsize=13, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'C{i}' for i in range(k)])
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. 激进模式切换
    ax3 = axes[1, 0]
    
    if 'aggressive_switch_rate' in df_vehicles.columns:
        data = [df_vehicles[df_vehicles['cluster']==cid]['aggressive_switch_rate'] 
               for cid in range(k)]
        
        bp = ax3.boxplot(data, labels=[f'C{i}' for i in range(k)],
                        patch_artist=True, widths=0.6)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax3.set_xlabel('Cluster', fontweight='bold')
        ax3.set_ylabel('Aggressive Switch Rate', fontweight='bold')
        ax3.set_title('Aggressive Mode Switch Rate', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. 主导模式分布
    ax4 = axes[1, 1]
    
    if 'dominant_mode' in df_vehicles.columns:
        mode_counts = []
        for cid in range(k):
            cluster_data = df_vehicles[df_vehicles['cluster'] == cid]
            mode_dist = cluster_data['dominant_mode'].value_counts()
            mode_counts.append(mode_dist)
        
        # 堆叠柱状图
        x = np.arange(k)
        width = 0.6
        bottom = np.zeros(k)
        
        for mode_id in range(4):
            heights = [mode_counts[cid].get(mode_id, 0) if cid < len(mode_counts) else 0 
                      for cid in range(k)]
            ax4.bar(x, heights, width, bottom=bottom, 
                   label=f'Event Mode {mode_id}', alpha=0.8)
            bottom += heights
        
        ax4.set_xlabel('Cluster', fontweight='bold')
        ax4.set_ylabel('Vehicle Count', fontweight='bold')
        ax4.set_title('Dominant Event Mode Distribution', fontsize=13, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels([f'C{i}' for i in range(k)])
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Mode Switching Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('./results/mode_switching_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: mode_switching_analysis.png")


# ==================== Main ====================
def main():
    plot_comprehensive_analysis()
    plot_radar_chart()
    plot_detailed_features()
    plot_temporal_patterns()
    plot_range_anxiety()
    plot_mode_switching()
    
    print("\n" + "="*70)
    print("✅ All Visualizations Complete!")
    print("="*70)
    print("\n📁 Generated files:")
    print("   1. comprehensive_advanced.png - 综合分析（PCA+5特征）")
    print("   2. radar_advanced.png - 雷达图（6维特征）")
    print("   3. detailed_features_advanced.png - 详细特征对比（12个）")
    print("   4. temporal_patterns.png - 时间偏好分析")
    print("   5. range_anxiety_analysis.png - 续航焦虑分析（4子图）")
    print("   6. mode_switching_analysis.png - 模式切换分析（4子图）")
    print("="*70)


if __name__ == "__main__":
    main()