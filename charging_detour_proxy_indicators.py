"""
充电绕行的替代指标
无法计算真实绕行距离，但可以用这些指标近似反映充电便利性需求
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("="*70)
print("🛣️ Charging Detour Proxy Indicators")
print("="*70)

# 加载数据
df_events = pd.read_csv('./results/event_table.csv')
df_vehicles = pd.read_csv('./results/vehicle_features_clustered_advanced.csv')

print(f"✅ Loaded {len(df_events):,} events, {len(df_vehicles):,} vehicles")


# ==================== 方法1：充电紧急度指数 ====================
def compute_charging_urgency_index(df_events):
    """
    充电紧急度 = 当事件结束时SOC很低，说明"必须马上充"
    高紧急度 → 可能接受绕行
    低紧急度 → 可能等到方便时再充
    """
    print("\n" + "="*70)
    print("⚡ Method 1: Charging Urgency Index")
    print("="*70)
    
    charging_urgency = []
    
    for vehicle_id in df_events['vehicle_id'].unique():
        vehicle_events = df_events[df_events['vehicle_id'] == vehicle_id].sort_values('start_time')
        
        for i in range(len(vehicle_events) - 1):
            current = vehicle_events.iloc[i]
            next_event = vehicle_events.iloc[i+1]
            
            # 如果下次起始SOC > 当前结束SOC，说明充电了
            if next_event['soc_start'] > current['soc_end']:
                urgency = {
                    'vehicle_id': vehicle_id,
                    'soc_before_charging': current['soc_end'],
                    
                    # 紧急度指数（0-1，越大越紧急）
                    'urgency_index': 1 - (current['soc_end'] / 100),
                    
                    # 分类
                    'urgency_level': (
                        'Critical' if current['soc_end'] < 20 else
                        'High' if current['soc_end'] < 40 else
                        'Medium' if current['soc_end'] < 60 else
                        'Low'
                    ),
                    
                    # 推测的绕行意愿
                    'detour_willingness': (
                        'High' if current['soc_end'] < 20 else  # 愿意绕远路充电
                        'Medium' if current['soc_end'] < 40 else
                        'Low'  # 可能只在顺路充电桩充
                    )
                }
                charging_urgency.append(urgency)
    
    df_urgency = pd.DataFrame(charging_urgency)
    
    print(f"\n✅ Charging urgency analysis:")
    print(f"   Total charging events: {len(df_urgency):,}")
    print(f"\n   Urgency distribution:")
    for level in ['Critical', 'High', 'Medium', 'Low']:
        count = (df_urgency['urgency_level'] == level).sum()
        pct = count / len(df_urgency) * 100
        print(f"      {level:10s}: {count:,} ({pct:.1f}%)")
    
    print(f"\n   Detour willingness:")
    for willingness in ['High', 'Medium', 'Low']:
        count = (df_urgency['detour_willingness'] == willingness).sum()
        pct = count / len(df_urgency) * 100
        print(f"      {willingness:10s}: {count:,} ({pct:.1f}%)")
    
    return df_urgency


# ==================== 方法2：充电便利性需求指数 ====================
def compute_charging_convenience_need(df_vehicles):
    """
    便利性需求 = 充电频率高 + 焦虑阈值高 → 需要桩网密集（不愿绕行）
    """
    print("\n" + "="*70)
    print("🏪 Method 2: Charging Convenience Need Index")
    print("="*70)
    
    # 计算便利性需求指数
    df_vehicles['charging_frequency_norm'] = df_vehicles['total_events'] / df_vehicles['total_events'].max()
    df_vehicles['anxiety_norm'] = df_vehicles['range_anxiety_threshold'] / 100
    
    # 便利性需求 = 高频充电 × 高焦虑
    df_vehicles['convenience_need_index'] = (
        df_vehicles['charging_frequency_norm'] * 0.5 +
        df_vehicles['anxiety_norm'] * 0.5
    )
    
    # 分类
    df_vehicles['convenience_need_level'] = pd.cut(
        df_vehicles['convenience_need_index'],
        bins=[0, 0.3, 0.6, 1.0],
        labels=['Low', 'Medium', 'High']
    )
    
    print(f"\n   Convenience need distribution:")
    for level in ['Low', 'Medium', 'High']:
        count = (df_vehicles['convenience_need_level'] == level).sum()
        pct = count / len(df_vehicles) * 100
        print(f"      {level:10s}: {count:,} ({pct:.1f}%)")
    
    print(f"\n   Interpretation:")
    print(f"      High need → 需要密集桩网，不愿绕行")
    print(f"      Low need → 可接受稀疏桩网，愿意适度绕行")
    
    return df_vehicles


# ==================== 方法3：充电可达性容忍度 ====================
def compute_charging_accessibility_tolerance(df_vehicles):
    """
    可达性容忍度 = 低焦虑 + 高速场景 → 愿意为快充绕行
    """
    print("\n" + "="*70)
    print("🎯 Method 3: Charging Accessibility Tolerance")
    print("="*70)
    
    # 低焦虑（愿意用到很低电量）
    low_anxiety_score = 1 - (df_vehicles['range_anxiety_threshold'] / 100)
    
    # 高速场景（可能是长途，愿意去高速服务区）
    high_speed_score = df_vehicles['speed_mean'] / df_vehicles['speed_mean'].max()
    
    # 容忍度 = 低焦虑 + 高速场景
    df_vehicles['accessibility_tolerance'] = (
        low_anxiety_score * 0.6 +
        high_speed_score * 0.4
    )
    
    # 分类
    df_vehicles['tolerance_level'] = pd.cut(
        df_vehicles['accessibility_tolerance'],
        bins=[0, 0.4, 0.7, 1.0],
        labels=['Low', 'Medium', 'High']
    )
    
    print(f"\n   Accessibility tolerance distribution:")
    for level in ['Low', 'Medium', 'High']:
        count = (df_vehicles['tolerance_level'] == level).sum()
        pct = count / len(df_vehicles) * 100
        print(f"      {level:10s}: {count:,} ({pct:.1f}%)")
    
    print(f"\n   Interpretation:")
    print(f"      High tolerance → 愿意绕行到高速服务区/快充站")
    print(f"      Low tolerance → 只接受住宅区/顺路充电桩")
    
    return df_vehicles


# ==================== 方法4：最大可接受绕行距离估算 ====================
def estimate_max_acceptable_detour(df_vehicles):
    """
    根据用户特征，估算最大可接受绕行距离
    """
    print("\n" + "="*70)
    print("📏 Method 4: Max Acceptable Detour Estimation")
    print("="*70)
    
    # 估算公式（启发式）
    # 低焦虑用户可接受更长绕行（因为他们会规划）
    # 高速用户可接受更长绕行（因为高速开得快，时间成本低）
    # 高频用户不接受绕行（因为太频繁了）
    
    anxiety_factor = (100 - df_vehicles['range_anxiety_threshold']) / 100  # 低焦虑 → 高值
    speed_factor = df_vehicles['speed_mean'] / 50  # 50km/h为基准
    frequency_factor = 1 / (df_vehicles['total_events'] / 10 + 1)  # 高频 → 低值
    
    # 最大可接受绕行距离（km）
    df_vehicles['max_acceptable_detour_km'] = (
        10 * anxiety_factor * 0.4 +
        20 * speed_factor * 0.4 +
        5 * frequency_factor * 0.2
    )
    
    # 按簇统计
    print(f"\n   Estimated max acceptable detour by cluster:")
    for cluster_id in df_vehicles['cluster'].unique():
        cluster_data = df_vehicles[df_vehicles['cluster'] == cluster_id]
        mean_detour = cluster_data['max_acceptable_detour_km'].mean()
        std_detour = cluster_data['max_acceptable_detour_km'].std()
        print(f"      Cluster {cluster_id}: {mean_detour:.1f} ± {std_detour:.1f} km")
    
    print(f"\n   Overall:")
    print(f"      Mean: {df_vehicles['max_acceptable_detour_km'].mean():.1f} km")
    print(f"      Median: {df_vehicles['max_acceptable_detour_km'].median():.1f} km")
    
    return df_vehicles


# ==================== 可视化 ====================
def visualize_detour_indicators(df_vehicles, df_urgency):
    """
    可视化充电绕行相关指标
    """
    print("\n🎨 Creating visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    k = df_vehicles['cluster'].nunique()
    colors = plt.cm.Set3(np.linspace(0, 1, k))
    
    # 1. 充电紧急度分布
    ax1 = axes[0, 0]
    urgency_counts = df_urgency['urgency_level'].value_counts()
    ax1.bar(range(len(urgency_counts)), urgency_counts.values, 
           color=['red', 'orange', 'yellow', 'green'], alpha=0.7)
    ax1.set_xticks(range(len(urgency_counts)))
    ax1.set_xticklabels(urgency_counts.index, rotation=45)
    ax1.set_ylabel('Count', fontweight='bold')
    ax1.set_title('Charging Urgency Distribution', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. 绕行意愿分布
    ax2 = axes[0, 1]
    willingness_counts = df_urgency['detour_willingness'].value_counts()
    ax2.bar(range(len(willingness_counts)), willingness_counts.values,
           color=['darkred', 'orange', 'lightgreen'], alpha=0.7)
    ax2.set_xticks(range(len(willingness_counts)))
    ax2.set_xticklabels(willingness_counts.index)
    ax2.set_ylabel('Count', fontweight='bold')
    ax2.set_title('Detour Willingness', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. 便利性需求 by cluster
    ax3 = axes[0, 2]
    data = [df_vehicles[df_vehicles['cluster']==i]['convenience_need_index'] for i in range(k)]
    bp = ax3.boxplot(data, labels=[f'C{i}' for i in range(k)], patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax3.set_xlabel('Cluster', fontweight='bold')
    ax3.set_ylabel('Convenience Need Index', fontweight='bold')
    ax3.set_title('Convenience Need by Cluster', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. 可达性容忍度 by cluster
    ax4 = axes[1, 0]
    data = [df_vehicles[df_vehicles['cluster']==i]['accessibility_tolerance'] for i in range(k)]
    bp = ax4.boxplot(data, labels=[f'C{i}' for i in range(k)], patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax4.set_xlabel('Cluster', fontweight='bold')
    ax4.set_ylabel('Accessibility Tolerance', fontweight='bold')
    ax4.set_title('Accessibility Tolerance by Cluster', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. 最大可接受绕行距离 by cluster
    ax5 = axes[1, 1]
    data = [df_vehicles[df_vehicles['cluster']==i]['max_acceptable_detour_km'] for i in range(k)]
    bp = ax5.boxplot(data, labels=[f'C{i}' for i in range(k)], patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax5.set_xlabel('Cluster', fontweight='bold')
    ax5.set_ylabel('Max Acceptable Detour (km)', fontweight='bold')
    ax5.set_title('Max Acceptable Detour by Cluster', fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. 绕行意愿 vs 续航焦虑
    ax6 = axes[1, 2]
    for cid in range(k):
        cluster_data = df_vehicles[df_vehicles['cluster'] == cid]
        ax6.scatter(cluster_data['range_anxiety_threshold'],
                   cluster_data['max_acceptable_detour_km'],
                   c=[colors[cid]], label=f'C{cid}', alpha=0.6, s=30)
    ax6.set_xlabel('Range Anxiety Threshold (%)', fontweight='bold')
    ax6.set_ylabel('Max Acceptable Detour (km)', fontweight='bold')
    ax6.set_title('Anxiety vs Detour Tolerance', fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Charging Detour Proxy Indicators', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('./results/charging_detour_indicators.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: charging_detour_indicators.png")


# ==================== Main ====================
def main():
    # 方法1：充电紧急度
    df_urgency = compute_charging_urgency_index(df_events)
    df_urgency.to_csv('./results/charging_urgency.csv', index=False, encoding='utf-8-sig')
    
    # 方法2：便利性需求
    df_vehicles = compute_charging_convenience_need(df_vehicles)
    
    # 方法3：可达性容忍度
    df_vehicles = compute_charging_accessibility_tolerance(df_vehicles)
    
    # 方法4：最大可接受绕行距离
    df_vehicles = estimate_max_acceptable_detour(df_vehicles)
    
    # 保存
    df_vehicles.to_csv('./results/vehicle_features_with_detour.csv', index=False, encoding='utf-8-sig')
    print(f"\n💾 Saved: vehicle_features_with_detour.csv")
    
    # 可视化
    visualize_detour_indicators(df_vehicles, df_urgency)
    
    # 业务建议
    print("\n" + "="*70)
    print("💡 Business Recommendations")
    print("="*70)
    
    for cluster_id in df_vehicles['cluster'].unique():
        cluster_data = df_vehicles[df_vehicles['cluster'] == cluster_id]
        
        print(f"\n🔷 Cluster {cluster_id}:")
        print(f"   Avg max detour: {cluster_data['max_acceptable_detour_km'].mean():.1f} km")
        print(f"   Convenience need: {cluster_data['convenience_need_index'].mean():.2f}")
        print(f"   Accessibility tolerance: {cluster_data['accessibility_tolerance'].mean():.2f}")
        
        # 桩网密度建议
        if cluster_data['convenience_need_index'].mean() > 0.6:
            print(f"   → 需要高密度桩网（<2km间距）")
        elif cluster_data['convenience_need_index'].mean() > 0.3:
            print(f"   → 中等密度桩网（2-5km间距）")
        else:
            print(f"   → 低密度桩网可接受（>5km间距）")
    
    print("\n" + "="*70)
    print("✅ Complete!")
    print("="*70)


if __name__ == "__main__":
    main()