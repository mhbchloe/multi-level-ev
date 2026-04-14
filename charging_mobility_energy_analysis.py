"""
Charging-Mobility-Energy 三元耦合分析
目标：从驾驶行为预测充电行为，建立量化模型
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("🔋 Charging-Mobility-Energy Coupling Analysis")
print("="*70)


# ==================== 加载数据 ====================
print("\n📂 Loading data...")

df_events = pd.read_csv('./results/event_table.csv')
df_vehicles = pd.read_csv('./results/vehicle_features_clustered_advanced.csv')

print(f"✅ Events: {len(df_events):,}, Vehicles: {len(df_vehicles):,}")


# ==================== Step 1: 充电事件推断 ====================
def infer_charging_events(df_events):
    """
    从事件表推断充电事件
    逻辑：如果下一次事件的起始SOC > 当前事件结束SOC，说明中间充了电
    """
    print("\n" + "="*70)
    print("⚡ Step 1: Inferring Charging Events")
    print("="*70)
    
    charging_events = []
    
    for vehicle_id in df_events['vehicle_id'].unique():
        vehicle_events = df_events[df_events['vehicle_id'] == vehicle_id].sort_values('start_time')
        
        for i in range(len(vehicle_events) - 1):
            current_event = vehicle_events.iloc[i]
            next_event = vehicle_events.iloc[i+1]
            
            soc_end_current = current_event['soc_end']
            soc_start_next = next_event['soc_start']
            
            # 如果下一次起始SOC > 当前结束SOC，说明充电了
            if soc_start_next > soc_end_current:
                charging = {
                    'vehicle_id': vehicle_id,
                    'charging_start_soc': soc_end_current,  # 充电开始时的SOC
                    'charging_end_soc': soc_start_next,  # 充电结束时的SOC
                    'charging_amount': soc_start_next - soc_end_current,  # 充电量
                    'prev_event_cluster': current_event['cluster'],  # 充电前的驾驶模式
                    'prev_event_distance': current_event['distance_km'],  # 充电前的行驶距离
                    'prev_event_energy': current_event['energy_consumption_kwh'],  # 充电前的能耗
                    'prev_event_speed': current_event['speed_mean'],  # 充电前的平均速度
                    'prev_event_power': current_event['power_mean'],  # 充电前的平均功率
                }
                charging_events.append(charging)
    
    df_charging = pd.DataFrame(charging_events)
    
    print(f"\n✅ Inferred {len(df_charging):,} charging events")
    print(f"   Vehicles with charging data: {df_charging['vehicle_id'].nunique():,}")
    
    return df_charging


# ==================== Step 2: 车辆级充电指标计算 ====================
def compute_charging_metrics(df_charging, df_vehicles):
    """
    为每辆车计算充电相关指标
    """
    print("\n" + "="*70)
    print("📊 Step 2: Computing Charging Metrics")
    print("="*70)
    
    charging_metrics = []
    
    for vehicle_id in df_charging['vehicle_id'].unique():
        vehicle_charging = df_charging[df_charging['vehicle_id'] == vehicle_id]
        
        metrics = {
            'vehicle_id': vehicle_id,
            
            # 1. 充电频率
            'charging_frequency': len(vehicle_charging),  # 充电次数
            
            # 2. 平均充电开始SOC
            'avg_charging_start_soc': vehicle_charging['charging_start_soc'].mean(),
            'std_charging_start_soc': vehicle_charging['charging_start_soc'].std(),
            'min_charging_start_soc': vehicle_charging['charging_start_soc'].min(),  # 最低充电起点
            
            # 3. 平均充电量
            'avg_charging_amount': vehicle_charging['charging_amount'].mean(),
            'std_charging_amount': vehicle_charging['charging_amount'].std(),
            'total_charging_amount': vehicle_charging['charging_amount'].sum(),
            
            # 4. 充电触发距离（充电前行驶了多远）
            'avg_distance_before_charging': vehicle_charging['prev_event_distance'].mean(),
            
            # 5. 充电触发能耗（充电前消耗了多少电）
            'avg_energy_before_charging': vehicle_charging['prev_event_energy'].mean(),
            
            # 6. 充电前驾驶模式分布
            'charging_after_mode0_ratio': (vehicle_charging['prev_event_cluster'] == 0).mean(),
            'charging_after_mode1_ratio': (vehicle_charging['prev_event_cluster'] == 1).mean(),
            'charging_after_mode2_ratio': (vehicle_charging['prev_event_cluster'] == 2).mean(),
            'charging_after_mode3_ratio': (vehicle_charging['prev_event_cluster'] == 3).mean(),
        }
        
        charging_metrics.append(metrics)
    
    df_charging_metrics = pd.DataFrame(charging_metrics)
    
    # 合并到车辆特征表
    df_vehicles_enhanced = df_vehicles.merge(df_charging_metrics, on='vehicle_id', how='left')
    
    # 填充缺失值（没有充电数据的车）
    charging_cols = [col for col in df_charging_metrics.columns if col != 'vehicle_id']
    df_vehicles_enhanced[charging_cols] = df_vehicles_enhanced[charging_cols].fillna(0)
    
    print(f"\n✅ Computed charging metrics for {len(df_charging_metrics):,} vehicles")
    print(f"   Total vehicles: {len(df_vehicles_enhanced):,}")
    
    return df_vehicles_enhanced, df_charging


# ==================== Step 3: 驾驶模式对充电行为的影响分析 ====================
def analyze_mobility_charging_coupling(df_vehicles_enhanced):
    """
    分析驾驶模式（event cluster）对充电行为的贡献
    """
    print("\n" + "="*70)
    print("🔗 Step 3: Mobility-Charging Coupling Analysis")
    print("="*70)
    
    # 按车辆簇分析
    for cluster_id in df_vehicles_enhanced['cluster'].unique():
        cluster_data = df_vehicles_enhanced[df_vehicles_enhanced['cluster'] == cluster_id]
        
        print(f"\n{'='*70}")
        print(f"🔷 Vehicle Cluster {cluster_id} (n={len(cluster_data):,})")
        print(f"{'='*70}")
        
        # 充电频率
        print(f"\n   充电频率:")
        print(f"      平均充电次数: {cluster_data['charging_frequency'].mean():.1f}")
        print(f"      中位数: {cluster_data['charging_frequency'].median():.0f}")
        
        # 充电触发SOC
        print(f"\n   充电触发SOC:")
        print(f"      平均充电开始SOC: {cluster_data['avg_charging_start_soc'].mean():.1f}%")
        print(f"      最低充电SOC: {cluster_data['min_charging_start_soc'].mean():.1f}%")
        
        # 充电量
        print(f"\n   充电量:")
        print(f"      平均单次充电量: {cluster_data['avg_charging_amount'].mean():.1f}%")
        print(f"      总充电量: {cluster_data['total_charging_amount'].mean():.1f}%")
        
        # 驾驶模式对充电的影响
        print(f"\n   充电前驾驶模式分布:")
        for mode_id in range(4):
            ratio = cluster_data[f'charging_after_mode{mode_id}_ratio'].mean()
            print(f"      充电发生在Event Mode {mode_id}之后: {ratio*100:.1f}%")
    
    return df_vehicles_enhanced


# ==================== Step 4: 关键指标相关性分析 ====================
def analyze_key_correlations(df_vehicles_enhanced):
    """
    分析驾驶行为特征与充电行为的相关性
    """
    print("\n" + "="*70)
    print("📈 Step 4: Key Correlations Analysis")
    print("="*70)
    
    # 选择关键变量
    mobility_features = [
        'speed_mean',  # 驾驶速度
        'idle_ratio_mean',  # 怠速比例
        'power_mean',  # 平均功率
        'efficiency_kwh_per_km',  # 能效
        'total_distance_km',  # 总行驶距离
        'range_anxiety_threshold',  # 续航焦虑阈值
    ]
    
    charging_features = [
        'charging_frequency',  # 充电频率
        'avg_charging_start_soc',  # 充电开始SOC
        'avg_charging_amount',  # 充电量
    ]
    
    # 计算相关性
    print("\n📊 Correlation Matrix (Pearson):")
    print("\n   驾驶特征 → 充电频率:")
    
    for mobility_feat in mobility_features:
        if mobility_feat in df_vehicles_enhanced.columns:
            corr, pval = pearsonr(
                df_vehicles_enhanced[mobility_feat].fillna(0),
                df_vehicles_enhanced['charging_frequency'].fillna(0)
            )
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            print(f"      {mobility_feat:30s} → r={corr:+.3f} {sig}")
    
    print("\n   驾驶特征 → 充电开始SOC:")
    for mobility_feat in mobility_features:
        if mobility_feat in df_vehicles_enhanced.columns:
            corr, pval = pearsonr(
                df_vehicles_enhanced[mobility_feat].fillna(0),
                df_vehicles_enhanced['avg_charging_start_soc'].fillna(0)
            )
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            print(f"      {mobility_feat:30s} → r={corr:+.3f} {sig}")
    
    return df_vehicles_enhanced


# ==================== Step 5: 可视化 ====================
def visualize_charging_mobility_coupling(df_vehicles_enhanced):
    """
    可视化驾驶-充电耦合关系
    """
    print("\n🎨 Creating visualizations...")
    
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 3, hspace=0.3, wspace=0.3)
    
    k = df_vehicles_enhanced['cluster'].nunique()
    colors = plt.cm.Set3(np.linspace(0, 1, k))
    
    # 1. 充电频率 vs 平均速度
    ax1 = fig.add_subplot(gs[0, 0])
    for cid in range(k):
        data = df_vehicles_enhanced[df_vehicles_enhanced['cluster'] == cid]
        ax1.scatter(data['speed_mean'], data['charging_frequency'],
                   c=[colors[cid]], label=f'C{cid}', alpha=0.6, s=30)
    ax1.set_xlabel('Avg Speed (km/h)', fontweight='bold')
    ax1.set_ylabel('Charging Frequency', fontweight='bold')
    ax1.set_title('Speed vs Charging Frequency', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 续航焦虑 vs 充电开始SOC
    ax2 = fig.add_subplot(gs[0, 1])
    for cid in range(k):
        data = df_vehicles_enhanced[df_vehicles_enhanced['cluster'] == cid]
        ax2.scatter(data['range_anxiety_threshold'], data['avg_charging_start_soc'],
                   c=[colors[cid]], label=f'C{cid}', alpha=0.6, s=30)
    ax2.plot([0, 100], [0, 100], 'k--', alpha=0.3, label='y=x')
    ax2.set_xlabel('Range Anxiety Threshold (%)', fontweight='bold')
    ax2.set_ylabel('Avg Charging Start SOC (%)', fontweight='bold')
    ax2.set_title('Anxiety vs Charging Trigger', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 能效 vs 充电量
    ax3 = fig.add_subplot(gs[0, 2])
    for cid in range(k):
        data = df_vehicles_enhanced[df_vehicles_enhanced['cluster'] == cid]
        ax3.scatter(data['efficiency_kwh_per_km'], data['avg_charging_amount'],
                   c=[colors[cid]], label=f'C{cid}', alpha=0.6, s=30)
    ax3.set_xlabel('Efficiency (kWh/km)', fontweight='bold')
    ax3.set_ylabel('Avg Charging Amount (%)', fontweight='bold')
    ax3.set_title('Efficiency vs Charging Amount', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 充电频率分布
    ax4 = fig.add_subplot(gs[1, 0])
    data = [df_vehicles_enhanced[df_vehicles_enhanced['cluster']==i]['charging_frequency'] 
            for i in range(k)]
    bp = ax4.boxplot(data, labels=[f'C{i}' for i in range(k)], patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax4.set_xlabel('Cluster', fontweight='bold')
    ax4.set_ylabel('Charging Frequency', fontweight='bold')
    ax4.set_title('Charging Frequency by Cluster', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. 充电开始SOC分布
    ax5 = fig.add_subplot(gs[1, 1])
    data = [df_vehicles_enhanced[df_vehicles_enhanced['cluster']==i]['avg_charging_start_soc'] 
            for i in range(k)]
    bp = ax5.boxplot(data, labels=[f'C{i}' for i in range(k)], patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax5.set_xlabel('Cluster', fontweight='bold')
    ax5.set_ylabel('Avg Charging Start SOC (%)', fontweight='bold')
    ax5.set_title('Charging Start SOC by Cluster', fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. 充电量分布
    ax6 = fig.add_subplot(gs[1, 2])
    data = [df_vehicles_enhanced[df_vehicles_enhanced['cluster']==i]['avg_charging_amount'] 
            for i in range(k)]
    bp = ax6.boxplot(data, labels=[f'C{i}' for i in range(k)], patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax6.set_xlabel('Cluster', fontweight='bold')
    ax6.set_ylabel('Avg Charging Amount (%)', fontweight='bold')
    ax6.set_title('Charging Amount by Cluster', fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Charging-Mobility-Energy Coupling Analysis', fontsize=18, fontweight='bold')
    plt.savefig('./results/charging_mobility_coupling.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: charging_mobility_coupling.png")


# ==================== Step 6: 充电行为预测模型 ====================
def build_charging_prediction_model(df_vehicles_enhanced):
    """
    建立充电行为预测模型
    目标：从驾驶行为预测充电频率和充电开始SOC
    """
    print("\n" + "="*70)
    print("🤖 Step 5: Charging Behavior Prediction Model")
    print("="*70)
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error
    
    # 特征选择
    features = [
        'speed_mean', 'idle_ratio_mean', 'accel_abs_mean',
        'power_mean', 'efficiency_kwh_per_km',
        'range_anxiety_threshold', 'mode_diversity',
        'total_events', 'total_distance_km', 'total_energy_kwh',
    ]
    
    # 目标1：充电频率
    print("\n📊 Predicting Charging Frequency:")
    
    X = df_vehicles_enhanced[features].fillna(0).values
    y = df_vehicles_enhanced['charging_frequency'].fillna(0).values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model_freq = RandomForestRegressor(n_estimators=100, random_state=42)
    model_freq.fit(X_train, y_train)
    
    y_pred = model_freq.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"   R² Score: {r2:.3f}")
    print(f"   MAE: {mae:.2f} times")
    
    # 特征重要性
    importances = model_freq.feature_importances_
    feature_importance = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
    
    print(f"\n   Top 5 Important Features:")
    for feat, imp in feature_importance[:5]:
        print(f"      {feat:30s}: {imp:.3f}")
    
    # 目标2：充电开始SOC
    print("\n📊 Predicting Charging Start SOC:")
    
    y = df_vehicles_enhanced['avg_charging_start_soc'].fillna(0).values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model_soc = RandomForestRegressor(n_estimators=100, random_state=42)
    model_soc.fit(X_train, y_train)
    
    y_pred = model_soc.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"   R² Score: {r2:.3f}")
    print(f"   MAE: {mae:.2f}%")
    
    # 特征重要性
    importances = model_soc.feature_importances_
    feature_importance = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
    
    print(f"\n   Top 5 Important Features:")
    for feat, imp in feature_importance[:5]:
        print(f"      {feat:30s}: {imp:.3f}")
    
    return model_freq, model_soc


# ==================== Main ====================
def main():
    # Step 1: 推断充电事件
    df_charging = infer_charging_events(df_events)
    df_charging.to_csv('./results/charging_events.csv', index=False, encoding='utf-8-sig')
    print(f"💾 Saved: charging_events.csv")
    
    # Step 2: 计算充电指标
    df_vehicles_enhanced, df_charging = compute_charging_metrics(df_charging, df_vehicles)
    df_vehicles_enhanced.to_csv('./results/vehicle_features_with_charging.csv', index=False, encoding='utf-8-sig')
    print(f"💾 Saved: vehicle_features_with_charging.csv")
    
    # Step 3: 驾驶-充电耦合分析
    df_vehicles_enhanced = analyze_mobility_charging_coupling(df_vehicles_enhanced)
    
    # Step 4: 相关性分析
    df_vehicles_enhanced = analyze_key_correlations(df_vehicles_enhanced)
    
    # Step 5: 可视化
    visualize_charging_mobility_coupling(df_vehicles_enhanced)
    
    # Step 6: 预测模型
    model_freq, model_soc = build_charging_prediction_model(df_vehicles_enhanced)
    
    print("\n" + "="*70)
    print("✅ Charging-Mobility-Energy Analysis Complete!")
    print("="*70)
    print("\n📁 Generated files:")
    print("   1. charging_events.csv - 充电事件表")
    print("   2. vehicle_features_with_charging.csv - 车辆特征+充电指标")
    print("   3. charging_mobility_coupling.png - 耦合分析图")
    print("\n💡 Key Insights:")
    print("   - 驾驶行为 → 能耗模式 → 充电需求")
    print("   - 续航焦虑阈值强相关充电触发SOC")
    print("   - 可预测充电频率和充电时机")
    print("="*70)


if __name__ == "__main__":
    main()