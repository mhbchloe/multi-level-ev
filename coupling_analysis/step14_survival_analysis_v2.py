"""
Step 14 (v2): Survival Analysis
Kaplan-Meier曲线：分析不同车辆画像的"充电忍耐度"
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import os
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['font.size'] = 12

print("="*70)
print("📈 Step 14 (v2): Survival Analysis")
print("="*70)

input_dir = "./coupling_analysis/results/"
df = pd.read_csv(os.path.join(input_dir, 'inter_charge_trips_v2.csv'))

print(f"\n📊 Dataset: {len(df):,} trips from {df['vehicle_id'].nunique():,} vehicles")

# ========== 1. 数据统计 ==========
print("\n📊 Charging Behavior Statistics by Vehicle Archetype:")
print("="*70)

for vtype in sorted(df['vehicle_type'].unique()):
    subset = df[df['vehicle_type'] == vtype]
    print(f"\n{vtype}:")
    print(f"   Trips: {len(subset):,}")
    print(f"   Avg Trigger SOC: {subset['charge_trigger_soc'].mean():.1f}% (±{subset['charge_trigger_soc'].std():.1f}%)")
    print(f"   Avg SOC Drop: {subset['trip_total_soc_drop'].mean():.1f}% (±{subset['trip_total_soc_drop'].std():.1f}%)")
    print(f"   Avg Charging Gain: {subset['charge_gain_soc'].mean():.1f}%")

# ========== 2. 箱线图对比 ==========
print("\n📈 Generating comparison boxplots...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

# 图1：不同车辆类型的充电触发SOC
ax = axes[0, 0]
sns.boxplot(data=df, x='vehicle_type', y='charge_trigger_soc', ax=ax, palette=colors, width=0.6)
ax.set_title('Charge Trigger SOC by Vehicle Archetype\n(Higher = Earlier charging = Higher anxiety)', 
             fontweight='bold', fontsize=12)
ax.set_ylabel('SOC Level when charging triggered (%)', fontweight='bold')
ax.set_xlabel('Vehicle Archetype', fontweight='bold')
ax.grid(alpha=0.3, axis='y')

# 图2：不同车辆类型的SOC消耗量
ax = axes[0, 1]
sns.boxplot(data=df, x='vehicle_type', y='trip_total_soc_drop', ax=ax, palette=colors, width=0.6)
ax.set_title('SOC Consumption per Trip by Vehicle Archetype\n(Higher = More intense driving)', 
             fontweight='bold', fontsize=12)
ax.set_ylabel('Total SOC drop during trip (%)', fontweight='bold')
ax.set_xlabel('Vehicle Archetype', fontweight='bold')
ax.grid(alpha=0.3, axis='y')

# 图3：激进片段比例
ax = axes[1, 0]
sns.boxplot(data=df, x='vehicle_type', y='ratio_aggressive', ax=ax, palette=colors, width=0.6)
ax.set_title('Aggressive Driving Ratio by Vehicle Archetype\n(Higher = More aggressive driving patterns)', 
             fontweight='bold', fontsize=12)
ax.set_ylabel('Ratio of Aggressive Segments', fontweight='bold')
ax.set_xlabel('Vehicle Archetype', fontweight='bold')
ax.grid(alpha=0.3, axis='y')

# 图4：充电增益
ax = axes[1, 1]
sns.boxplot(data=df, x='vehicle_type', y='charge_gain_soc', ax=ax, palette=colors, width=0.6)
ax.set_title('Charging Gain (SOC %) by Vehicle Archetype\n(How much do they charge each time?)', 
             fontweight='bold', fontsize=12)
ax.set_ylabel('SOC Gain per Charging (%)', fontweight='bold')
ax.set_xlabel('Vehicle Archetype', fontweight='bold')
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(input_dir, 'charging_behavior_boxplots_v2.png'), dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Saved: charging_behavior_boxplots_v2.png")

# ========== 3. Kaplan-Meier 生存曲线 ==========
print("\n⏳ Generating Kaplan-Meier Survival Curves...")

kmf = KaplanMeierFitter()
fig, ax = plt.subplots(figsize=(13, 8))

archetypes = sorted(df['vehicle_type'].unique())

for i, archetype in enumerate(archetypes):
    subset = df[df['vehicle_type'] == archetype]
    
    # T: 行程中消耗的总SOC（作为"时间"）
    # E: 所有样本都发生了充电事件（event=1）
    T = subset['trip_total_soc_drop'].values
    E = np.ones(len(T))
    
    kmf.fit(T, event_observed=E, label=f"{archetype} (n={len(subset):,})")
    kmf.plot_survival_function(ax=ax, color=colors[i], linewidth=2.5, ci_show=True)

ax.set_xlabel('Total SOC Consumed since last charging (%)', fontweight='bold', fontsize=13)
ax.set_ylabel('Probability of continuing (not charging yet)', fontweight='bold', fontsize=13)
ax.set_title('Kaplan-Meier Curve: Charging Endurance by Vehicle Archetype\n' + 
             'Question: How much SOC can each type consume before must charging?',
             fontweight='bold', fontsize=14, pad=20)
ax.grid(alpha=0.3)
ax.set_xlim(0, 100)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=11, loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(input_dir, 'survival_curve_kaplan_meier_v2.png'), dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Saved: survival_curve_kaplan_meier_v2.png")

# ========== 4. Log-Rank 检验 ==========
print("\n🔬 Log-rank Test Results (Statistical Significance):")
print("="*70)

results_file = os.path.join(input_dir, 'logrank_test_results_v2.txt')
with open(results_file, 'w', encoding='utf-8') as f:
    f.write("Log-rank Test: Testing if charging endurance curves are significantly different\n")
    f.write("="*70 + "\n\n")
    
    for i in range(len(archetypes)):
        for j in range(i+1, len(archetypes)):
            subset_i = df[df['vehicle_type'] == archetypes[i]]
            subset_j = df[df['vehicle_type'] == archetypes[j]]
            
            T_i = subset_i['trip_total_soc_drop'].values
            T_j = subset_j['trip_total_soc_drop'].values
            E_i = np.ones(len(T_i))
            E_j = np.ones(len(T_j))
            
            results = logrank_test(T_i, T_j, event_observed_A=E_i, event_observed_B=E_j)
            
            sig_indicator = "***" if results.p_value < 0.001 else ("**" if results.p_value < 0.01 else ("*" if results.p_value < 0.05 else "ns"))
            
            comparison_str = f"{archetypes[i]} vs {archetypes[j]}:\n" + \
                           f"  p-value: {results.p_value:.4e} {sig_indicator}\n" + \
                           f"  Test Statistic: {results.test_statistic:.4f}\n\n"
            
            print(comparison_str)
            f.write(comparison_str)

print(f"✅ Saved: {results_file}")

# ========== 5. 中位数生存时间 ==========
print(f"\n{'='*70}")
print("📊 Median Charging Endurance (Median SOC Consumption):")
print(f"{'='*70}")

for archetype in archetypes:
    subset = df[df['vehicle_type'] == archetype]
    median_soc = subset['trip_total_soc_drop'].median()
    q1 = subset['trip_total_soc_drop'].quantile(0.25)
    q3 = subset['trip_total_soc_drop'].quantile(0.75)
    
    print(f"\n{archetype}:")
    print(f"   Median: {median_soc:.1f}% (IQR: {q1:.1f}%-{q3:.1f}%)")

print("\n✅ Step 14 Complete!")