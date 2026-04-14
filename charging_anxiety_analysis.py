"""
Phase 3: 充电焦虑分析 + 放电模式影响
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("📊 Phase 3: Charging Anxiety & Discharge Pattern Analysis")
print("="*70)

# 加载完整数据
df = pd.read_csv('./results/vehicle_complete_profile.csv')

print(f"✅ Loaded {len(df):,} vehicles")

# 定义充电焦虑指标
print("\n🔍 Computing charging anxiety metrics...")

df['charging_anxiety_score'] = (
    (100 - df['avg_charging_trigger_soc']) * 0.3 +  # 触发SOC越低，焦虑越高
    df['charging_freq_per_100km'] * 20 +             # 充电频率越高，焦虑越高
    (df['std_charging_trigger_soc']) * 0.2 +         # 触发SOC波动越大，焦虑越高
    (50 - df['min_soc_ever']) * 0.3                  # 历史最低SOC越低，焦虑越高
)

# 归一化到0-100
df['charging_anxiety_score'] = (
    (df['charging_anxiety_score'] - df['charging_anxiety_score'].min()) / 
    (df['charging_anxiety_score'].max() - df['charging_anxiety_score'].min()) * 100
)

# 分级
df['anxiety_level'] = pd.cut(
    df['charging_anxiety_score'],
    bins=[0, 25, 50, 75, 100],
    labels=['低焦虑', '中等焦虑', '较高焦虑', '高焦虑']
)

print(f"✅ Anxiety levels:")
print(df['anxiety_level'].value_counts())

# 按主导放电模式分组
print(f"\n📊 Analysis by dominant discharge cluster:")

for cluster in sorted(df['dominant_cluster'].dropna().unique()):
    cluster_data = df[df['dominant_cluster'] == cluster]
    
    print(f"\n   Cluster {int(cluster)} (n={len(cluster_data):,}):")
    print(f"      平均充电触发SOC: {cluster_data['avg_charging_trigger_soc'].mean():.1f}%")
    print(f"      平均充电频率(次/100km): {cluster_data['charging_freq_per_100km'].mean():.2f}")
    print(f"      平均焦虑分数: {cluster_data['charging_anxiety_score'].mean():.1f}")
    print(f"      高焦虑车辆占比: {(cluster_data['anxiety_level'] == '高焦虑').sum() / len(cluster_data) * 100:.1f}%")

# 可视化
print(f"\n📈 Generating visualizations...")

fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

# 图1: 充电触发SOC分布（按主导cluster）
ax = fig.add_subplot(gs[0, :2])
for cluster in sorted(df['dominant_cluster'].dropna().unique()):
    data = df[df['dominant_cluster'] == cluster]['avg_charging_trigger_soc']
    ax.hist(data, bins=30, alpha=0.6, label=f'Cluster {int(cluster)}')
ax.set_xlabel('平均充电触发SOC (%)', fontsize=11)
ax.set_ylabel('车辆数', fontsize=11)
ax.set_title('1. 充电触发SOC分布（按主导放电模式）', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# 图2: 充电频率对比（箱线图）
ax = fig.add_subplot(gs[0, 2:])
data_list = [df[df['dominant_cluster'] == c]['charging_freq_per_100km'] 
             for c in sorted(df['dominant_cluster'].dropna().unique())]
ax.boxplot(data_list, labels=[f'C{int(c)}' for c in sorted(df['dominant_cluster'].dropna().unique())])
ax.set_ylabel('充电频率 (次/100km)', fontsize=11)
ax.set_title('2. 充电频率对比', fontsize=13, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

# 图3: 放电模式占比 vs 充电触发SOC
ax = fig.add_subplot(gs[1, 0])
for cluster in [0, 1, 2]:
    ax.scatter(df[f'cluster_{cluster}_ratio'], df['avg_charging_trigger_soc'],
               alpha=0.3, s=20, label=f'Cluster {cluster} 占比')
ax.set_xlabel('放电模式占比', fontsize=11)
ax.set_ylabel('平均充电触发SOC (%)', fontsize=11)
ax.set_title('3. 放电模式占比 vs 充电触发SOC', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# 图4: 充电频率 vs 焦虑分数
ax = fig.add_subplot(gs[1, 1])
scatter = ax.scatter(df['charging_freq_per_100km'], df['charging_anxiety_score'],
                     c=df['dominant_cluster'], cmap='viridis', alpha=0.5, s=30)
ax.set_xlabel('充电频率 (次/100km)', fontsize=11)
ax.set_ylabel('焦虑分数', fontsize=11)
ax.set_title('4. 充电频率 vs 焦虑分数', fontsize=12, fontweight='bold')
plt.colorbar(scatter, ax=ax, label='主导Cluster')
ax.grid(alpha=0.3)

# 图5: 焦虑等级分布（堆叠条形图）
ax = fig.add_subplot(gs[1, 2:])
anxiety_by_cluster = df.groupby(['dominant_cluster', 'anxiety_level']).size().unstack(fill_value=0)
anxiety_by_cluster.plot(kind='bar', stacked=True, ax=ax, 
                        color=['green', 'yellow', 'orange', 'red'], alpha=0.8)
ax.set_xlabel('主导放电Cluster', fontsize=11)
ax.set_ylabel('车辆数', fontsize=11)
ax.set_title('5. 焦虑等级分布（按放电模式）', fontsize=12, fontweight='bold')
ax.set_xticklabels([f'C{int(c)}' for c in anxiety_by_cluster.index], rotation=0)
ax.legend(title='焦虑等级', fontsize=9)
ax.grid(alpha=0.3, axis='y')

# 图6-8: 三个cluster的详细对比
for i, cluster in enumerate(sorted(df['dominant_cluster'].dropna().unique())):
    ax = fig.add_subplot(gs[2, i])
    cluster_data = df[df['dominant_cluster'] == cluster]
    
    metrics = ['充电触发SOC', '充电频率\n(次/100km)', '焦虑分数']
    values = [
        cluster_data['avg_charging_trigger_soc'].mean(),
        cluster_data['charging_freq_per_100km'].mean() * 10,  # 缩放
        cluster_data['charging_anxiety_score'].mean()
    ]
    
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    ax.bar(metrics, values, color=colors, alpha=0.7)
    ax.set_title(f'Cluster {int(cluster)} 特征\n(n={len(cluster_data):,})', 
                 fontsize=11, fontweight='bold')
    ax.set_ylabel('值', fontsize=10)
    ax.grid(alpha=0.3, axis='y')

# 图9: 相关性热图
ax = fig.add_subplot(gs[2, 3])
corr_cols = ['avg_charging_trigger_soc', 'charging_freq_per_100km', 
             'charging_anxiety_score', 'cluster_0_ratio', 'cluster_1_ratio', 'cluster_2_ratio']
corr = df[corr_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            xticklabels=['触发SOC', '频率', '焦虑', 'C0占比', 'C1占比', 'C2占比'],
            yticklabels=['触发SOC', '频率', '焦虑', 'C0占比', 'C1占比', 'C2占比'],
            ax=ax, cbar_kws={'label': '相关系数'})
ax.set_title('9. 特征相关性热图', fontsize=12, fontweight='bold')

plt.suptitle('放电模式对充电行为和焦虑的影响分析', fontsize=16, fontweight='bold', y=0.995)

plt.savefig('./results/charging_anxiety_analysis.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: charging_anxiety_analysis.png")

# 统计检验
print(f"\n📊 Statistical Tests:")

for metric in ['avg_charging_trigger_soc', 'charging_freq_per_100km', 'charging_anxiety_score']:
    groups = [df[df['dominant_cluster'] == c][metric].dropna() 
              for c in sorted(df['dominant_cluster'].dropna().unique())]
    
    f_stat, p_value = stats.f_oneway(*groups)
    
    print(f"\n   {metric}:")
    print(f"      F-statistic: {f_stat:.3f}")
    print(f"      P-value: {p_value:.6f}")
    print(f"      结论: {'显著差异 ✓' if p_value < 0.05 else '无显著差异'}")

# 保存最终数据
df.to_csv('./results/vehicle_with_anxiety.csv', index=False, encoding='utf-8-sig')
print(f"\n💾 Saved: vehicle_with_anxiety.csv")

print(f"\n{'='*70}")
print(f"✅ Phase 3 Complete!")
print(f"{'='*70}")
print(f"\n📁 Generated files:")
print(f"   1. vehicle_with_anxiety.csv - 包含焦虑指标的车辆画像")
print(f"   2. charging_anxiety_analysis.png - 完整分析可视化")
print(f"{'='*70}")