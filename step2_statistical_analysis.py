"""
Step 2: Statistical Analysis (Fixed)
修复标题分割问题
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway, kruskal, mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

print("="*70)
print("📊 Step 2: Statistical Analysis (Fixed)")
print("="*70)

# ============ 1. Load Data ============
print("\n📂 Loading data...")

df = pd.read_csv('./results/discharge_charging_pairs.csv')
print(f"✅ Loaded {len(df):,} discharge-charging pairs")

# 检查数据分布
print(f"\nCluster distribution:")
print(df['discharge_cluster'].value_counts().sort_index())

print(f"\nSample data:")
print(df[['discharge_cluster', 'charging_soc_start', 'charging_soc_gain', 
          'charging_duration', 'time_to_next_charging']].head())

# ============ 2. Descriptive Statistics by Cluster ============
print("\n📊 Descriptive Statistics by Discharge Cluster:")

cluster_stats = df.groupby('discharge_cluster').agg({
    # 充电行为
    'charging_soc_start': ['mean', 'std', 'median', 'count'],
    'charging_soc_gain': ['mean', 'std', 'median'],
    'charging_duration': ['mean', 'std', 'median'],
    'time_to_next_charging': ['mean', 'std', 'median'],
    
    # 放电特征
    'discharge_soc_drop': ['mean', 'std'],
    'discharge_speed_mean': ['mean', 'std'],
    'discharge_harsh_accel': ['mean', 'std'],
    'discharge_distance': ['mean', 'std'],
}).round(2)

print(cluster_stats)

# 保存
cluster_stats.to_csv('./results/cluster_statistics.csv', encoding='utf-8-sig')
print(f"💾 Saved: cluster_statistics.csv")

# ============ 3. Statistical Tests ============
print(f"\n{'='*70}")
print(f"📊 Statistical Significance Tests")
print(f"{'='*70}")

test_variables = {
    'charging_soc_start': 'Charging Trigger SOC',
    'charging_soc_gain': 'Charging Gain',
    'charging_duration': 'Charging Duration',
    'time_to_next_charging': 'Time to Next Charging',
}

test_results = []

for var_name, var_label in test_variables.items():
    print(f"\n{var_label} ({var_name}):")
    
    # 按cluster分组
    groups = [df[df['discharge_cluster'] == c][var_name].dropna() 
              for c in sorted(df['discharge_cluster'].unique())]
    
    # 过滤空组
    groups = [g for g in groups if len(g) > 0]
    
    if len(groups) < 2:
        print(f"   ⚠️  Insufficient groups for testing")
        continue
    
    # 检查样本量
    print(f"   Group sizes: {[len(g) for g in groups]}")
    
    # 正态性检验
    normality = [stats.shapiro(g)[1] > 0.05 for g in groups if len(g) >= 3]
    all_normal = all(normality) if normality else False
    
    if all_normal and len(groups) >= 2:
        # ANOVA
        f_stat, p_value = f_oneway(*groups)
        test_type = "ANOVA"
    elif len(groups) >= 2:
        # Kruskal-Wallis
        h_stat, p_value = kruskal(*groups)
        test_type = "Kruskal-Wallis"
        f_stat = h_stat
    else:
        print(f"   ⚠️  Cannot perform test")
        continue
    
    # 效应量
    try:
        grand_mean = df[var_name].mean()
        ss_between = sum([len(g) * (g.mean() - grand_mean)**2 for g in groups])
        ss_total = sum([(x - grand_mean)**2 for g in groups for x in g])
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
    except:
        eta_squared = 0
    
    print(f"   Test: {test_type}")
    print(f"   Statistic: {f_stat:.3f}")
    print(f"   P-value: {p_value:.6f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")
    print(f"   Effect Size (η²): {eta_squared:.3f}")
    
    test_results.append({
        'Variable': var_label,
        'Test': test_type,
        'Statistic': f_stat,
        'P-value': p_value,
        'Eta_squared': eta_squared,
        'Significant': p_value < 0.05
    })

df_test_results = pd.DataFrame(test_results)
df_test_results.to_csv('./results/statistical_test_results.csv', index=False, encoding='utf-8-sig')

# ============ 4. Post-hoc Analysis ============
print(f"\n{'='*70}")
print(f"📊 Post-hoc Pairwise Comparisons")
print(f"{'='*70}")

from itertools import combinations

clusters = sorted(df['discharge_cluster'].unique())
n_comparisons = len(list(combinations(clusters, 2)))
alpha_corrected = 0.05 / n_comparisons if n_comparisons > 0 else 0.05

print(f"\nNumber of clusters: {len(clusters)}")
print(f"Pairwise comparisons: {n_comparisons}")
print(f"Bonferroni corrected α: {alpha_corrected:.4f}")

for var_name, var_label in test_variables.items():
    print(f"\n{var_label}:")
    
    for c1, c2 in combinations(clusters, 2):
        group1 = df[df['discharge_cluster'] == c1][var_name].dropna()
        group2 = df[df['discharge_cluster'] == c2][var_name].dropna()
        
        if len(group1) < 2 or len(group2) < 2:
            print(f"   Cluster {c1} vs {c2}: Insufficient data")
            continue
        
        # Mann-Whitney U test
        try:
            u_stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
            
            # Cohen's d
            mean_diff = group1.mean() - group2.mean()
            pooled_std = np.sqrt((group1.std()**2 + group2.std()**2) / 2)
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
            
            sig_symbol = '***' if p_value < alpha_corrected else ''
            
            print(f"   Cluster {c1} vs {c2}: p={p_value:.4f} {sig_symbol}, Cohen's d={cohens_d:.3f}")
        except Exception as e:
            print(f"   Cluster {c1} vs {c2}: Error - {str(e)}")

# ============ 5. Visualization ============
print(f"\n📈 Generating visualizations...")

fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)

colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']

# === Row 1: Box plots ===
charging_vars = [
    ('charging_soc_start', 'Charging Trigger SOC', '%'),
    ('charging_soc_gain', 'Charging Gain', '%'),
    ('charging_duration', 'Charging Duration', 'minutes'),
    ('time_to_next_charging', 'Time to Next Charging', 'hours')
]

for idx, (var, title, unit) in enumerate(charging_vars):
    ax = fig.add_subplot(gs[0, idx])
    
    # 准备数据
    data_to_plot = []
    labels = []
    for cluster in sorted(df['discharge_cluster'].unique()):
        data = df[df['discharge_cluster'] == cluster][var].dropna()
        
        if len(data) == 0:
            continue
        
        # 单位转换
        if var == 'charging_duration':
            data = data / 60
        elif var == 'time_to_next_charging':
            data = data / 3600
        
        data_to_plot.append(data)
        labels.append(f'C{cluster}')
    
    if len(data_to_plot) == 0:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
        ax.set_title(title, fontsize=11, fontweight='bold')
        continue
    
    # 箱线图
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                    widths=0.6, showfliers=False)
    
    for patch, color in zip(bp['boxes'], colors[:len(data_to_plot)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel(f'{title} ({unit})', fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    # 显著性标记
    test_result = df_test_results[df_test_results['Variable'] == title]
    if not test_result.empty and test_result.iloc[0]['Significant']:
        ax.text(0.5, 0.95, '***', transform=ax.transAxes, 
                ha='center', va='top', fontsize=16, color='red', fontweight='bold')

# === Row 2: Violin plots ===
discharge_vars = [
    ('discharge_speed_mean', 'Avg Speed', 'km/h'),
    ('discharge_harsh_accel', 'Harsh Accel', 'count'),
    ('discharge_soc_drop', 'SOC Drop', '%'),
    ('discharge_distance', 'Distance', 'km')
]

for idx, (var, title, unit) in enumerate(discharge_vars):
    ax = fig.add_subplot(gs[1, idx])
    
    # 准备数据
    plot_data = []
    positions = []
    plot_colors = []
    
    for i, cluster in enumerate(sorted(df['discharge_cluster'].unique())):
        data = df[df['discharge_cluster'] == cluster][var].dropna()
        if len(data) > 0:
            plot_data.append(data.values)
            positions.append(i)
            plot_colors.append(colors[cluster % len(colors)])
    
    if len(plot_data) == 0:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
        ax.set_title(f'{title} ({unit})', fontsize=11, fontweight='bold')
        continue
    
    # 小提琴图
    parts = ax.violinplot(plot_data, positions=positions, widths=0.7,
                          showmeans=True, showmedians=True)
    
    for pc, color in zip(parts['bodies'], plot_colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    
    ax.set_xticks(positions)
    ax.set_xticklabels([f'C{c}' for c in sorted(df['discharge_cluster'].unique()) 
                        if len(df[df['discharge_cluster'] == c][var].dropna()) > 0])
    ax.set_ylabel(f'{title} ({unit})', fontsize=10, fontweight='bold')
    ax.set_title(f'Discharge: {title}', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

# === Row 3: Scatter plots ===
scatter_pairs = [
    ('discharge_soc_drop', 'charging_soc_start', 'SOC Drop vs Trigger SOC'),
    ('discharge_distance', 'charging_soc_gain', 'Distance vs Charging Gain'),
    ('discharge_speed_mean', 'charging_duration', 'Speed vs Duration'),
    ('discharge_harsh_accel', 'time_to_next_charging', 'Harsh Accel vs Time')
]

for idx, (x_var, y_var, title) in enumerate(scatter_pairs):
    ax = fig.add_subplot(gs[2, idx])
    
    for cluster in sorted(df['discharge_cluster'].unique()):
        data = df[df['discharge_cluster'] == cluster]
        x = data[x_var].dropna()
        y = data[y_var].dropna()
        
        # 确保x和y长度一致
        common_idx = data[[x_var, y_var]].dropna().index
        x = data.loc[common_idx, x_var]
        y = data.loc[common_idx, y_var]
        
        if len(x) == 0:
            continue
        
        # 单位转换
        if y_var == 'charging_duration':
            y = y / 60
        elif y_var == 'time_to_next_charging':
            y = y / 3600
        
        ax.scatter(x, y, alpha=0.4, s=20, label=f'C{cluster}', 
                   color=colors[cluster % len(colors)])
    
    ax.set_xlabel(x_var.replace('_', ' ').title(), fontsize=9)
    ax.set_ylabel(y_var.replace('_', ' ').title(), fontsize=9)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.3)

plt.suptitle('Statistical Analysis: Discharge Profile Impact on Charging Behavior', 
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig('./results/statistical_analysis.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: statistical_analysis.png")

# ============ 6. Summary Report ============
print(f"\n{'='*70}")
print(f"📊 Analysis Summary")
print(f"{'='*70}")

print(f"\nSample Distribution:")
for cluster in sorted(df['discharge_cluster'].unique()):
    count = (df['discharge_cluster'] == cluster).sum()
    print(f"   Cluster {cluster}: {count:,} samples ({count/len(df)*100:.1f}%)")

print(f"\nStatistical Test Results:")
if len(df_test_results) > 0:
    print(df_test_results.to_string(index=False))
else:
    print("   No significant differences found")

print(f"\n{'='*70}")
print(f"✅ Step 2 Complete!")
print(f"{'='*70}")
print(f"\n📁 Generated files:")
print(f"   1. cluster_statistics.csv")
print(f"   2. statistical_test_results.csv")
print(f"   3. statistical_analysis.png")
print(f"{'='*70}")