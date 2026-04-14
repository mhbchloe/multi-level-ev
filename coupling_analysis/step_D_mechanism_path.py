"""
Step D: 3.3.4 行为–能耗耦合路径机制图
- 综合可视化
- 结构方程路径图
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import json
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("📊 Step D: Mechanism Path Diagram (Section 3.3.4)")
print("=" * 70)

OUTPUT_DIR = "./coupling_analysis/results/"
FIGURE_DIR = "./coupling_analysis/figures/"

df = pd.read_csv(os.path.join(OUTPUT_DIR, 'coupling_analysis_dataset.csv'))
print(f"   Loaded: {len(df):,} trips")

# ============================================================
# 1. 计算路径系数 (标准化回归)
# ============================================================
print(f"\n📊 Computing path coefficients...")

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

vars_needed = ['aggressiveness_index', 'soc_rate_per_hr', 'soc_drop',
               'charge_trigger_soc', 'charge_gain_soc']
vars_available = [v for v in vars_needed if v in df.columns]

df_path = df[vars_available].dropna()
scaler = StandardScaler()
df_std = pd.DataFrame(scaler.fit_transform(df_path), columns=vars_available)

# 路径 1: Aggressive → SOC_rate (能耗加速)
lr1 = LinearRegression().fit(df_std[['aggressiveness_index']], df_std['soc_rate_per_hr'])
beta_agg_to_rate = lr1.coef_[0]
r1, _ = stats.pearsonr(df_path['aggressiveness_index'], df_path['soc_rate_per_hr'])

# 路径 2: SOC_rate → Trigger SOC
lr2 = LinearRegression().fit(df_std[['soc_rate_per_hr', 'aggressiveness_index']],
                              df_std['charge_trigger_soc'])
beta_rate_to_trigger = lr2.coef_[0]
beta_agg_to_trigger_direct = lr2.coef_[1]

# 路径 3: Trigger SOC → SOC_gain
lr3 = LinearRegression().fit(df_std[['charge_trigger_soc', 'aggressiveness_index']],
                              df_std['charge_gain_soc'])
beta_trigger_to_gain = lr3.coef_[0]

paths = {
    'Aggressive → SOC Rate': round(beta_agg_to_rate, 3),
    'SOC Rate → Trigger SOC': round(beta_rate_to_trigger, 3),
    'Aggressive → Trigger SOC (direct)': round(beta_agg_to_trigger_direct, 3),
    'Trigger SOC → SOC Gain': round(beta_trigger_to_gain, 3),
}

print(f"\n   Path Coefficients (standardized β):")
for path_name, beta in paths.items():
    direction = "↑" if beta > 0 else "↓"
    print(f"      {path_name:<45} β = {beta:+.3f} {direction}")

indirect = beta_agg_to_rate * beta_rate_to_trigger
total = indirect + beta_agg_to_trigger_direct
print(f"\n   Indirect effect (via SOC rate): {indirect:+.3f}")
print(f"   Direct effect:                  {beta_agg_to_trigger_direct:+.3f}")
print(f"   Total effect:                   {total:+.3f}")

# ============================================================
# 2. 机制路径图 (手绘风格)
# ============================================================
print(f"\n📈 Drawing mechanism path diagram...")

fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(-1, 11)
ax.set_ylim(-1, 8)
ax.axis('off')

# 节点位置
nodes = {
    'Aggressive\nDriving': (1, 6),
    'Energy\nConsumption\nRate': (5, 6),
    'Range\nAnxiety': (5, 3.5),
    'Charging\nTrigger SOC↑': (9, 6),
    'Charging\nAmount↑': (9, 3.5),
    'Detour\nDistance↑': (9, 1),
}

node_colors = {
    'Aggressive\nDriving': '#e74c3c',
    'Energy\nConsumption\nRate': '#f39c12',
    'Range\nAnxiety': '#9b59b6',
    'Charging\nTrigger SOC↑': '#3498db',
    'Charging\nAmount↑': '#2ecc71',
    'Detour\nDistance↑': '#1abc9c',
}

# 绘制节点
for name, (x, y) in nodes.items():
    color = node_colors[name]
    box = mpatches.FancyBboxPatch((x-1.2, y-0.7), 2.4, 1.4,
                                   boxstyle="round,pad=0.15",
                                   facecolor=color, alpha=0.2,
                                   edgecolor=color, linewidth=2.5)
    ax.add_patch(box)
    ax.text(x, y, name, ha='center', va='center',
            fontsize=11, fontweight='bold', color=color)

# 绘制箭头和路径系数
arrows = [
    ('Aggressive\nDriving', 'Energy\nConsumption\nRate',
     f'β={beta_agg_to_rate:+.2f}', 'black'),
    ('Energy\nConsumption\nRate', 'Range\nAnxiety',
     'Uncertainty↑', '#9b59b6'),
    ('Range\nAnxiety', 'Charging\nTrigger SOC↑',
     f'β={beta_rate_to_trigger:+.2f}', 'black'),
    ('Charging\nTrigger SOC↑', 'Charging\nAmount↑',
     f'β={beta_trigger_to_gain:+.2f}', 'black'),
    ('Charging\nTrigger SOC↑', 'Detour\nDistance↑',
     'More options', '#1abc9c'),
]

for src, dst, label, color in arrows:
    x1, y1 = nodes[src]
    x2, y2 = nodes[dst]

    # 调整箭头起止点 (从框边缘出发)
    dx, dy = x2 - x1, y2 - y1
    dist = np.sqrt(dx**2 + dy**2)
    offset = 1.3 / dist

    ax.annotate('',
                xy=(x2 - dx*offset, y2 - dy*offset),
                xytext=(x1 + dx*offset, y1 + dy*offset),
                arrowprops=dict(arrowstyle='->', color=color,
                               lw=2.5, connectionstyle='arc3,rad=0.05'))

    # 标签
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    ax.text(mx, my + 0.3, label, ha='center', va='bottom',
            fontsize=9, fontweight='bold', color=color,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                     edgecolor=color, alpha=0.8))

# 直接效应虚线
x1, y1 = nodes['Aggressive\nDriving']
x2, y2 = nodes['Charging\nTrigger SOC↑']
ax.annotate('',
            xy=(x2 - 1.3, y2),
            xytext=(x1 + 1.3, y1),
            arrowprops=dict(arrowstyle='->', color='grey', lw=1.5,
                           linestyle='dashed', connectionstyle='arc3,rad=-0.3'))
ax.text(5, 7.3, f'Direct: β={beta_agg_to_trigger_direct:+.2f}',
        ha='center', fontsize=9, color='grey', fontstyle='italic')

# 标题
ax.text(5, -0.5,
        'Coupling Mechanism: Aggressive Driving → Energy Acceleration → '
        'Range Anxiety → Earlier Charging → More Energy & Detour',
        ha='center', fontsize=11, fontstyle='italic', color='#555')

ax.set_title('Section 3.3.4: Behavior–Energy–Charging Coupling Mechanism',
             fontsize=15, fontweight='bold', pad=20)

fig.savefig(os.path.join(FIGURE_DIR, 'fig_334_mechanism_path.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(FIGURE_DIR, 'fig_334_mechanism_path.pdf'), bbox_inches='tight')
plt.close(fig)
print(f"   ✅ Saved: fig_334_mechanism_path.png/pdf")

# ============================================================
# 3. 综合相关性热力图
# ============================================================
fig, ax = plt.subplots(figsize=(12, 10))

corr_vars = [
    'aggressiveness_index', 'speed_cv', 'idle_ratio',
    'soc_rate_per_hr', 'soc_drop', 'trip_duration_hrs',
    'charge_trigger_soc', 'charge_gain_soc', 'charge_duration_min',
]
corr_vars = [v for v in corr_vars if v in df.columns]

labels_pretty = {
    'aggressiveness_index': 'Aggressiveness',
    'speed_cv': 'Speed CV',
    'idle_ratio': 'Idle Ratio',
    'soc_rate_per_hr': 'SOC Rate (%/hr)',
    'soc_drop': 'SOC Drop',
    'trip_duration_hrs': 'Duration (hrs)',
    'charge_trigger_soc': 'Trigger SOC ★',
    'charge_gain_soc': 'SOC Gain ★',
    'charge_duration_min': 'Charge Duration ★',
}

corr = df[corr_vars].corr(method='spearman')
pretty_labels = [labels_pretty.get(v, v) for v in corr_vars]

im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
plt.colorbar(im, ax=ax, label='Spearman ρ', shrink=0.8)

ax.set_xticks(range(len(corr_vars)))
ax.set_yticks(range(len(corr_vars)))
ax.set_xticklabels(pretty_labels, rotation=45, ha='right', fontsize=10)
ax.set_yticklabels(pretty_labels, fontsize=10)

for i in range(len(corr_vars)):
    for j in range(len(corr_vars)):
        val = corr.values[i, j]
        color = 'white' if abs(val) > 0.5 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                fontsize=8, color=color, fontweight='bold' if abs(val) > 0.3 else 'normal')

ax.set_title('Spearman Correlation: Driving Behavior ↔ Charging Decisions\n(★ = dependent variables)',
             fontsize=13, fontweight='bold', pad=15)

fig.savefig(os.path.join(FIGURE_DIR, 'fig_334_correlation_heatmap.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(FIGURE_DIR, 'fig_334_correlation_heatmap.pdf'), bbox_inches='tight')
plt.close(fig)
print(f"   ✅ Saved: fig_334_correlation_heatmap.png/pdf")

# 保存结果
all_results = {
    'path_coefficients': paths,
    'indirect_effect': float(indirect),
    'direct_effect': float(beta_agg_to_trigger_direct),
    'total_effect': float(total),
}
with open(os.path.join(OUTPUT_DIR, 'step_D_results.json'), 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\n✅ Step D Complete!")
print(f"\n{'=' * 70}")
print(f"📋 ALL FIGURES FOR SECTION 3.3:")
print(f"{'=' * 70}")
for fn in sorted(os.listdir(FIGURE_DIR)):
    if fn.startswith('fig_33'):
        fp = os.path.join(FIGURE_DIR, fn)
        print(f"   {fn:<50} {os.path.getsize(fp)/1024:.0f} KB")