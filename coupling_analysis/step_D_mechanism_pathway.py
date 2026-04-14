"""
Step D: 3.3.4 行为-能耗-充电耦合机制路径图
基于 Step B/C/F 的实际发现重新构建

核心发现:
1. 驾驶行为 → 充电决策的直接效应极弱 (ρ≈-0.04)
2. soc_drop 是充电决策的绝对主导因素 (|SHAP|=12.5)
3. 驾驶行为通过加速 SOC 消耗间接传导
4. 车辆画像存在独立于驾驶行为的充电偏好差异
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import json
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("📊 Step D: Behavior-Energy-Charging Coupling Mechanism")
print("   (Section 3.3.4) — Updated with actual findings")
print("=" * 70)

OUTPUT_DIR = "./coupling_analysis/results/"
FIGURE_DIR = "./coupling_analysis/figures/"
os.makedirs(FIGURE_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(OUTPUT_DIR, 'coupling_dataset_with_genes.csv'))
print(f"   Loaded: {len(df):,} trips")

# 自动检测 ratio 列
ratio_cols = sorted([c for c in df.columns if c.startswith('ratio_')])
print(f"   Ratio columns: {ratio_cols}")

# ============================================================
# 1. 计算所有路径系数 (标准化回归系数)
# ============================================================
print(f"\n{'=' * 70}")
print("📐 D1: Path Coefficient Estimation")
print(f"{'=' * 70}")

def standardized_regression(df, x_cols, y_col):
    """计算标准化回归系数"""
    data = df[x_cols + [y_col]].dropna()
    if len(data) < 100:
        return {col: 0.0 for col in x_cols}, 0.0

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X = scaler_x.fit_transform(data[x_cols].values)
    y = scaler_y.fit_transform(data[[y_col]].values).ravel()

    reg = LinearRegression().fit(X, y)
    betas = {col: float(b) for col, b in zip(x_cols, reg.coef_)}
    r2 = float(reg.score(X, y))
    return betas, r2

# --- 路径 A: 行程基因 → soc_drop ---
print("\n   Path A: Trip Genes → SOC Drop")
betas_a, r2_a = standardized_regression(df, ratio_cols, 'soc_drop')
for col, b in betas_a.items():
    print(f"      {col:<25} β = {b:+.4f}")
print(f"      R² = {r2_a:.4f}")

# --- 路径 B: soc_drop → charge_trigger_soc ---
print("\n   Path B: SOC Drop → Trigger SOC")
betas_b, r2_b = standardized_regression(df, ['soc_drop'], 'charge_trigger_soc')
print(f"      soc_drop               β = {betas_b['soc_drop']:+.4f}")
print(f"      R² = {r2_b:.4f}")

# --- 路径 C: 行程基因 → trigger_soc (直接效应) ---
print("\n   Path C: Trip Genes → Trigger SOC (direct)")
betas_c, r2_c = standardized_regression(df, ratio_cols, 'charge_trigger_soc')
for col, b in betas_c.items():
    print(f"      {col:<25} β = {b:+.4f}")
print(f"      R² = {r2_c:.4f}")

# --- 路径 D: 行程基因 + soc_drop → trigger_soc (控制中介变量后) ---
print("\n   Path D: Trip Genes + SOC Drop → Trigger SOC (controlled)")
betas_d, r2_d = standardized_regression(df, ratio_cols + ['soc_drop'], 'charge_trigger_soc')
for col, b in betas_d.items():
    print(f"      {col:<25} β = {b:+.4f}")
print(f"      R² = {r2_d:.4f}")

# --- 路径 E: trigger_soc → charge_gain_soc ---
print("\n   Path E: Trigger SOC → SOC Gain")
betas_e, r2_e = standardized_regression(df, ['charge_trigger_soc', 'soc_drop'],
                                         'charge_gain_soc')
for col, b in betas_e.items():
    print(f"      {col:<25} β = {b:+.4f}")
print(f"      R² = {r2_e:.4f}")

# --- 路径 F: 车辆画像效应 ---
if 'vehicle_archetype' in df.columns:
    print("\n   Path F: Vehicle Archetype → Trigger SOC")
    dummies = pd.get_dummies(df['vehicle_archetype'], prefix='vtype')
    df_temp = pd.concat([df, dummies], axis=1)
    vtype_cols = list(dummies.columns)
    betas_f, r2_f = standardized_regression(df_temp, vtype_cols, 'charge_trigger_soc')
    for col, b in betas_f.items():
        print(f"      {col:<25} β = {b:+.4f}")
    print(f"      R² = {r2_f:.4f}")
else:
    betas_f, r2_f = {}, 0.0

# --- 中介效应分解 ---
print(f"\n{'=' * 70}")
print("📊 D2: Mediation Analysis (Indirect Effect)")
print(f"{'=' * 70}")

# 间接效应 = β(genes→soc_drop) × β(soc_drop→trigger_soc)
beta_soc_drop_to_trigger = betas_d.get('soc_drop', betas_b['soc_drop'])

print(f"\n   Indirect effect via SOC Drop:")
for col in ratio_cols:
    indirect = betas_a[col] * beta_soc_drop_to_trigger
    direct = betas_d.get(col, 0)
    total = betas_c.get(col, 0)
    ratio = abs(indirect) / (abs(indirect) + abs(direct)) * 100 if (abs(indirect) + abs(direct)) > 0 else 0

    print(f"      {col}:")
    print(f"         Direct:   β = {direct:+.4f}")
    print(f"         Indirect: β = {indirect:+.4f}  ({betas_a[col]:+.4f} × {beta_soc_drop_to_trigger:+.4f})")
    print(f"         Total:    β = {total:+.4f}")
    print(f"         Mediation ratio: {ratio:.1f}%")

# ============================================================
# 2. 相关性热力图
# ============================================================
print(f"\n{'=' * 70}")
print("📈 D3: Correlation Heatmap")
print(f"{'=' * 70}")

heatmap_vars = ratio_cols + ['soc_drop', 'trip_duration_hrs',
                              'charge_trigger_soc', 'charge_gain_soc']
heatmap_vars = [v for v in heatmap_vars if v in df.columns]

heatmap_labels = []
for v in heatmap_vars:
    label = v.replace('ratio_', '').replace('_', ' ').title()
    if v == 'charge_trigger_soc':
        label = 'Trigger SOC ★'
    elif v == 'charge_gain_soc':
        label = 'SOC Gain ★'
    elif v == 'soc_drop':
        label = 'SOC Drop'
    elif v == 'trip_duration_hrs':
        label = 'Duration (hrs)'
    heatmap_labels.append(label)

corr = df[heatmap_vars].corr(method='spearman')

fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

ax.set_xticks(range(len(heatmap_labels)))
ax.set_yticks(range(len(heatmap_labels)))
ax.set_xticklabels(heatmap_labels, rotation=45, ha='right', fontsize=10)
ax.set_yticklabels(heatmap_labels, fontsize=10)

for i in range(len(corr)):
    for j in range(len(corr)):
        val = corr.iloc[i, j]
        color = 'white' if abs(val) > 0.5 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                color=color, fontsize=9, fontweight='bold')

plt.colorbar(im, ax=ax, label='Spearman ρ', shrink=0.8)
ax.set_title('Spearman Correlation: Trip Genes ↔ Charging Decisions\n(★ = dependent variables)',
             fontweight='bold', fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(FIGURE_DIR, 'fig_334_correlation_heatmap.png'),
            dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(FIGURE_DIR, 'fig_334_correlation_heatmap.pdf'),
            bbox_inches='tight')
plt.close(fig)
print("   ✅ fig_334_correlation_heatmap.png/pdf")

# ============================================================
# 3. 机制路径图 (核心图)
# ============================================================
print(f"\n{'=' * 70}")
print("🗺️ D4: Mechanism Pathway Diagram")
print(f"{'=' * 70}")

fig, ax = plt.subplots(figsize=(16, 11))
ax.set_xlim(0, 16)
ax.set_ylim(0, 11)
ax.axis('off')

# ── 定义节点位置和样式 ──
nodes = {
    'genes': {
        'xy': (1.5, 8.5), 'w': 3.2, 'h': 1.6,
        'color': '#FADBD8', 'edge': '#E74C3C',
        'label': 'Micro Driving\nMode Sequence\n(Trip Genes)',
        'sublabel': 'Long Idle / Urban /\nShort Idle / Highway'
    },
    'soc_drop': {
        'xy': (7.5, 8.5), 'w': 3.0, 'h': 1.6,
        'color': '#FDF2E9', 'edge': '#F39C12',
        'label': 'Cumulative\nSOC Drop\n(Energy State)',
        'sublabel': '|SHAP| = 12.5'
    },
    'trigger': {
        'xy': (13.0, 8.5), 'w': 2.8, 'h': 1.6,
        'color': '#D4E6F1', 'edge': '#2980B9',
        'label': 'Charging\nTrigger SOC',
        'sublabel': 'Y₁'
    },
    'gain': {
        'xy': (13.0, 4.5), 'w': 2.8, 'h': 1.6,
        'color': '#D5F5E3', 'edge': '#27AE60',
        'label': 'Charging\nAmount (Gain)',
        'sublabel': 'Y₂'
    },
    'vtype': {
        'xy': (7.5, 4.5), 'w': 3.0, 'h': 1.6,
        'color': '#E8DAEF', 'edge': '#8E44AD',
        'label': 'Vehicle\nArchetype',
        'sublabel': '|SHAP| = 0.8'
    },
    'deviation': {
        'xy': (1.5, 4.5), 'w': 3.2, 'h': 1.6,
        'color': '#D1F2EB', 'edge': '#1ABC9C',
        'label': 'Energy\nDeviation',
        'sublabel': '(Detour Proxy)'
    },
}

# 绘制节点
for key, node in nodes.items():
    x, y = node['xy']
    w, h = node['w'], node['h']
    fancy = FancyBboxPatch((x - w/2, y - h/2), w, h,
                           boxstyle="round,pad=0.15",
                           facecolor=node['color'],
                           edgecolor=node['edge'],
                           linewidth=2.5)
    ax.add_patch(fancy)
    ax.text(x, y + 0.15, node['label'], ha='center', va='center',
            fontsize=10, fontweight='bold', linespacing=1.2)
    if node.get('sublabel'):
        ax.text(x, y - h/2 + 0.25, node['sublabel'], ha='center', va='center',
                fontsize=8, color='grey', style='italic')

# ── 定义箭头和路径系数 ──
def draw_arrow(ax, start, end, label, color='black', style='->', lw=2.5,
               connectionstyle='arc3,rad=0', fontsize=11, label_offset=(0, 0),
               linestyle='-', alpha=1.0):
    """绘制带标签的箭头"""
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle=style, color=color,
        linewidth=lw, linestyle=linestyle,
        connectionstyle=connectionstyle,
        mutation_scale=20, alpha=alpha,
        zorder=3
    )
    ax.add_patch(arrow)

    mid_x = (start[0] + end[0]) / 2 + label_offset[0]
    mid_y = (start[1] + end[1]) / 2 + label_offset[1]
    ax.text(mid_x, mid_y, label, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color=color,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=color, alpha=0.9))

# 获取实际路径系数
# 找到行程基因中影响 soc_drop 最大的
max_gene_to_soc = max(betas_a.items(), key=lambda x: abs(x[1]))
avg_gene_to_soc = np.mean([abs(v) for v in betas_a.values()])

# Path 1: Genes → SOC Drop (主路径)
draw_arrow(ax, (3.1, 8.5), (6.0, 8.5),
           f'β={avg_gene_to_soc:+.3f}\n(avg)',
           color='#E74C3C', lw=3, label_offset=(0, 0.5))

# Path 2: SOC Drop → Trigger SOC (最强路径)
draw_arrow(ax, (9.0, 8.5), (11.6, 8.5),
           f'β={betas_b["soc_drop"]:+.3f}',
           color='#D35400', lw=4, label_offset=(0, 0.5))

# Path 3: Genes → Trigger SOC (直接效应，弱，虚线)
avg_gene_direct = np.mean([abs(betas_d.get(c, 0)) for c in ratio_cols])
draw_arrow(ax, (2.5, 7.7), (12.0, 9.3),
           f'Direct: β≈{avg_gene_direct:+.3f}',
           color='#95A5A6', lw=1.5, linestyle='--',
           connectionstyle='arc3,rad=-0.2',
           label_offset=(2.0, 1.0), alpha=0.6)

# Path 4: Trigger SOC → Gain
beta_trigger_gain = betas_e.get('charge_trigger_soc', -0.50)
draw_arrow(ax, (13.0, 7.7), (13.0, 5.3),
           f'β={beta_trigger_gain:+.3f}',
           color='#2980B9', lw=3, label_offset=(1.0, 0))

# Path 5: Vehicle Archetype → Trigger SOC
if betas_f:
    max_vtype_beta = max(betas_f.values(), key=abs)
    draw_arrow(ax, (9.0, 5.0), (11.6, 8.0),
               f'β≈{max_vtype_beta:+.3f}',
               color='#8E44AD', lw=2,
               connectionstyle='arc3,rad=0.2',
               label_offset=(0.3, 0))

# Path 6: Genes → Energy Deviation
draw_arrow(ax, (1.5, 7.7), (1.5, 5.3),
           f'ρ=0.79',
           color='#1ABC9C', lw=2.5, label_offset=(0.8, 0))

# ── 添加效应大小标注框 ──
info_text = (
    "Effect Size Summary\n"
    "─────────────────────\n"
    f"SOC Drop → Trigger SOC:  |SHAP|=12.5  ██████████\n"
    f"Trip Duration → Trigger: |SHAP|= 1.5  █\n"
    f"Vehicle Type → Trigger:  |SHAP|= 0.8  ▌\n"
    f"Trip Genes → Trigger:    |SHAP|≈ 0.3  ▏\n"
    "─────────────────────\n"
    "Model: XGBoost, R²=0.559"
)
ax.text(5.0, 2.0, info_text, fontsize=9, fontfamily='monospace',
        va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#FDFEFE',
                  edgecolor='#BDC3C7', linewidth=1.5))

# ── 添加核心结论 ──
conclusion = (
    "Core Finding: Charging decisions are primarily \"SOC-threshold-driven\".\n"
    "Driving behavior influences charging indirectly through energy depletion,\n"
    "but users ultimately respond to remaining battery level, not driving style."
)
ax.text(8.0, 0.6, conclusion, fontsize=10, ha='center', va='center',
        style='italic', color='#2C3E50',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#EBF5FB',
                  edgecolor='#2980B9', linewidth=1.5))

ax.set_title('Section 3.3.4: Behavior–Energy–Charging Coupling Mechanism\n'
             '(Based on Empirical Findings)',
             fontsize=15, fontweight='bold', pad=20)

fig.savefig(os.path.join(FIGURE_DIR, 'fig_334_mechanism_pathway.png'),
            dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(FIGURE_DIR, 'fig_334_mechanism_pathway.pdf'),
            bbox_inches='tight')
plt.close(fig)
print("   ✅ fig_334_mechanism_pathway.png/pdf")

# ============================================================
# 4. 效应对比图 (补充)
# ============================================================
print(f"\n{'=' * 70}")
print("📊 D5: Effect Comparison Chart")
print(f"{'=' * 70}")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# (a) 直接 vs 间接效应对比
ax = axes[0]
categories = [c.replace('ratio_', '').replace('_', ' ').title() for c in ratio_cols]
direct_effects = [abs(betas_d.get(c, 0)) for c in ratio_cols]
indirect_effects = [abs(betas_a[c] * beta_soc_drop_to_trigger) for c in ratio_cols]

x_pos = np.arange(len(categories))
width = 0.35
bars1 = ax.bar(x_pos - width/2, direct_effects, width, label='Direct Effect',
               color='#3498DB', alpha=0.8)
bars2 = ax.bar(x_pos + width/2, indirect_effects, width, label='Indirect (via SOC Drop)',
               color='#E74C3C', alpha=0.8)

ax.set_xticks(x_pos)
ax.set_xticklabels(categories, rotation=15, fontsize=9)
ax.set_ylabel('|Standardized β|')
ax.set_title('(a) Direct vs Indirect Effects\non Trigger SOC', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis='y')

# (b) SHAP 贡献层级 (从 Step F 结果)
ax = axes[1]
try:
    with open(os.path.join(OUTPUT_DIR, 'step_F_results.json')) as f:
        stepf = json.load(f)
    shap_imp = stepf['feature_importance']

    # 按层级分组
    groups = {
        'SOC Drop': shap_imp.get('soc_drop', 0),
        'Trip Duration': shap_imp.get('trip_duration_hrs', 0),
        'Vehicle Type': sum(v for k, v in shap_imp.items() if 'vtype' in k),
        'Trip Genes\n(all ratios)': sum(v for k, v in shap_imp.items() if 'ratio' in k),
    }
    g_names = list(groups.keys())
    g_vals = list(groups.values())
    g_colors = ['#E74C3C', '#F39C12', '#8E44AD', '#2ECC71']

    bars = ax.barh(range(len(g_names)), g_vals, color=g_colors, alpha=0.8)
    ax.set_yticks(range(len(g_names)))
    ax.set_yticklabels(g_names, fontsize=11)
    ax.set_xlabel('Sum |SHAP|')
    ax.set_title('(b) SHAP Contribution by Category', fontweight='bold')

    for bar, v in zip(bars, g_vals):
        ax.text(v + 0.1, bar.get_y() + bar.get_height()/2,
                f'{v:.2f}', va='center', fontweight='bold', fontsize=10)
    ax.grid(alpha=0.2, axis='x')
except:
    ax.text(0.5, 0.5, 'Step F results not found', ha='center', va='center',
            transform=ax.transAxes)

# (c) 中介效应比例
ax = axes[2]
mediation_ratios = []
mediation_labels = []
for col in ratio_cols:
    indirect = abs(betas_a[col] * beta_soc_drop_to_trigger)
    direct = abs(betas_d.get(col, 0))
    total = indirect + direct
    if total > 0.001:
        ratio = indirect / total * 100
        mediation_ratios.append(ratio)
        mediation_labels.append(col.replace('ratio_', '').replace('_', ' ').title())

if mediation_ratios:
    colors_med = ['#2ECC71' if r > 50 else '#E74C3C' for r in mediation_ratios]
    bars = ax.barh(range(len(mediation_labels)), mediation_ratios,
                   color=colors_med, alpha=0.8)
    ax.axvline(50, color='black', linestyle='--', linewidth=1.5, label='50% threshold')
    ax.set_yticks(range(len(mediation_labels)))
    ax.set_yticklabels(mediation_labels, fontsize=10)
    ax.set_xlabel('Mediation Ratio (%)')
    ax.set_xlim(0, 100)
    ax.set_title('(c) % Effect Mediated by SOC Drop', fontweight='bold')
    ax.legend()

    for bar, v in zip(bars, mediation_ratios):
        ax.text(v + 1, bar.get_y() + bar.get_height()/2,
                f'{v:.0f}%', va='center', fontweight='bold', fontsize=10)
    ax.grid(alpha=0.2, axis='x')

plt.suptitle('Section 3.3.4: Effect Decomposition & Mediation Analysis',
             fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(FIGURE_DIR, 'fig_334_effect_comparison.png'),
            dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(FIGURE_DIR, 'fig_334_effect_comparison.pdf'),
            bbox_inches='tight')
plt.close(fig)
print("   ✅ fig_334_effect_comparison.png/pdf")

# ============================================================
# 5. 保存所有结果
# ============================================================
results_d = {
    'path_coefficients': {
        'genes_to_soc_drop': betas_a,
        'soc_drop_to_trigger': betas_b,
        'genes_to_trigger_direct': {c: betas_d.get(c, 0) for c in ratio_cols},
        'genes_to_trigger_total': betas_c,
        'trigger_to_gain': betas_e,
        'vehicle_type_to_trigger': betas_f,
    },
    'r_squared': {
        'genes_to_soc_drop': r2_a,
        'soc_drop_to_trigger': r2_b,
        'genes_to_trigger_direct': r2_c,
        'full_model': r2_d,
        'trigger_to_gain': r2_e,
    },
    'mediation': {},
    'key_findings': [
        "Charging decisions are primarily SOC-threshold-driven",
        f"SOC Drop dominates prediction (|SHAP|=12.5, β={betas_b['soc_drop']:+.3f})",
        "Trip gene composition has weak direct effect on charging trigger",
        "Driving behavior influences charging indirectly through energy depletion",
        "Vehicle archetype shows independent charging preference differences",
    ]
}

for col in ratio_cols:
    indirect = betas_a[col] * beta_soc_drop_to_trigger
    direct = betas_d.get(col, 0)
    total = betas_c.get(col, 0)
    ratio_med = abs(indirect) / (abs(indirect) + abs(direct)) * 100 \
                if (abs(indirect) + abs(direct)) > 0 else 0
    results_d['mediation'][col] = {
        'direct': float(direct),
        'indirect': float(indirect),
        'total': float(total),
        'mediation_pct': float(ratio_med),
    }

with open(os.path.join(OUTPUT_DIR, 'step_D_results.json'), 'w') as f:
    json.dump(results_d, f, indent=2, ensure_ascii=False)

print(f"\n💾 Saved: step_D_results.json")

# 打印最终结论
print(f"\n{'=' * 70}")
print("📋 Section 3.3.4: Key Findings")
print(f"{'=' * 70}")
for i, finding in enumerate(results_d['key_findings'], 1):
    print(f"   {i}. {finding}")

print(f"\n📊 All figures generated:")
for fn in sorted(os.listdir(FIGURE_DIR)):
    if fn.startswith('fig_334'):
        fp = os.path.join(FIGURE_DIR, fn)
        print(f"   {fn:<55} {os.path.getsize(fp)/1024:.0f} KB")

print(f"\n✅ Step D Complete!")