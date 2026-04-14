"""
Step F: 3.3.2 用行程基因 (片段占比) 预测充电触发 SOC
自动检测 ratio 列名，不再硬编码

X = [ratio_long_idle, ratio_urban, ratio_short_idle, ratio_highway,
     trip_duration_hrs, soc_drop, vehicle_archetype_dummies]
Y = charge_trigger_soc
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import cross_val_score, KFold
import xgboost as xgb
import shap
import builtins
import json
import os
import warnings
warnings.filterwarnings('ignore')

# shap 兼容补丁
_original_float = builtins.float
def _safe_float(x):
    if isinstance(x, str):
        x = x.strip('[]')
    return _original_float(x)

def shap_tree_safe(model):
    builtins.float = _safe_float
    try:
        exp = shap.TreeExplainer(model)
    finally:
        builtins.float = _original_float
    return exp

print("=" * 70)
print("📊 Step F: Trip Genes → Charging Trigger SOC (Section 3.3.2)")
print("=" * 70)

OUTPUT_DIR = "./coupling_analysis/results/"
FIGURE_DIR = "./coupling_analysis/figures/"
os.makedirs(FIGURE_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(OUTPUT_DIR, 'coupling_dataset_with_genes.csv'))
print(f"   Loaded: {len(df):,} trips")

# ── 自动检测 ratio 列 ──
ratio_cols = sorted([c for c in df.columns if c.startswith('ratio_')])
print(f"   Detected ratio columns: {ratio_cols}")

# 找到最"激进"的列 (highway 或包含 aggressive 的)
# 按该列均值的 soc_rate 相关性排序，或者直接用 highway
highway_col = None
for col in ratio_cols:
    if 'highway' in col.lower() or 'aggressive' in col.lower():
        highway_col = col
        break
if highway_col is None:
    # 找与 soc_drop 相关性最高的 ratio 列
    corrs = {col: abs(df[col].corr(df['soc_drop'])) for col in ratio_cols if col in df.columns}
    highway_col = max(corrs, key=corrs.get) if corrs else ratio_cols[-1]

print(f"   Most 'aggressive' column: {highway_col}")

# 加载聚类颜色 (如果有)
cluster_colors_map = {}
try:
    with open('./analysis_complete_vehicles/results/clustering_v3/cluster_names.json') as f:
        cn = json.load(f)
    for k, v in cn.items():
        short = v['short'].lower().replace(' ', '_')
        col_name = f"ratio_{short}"
        cluster_colors_map[col_name] = v['color']
except:
    pass

# 默认颜色
default_colors = ['#5B9BD5', '#70AD47', '#C0504D', '#FFC000', '#9b59b6', '#1abc9c']
for i, col in enumerate(ratio_cols):
    if col not in cluster_colors_map:
        cluster_colors_map[col] = default_colors[i % len(default_colors)]

# ============================================================
# 1. 变量定义
# ============================================================
# X1: 行程基因
gene_cols = ratio_cols

# X2: 累积状态
accum_cols = ['trip_duration_hrs', 'soc_drop']

# X3: 宏观标签 (如有)
macro_cols = []
if 'vehicle_archetype' in df.columns:
    dummies = pd.get_dummies(df['vehicle_archetype'], prefix='vtype')
    df = pd.concat([df, dummies], axis=1)
    macro_cols = list(dummies.columns)

feature_cols = gene_cols + accum_cols + macro_cols
feature_cols = [f for f in feature_cols if f in df.columns]

target_col = 'charge_trigger_soc'

print(f"\n   X1 (Trip Genes):  {gene_cols}")
print(f"   X2 (Cumulative):  {accum_cols}")
print(f"   X3 (Macro):       {macro_cols if macro_cols else 'N/A'}")
print(f"   All features:     {feature_cols}")
print(f"   Y:                {target_col}")

# ============================================================
# 2. 相关性分析
# ============================================================
print(f"\n{'=' * 70}")
print("📈 F1: Correlation Analysis")
print(f"{'=' * 70}")

for col in gene_cols + accum_cols:
    if col in df.columns:
        valid = df[[col, target_col]].dropna()
        rho, p = stats.spearmanr(valid[col], valid[target_col])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"   {col:<30} ρ = {rho:+.4f}  p={'<0.001' if p<0.001 else f'{p:.4f}'} {sig}")

# ============================================================
# 3. XGBoost
# ============================================================
print(f"\n{'=' * 70}")
print("🤖 F2: XGBoost Prediction")
print(f"{'=' * 70}")

df_model = df[feature_cols + [target_col]].dropna()
X = df_model[feature_cols].values
y = df_model[target_col].values

print(f"   Features ({len(feature_cols)}): {feature_cols}")
print(f"   Samples: {len(X):,}")

model = xgb.XGBRegressor(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, n_jobs=-1,
)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2 = cross_val_score(model, X, y, cv=kf, scoring='r2')
cv_mae = -cross_val_score(model, X, y, cv=kf, scoring='neg_mean_absolute_error')

print(f"\n   5-Fold CV:")
print(f"      R²:  {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
print(f"      MAE: {cv_mae.mean():.2f} ± {cv_mae.std():.2f}")

model.fit(X, y)

# ============================================================
# 4. SHAP
# ============================================================
print(f"\n{'=' * 70}")
print("🔍 F3: SHAP Explanation")
print(f"{'=' * 70}")

n_shap = min(10000, len(X))
idx = np.random.RandomState(42).choice(len(X), n_shap, replace=False)
X_shap = X[idx]

explainer = shap_tree_safe(model)
shap_values = explainer.shap_values(X_shap)

abs_shap = np.abs(shap_values).mean(axis=0)
imp_order = np.argsort(abs_shap)[::-1]

print(f"   SHAP on {n_shap:,} samples")
print(f"\n   Feature Importance:")
for rank, i in enumerate(imp_order):
    print(f"      {rank+1}. {feature_cols[i]:<30} |SHAP|={abs_shap[i]:.4f}")

# ============================================================
# 5. 可视化
# ============================================================
print(f"\n📈 Generating figures...")

# --- Fig 1: SHAP Beeswarm ---
fig_bee, _ = plt.subplots(figsize=(10, 7))
shap.summary_plot(shap_values, X_shap, feature_names=feature_cols,
                  show=False, max_display=len(feature_cols))
plt.title(f'SHAP Summary: Trigger SOC (R²={cv_r2.mean():.3f})',
          fontweight='bold', fontsize=13)
plt.tight_layout()
fig_bee.savefig(os.path.join(FIGURE_DIR, 'fig_332_gene_shap_beeswarm.png'),
                dpi=300, bbox_inches='tight')
plt.close(fig_bee)
print("   ✅ fig_332_gene_shap_beeswarm.png")

# --- Fig 2: 主面板 (2x2) ---
fig, axes = plt.subplots(2, 2, figsize=(16, 13))

# (a) SHAP Bar
ax = axes[0, 0]
n_feat = len(feature_cols)
ax.barh(range(n_feat), abs_shap[imp_order][::-1], color='#e74c3c', alpha=0.8)
ax.set_yticks(range(n_feat))
ax.set_yticklabels([feature_cols[i] for i in imp_order][::-1], fontsize=10)
ax.set_xlabel('Mean |SHAP|')
ax.set_title(f'(a) Feature Importance (R²={cv_r2.mean():.3f})', fontweight='bold')
ax.grid(alpha=0.2, axis='x')

# (b) SHAP Dependence: highway/aggressive column
ax = axes[0, 1]
if highway_col in feature_cols:
    hw_idx = feature_cols.index(highway_col)
    shap.dependence_plot(hw_idx, shap_values, X_shap,
                         feature_names=feature_cols, ax=ax, show=False)
    ax.set_title(f'(b) SHAP: {highway_col} → Trigger SOC', fontweight='bold')

# (c) 最激进列 vs Trigger SOC 趋势线
ax = axes[1, 0]
x_hw = df[highway_col].values
y_trig = df['charge_trigger_soc'].values
valid = ~(np.isnan(x_hw) | np.isnan(y_trig))
x_hw, y_trig = x_hw[valid], y_trig[valid]

n_bins = 15
q99 = np.percentile(x_hw, 99)
if q99 > 0.01:
    bins = np.linspace(0, q99, n_bins + 1)
    centers, means, ses = [], [], []
    for i in range(len(bins)-1):
        mask = (x_hw >= bins[i]) & (x_hw < bins[i+1])
        if mask.sum() > 20:
            centers.append((bins[i]+bins[i+1])/2)
            means.append(np.mean(y_trig[mask]))
            ses.append(np.std(y_trig[mask]) / np.sqrt(mask.sum()))

    centers = np.array(centers)
    means = np.array(means)
    ses = np.array(ses)

    ax.plot(centers, means, 'ro-', linewidth=2.5, markersize=7, label='Mean Trigger SOC')
    ax.fill_between(centers, means-1.96*ses, means+1.96*ses, alpha=0.3, color='red')
    rho_hw, p_hw = stats.spearmanr(x_hw, y_trig)
    ax.set_xlabel(f'{highway_col}', fontsize=12)
    ax.set_ylabel('Mean Trigger SOC (%)', fontsize=12)
    ax.set_title(f'(c) {highway_col} → Trigger SOC (ρ={rho_hw:.3f})', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
else:
    ax.text(0.5, 0.5, f'{highway_col} has too little variance',
            ha='center', va='center', transform=ax.transAxes)

# (d) 所有基因类型 vs Trigger SOC
ax = axes[1, 1]
for col in ratio_cols:
    label = col.replace('ratio_', '').replace('_', ' ').title()
    color = cluster_colors_map.get(col, 'grey')

    df_temp = df[[col, 'charge_trigger_soc']].dropna()
    q99 = df_temp[col].quantile(0.99)
    if q99 < 0.01:
        continue
    bins = np.linspace(0, q99, 11)
    c, m = [], []
    for i in range(len(bins)-1):
        mask = (df_temp[col] >= bins[i]) & (df_temp[col] < bins[i+1])
        if mask.sum() > 20:
            c.append((bins[i]+bins[i+1])/2)
            m.append(df_temp.loc[mask, 'charge_trigger_soc'].mean())
    if len(c) > 2:
        ax.plot(c, m, '-o', color=color, linewidth=2, markersize=5, label=label)

ax.set_xlabel('Segment Type Ratio', fontsize=12)
ax.set_ylabel('Mean Trigger SOC (%)', fontsize=12)
ax.set_title('(d) All Trip Genes → Trigger SOC', fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.suptitle(f'Section 3.3.2: Trip Gene Composition → Trigger SOC\n'
             f'(CV R²={cv_r2.mean():.3f}, MAE={cv_mae.mean():.2f})',
             fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(FIGURE_DIR, 'fig_332_gene_analysis.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(FIGURE_DIR, 'fig_332_gene_analysis.pdf'), bbox_inches='tight')
plt.close(fig)
print("   ✅ fig_332_gene_analysis.png/pdf")

# --- Fig 3: SHAP Dependence for soc_drop ---
if 'soc_drop' in feature_cols:
    fig_dep, ax_dep = plt.subplots(figsize=(8, 6))
    soc_idx = feature_cols.index('soc_drop')
    shap.dependence_plot(soc_idx, shap_values, X_shap,
                         feature_names=feature_cols, ax=ax_dep, show=False)
    ax_dep.set_title('SHAP Dependence: soc_drop → Trigger SOC', fontweight='bold')
    plt.tight_layout()
    fig_dep.savefig(os.path.join(FIGURE_DIR, 'fig_332_shap_dep_soc_drop.png'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig_dep)
    print("   ✅ fig_332_shap_dep_soc_drop.png")

# ============================================================
# 6. 保存结果
# ============================================================
results = {
    'n_trips': len(df_model),
    'cv_r2': float(cv_r2.mean()),
    'cv_r2_std': float(cv_r2.std()),
    'cv_mae': float(cv_mae.mean()),
    'cv_mae_std': float(cv_mae.std()),
    'ratio_columns': ratio_cols,
    'highway_column': highway_col,
    'feature_importance': {feature_cols[i]: float(abs_shap[i])
                          for i in range(len(feature_cols))},
    'correlations': {},
}

for col in gene_cols + accum_cols:
    if col in df.columns:
        valid = df[[col, target_col]].dropna()
        rho, p = stats.spearmanr(valid[col], valid[target_col])
        results['correlations'][col] = {'rho': float(rho), 'p': float(p)}

with open(os.path.join(OUTPUT_DIR, 'step_F_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n💾 Saved: step_F_results.json")
print(f"\n✅ Step F Complete!")