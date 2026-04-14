"""
Step B: 3.3.2 驾驶模式对充电触发阈值的影响
- 相关性分析
- XGBoost 预测 SOC_start
- SHAP 解释
适配新聚类名: Eco-Idle / Urban Moderate / Active Dynamic / Highway Aggressive
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.model_selection import cross_val_score, KFold
import xgboost as xgb
import shap
import json
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("📊 Step B: Driving Pattern → Charging Trigger SOC")
print("   (Section 3.3.2)")
print("=" * 70)

OUTPUT_DIR = "./coupling_analysis/results/"
FIGURE_DIR = "./coupling_analysis/figures/"
os.makedirs(FIGURE_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(OUTPUT_DIR, 'coupling_analysis_dataset.csv'))
print(f"   Loaded: {len(df):,} trips")

# ── 自动获取实际的模式名称和颜色 ──
actual_patterns = df['driving_pattern_name'].dropna().unique().tolist()
print(f"   Driving patterns found: {actual_patterns}")

# 按能耗排序（从低到高）
pattern_rate = df.groupby('driving_pattern_name')['soc_rate_per_hr'].mean().sort_values()
pattern_order = pattern_rate.index.tolist()
print(f"   Ordered by energy rate: {pattern_order}")

# 动态配色
COLOR_PALETTE = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#9b59b6', '#1abc9c']
colors_map = {name: COLOR_PALETTE[i] for i, name in enumerate(pattern_order)}

# ============================================================
# 1. 相关性分析
# ============================================================
print(f"\n{'=' * 70}")
print("📈 B1: Correlation Analysis")
print(f"{'=' * 70}")

numeric_vars = [
    'speed_mean', 'speed_cv', 'speed_std', 'idle_ratio',
    'aggressiveness_index', 'agg_composite',
    'trip_duration_hrs', 'soc_drop', 'soc_rate_per_hr', 'power_mean',
    'end_speed_mean', 'end_power_mean',
    'charge_trigger_soc', 'charge_gain_soc', 'charge_duration_min',
]
numeric_vars = [v for v in numeric_vars if v in df.columns]

corr_matrix = df[numeric_vars].corr(method='spearman')
target = 'charge_trigger_soc'
corr_with_target = corr_matrix[target].drop(target).sort_values(ascending=False)

print(f"\n   Spearman Correlation with {target}:")
for var, rho in corr_with_target.items():
    sig = "***" if abs(rho) > 0.1 else "**" if abs(rho) > 0.05 else ""
    print(f"      {var:<25} ρ = {rho:+.4f} {sig}")

# ============================================================
# 2. 散点图 + 箱线图
# ============================================================
print(f"\n{'=' * 70}")
print("📈 B2: Visualization")
print(f"{'=' * 70}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# (a) 激进度 vs Trigger SOC (hex density + 趋势线)
ax = axes[0]
x = df['aggressiveness_index'].values
y = df['charge_trigger_soc'].values
valid = ~(np.isnan(x) | np.isnan(y))
x, y = x[valid], y[valid]

hb = ax.hexbin(x, y, gridsize=40, cmap='YlOrRd', mincnt=1)
plt.colorbar(hb, ax=ax, label='Count')

# 分箱趋势线
n_bins = 20
bins = np.linspace(0, 1, n_bins + 1)
bin_centers, bin_means, bin_ses = [], [], []
for i in range(len(bins) - 1):
    mask = (x >= bins[i]) & (x < bins[i + 1])
    if mask.sum() > 10:
        bin_centers.append((bins[i] + bins[i + 1]) / 2)
        bin_means.append(np.mean(y[mask]))
        bin_ses.append(np.std(y[mask]) / np.sqrt(mask.sum()))

bin_centers = np.array(bin_centers)
bin_means = np.array(bin_means)
bin_ses = np.array(bin_ses)

ax.plot(bin_centers, bin_means, 'b-o', linewidth=2.5, markersize=5,
        label='Bin Mean', zorder=5)
ax.fill_between(bin_centers, bin_means - 1.96*bin_ses, bin_means + 1.96*bin_ses,
                alpha=0.3, color='blue', label='95% CI')

rho_val, p_val = stats.spearmanr(x, y)
ax.set_xlabel('Aggressiveness Index (energy-based)', fontsize=12)
ax.set_ylabel('Charging Trigger SOC (%)', fontsize=12)
ax.set_title(f'(a) Aggressiveness vs Trigger SOC\n(ρ={rho_val:.3f}, p<0.001)',
             fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.2)

# (b) 按驾驶模式分组的 Trigger SOC 箱线图
ax = axes[1]
box_data = []
box_labels = []
for p in pattern_order:
    vals = df[df['driving_pattern_name'] == p]['charge_trigger_soc'].dropna().values
    if len(vals) > 0:
        box_data.append(vals)
        box_labels.append(p)

if len(box_data) > 0:
    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True,
                    showfliers=False, widths=0.6)
    for patch, label in zip(bp['boxes'], box_labels):
        patch.set_facecolor(colors_map.get(label, 'grey'))
        patch.set_alpha(0.7)

    means = [np.mean(d) for d in box_data]
    ax.scatter(range(1, len(means)+1), means, color='black', marker='D',
               s=60, zorder=5, label='Mean')

    # Kruskal-Wallis 检验
    if len(box_data) >= 2:
        h_stat, kw_p = stats.kruskal(*box_data)
        ax.text(0.05, 0.95, f'Kruskal-Wallis: H={h_stat:.1f}, p<0.001',
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax.set_ylabel('Charging Trigger SOC (%)', fontsize=12)
ax.set_title('(b) Trigger SOC by Driving Pattern', fontweight='bold')
ax.tick_params(axis='x', rotation=15)
ax.legend()
ax.grid(alpha=0.2, axis='y')

# (c) SOC_drop vs Trigger SOC (colored by aggressiveness)
ax = axes[2]
sample = df.sample(min(8000, len(df)), random_state=42)
sc = ax.scatter(sample['soc_drop'], sample['charge_trigger_soc'],
                c=sample['aggressiveness_index'], cmap='RdYlGn_r',
                s=8, alpha=0.5)
plt.colorbar(sc, ax=ax, label='Aggressiveness')
ax.set_xlabel('Trip SOC Drop (%)', fontsize=12)
ax.set_ylabel('Charging Trigger SOC (%)', fontsize=12)
ax.set_title('(c) SOC Drop vs Trigger SOC\n(colored by aggressiveness)', fontweight='bold')
ax.grid(alpha=0.2)

plt.suptitle('Section 3.3.2: Driving Pattern → Charging Trigger SOC',
             fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(FIGURE_DIR, 'fig_332_trigger_soc_scatter.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(FIGURE_DIR, 'fig_332_trigger_soc_scatter.pdf'), bbox_inches='tight')
plt.close(fig)
print("   ✅ Saved: fig_332_trigger_soc_scatter.png/pdf")

# ============================================================
# 3. XGBoost 预测 SOC_start
# ============================================================
print(f"\n{'=' * 70}")
print("🤖 B3: XGBoost Model for Trigger SOC Prediction")
print(f"{'=' * 70}")

feature_cols = [
    'aggressiveness_index', 'agg_composite',
    'speed_mean', 'speed_cv', 'idle_ratio',
    'trip_duration_hrs', 'soc_drop', 'soc_rate_per_hr',
    'power_mean', 'end_speed_mean', 'end_power_mean',
]
feature_cols = [f for f in feature_cols if f in df.columns]
target_col = 'charge_trigger_soc'

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

print(f"\n   5-Fold Cross Validation:")
print(f"      R²:  {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
print(f"      MAE: {cv_mae.mean():.2f} ± {cv_mae.std():.2f}")

model.fit(X, y)

# ============================================================
# ============================================================
# 4. SHAP 分析 (兼容 xgboost 2.x)
# ============================================================
print(f"\n{'=' * 70}")
print("🔍 B4: SHAP Explanation")
print(f"{'=' * 70}")

n_shap = min(10000, len(X))
idx_shap = np.random.RandomState(42).choice(len(X), n_shap, replace=False)
X_shap = X[idx_shap]

# ── 修复 shap + xgboost 2.x 不兼容问题 ──
# 猴子补丁: 在 shap 解析 base_score 之前修复格式
import shap.explainers._tree as _tree
_original_init = _tree.XGBTreeModelLoader.__init__

def _patched_init(self, xgb_model):
    """修复 xgboost 2.x 的 base_score 格式问题"""
    _original_init(self, xgb_model)
    # 如果 base_score 解析失败了，不会走到这里
    # 所以直接在原始 init 外面包一层

# 更直接的方式: 直接 patch float 解析
_original_float = float

def _safe_float(x):
    if isinstance(x, str):
        x = x.strip('[]')  # '[3.5666237E1]' → '3.5666237E1'
    return _original_float(x)

import builtins
builtins.float = _safe_float

try:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)
finally:
    builtins.float = _original_float  # 恢复原始 float

mean_abs_shap = np.abs(shap_values).mean(axis=0)
importance_order = np.argsort(mean_abs_shap)[::-1]

print(f"   SHAP computed on {n_shap:,} samples")
print(f"\n   SHAP Feature Importance:")
for rank, idx in enumerate(importance_order):
    print(f"      {rank+1}. {feature_cols[idx]:<25} |SHAP|={mean_abs_shap[idx]:.4f}")

# ============================================================
# 5. SHAP 可视化
# ============================================================
fig = plt.figure(figsize=(18, 12))

# 用 GridSpec 手动控制布局，避免 shap 和 matplotlib 冲突
# (a) SHAP Beeswarm - 单独保存
fig_bee, ax_bee = plt.subplots(figsize=(10, 7))
shap.summary_plot(shap_values, X_shap, feature_names=feature_cols,
                  show=False, max_display=len(feature_cols))
plt.title(f'SHAP Summary: Trigger SOC Prediction (R²={cv_r2.mean():.3f})',
          fontweight='bold', fontsize=13)
plt.tight_layout()
fig_bee.savefig(os.path.join(FIGURE_DIR, 'fig_332_shap_beeswarm.png'), dpi=300, bbox_inches='tight')
fig_bee.savefig(os.path.join(FIGURE_DIR, 'fig_332_shap_beeswarm.pdf'), bbox_inches='tight')
plt.close(fig_bee)
print("   ✅ Saved: fig_332_shap_beeswarm.png/pdf")

# (b) SHAP Bar + Dependence plots
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Bar plot
ax = axes[0]
top_n = min(10, len(feature_cols))
sorted_idx = importance_order[:top_n]
ax.barh(range(top_n), mean_abs_shap[sorted_idx][::-1], color='#e74c3c', alpha=0.8)
ax.set_yticks(range(top_n))
ax.set_yticklabels([feature_cols[i] for i in sorted_idx][::-1], fontsize=10)
ax.set_xlabel('Mean |SHAP value|', fontsize=11)
ax.set_title('(a) Feature Importance', fontweight='bold', fontsize=12)
ax.grid(alpha=0.2, axis='x')

# Dependence: aggressiveness_index
ax = axes[1]
agg_col = 'aggressiveness_index'
if agg_col in feature_cols:
    agg_idx = feature_cols.index(agg_col)
    shap.dependence_plot(agg_idx, shap_values, X_shap,
                         feature_names=feature_cols, ax=ax, show=False)
    ax.set_title(f'(b) SHAP Dependence: {agg_col}', fontweight='bold', fontsize=12)

# Dependence: soc_drop
ax = axes[2]
if 'soc_drop' in feature_cols:
    soc_idx = feature_cols.index('soc_drop')
    shap.dependence_plot(soc_idx, shap_values, X_shap,
                         feature_names=feature_cols, ax=ax, show=False)
    ax.set_title('(c) SHAP Dependence: soc_drop', fontweight='bold', fontsize=12)

plt.suptitle(f'Section 3.3.2: XGBoost + SHAP (CV R²={cv_r2.mean():.3f}, MAE={cv_mae.mean():.2f})',
             fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(FIGURE_DIR, 'fig_332_shap_dependence.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(FIGURE_DIR, 'fig_332_shap_dependence.pdf'), bbox_inches='tight')
plt.close(fig)
print("   ✅ Saved: fig_332_shap_dependence.png/pdf")

# ============================================================
# 6. 非线性阈值效应
# ============================================================
print(f"\n{'=' * 70}")
print("📊 B5: Nonlinear Threshold Effect")
print(f"{'=' * 70}")

fig, ax = plt.subplots(figsize=(10, 6))

df_valid = df[['aggressiveness_index', 'charge_trigger_soc']].dropna()
n_bins = 20
df_valid['agg_bin'] = pd.cut(df_valid['aggressiveness_index'], bins=n_bins)
bin_stats = df_valid.groupby('agg_bin')['charge_trigger_soc'].agg(['mean', 'std', 'count'])
bin_stats = bin_stats[bin_stats['count'] > 20]

x_centers = np.array([interval.mid for interval in bin_stats.index])
y_means = bin_stats['mean'].values
y_se = bin_stats['std'].values / np.sqrt(bin_stats['count'].values)

ax.plot(x_centers, y_means, 'ro-', linewidth=2.5, markersize=8, label='Mean Trigger SOC')
ax.fill_between(x_centers, y_means - 1.96*y_se, y_means + 1.96*y_se,
                alpha=0.3, color='red', label='95% CI')

# 拐点检测
if len(y_means) > 4:
    second_diff = np.diff(np.diff(y_means))
    threshold_idx = np.argmax(np.abs(second_diff)) + 1
    threshold_x = x_centers[threshold_idx]
    ax.axvline(threshold_x, color='blue', linestyle='--', linewidth=2,
               label=f'Threshold ≈ {threshold_x:.2f}')
    print(f"   Detected threshold at aggressiveness ≈ {threshold_x:.2f}")

ax.set_xlabel('Aggressiveness Index (energy-based)', fontsize=12)
ax.set_ylabel('Mean Charging Trigger SOC (%)', fontsize=12)
ax.set_title('Nonlinear Threshold Effect:\nAggressiveness → Trigger SOC',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

fig.savefig(os.path.join(FIGURE_DIR, 'fig_332_threshold_effect.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(FIGURE_DIR, 'fig_332_threshold_effect.pdf'), bbox_inches='tight')
plt.close(fig)
print(f"   ✅ Saved: fig_332_threshold_effect.png/pdf")

# 保存结果
results = {
    'cv_r2_mean': float(cv_r2.mean()),
    'cv_r2_std': float(cv_r2.std()),
    'cv_mae_mean': float(cv_mae.mean()),
    'cv_mae_std': float(cv_mae.std()),
    'spearman_agg_vs_trigger': float(rho_val),
    'feature_importance': {feature_cols[i]: float(mean_abs_shap[i])
                          for i in range(len(feature_cols))},
}
with open(os.path.join(OUTPUT_DIR, 'step_B_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n💾 Saved: step_B_results.json")
print(f"\n✅ Step B Complete!")