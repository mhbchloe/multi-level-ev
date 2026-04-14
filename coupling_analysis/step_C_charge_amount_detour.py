"""
Step C: 3.3.3 对充电量与空间绕行的影响
- 驾驶模式 → SOC_gain
- 驾驶模式 → 绕行距离 (能耗偏离度代理)
- XGBoost + SHAP
适配: xgboost 2.x + shap 兼容补丁
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
import builtins
import json
import os
import warnings
warnings.filterwarnings('ignore')

# ── shap + xgboost 2.x 兼容补丁 ──
_original_float = builtins.float

def _safe_float(x):
    if isinstance(x, str):
        x = x.strip('[]')
    return _original_float(x)

def shap_tree_explainer_safe(model):
    """安全创建 TreeExplainer，兼容 xgboost 2.x"""
    builtins.float = _safe_float
    try:
        explainer = shap.TreeExplainer(model)
    finally:
        builtins.float = _original_float
    return explainer

print("=" * 70)
print("📊 Step C: Driving Pattern → Charge Amount & Detour")
print("   (Section 3.3.3)")
print("=" * 70)

OUTPUT_DIR = "./coupling_analysis/results/"
FIGURE_DIR = "./coupling_analysis/figures/"
os.makedirs(FIGURE_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(OUTPUT_DIR, 'coupling_analysis_dataset.csv'))
print(f"   Loaded: {len(df):,} trips")

# ── 自动获取模式名称 ──
pattern_rate = df.groupby('driving_pattern_name')['soc_rate_per_hr'].mean().sort_values()
pattern_order = pattern_rate.index.tolist()
print(f"   Patterns (by energy): {pattern_order}")

COLOR_PALETTE = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#9b59b6', '#1abc9c']
colors_map = {name: COLOR_PALETTE[i] for i, name in enumerate(pattern_order)}

# ============================================================
# 1. 驾驶模式 → SOC_gain 分析
# ============================================================
print(f"\n{'=' * 70}")
print("⚡ C1: Driving Pattern → Charging Amount (SOC_gain)")
print(f"{'=' * 70}")

rho_gain, p_gain = stats.spearmanr(
    df['aggressiveness_index'].dropna(),
    df.loc[df['aggressiveness_index'].notna(), 'charge_gain_soc']
)
print(f"   Spearman: ρ={rho_gain:.4f}, p={'<0.001' if p_gain < 0.001 else f'{p_gain:.4f}'}")

# XGBoost for SOC_gain
feature_cols = [
    'aggressiveness_index', 'agg_composite',
    'speed_mean', 'speed_cv', 'idle_ratio',
    'trip_duration_hrs', 'soc_drop', 'soc_rate_per_hr',
    'power_mean', 'end_speed_mean', 'end_power_mean',
    'charge_trigger_soc',
]
feature_cols = [f for f in feature_cols if f in df.columns]

target_col = 'charge_gain_soc'
df_model = df[feature_cols + [target_col]].dropna()
X = df_model[feature_cols].values
y = df_model[target_col].values

print(f"   Features ({len(feature_cols)}): {feature_cols}")
print(f"   Samples: {len(X):,}")

model_gain = xgb.XGBRegressor(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2_gain = cross_val_score(model_gain, X, y, cv=kf, scoring='r2')
cv_mae_gain = -cross_val_score(model_gain, X, y, cv=kf, scoring='neg_mean_absolute_error')
print(f"   XGBoost CV R²: {cv_r2_gain.mean():.4f} ± {cv_r2_gain.std():.4f}")
print(f"   XGBoost CV MAE: {cv_mae_gain.mean():.2f} ± {cv_mae_gain.std():.2f}")

model_gain.fit(X, y)

# SHAP (使用兼容补丁)
n_shap = min(10000, len(X))
idx_shap = np.random.RandomState(42).choice(len(X), n_shap, replace=False)
X_shap = X[idx_shap]

print("   Computing SHAP values...")
explainer_gain = shap_tree_explainer_safe(model_gain)
shap_gain = explainer_gain.shap_values(X_shap)

abs_shap_gain = np.abs(shap_gain).mean(axis=0)
importance_gain = np.argsort(abs_shap_gain)[::-1]

print(f"   SHAP computed on {n_shap:,} samples")
print(f"\n   Top SHAP features for SOC_gain:")
for rank, i in enumerate(importance_gain[:8]):
    print(f"      {rank+1}. {feature_cols[i]:<25} |SHAP|={abs_shap_gain[i]:.4f}")

# ============================================================
# 2. 绕行距离近似计算 (Energy Deviation Index)
# ============================================================
print(f"\n{'=' * 70}")
print("🗺️ C2: Detour Distance Proxy (Energy Deviation)")
print(f"{'=' * 70}")

# 能耗偏离度: 实际能耗 / 同速度段的中位数能耗
# > 1 表示比同速度段的中位数多消耗了能量（可能绕路/激进）
df_valid = df[['speed_mean', 'soc_rate_per_hr', 'trip_duration_hrs',
               'aggressiveness_index', 'charge_trigger_soc',
               'driving_pattern_name']].dropna().copy()

n_speed_bins = 10
df_valid['speed_bin'] = pd.qcut(df_valid['speed_mean'], n_speed_bins, duplicates='drop')

# ── 关键修复: .astype(float) 避免 Categorical 类型问题 ──
baseline = df_valid.groupby('speed_bin')['soc_rate_per_hr'].median()
df_valid['baseline_rate'] = df_valid['speed_bin'].map(baseline).astype(float)

df_valid['energy_deviation'] = (
    df_valid['soc_rate_per_hr'] / df_valid['baseline_rate'].clip(lower=0.01)
)
df_valid['energy_deviation'] = df_valid['energy_deviation'].clip(0.1, 10)

print(f"   Samples: {len(df_valid):,}")
print(f"   Energy Deviation: mean={df_valid['energy_deviation'].mean():.3f}, "
      f"std={df_valid['energy_deviation'].std():.3f}")
print(f"   > 1 means higher-than-baseline consumption (potential detour/aggressive)")

# 估算行驶距离 (km) 并计算距离偏离度
df['est_distance_km'] = df['speed_mean'] * df['trip_duration_hrs']

df_dist = df[['speed_mean', 'est_distance_km', 'aggressiveness_index',
              'charge_trigger_soc', 'trip_duration_hrs']].dropna().copy()
df_dist['speed_bin'] = pd.qcut(df_dist['speed_mean'], n_speed_bins, duplicates='drop')
baseline_dist = df_dist.groupby('speed_bin')['est_distance_km'].median()
df_dist['baseline_dist'] = df_dist['speed_bin'].map(baseline_dist).astype(float)
df_dist['detour_ratio'] = (
    df_dist['est_distance_km'] / df_dist['baseline_dist'].clip(lower=0.1)
)
df_dist['detour_ratio'] = df_dist['detour_ratio'].clip(0.1, 10)

print(f"   Detour Ratio: mean={df_dist['detour_ratio'].mean():.3f}, "
      f"std={df_dist['detour_ratio'].std():.3f}")

# 合并回主表
df['energy_deviation'] = np.nan
df.loc[df_valid.index, 'energy_deviation'] = df_valid['energy_deviation'].values

# 相关性
rho_dev, p_dev = stats.spearmanr(
    df_valid['aggressiveness_index'], df_valid['energy_deviation']
)
rho_trigger_dev, p_trigger_dev = stats.spearmanr(
    df_valid['charge_trigger_soc'], df_valid['energy_deviation']
)
print(f"\n   Correlations:")
print(f"      Aggressiveness → Energy Deviation: ρ={rho_dev:.4f}")
print(f"      Trigger SOC → Energy Deviation:    ρ={rho_trigger_dev:.4f}")

# ============================================================
# 3. XGBoost + SHAP for Energy Deviation
# ============================================================
print(f"\n{'=' * 70}")
print("🤖 C3: XGBoost for Energy Deviation")
print(f"{'=' * 70}")

dev_feature_cols = [
    'aggressiveness_index', 'agg_composite',
    'speed_mean', 'speed_cv', 'idle_ratio',
    'trip_duration_hrs', 'soc_drop', 'charge_trigger_soc',
    'power_mean', 'end_speed_mean',
]
dev_feature_cols = [f for f in dev_feature_cols if f in df_valid.columns]

df_dev_model = df_valid[dev_feature_cols + ['energy_deviation']].dropna()
X_dev = df_dev_model[dev_feature_cols].values
y_dev = df_dev_model['energy_deviation'].values

print(f"   Features: {dev_feature_cols}")
print(f"   Samples: {len(X_dev):,}")

model_dev = xgb.XGBRegressor(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
)

cv_r2_dev = cross_val_score(model_dev, X_dev, y_dev, cv=kf, scoring='r2')
print(f"   XGBoost CV R²: {cv_r2_dev.mean():.4f} ± {cv_r2_dev.std():.4f}")

model_dev.fit(X_dev, y_dev)

n_shap_dev = min(10000, len(X_dev))
idx_dev = np.random.RandomState(42).choice(len(X_dev), n_shap_dev, replace=False)
X_shap_dev = X_dev[idx_dev]

print("   Computing SHAP values...")
explainer_dev = shap_tree_explainer_safe(model_dev)
shap_dev = explainer_dev.shap_values(X_shap_dev)

abs_shap_dev = np.abs(shap_dev).mean(axis=0)
importance_dev = np.argsort(abs_shap_dev)[::-1]

print(f"\n   Top SHAP features for Energy Deviation:")
for rank, i in enumerate(importance_dev[:8]):
    print(f"      {rank+1}. {dev_feature_cols[i]:<25} |SHAP|={abs_shap_dev[i]:.4f}")

# ============================================================
# 4. 可视化 (3.3.3 全部图)
# ============================================================
print(f"\n{'=' * 70}")
print("📈 C4: Generating Section 3.3.3 Figures")
print(f"{'=' * 70}")

fig = plt.figure(figsize=(20, 14))
gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.30)

# (a) 驾驶模式 vs SOC_gain 箱线图
ax = fig.add_subplot(gs[0, 0])
box_data_gain = []
box_labels_gain = []
for p in pattern_order:
    vals = df[df['driving_pattern_name'] == p]['charge_gain_soc'].dropna().values
    if len(vals) > 0:
        box_data_gain.append(vals)
        box_labels_gain.append(p)

if len(box_data_gain) > 0:
    bp = ax.boxplot(box_data_gain, labels=box_labels_gain, patch_artist=True,
                    showfliers=False, widths=0.6)
    for patch, label in zip(bp['boxes'], box_labels_gain):
        patch.set_facecolor(colors_map.get(label, 'grey'))
        patch.set_alpha(0.7)
    means = [np.mean(d) for d in box_data_gain]
    ax.scatter(range(1, len(means)+1), means, color='black', marker='D', s=60, zorder=5)
ax.set_ylabel('SOC Gain (%)')
ax.set_title('(a) Charging Amount by Pattern', fontweight='bold')
ax.tick_params(axis='x', rotation=15)
ax.grid(alpha=0.2, axis='y')

# (b) Aggressiveness vs SOC_gain 趋势线
ax = fig.add_subplot(gs[0, 1])
n_bins = 20
bins = np.linspace(0, 1, n_bins + 1)
df_temp = df[['aggressiveness_index', 'charge_gain_soc']].dropna()
centers, means_g, ses_g = [], [], []
for i in range(len(bins) - 1):
    mask = (df_temp['aggressiveness_index'] >= bins[i]) & (df_temp['aggressiveness_index'] < bins[i+1])
    n = mask.sum()
    if n > 20:
        centers.append((bins[i] + bins[i+1]) / 2)
        vals = df_temp.loc[mask, 'charge_gain_soc']
        means_g.append(vals.mean())
        ses_g.append(vals.std() / np.sqrt(n))

centers = np.array(centers)
means_g = np.array(means_g)
ses_g = np.array(ses_g)

ax.plot(centers, means_g, 'ro-', linewidth=2.5, markersize=7, label='Mean SOC Gain')
ax.fill_between(centers, means_g - 1.96*ses_g, means_g + 1.96*ses_g,
                alpha=0.3, color='red', label='95% CI')
ax.set_xlabel('Aggressiveness Index')
ax.set_ylabel('Mean SOC Gain (%)')
ax.set_title(f'(b) Aggressiveness → Charging Amount\n(ρ={rho_gain:.3f})', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# (c) SHAP for SOC_gain
ax = fig.add_subplot(gs[0, 2])
top_n = min(8, len(feature_cols))
sorted_idx = importance_gain[:top_n]
ax.barh(range(top_n), abs_shap_gain[sorted_idx][::-1], color='#e74c3c', alpha=0.8)
ax.set_yticks(range(top_n))
ax.set_yticklabels([feature_cols[i] for i in sorted_idx][::-1], fontsize=10)
ax.set_xlabel('Mean |SHAP|')
ax.set_title(f'(c) SHAP for SOC Gain\n(R²={cv_r2_gain.mean():.3f})', fontweight='bold')
ax.grid(alpha=0.2, axis='x')

# (d) Aggressiveness vs Energy Deviation
ax = fig.add_subplot(gs[1, 0])
df_plot = df_valid.sample(min(5000, len(df_valid)), random_state=42)
hb = ax.hexbin(df_plot['aggressiveness_index'], df_plot['energy_deviation'],
               gridsize=30, cmap='YlOrRd', mincnt=1)
plt.colorbar(hb, ax=ax, label='Count')
ax.axhline(1.0, color='blue', linestyle='--', linewidth=1.5, label='Baseline')
ax.set_xlabel('Aggressiveness Index')
ax.set_ylabel('Energy Deviation (actual/baseline)')
ax.set_title(f'(d) Aggressiveness → Energy Deviation\n(ρ={rho_dev:.3f})', fontweight='bold')
ax.legend()
ax.grid(alpha=0.2)

# (e) Trigger SOC vs Energy Deviation
ax = fig.add_subplot(gs[1, 1])
hb = ax.hexbin(df_valid['charge_trigger_soc'], df_valid['energy_deviation'],
               gridsize=30, cmap='YlOrRd', mincnt=1)
plt.colorbar(hb, ax=ax, label='Count')
ax.axhline(1.0, color='blue', linestyle='--', linewidth=1.5)
ax.set_xlabel('Trigger SOC (%)')
ax.set_ylabel('Energy Deviation')
ax.set_title(f'(e) Trigger SOC → Deviation\n(ρ={rho_trigger_dev:.3f})', fontweight='bold')
ax.grid(alpha=0.2)

# (f) 驾驶模式 vs Energy Deviation 箱线图
ax = fig.add_subplot(gs[1, 2])
box_data_dev = []
box_labels_dev = []
for p in pattern_order:
    vals = df_valid[df_valid['driving_pattern_name'] == p]['energy_deviation'].dropna().values
    if len(vals) > 0:
        box_data_dev.append(vals)
        box_labels_dev.append(p)

if len(box_data_dev) > 0:
    bp = ax.boxplot(box_data_dev, labels=box_labels_dev, patch_artist=True,
                    showfliers=False, widths=0.6)
    for patch, label in zip(bp['boxes'], box_labels_dev):
        patch.set_facecolor(colors_map.get(label, 'grey'))
        patch.set_alpha(0.7)
ax.axhline(1.0, color='blue', linestyle='--', alpha=0.5, label='Baseline')
ax.set_ylabel('Energy Deviation')
ax.set_title('(f) Energy Deviation by Pattern', fontweight='bold')
ax.tick_params(axis='x', rotation=15)
ax.legend()
ax.grid(alpha=0.2, axis='y')

plt.suptitle('Section 3.3.3: Impact on Charging Amount & Spatial Deviation',
             fontsize=15, fontweight='bold')
fig.savefig(os.path.join(FIGURE_DIR, 'fig_333_charge_detour.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(FIGURE_DIR, 'fig_333_charge_detour.pdf'), bbox_inches='tight')
plt.close('all')
print(f"   ✅ Saved: fig_333_charge_detour.png/pdf")

# ── SHAP Beeswarm for SOC_gain (单独保存) ──
fig_bee, _ = plt.subplots(figsize=(10, 7))
shap.summary_plot(shap_gain, X_shap, feature_names=feature_cols,
                  show=False, max_display=len(feature_cols))
plt.title(f'SHAP Summary: SOC Gain Prediction (R²={cv_r2_gain.mean():.3f})',
          fontweight='bold', fontsize=13)
plt.tight_layout()
fig_bee.savefig(os.path.join(FIGURE_DIR, 'fig_333_shap_soc_gain_beeswarm.png'),
                dpi=300, bbox_inches='tight')
plt.close(fig_bee)
print(f"   ✅ Saved: fig_333_shap_soc_gain_beeswarm.png")

# ── SHAP Beeswarm for Energy Deviation (单独保存) ──
fig_bee2, _ = plt.subplots(figsize=(10, 7))
shap.summary_plot(shap_dev, X_shap_dev, feature_names=dev_feature_cols,
                  show=False, max_display=len(dev_feature_cols))
plt.title(f'SHAP Summary: Energy Deviation (R²={cv_r2_dev.mean():.3f})',
          fontweight='bold', fontsize=13)
plt.tight_layout()
fig_bee2.savefig(os.path.join(FIGURE_DIR, 'fig_333_shap_energy_deviation_beeswarm.png'),
                 dpi=300, bbox_inches='tight')
plt.close(fig_bee2)
print(f"   ✅ Saved: fig_333_shap_energy_deviation_beeswarm.png")

# ── SHAP Dependence plots ──
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Dependence: aggressiveness → SOC_gain
ax = axes[0]
if 'aggressiveness_index' in feature_cols:
    agg_idx = feature_cols.index('aggressiveness_index')
    shap.dependence_plot(agg_idx, shap_gain, X_shap,
                         feature_names=feature_cols, ax=ax, show=False)
    ax.set_title('(a) Aggressiveness → SOC Gain SHAP', fontweight='bold')

# Dependence: charge_trigger_soc → SOC_gain
ax = axes[1]
if 'charge_trigger_soc' in feature_cols:
    trig_idx = feature_cols.index('charge_trigger_soc')
    shap.dependence_plot(trig_idx, shap_gain, X_shap,
                         feature_names=feature_cols, ax=ax, show=False)
    ax.set_title('(b) Trigger SOC → SOC Gain SHAP', fontweight='bold')

# Dependence: aggressiveness → Energy Deviation
ax = axes[2]
if 'aggressiveness_index' in dev_feature_cols:
    agg_dev_idx = dev_feature_cols.index('aggressiveness_index')
    shap.dependence_plot(agg_dev_idx, shap_dev, X_shap_dev,
                         feature_names=dev_feature_cols, ax=ax, show=False)
    ax.set_title('(c) Aggressiveness → Deviation SHAP', fontweight='bold')

plt.suptitle('Section 3.3.3: SHAP Dependence Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(FIGURE_DIR, 'fig_333_shap_dependence.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(FIGURE_DIR, 'fig_333_shap_dependence.pdf'), bbox_inches='tight')
plt.close(fig)
print(f"   ✅ Saved: fig_333_shap_dependence.png/pdf")

# ============================================================
# 5. 保存结果
# ============================================================
df.to_csv(os.path.join(OUTPUT_DIR, 'coupling_analysis_dataset.csv'), index=False)

results_c = {
    'soc_gain_spearman_with_aggressiveness': float(rho_gain),
    'soc_gain_xgb_r2': float(cv_r2_gain.mean()),
    'soc_gain_xgb_mae': float(cv_mae_gain.mean()),
    'energy_deviation_mean': float(df_valid['energy_deviation'].mean()),
    'energy_deviation_std': float(df_valid['energy_deviation'].std()),
    'energy_dev_xgb_r2': float(cv_r2_dev.mean()),
    'corr_aggressiveness_deviation': float(rho_dev),
    'corr_trigger_soc_deviation': float(rho_trigger_dev),
    'soc_gain_shap_top5': {feature_cols[i]: float(abs_shap_gain[i])
                           for i in importance_gain[:5]},
    'deviation_shap_top5': {dev_feature_cols[i]: float(abs_shap_dev[i])
                            for i in importance_dev[:5]},
}

with open(os.path.join(OUTPUT_DIR, 'step_C_results.json'), 'w') as f:
    json.dump(results_c, f, indent=2)

print(f"\n💾 Saved: step_C_results.json")
print(f"💾 Updated: coupling_analysis_dataset.csv")

# 打印图表清单
print(f"\n📊 All figures generated:")
for fn in sorted(os.listdir(FIGURE_DIR)):
    if fn.startswith('fig_333'):
        fp = os.path.join(FIGURE_DIR, fn)
        print(f"   {fn:<55} {os.path.getsize(fp)/1024:.0f} KB")

print(f"\n✅ Step C Complete!")