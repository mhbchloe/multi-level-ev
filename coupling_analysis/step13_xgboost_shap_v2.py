"""
Step 13 (v2): Feature Importance & Impact Analysis (No SHAP)
用传统方法分析特征重要性和驾驶模式对充电的影响
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['font.size'] = 11

print("="*70)
print("🧠 Step 13 (v2): Feature Importance & Impact Analysis")
print("="*70)

input_dir = "./coupling_analysis/results/"
df = pd.read_csv(os.path.join(input_dir, 'inter_charge_trips_v2.csv'))

print(f"\n📊 Dataset: {len(df):,} trips from {df['vehicle_id'].nunique():,} vehicles")

# ========== 1. 数据准备 ==========
print("\n📋 Data Preparation...")

features = [
    'ratio_moderate', 'ratio_conservative', 'ratio_aggressive', 'ratio_highway',
    'trip_avg_power', 'trip_avg_speed', 'trip_acc_std',
    'trip_total_soc_drop', 'trip_duration_hrs', 'end_stage_power'
]

X = df[features].fillna(0)
y = df['charge_trigger_soc']

valid_idx = (y >= 0) & (y <= 100)
X = X[valid_idx]
y = y[valid_idx]

print(f"   Features: {len(features)}")
print(f"   Samples: {len(X):,}")
print(f"   Target Mean: {y.mean():.2f}%")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========== 2. XGBoost 模型训练 ==========
print("\n🎯 Training XGBoost Model...")

model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    tree_method='hist'
)

model.fit(X_train, y_train, verbose=0)

y_pred_test = model.predict(X_test)
r2_test = r2_score(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"   ✅ Test R²: {r2_test:.4f}")
print(f"   ✅ Test MAE: {mae_test:.2f}%")
print(f"   ✅ Test RMSE: {rmse_test:.2f}%")

# ========== 3. 特征重要性分析 ==========
print("\n📊 Feature Importance Analysis...")

# 3.1 XGBoost 内置特征重要性
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🏆 Top Features (by XGBoost gain):")
for i, row in feature_importance.head(5).iterrows():
    print(f"   {row['feature']}: {row['importance']:.4f}")

# 3.2 绘制特征重要性柱状图
fig, ax = plt.subplots(figsize=(11, 7))
feature_importance_sorted = feature_importance.sort_values('importance', ascending=True)

colors_list = ['#FF6B6B' if 'aggressive' in f else '#4ECDC4' if 'ratio' in f else '#45B7D1' 
               for f in feature_importance_sorted['feature']]

ax.barh(feature_importance_sorted['feature'], feature_importance_sorted['importance'], 
        color=colors_list, edgecolor='black', linewidth=1.2)
ax.set_xlabel('XGBoost Feature Importance (Gain)', fontweight='bold', fontsize=12)
ax.set_title('Which Factors Most Strongly Determine Charging Trigger SOC?', 
             fontweight='bold', fontsize=13, pad=15)
ax.grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(os.path.join(input_dir, 'feature_importance_xgboost_v2.png'), dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Saved: feature_importance_xgboost_v2.png")

# ========== 4. 驾驶模式与充电决策的相关性 ==========
print("\n📈 Correlation Analysis: Driving Patterns vs Charging Trigger...")

# 计算各特征与目标变量的皮尔逊相关系数
correlations = []
for feat in features:
    corr, pvalue = stats.pearsonr(X[feat], y)
    correlations.append({
        'feature': feat,
        'correlation': corr,
        'p_value': pvalue,
        'significant': 'Yes' if pvalue < 0.05 else 'No'
    })

df_corr = pd.DataFrame(correlations).sort_values('correlation', ascending=False)

print("\n📊 Correlation with Charge Trigger SOC:")
print(df_corr.to_string(index=False))

# 3.3 绘制相关性柱状图
fig, ax = plt.subplots(figsize=(11, 7))
df_corr_sorted = df_corr.sort_values('correlation', ascending=True)

colors_corr = ['#FF6B6B' if x > 0 else '#4ECDC4' for x in df_corr_sorted['correlation']]
ax.barh(df_corr_sorted['feature'], df_corr_sorted['correlation'], 
        color=colors_corr, edgecolor='black', linewidth=1.2)
ax.set_xlabel('Pearson Correlation with Charge Trigger SOC', fontweight='bold', fontsize=12)
ax.set_title('Correlation Analysis: Driving Patterns vs When to Charge\n(Positive = Earlier charging)', 
             fontweight='bold', fontsize=13, pad=15)
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax.grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(os.path.join(input_dir, 'correlation_analysis_v2.png'), dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Saved: correlation_analysis_v2.png")

# ========== 5. 激进片段比例的深入分析 ==========
print("\n⭐ Deep Dive: Aggressive Driving Ratio Impact...")

# 按激进片段比例分组
df['aggressive_bin'] = pd.cut(df['ratio_aggressive'], 
                               bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                               labels=['0-20%', '20-40%', '40-60%', '60-80%', '80-100%'])

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 左图：激进比例 vs 充电触发SOC
ax = axes[0]
agg_stats = df.groupby('aggressive_bin')['charge_trigger_soc'].agg(['mean', 'std', 'count'])
x_pos = np.arange(len(agg_stats))
ax.bar(x_pos, agg_stats['mean'], yerr=agg_stats['std'], capsize=5, 
       color='#FF6B6B', edgecolor='black', linewidth=1.5, alpha=0.7)
ax.set_xticks(x_pos)
ax.set_xticklabels(agg_stats.index)
ax.set_ylabel('Avg Charge Trigger SOC (%)', fontweight='bold', fontsize=12)
ax.set_xlabel('Aggressive Driving Ratio Range', fontweight='bold', fontsize=12)
ax.set_title('Impact of Aggressive Driving Ratio on Charging Trigger\n(Higher ratio → Earlier charging = Range Anxiety)', 
             fontweight='bold', fontsize=12, pad=15)
ax.grid(alpha=0.3, axis='y')

# 添加样本量标签
for i, (idx, row) in enumerate(agg_stats.iterrows()):
    ax.text(i, row['mean'] + row['std'] + 1, f"n={int(row['count'])}", 
            ha='center', fontsize=10, fontweight='bold')

# 右图：激进比例 vs SOC消耗
ax = axes[1]
drop_stats = df.groupby('aggressive_bin')['trip_total_soc_drop'].agg(['mean', 'std', 'count'])
ax.bar(x_pos, drop_stats['mean'], yerr=drop_stats['std'], capsize=5, 
       color='#4ECDC4', edgecolor='black', linewidth=1.5, alpha=0.7)
ax.set_xticks(x_pos)
ax.set_xticklabels(drop_stats.index)
ax.set_ylabel('Avg SOC Consumption per Trip (%)', fontweight='bold', fontsize=12)
ax.set_xlabel('Aggressive Driving Ratio Range', fontweight='bold', fontsize=12)
ax.set_title('Energy Consumption by Aggressive Driving Ratio\n(Higher ratio → More energy intensive)', 
             fontweight='bold', fontsize=12, pad=15)
ax.grid(alpha=0.3, axis='y')

for i, (idx, row) in enumerate(drop_stats.iterrows()):
    ax.text(i, row['mean'] + row['std'] + 1, f"n={int(row['count'])}", 
            ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(input_dir, 'aggressive_ratio_detailed_analysis_v2.png'), dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Saved: aggressive_ratio_detailed_analysis_v2.png")

# ========== 6. 关键统计发现 ==========
print(f"\n{'='*70}")
print("📋 KEY FINDINGS (3.3.2):")
print(f"{'='*70}")

# 激进比例的统计
agg_corr = df_corr[df_corr['feature'] == 'ratio_aggressive'].iloc[0]
print(f"\n⭐ Aggressive Driving Ratio:")
print(f"   Correlation with Trigger SOC: {agg_corr['correlation']:.4f}")
print(f"   P-value: {agg_corr['p_value']:.4e} (Significant: {agg_corr['significant']})")
print(f"   Rank by importance: #{(feature_importance['feature'] == 'ratio_aggressive').idxmax() + 1}/{len(features)}")

# 分组统计
print(f"\n📊 Behavior Pattern (by Aggressive Ratio):")
low_agg = df[df['ratio_aggressive'] < 0.2]
high_agg = df[df['ratio_aggressive'] >= 0.6]
print(f"\n   Low Aggression (0-20%):")
print(f"      Avg Trigger SOC: {low_agg['charge_trigger_soc'].mean():.1f}%")
print(f"      Avg SOC Drop: {low_agg['trip_total_soc_drop'].mean():.1f}%")
print(f"      Sample size: {len(low_agg):,}")

print(f"\n   High Aggression (60-100%):")
print(f"      Avg Trigger SOC: {high_agg['charge_trigger_soc'].mean():.1f}%")
print(f"      Avg SOC Drop: {high_agg['trip_total_soc_drop'].mean():.1f}%")
print(f"      Sample size: {len(high_agg):,}")

# T检验
t_stat, p_val = stats.ttest_ind(low_agg['charge_trigger_soc'], high_agg['charge_trigger_soc'])
print(f"\n   Statistical Test (t-test):")
print(f"      t-statistic: {t_stat:.4f}")
print(f"      p-value: {p_val:.4e}")
print(f"      Significant difference: {'YES' if p_val < 0.05 else 'NO'}")

print(f"\n{'='*70}")
print("✅ Step 13 Complete!")
print(f"{'='*70}")