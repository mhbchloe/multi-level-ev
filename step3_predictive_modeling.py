"""
Step 3: Predictive Modeling (Fixed for SHAP compatibility)
修复XGBoost与SHAP兼容性问题
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# SHAP导入（带兼容性处理）
try:
    import shap
    SHAP_AVAILABLE = True
    print("✅ SHAP version:", shap.__version__)
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️  SHAP not available")

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

print("="*70)
print("🤖 Step 3: Predictive Modeling (Fixed)")
print("="*70)

# ============ 1. Load and Prepare Data ============
print("\n📂 Loading data...")

df = pd.read_csv('./results/discharge_charging_pairs.csv')
print(f"✅ Loaded {len(df):,} samples")

# 特征选择
feature_cols = [
    'discharge_cluster',
    'discharge_speed_mean',
    'discharge_speed_std',
    'discharge_harsh_accel',
    'discharge_harsh_brake',
    'discharge_idle_ratio',
    'discharge_soc_drop',
    'discharge_soc_start',
    'discharge_energy_kwh',
    'discharge_efficiency_kwh_per_km',
    'discharge_duration',
    'discharge_distance',
    'vehicle_total_chargings',
    'vehicle_avg_soc_gain',
]

# 目标变量
target_vars = {
    'charging_soc_start': 'Charging Trigger SOC (%)',
    'charging_soc_gain': 'Charging Gain (%)',
    'charging_duration': 'Charging Duration (seconds)',
    'time_to_next_charging': 'Time to Next Charging (seconds)'
}

available_features = [col for col in feature_cols if col in df.columns]
print(f"✅ Selected {len(available_features)} features")

X = df[available_features].copy()
X = X.fillna(X.median())

# One-hot encode cluster
X_encoded = pd.get_dummies(X, columns=['discharge_cluster'], prefix='cluster')

print(f"✅ Feature matrix shape: {X_encoded.shape}")

# ============ 2. Model Training Function ============

def train_and_evaluate_models(X, y, target_name):
    """训练并评估模型"""
    
    print(f"\n{'='*70}")
    print(f"🎯 Target: {target_name}")
    print(f"{'='*70}")
    
    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    # ========== Linear Regression ==========
    print(f"\n📊 Linear Regression:")
    
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    y_pred_lr = lr_model.predict(X_test_scaled)
    
    lr_mse = mean_squared_error(y_test, y_pred_lr)
    lr_mae = mean_absolute_error(y_test, y_pred_lr)
    lr_r2 = r2_score(y_test, y_pred_lr)
    
    cv_scores = cross_val_score(lr_model, X_train_scaled, y_train, cv=5, scoring='r2')
    
    print(f"   MSE: {lr_mse:.4f}")
    print(f"   MAE: {lr_mae:.4f}")
    print(f"   R²: {lr_r2:.4f}")
    print(f"   CV R²: {cv_scores.mean():.4f}±{cv_scores.std():.4f}")
    
    results['Linear Regression'] = {
        'model': lr_model,
        'scaler': scaler,
        'y_pred': y_pred_lr,
        'mse': lr_mse,
        'mae': lr_mae,
        'r2': lr_r2,
        'cv_r2': cv_scores.mean()
    }
    
    # ========== Ridge ==========
    print(f"\n📊 Ridge Regression:")
    
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train_scaled, y_train)
    y_pred_ridge = ridge_model.predict(X_test_scaled)
    
    ridge_mse = mean_squared_error(y_test, y_pred_ridge)
    ridge_mae = mean_absolute_error(y_test, y_pred_ridge)
    ridge_r2 = r2_score(y_test, y_pred_ridge)
    
    print(f"   MSE: {ridge_mse:.4f}")
    print(f"   MAE: {ridge_mae:.4f}")
    print(f"   R²: {ridge_r2:.4f}")
    
    results['Ridge'] = {
        'model': ridge_model,
        'scaler': scaler,
        'y_pred': y_pred_ridge,
        'mse': ridge_mse,
        'mae': ridge_mae,
        'r2': ridge_r2
    }
    
    # ========== XGBoost ==========
    print(f"\n📊 XGBoost:")
    
    # 使用兼容的XGBoost配置
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        base_score=0.5  # 显式设置base_score为单个值
    )
    
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    
    xgb_mse = mean_squared_error(y_test, y_pred_xgb)
    xgb_mae = mean_absolute_error(y_test, y_pred_xgb)
    xgb_r2 = r2_score(y_test, y_pred_xgb)
    
    cv_scores_xgb = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='r2')
    
    print(f"   MSE: {xgb_mse:.4f}")
    print(f"   MAE: {xgb_mae:.4f}")
    print(f"   R²: {xgb_r2:.4f}")
    print(f"   CV R²: {cv_scores_xgb.mean():.4f}±{cv_scores_xgb.std():.4f}")
    
    # Feature importance (XGBoost内置)
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    results['XGBoost'] = {
        'model': xgb_model,
        'y_pred': y_pred_xgb,
        'mse': xgb_mse,
        'mae': xgb_mae,
        'r2': xgb_r2,
        'cv_r2': cv_scores_xgb.mean(),
        'feature_importance': feature_importance
    }
    
    results['X_test'] = X_test
    results['y_test'] = y_test
    results['X_train'] = X_train
    results['y_train'] = y_train
    
    return results

# ============ 3. Train Models ============
all_results = {}

for target_col, target_label in target_vars.items():
    if target_col not in df.columns:
        print(f"⚠️  {target_col} not found, skipping...")
        continue
    
    y = df[target_col].fillna(df[target_col].median())
    results = train_and_evaluate_models(X_encoded, y, target_label)
    all_results[target_col] = results

# ============ 4. SHAP Analysis (带错误处理) ============
print(f"\n{'='*70}")
print(f"📊 Feature Importance Analysis")
print(f"{'='*70}")

fig_shap = plt.figure(figsize=(20, 12))
gs_shap = fig_shap.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

for idx, (target_col, target_label) in enumerate(list(target_vars.items())[:4]):
    if target_col not in all_results:
        continue
    
    print(f"\n🔍 Feature importance for: {target_label}")
    
    xgb_model = all_results[target_col]['XGBoost']['model']
    X_test = all_results[target_col]['X_test']
    feature_importance = all_results[target_col]['XGBoost']['feature_importance']
    
    ax = fig_shap.add_subplot(gs_shap[idx // 2, idx % 2])
    
    if SHAP_AVAILABLE:
        try:
            # 尝试使用TreeExplainer
            explainer = shap.TreeExplainer(xgb_model)
            shap_values = explainer.shap_values(X_test)
            
            # SHAP summary plot
            shap.summary_plot(
                shap_values, 
                X_test, 
                plot_type="bar",
                show=False,
                max_display=10
            )
            
            ax.set_title(f'SHAP: {target_label}', fontsize=12, fontweight='bold')
            
            print(f"   ✅ SHAP analysis complete")
            
        except Exception as e:
            print(f"   ⚠️  SHAP TreeExplainer failed: {str(e)}")
            print(f"   → Using XGBoost built-in feature importance instead")
            
            # 降级到XGBoost内置重要性
            top_features = feature_importance.head(10)
            
            ax.barh(range(len(top_features)), top_features['importance'].values, 
                    color='#3498db', alpha=0.8)
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['feature'].values, fontsize=9)
            ax.set_xlabel('Feature Importance', fontsize=10, fontweight='bold')
            ax.set_title(f'XGBoost Feature Importance: {target_label}', 
                        fontsize=12, fontweight='bold')
            ax.grid(alpha=0.3, axis='x')
            ax.invert_yaxis()
    else:
        # 使用XGBoost内置重要性
        top_features = feature_importance.head(10)
        
        ax.barh(range(len(top_features)), top_features['importance'].values, 
                color='#3498db', alpha=0.8)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'].values, fontsize=9)
        ax.set_xlabel('Feature Importance', fontsize=10, fontweight='bold')
        ax.set_title(f'XGBoost Feature Importance: {target_label}', 
                    fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3, axis='x')
        ax.invert_yaxis()

plt.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('./results/feature_importance.png', dpi=300, bbox_inches='tight')
print(f"\n✅ Saved: feature_importance.png")

# ============ 5. Model Comparison Visualization ============
print(f"\n📈 Generating model comparison visualizations...")

fig_comp = plt.figure(figsize=(20, 14))
gs_comp = fig_comp.add_gridspec(3, 4, hspace=0.35, wspace=0.3)

colors_model = {'Linear Regression': '#3498db', 'Ridge': '#2ecc71', 'XGBoost': '#e74c3c'}

# === Row 1: Performance Comparison ===
for idx, (target_col, target_label) in enumerate(list(target_vars.items())[:4]):
    if target_col not in all_results:
        continue
    
    ax = fig_comp.add_subplot(gs_comp[0, idx])
    
    models = ['Linear Regression', 'Ridge', 'XGBoost']
    r2_scores = [all_results[target_col][m]['r2'] for m in models]
    
    bars = ax.bar(range(len(models)), r2_scores, 
                  color=[colors_model[m] for m in models], alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(['Linear', 'Ridge', 'XGBoost'], rotation=15, ha='right', fontsize=9)
    ax.set_ylabel('R² Score', fontsize=10, fontweight='bold')
    ax.set_title(f'{target_label}\nModel Comparison', fontsize=11, fontweight='bold')
    ax.set_ylim([0, max(r2_scores) * 1.2] if max(r2_scores) > 0 else [0, 1])
    ax.grid(alpha=0.3, axis='y')
    
    for bar, score in zip(bars, r2_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{score:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# === Row 2: Prediction vs Actual ===
for idx, (target_col, target_label) in enumerate(list(target_vars.items())[:4]):
    if target_col not in all_results:
        continue
    
    ax = fig_comp.add_subplot(gs_comp[1, idx])
    
    y_test = all_results[target_col]['y_test']
    y_pred = all_results[target_col]['XGBoost']['y_pred']
    
    ax.scatter(y_test, y_pred, alpha=0.3, s=20, color='#3498db')
    
    # 45度线
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
    
    ax.set_xlabel('Actual', fontsize=10, fontweight='bold')
    ax.set_ylabel('Predicted', fontsize=10, fontweight='bold')
    ax.set_title(f'{target_label}\nXGBoost Predictions', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    # R² annotation
    r2 = all_results[target_col]['XGBoost']['r2']
    ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
            fontsize=10, fontweight='bold', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# === Row 3: Residual Analysis ===
for idx, (target_col, target_label) in enumerate(list(target_vars.items())[:4]):
    if target_col not in all_results:
        continue
    
    ax = fig_comp.add_subplot(gs_comp[2, idx])
    
    y_test = all_results[target_col]['y_test']
    y_pred = all_results[target_col]['XGBoost']['y_pred']
    residuals = y_test - y_pred
    
    ax.scatter(y_pred, residuals, alpha=0.3, s=20, color='#e74c3c')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=2)
    
    # 添加残差分布
    ax2 = ax.twinx()
    ax2.hist(residuals, bins=30, alpha=0.3, color='green', orientation='horizontal')
    ax2.set_ylabel('Frequency', fontsize=9, color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    
    ax.set_xlabel('Predicted', fontsize=10, fontweight='bold')
    ax.set_ylabel('Residuals', fontsize=10, fontweight='bold')
    ax.set_title(f'{target_label}\nResidual Plot', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)

plt.suptitle('Predictive Modeling: Linear Regression vs XGBoost', 
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig('./results/model_comparison.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: model_comparison.png")

# ============ 6. Save Summary ============
summary_data = []

for target_col, target_label in target_vars.items():
    if target_col not in all_results:
        continue
    
    for model_name in ['Linear Regression', 'Ridge', 'XGBoost']:
        result = all_results[target_col][model_name]
        
        summary_data.append({
            'Target': target_label,
            'Model': model_name,
            'MSE': result['mse'],
            'MAE': result['mae'],
            'R²': result['r2'],
            'CV R²': result.get('cv_r2', np.nan)
        })

df_summary = pd.DataFrame(summary_data)
df_summary.to_csv('./results/model_performance_summary.csv', index=False, encoding='utf-8-sig')
print(f"💾 Saved: model_performance_summary.csv")

# 保存特征重要性
for target_col, target_label in target_vars.items():
    if target_col not in all_results:
        continue
    
    importance_df = all_results[target_col]['XGBoost']['feature_importance']
    filename = f"./results/feature_importance_{target_col}.csv"
    importance_df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"💾 Saved: {filename}")

print(f"\n{'='*70}")
print(f"✅ Step 3 Complete!")
print(f"{'='*70}")
print(f"\n📁 Generated files:")
print(f"   1. feature_importance.png")
print(f"   2. model_comparison.png")
print(f"   3. model_performance_summary.csv")
print(f"   4. feature_importance_*.csv (for each target)")
print(f"{'='*70}")