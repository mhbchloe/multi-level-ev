"""
Step 3.3: Charging-Mobility-Energy Coupling Analysis (带缓存)
支持断点续传和中间结果保存
"""

import numpy as np
import pandas as pd
import os
import json
import warnings
import pickle
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import shap
from tqdm import tqdm

warnings.filterwarnings('ignore')

rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 10
rcParams['figure.dpi'] = 150

print("=" * 80)
print("🔗 STEP 3.3: CHARGING-MOBILITY-ENERGY COUPLING (WITH CACHE)")
print("=" * 80)

# ============================================================
# 0. 配置和缓存路径
# ============================================================
CONFIG = {
    'segments_path': './coupling_analysis/results/segments_integrated_complete.csv',
    'trips_path': './coupling_analysis/results/inter_charge_trips.csv',
    'vehicles_path': './vehicle_clustering/results/vehicle_clustering_gmm_k4.csv',
    'save_dir': './coupling_analysis/results/coupling_analysis/',
    'figure_dir': './coupling_analysis/results/coupling_figures/',
    'cache_dir': './coupling_analysis/results/coupling_cache/',  # 缓存目录
    'seed': 42,
}

os.makedirs(CONFIG['save_dir'], exist_ok=True)
os.makedirs(CONFIG['figure_dir'], exist_ok=True)
os.makedirs(CONFIG['cache_dir'], exist_ok=True)

# 缓存文件路径
CACHE_FILES = {
    'trip_genes': os.path.join(CONFIG['cache_dir'], 'trip_genes_df.pkl'),
    'trips_enhanced': os.path.join(CONFIG['cache_dir'], 'trips_enhanced.pkl'),
    'df_model': os.path.join(CONFIG['cache_dir'], 'df_model.pkl'),
    'X_scaled': os.path.join(CONFIG['cache_dir'], 'X_scaled.pkl'),
    'y': os.path.join(CONFIG['cache_dir'], 'y.pkl'),
    'feature_cols': os.path.join(CONFIG['cache_dir'], 'feature_cols.json'),
    'model': os.path.join(CONFIG['cache_dir'], 'xgboost_model.pkl'),
}

VEHICLE_NAMES = {
    0: 'Long-Distance Highway (LDH)',
    1: 'Stationary/Occasional (SOC)',
    2: 'Urban Commuter (UCO)',
    3: 'Multi-purpose Mixed (MUM)',
}

# ============================================================
# 辅助函数：保存/加载缓存
# ============================================================
def save_cache(obj, cache_key):
    """保存对象到缓存"""
    path = CACHE_FILES[cache_key]
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"   ✓ Cached {cache_key}: {size_mb:.1f} MB")

def load_cache(cache_key):
    """从缓存加载对象"""
    path = CACHE_FILES[cache_key]
    if os.path.exists(path):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"   ✓ Loaded {cache_key} from cache: {size_mb:.1f} MB")
        return obj
    return None

def cache_exists(cache_key):
    """检查缓存是否存在"""
    return os.path.exists(CACHE_FILES[cache_key])

# ============================================================
# 1. 加载数据
# ============================================================
print(f"\n【STEP 1】Loading Data")
print("=" * 80)

segments_df = pd.read_csv(CONFIG['segments_path'])
trips_df = pd.read_csv(CONFIG['trips_path'])
vehicles_df = pd.read_csv(CONFIG['vehicles_path'])

print(f"   ✓ Segments: {len(segments_df):,}")
print(f"   ✓ Trips: {len(trips_df):,}")
print(f"   ✓ Vehicles: {len(vehicles_df):,}")

# 统一数据类型
segments_df['trip_id'] = segments_df['trip_id'].astype(str)
segments_df['vehicle_id'] = segments_df['vehicle_id'].astype(str)
trips_df['trip_id'] = trips_df['trip_id'].astype(str)
trips_df['vehicle_id'] = trips_df['vehicle_id'].astype(str)
vehicles_df['vehicle_id'] = vehicles_df['vehicle_id'].astype(str)

print(f"   ✓ Data types unified")

# ============================================================
# 2. 构建行程基因 (带缓存)
# ============================================================
print(f"\n【STEP 2】Building Trip Genes (with cache)")
print("=" * 80)

if cache_exists('trip_genes'):
    print(f"   Found trip_genes cache, loading...")
    trip_genes_df = load_cache('trip_genes')
else:
    print(f"   No cache found, computing trip genes...")
    trip_genes = []

    for trip_id in tqdm(segments_df['trip_id'].unique(), 
                       desc="   Processing trips", ncols=80):
        trip_segments = segments_df[segments_df['trip_id'] == trip_id]
        
        if len(trip_segments) == 0:
            continue
        
        trip_data = {
            'trip_id': trip_id,
            'vehicle_id': trip_segments['vehicle_id'].iloc[0],
        }
        
        # 片段基因：4种聚类的占比
        n_segs = len(trip_segments)
        for c in range(4):
            ratio = (trip_segments['cluster_id'] == c).sum() / n_segs
            trip_data[f'seg_cluster_{c}_ratio'] = ratio
        
        # 激进指标
        aggressive_ratio = ((trip_segments['cluster_id'] == 1) | 
                           (trip_segments['cluster_id'] == 2)).sum() / n_segs
        trip_data['aggressive_ratio'] = aggressive_ratio
        
        # 稳定指标
        stable_ratio = (trip_segments['cluster_id'] == 0).sum() / n_segs
        trip_data['stable_ratio'] = stable_ratio
        
        trip_genes.append(trip_data)

    trip_genes_df = pd.DataFrame(trip_genes)
    save_cache(trip_genes_df, 'trip_genes')

print(f"   ✓ Trip genes: {len(trip_genes_df):,} trips")

# ============================================================
# 3. 数据合并 (带缓存)
# ============================================================
print(f"\n【STEP 3】Data Merging (with cache)")
print("=" * 80)

if cache_exists('trips_enhanced'):
    print(f"   Found trips_enhanced cache, loading...")
    trips_enhanced = load_cache('trips_enhanced')
else:
    print(f"   No cache found, merging...")
    
    # 合并到 trips_df
    trips_enhanced = trips_df.merge(trip_genes_df, on=['trip_id', 'vehicle_id'], how='inner')
    print(f"   After trip gene merge: {len(trips_enhanced):,} rows")
    
    # 合并车辆画像
    vehicles_selected = vehicles_df[['vehicle_id', 'vehicle_cluster', 'cluster_label']].copy()
    trips_enhanced = trips_enhanced.merge(vehicles_selected, on='vehicle_id', how='left')
    print(f"   After vehicle merge: {len(trips_enhanced):,} rows")
    
    save_cache(trips_enhanced, 'trips_enhanced')

print(f"   ✓ Enhanced trips: {len(trips_enhanced):,} rows")

# ============================================================
# 4. 数据清洗 (带缓存)
# ============================================================
print(f"\n【STEP 4】Data Cleaning (with cache)")
print("=" * 80)

if cache_exists('df_model'):
    print(f"   Found df_model cache, loading...")
    df_model = load_cache('df_model')
    with open(CACHE_FILES['feature_cols'], 'r') as f:
        feature_cols = json.load(f)
else:
    print(f"   No cache found, cleaning...")
    
    target_col = 'charge_trigger_soc'
    
    # 删除缺失值
    required_cols = [
        'trip_id', 'vehicle_id', 'trip_duration_hrs', 'soc_drop',
        'seg_cluster_0_ratio', 'seg_cluster_1_ratio', 'seg_cluster_2_ratio', 'seg_cluster_3_ratio',
        'aggressive_ratio', 'stable_ratio',
        'vehicle_cluster', 'cluster_label',
        target_col
    ]
    
    available_cols = [col for col in required_cols if col in trips_enhanced.columns]
    df_model = trips_enhanced[available_cols].dropna()
    
    print(f"   Before cleaning: {len(trips_enhanced):,}")
    print(f"   After removing NaN: {len(df_model):,}")
    
    # 异常值处理
    for col in ['trip_duration_hrs', 'soc_drop', target_col]:
        q1, q99 = df_model[col].quantile([0.01, 0.99])
        df_model[col] = df_model[col].clip(q1, q99)
    
    print(f"   ✓ Outliers clipped")
    
    # 定义特征
    feature_cols = [
        'seg_cluster_0_ratio', 'seg_cluster_1_ratio', 'seg_cluster_2_ratio', 'seg_cluster_3_ratio',
        'aggressive_ratio', 'stable_ratio',
        'trip_duration_hrs', 'soc_drop',
    ]
    
    save_cache(df_model, 'df_model')
    with open(CACHE_FILES['feature_cols'], 'w') as f:
        json.dump(feature_cols, f)

print(f"   ✓ Model data: {len(df_model):,} rows")
print(f"   ✓ Features: {len(feature_cols)}")

# ============================================================
# 5. 特征标准化 (带缓存)
# ============================================================
print(f"\n【STEP 5】Feature Scaling (with cache)")
print("=" * 80)

if cache_exists('X_scaled') and cache_exists('y'):
    print(f"   Found X_scaled and y cache, loading...")
    X_scaled = load_cache('X_scaled')
    y = load_cache('y')
else:
    print(f"   No cache found, scaling...")
    
    target_col = 'charge_trigger_soc'
    X = df_model[feature_cols].copy()
    y = df_model[target_col].copy()
    
    # 编码车辆聚类
    le_vehicle = LabelEncoder()
    vehicle_cluster_encoded = le_vehicle.fit_transform(df_model['vehicle_cluster'].astype(str))
    X['vehicle_cluster'] = vehicle_cluster_encoded
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"   ✓ Scaled to shape: {X_scaled.shape}")
    
    save_cache(X_scaled, 'X_scaled')
    save_cache(y, 'y')

print(f"   ✓ X_scaled: {X_scaled.shape}")
print(f"   ✓ y: {len(y):,} samples")

# ============================================================
# 6. 特征相关性分析
# ============================================================
print(f"\n【STEP 6】Feature Correlation Analysis")
print("=" * 80)

feature_cols_with_vehicle = feature_cols + ['vehicle_cluster']

# 重建 X 用于相关性分析
X_for_corr = df_model[feature_cols].copy()
le_vehicle = LabelEncoder()
vehicle_cluster_encoded = le_vehicle.fit_transform(df_model['vehicle_cluster'].astype(str))
X_for_corr['vehicle_cluster'] = vehicle_cluster_encoded
X_for_corr[target_col] = df_model[target_col].values

correlation = X_for_corr.corr()[target_col].sort_values(ascending=False)

print(f"\n   Correlation with '{target_col}':")
for feat, corr_val in correlation.items():
    if feat != target_col:
        print(f"      {feat:<30} {corr_val:>+.4f}")

# 绘制相关性图
fig, ax = plt.subplots(figsize=(10, 6))
corr_plot = correlation.drop(target_col, errors='ignore')
colors = ['#2ecc71' if x > 0.05 else '#e74c3c' if x < -0.05 else '#95a5a6' 
          for x in corr_plot.values]
corr_plot.plot(kind='barh', ax=ax, color=colors, edgecolor='black', linewidth=1)
ax.set_xlabel('Correlation Coefficient', fontweight='bold')
ax.set_title('Feature Correlation with Charge Trigger SOC', fontweight='bold', fontsize=12)
ax.axvline(0, color='black', linestyle='-', linewidth=0.8)
ax.grid(alpha=0.3, axis='x')
plt.tight_layout()

for fmt, dpi in [('.png', 300), ('.pdf', None)]:
    path = os.path.join(CONFIG['figure_dir'], f'fig_correlation_with_target{fmt}')
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
plt.close()
print(f"\n   ✓ Saved: fig_correlation_with_target.png/pdf")

# ============================================================
# 7. XGBoost 模型训练 (带缓存)
# ============================================================
print(f"\n【STEP 7】XGBoost Model Training (with cache)")
print("=" * 80)

if cache_exists('model'):
    print(f"   Found model cache, loading...")
    model = load_cache('model')
else:
    print(f"   No cache found, training...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=CONFIG['seed']
    )
    
    print(f"   Train: {len(X_train):,}, Test: {len(X_test):,}")
    
    model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=CONFIG['seed'],
        n_jobs=-1,
        verbosity=0,
    )
    
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    save_cache(model, 'model')

# 评估
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=CONFIG['seed']
)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n   📊 Model Performance:")
print(f"      R² Score: {r2:.4f}")
print(f"      MAE: {mae:.4f}%")
print(f"      RMSE: {rmse:.4f}%")

# ============================================================
# 8. 特征重要性
# ============================================================
print(f"\n【STEP 8】Feature Importance Analysis")
print("=" * 80)

feature_importance = pd.DataFrame({
    'feature': feature_cols_with_vehicle,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n   Top 10 Important Features:")
for idx, (i, row) in enumerate(feature_importance.head(10).iterrows()):
    print(f"      {idx+1}. {row['feature']:<30} {row['importance']:>8.4f}")

# 绘制特征重要性
fig, ax = plt.subplots(figsize=(10, 8))
top_features = feature_importance.head(12)
ax.barh(top_features['feature'], top_features['importance'], 
       color='#3498db', edgecolor='black', linewidth=1)
ax.set_xlabel('Importance Score', fontweight='bold')
ax.set_title('XGBoost Feature Importance', fontweight='bold', fontsize=12)
ax.grid(alpha=0.3, axis='x')
plt.tight_layout()

for fmt, dpi in [('.png', 300), ('.pdf', None)]:
    path = os.path.join(CONFIG['figure_dir'], f'fig_feature_importance{fmt}')
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
plt.close()
print(f"\n   ✓ Saved: fig_feature_importance.png/pdf")

# ============================================================
# 9. SHAP 解释
# ============================================================
print(f"\n【STEP 9】SHAP Interpretation")
print("=" * 80)

print(f"\n   Computing SHAP values...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

print(f"   ✓ SHAP values computed")

# SHAP summary plot
fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, 
                 feature_names=feature_cols_with_vehicle,
                 plot_type='bar', show=False)
plt.title('SHAP Feature Importance', fontweight='bold', fontsize=12)
plt.tight_layout()

for fmt, dpi in [('.png', 300), ('.pdf', None)]:
    path = os.path.join(CONFIG['figure_dir'], f'fig_shap_summary{fmt}')
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: fig_shap_summary.png/pdf")

# ============================================================
# 10. 关键发现
# ============================================================
print(f"\n【STEP 10】Key Findings")
print("=" * 80)

print(f"""
【发现 1】激进驾驶与充电阈值的关系
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   相关性：{correlation.get('aggressive_ratio', 0):+.4f}
""")

agg_corr = correlation.get('aggressive_ratio', 0)
if agg_corr > 0.05:
    print(f"   ✓ 正相关：驾驶越激进 → 充电阈值越高（更早充电）")
else:
    print(f"   激进驾驶的影响较弱")

print(f"""
【发现 2】模型性能
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   R² Score: {r2:.4f}
   MAE: {mae:.4f}%
   Top feature: {feature_importance.iloc[0]['feature']}
""")

# ============================================================
# 11. 保存结果
# ============================================================
print(f"\n【STEP 11】Saving Results")
print("=" * 80)

results = {
    'model_performance': {
        'r2_score': float(r2),
        'mae': float(mae),
        'rmse': float(rmse),
    },
    'feature_importance': feature_importance.to_dict('records'),
    'feature_correlations': {str(k): float(v) for k, v in correlation.items()},
}

results_path = os.path.join(CONFIG['save_dir'], 'coupling_model_results.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"   ✓ {results_path}")

print("\n" + "=" * 80)
print("✅ COUPLING ANALYSIS COMPLETE!")
print("=" * 80)

print(f"""
Cache Info:
   Cache directory: {CONFIG['cache_dir']}
   Files cached: {len([f for f in CACHE_FILES.values() if os.path.exists(f)])}/{len(CACHE_FILES)}
   
Next run will load from cache (much faster!)
   
Model Results:
   R² = {r2:.4f}
   MAE = {mae:.4f}%
   Top feature: {feature_importance.iloc[0]['feature']}
""")

print("=" * 80)