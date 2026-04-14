"""
Step A: 构建 3.3 节所需的完整变量体系 (修复版)
修复: 聚类命名按能耗速率排序，而非 speed_cv
修复: 激进度指标基于综合能耗，而非单一 speed_cv
"""

import pandas as pd
import numpy as np   
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("📊 Step A: Building Variable System for Coupling Analysis (FIXED)")
print("=" * 70)

OUTPUT_DIR = "./coupling_analysis/results/"
FIGURE_DIR = "./coupling_analysis/figures/"
os.makedirs(FIGURE_DIR, exist_ok=True)

# ============================================================
# 1. 加载行程数据
# ============================================================
df = pd.read_csv(os.path.join(OUTPUT_DIR, 'inter_charge_trips.csv'))
print(f"\n📂 Loaded: {len(df):,} trips, {df['vehicle_id'].nunique():,} vehicles")

# ============================================================
# 2. 驾驶模式聚类
# ============================================================
print(f"\n{'=' * 70}")
print("🚗 Step A1: Driving Pattern Clustering")
print(f"{'=' * 70}")

cluster_features = ['speed_mean', 'speed_cv', 'idle_ratio', 'power_mean', 'soc_rate_per_hr']
available_features = [f for f in cluster_features if f in df.columns]
print(f"   Clustering features: {available_features}")

# IQR 裁剪异常值
df_clean = df.copy()
for feat in available_features:
    q1, q3 = df_clean[feat].quantile([0.01, 0.99])
    df_clean[feat] = df_clean[feat].clip(q1, q3)

X = df_clean[available_features].fillna(0).values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 选择最优 k
print("\n   Finding optimal k...")
sil_scores = {}
for k in range(3, 7):
    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels, sample_size=min(10000, len(X_scaled)), random_state=42)
    sil_scores[k] = sil
    print(f"      k={k}: Silhouette={sil:.4f}")

best_k = max(sil_scores, key=sil_scores.get)
print(f"   ✅ Best k={best_k} (Silhouette={sil_scores[best_k]:.4f})")

km_final = KMeans(n_clusters=best_k, n_init=20, random_state=42)
df['driving_pattern'] = km_final.fit_predict(X_scaled)

# ── 关键修复: 按 soc_rate_per_hr (能耗速率) 排序命名 ──
# 能耗越高 → 越激进
pattern_stats = df.groupby('driving_pattern').agg({
    'speed_mean': 'mean',
    'speed_cv': 'mean',
    'idle_ratio': 'mean',
    'power_mean': 'mean',
    'soc_rate_per_hr': 'mean',
}).round(2)

# 按能耗速率从低到高排序
rate_order = pattern_stats['soc_rate_per_hr'].sort_values().index.tolist()
pattern_names_ordered = ['Eco-Idle', 'Urban Moderate', 'Active Dynamic', 'Highway Aggressive'][:best_k]

pattern_name_map = {}
for rank, cluster_id in enumerate(rate_order):
    pattern_name_map[cluster_id] = pattern_names_ordered[rank]

df['driving_pattern_name'] = df['driving_pattern'].map(pattern_name_map)

print(f"\n   Driving Pattern Profiles (sorted by energy consumption):")
print(f"   {'Pattern':<20} {'Count':>8} {'Speed':>8} {'CV':>8} {'Idle%':>8} {'Power':>10} {'SOC/hr':>8}")
print(f"   {'─'*20} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*10} {'─'*8}")
for cluster_id in rate_order:
    name = pattern_name_map[cluster_id]
    row = pattern_stats.loc[cluster_id]
    n = (df['driving_pattern'] == cluster_id).sum()
    print(f"   {name:<20} {n:>8,} {row['speed_mean']:>8.1f} {row['speed_cv']:>8.3f} "
          f"{row['idle_ratio']:>8.2%} {row['power_mean']:>10.0f} {row['soc_rate_per_hr']:>8.2f}")

# 验证命名合理性
print(f"\n   ✅ Naming verification:")
least_aggressive = pattern_stats.loc[rate_order[0]]
most_aggressive = pattern_stats.loc[rate_order[-1]]
print(f"      Least aggressive: SOC rate={least_aggressive['soc_rate_per_hr']:.2f} %/hr, "
      f"speed={least_aggressive['speed_mean']:.1f} km/h")
print(f"      Most aggressive:  SOC rate={most_aggressive['soc_rate_per_hr']:.2f} %/hr, "
      f"speed={most_aggressive['speed_mean']:.1f} km/h")

# ============================================================
# 3. 激进度指标 (基于综合能耗，而非单一 speed_cv)
# ============================================================
print(f"\n{'=' * 70}")
print("📊 Step A2: Computing Aggressiveness Index (Energy-based)")
print(f"{'=' * 70}")

# ── 关键修复: 激进度 = 归一化的能耗速率 ──
# soc_rate_per_hr 越高 → 驾驶越激进（能量消耗越快）
rate_min = df['soc_rate_per_hr'].quantile(0.01)
rate_max = df['soc_rate_per_hr'].quantile(0.99)
df['aggressiveness_index'] = ((df['soc_rate_per_hr'] - rate_min) / (rate_max - rate_min)).clip(0, 1)

# 也保留一个多维度综合指标
# 综合考虑: 高能耗 + 高速度 + 高加速变异 + 低停车比
df['agg_composite'] = (
    0.4 * df['aggressiveness_index'] +                                           # 能耗主导
    0.2 * ((df['speed_mean'] - df['speed_mean'].quantile(0.01)) /
           (df['speed_mean'].quantile(0.99) - df['speed_mean'].quantile(0.01))).clip(0, 1) +
    0.2 * ((df['speed_cv'] - df['speed_cv'].quantile(0.01)) /
           (df['speed_cv'].quantile(0.99) - df['speed_cv'].quantile(0.01))).clip(0, 1) +
    0.2 * (1 - df['idle_ratio'].clip(0, 1))                                     # 低空闲 = 更激进
)

# 离散标记
aggressive_patterns = [k for k, v in pattern_name_map.items()
                       if 'Aggressive' in v or 'Dynamic' in v]
df['is_aggressive'] = df['driving_pattern'].isin(aggressive_patterns).astype(int)

print(f"   Aggressiveness Index (energy-based): "
      f"mean={df['aggressiveness_index'].mean():.3f}, std={df['aggressiveness_index'].std():.3f}")
print(f"   Composite Aggressiveness: "
      f"mean={df['agg_composite'].mean():.3f}, std={df['agg_composite'].std():.3f}")
print(f"   Aggressive trips (discrete): {df['is_aggressive'].sum():,} "
      f"({df['is_aggressive'].mean()*100:.1f}%)")

# ============================================================
# 4. 变量体系
# ============================================================
print(f"\n{'=' * 70}")
print("📋 Step A3: Final Variable System")
print(f"{'=' * 70}")

x_vars = {
    '微观驾驶行为': ['speed_mean', 'speed_cv', 'speed_std', 'idle_ratio',
                 'aggressiveness_index', 'agg_composite', 'driving_pattern_name'],
    '中观能耗状态': ['trip_duration_hrs', 'soc_drop', 'soc_rate_per_hr',
                 'power_mean', 'power_std'],
    '末端特征':     ['end_speed_mean', 'end_power_mean', 'soc_end_trip'],
}

y_vars = {
    '充电触发阈值': ['charge_trigger_soc'],
    '充电补能量':   ['charge_gain_soc'],
    '充电时长':     ['charge_duration_min'],
    '充电类型':     ['charge_type'],
}

print("\n   X Variables (Independent):")
for category, vars_list in x_vars.items():
    available = [v for v in vars_list if v in df.columns]
    print(f"      {category}: {available}")

print("\n   Y Variables (Dependent):")
for category, vars_list in y_vars.items():
    available = [v for v in vars_list if v in df.columns]
    print(f"      {category}: {available}")

# ============================================================
# 5. 保存
# ============================================================
output_path = os.path.join(OUTPUT_DIR, 'coupling_analysis_dataset.csv')
df.to_csv(output_path, index=False)
print(f"\n💾 Saved: {output_path}")
print(f"   Shape: {df.shape}")
print(f"   Size: {os.path.getsize(output_path)/1024/1024:.1f} MB")

# ============================================================
# 6. 可视化
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

colors = {
    'Eco-Idle': '#2ecc71',
    'Urban Moderate': '#3498db',
    'Active Dynamic': '#f39c12',
    'Highway Aggressive': '#e74c3c',
}

# (a) 模式分布 (按能耗排序)
ax = axes[0]
ordered_names = [pattern_name_map[c] for c in rate_order]
ordered_counts = [(df['driving_pattern_name'] == n).sum() for n in ordered_names]
bars = ax.bar(ordered_names, ordered_counts,
              color=[colors.get(n, 'grey') for n in ordered_names])
for bar, v in zip(bars, ordered_counts):
    ax.text(bar.get_x()+bar.get_width()/2, v, f'{v:,}',
            ha='center', va='bottom', fontweight='bold', fontsize=9)
ax.set_ylabel('Number of Trips')
ax.set_title('(a) Driving Pattern Distribution\n(ordered by energy rate)', fontweight='bold')
ax.tick_params(axis='x', rotation=15)
ax.grid(alpha=0.3, axis='y')

# (b) Speed vs Power (colored by pattern) - 更直观
ax = axes[1]
sample = df.sample(min(5000, len(df)), random_state=42)
for name in ordered_names:
    mask = sample['driving_pattern_name'] == name
    ax.scatter(sample[mask]['speed_mean'], sample[mask]['power_mean'],
               c=colors.get(name, 'grey'), s=8, alpha=0.4, label=name)
ax.set_xlabel('Average Speed (km/h)')
ax.set_ylabel('Average Power (W)')
ax.set_title('(b) Speed vs Power by Pattern', fontweight='bold')
ax.legend(markerscale=3, fontsize=9)
ax.grid(alpha=0.2)

# (c) Aggressiveness Index 分布
ax = axes[2]
ax.hist(df['aggressiveness_index'], bins=50, color='#e74c3c', alpha=0.7, edgecolor='white')
ax.axvline(df['aggressiveness_index'].mean(), color='black', linestyle='--',
           label=f'Mean={df["aggressiveness_index"].mean():.3f}')
ax.set_xlabel('Aggressiveness Index (energy-based)')
ax.set_ylabel('Count')
ax.set_title('(c) Aggressiveness Index Distribution', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3, axis='y')

plt.suptitle('Driving Pattern Analysis (Energy-based Naming)', fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(FIGURE_DIR, 'fig_driving_patterns.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(FIGURE_DIR, 'fig_driving_patterns.pdf'), bbox_inches='tight')
plt.close(fig)
print(f"   ✅ Saved: fig_driving_patterns.png/pdf")

print(f"\n✅ Step A Complete!")