"""
Step 11: Fast Feature Diagnosis and Optimization (Lightweight Version)
快速特征诊断与优化（轻量化版本）

功能：
1. 特征工程优化：从36维提取15-20个关键特征
2. 特征诊断：Fisher Score、相关性矩阵、PCA方差分析
3. 特征选择：Fisher排序 + 相关性去重
4. 算法对比：GMM vs KMeans（K=2-6）
5. 最优聚类：最优算法+K值
6. 6张可视化图表
7. 诊断报告和输出文件

运行时间：约3-5分钟
"""

import numpy as np
import pandas as pd
import os
import json
import warnings
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from scipy.stats import entropy as scipy_entropy
from tqdm import tqdm

warnings.filterwarnings('ignore')

rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 11
rcParams['figure.dpi'] = 150

print("=" * 80)
print("🚀 STEP 11: FAST FEATURE DIAGNOSIS AND OPTIMIZATION")
print("   (Lightweight Version — Target: 3-5 minutes)")
print("=" * 80)

# ============================================================
# 配置
# ============================================================
CONFIG = {
    'vehicle_features_path': './coupling_analysis/results/vehicles_aggregated_features.csv',
    'segments_path': './coupling_analysis/results/segments_integrated_complete.csv',
    'results_dir': './vehicle_clustering/results/',
    'figures_dir': './vehicle_clustering/results/figures_diagnosis_fast/',
    'seed': 42,
    'k_range': range(2, 7),          # K=2..6
    'corr_threshold': 0.80,           # 相关性去重阈值
    'top_n_features': 18,             # 目标特征数 (15-20)
}

os.makedirs(CONFIG['results_dir'], exist_ok=True)
os.makedirs(CONFIG['figures_dir'], exist_ok=True)

# ============================================================
# STEP 1: 加载数据
# ============================================================
print(f"\n{'='*80}")
print("【STEP 1】Loading Data")
print("=" * 80)

vehicle_features = pd.read_csv(CONFIG['vehicle_features_path'])
print(f"   ✓ Vehicle features: {len(vehicle_features):,} vehicles, {len(vehicle_features.columns)} columns")

segments_df = pd.read_csv(CONFIG['segments_path'])
print(f"   ✓ Segments: {len(segments_df):,} rows")

# 确认 vehicle_id 对齐
common_ids = set(vehicle_features['vehicle_id']) & set(segments_df['vehicle_id'])
vehicle_features = vehicle_features[vehicle_features['vehicle_id'].isin(common_ids)].reset_index(drop=True)
print(f"   ✓ Common vehicles: {len(vehicle_features):,}")

# ============================================================
# STEP 2: 特征工程优化
# ============================================================
print(f"\n{'='*80}")
print("【STEP 2】Feature Engineering — Extracting Optimized Features")
print("=" * 80)

print("   Computing per-vehicle optimized features from segments...")

# 预先按 vehicle_id 分组（加速）
seg_grouped = segments_df.groupby('vehicle_id')

# 识别 cluster_label 列名
cluster_col = None
for candidate in ['cluster_label', 'cluster', 'segment_cluster', 'label']:
    if candidate in segments_df.columns:
        cluster_col = candidate
        break
if cluster_col is None:
    raise ValueError("Cannot find cluster label column in segments_df. "
                     "Expected one of: cluster_label, cluster, segment_cluster, label")

# 识别 SOC 相关列
soc_col = None
for candidate in ['soc', 'SOC', 'soc_start', 'soc_end']:
    if candidate in segments_df.columns:
        soc_col = candidate
        break

# 识别时长列
dur_col = None
for candidate in ['duration_seconds', 'duration_min', 'duration']:
    if candidate in segments_df.columns:
        dur_col = candidate
        break

# 识别速度列
speed_col = None
for candidate in ['avg_speed', 'speed_mean', 'speed']:
    if candidate in segments_df.columns:
        speed_col = candidate
        break

print(f"   ✓ Cluster column:   '{cluster_col}'")
print(f"   ✓ SOC column:       '{soc_col or 'N/A'}'")
print(f"   ✓ Duration column:  '{dur_col or 'N/A'}'")
print(f"   ✓ Speed column:     '{speed_col or 'N/A'}'")

optimized_rows = []

for vid in tqdm(vehicle_features['vehicle_id'], desc="   Extracting features", ncols=80):
    if vid not in seg_grouped.groups:
        continue
    grp = seg_grouped.get_group(vid).copy()
    grp = grp.sort_values('start_dt') if 'start_dt' in grp.columns else grp
    labels_seq = grp[cluster_col].values
    n_seg = len(labels_seq)

    # ── 1. 转移矩阵特征 ──────────────────────────────────────
    # 构建4×4转移矩阵 T
    T = np.zeros((4, 4))
    if n_seg >= 2:
        for i in range(n_seg - 1):
            src = int(labels_seq[i]) if 0 <= labels_seq[i] <= 3 else None
            dst = int(labels_seq[i + 1]) if 0 <= labels_seq[i + 1] <= 3 else None
            if src is not None and dst is not None:
                T[src, dst] += 1
        row_sums = T.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        T = T / row_sums  # 归一化

    highway_preference  = (T[2, :].sum() + T[:, 2].sum()) / 2
    urban_affinity      = (T[1, :].sum() + T[:, 1].sum()) / 2
    idle_preference     = (T[0, :].sum() + T[:, 0].sum()) / 2
    mode_stability      = np.trace(T) / 4
    transition_diversity = np.count_nonzero(T > 0.01) / 16

    # ── 2. 驾驶行为特征 ──────────────────────────────────────
    counts = Counter(labels_seq)
    total = n_seg

    idle_time_prop    = counts.get(0, 0) / total
    urban_time_prop   = counts.get(1, 0) / total
    highway_time_prop = counts.get(2, 0) / total
    accel_time_prop   = counts.get(3, 0) / total

    # SOC下降速率
    avg_soc_drop_rate = 0.0
    if soc_col and soc_col in grp.columns:
        soc_vals = grp[soc_col].dropna().values
        if len(soc_vals) >= 2:
            soc_diff = np.diff(soc_vals)
            drops = soc_diff[soc_diff < 0]
            if dur_col and dur_col in grp.columns:
                dur_vals = grp[dur_col].dropna().values
                total_time = dur_vals.sum() if len(dur_vals) > 0 else 1
                avg_soc_drop_rate = abs(drops.sum()) / max(total_time, 1)
            else:
                avg_soc_drop_rate = abs(drops.mean()) if len(drops) > 0 else 0.0

    # 速度波动
    speed_variance = 0.0
    if speed_col and speed_col in grp.columns:
        spd = grp[speed_col].dropna().values
        if len(spd) >= 2:
            speed_variance = float(np.var(spd))

    # 平均行程距离（用时长×速度代理）
    trip_distance_avg = 0.0
    if dur_col in grp.columns and speed_col and speed_col in grp.columns:
        dur_vals = grp[dur_col].fillna(0).values
        spd_vals = grp[speed_col].fillna(0).values
        trip_distance_avg = float(np.mean(dur_vals * spd_vals))

    # ── 3. 演化特征 ──────────────────────────────────────────
    sequence_length = n_seg

    # 模式切换频率
    if n_seg >= 2:
        switches = sum(1 for i in range(n_seg - 1) if labels_seq[i] != labels_seq[i + 1])
        mode_switching_freq = switches / (n_seg - 1)
    else:
        mode_switching_freq = 0.0

    # 稳定性指数（最长连续相同状态 / 总长度）
    if n_seg >= 1:
        max_run = 1
        cur_run = 1
        for i in range(1, n_seg):
            if labels_seq[i] == labels_seq[i - 1]:
                cur_run += 1
                max_run = max(max_run, cur_run)
            else:
                cur_run = 1
        stability_index = max_run / n_seg
    else:
        stability_index = 1.0

    # 模式熵
    probs = np.array([counts.get(c, 0) / total for c in range(4)])
    mode_entropy = float(scipy_entropy(probs + 1e-9))

    # 节奏一致性：连续运行长度的变异系数倒数
    run_lengths = []
    if n_seg >= 1:
        cur_run = 1
        for i in range(1, n_seg):
            if labels_seq[i] == labels_seq[i - 1]:
                cur_run += 1
            else:
                run_lengths.append(cur_run)
                cur_run = 1
        run_lengths.append(cur_run)
    if len(run_lengths) >= 2:
        cv = np.std(run_lengths) / (np.mean(run_lengths) + 1e-9)
        rhythm_consistency = 1.0 / (1.0 + cv)
    else:
        rhythm_consistency = 1.0

    optimized_rows.append({
        'vehicle_id': vid,
        # 转移特征
        'highway_preference':   highway_preference,
        'urban_affinity':       urban_affinity,
        'idle_preference':      idle_preference,
        'mode_stability':       mode_stability,
        'transition_diversity': transition_diversity,
        # 驾驶行为特征
        'idle_time_prop':       idle_time_prop,
        'urban_time_prop':      urban_time_prop,
        'highway_time_prop':    highway_time_prop,
        'accel_time_prop':      accel_time_prop,
        'avg_soc_drop_rate':    avg_soc_drop_rate,
        'speed_variance':       speed_variance,
        'trip_distance_avg':    trip_distance_avg,
        # 演化特征
        'sequence_length':      sequence_length,
        'mode_switching_freq':  mode_switching_freq,
        'stability_index':      stability_index,
        'mode_entropy':         mode_entropy,
        'rhythm_consistency':   rhythm_consistency,
    })

df_opt = pd.DataFrame(optimized_rows)
print(f"\n   ✓ Optimized features extracted: {len(df_opt):,} vehicles × {len(df_opt.columns) - 1} features")

# 合并原始分布特征（cluster_X_ratio）
dist_cols = [f'cluster_{c}_ratio' for c in range(4)]
dist_cols_available = [c for c in dist_cols if c in vehicle_features.columns]
if dist_cols_available:
    df_dist = vehicle_features[['vehicle_id'] + dist_cols_available].copy()
    df_opt = df_opt.merge(df_dist, on='vehicle_id', how='left')
    print(f"   ✓ Added {len(dist_cols_available)} distribution features")

total_opt_features = len(df_opt.columns) - 1
print(f"   ✓ Total candidate features: {total_opt_features}")

# ============================================================
# STEP 3: 特征诊断
# ============================================================
print(f"\n{'='*80}")
print("【STEP 3】Feature Diagnosis (Fisher Score, Correlation, PCA)")
print("=" * 80)

feature_cols_all = [c for c in df_opt.columns if c != 'vehicle_id']

# 填充缺失值
X_raw = df_opt[feature_cols_all].copy()
X_raw = X_raw.fillna(X_raw.median())

# 标准化
scaler = StandardScaler()
X_scaled_all = scaler.fit_transform(X_raw)
X_scaled_all = np.nan_to_num(X_scaled_all, nan=0.0, posinf=0.0, neginf=0.0)

# ── 3.1 临时K-Means粗聚类，用于计算Fisher Score ──
print("   Computing preliminary clustering for Fisher Score...")
tmp_kmeans = KMeans(n_clusters=3, n_init=10, random_state=CONFIG['seed'])
tmp_labels = tmp_kmeans.fit_predict(X_scaled_all)

# ── 3.2 Fisher判别分数 ──
def compute_fisher_scores(X, labels):
    """计算每个特征的Fisher判别分数 (between-class / within-class variance)."""
    unique_classes = np.unique(labels)
    overall_mean = X.mean(axis=0)
    n_total = len(X)
    n_features = X.shape[1]

    between_var = np.zeros(n_features)
    within_var = np.zeros(n_features)

    for c in unique_classes:
        mask = labels == c
        n_c = mask.sum()
        class_mean = X[mask].mean(axis=0)
        between_var += n_c * (class_mean - overall_mean) ** 2
        within_var += ((X[mask] - class_mean) ** 2).sum(axis=0)

    within_var = np.maximum(within_var / n_total, 1e-10)
    fisher = between_var / within_var
    return fisher

fisher_scores = compute_fisher_scores(X_scaled_all, tmp_labels)
fisher_df = pd.DataFrame({
    'feature': feature_cols_all,
    'fisher_score': fisher_scores
}).sort_values('fisher_score', ascending=False).reset_index(drop=True)

print(f"\n   Top 10 features by Fisher Score:")
for _, row in fisher_df.head(10).iterrows():
    print(f"      {row['feature']:35s} Fisher={row['fisher_score']:.4f}")

# ── 3.3 特征相关性矩阵 ──
corr_matrix = X_raw.corr()

# ── 3.4 PCA方差分析 ──
pca_full = PCA(random_state=CONFIG['seed'])
pca_full.fit(X_scaled_all)
explained_var = pca_full.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)

n_components_90 = int(np.argmax(cumulative_var >= 0.90)) + 1
n_components_95 = int(np.argmax(cumulative_var >= 0.95)) + 1
print(f"\n   PCA: {n_components_90} PCs for 90% variance, {n_components_95} for 95%")

# ============================================================
# STEP 4: 特征选择（Fisher + 相关性去重）
# ============================================================
print(f"\n{'='*80}")
print("【STEP 4】Feature Selection (Fisher Score + Correlation Dedup)")
print("=" * 80)

# 按Fisher Score排序后，相关性去重
ordered_features = fisher_df['feature'].tolist()
selected_features = []

for feat in ordered_features:
    if len(selected_features) >= CONFIG['top_n_features']:
        break
    # 检查与已选特征的相关性
    if not selected_features:
        selected_features.append(feat)
        continue
    corrs = corr_matrix.loc[feat, selected_features].abs()
    if corrs.max() <= CONFIG['corr_threshold']:
        selected_features.append(feat)

# 若选出特征不足 15 个，放宽阈值继续补充
if len(selected_features) < 15:
    for feat in ordered_features:
        if feat in selected_features:
            continue
        if len(selected_features) >= 15:
            break
        selected_features.append(feat)

print(f"\n   ✓ Selected {len(selected_features)} optimal features:")
for i, feat in enumerate(selected_features, 1):
    score = fisher_df.loc[fisher_df['feature'] == feat, 'fisher_score'].values[0]
    print(f"      {i:2d}. {feat:35s} Fisher={score:.4f}")

# ============================================================
# STEP 5: 算法对比（GMM vs KMeans，K=2~6）
# ============================================================
print(f"\n{'='*80}")
print("【STEP 5】Algorithm Comparison: GMM vs KMeans (K=2~6)")
print("=" * 80)

# 使用优化特征子集
X_sel = X_raw[selected_features].copy()
X_sel = X_sel.fillna(X_sel.median())
scaler_sel = StandardScaler()
X_sel_scaled = scaler_sel.fit_transform(X_sel)
X_sel_scaled = np.nan_to_num(X_sel_scaled, nan=0.0, posinf=0.0, neginf=0.0)

comparison_results = {}
best_overall = {'sil': -1, 'algo': None, 'k': None, 'labels': None}

for algo_name in ['KMeans', 'GMM']:
    comparison_results[algo_name] = {}
    print(f"\n   📊 {algo_name}:")

    for k in CONFIG['k_range']:
        try:
            if algo_name == 'KMeans':
                model = KMeans(n_clusters=k, n_init=20, random_state=CONFIG['seed'], max_iter=300)
                labels = model.fit_predict(X_sel_scaled)
            else:
                # GMM: try covariance types, pick best
                best_gmm_sil = -1
                best_gmm_labels = None
                for cov in ['tied', 'diag', 'spherical']:
                    try:
                        gmm = GaussianMixture(
                            n_components=k, covariance_type=cov,
                            n_init=5, random_state=CONFIG['seed'],
                            max_iter=300, reg_covar=1e-4
                        )
                        gmm.fit(X_sel_scaled)
                        lbl = gmm.predict(X_sel_scaled)
                        s = silhouette_score(X_sel_scaled, lbl,
                                             sample_size=min(3000, len(lbl)),
                                             random_state=CONFIG['seed'])
                        if s > best_gmm_sil:
                            best_gmm_sil = s
                            best_gmm_labels = lbl
                    except Exception:
                        pass
                labels = best_gmm_labels
                if labels is None:
                    continue

            sil = silhouette_score(X_sel_scaled, labels,
                                   sample_size=min(3000, len(labels)),
                                   random_state=CONFIG['seed'])
            db  = davies_bouldin_score(X_sel_scaled, labels)
            ch  = calinski_harabasz_score(X_sel_scaled, labels)

            comparison_results[algo_name][k] = {
                'silhouette': float(sil),
                'davies_bouldin': float(db),
                'calinski_harabasz': float(ch),
                'labels': labels.tolist(),
            }

            print(f"      K={k}: Sil={sil:.4f}  DB={db:.4f}  CH={ch:.1f}")

            if sil > best_overall['sil']:
                best_overall = {'sil': sil, 'algo': algo_name, 'k': k,
                                'labels': labels}

        except Exception as e:
            print(f"      K={k}: Failed ({str(e)[:40]})")

print(f"\n   🏆 Best overall: {best_overall['algo']} K={best_overall['k']} "
      f"(Silhouette={best_overall['sil']:.4f})")

# ============================================================
# STEP 6: 最优聚类分析
# ============================================================
print(f"\n{'='*80}")
print("【STEP 6】Optimal Clustering Analysis")
print("=" * 80)

opt_labels = np.array(best_overall['labels'])
unique_clusters = sorted(np.unique(opt_labels))

print(f"\n   Cluster Distribution:")
cluster_stats = {}
for c in unique_clusters:
    mask = opt_labels == c
    n = mask.sum()
    pct = n / len(opt_labels) * 100
    print(f"      C{c}: {n:>6,} vehicles ({pct:>5.1f}%)")

    # 特征均值
    feat_means = X_sel.iloc[mask][selected_features].mean()
    cluster_stats[int(c)] = {
        'size': int(n),
        'pct': float(pct),
        'feature_means': feat_means.to_dict(),
    }

# 自动标记
def auto_label(stats, selected):
    """根据特征均值自动为簇生成描述标签."""
    labels_map = {}
    for c, st in stats.items():
        fm = st['feature_means']
        desc = []
        if 'highway_time_prop' in fm and fm['highway_time_prop'] > 0.25:
            desc.append('Highway-dominant')
        elif 'idle_time_prop' in fm and fm['idle_time_prop'] > 0.50:
            desc.append('Idle-heavy')
        elif 'urban_time_prop' in fm and fm['urban_time_prop'] > 0.40:
            desc.append('Urban-focused')
        if 'stability_index' in fm and fm['stability_index'] > 0.60:
            desc.append('Stable')
        if 'mode_switching_freq' in fm and fm['mode_switching_freq'] > 0.50:
            desc.append('High-switching')
        labels_map[c] = ' + '.join(desc) if desc else f'Cluster {c}'
    return labels_map

cluster_labels_map = auto_label(cluster_stats, selected_features)
print(f"\n   Auto-labels:")
for c, lbl in cluster_labels_map.items():
    print(f"      C{c}: {lbl}")

# ============================================================
# STEP 7: 与原方案对比（读取旧聚类结果如存在）
# ============================================================
orig_labels = None
orig_label_col = None
orig_result_path = os.path.join(CONFIG['results_dir'], 'vehicle_clustering_results.csv')
for candidate_path in [
    orig_result_path,
    os.path.join(CONFIG['results_dir'], 'vehicle_clustering_improved_3d.csv'),
    os.path.join(CONFIG['results_dir'], 'vehicle_clustering_gmm_k4.csv'),
]:
    if os.path.exists(candidate_path):
        try:
            df_orig = pd.read_csv(candidate_path)
            for c in ['vehicle_cluster', 'cluster', 'label', 'cluster_label']:
                if c in df_orig.columns:
                    orig_labels_full = df_orig.set_index('vehicle_id')[c]
                    orig_label_col = c
                    orig_labels_series = orig_labels_full.reindex(df_opt['vehicle_id']).dropna()
                    if len(orig_labels_series) > 100:
                        orig_labels = orig_labels_series.values.astype(int)
                        print(f"\n   ✓ Loaded original clustering from: {candidate_path}")
                        break
        except Exception:
            pass
        if orig_labels is not None:
            break

# ============================================================
# STEP 8: 可视化
# ============================================================
print(f"\n{'='*80}")
print("【STEP 8】Generating 6 Visualizations")
print("=" * 80)

# ─────────────────────────────────────────────────────────────
# 图1: 特征重要性排序
# ─────────────────────────────────────────────────────────────
print("   📊 Figure 1: Feature Importance...")
fig, ax = plt.subplots(figsize=(10, 7))
top_n = min(20, len(fisher_df))
df_plot = fisher_df.head(top_n).iloc[::-1]  # 从高到低（横向条形）
colors_bar = ['#2ecc71' if f in selected_features else '#95a5a6'
              for f in df_plot['feature']]
bars = ax.barh(df_plot['feature'], df_plot['fisher_score'], color=colors_bar)
ax.set_xlabel('Fisher Discriminant Score', fontsize=12)
ax.set_title('Feature Importance Ranking (Fisher Score)\n'
             'Green = Selected, Gray = Excluded', fontsize=13)
ax.axvline(x=fisher_df.loc[fisher_df['feature'].isin(selected_features),
                             'fisher_score'].min(),
           color='red', linestyle='--', alpha=0.6, label='Selection threshold')
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(CONFIG['figures_dir'], '01_feature_importance.png'), dpi=150)
plt.close(fig)
print("      ✓ 01_feature_importance.png")

# ─────────────────────────────────────────────────────────────
# 图2: 特征相关性热力图（仅选出的特征）
# ─────────────────────────────────────────────────────────────
print("   📊 Figure 2: Feature Correlation Heatmap...")
corr_sel = corr_matrix.loc[selected_features, selected_features]
fig, ax = plt.subplots(figsize=(max(10, len(selected_features) * 0.6),
                                max(8, len(selected_features) * 0.6)))
im = ax.imshow(corr_sel.values, vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_xticks(range(len(selected_features)))
ax.set_yticks(range(len(selected_features)))
ax.set_xticklabels(selected_features, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(selected_features, fontsize=9)
ax.set_title('Feature Correlation Matrix (Selected Features)', fontsize=13)
# 标注相关系数
for i in range(len(selected_features)):
    for j in range(len(selected_features)):
        val = corr_sel.values[i, j]
        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                fontsize=7, color='black' if abs(val) < 0.7 else 'white')
plt.tight_layout()
fig.savefig(os.path.join(CONFIG['figures_dir'], '02_feature_correlation.png'), dpi=150)
plt.close(fig)
print("      ✓ 02_feature_correlation.png")

# ─────────────────────────────────────────────────────────────
# 图3: PCA方差分析
# ─────────────────────────────────────────────────────────────
print("   📊 Figure 3: PCA Variance Analysis...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

n_show = min(len(explained_var), 20)
axes[0].bar(range(1, n_show + 1), explained_var[:n_show] * 100, color='#3498db', alpha=0.8)
axes[0].set_xlabel('Principal Component', fontsize=11)
axes[0].set_ylabel('Explained Variance (%)', fontsize=11)
axes[0].set_title('Individual Explained Variance', fontsize=12)
axes[0].set_xticks(range(1, n_show + 1))

axes[1].plot(range(1, n_show + 1), cumulative_var[:n_show] * 100,
             marker='o', color='#e74c3c', linewidth=2, markersize=5)
axes[1].axhline(y=90, color='gray', linestyle='--', alpha=0.7, label='90%')
axes[1].axhline(y=95, color='black', linestyle='--', alpha=0.7, label='95%')
axes[1].fill_between(range(1, n_show + 1), cumulative_var[:n_show] * 100, alpha=0.2, color='#e74c3c')
axes[1].set_xlabel('Number of Components', fontsize=11)
axes[1].set_ylabel('Cumulative Explained Variance (%)', fontsize=11)
axes[1].set_title('Cumulative Explained Variance', fontsize=12)
axes[1].legend()
axes[1].set_xticks(range(1, n_show + 1))

plt.suptitle(f'PCA Variance Analysis — {len(selected_features)} Optimized Features\n'
             f'90% variance: {n_components_90} PCs  |  95% variance: {n_components_95} PCs',
             fontsize=12, y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(CONFIG['figures_dir'], '03_pca_variance.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)
print("      ✓ 03_pca_variance.png")

# ─────────────────────────────────────────────────────────────
# 图4: 算法对比 GMM vs KMeans
# ─────────────────────────────────────────────────────────────
print("   📊 Figure 4: Algorithm Comparison (GMM vs KMeans)...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
k_vals = list(CONFIG['k_range'])
metrics_titles = ['Silhouette Score (↑)', 'Davies-Bouldin (↓)', 'Calinski-Harabasz (↑)']
metrics_keys   = ['silhouette', 'davies_bouldin', 'calinski_harabasz']
colors_algo    = {'GMM': '#e74c3c', 'KMeans': '#3498db'}
markers_algo   = {'GMM': 'o', 'KMeans': 's'}

for ax, mkey, mtitle in zip(axes, metrics_keys, metrics_titles):
    for algo_name in ['GMM', 'KMeans']:
        vals = [comparison_results[algo_name].get(k, {}).get(mkey, np.nan)
                for k in k_vals]
        ax.plot(k_vals, vals, marker=markers_algo[algo_name],
                color=colors_algo[algo_name], label=algo_name,
                linewidth=2, markersize=7)
    ax.set_xlabel('Number of Clusters K', fontsize=11)
    ax.set_ylabel(mtitle, fontsize=11)
    ax.set_title(mtitle, fontsize=12)
    ax.set_xticks(k_vals)
    ax.legend()
    ax.grid(alpha=0.3)

# 标注最优点
best_k = best_overall['k']
best_algo = best_overall['algo']
best_sil = comparison_results[best_algo].get(best_k, {}).get('silhouette', np.nan)
if not np.isnan(best_sil):
    axes[0].annotate(f'Best\n{best_algo} K={best_k}',
                     xy=(best_k, best_sil),
                     xytext=(best_k + 0.3, best_sil + 0.02),
                     arrowprops=dict(arrowstyle='->', color='green'),
                     color='green', fontsize=9)

plt.suptitle('Algorithm Comparison: GMM vs KMeans', fontsize=13, y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(CONFIG['figures_dir'], '04_algorithm_comparison.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)
print("      ✓ 04_algorithm_comparison.png")

# ─────────────────────────────────────────────────────────────
# 图5: 最优聚类 PCA 投影
# ─────────────────────────────────────────────────────────────
print("   📊 Figure 5: Optimal Clustering PCA Projection...")
pca2 = PCA(n_components=2, random_state=CONFIG['seed'])
X_pca2 = pca2.fit_transform(X_sel_scaled)

fig, ax = plt.subplots(figsize=(9, 7))
palette = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
for c in unique_clusters:
    mask = opt_labels == c
    label_str = cluster_labels_map.get(c, f'C{c}')
    ax.scatter(X_pca2[mask, 0], X_pca2[mask, 1],
               c=palette[c % len(palette)], label=f'C{c}: {label_str}',
               alpha=0.5, s=20, edgecolors='none')

ax.set_xlabel(f'PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=11)
ax.set_ylabel(f'PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=11)
ax.set_title(f'Optimal Clustering: {best_overall["algo"]} K={best_overall["k"]}\n'
             f'Silhouette={best_overall["sil"]:.4f}  |  {len(selected_features)} optimized features',
             fontsize=12)
ax.legend(loc='upper right', fontsize=9)
ax.grid(alpha=0.2)
plt.tight_layout()
fig.savefig(os.path.join(CONFIG['figures_dir'], '05_optimal_clustering.png'), dpi=150)
plt.close(fig)
print("      ✓ 05_optimal_clustering.png")

# ─────────────────────────────────────────────────────────────
# 图6: 聚类结果对比（新 vs 旧）
# ─────────────────────────────────────────────────────────────
print("   📊 Figure 6: Cluster Comparison...")
if orig_labels is not None and len(orig_labels) == len(opt_labels):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, lbl_arr, title in [
        (axes[0], orig_labels, 'Original Clustering'),
        (axes[1], opt_labels,  f'Optimized ({best_overall["algo"]} K={best_overall["k"]})'),
    ]:
        sil_val = silhouette_score(X_sel_scaled, lbl_arr,
                                   sample_size=min(3000, len(lbl_arr)),
                                   random_state=CONFIG['seed'])
        for c in sorted(np.unique(lbl_arr)):
            mask = lbl_arr == c
            ax.scatter(X_pca2[mask, 0], X_pca2[mask, 1],
                       c=palette[c % len(palette)], label=f'C{c}',
                       alpha=0.5, s=18, edgecolors='none')
        ax.set_title(f'{title}\nSilhouette={sil_val:.4f}', fontsize=12)
        ax.set_xlabel('PC1', fontsize=10)
        ax.set_ylabel('PC2', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2)

    plt.suptitle('Clustering Comparison: Original vs Optimized', fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(CONFIG['figures_dir'], '06_cluster_comparison.png'), dpi=150)
    plt.close(fig)
    print("      ✓ 06_cluster_comparison.png")
else:
    # 当没有原始聚类时，用簇内特征分布作为对比
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    top6_feats = selected_features[:6]

    for i, feat in enumerate(top6_feats):
        ax = axes[i]
        for c in unique_clusters:
            mask = opt_labels == c
            vals = X_sel.iloc[mask][feat].dropna().values
            ax.hist(vals, bins=30, alpha=0.5, label=f'C{c}',
                    color=palette[c % len(palette)], density=True)
        ax.set_title(feat, fontsize=10)
        ax.set_xlabel('Value', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.legend(fontsize=8)

    plt.suptitle(f'Feature Distributions by Cluster\n'
                 f'{best_overall["algo"]} K={best_overall["k"]} | '
                 f'Sil={best_overall["sil"]:.4f}', fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(CONFIG['figures_dir'], '06_cluster_comparison.png'), dpi=150)
    plt.close(fig)
    print("      ✓ 06_cluster_comparison.png (feature distributions)")

# ============================================================
# STEP 9: 保存结果文件
# ============================================================
print(f"\n{'='*80}")
print("【STEP 9】Saving Output Files")
print("=" * 80)

# ── 9.1 优化特征 CSV ──
df_opt_final = df_opt[['vehicle_id'] + selected_features].copy()
df_opt_final['cluster'] = opt_labels
feat_path = os.path.join(CONFIG['results_dir'], 'features_optimized_final.csv')
df_opt_final.to_csv(feat_path, index=False)
print(f"   ✓ features_optimized_final.csv  ({len(df_opt_final):,} vehicles × {len(selected_features)} features)")

# ── 9.2 聚类结果 CSV ──
df_cluster_out = df_opt[['vehicle_id']].copy()
df_cluster_out['cluster'] = opt_labels
df_cluster_out['cluster_label'] = [cluster_labels_map.get(c, f'C{c}') for c in opt_labels]
clust_path = os.path.join(CONFIG['results_dir'], 'vehicle_clustering_optimized.csv')
df_cluster_out.to_csv(clust_path, index=False)
print(f"   ✓ vehicle_clustering_optimized.csv")

# ── 9.3 特征评分汇总 JSON ──
scores_summary = {
    'feature_fisher_scores': fisher_df.set_index('feature')['fisher_score'].to_dict(),
    'selected_features': selected_features,
    'n_selected': len(selected_features),
    'corr_threshold': CONFIG['corr_threshold'],
}
scores_path = os.path.join(CONFIG['results_dir'], 'feature_scores_summary.json')
with open(scores_path, 'w') as f:
    json.dump(scores_summary, f, indent=2)
print(f"   ✓ feature_scores_summary.json")

# ── 9.4 算法对比 JSON ──
algo_json = {}
for algo_name in comparison_results:
    algo_json[algo_name] = {}
    for k, res in comparison_results[algo_name].items():
        algo_json[algo_name][str(k)] = {
            kk: vv for kk, vv in res.items() if kk != 'labels'
        }
algo_path = os.path.join(CONFIG['results_dir'], 'algorithm_comparison_fast.json')
with open(algo_path, 'w') as f:
    json.dump(algo_json, f, indent=2)
print(f"   ✓ algorithm_comparison_fast.json")

# ── 9.5 诊断报告 TXT ──
report_path = os.path.join(CONFIG['results_dir'], 'diagnostic_report_fast.txt')

# 原方案 Silhouette（如有）
orig_sil_str = 'N/A'
if orig_labels is not None and len(orig_labels) == len(opt_labels):
    try:
        orig_sil = silhouette_score(X_sel_scaled, orig_labels,
                                    sample_size=min(3000, len(orig_labels)),
                                    random_state=CONFIG['seed'])
        orig_sil_str = f'{orig_sil:.4f}'
    except Exception:
        pass

with open(report_path, 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("  FAST FEATURE DIAGNOSIS AND OPTIMIZATION REPORT\n")
    f.write("  快速特征诊断与优化报告\n")
    f.write("=" * 70 + "\n\n")

    f.write("─" * 70 + "\n")
    f.write("1. FEATURE DIAGNOSIS SUMMARY | 特征诊断总结\n")
    f.write("─" * 70 + "\n")
    f.write(f"  Total candidate features:  {total_opt_features}\n")
    f.write(f"  PCA 90% variance needs:    {n_components_90} components\n")
    f.write(f"  PCA 95% variance needs:    {n_components_95} components\n\n")
    f.write("  Top 10 features (Fisher Score):\n")
    for i, row in fisher_df.head(10).iterrows():
        marker = '✓' if row['feature'] in selected_features else ' '
        f.write(f"    {marker} {i+1:2d}. {row['feature']:35s} {row['fisher_score']:.4f}\n")
    f.write("\n")

    f.write("─" * 70 + "\n")
    f.write("2. OPTIMAL FEATURE SUBSET | 最优特征子集\n")
    f.write("─" * 70 + "\n")
    f.write(f"  Selected {len(selected_features)} features "
            f"(correlation threshold: {CONFIG['corr_threshold']}):\n\n")
    for i, feat in enumerate(selected_features, 1):
        score = fisher_df.loc[fisher_df['feature'] == feat, 'fisher_score'].values[0]
        f.write(f"    {i:2d}. {feat:35s} Fisher={score:.4f}\n")
    f.write("\n")

    f.write("─" * 70 + "\n")
    f.write("3. ALGORITHM COMPARISON | 算法对比结果\n")
    f.write("─" * 70 + "\n")
    f.write(f"  {'Algorithm':>10}  {'K':>4}  {'Silhouette':>12}  {'DB':>10}  {'CH':>12}\n")
    f.write(f"  {'─'*10}  {'─'*4}  {'─'*12}  {'─'*10}  {'─'*12}\n")
    for algo_name in ['GMM', 'KMeans']:
        for k in CONFIG['k_range']:
            res = comparison_results[algo_name].get(k)
            if res:
                marker = ' ← BEST' if (algo_name == best_overall['algo'] and
                                        k == best_overall['k']) else ''
                f.write(f"  {algo_name:>10}  {k:>4}  "
                        f"{res['silhouette']:>12.4f}  "
                        f"{res['davies_bouldin']:>10.4f}  "
                        f"{res['calinski_harabasz']:>12.1f}"
                        f"{marker}\n")
    f.write("\n")

    f.write("─" * 70 + "\n")
    f.write("4. OPTIMAL CLUSTERING SCHEME | 最优聚类方案\n")
    f.write("─" * 70 + "\n")
    f.write(f"  Algorithm:         {best_overall['algo']}\n")
    f.write(f"  K (clusters):      {best_overall['k']}\n")
    f.write(f"  Silhouette Score:  {best_overall['sil']:.4f}\n\n")
    f.write("  Cluster breakdown:\n")
    for c, st in cluster_stats.items():
        f.write(f"    C{c} ({cluster_labels_map.get(c, '')}): "
                f"{st['size']:,} vehicles ({st['pct']:.1f}%)\n")
    f.write("\n")

    f.write("─" * 70 + "\n")
    f.write("5. IMPROVEMENT vs ORIGINAL | 改进效果对比\n")
    f.write("─" * 70 + "\n")
    f.write(f"  Original Silhouette:   {orig_sil_str}\n")
    f.write(f"  Optimized Silhouette:  {best_overall['sil']:.4f}\n")
    f.write(f"  Feature reduction:     {total_opt_features} → {len(selected_features)}\n\n")

    f.write("─" * 70 + "\n")
    f.write("6. RECOMMENDATIONS | 后续建议\n")
    f.write("─" * 70 + "\n")
    f.write("  1. Use features_optimized_final.csv for downstream analysis\n")
    f.write("  2. Selected features capture transition, behavior, and evolution\n")
    f.write(f"  3. {best_overall['algo']} K={best_overall['k']} is the recommended scheme\n")
    f.write("  4. Consider retraining with these features for coupling analysis\n")
    f.write("  5. Cluster labels can be refined with domain knowledge\n")

print(f"   ✓ diagnostic_report_fast.txt")

# ============================================================
# 完成
# ============================================================
print("\n" + "=" * 80)
print("✅ STEP 11 COMPLETE!")
print("=" * 80)

print(f"""
Summary:
  - Vehicles:          {len(df_opt_final):,}
  - Input features:    {total_opt_features}
  - Selected features: {len(selected_features)}
  - Best algorithm:    {best_overall['algo']}
  - Best K:            {best_overall['k']}
  - Silhouette Score:  {best_overall['sil']:.4f}

Cluster Breakdown:""")
for c, st in cluster_stats.items():
    lbl = cluster_labels_map.get(c, '')
    print(f"  C{c} ({lbl:30s}): {st['size']:,} vehicles ({st['pct']:.1f}%)")

print(f"""
Output Files (vehicle_clustering/results/):
  ├── diagnostic_report_fast.txt
  ├── features_optimized_final.csv
  ├── feature_scores_summary.json
  ├── algorithm_comparison_fast.json
  ├── vehicle_clustering_optimized.csv
  └── figures_diagnosis_fast/
      ├── 01_feature_importance.png
      ├── 02_feature_correlation.png
      ├── 03_pca_variance.png
      ├── 04_algorithm_comparison.png
      ├── 05_optimal_clustering.png
      └── 06_cluster_comparison.png
""")
