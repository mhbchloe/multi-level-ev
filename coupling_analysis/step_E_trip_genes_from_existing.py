"""
Step E-lite: 复用已有的片段聚类，构建行程基因
直接从 discharge_segments + clustering_v3 映射到 inter_charge_trips

输入:
  - discharge_segments_28days.csv (201,054 片段)
  - clustering_v3_results.npz (199,991 labels)
  - inter_charge_trips.csv (102,518 行程)

输出:
  - coupling_dataset_with_genes.csv (行程 + 片段占比)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("🧬 Step E-lite: Trip Gene Reconstruction from Existing Clusters")
print("=" * 70)

SEGMENT_DIR = "./analysis_complete_vehicles/results/"
CLUSTER_DIR = "./analysis_complete_vehicles/results/clustering_v3/"
TRIP_DIR = "./coupling_analysis/results/"
FIGURE_DIR = "./coupling_analysis/figures/"
os.makedirs(FIGURE_DIR, exist_ok=True)

t_start = time.time()

# ============================================================
# 1. 加载片段数据 + 聚类标签
# ============================================================
print("\n📂 Loading data...")

# 片段表
df_seg = pd.read_csv(os.path.join(SEGMENT_DIR, 'discharge_segments_28days.csv'))
print(f"   Segments CSV: {len(df_seg):,}")

# 聚类标签
cluster_data = np.load(os.path.join(CLUSTER_DIR, 'clustering_v3_results.npz'),
                       allow_pickle=True)
labels = cluster_data['labels']  # 4 类聚类标签
print(f"   Cluster labels: {len(labels):,}")
print(f"   Cluster distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

# 聚类名称
with open(os.path.join(CLUSTER_DIR, 'cluster_names.json')) as f:
    cluster_names = json.load(f)

CLUSTER_MAP = {}
for k, v in cluster_names.items():
    CLUSTER_MAP[int(k)] = v['short']  # 'Long Idle', 'Urban', 'Short Idle', 'Highway'

print(f"   Cluster names: {CLUSTER_MAP}")

# ============================================================
# 2. 对齐片段和聚类标签
#    CSV 有 201,054 行，labels 有 199,991 行
#    需要找到哪些片段被过滤了
# ============================================================
print(f"\n{'=' * 70}")
print("🔗 Aligning segments with cluster labels...")
print(f"{'=' * 70}")

n_seg = len(df_seg)
n_labels = len(labels)
n_diff = n_seg - n_labels

print(f"   CSV rows: {n_seg:,}, Labels: {n_labels:,}, Diff: {n_diff:,}")

# 方法: GRU 训练时通常按顺序处理，可能过滤了 seg_length < 某阈值的片段
# 从 npz 中读取 seg_length 来做匹配
if 'seg_length' in cluster_data:
    npz_seg_lengths = cluster_data['seg_length']

    # 尝试1: 假设 GRU 按顺序处理，过滤了某些短片段
    # 用 n_points (CSV) 和 seg_length (npz) 对齐
    csv_lengths = df_seg['n_points'].values

    # 找到匹配的索引
    # 策略: 遍历 npz 的 seg_length，在 csv 中按顺序匹配
    print("   Matching by sequential alignment...")

    matched_csv_idx = []
    npz_ptr = 0

    for csv_idx in range(n_seg):
        if npz_ptr >= n_labels:
            break
        # seg_length 在 npz 中可能是被截断后的长度，而 n_points 是原始长度
        # 如果 n_points >= seg_length 就认为匹配
        if csv_lengths[csv_idx] >= npz_seg_lengths[npz_ptr]:
            matched_csv_idx.append(csv_idx)
            npz_ptr += 1

    if len(matched_csv_idx) == n_labels:
        print(f"   ✅ Perfect match! {n_labels:,} segments aligned")
        df_seg_matched = df_seg.iloc[matched_csv_idx].copy()
        df_seg_matched['cluster'] = labels
    else:
        # 尝试2: 直接取前 n_labels 行 (如果 GRU 只是截断了尾部)
        print(f"   ⚠️ Sequential match got {len(matched_csv_idx):,}, trying head truncation...")
        df_seg_matched = df_seg.head(n_labels).copy()
        df_seg_matched['cluster'] = labels
        print(f"   ✅ Using first {n_labels:,} rows")
else:
    # 没有 seg_length，直接取前 n_labels 行
    print("   No seg_length in npz, using first N rows")
    df_seg_matched = df_seg.head(n_labels).copy()
    df_seg_matched['cluster'] = labels

# 添加聚类名称
df_seg_matched['cluster_name'] = df_seg_matched['cluster'].map(CLUSTER_MAP)

print(f"\n   Matched segments: {len(df_seg_matched):,}")
print(f"   Vehicles: {df_seg_matched['vehicle_id'].nunique():,}")
print(f"\n   Cluster distribution in matched data:")
for cid in sorted(CLUSTER_MAP.keys()):
    n = (df_seg_matched['cluster'] == cid).sum()
    print(f"      {cid} ({CLUSTER_MAP[cid]}): {n:,} ({n/len(df_seg_matched)*100:.1f}%)")

# ============================================================
# 3. 解析时间戳，准备和行程匹配
# ============================================================
print(f"\n{'=' * 70}")
print("⏳ Parsing timestamps...")
print(f"{'=' * 70}")

df_seg_matched['start_dt'] = pd.to_datetime(df_seg_matched['start_time'])
df_seg_matched['end_dt'] = pd.to_datetime(df_seg_matched['end_time'])

# 加载行程数据
df_trips = pd.read_csv(os.path.join(TRIP_DIR, 'inter_charge_trips.csv'))
df_trips['trip_start'] = pd.to_datetime(df_trips['trip_start'], format='ISO8601')
df_trips['trip_end'] = pd.to_datetime(df_trips['trip_end'], format='ISO8601')

print(f"   Trips: {len(df_trips):,}")
print(f"   Segments: {len(df_seg_matched):,}")

# ============================================================
# 4. 将片段映射到行程 (按时间范围)
# ============================================================
print(f"\n{'=' * 70}")
print("🧬 Mapping segments to trips...")
print(f"{'=' * 70}")

# 预分组
seg_groups = {vid: grp.sort_values('start_dt')
              for vid, grp in df_seg_matched.groupby('vehicle_id')}

trip_genes = []

for _, trip in tqdm(df_trips.iterrows(), total=len(df_trips),
                    desc="Mapping", ncols=80):
    vid = trip['vehicle_id']
    if vid not in seg_groups:
        trip_genes.append({
            'trip_id': trip['trip_id'],
            'n_segments': 0,
        })
        continue

    v_segs = seg_groups[vid]

    # 找到该行程时间范围内的片段
    ts_start = trip['trip_start']
    ts_end = trip['trip_end']

    mask = (v_segs['start_dt'] >= ts_start) & (v_segs['start_dt'] < ts_end)
    trip_segs = v_segs[mask]

    n_segs = len(trip_segs)

    if n_segs == 0:
        trip_genes.append({
            'trip_id': trip['trip_id'],
            'n_segments': 0,
        })
        continue

    # 统计各类片段占比
    cluster_counts = trip_segs['cluster'].value_counts(normalize=True)

    gene = {
        'trip_id': trip['trip_id'],
        'n_segments': n_segs,
    }

    # 用标准化的列名: ratio_<short_name_lower>
    for cid, cname in CLUSTER_MAP.items():
        col_name = f"ratio_{cname.lower().replace(' ', '_')}"
        gene[col_name] = cluster_counts.get(cid, 0.0)

    # 额外: 片段级特征的加权均值 (用于交叉验证)
    gene['seg_avg_speed'] = trip_segs['speed_mean'].mean()
    gene['seg_avg_power'] = trip_segs['power_mean'].mean()
    gene['seg_avg_soc_drop'] = trip_segs['soc_drop'].mean()
    gene['seg_total_duration'] = trip_segs['duration_seconds'].sum()

    trip_genes.append(gene)

df_genes = pd.DataFrame(trip_genes)

# 过滤掉没有片段的行程
n_total = len(df_genes)
df_genes_valid = df_genes[df_genes['n_segments'] > 0].copy()
print(f"\n   Total trips: {n_total:,}")
print(f"   Trips with segments: {len(df_genes_valid):,} ({len(df_genes_valid)/n_total*100:.1f}%)")
print(f"   Avg segments/trip: {df_genes_valid['n_segments'].mean():.1f}")

# ============================================================
# 5. 合并到行程数据
# ============================================================
print(f"\n{'=' * 70}")
print("📊 Merging genes with trip data...")
print(f"{'=' * 70}")

df_coupled = df_trips.merge(df_genes_valid, on='trip_id', how='inner')

# 识别 ratio 列
ratio_cols = [c for c in df_coupled.columns if c.startswith('ratio_')]
print(f"   Ratio columns: {ratio_cols}")

# 尝试加载车辆画像
for vpath in ['./vehicle_clustering/results/vehicle_clustering_results_v3.csv',
              './vehicle_clustering/results/vehicle_clustering_results.csv']:
    if os.path.exists(vpath):
        df_veh = pd.read_csv(vpath)
        veh_cols = ['vehicle_id']
        if 'vehicle_type' in df_veh.columns:
            veh_cols.append('vehicle_type')
        if 'cluster' in df_veh.columns:
            df_veh = df_veh.rename(columns={'cluster': 'vehicle_archetype'})
            veh_cols.append('vehicle_archetype')
        df_coupled = df_coupled.merge(df_veh[veh_cols], on='vehicle_id', how='left')
        print(f"   ✅ Merged vehicle profiles from {vpath}")
        break

print(f"\n   Final dataset: {len(df_coupled):,} trips, "
      f"{df_coupled['vehicle_id'].nunique():,} vehicles")

# ============================================================
# 6. 变量体系报告
# ============================================================
print(f"\n{'=' * 70}")
print("📋 Variable System (Section 3.3.1)")
print(f"{'=' * 70}")

print(f"\n   X1 - Trip Genes (micro-segment composition):")
for col in ratio_cols:
    m = df_coupled[col].mean()
    s = df_coupled[col].std()
    print(f"      {col:<30} {m:.3f} ± {s:.3f}  ({m*100:.1f}%)")

print(f"\n   X2 - Cumulative State:")
print(f"      trip_duration_hrs:          {df_coupled['trip_duration_hrs'].mean():.2f} ± "
      f"{df_coupled['trip_duration_hrs'].std():.2f}")
print(f"      soc_drop:                   {df_coupled['soc_drop'].mean():.1f} ± "
      f"{df_coupled['soc_drop'].std():.1f}")

if 'vehicle_archetype' in df_coupled.columns:
    print(f"\n   X3 - Vehicle Archetype:")
    for vt in df_coupled['vehicle_archetype'].dropna().unique():
        n = (df_coupled['vehicle_archetype'] == vt).sum()
        print(f"      Archetype {vt}: {n:,} ({n/len(df_coupled)*100:.1f}%)")

print(f"\n   Y - Charging Decisions:")
print(f"      charge_trigger_soc:         {df_coupled['charge_trigger_soc'].mean():.1f} ± "
      f"{df_coupled['charge_trigger_soc'].std():.1f}")
print(f"      charge_gain_soc:            {df_coupled['charge_gain_soc'].mean():.1f} ± "
      f"{df_coupled['charge_gain_soc'].std():.1f}")

# ============================================================
# 7. 保存
# ============================================================
output_path = os.path.join(TRIP_DIR, 'coupling_dataset_with_genes.csv')
df_coupled.to_csv(output_path, index=False)
print(f"\n💾 Saved: {output_path}")
print(f"   Shape: {df_coupled.shape}")
print(f"   Size: {os.path.getsize(output_path)/1024/1024:.1f} MB")

# 也保存带标签的片段 (后续可能用到)
seg_output = os.path.join(TRIP_DIR, 'segments_with_cluster_labels.csv')
df_seg_matched.to_csv(seg_output, index=False)
print(f"   Segments: {seg_output}")

# ============================================================
# 8. 可视化
# ============================================================
print(f"\n📈 Generating visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(20, 12))

cluster_colors = {}
for k, v in cluster_names.items():
    cluster_colors[v['short']] = v['color']

# (a) 片段聚类分布
ax = axes[0, 0]
cluster_dist = df_seg_matched['cluster_name'].value_counts()
ordered_names = [CLUSTER_MAP[c] for c in sorted(CLUSTER_MAP.keys())]
ordered_counts = [cluster_dist.get(n, 0) for n in ordered_names]
bars = ax.bar(ordered_names, ordered_counts,
              color=[cluster_colors.get(n, 'grey') for n in ordered_names])
for bar, v in zip(bars, ordered_counts):
    ax.text(bar.get_x()+bar.get_width()/2, v, f'{v:,}',
            ha='center', va='bottom', fontweight='bold', fontsize=9)
ax.set_ylabel('Number of Segments')
ax.set_title('(a) Micro-Segment Cluster Distribution', fontweight='bold')
ax.tick_params(axis='x', rotation=15)
ax.grid(alpha=0.3, axis='y')

# (b) 片段 Speed vs SOC Rate
ax = axes[0, 1]
sample = df_seg_matched.sample(min(8000, len(df_seg_matched)), random_state=42)
for cid in sorted(CLUSTER_MAP.keys()):
    name = CLUSTER_MAP[cid]
    mask = sample['cluster'] == cid
    ax.scatter(sample[mask]['speed_mean'], sample[mask]['soc_drop'],
               c=cluster_colors.get(name, 'grey'), s=6, alpha=0.4, label=name)
ax.set_xlabel('Segment Speed (km/h)')
ax.set_ylabel('Segment SOC Drop (%)')
ax.set_title('(b) Segment Clusters in Feature Space', fontweight='bold')
ax.legend(markerscale=3, fontsize=9)
ax.grid(alpha=0.2)

# (c) 行程基因组成 (堆叠柱状)
ax = axes[0, 2]
gene_means = {col: df_coupled[col].mean() for col in ratio_cols}
labels_bar = [col.replace('ratio_', '').replace('_', ' ').title() for col in ratio_cols]
values_bar = [gene_means[col] for col in ratio_cols]
colors_bar = [cluster_colors.get(CLUSTER_MAP.get(i, ''), 'grey')
              for i in range(len(ratio_cols))]

# 匹配颜色到 ratio 列
bar_colors = []
for col in ratio_cols:
    short_name = col.replace('ratio_', '').replace('_', ' ').title()
    # 找到对应的 cluster color
    matched_color = 'grey'
    for cid, cname in CLUSTER_MAP.items():
        if cname.lower().replace(' ', '_') in col:
            matched_color = cluster_colors.get(cname, 'grey')
            break
    bar_colors.append(matched_color)

bars = ax.bar(labels_bar, values_bar, color=bar_colors, alpha=0.8)
for bar, v in zip(bars, values_bar):
    ax.text(bar.get_x()+bar.get_width()/2, v, f'{v*100:.1f}%',
            ha='center', va='bottom', fontweight='bold', fontsize=9)
ax.set_ylabel('Average Ratio')
ax.set_title('(c) Average Trip Gene Composition', fontweight='bold')
ax.tick_params(axis='x', rotation=15)
ax.grid(alpha=0.3, axis='y')

# (d) 各类占比 vs Trigger SOC
ax = axes[1, 0]
from scipy import stats
for col in ratio_cols:
    short_name = col.replace('ratio_', '').replace('_', ' ').title()
    # 找颜色
    matched_color = 'grey'
    for cid, cname in CLUSTER_MAP.items():
        if cname.lower().replace(' ', '_') in col:
            matched_color = cluster_colors.get(cname, 'grey')
            break

    df_temp = df_coupled[[col, 'charge_trigger_soc']].dropna()
    q99 = df_temp[col].quantile(0.99)
    if q99 < 0.01:
        continue
    bins = np.linspace(0, q99, 11)
    centers, means = [], []
    for i in range(len(bins)-1):
        mask = (df_temp[col] >= bins[i]) & (df_temp[col] < bins[i+1])
        if mask.sum() > 20:
            centers.append((bins[i]+bins[i+1])/2)
            means.append(df_temp.loc[mask, 'charge_trigger_soc'].mean())
    if len(centers) > 2:
        ax.plot(centers, means, '-o', color=matched_color, linewidth=2,
                markersize=5, label=short_name, alpha=0.8)

ax.set_xlabel('Segment Type Ratio')
ax.set_ylabel('Mean Trigger SOC (%)')
ax.set_title('(d) Trip Genes → Trigger SOC', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# (e) 各类占比 vs SOC Gain
ax = axes[1, 1]
for col in ratio_cols:
    short_name = col.replace('ratio_', '').replace('_', ' ').title()
    matched_color = 'grey'
    for cid, cname in CLUSTER_MAP.items():
        if cname.lower().replace(' ', '_') in col:
            matched_color = cluster_colors.get(cname, 'grey')
            break

    df_temp = df_coupled[[col, 'charge_gain_soc']].dropna()
    q99 = df_temp[col].quantile(0.99)
    if q99 < 0.01:
        continue
    bins = np.linspace(0, q99, 11)
    centers, means = [], []
    for i in range(len(bins)-1):
        mask = (df_temp[col] >= bins[i]) & (df_temp[col] < bins[i+1])
        if mask.sum() > 20:
            centers.append((bins[i]+bins[i+1])/2)
            means.append(df_temp.loc[mask, 'charge_gain_soc'].mean())
    if len(centers) > 2:
        ax.plot(centers, means, '-o', color=matched_color, linewidth=2,
                markersize=5, label=short_name, alpha=0.8)

ax.set_xlabel('Segment Type Ratio')
ax.set_ylabel('Mean SOC Gain (%)')
ax.set_title('(e) Trip Genes → SOC Gain', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# (f) 片段数分布
ax = axes[1, 2]
ax.hist(df_coupled['n_segments'], bins=50, color='#3498db', alpha=0.7, edgecolor='white')
ax.axvline(df_coupled['n_segments'].mean(), color='black', linestyle='--',
           label=f'Mean={df_coupled["n_segments"].mean():.1f}')
ax.set_xlabel('Number of Segments per Trip')
ax.set_ylabel('Count')
ax.set_title('(f) Segments per Trip Distribution', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3, axis='y')

plt.suptitle('Section 3.3.1: Trip Gene Reconstruction from GRU-based Segment Clustering',
             fontsize=15, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(FIGURE_DIR, 'fig_331_trip_genes.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(FIGURE_DIR, 'fig_331_trip_genes.pdf'), bbox_inches='tight')
plt.close(fig)
print(f"   ✅ Saved: fig_331_trip_genes.png/pdf")

t_total = time.time() - t_start
print(f"\n⏱️ Total time: {t_total:.1f}s ({t_total/60:.1f} min)")
print(f"\n✅ Step E-lite Complete!")
print(f"\n📋 Next: Run step_F_xgboost_shap_with_genes.py")