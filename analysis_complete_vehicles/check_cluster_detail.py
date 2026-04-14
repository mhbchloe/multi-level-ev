import numpy as np
import json

# 加载
data = np.load('./analysis_complete_vehicles/results/clustering_v2/physical_features.npz')
km_labels = data['kmeans_labels']
seg_types = data['seg_types']

with open('./analysis_complete_vehicles/results/clustering_v2/cluster_feature_profiles.json', 'r') as f:
    profiles = json.load(f)

stats = profiles['cluster_stats']

print("=" * 90)
print("各簇详细统计")
print("=" * 90)

# 关键特征
keys = [
    ('size',           '样本数',       ''),
    ('driving_pct',    '行驶占比',     '%'),
    ('speed_mean_kmh', '平均速度',     'km/h'),
    ('speed_max_kmh',  '最大速度',     'km/h'),
    ('acc_max',        '最大|加速度|', 'm/s²'),
    ('heading_std',    '航向波动',     '°'),
    ('soc_delta',      'SOC消耗',      '%'),
    ('power_max_kw',   '最大功率',     'kW'),
    ('seg_length',     '片段长度',     'steps'),
]

header = f"{'指标':>14} {'单位':>6}"
for c in ['0','1','2','3']:
    header += f"  {'C'+c:>12}"
print(header)
print("-" * 90)

for key, name, unit in keys:
    line = f"{name:>14} {unit:>6}"
    for c in ['0','1','2','3']:
        v = stats[c][key]
        if isinstance(v, int) or (isinstance(v, float) and abs(v) > 100):
            line += f"  {v:>12.1f}"
        else:
            line += f"  {v:>12.4f}"
    print(line)

# 各簇样本的中位数和分位数
print("\n\n" + "=" * 90)
print("各簇特征分位数（检查极端值）")
print("=" * 90)

feat_keys = ['speed_mean_kmh', 'acc_max_abs', 'soc_delta', 'power_max_kw', 'seg_length']
feat_names = ['速度(km/h)', '|加速度|max(m/s²)', 'SOC消耗(%)', '功率max(kW)', '片段长度']

for fk, fn in zip(feat_keys, feat_names):
    print(f"\n  {fn}:")
    for c in sorted(np.unique(km_labels)):
        mask = km_labels == c
        vals = data[fk][mask]
        print(f"    C{c} (n={mask.sum():>6}): "
              f"p5={np.percentile(vals,5):>8.2f}  "
              f"p25={np.percentile(vals,25):>8.2f}  "
              f"median={np.median(vals):>8.2f}  "
              f"p75={np.percentile(vals,75):>8.2f}  "
              f"p95={np.percentile(vals,95):>8.2f}  "
              f"max={np.max(vals):>8.2f}")