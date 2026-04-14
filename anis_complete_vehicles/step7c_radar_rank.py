"""
雷达图：基于排名的归一化，只表示相对高低
"""

import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import rankdata

warnings.filterwarnings('ignore')

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
rcParams['font.size'] = 11
rcParams['axes.unicode_minus'] = False
rcParams['figure.dpi'] = 150

CONFIG = {
    'result_path': './analysis_complete_vehicles/results/clustering_v3/clustering_v3_results.npz',
    'save_dir':    './analysis_complete_vehicles/results/clustering_v3',
    'seed':        42,
}

CLUSTER_META = {
    0: {'short': 'Long Idle',   'color': '#5B9BD5'},
    1: {'short': 'Urban',       'color': '#70AD47'},
    2: {'short': 'Short Idle',  'color': '#C0504D'},
    3: {'short': 'Highway',     'color': '#FFC000'},
}


def main():
    cfg = CONFIG
    data = np.load(cfg['result_path'])
    labels = data['labels']
    seg_types = data['seg_types']

    phys = {}
    for key in ['avg_speed', 'avg_speed_mov', 'speed_std', 'speed_max',
                'acc_std_mov', 'heading_change', 'idle_ratio',
                'soc_rate', 'power_mean', 'seg_length']:
        phys[key] = data[key]

    unique = sorted(np.unique(labels))

    # 各簇均值
    stats = {}
    for c in unique:
        mask = labels == c
        stats[c] = {}
        for k in phys:
            stats[c][k] = float(np.mean(phys[k][mask]))

    # ============================================================
    # 雷达图特征（统一用简洁的标签）
    # ============================================================
    radar_feats = [
        ('avg_speed_mov', 'Speed'),
        ('speed_std',     'Speed\nVariation'),
        ('acc_std_mov',   'Acceleration\nVariation'),
        ('heading_change','Turning\nIntensity'),
        ('soc_rate',      'Energy\nConsumption'),
        ('idle_ratio',    'Idle\nRatio'),
        ('seg_length',    'Segment\nDuration'),
    ]

    feat_keys   = [r[0] for r in radar_feats]
    feat_labels = [r[1] for r in radar_feats]
    n_feats     = len(feat_keys)

    # ============================================================
    # 排名归一化：每个特征，4个簇排名 → 映射到 [0.15, 1.0]
    # 排名1(最小)→0.15, 排名4(最大)→1.0
    # 这样最低的也不会贴在圆心
    # ============================================================
    rank_norm = {}
    for c in unique:
        rank_norm[c] = []

    for key in feat_keys:
        vals = [stats[c][key] for c in unique]
        ranks = rankdata(vals, method='average')  # [1,2,3,4]
        # 映射到 [0.15, 1.0]
        for i, c in enumerate(unique):
            normalized = 0.15 + (ranks[i] - 1) / (len(unique) - 1) * 0.85
            rank_norm[c].append(normalized)

    # ============================================================
    # 绘图
    # ============================================================
    angles = np.linspace(0, 2 * np.pi, n_feats, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    for c in unique:
        vals = rank_norm[c] + rank_norm[c][:1]
        meta = CLUSTER_META[c]
        ax.plot(angles, vals, 'o-', linewidth=2.5, markersize=9,
                label=meta['short'], color=meta['color'], zorder=3)
        ax.fill(angles, vals, alpha=0.08, color=meta['color'])

    # 刻度标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feat_labels, fontsize=11, fontweight='bold')

    # Y 轴：Low / Medium / High
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0.15, 0.43, 0.72, 1.0])
    ax.set_yticklabels(['Low', '', '', 'High'], fontsize=9, color='gray')

    # 添加 Low/High 标注
    ax.annotate('Low', xy=(0, 0.15), fontsize=8, color='gray', ha='center')
    ax.annotate('High', xy=(0, 1.05), fontsize=8, color='gray', ha='center')

    ax.legend(loc='upper right', bbox_to_anchor=(1.32, 1.1),
              fontsize=11, framealpha=0.9)

    ax.set_title('Cluster Feature Profiles\n(Rank-based: Low → High)',
                 fontsize=14, fontweight='bold', pad=25)

    ax.spines['polar'].set_visible(False)
    ax.grid(True, alpha=0.25, linewidth=0.8)

    for fmt, dpi in [('.png', 300), ('.pdf', None)]:
        path = os.path.join(cfg['save_dir'], f'paper_radar_rank{fmt}')
        fig.savefig(path, dpi=dpi if dpi else None, bbox_inches='tight')
        print(f"   Saved: {path}")
    plt.close(fig)

    # ============================================================
    # 打印排名表（确认正确性）
    # ============================================================
    print(f"\n   📋 各簇特征排名 (1=最低, 4=最高):")
    hdr = f"   {'Feature':>22}"
    for c in unique:
        hdr += f"  {CLUSTER_META[c]['short']:>12}"
    print(hdr)
    print(f"   {'─'*22}" + "  ".join(['─'*12]*len(unique)))

    for i, (key, label) in enumerate(radar_feats):
        vals = [stats[c][key] for c in unique]
        ranks = rankdata(vals, method='average')
        line = f"   {label.replace(chr(10),' '):>22}"
        for j, c in enumerate(unique):
            # 显示：排名 (真实值)
            v = vals[j]
            if abs(v) >= 100:
                line += f"  #{int(ranks[j])} ({v:>.0f})"
            elif abs(v) >= 1:
                line += f"  #{int(ranks[j])} ({v:>.2f})"
            elif abs(v) >= 0.001:
                line += f"  #{int(ranks[j])} ({v:>.4f})"
            else:
                line += f"  #{int(ranks[j])} ({v:>.5f})"
        print(line)

    print(f"\n   ✅ 完成！")
    print(f"   关键：图中只表示相对高低（排名），不受绝对量级影响")
    print(f"   Y轴含义：靠外=该特征相对更高，靠内=相对更低\n")


if __name__ == '__main__':
    main()