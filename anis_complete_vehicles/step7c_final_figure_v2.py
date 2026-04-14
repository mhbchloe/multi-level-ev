"""
Step 7c Final v2: 论文级聚类特征对比图
用分组柱状图替代雷达图，更清晰
"""

import numpy as np
import os
import json
import warnings
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as patheffects
from matplotlib import rcParams
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
rcParams['font.size'] = 11
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 13
rcParams['axes.titleweight'] = 'bold'
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['legend.fontsize'] = 9
rcParams['axes.unicode_minus'] = False
rcParams['figure.dpi'] = 150

CONFIG = {
    'result_path': './analysis_complete_vehicles/results/clustering_v3/clustering_v3_results.npz',
    'save_dir':    './analysis_complete_vehicles/results/clustering_v3',
    'seed':        42,
}

CLUSTER_META = {
    0: {'name': 'Long Stationary Idle',  'cn': '长时静态怠速', 'short': 'Long Idle',
        'color': '#5B9BD5'},
    1: {'name': 'Urban Cruising',        'cn': '城市中速行驶', 'short': 'Urban',
        'color': '#70AD47'},
    2: {'name': 'Short Active Idle',     'cn': '短时活跃怠速', 'short': 'Short Idle',
        'color': '#C0504D'},
    3: {'name': 'Highway Driving',       'cn': '高速持续行驶', 'short': 'Highway',
        'color': '#FFC000'},
}


def main():
    cfg = CONFIG
    os.makedirs(cfg['save_dir'], exist_ok=True)

    print("=" * 70)
    print("Paper Figures v2")
    print("=" * 70)

    data = np.load(cfg['result_path'])
    labels    = data['labels']
    seg_types = data['seg_types']
    z_pca     = data['z_pca']

    phys = {}
    for key in ['avg_speed', 'avg_speed_mov', 'speed_std', 'speed_max',
                'acc_std_mov', 'heading_change', 'idle_ratio',
                'soc_rate', 'power_mean', 'seg_length']:
        phys[key] = data[key]

    unique = sorted(np.unique(labels))
    n_total = len(labels)

    # 各簇统计
    stats = {}
    for c in unique:
        mask = labels == c
        n = mask.sum()
        stats[c] = {
            'size': int(n), 'pct': n / n_total * 100,
            'driving_pct': (seg_types[mask] == 0).sum() / n * 100,
        }
        for k in phys:
            vals = phys[k][mask]
            stats[c][f'{k}_mean'] = float(np.mean(vals))
            stats[c][f'{k}_med']  = float(np.median(vals))

    colors = {c: CLUSTER_META[c]['color'] for c in unique}
    shorts = {c: CLUSTER_META[c]['short'] for c in unique}

    # ================================================================
    # 图 1: 论文主图 2×3
    # ================================================================
    print("Figure 1: 2x3 main figure...")

    pca2 = PCA(n_components=2, random_state=cfg['seed'])
    z_2d = pca2.fit_transform(z_pca)
    ev = pca2.explained_variance_ratio_

    fig = plt.figure(figsize=(18, 11))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.32,
                           left=0.06, right=0.97, top=0.93, bottom=0.06)

    # (a) PCA
    ax = fig.add_subplot(gs[0, 0])
    np.random.seed(cfg['seed'])
    idx = np.random.choice(len(z_2d), min(10000, len(z_2d)), replace=False)
    for c in unique:
        m = labels[idx] == c
        ax.scatter(z_2d[idx][m, 0], z_2d[idx][m, 1],
                   c=colors[c], s=5, alpha=0.35, edgecolors='none',
                   label=f'{shorts[c]} (n={stats[c]["size"]:,})')
    ax.set_xlabel(f'PC1 ({ev[0]:.1%})')
    ax.set_ylabel(f'PC2 ({ev[1]:.1%})')
    ax.set_title('(a) Cluster Distribution in PCA Space')
    ax.legend(markerscale=5, fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.15)

    # (b) 行驶/怠速组成
    ax = fig.add_subplot(gs[0, 1])
    x = np.arange(len(unique))
    w = 0.6
    drv = [stats[c]['driving_pct'] for c in unique]
    idl = [100 - stats[c]['driving_pct'] for c in unique]
    b1 = ax.bar(x, drv, w, color='#4C72B0', label='Driving', edgecolor='white')
    b2 = ax.bar(x, idl, w, bottom=drv, color='#DD8452', label='Idle', edgecolor='white')
    for bar, p in zip(b1, drv):
        if p > 8:
            ax.text(bar.get_x() + w / 2, p / 2, f'{p:.0f}%',
                    ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    for bar, p, bot in zip(b2, idl, drv):
        if p > 8:
            ax.text(bar.get_x() + w / 2, bot + p / 2, f'{p:.0f}%',
                    ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    for i, c in enumerate(unique):
        ax.text(i, 104, f'n={stats[c]["size"]:,}', ha='center', fontsize=7, color='gray')
    ax.set_xticks(x)
    ax.set_xticklabels([shorts[c] for c in unique], fontsize=9)
    ax.set_ylabel('Composition (%)')
    ax.set_title('(b) Driving / Idle Composition')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 112)
    ax.grid(True, alpha=0.15, axis='y')

    # (c) 气泡图
    ax = fig.add_subplot(gs[0, 2])
    for c in unique:
        s = stats[c]
        bubble = s['size'] / n_total * 4000
        ax.scatter(s['avg_speed_mov_mean'], s['soc_rate_mean'],
                   s=bubble, c=colors[c], alpha=0.75,
                   edgecolors='#333', linewidths=1.5, zorder=5)
        ax.annotate(shorts[c],
                    (s['avg_speed_mov_mean'], s['soc_rate_mean']),
                    fontsize=9, fontweight='bold', ha='center', va='center',
                    color='white',
                    path_effects=[patheffects.withStroke(linewidth=2, foreground='#333')])
    ax.set_xlabel('Avg. Speed when Moving (km/h)')
    ax.set_ylabel('SOC Rate (%/min)')
    ax.set_title('(c) Speed vs Energy Consumption')
    ax.grid(True, alpha=0.15)

    # (d)(e)(f) 箱线图
    box_cfgs = [
        ('avg_speed_mov', '(d) Speed when Moving',        'Speed (km/h)'),
        ('acc_std_mov',   '(e) Acceleration Variation',    'Acc. Std (m/s\u00B2)'),
        ('soc_rate',      '(f) Energy Consumption Rate',   'SOC Rate (%/min)'),
    ]
    for gi, (feat, title, ylabel) in enumerate(box_cfgs):
        ax = fig.add_subplot(gs[1, gi])
        bd = [phys[feat][labels == c] for c in unique]
        bp = ax.boxplot(bd, labels=[shorts[c] for c in unique],
                        patch_artist=True, showfliers=False, widths=0.55,
                        medianprops=dict(color='#E67E22', linewidth=2))
        for patch, c in zip(bp['boxes'], unique):
            patch.set_facecolor(colors[c])
            patch.set_alpha(0.75)
            patch.set_edgecolor('#555')
        for el in ['whiskers', 'caps']:
            for it in bp[el]:
                it.set_color('#555'); it.set_linewidth(1.2)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.15, axis='y')
        ax.tick_params(axis='x', rotation=15)

    for fmt, dpi in [('.png', 300), ('.pdf', None)]:
        path = os.path.join(cfg['save_dir'], f'paper_main_figure{fmt}')
        fig.savefig(path, dpi=dpi if dpi else None, bbox_inches='tight')
        print(f"   Saved: {path}")
    plt.close(fig)

    # ================================================================
    # 图 2: 特征对比图（替代雷达图，横向分组柱状图 + 数值标注）
    # ================================================================
    print("Figure 2: Feature comparison (horizontal grouped bars)...")

    feat_cfgs = [
        ('avg_speed_mov_mean', 'Avg. Speed (moving)',     'km/h',  '.1f'),
        ('speed_max_mean',     'Max Speed',               'km/h',  '.1f'),
        ('acc_std_mov_mean',   'Acc. Variation (moving)', 'm/s²',  '.4f'),
        ('heading_change_mean','Total Heading Change',    '°',     '.0f'),
        ('soc_rate_mean',      'SOC Consumption Rate',    '%/min', '.2f'),
        ('idle_ratio_mean',    'Idle Ratio',              '',      '.2f'),
        ('seg_length_mean',    'Segment Duration',        'steps', '.0f'),
        ('power_mean_mean',    'Avg. Power',              'kW',    '.3f'),
    ]

    n_feats = len(feat_cfgs)
    n_clusters = len(unique)

    fig, ax = plt.subplots(figsize=(12, 8))

    y_positions = np.arange(n_feats)
    bar_height = 0.18
    offsets = np.linspace(-(n_clusters - 1) / 2 * bar_height,
                          (n_clusters - 1) / 2 * bar_height,
                          n_clusters)

    # 每个特征归一化到 [0, 1]（仅用于柱子长度）
    for fi, (key, label, unit, fmt) in enumerate(feat_cfgs):
        vals = [stats[c][key] for c in unique]
        vmax = max(abs(v) for v in vals)
        if vmax < 1e-10:
            vmax = 1.0

        for ci, c in enumerate(unique):
            v = vals[ci]
            bar_len = abs(v) / vmax  # 归一化长度
            y = y_positions[fi] + offsets[ci]

            bar = ax.barh(y, bar_len, height=bar_height,
                          color=colors[c], edgecolor='white', linewidth=0.5,
                          alpha=0.85)

            # 数值标注
            ax.text(bar_len + 0.02, y, f'{v:{fmt}}',
                    va='center', ha='left', fontsize=8, color='#333')

    # Y 轴标签
    ylabels = []
    for key, label, unit, fmt in feat_cfgs:
        if unit:
            ylabels.append(f'{label} ({unit})')
        else:
            ylabels.append(label)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(ylabels, fontsize=10)
    ax.set_xlim(0, 1.35)
    ax.set_xlabel('Normalized Value', fontsize=11)
    ax.set_title('Cluster Feature Comparison', fontsize=14, fontweight='bold')

    # 图例
    legend_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=colors[c], alpha=0.85)
                      for c in unique]
    legend_labels = [f'{shorts[c]}' for c in unique]
    ax.legend(legend_handles, legend_labels, loc='lower right', fontsize=10, framealpha=0.9)

    ax.grid(True, alpha=0.15, axis='x')
    ax.invert_yaxis()

    for fmt, dpi in [('.png', 300), ('.pdf', None)]:
        path = os.path.join(cfg['save_dir'], f'paper_feature_bars{fmt}')
        fig.savefig(path, dpi=dpi if dpi else None, bbox_inches='tight')
        print(f"   Saved: {path}")
    plt.close(fig)

    # ================================================================
    # 图 3: 四类特征卡片式对比图（最适合论文的方式）
    # ================================================================
    print("Figure 3: Cluster profile cards...")

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.25,
                           left=0.08, right=0.95, top=0.92, bottom=0.06)

    card_feats = [
        ('avg_speed_mov_mean', 'Speed\n(moving)', 'km/h'),
        ('acc_std_mov_mean',   'Acc.\nVar.',       'm/s²'),
        ('heading_change_mean','Heading\nChange',  '°'),
        ('soc_rate_mean',      'Energy\nRate',     '%/min'),
        ('idle_ratio_mean',    'Idle\nRatio',      ''),
        ('seg_length_mean',    'Duration',         'steps'),
    ]

    n_card_feats = len(card_feats)

    for ci, c in enumerate(unique):
        row, col = ci // 2, ci % 2
        ax = fig.add_subplot(gs[row, col])

        s = stats[c]
        meta = CLUSTER_META[c]

        # 该簇各特征的柱状图
        vals = [s[k] for k, _, _ in card_feats]

        # 同时画所有簇的同特征值做背景参考线
        all_vals_per_feat = []
        for k, _, _ in card_feats:
            fv = [stats[cc][k] for cc in unique]
            all_vals_per_feat.append(fv)

        x = np.arange(n_card_feats)
        w_bar = 0.6

        # 背景：其他簇的值（灰色细线）
        for other_c in unique:
            if other_c == c:
                continue
            other_vals = [stats[other_c][k] for k, _, _ in card_feats]
            # 归一化
            for fi in range(n_card_feats):
                fmax = max(abs(v) for v in all_vals_per_feat[fi])
                if fmax < 1e-10:
                    fmax = 1.0
                other_norm = abs(other_vals[fi]) / fmax
                ax.barh(fi, other_norm, height=0.15, color='lightgray',
                        alpha=0.4, edgecolor='none')

        # 前景：当前簇
        for fi in range(n_card_feats):
            fmax = max(abs(v) for v in all_vals_per_feat[fi])
            if fmax < 1e-10:
                fmax = 1.0
            v_norm = abs(vals[fi]) / fmax

            ax.barh(fi, v_norm, height=0.45, color=meta['color'],
                    alpha=0.85, edgecolor='white', linewidth=0.5)

            # 数值标注
            v = vals[fi]
            if abs(v) >= 100:
                txt = f'{v:.0f}'
            elif abs(v) >= 1:
                txt = f'{v:.2f}'
            elif abs(v) >= 0.001:
                txt = f'{v:.4f}'
            else:
                txt = f'{v:.5f}'
            ax.text(v_norm + 0.03, fi, txt, va='center', fontsize=9, color='#333')

        ylabels_card = [f'{lab}\n({u})' if u else lab for _, lab, u in card_feats]
        ax.set_yticks(range(n_card_feats))
        ax.set_yticklabels(ylabels_card, fontsize=9)
        ax.set_xlim(0, 1.4)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.1, axis='x')

        title = f'C{c}: {meta["short"]}\n(n={s["size"]:,}, {s["driving_pct"]:.0f}% driving)'
        ax.set_title(title, fontsize=12, fontweight='bold', color=meta['color'])

    plt.suptitle('Cluster Feature Profiles (colored = this cluster, gray = others)',
                 fontsize=14, fontweight='bold')

    for fmt, dpi in [('.png', 300), ('.pdf', None)]:
        path = os.path.join(cfg['save_dir'], f'paper_cluster_cards{fmt}')
        fig.savefig(path, dpi=dpi if dpi else None, bbox_inches='tight')
        print(f"   Saved: {path}")
    plt.close(fig)

    # ================================================================
    # LaTeX 表格
    # ================================================================
    print(f"\n{'='*70}")
    print("LaTeX Table:")
    print("="*70)
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Characteristics of four identified driving pattern clusters.}")
    print(r"\label{tab:clusters}")
    print(r"\resizebox{\textwidth}{!}{")
    print(r"\begin{tabular}{lrrrr}")
    print(r"\toprule")
    print(r"Feature & \textbf{Long Idle} & \textbf{Urban} & \textbf{Short Idle} & \textbf{Highway} \\")
    print(r"\midrule")

    ltx_rows = [
        ('size',               'Samples',               'd',    ''),
        ('driving_pct',        'Driving ratio (\\%)',    '.1f',  ''),
        ('avg_speed_mov_mean', 'Avg. speed (km/h)',     '.1f',  ''),
        ('speed_max_mean',     'Max speed (km/h)',      '.1f',  ''),
        ('acc_std_mov_mean',   'Acc. std. (m/s$^2$)',   '.5f',  ''),
        ('heading_change_mean','Heading change ($^\\circ$)', '.1f', ''),
        ('idle_ratio_mean',    'Idle ratio',            '.3f',  ''),
        ('soc_rate_mean',      'SOC rate (\\%/min)',    '.2f',  ''),
        ('power_mean_mean',    'Avg. power (kW)',       '.4f',  ''),
        ('seg_length_mean',    'Duration (steps)',      '.0f',  ''),
    ]

    for key, name, fmt, _ in ltx_rows:
        line = name
        for c in unique:
            v = stats[c].get(key, 0)
            line += f" & {v:{fmt}}"
        line += r" \\"
        print(line)

    print(r"\bottomrule")
    print(r"\end{tabular}}")
    print(r"\end{table}")

    # 保存
    with open(os.path.join(cfg['save_dir'], 'cluster_names.json'), 'w', encoding='utf-8') as f:
        json.dump({str(c): CLUSTER_META[c] for c in unique}, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print("Done!")
    print("="*70)
    for fn in sorted(os.listdir(cfg['save_dir'])):
        if fn.startswith('paper_'):
            fp = os.path.join(cfg['save_dir'], fn)
            print(f"   {fn:<50} {os.path.getsize(fp)/1024:>8.1f} KB")
    print()


if __name__ == '__main__':
    main()