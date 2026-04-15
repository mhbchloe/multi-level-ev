"""
Step 7c Final: 论文级聚类分析图
四类：长时怠速 / 城市行驶 / 短时怠速 / 高速行驶
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

# ============================================================
CONFIG = {
    'result_path': './analysis_complete_vehicles/results/clustering_v3/clustering_v3_results.npz',
    'save_dir':    './analysis_complete_vehicles/results/clustering_v3',
    'seed':        42,
}

# 四类配色与命名
CLUSTER_META = {
    0: {'name': 'Long Stationary Idle',  'name_cn': '长时静态怠速', 'short': 'Long Idle',
        'color': '#5B9BD5', 'icon': '🅿️'},
    1: {'name': 'Urban Cruising',        'name_cn': '城市中速行驶', 'short': 'Urban',
        'color': '#70AD47', 'icon': '🚗'},
    2: {'name': 'Short Active Idle',     'name_cn': '短时活跃怠速', 'short': 'Short Idle',
        'color': '#C0504D', 'icon': '🔴'},
    3: {'name': 'Highway Driving',       'name_cn': '高速持续行驶', 'short': 'Highway',
        'color': '#FFC000', 'icon': '🛣️'},
}


def main():
    cfg = CONFIG
    os.makedirs(cfg['save_dir'], exist_ok=True)

    print("=" * 70)
    print("🎨 Final Paper Figures")
    print("=" * 70)

    # 加载
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
            'size': int(n), 'pct': n/n_total*100,
            'driving_pct': (seg_types[mask]==0).sum()/n*100,
        }
        for k in phys:
            stats[c][f'{k}_mean'] = float(np.mean(phys[k][mask]))
            stats[c][f'{k}_med']  = float(np.median(phys[k][mask]))
            stats[c][f'{k}_p25']  = float(np.percentile(phys[k][mask], 25))
            stats[c][f'{k}_p75']  = float(np.percentile(phys[k][mask], 75))

    for c in unique:
        s = stats[c]
        m = CLUSTER_META[c]
        print(f"   {m['icon']} C{c} {m['name']:.<30} n={s['size']:>6,} ({s['pct']:.1f}%)  "
              f"drv={s['driving_pct']:.0f}%  spd={s['avg_speed_mov_mean']:.1f}km/h  "
              f"idle={s['idle_ratio_mean']:.2f}  dur={s['seg_length_mean']:.0f}")

    colors = {c: CLUSTER_META[c]['color'] for c in unique}

    # ================================================================
    # 图 1: 2×3 综合大图
    # ================================================================
    print(f"\n🎨 Figure 1: 2×3 综合大图...")

    pca2 = PCA(n_components=2, random_state=cfg['seed'])
    z_2d = pca2.fit_transform(z_pca)
    ev = pca2.explained_variance_ratio_

    fig = plt.figure(figsize=(18, 11))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.32,
                           left=0.06, right=0.97, top=0.93, bottom=0.06)

    # (a) PCA 散点
    ax = fig.add_subplot(gs[0, 0])
    np.random.seed(cfg['seed'])
    idx = np.random.choice(len(z_2d), min(10000, len(z_2d)), replace=False)
    for c in unique:
        m = labels[idx] == c
        ax.scatter(z_2d[idx][m, 0], z_2d[idx][m, 1],
                   c=colors[c], s=5, alpha=0.35, edgecolors='none',
                   label=f'{CLUSTER_META[c]["short"]} (n={stats[c]["size"]:,})')
    ax.set_xlabel(f'PC1 ({ev[0]:.1%})')
    ax.set_ylabel(f'PC2 ({ev[1]:.1%})')
    ax.set_title('(a) Cluster Distribution in PCA Space')
    ax.legend(markerscale=5, fontsize=8, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.15)

    # (b) 行驶/怠速 + 簇大小
    ax = fig.add_subplot(gs[0, 1])
    x = np.arange(len(unique))
    w = 0.6
    drv = [stats[c]['driving_pct'] for c in unique]
    idl = [100 - stats[c]['driving_pct'] for c in unique]
    xlbl = [f'{CLUSTER_META[c]["short"]}' for c in unique]

    b1 = ax.bar(x, drv, w, color='#4C72B0', label='Driving', edgecolor='white')
    b2 = ax.bar(x, idl, w, bottom=drv, color='#DD8452', label='Idle', edgecolor='white')

    for bar, p in zip(b1, drv):
        if p > 8:
            ax.text(bar.get_x()+w/2, p/2, f'{p:.0f}%',
                    ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    for bar, p, bot in zip(b2, idl, drv):
        if p > 8:
            ax.text(bar.get_x()+w/2, bot+p/2, f'{p:.0f}%',
                    ha='center', va='center', fontsize=9, fontweight='bold', color='white')

    # 簇大小标注在顶部
    for i, c in enumerate(unique):
        ax.text(i, 103, f'n={stats[c]["size"]:,}', ha='center', fontsize=8, color='gray')

    ax.set_xticks(x)
    ax.set_xticklabels(xlbl, fontsize=9)
    ax.set_ylabel('Composition (%)')
    ax.set_title('(b) Driving / Idle Composition')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 112)
    ax.grid(True, alpha=0.15, axis='y')

    # (c) 速度 vs 能耗 气泡图
    ax = fig.add_subplot(gs[0, 2])
    for c in unique:
        s = stats[c]
        bubble = s['size'] / n_total * 4000
        ax.scatter(s['avg_speed_mov_mean'], s['soc_rate_mean'],
                   s=bubble, c=colors[c], alpha=0.75,
                   edgecolors='#333', linewidths=1.5, zorder=5)
        ax.annotate(CLUSTER_META[c]['short'],
                    (s['avg_speed_mov_mean'], s['soc_rate_mean']),
                    fontsize=9, fontweight='bold', ha='center', va='center',
                    color='white',
                    path_effects=[patheffects.withStroke(linewidth=2, foreground='#333')])
    ax.set_xlabel('Avg. Speed when Moving (km/h)')
    ax.set_ylabel('SOC Consumption Rate (%/min)')
    ax.set_title('(c) Speed vs Energy Consumption')
    ax.grid(True, alpha=0.15)

    # (d)(e)(f) 三个箱线图
    box_configs = [
        ('avg_speed_mov', '(d) Speed when Moving', 'Speed (km/h)'),
        ('acc_std_mov',   '(e) Acceleration Variation', 'Acc. Std (m/s²)'),
        ('soc_rate',      '(f) Energy Consumption Rate', 'SOC Rate (%/min)'),
    ]

    for gi, (feat, title, ylabel) in enumerate(box_configs):
        ax = fig.add_subplot(gs[1, gi])
        box_data = [phys[feat][labels == c] for c in unique]

        bp = ax.boxplot(box_data,
                        labels=[CLUSTER_META[c]['short'] for c in unique],
                        patch_artist=True, showfliers=False, widths=0.55,
                        medianprops=dict(color='#E67E22', linewidth=2))
        for patch, c in zip(bp['boxes'], unique):
            patch.set_facecolor(colors[c])
            patch.set_alpha(0.75)
            patch.set_edgecolor('#555')
        for el in ['whiskers', 'caps']:
            for it in bp[el]:
                it.set_color('#555')
                it.set_linewidth(1.2)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.15, axis='y')
        # x 轴标签旋转
        ax.tick_params(axis='x', rotation=15)

    for fmt, dpi in [('.png', 300), ('.pdf', None)]:
        path = os.path.join(cfg['save_dir'], f'paper_main_figure{fmt}')
        fig.savefig(path, dpi=dpi if dpi else None, bbox_inches='tight')
        print(f"   📊 Saved: {path}")
    plt.close(fig)

    # ================================================================
    # 图 2: 分组对比柱状图（代替雷达图，更清晰）
    # ================================================================
    print(f"\n🎨 Figure 2: 特征对比柱状图...")

    feat_configs = [
        ('avg_speed_mov_mean', 'Avg. Speed\n(moving)', 'km/h'),
        ('speed_max_mean',     'Max Speed',             'km/h'),
        ('acc_std_mov_mean',   'Acc. Variation',        'm/s²'),
        ('heading_change_mean','Heading\nChange',       '°'),
        ('soc_rate_mean',      'Energy Rate',           '%/min'),
        ('idle_ratio_mean',    'Idle Ratio',            ''),
        ('seg_length_mean',    'Duration',              'steps'),
        ('power_mean_mean',    'Avg. Power',            'kW'),
    ]

    n_feats = len(feat_configs)
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    axes = axes.flatten()

    x = np.arange(len(unique))
    w = 0.65

    for i, (key, title, unit) in enumerate(feat_configs):
        ax = axes[i]
        vals = [stats[c][key] for c in unique]
        bars = ax.bar(x, vals, w,
                      color=[colors[c] for c in unique],
                      edgecolor='white', linewidth=1.5)

        # 数值标注
        for bar, v in zip(bars, vals):
            if abs(v) >= 100:
                txt = f'{v:.0f}'
            elif abs(v) >= 1:
                txt = f'{v:.1f}'
            elif abs(v) >= 0.001:
                txt = f'{v:.3f}'
            else:
                txt = f'{v:.5f}'
            ax.text(bar.get_x()+w/2, bar.get_height()*1.02 + max(vals)*0.02,
                    txt, ha='center', va='bottom', fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([CLUSTER_META[c]['short'] for c in unique], fontsize=8, rotation=15)
        ylabel = f'{title}\n({unit})' if unit else title
        ax.set_title(ylabel, fontsize=11)
        ax.grid(True, alpha=0.15, axis='y')
        ax.set_ylim(0, max(vals) * 1.25 if max(vals) > 0 else 1)

    plt.suptitle('Cluster Feature Comparison (Mean Values)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    for fmt, dpi in [('.png', 300), ('.pdf', None)]:
        path = os.path.join(cfg['save_dir'], f'paper_feature_comparison{fmt}')
        fig.savefig(path, dpi=dpi if dpi else None, bbox_inches='tight')
        print(f"   📊 Saved: {path}")
    plt.close(fig)

    # ================================================================
    # 图 3: 改进雷达图（对数缩放 + 分组）
    # ================================================================
    print(f"\n🎨 Figure 3: 改进雷达��（对数缩放）...")

    radar_feats = [
        ('avg_speed_mov_mean', 'Avg. Speed\n(km/h)'),
        ('acc_std_mov_mean',   'Acc. Var.\n(m/s²)'),
        ('heading_change_mean','Heading Δ\n(°)'),
        ('soc_rate_mean',      'Energy Rate\n(%/min)'),
        ('idle_ratio_mean',    'Idle Ratio'),
        ('seg_length_mean',    'Duration\n(steps)'),
    ]

    feat_keys = [r[0] for r in radar_feats]
    feat_labels = [r[1] for r in radar_feats]
    n_rf = len(radar_feats)

    # 取各簇值，log1p 缩放后再归一化
    raw = {}
    for c in unique:
        raw[c] = [stats[c][k] for k in feat_keys]

    # log1p 缩放（压缩极端值差异）
    log_vals = {}
    for c in unique:
        log_vals[c] = [np.log1p(abs(v)) for v in raw[c]]

    arr = np.array([log_vals[c] for c in unique])
    fmin, fmax = arr.min(0), arr.max(0)
    frng = fmax - fmin
    frng[frng < 1e-10] = 1.0

    norm = {c: ((np.array(log_vals[c]) - fmin) / frng).tolist() for c in unique}

    angles = np.linspace(0, 2*np.pi, n_rf, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    for c in unique:
        vals = norm[c] + norm[c][:1]
        ax.plot(angles, vals, 'o-', lw=2.5, ms=8,
                label=f'{CLUSTER_META[c]["icon"]} {CLUSTER_META[c]["short"]}',
                color=colors[c])
        ax.fill(angles, vals, alpha=0.08, color=colors[c])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feat_labels, fontsize=11)
    ax.set_ylim(-0.05, 1.15)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0', '0.25', '0.5', '0.75', '1.0'], fontsize=8, color='gray')
    ax.legend(loc='upper right', bbox_to_anchor=(1.38, 1.1), fontsize=11, framealpha=0.9)
    ax.set_title('Cluster Feature Profiles\n(log-scaled normalization)',
                 fontsize=14, fontweight='bold', pad=25)
    ax.spines['polar'].set_visible(False)
    ax.grid(True, alpha=0.3)

    for fmt, dpi in [('.png', 300), ('.pdf', None)]:
        path = os.path.join(cfg['save_dir'], f'paper_radar_log{fmt}')
        fig.savefig(path, dpi=dpi if dpi else None, bbox_inches='tight')
        print(f"   📊 Saved: {path}")
    plt.close(fig)

    # 打印原始值 + log值
    print(f"\n   📋 雷达图数值:")
    hdr = f"   {'Feature':>18}"
    for c in unique:
        hdr += f"  {CLUSTER_META[c]['short']:>14}"
    print(hdr)
    print(f"   {'─'*18}" + "  ".join(['─'*14]*len(unique)))
    for i, (key, label) in enumerate(radar_feats):
        line = f"   {label.replace(chr(10),' '):>18}"
        for c in unique:
            v = raw[c][i]
            if abs(v) >= 100:
                line += f"  {v:>14.1f}"
            elif abs(v) >= 1:
                line += f"  {v:>14.2f}"
            else:
                line += f"  {v:>14.5f}"
        print(line)

    # ================================================================
    # 图 4: 论文 Table 用的统计表（LaTeX 格式）
    # ================================================================
    print(f"\n📋 LaTeX 表格:")
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Characteristics of four identified driving pattern clusters}")
    print(r"\label{tab:clusters}")
    print(r"\begin{tabular}{lcccc}")
    print(r"\hline")
    print(r"Feature & Long Idle & Urban & Short Idle & Highway \\")
    print(r"\hline")

    latex_rows = [
        ('size',               'Samples',                 'd',    ''),
        ('driving_pct',        'Driving ratio',           '.1f',  '\\%'),
        ('avg_speed_mov_mean', 'Avg. speed (moving)',     '.1f',  'km/h'),
        ('speed_max_mean',     'Max speed',               '.1f',  'km/h'),
        ('acc_std_mov_mean',   'Acc. variation',          '.5f',  'm/s$^2$'),
        ('heading_change_mean','Heading change',          '.1f',  '$^\\circ$'),
        ('idle_ratio_mean',    'Idle ratio',              '.3f',  ''),
        ('soc_rate_mean',      'SOC rate',                '.2f',  '\\%/min'),
        ('power_mean_mean',    'Avg. power',              '.4f',  'kW'),
        ('seg_length_mean',    'Duration',                '.0f',  'steps'),
    ]

    for key, name, fmt, unit in latex_rows:
        unit_str = f' ({unit})' if unit else ''
        line = f"{name}{unit_str}"
        for c in unique:
            v = stats[c].get(key, 0)
            line += f" & {v:{fmt}}"
        line += r" \\"
        print(line)

    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")

    # 保存
    with open(os.path.join(cfg['save_dir'], 'cluster_names.json'), 'w', encoding='utf-8') as f:
        json.dump({str(c): CLUSTER_META[c] for c in unique}, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print("✅ 完成！")
    print("="*70)
    print(f"\n   📁 输出:")
    for fn in sorted(os.listdir(cfg['save_dir'])):
        if fn.startswith('paper_'):
            fp = os.path.join(cfg['save_dir'], fn)
            print(f"      {fn:<50} {os.path.getsize(fp)/1024:>8.1f} KB")

    print(f"\n   📌 四类命名:")
    for c in unique:
        m = CLUSTER_META[c]
        s = stats[c]
        print(f"      {m['icon']} C{c}: {m['name_cn']} ({m['name']})")
        print(f"           n={s['size']:,} | drv={s['driving_pct']:.0f}% | "
              f"spd={s['avg_speed_mov_mean']:.1f}km/h | idle={s['idle_ratio_mean']:.2f} | "
              f"dur={s['seg_length_mean']:.0f} | soc={s['soc_rate_mean']:.2f}%/min")

    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    main()