"""
Step 7 v3: Latent-Space Clustering
改动：加入隐向量质量预检查，其余不变
"""

import numpy as np
import h5py
import os
import json
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from tqdm import tqdm

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
    'h5_path':       './analysis_complete_vehicles/results/dual_channel_dataset.h5',
    'latent_path':   './analysis_complete_vehicles/results/checkpoints_v2/latent_vectors.npz',
    'save_dir':      './analysis_complete_vehicles/results/clustering_v3',
    'seed':          42,
    'n_clusters':    4,
    'pca_variance':  0.95,
}

CLUSTER_COLORS = {0: '#5B9BD5', 1: '#70AD47', 2: '#C0504D', 3: '#FFC000'}
SPEED_THRESH = 0.5


def denormalize(val, vmin, vmax):
    return val * (vmax - vmin) + vmin


# ============================================================
# 0. Latent vector quality check (NEW)
# ============================================================
def check_latent_quality(z_final, z_B, z_E, seg_types, save_dir):
    """Quick diagnostic before clustering"""
    print("=" * 70)
    print("Step 0: Latent Vector Quality Check")
    print("=" * 70)

    n, d = z_final.shape
    print(f"   Shape: {z_final.shape}")

    # Dead dimensions
    dim_std = np.std(z_final, axis=0)
    n_dead = (dim_std < 1e-3).sum()
    n_active = d - n_dead
    print(f"   Active dims: {n_active}/{d} (dead: {n_dead})")

    # Norm stats
    norms = np.linalg.norm(z_final, axis=1)
    print(f"   L2 norms: mean={np.mean(norms):.3f}, std={np.std(norms):.3f}, "
          f"min={np.min(norms):.3f}, max={np.max(norms):.3f}")

    # Orthogonality
    if z_B is not None and z_E is not None:
        cos_sim = np.sum(z_B * z_E, axis=1) / (
            np.linalg.norm(z_B, axis=1) * np.linalg.norm(z_E, axis=1) + 1e-8
        )
        print(f"   Orthogonality: mean|cos(z_B, z_E)| = {np.mean(np.abs(cos_sim)):.4f}")

    # Driving vs Idle separation
    drv_mask = seg_types == 0
    idle_mask = seg_types == 1
    if drv_mask.sum() > 0 and idle_mask.sum() > 0:
        drv_mean = np.mean(z_final[drv_mask], axis=0)
        idle_mean = np.mean(z_final[idle_mask], axis=0)
        centroid_dist = np.linalg.norm(drv_mean - idle_mean)
        print(f"   Driving vs Idle centroid distance: {centroid_dist:.4f}")

        # Check if driving/idle are separable via simple PCA
        pca_2d = PCA(n_components=2, random_state=42).fit_transform(z_final)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        ax = axes[0]
        np.random.seed(42)
        n_show = min(5000, drv_mask.sum(), idle_mask.sum())
        drv_idx = np.random.choice(np.where(drv_mask)[0], n_show, replace=False)
        idle_idx = np.random.choice(np.where(idle_mask)[0], n_show, replace=False)
        ax.scatter(pca_2d[drv_idx, 0], pca_2d[drv_idx, 1],
                   c='#4C72B0', s=5, alpha=0.3, label=f'Driving (n={drv_mask.sum():,})')
        ax.scatter(pca_2d[idle_idx, 0], pca_2d[idle_idx, 1],
                   c='#DD8452', s=5, alpha=0.3, label=f'Idle (n={idle_mask.sum():,})')
        ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
        ax.set_title('Latent Space: Driving vs Idle')
        ax.legend(markerscale=4, fontsize=9); ax.grid(alpha=0.15)

        ax = axes[1]
        ax.bar(['Active', 'Dead'], [n_active, n_dead],
               color=['steelblue', 'lightcoral'], edgecolor='white')
        ax.set_ylabel('Dimensions')
        ax.set_title(f'Dimension Activation ({n_active}/{d} active)')
        ax.grid(alpha=0.15, axis='y')

        plt.suptitle('Latent Vector Quality Check', fontweight='bold', fontsize=14)
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, 'latent_quality_check.png'), dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"   Saved: latent_quality_check.png")

        if centroid_dist < 0.5:
            print(f"\n   WARNING: Driving/Idle centroid distance is very small ({centroid_dist:.4f})")
            print(f"   The latent space may not have learned meaningful separations.")
            print(f"   Consider checking Step 6 training quality.")
        else:
            print(f"\n   OK: Latent space shows separation (dist={centroid_dist:.4f})")

    return n_active


# ============================================================
# 1. Physical features (validation only)
# ============================================================
def extract_physical_features(h5_path):
    print("\n" + "=" * 70)
    print("Step 1: Extract Physical Features (for validation only)")
    print("=" * 70)

    with h5py.File(h5_path, 'r') as f:
        offsets   = f['offsets'][:]
        lengths   = f['lengths'][:]
        seg_types = f['segment_types'][:]
        n = len(lengths)
        drv_min, drv_max = f['driving_min'][:], f['driving_max'][:]
        eng_min, eng_max = f['energy_min'][:], f['energy_max'][:]
        drv_all = f['driving_packed'][:]
        eng_all = f['energy_packed'][:]

    for col in range(drv_all.shape[1]):
        drv_all[:, col] = denormalize(drv_all[:, col], drv_min[col], drv_max[col])
    for col in range(eng_all.shape[1]):
        eng_all[:, col] = denormalize(eng_all[:, col], eng_min[col], eng_max[col])
    drv_all[:, 0] *= 3.6

    avg_speed      = np.zeros(n, np.float32)
    avg_speed_mov  = np.zeros(n, np.float32)
    speed_std      = np.zeros(n, np.float32)
    speed_max      = np.zeros(n, np.float32)
    acc_std_mov    = np.zeros(n, np.float32)
    heading_change = np.zeros(n, np.float32)
    idle_ratio     = np.zeros(n, np.float32)
    soc_rate       = np.zeros(n, np.float32)
    power_mean     = np.zeros(n, np.float32)
    seg_length     = lengths.astype(np.float32)

    for i in tqdm(range(n), desc="   Extracting", ncols=80, mininterval=1):
        s, e = offsets[i], offsets[i + 1]
        L = e - s
        if L < 2:
            continue
        drv, eng = drv_all[s:e], eng_all[s:e]
        sp, ac, hd = drv[:, 0], drv[:, 1], drv[:, 2]

        avg_speed[i] = np.mean(sp)
        speed_std[i] = np.std(sp)
        speed_max[i] = np.max(sp)
        mov = sp > SPEED_THRESH
        if mov.sum() > 0: avg_speed_mov[i] = np.mean(sp[mov])
        if mov.sum() > 1: acc_std_mov[i] = np.std(ac[mov])
        idle_ratio[i] = 1.0 - mov.mean()
        if L > 1:
            hd_diff = np.abs(np.diff(hd))
            heading_change[i] = np.sum(np.minimum(hd_diff, 360.0 - hd_diff))
        soc_drop = eng[0, 0] - eng[-1, 0]
        dur_min = L / 60.0
        soc_rate[i] = soc_drop / dur_min if dur_min > 0 else 0
        power_mean[i] = np.mean(eng[:, 3]) / 1000.0

    phys = {
        'avg_speed': avg_speed, 'avg_speed_mov': avg_speed_mov,
        'speed_std': speed_std, 'speed_max': speed_max,
        'acc_std_mov': acc_std_mov, 'heading_change': heading_change,
        'idle_ratio': idle_ratio, 'soc_rate': soc_rate,
        'power_mean': power_mean, 'seg_length': seg_length,
    }
    print(f"   Done: {n:,} segments, {len(phys)} features")
    return phys, seg_types


# ============================================================
# 2. Latent clustering
# ============================================================
def run_latent_clustering(z_final, cfg):
    print(f"\n{'=' * 70}")
    print("Step 2: Latent Vector Clustering")
    print("=" * 70)

    K = cfg['n_clusters']
    seed = cfg['seed']

    scaler = StandardScaler()
    z_scaled = scaler.fit_transform(z_final)

    var = z_final.var(axis=0)
    active = var > 1e-6
    z_active = z_scaled[:, active]
    print(f"   Active dims: {active.sum()}/{z_final.shape[1]}")

    pca = PCA(n_components=cfg['pca_variance'], random_state=seed)
    z_pca = pca.fit_transform(z_active)
    print(f"   PCA: {z_active.shape[1]} -> {z_pca.shape[1]} dims "
          f"(variance: {pca.explained_variance_ratio_.sum():.2%})")

    # KMeans
    km = KMeans(n_clusters=K, n_init=20, max_iter=500, random_state=seed)
    km_labels = km.fit_predict(z_pca)
    km_sil = silhouette_score(z_pca, km_labels,
                              sample_size=min(10000, len(km_labels)), random_state=seed)
    km_ch = calinski_harabasz_score(z_pca, km_labels)
    km_db = davies_bouldin_score(z_pca, km_labels)

    # GMM
    gmm = GaussianMixture(n_components=K, covariance_type='full',
                           n_init=5, random_state=seed, max_iter=300)
    gmm.fit(z_pca)
    gmm_labels = gmm.predict(z_pca)
    gmm_sil = silhouette_score(z_pca, gmm_labels,
                               sample_size=min(10000, len(gmm_labels)), random_state=seed)
    gmm_ch = calinski_harabasz_score(z_pca, gmm_labels)
    gmm_db = davies_bouldin_score(z_pca, gmm_labels)

    print(f"\n   {'Method':>10}  {'Silhouette':>10}  {'CH':>10}  {'DB':>10}")
    print(f"   {'─' * 10}  {'─' * 10}  {'─' * 10}  {'─' * 10}")
    print(f"   {'KMeans':>10}  {km_sil:>10.4f}  {km_ch:>10.1f}  {km_db:>10.4f}")
    print(f"   {'GMM':>10}  {gmm_sil:>10.4f}  {gmm_ch:>10.1f}  {gmm_db:>10.4f}")

    if km_sil >= gmm_sil:
        best_name, best_labels = 'KMeans_Latent', km_labels
    else:
        best_name, best_labels = 'GMM_Latent', gmm_labels

    eval_results = {
        'KMeans_Latent': {'sil': km_sil, 'ch': km_ch, 'db': km_db, 'labels': km_labels},
        'GMM_Latent':    {'sil': gmm_sil, 'ch': gmm_ch, 'db': gmm_db, 'labels': gmm_labels},
    }

    best_sil = eval_results[best_name]['sil']
    print(f"\n   Best: {best_name} (Silhouette = {best_sil:.4f})")

    return best_name, best_labels, eval_results, z_pca, pca


# ============================================================
# 3. Physical validation
# ============================================================
def validate_with_physics(best_name, labels, phys, seg_types, cfg):
    print(f"\n{'=' * 70}")
    print(f"Step 3: Physical Validation [{best_name}]")
    print("=" * 70)

    unique = sorted(np.unique(labels))
    stats = {}
    for c in unique:
        mask = labels == c
        n = mask.sum()
        stats[c] = {
            'size': int(n),
            'pct': float(n / len(labels) * 100),
            'driving_pct': float((seg_types[mask] == 0).sum() / n * 100),
            'idle_pct': float((seg_types[mask] == 1).sum() / n * 100),
        }
        for key in phys:
            vals = phys[key][mask]
            stats[c][f'{key}_mean']   = float(np.mean(vals))
            stats[c][f'{key}_median'] = float(np.median(vals))
            stats[c][f'{key}_p25']    = float(np.percentile(vals, 25))
            stats[c][f'{key}_p75']    = float(np.percentile(vals, 75))

    rows = [
        ('size',               'Size',         '',      'd'),
        ('driving_pct',        'Driving %',    '%',     '.1f'),
        ('avg_speed_mean',     'Avg Speed',    'km/h',  '.2f'),
        ('avg_speed_mov_mean', 'Moving Speed', 'km/h',  '.2f'),
        ('speed_max_mean',     'Max Speed',    'km/h',  '.2f'),
        ('acc_std_mov_mean',   'Acc Std',      'm/s^2', '.5f'),
        ('heading_change_mean','Heading Chg',  'deg',   '.1f'),
        ('idle_ratio_mean',    'Idle Ratio',   '',      '.3f'),
        ('soc_rate_mean',      'SOC Rate',     '%/min', '.4f'),
        ('power_mean_mean',    'Avg Power',    'kW',    '.4f'),
        ('seg_length_mean',    'Duration',     'steps', '.0f'),
    ]

    hdr = f"   {'Metric':>16} {'Unit':>8}"
    for c in unique: hdr += f"  {'C' + str(c):>12}"
    print(hdr)
    print(f"   {'─' * 16} {'─' * 8}  " + "  ".join(['─' * 12] * len(unique)))
    for key, name, unit, fmt in rows:
        line = f"   {name:>16} {unit:>8}"
        for c in unique:
            v = stats[c].get(key, 0)
            line += f"  {v:>12{fmt}}"
        print(line)

    # Auto label
    print(f"\n   Auto Labels:")
    for c in unique:
        s = stats[c]
        drv, spd = s['driving_pct'], s['avg_speed_mov_mean']
        idle, dur = s['idle_ratio_mean'], s['seg_length_mean']

        if drv > 70 and spd > 30:
            label = "Highway Driving"
        elif drv > 70 and spd <= 30:
            label = "Urban Driving"
        elif drv > 30:
            label = "Mixed Driving"
        elif idle > 0.8 and dur > 100:
            label = "Long Idle"
        elif idle > 0.8:
            label = "Short Idle"
        else:
            label = "Unclassified"

        stats[c]['label'] = label
        print(f"     C{c}: {label} (drv={drv:.0f}%, spd={spd:.1f}km/h, idle={idle:.2f}, dur={dur:.0f})")

    return stats


# ============================================================
# 4. Visualization
# ============================================================
def plot_all(labels, phys, seg_types, z_pca, stats, best_name, eval_results, save_dir, cfg):
    print(f"\n{'=' * 70}")
    print("Step 4: Visualization")
    print("=" * 70)

    unique = sorted(np.unique(labels))
    colors = {c: CLUSTER_COLORS.get(c, f'C{c}') for c in unique}

    pca_2d = PCA(n_components=2, random_state=cfg['seed']).fit_transform(z_pca)
    ev = PCA(n_components=2, random_state=cfg['seed']).fit(z_pca).explained_variance_ratio_

    # ---- KMeans vs GMM comparison ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    np.random.seed(cfg['seed'])
    idx = np.random.choice(len(pca_2d), min(10000, len(pca_2d)), replace=False)

    for ax, (method_name, result) in zip(axes, eval_results.items()):
        method_labels = result['labels']
        for c in unique:
            m = method_labels[idx] == c
            lbl = stats.get(c, {}).get('label', f'C{c}')
            n_c = (method_labels == c).sum()
            ax.scatter(pca_2d[idx][m, 0], pca_2d[idx][m, 1],
                       c=colors[c], s=6, alpha=0.4,
                       label=f'C{c}: {lbl} (n={n_c:,})', edgecolors='none')
        ax.set_xlabel(f'Latent PC1 ({ev[0]:.1%})')
        ax.set_ylabel(f'Latent PC2 ({ev[1]:.1%})')
        ax.set_title(f'{method_name} (Sil={result["sil"]:.4f})')
        ax.legend(markerscale=3, fontsize=7)
        ax.grid(True, alpha=0.15)

    plt.suptitle('Latent-Space Clustering: KMeans vs GMM', fontweight='bold', fontsize=14)
    plt.tight_layout()
    for fmt, dpi in [('.png', 300), ('.pdf', None)]:
        fig.savefig(os.path.join(save_dir, f'clustering_comparison{fmt}'),
                    dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"   Saved: clustering_comparison.png/pdf")

    # ---- 2x3 main figure (best method) ----
    fig = plt.figure(figsize=(18, 11))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.30,
                           left=0.06, right=0.97, top=0.93, bottom=0.06)

    # (a) PCA scatter
    ax = fig.add_subplot(gs[0, 0])
    for c in unique:
        m = labels[idx] == c
        lbl = stats[c].get('label', f'C{c}')
        ax.scatter(pca_2d[idx][m, 0], pca_2d[idx][m, 1],
                   c=colors[c], s=6, alpha=0.4,
                   label=f'C{c}: {lbl} (n={stats[c]["size"]:,})', edgecolors='none')
    ax.set_xlabel(f'Latent PC1 ({ev[0]:.1%})')
    ax.set_ylabel(f'Latent PC2 ({ev[1]:.1%})')
    ax.set_title(f'(a) {best_name}: Latent Space')
    ax.legend(markerscale=4, fontsize=7); ax.grid(True, alpha=0.15)

    # (b) Driving vs Idle
    ax = fig.add_subplot(gs[0, 1])
    drv = [stats[c]['driving_pct'] for c in unique]
    idl = [stats[c]['idle_pct'] for c in unique]
    xl = [f'C{c}\n(n={stats[c]["size"]:,})' for c in unique]
    b1 = ax.bar(xl, drv, color='#4C72B0', label='Driving')
    b2 = ax.bar(xl, idl, bottom=drv, color='#DD8452', label='Idle')
    for bar, p in zip(b1, drv):
        if p > 5:
            ax.text(bar.get_x() + bar.get_width() / 2, p / 2, f'{p:.0f}%',
                    ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    for bar, p, bot in zip(b2, idl, drv):
        if p > 5:
            ax.text(bar.get_x() + bar.get_width() / 2, bot + p / 2, f'{p:.0f}%',
                    ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    ax.set_ylabel('%'); ax.set_title('(b) Driving vs Idle'); ax.legend(); ax.set_ylim(0, 108)

    # (c-f) boxplots
    box_features = [
        ('avg_speed_mov', 'Speed when moving (km/h)', '(c) Speed Distribution'),
        ('acc_std_mov',   'Acc. Std (m/s^2)',          '(d) Acceleration Variation'),
        ('soc_rate',      'SOC Rate (%/min)',           '(e) Energy Consumption'),
        ('idle_ratio',    'Idle Ratio',                 '(f) Idle Ratio'),
    ]
    positions = [(0, 2), (1, 0), (1, 1), (1, 2)]

    for (feat_key, ylabel, title), (r, c_pos) in zip(box_features, positions):
        ax = fig.add_subplot(gs[r, c_pos])
        data = [phys[feat_key][labels == c] for c in unique]
        bp = ax.boxplot(data, labels=[f'C{c}' for c in unique],
                        patch_artist=True, showfliers=False, widths=0.55,
                        medianprops=dict(color='#E67E22', linewidth=2))
        for patch, c in zip(bp['boxes'], unique):
            patch.set_facecolor(colors[c]); patch.set_alpha(0.75); patch.set_edgecolor('#555')
        for el in ['whiskers', 'caps']:
            for it in bp[el]: it.set_color('#555'); it.set_linewidth(1.2)
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.grid(True, alpha=0.15, axis='y')

    sil = eval_results[best_name]['sil']
    plt.suptitle(f'Best: {best_name} (Silhouette={sil:.4f})', fontweight='bold', fontsize=14)
    for fmt, dpi in [('.png', 300), ('.pdf', None)]:
        fig.savefig(os.path.join(save_dir, f'paper_figure{fmt}'), dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"   Saved: paper_figure.png/pdf")

    # ---- Radar ----
    radar_cfg = [
        ('avg_speed_mov', 'Avg Speed\n(moving)',  'km/h'),
        ('speed_std',     'Speed Std',             'km/h'),
        ('acc_std_mov',   'Acc Std\n(moving)',     'm/s^2'),
        ('heading_change','Heading\nChange',       'deg'),
        ('soc_rate',      'Energy\nRate',          '%/min'),
        ('idle_ratio',    'Idle\nRatio',           ''),
        ('seg_length',    'Duration',              'steps'),
    ]
    feat_keys = [r[0] for r in radar_cfg]
    feat_labels = [f'{r[1]}\n({r[2]})' if r[2] else r[1] for r in radar_cfg]
    n_feats = len(feat_keys)
    angles = np.linspace(0, 2 * np.pi, n_feats, endpoint=False).tolist()
    angles += angles[:1]

    raw = {}
    for c in unique:
        mask = labels == c
        raw[c] = [float(np.mean(phys[k][mask])) for k in feat_keys]
    arr = np.array([raw[c] for c in unique])
    fmin, fmax = arr.min(0), arr.max(0)
    frng = fmax - fmin; frng[frng < 1e-10] = 1.0
    norm = {c: ((np.array(raw[c]) - fmin) / frng).tolist() for c in unique}

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    for c in unique:
        vals = norm[c] + norm[c][:1]
        lbl = stats[c].get('label', f'C{c}')
        ax.plot(angles, vals, 'o-', lw=2.2, ms=7, label=f'C{c}: {lbl}', color=colors[c])
        ax.fill(angles, vals, alpha=0.08, color=colors[c])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feat_labels, fontsize=10)
    ax.set_ylim(-0.05, 1.15)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=10)
    ax.set_title(f'{best_name}: Cluster Profiles', fontsize=14, fontweight='bold', pad=25)
    ax.spines['polar'].set_visible(False)
    ax.grid(True, alpha=0.3)

    for fmt, dpi in [('.png', 300), ('.pdf', None)]:
        fig.savefig(os.path.join(save_dir, f'paper_radar{fmt}'), dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"   Saved: paper_radar.png/pdf")

    # Print radar values
    print(f"\n   Radar chart raw values (mean):")
    hdr = f"   {'Feature':>20} {'Unit':>8}"
    for c in unique: hdr += f"  {'C' + str(c):>10}"
    print(hdr)
    print(f"   {'─' * 20} {'─' * 8}  " + "  ".join(['─' * 10] * len(unique)))
    for i, (key, name, unit) in enumerate(radar_cfg):
        line = f"   {name.replace(chr(10), ' '):>20} {unit:>8}"
        for c in unique: line += f"  {raw[c][i]:>10.4f}"
        print(line)


# ============================================================
# Main
# ============================================================
def main():
    cfg = CONFIG
    os.makedirs(cfg['save_dir'], exist_ok=True)

    print("=" * 70)
    print("Step 7 v3: Latent-Space Clustering")
    print("=" * 70)

    # Load latent vectors
    latent_data = np.load(cfg['latent_path'])
    z_final   = latent_data['z_final']
    seg_types = latent_data['seg_types']
    z_B = latent_data.get('z_B', None)
    z_E = latent_data.get('z_E', None)
    print(f"   z_final: {z_final.shape}")
    if z_B is not None:
        print(f"   z_B: {z_B.shape}, z_E: {z_E.shape}")

    # Step 0: Quality check
    n_active = check_latent_quality(z_final, z_B, z_E, seg_types, cfg['save_dir'])

    # Step 1: Physical features (validation only)
    phys, seg_types_h5 = extract_physical_features(cfg['h5_path'])

    # Step 2: Latent clustering
    best_name, best_labels, eval_results, z_pca, pca = \
        run_latent_clustering(z_final, cfg)

    # Step 3: Physical validation
    stats = validate_with_physics(best_name, best_labels, phys, seg_types, cfg)

    # Step 4: Visualization
    plot_all(best_labels, phys, seg_types, z_pca, stats, best_name, eval_results,
             cfg['save_dir'], cfg)

    # Save
    print(f"\n   Saving...")
    save_data = {k: v for k, v in phys.items()}
    save_data['labels']    = best_labels
    save_data['seg_types'] = seg_types
    save_data['z_pca']     = z_pca
    np.savez(os.path.join(cfg['save_dir'], 'clustering_v3_results.npz'), **save_data)

    summary = {
        'best_method': best_name,
        'n_clusters': cfg['n_clusters'],
        'pca_dims': z_pca.shape[1],
        'pca_variance_retained': float(pca.explained_variance_ratio_.sum()),
        'n_active_dims': int(n_active),
        'eval_results': {name: {'sil': r['sil'], 'ch': r['ch'], 'db': r['db']}
                         for name, r in eval_results.items()},
        'cluster_stats': {str(k): v for k, v in stats.items()},
    }
    with open(os.path.join(cfg['save_dir'], 'clustering_v3_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print("Done!")
    print("=" * 70)
    print(f"   Latent: {z_final.shape[1]}d -> {n_active} active -> PCA {z_pca.shape[1]}d")
    print(f"   Best: {best_name} (Sil={eval_results[best_name]['sil']:.4f})")
    for fn in sorted(os.listdir(cfg['save_dir'])):
        fp = os.path.join(cfg['save_dir'], fn)
        if os.path.isfile(fp):
            print(f"   {fn:<50} {os.path.getsize(fp) / 1024:>8.1f} KB")
    print()


if __name__ == '__main__':
    main()