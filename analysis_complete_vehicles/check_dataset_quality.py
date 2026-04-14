"""
检查 dual_channel_dataset.h5 数据质量
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import os

filepath = './analysis_complete_vehicles/results/dual_channel_dataset.h5'

print("=" * 70)
print("🔍 HDF5 Dataset Quality Check")
print("=" * 70)

# ============ 1. 基本信息 ============
print("\n📦 [1] Basic Info")
with h5py.File(filepath, 'r') as f:
    # 属性
    for k, v in f.attrs.items():
        print(f"   {k}: {v}")

    # 各数据集 shape / dtype
    print("\n   Datasets:")
    for key in f.keys():
        ds = f[key]
        print(f"   - {key:<22}: shape={ds.shape}, dtype={ds.dtype}")

    # 读取数据
    X_drv  = f['driving_sequences'][:]
    X_eng  = f['energy_sequences'][:]
    lens   = f['lengths'][:]
    mask   = f['valid_mask'][:]
    d_min  = f['driving_min'][:]
    d_max  = f['driving_max'][:]
    e_min  = f['energy_min'][:]
    e_max  = f['energy_max'][:]

DRIVING_FEATURES = ['speed', 'acc', 'heading']
ENERGY_FEATURES  = ['soc', 'voltage', 'current', 'power']

# ============ 2. 序列长度分布 ============
print("\n📏 [2] Sequence Length Distribution")
print(f"   Min    : {lens.min()}")
print(f"   Max    : {lens.max()}")
print(f"   Mean   : {lens.mean():.2f}")
print(f"   Median : {np.median(lens):.2f}")
print(f"   Std    : {lens.std():.2f}")

# 分箱统计
bins = [0, 50, 100, 200, 500, 1000]
counts, _ = np.histogram(lens, bins=bins)
print(f"\n   Length distribution:")
for i in range(len(counts)):
    bar = '█' * int(counts[i] / max(counts) * 30)
    print(f"   [{bins[i]:>4} ~{bins[i+1]:>4}] {counts[i]:>6} | {bar}")

# ============ 3. Padding 比例 ============
print("\n🧩 [3] Padding Ratio")
total_steps = X_drv.shape[0] * X_drv.shape[1]
valid_steps = mask.sum()
pad_steps   = total_steps - valid_steps
print(f"   Total timesteps : {total_steps:,}")
print(f"   Valid timesteps : {valid_steps:,} ({100*valid_steps/total_steps:.1f}%)")
print(f"   Padding steps   : {pad_steps:,}   ({100*pad_steps/total_steps:.1f}%)")

# ============ 4. 归一化范围检查 ============
print("\n📊 [4] Normalization Range Check (valid timesteps only)")
print(f"\n   Driving features (raw min/max saved):")
for i, feat in enumerate(DRIVING_FEATURES):
    print(f"   - {feat:<10}: raw_min={d_min[i]:.4f}, raw_max={d_max[i]:.4f}")

print(f"\n   Energy features (raw min/max saved):")
for i, feat in enumerate(ENERGY_FEATURES):
    print(f"   - {feat:<10}: raw_min={e_min[i]:.4f}, raw_max={e_max[i]:.4f}")

# 检查归一化后有效值是否在 [0,1]
drv_valid = X_drv[mask]
eng_valid = X_eng[mask]
print(f"\n   Normalized driving - min={drv_valid.min():.4f}, max={drv_valid.max():.4f} (should be [0,1])")
print(f"   Normalized energy  - min={eng_valid.min():.4f}, max={eng_valid.max():.4f} (should be [0,1])")

# ============ 5. NaN / Inf 检查 ============
print("\n🚨 [5] NaN / Inf Check")
for name, arr in [('driving_sequences', X_drv), ('energy_sequences', X_eng)]:
    n_nan = np.isnan(arr).sum()
    n_inf = np.isinf(arr).sum()
    status = "✅ Clean" if n_nan == 0 and n_inf == 0 else "❌ Has issues!"
    print(f"   {name:<22}: NaN={n_nan}, Inf={n_inf}  {status}")

# ============ 6. 各特征统计（只看有效位置） ============
print("\n📈 [6] Per-Feature Statistics (valid timesteps only)")
print(f"\n   {'Feature':<12} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Zero%':>8}")
print("   " + "-" * 55)

for i, feat in enumerate(DRIVING_FEATURES):
    vals = X_drv[:, :, i][mask]
    zero_pct = 100 * (vals == 0).sum() / len(vals)
    print(f"   {feat:<12} {vals.mean():>8.4f} {vals.std():>8.4f} "
          f"{vals.min():>8.4f} {vals.max():>8.4f} {zero_pct:>7.1f}%")

for i, feat in enumerate(ENERGY_FEATURES):
    vals = X_eng[:, :, i][mask]
    zero_pct = 100 * (vals == 0).sum() / len(vals)
    print(f"   {feat:<12} {vals.mean():>8.4f} {vals.std():>8.4f} "
          f"{vals.min():>8.4f} {vals.max():>8.4f} {zero_pct:>7.1f}%")

# ============ 7. Padding 位置是否严格为 0 ============
print("\n🔎 [7] Padding Zero Check")
pad_mask = ~mask
for name, arr in [('driving', X_drv), ('energy', X_eng)]:
    pad_vals = arr[pad_mask]
    nonzero_in_pad = (pad_vals != 0).sum()
    status = "✅ All zero" if nonzero_in_pad == 0 else f"❌ {nonzero_in_pad} non-zero values!"
    print(f"   {name:<10} padding: {status}")

# ============ 8. 随机抽样可视化（保存图片） ============
print("\n🖼️  [8] Saving sample visualization...")

n_plot   = min(3, X_drv.shape[0])
fig, axes = plt.subplots(n_plot, 2, figsize=(14, 4 * n_plot))
if n_plot == 1:
    axes = axes[np.newaxis, :]

sample_indices = np.random.choice(X_drv.shape[0], n_plot, replace=False)

for row, idx in enumerate(sample_indices):
    seq_len = lens[idx]

    # 驾驶通道
    ax = axes[row, 0]
    for fi, feat in enumerate(DRIVING_FEATURES):
        ax.plot(range(seq_len), X_drv[idx, :seq_len, fi], label=feat, alpha=0.8)
    ax.axvline(seq_len, color='red', linestyle='--', alpha=0.5, label='end')
    ax.set_title(f"Sample {idx} | Driving (len={seq_len})")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Normalized Value")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # 能源通道
    ax = axes[row, 1]
    for fi, feat in enumerate(ENERGY_FEATURES):
        ax.plot(range(seq_len), X_eng[idx, :seq_len, fi], label=feat, alpha=0.8)
    ax.axvline(seq_len, color='red', linestyle='--', alpha=0.5, label='end')
    ax.set_title(f"Sample {idx} | Energy (len={seq_len})")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Normalized Value")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
save_path = './analysis_complete_vehicles/results/dataset_quality_check.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"   ✅ Saved to {save_path}")

# ============ 9. 总结 ============
print(f"\n{'=' * 70}")
print("📋 Quality Check Summary")
print(f"{'=' * 70}")

issues = []
if np.isnan(X_drv).any() or np.isnan(X_eng).any():
    issues.append("❌ NaN values detected!")
if np.isinf(X_drv).any() or np.isinf(X_eng).any():
    issues.append("❌ Inf values detected!")
if X_drv[mask].max() > 1.01 or X_drv[mask].min() < -0.01:
    issues.append("❌ Driving values out of [0,1] range!")
if X_eng[mask].max() > 1.01 or X_eng[mask].min() < -0.01:
    issues.append("❌ Energy values out of [0,1] range!")
if (X_drv[~mask] != 0).any() or (X_eng[~mask] != 0).any():
    issues.append("❌ Padding positions are not zero!")
if pad_steps / total_steps > 0.5:
    issues.append(f"⚠️  High padding ratio: {100*pad_steps/total_steps:.1f}% — consider trimming max_seq_len")

if issues:
    for issue in issues:
        print(f"   {issue}")
else:
    print("   ✅ All checks passed! Dataset is ready for GRU training.")

print(f"{'=' * 70}\n")