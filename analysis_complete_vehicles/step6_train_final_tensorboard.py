"""
Step 6: Train Dual-Channel GRU (using Step 5 model definition)
支持：
  - 后台运行（nohup / tmux 友好）
  - TensorBoard 完整监控
  - 自动保存训练状态，支持断点续训
  - 论文级可视化输出
"""

import torch
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无显示器后端，服务器必须
import matplotlib.pyplot as plt
from matplotlib import rcParams
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys
import json
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# ============================================================
# Logging（文件 + 终端双输出，关掉 VSCode 也能看日志）
# ============================================================
os.makedirs('./analysis_complete_vehicles/results/checkpoints_v2', exist_ok=True)

log_file = f'./analysis_complete_vehicles/results/checkpoints_v2/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

# ============================================================
# Import model from Step 5
# ============================================================
from step5_dual_gru_model import (
    DualChannelGRU,
    PackedHDF5Dataset,
    collate_fn,
    compute_loss,
)

# ============================================================
# Plot style
# ============================================================
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
rcParams['font.size'] = 11
rcParams['axes.titleweight'] = 'bold'
rcParams['axes.unicode_minus'] = False
rcParams['figure.dpi'] = 150

# ============================================================
# Config
# ============================================================
CONFIG = {
    'h5_path': './analysis_complete_vehicles/results/dual_channel_dataset.h5',
    'save_dir': './analysis_complete_vehicles/results/checkpoints_v2',
    'log_dir': './analysis_complete_vehicles/runs/dual_channel_gru_v2',
    'seed': 42,

    # Model
    'driving_input_dim': 3,
    'energy_input_dim': 4,
    'hidden_dim': 128,
    'num_layers': 2,
    'dropout': 0.2,
    'n_segment_types': 2,
    'type_embed_dim': 8,

    # Training
    'batch_size': 256,
    'num_workers': 4,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'lambda_rec': 1.0,
    'lambda_orth': 0.1,
    'max_epochs': 100,
    'patience': 15,
    'scheduler_factor': 0.5,
    'scheduler_patience': 5,
    'grad_clip': 1.0,

    # Resume
    'resume_from': None,  # 设为 checkpoint 路径即可断点续训
}

cfg = CONFIG
os.makedirs(cfg['save_dir'], exist_ok=True)
torch.manual_seed(cfg['seed'])
np.random.seed(cfg['seed'])

logger.info("=" * 70)
logger.info("Step 6: Training Dual-Channel GRU")
logger.info("=" * 70)
logger.info(f"Config: {json.dumps(cfg, indent=2, default=str)}")
logger.info(f"Log file: {log_file}")

# ============================================================
# Dataset
# ============================================================
logger.info("Loading dataset...")

train_dataset = PackedHDF5Dataset(cfg['h5_path'], split='train', seed=cfg['seed'])
val_dataset   = PackedHDF5Dataset(cfg['h5_path'], split='val',   seed=cfg['seed'])

train_loader = DataLoader(
    train_dataset, batch_size=cfg['batch_size'], shuffle=True,
    collate_fn=collate_fn, num_workers=cfg['num_workers'], pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=cfg['batch_size'], shuffle=False,
    collate_fn=collate_fn, num_workers=cfg['num_workers'], pin_memory=True
)

logger.info(f"Train: {len(train_dataset):,} samples, {len(train_loader)} batches")
logger.info(f"Val:   {len(val_dataset):,} samples, {len(val_loader)} batches")

# ============================================================
# Model
# ============================================================
logger.info("Initializing model...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Device: {device}")
if device.type == 'cuda':
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

model = DualChannelGRU(
    driving_input_dim=cfg['driving_input_dim'],
    energy_input_dim=cfg['energy_input_dim'],
    hidden_dim=cfg['hidden_dim'],
    num_layers=cfg['num_layers'],
    dropout=cfg['dropout'],
    n_segment_types=cfg['n_segment_types'],
    type_embed_dim=cfg['type_embed_dim'],
).to(device)

use_amp = device.type == 'cuda'
scaler = torch.amp.GradScaler('cuda') if use_amp else None

n_params = sum(p.numel() for p in model.parameters())
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(f"Parameters: {n_params:,} (trainable: {n_trainable:,})")
logger.info(f"AMP: {use_amp}")

# Print model architecture
logger.info(f"\nModel Architecture:\n{model}\n")

# ============================================================
# Optimizer & Scheduler
# ============================================================
optimizer = optim.Adam(
    model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay']
)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min',
    factor=cfg['scheduler_factor'],
    patience=cfg['scheduler_patience'],
    threshold=1e-4
)

# ============================================================
# Resume from checkpoint
# ============================================================
start_epoch = 0
best_val_loss = float('inf')
patience_counter = 0
global_step = 0
history = {k: [] for k in [
    'train_loss', 'val_loss',
    'train_rec', 'val_rec',
    'train_drv', 'val_drv',
    'train_eng', 'val_eng',
    'train_orth', 'val_orth',
    'lr',
]}

if cfg['resume_from'] and os.path.exists(cfg['resume_from']):
    logger.info(f"Resuming from: {cfg['resume_from']}")
    ckpt = torch.load(cfg['resume_from'], map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    start_epoch = ckpt.get('epoch', 0) + 1
    best_val_loss = ckpt.get('val_loss', float('inf'))
    global_step = ckpt.get('global_step', 0)
    if 'history' in ckpt:
        history = ckpt['history']
    logger.info(f"Resumed at epoch {start_epoch}, val_loss={best_val_loss:.4f}")

# ============================================================
# TensorBoard
# ============================================================
writer = SummaryWriter(cfg['log_dir'])
writer.add_text('Config', f'```json\n{json.dumps(cfg, indent=2, default=str)}\n```')

# Model architecture text
arch_text = f"```\n{model}\n```"
writer.add_text('Architecture', arch_text)

# Model parameter count per module
param_table = "| Module | Parameters |\n|--------|------------|\n"
for name, module in model.named_children():
    n_p = sum(p.numel() for p in module.parameters())
    param_table += f"| {name} | {n_p:,} |\n"
param_table += f"| **Total** | **{n_params:,}** |"
writer.add_text('Parameters', param_table)

# Model graph
try:
    dummy_d = torch.randn(2, 50, cfg['driving_input_dim']).to(device)
    dummy_e = torch.randn(2, 50, cfg['energy_input_dim']).to(device)
    dummy_l = torch.tensor([50, 30]).to(device)
    dummy_s = torch.tensor([0, 1]).to(device)
    writer.add_graph(model, (dummy_d, dummy_e, dummy_l, dummy_s))
    logger.info("TensorBoard: model graph added")
except Exception as e:
    logger.warning(f"TensorBoard graph failed: {e}")

logger.info(f"TensorBoard: tensorboard --logdir {cfg['log_dir']}")

# ============================================================
# Training Loop
# ============================================================
logger.info(f"\n{'=' * 70}")
logger.info(f"Training started: {datetime.now()}")
logger.info(f"{'=' * 70}\n")


def run_epoch(loader, training=True):
    global global_step
    model.train() if training else model.eval()
    epoch_losses = {k: [] for k in ['total', 'rec', 'drv', 'eng', 'orth']}

    prefix = 'Train' if training else 'Val'
    pbar = tqdm(loader, desc=f'  [{prefix}]', ncols=80, file=sys.stdout)

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch_idx, (drv_pad, eng_pad, lengths, seg_types) in enumerate(pbar):
            drv_pad   = drv_pad.to(device, non_blocking=True)
            eng_pad   = eng_pad.to(device, non_blocking=True)
            lengths   = lengths.to(device, non_blocking=True)
            seg_types = seg_types.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast('cuda'):
                    output = model(drv_pad, eng_pad, lengths, seg_types)
                    loss, ld = compute_loss(
                        output, drv_pad, eng_pad, lengths,
                        cfg['lambda_rec'], cfg['lambda_orth']
                    )
                if training:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
                    scaler.step(optimizer)
                    scaler.update()
            else:
                output = model(drv_pad, eng_pad, lengths, seg_types)
                loss, ld = compute_loss(
                    output, drv_pad, eng_pad, lengths,
                    cfg['lambda_rec'], cfg['lambda_orth']
                )
                if training:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
                    optimizer.step()

            epoch_losses['total'].append(ld['total'])
            epoch_losses['rec'].append(ld['reconstruction'])
            epoch_losses['drv'].append(ld['reconstruction_driving'])
            epoch_losses['eng'].append(ld['reconstruction_energy'])
            epoch_losses['orth'].append(ld['orthogonal'])

            # TensorBoard batch-level logging
            if training and (batch_idx + 1) % 10 == 0:
                writer.add_scalar('Batch/Total_Loss', ld['total'], global_step)
                writer.add_scalar('Batch/Reconstruction', ld['reconstruction'], global_step)
                writer.add_scalar('Batch/Driving_Rec', ld['reconstruction_driving'], global_step)
                writer.add_scalar('Batch/Energy_Rec', ld['reconstruction_energy'], global_step)
                writer.add_scalar('Batch/Orthogonal', ld['orthogonal'], global_step)

            if training:
                global_step += 1

            pbar.set_postfix({'L': f"{np.mean(epoch_losses['total']):.4f}"})

    return {k: float(np.mean(v)) for k, v in epoch_losses.items()}


for epoch in range(start_epoch, cfg['max_epochs']):
    epoch_start = datetime.now()
    logger.info(f"Epoch {epoch + 1}/{cfg['max_epochs']}")

    train_m = run_epoch(train_loader, training=True)
    val_m   = run_epoch(val_loader, training=False)
    current_lr = optimizer.param_groups[0]['lr']

    epoch_time = (datetime.now() - epoch_start).total_seconds()

    # ---- History ----
    for prefix, m in [('train', train_m), ('val', val_m)]:
        history[f'{prefix}_loss'].append(m['total'])
        history[f'{prefix}_rec'].append(m['rec'])
        history[f'{prefix}_drv'].append(m['drv'])
        history[f'{prefix}_eng'].append(m['eng'])
        history[f'{prefix}_orth'].append(m['orth'])
    history['lr'].append(current_lr)

    # ---- TensorBoard epoch-level ----
    writer.add_scalars('Epoch/Total_Loss', {
        'Train': train_m['total'], 'Val': val_m['total']}, epoch)
    writer.add_scalars('Epoch/Reconstruction', {
        'Train': train_m['rec'], 'Val': val_m['rec']}, epoch)
    writer.add_scalars('Epoch/Driving_Rec', {
        'Train': train_m['drv'], 'Val': val_m['drv']}, epoch)
    writer.add_scalars('Epoch/Energy_Rec', {
        'Train': train_m['eng'], 'Val': val_m['eng']}, epoch)
    writer.add_scalars('Epoch/Orthogonal', {
        'Train': train_m['orth'], 'Val': val_m['orth']}, epoch)
    writer.add_scalar('Epoch/Learning_Rate', current_lr, epoch)
    writer.add_scalar('Epoch/Epoch_Time_sec', epoch_time, epoch)

    # Gradient norm
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    grad_norm = total_norm ** 0.5
    writer.add_scalar('Epoch/Gradient_Norm', grad_norm, epoch)

    # Per-module gradient norms
    for name, module in model.named_children():
        m_norm = 0.0
        for p in module.parameters():
            if p.grad is not None:
                m_norm += p.grad.data.norm(2).item() ** 2
        writer.add_scalar(f'GradNorm/{name}', m_norm ** 0.5, epoch)

    # Weight norms
    for name, param in model.named_parameters():
        if param.requires_grad:
            writer.add_histogram(f'Weights/{name}', param.data, epoch)
            if param.grad is not None:
                writer.add_histogram(f'Gradients/{name}', param.grad.data, epoch)

    # Loss ratio
    if train_m['rec'] > 0:
        writer.add_scalar('Epoch/Orth_Rec_Ratio',
                           train_m['orth'] / train_m['rec'], epoch)
    if train_m['drv'] > 0:
        writer.add_scalar('Epoch/Energy_Driving_Ratio',
                           train_m['eng'] / train_m['drv'], epoch)

    # Generalization gap
    writer.add_scalar('Epoch/Generalization_Gap',
                       val_m['total'] - train_m['total'], epoch)

    writer.flush()  # 确保写入磁盘

    # ---- Log ----
    logger.info(
        f"  Train: {train_m['total']:.4f} "
        f"(rec={train_m['rec']:.4f} [drv={train_m['drv']:.4f} eng={train_m['eng']:.4f}] "
        f"orth={train_m['orth']:.4f})"
    )
    logger.info(
        f"  Val:   {val_m['total']:.4f} "
        f"(rec={val_m['rec']:.4f} [drv={val_m['drv']:.4f} eng={val_m['eng']:.4f}] "
        f"orth={val_m['orth']:.4f})"
    )
    logger.info(f"  LR: {current_lr:.6f} | GradNorm: {grad_norm:.4f} | Time: {epoch_time:.1f}s")

    # ---- Scheduler ----
    old_lr = current_lr
    scheduler.step(val_m['total'])
    new_lr = optimizer.param_groups[0]['lr']
    if new_lr != old_lr:
        logger.info(f"  LR reduced: {old_lr:.6f} -> {new_lr:.6f}")

    # ---- Save checkpoint (every epoch, for resume) ----
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_m['total'],
        'global_step': global_step,
        'history': history,
        'config': cfg,
    }, os.path.join(cfg['save_dir'], 'checkpoint_latest.pth'))

    # ---- Best model ----
    if val_m['total'] < best_val_loss:
        best_val_loss = val_m['total']
        patience_counter = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_m['total'],
            'global_step': global_step,
            'history': history,
            'config': cfg,
        }, os.path.join(cfg['save_dir'], 'model_best.pth'))
        logger.info(f"  *** Best model saved (val={val_m['total']:.6f}) ***")
    else:
        patience_counter += 1
        logger.info(f"  Patience: {patience_counter}/{cfg['patience']}")
        if patience_counter >= cfg['patience']:
            logger.info(f"\n  Early stopping at epoch {epoch + 1}")
            break

# ============================================================
# Extract embeddings
# ============================================================
logger.info(f"\n{'=' * 70}")
logger.info("Extracting embeddings from best model...")
logger.info("=" * 70)

import h5py

checkpoint = torch.load(os.path.join(cfg['save_dir'], 'model_best.pth'), map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

with h5py.File(cfg['h5_path'], 'r') as f:
    n_total = len(f['lengths'][:])

full_dataset = PackedHDF5Dataset.__new__(PackedHDF5Dataset)
full_dataset.h5_path = cfg['h5_path']
with h5py.File(cfg['h5_path'], 'r') as f:
    full_dataset.offsets       = f['offsets'][:]
    full_dataset.lengths       = f['lengths'][:]
    full_dataset.segment_types = f['segment_types'][:]
full_dataset.indices = np.arange(n_total)

full_loader = DataLoader(
    full_dataset, batch_size=cfg['batch_size'], shuffle=False,
    collate_fn=collate_fn, num_workers=cfg['num_workers'], pin_memory=True
)

all_z_final, all_z_B, all_z_E, all_seg_types = [], [], [], []

with torch.no_grad():
    for drv_pad, eng_pad, lengths, seg_types in tqdm(full_loader, desc="   Extract", ncols=80):
        drv_pad   = drv_pad.to(device)
        eng_pad   = eng_pad.to(device)
        lengths   = lengths.to(device)
        seg_types = seg_types.to(device)

        if use_amp:
            with torch.amp.autocast('cuda'):
                output = model(drv_pad, eng_pad, lengths, seg_types)
        else:
            output = model(drv_pad, eng_pad, lengths, seg_types)

        all_z_final.append(output['z_final'].cpu().numpy())
        all_z_B.append(output['z_B'].cpu().numpy())
        all_z_E.append(output['z_E'].cpu().numpy())
        all_seg_types.append(seg_types.cpu().numpy())

z_final       = np.vstack(all_z_final)
z_B           = np.vstack(all_z_B)
z_E           = np.vstack(all_z_E)
seg_types_all = np.concatenate(all_seg_types)

np.savez(os.path.join(cfg['save_dir'], 'latent_vectors.npz'),
         z_final=z_final, z_B=z_B, z_E=z_E, seg_types=seg_types_all)

logger.info(f"z_final: {z_final.shape}, z_B: {z_B.shape}, z_E: {z_E.shape}")

# ============================================================
# Paper-quality plots
# ============================================================
logger.info("Generating paper-quality plots...")

epochs_range = range(1, len(history['train_loss']) + 1)

# ---- Figure 1: Training curves (2x2) ----
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

ax = axes[0, 0]
ax.plot(epochs_range, history['train_loss'], 'b-', lw=2, label='Train')
ax.plot(epochs_range, history['val_loss'], 'r-', lw=2, label='Validation')
best_ep = np.argmin(history['val_loss']) + 1
ax.axvline(best_ep, color='green', ls='--', alpha=0.5, label=f'Best (epoch {best_ep})')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.set_title('(a) Total Loss')
ax.legend(fontsize=9); ax.grid(alpha=0.2)

ax = axes[0, 1]
ax.plot(epochs_range, history['train_drv'], 'b-', lw=2, label='Train Driving')
ax.plot(epochs_range, history['val_drv'], 'b--', lw=2, label='Val Driving')
ax.plot(epochs_range, history['train_eng'], 'r-', lw=2, label='Train Energy')
ax.plot(epochs_range, history['val_eng'], 'r--', lw=2, label='Val Energy')
ax.set_xlabel('Epoch'); ax.set_ylabel('Reconstruction Loss')
ax.set_title('(b) Driving vs Energy Reconstruction')
ax.legend(fontsize=8); ax.grid(alpha=0.2)

ax = axes[1, 0]
ax.plot(epochs_range, history['train_orth'], 'b-', lw=2, label='Train')
ax.plot(epochs_range, history['val_orth'], 'r-', lw=2, label='Validation')
ax.set_xlabel('Epoch'); ax.set_ylabel('Orthogonal Loss')
ax.set_title('(c) Orthogonal Constraint $\\|\\mathbf{z}_B \\cdot \\mathbf{z}_E\\|$')
ax.legend(fontsize=9); ax.grid(alpha=0.2)

ax = axes[1, 1]
ax.plot(epochs_range, history['lr'], 'g-', lw=2)
ax.set_xlabel('Epoch'); ax.set_ylabel('Learning Rate')
ax.set_title('(d) Learning Rate Schedule')
ax.set_yscale('log'); ax.grid(alpha=0.2)

plt.suptitle(f'Training Convergence (Best Val Loss = {best_val_loss:.4f} at Epoch {best_ep})',
             fontweight='bold', fontsize=14)
plt.tight_layout()
fig.savefig(os.path.join(cfg['save_dir'], 'fig_training_curves.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(cfg['save_dir'], 'fig_training_curves.pdf'), bbox_inches='tight')
plt.close(fig)
logger.info("   Saved: fig_training_curves.png/pdf")

# Add to TensorBoard
writer.add_figure('Paper/Training_Curves', fig, global_step=0)

# ---- Figure 2: Latent space analysis (2x2) ----
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# (a) Orthogonality
ax = axes[0, 0]
cos_sim = np.sum(z_B * z_E, axis=1) / (
    np.linalg.norm(z_B, axis=1) * np.linalg.norm(z_E, axis=1) + 1e-8
)
ax.hist(cos_sim, bins=150, alpha=0.8, color='steelblue', edgecolor='none', density=True)
ax.axvline(0, color='red', ls='--', lw=2, label='Ideal (orthogonal)')
mean_abs_cos = np.mean(np.abs(cos_sim))
ax.set_xlabel('Cosine Similarity (z_B, z_E)')
ax.set_ylabel('Density')
ax.set_title(f'(a) Channel Orthogonality (mean|cos|={mean_abs_cos:.4f})')
ax.legend(); ax.grid(alpha=0.2)

# (b) z_final norm distribution
ax = axes[0, 1]
z_norms = np.linalg.norm(z_final, axis=1)
ax.hist(z_norms, bins=150, alpha=0.8, color='coral', edgecolor='none', density=True)
ax.set_xlabel('L2 Norm')
ax.set_ylabel('Density')
ax.set_title(f'(b) Latent Vector Norms (mean={np.mean(z_norms):.2f}, std={np.std(z_norms):.2f})')
ax.grid(alpha=0.2)

# (c) PCA of z_final
from sklearn.decomposition import PCA
ax = axes[1, 0]
pca = PCA(n_components=2, random_state=cfg['seed'])
z_2d = pca.fit_transform(z_final)
ev = pca.explained_variance_ratio_

colors_map = {0: '#4C72B0', 1: '#DD8452'}
for st in [0, 1]:
    mask = seg_types_all == st
    label = 'Driving' if st == 0 else 'Idle'
    np.random.seed(cfg['seed'])
    if mask.sum() > 5000:
        sub = np.random.choice(np.where(mask)[0], 5000, replace=False)
    else:
        sub = np.where(mask)[0]
    ax.scatter(z_2d[sub, 0], z_2d[sub, 1], c=colors_map[st], s=3, alpha=0.3,
               label=f'{label} (n={mask.sum():,})', edgecolors='none')
ax.set_xlabel(f'PC1 ({ev[0]:.1%})')
ax.set_ylabel(f'PC2 ({ev[1]:.1%})')
ax.set_title('(c) Latent Space (PCA, colored by segment type)')
ax.legend(markerscale=5, fontsize=9); ax.grid(alpha=0.2)

# (d) Dimension activation
ax = axes[1, 1]
dim_std = np.std(z_final, axis=0)
dim_idx = np.argsort(dim_std)[::-1]
ax.bar(range(len(dim_std)), dim_std[dim_idx], color='mediumpurple', edgecolor='none', alpha=0.8)
n_dead = (dim_std < 1e-3).sum()
ax.axhline(1e-3, color='red', ls='--', lw=1, label=f'Dead threshold (n_dead={n_dead})')
ax.set_xlabel('Dimension (sorted)')
ax.set_ylabel('Std Dev')
ax.set_title(f'(d) Latent Dimension Activation ({z_final.shape[1]}d, {z_final.shape[1]-n_dead} active)')
ax.legend(fontsize=9); ax.grid(alpha=0.2)

plt.suptitle('Latent Space Analysis', fontweight='bold', fontsize=14)
plt.tight_layout()
fig.savefig(os.path.join(cfg['save_dir'], 'fig_latent_analysis.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(cfg['save_dir'], 'fig_latent_analysis.pdf'), bbox_inches='tight')
plt.close(fig)
logger.info("   Saved: fig_latent_analysis.png/pdf")

# ---- Figure 3: Embedding projections (TensorBoard) ----
try:
    n_embed = min(10000, len(z_final))
    embed_idx = np.random.choice(len(z_final), n_embed, replace=False)
    metadata = [f'{"Drv" if s == 0 else "Idle"}' for s in seg_types_all[embed_idx]]
    writer.add_embedding(
        torch.FloatTensor(z_final[embed_idx]),
        metadata=metadata,
        tag='z_final',
        global_step=0
    )
    writer.add_embedding(
        torch.FloatTensor(z_B[embed_idx]),
        metadata=metadata,
        tag='z_B (Driving Channel)',
        global_step=0
    )
    writer.add_embedding(
        torch.FloatTensor(z_E[embed_idx]),
        metadata=metadata,
        tag='z_E (Energy Channel)',
        global_step=0
    )
    logger.info("   TensorBoard embeddings added (z_final, z_B, z_E)")
except Exception as e:
    logger.warning(f"   Embedding projection failed: {e}")

# ============================================================
# Save summary
# ============================================================
summary = {
    'best_val_loss': float(best_val_loss),
    'best_epoch': int(best_ep),
    'total_epochs': len(history['train_loss']),
    'n_params': n_params,
    'z_final_shape': list(z_final.shape),
    'mean_abs_cos_sim': float(mean_abs_cos),
    'n_dead_dims': int(n_dead),
    'n_active_dims': int(z_final.shape[1] - n_dead),
    'final_train_loss': history['train_loss'][-1],
    'final_val_loss': history['val_loss'][-1],
    'config': cfg,
}
with open(os.path.join(cfg['save_dir'], 'training_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2, default=str)

writer.close()

logger.info(f"\n{'=' * 70}")
logger.info("TRAINING COMPLETE")
logger.info(f"{'=' * 70}")
logger.info(f"Best val loss: {best_val_loss:.6f} (epoch {best_ep})")
logger.info(f"z_final: {z_final.shape} ({z_final.shape[1]-n_dead} active dims)")
logger.info(f"Orthogonality: mean|cos| = {mean_abs_cos:.4f}")
logger.info(f"Files:")
for fn in sorted(os.listdir(cfg['save_dir'])):
    fp = os.path.join(cfg['save_dir'], fn)
    if os.path.isfile(fp):
        logger.info(f"   {fn:<45} {os.path.getsize(fp)/1024:>8.1f} KB")
logger.info(f"\nTensorBoard: tensorboard --logdir {cfg['log_dir']}")
logger.info(f"Resume: set resume_from='{os.path.join(cfg['save_dir'], 'checkpoint_latest.pth')}'")
logger.info("=" * 70)