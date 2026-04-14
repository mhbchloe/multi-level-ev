"""
完整双通道建模流程
- 141k完整行程数据
- 双通道自编码器（Direct Concatenation）
- 交叉注意力机制
- 完整评估和可视化
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
import os
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("🔥 Full Dual-Channel Pipeline on 141k Trip Data")
print("="*70)

# ==================== 配置 ====================
CONFIG = {
    'latent_dim': 8,
    'batch_size': 512,
    'epochs': 40,
    'lr': 0.0001,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'k_values': [3, 4, 5],
    'output_dir': './results/full_dual_channel'
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)

print(f"\n⚙️  Configuration:")
for key, value in CONFIG.items():
    if key != 'output_dir':
        print(f"   {key}: {value}")

# ==================== 模型架构 ====================
class CrossChannelAttention(nn.Module):
    """交叉通道注意力机制"""
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
        
    def forward(self, x1, x2):
        """x1查询x2的信息"""
        q = self.query(x1)
        k = self.key(x2)
        v = self.value(x2)
        
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        return out

class DualChannelAutoencoder(nn.Module):
    """双通道自编码器（Direct Concatenation融合）"""
    def __init__(self, driving_dim, energy_dim, latent_dim=8):
        super().__init__()
        
        print(f"\n🏗️  Building Dual-Channel Autoencoder:")
        print(f"   Driving channel: {driving_dim}D → {latent_dim}D")
        print(f"   Energy channel: {energy_dim}D → {latent_dim}D")
        print(f"   Cross-attention: {latent_dim}D ↔ {latent_dim}D")
        print(f"   Fused latent: {latent_dim * 4}D → {latent_dim * 2}D")
        
        # 驾驶通道编码器
        self.driving_encoder = nn.Sequential(
            nn.Linear(driving_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, latent_dim),
            nn.ReLU()
        )
        
        # 能量通道编码器
        self.energy_encoder = nn.Sequential(
            nn.Linear(energy_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, latent_dim),
            nn.ReLU()
        )
        
        # 交叉注意力（双向）
        self.cross_attn_d2e = CrossChannelAttention(latent_dim)  # 驾驶→能量
        self.cross_attn_e2d = CrossChannelAttention(latent_dim)  # 能量→驾驶
        
        # 融合层（直接拼接4个向量）
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 4, latent_dim * 3),
            nn.LayerNorm(latent_dim * 3),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(latent_dim * 3, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.ReLU()
        )
        
        # 驾驶通道解码器
        self.driving_decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, driving_dim)
        )
        
        # 能量通道解码器
        self.energy_decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, energy_dim)
        )
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"   Total parameters: {total_params:,}")
    
    def forward(self, driving_features, energy_features):
        # 独立编码
        driving_latent = self.driving_encoder(driving_features)
        energy_latent = self.energy_encoder(energy_features)
        
        # 交叉注意力
        driving_attended = self.cross_attn_e2d(driving_latent, energy_latent)
        energy_attended = self.cross_attn_d2e(energy_latent, driving_latent)
        
        # 直接拼接融合
        combined = torch.cat([
            driving_latent,      # 原始驾驶
            driving_attended,    # 能量增强的驾驶
            energy_latent,       # 原始能量
            energy_attended      # 驾驶增强的能量
        ], dim=1)
        
        fused_latent = self.fusion(combined)
        
        # 解码重构
        driving_recon = self.driving_decoder(driving_latent)
        energy_recon = self.energy_decoder(energy_latent)
        
        return driving_recon, energy_recon, fused_latent
    
    def encode(self, driving_features, energy_features):
        """仅编码（用于特征提取）"""
        driving_latent = self.driving_encoder(driving_features)
        energy_latent = self.energy_encoder(energy_features)
        
        driving_attended = self.cross_attn_e2d(driving_latent, energy_latent)
        energy_attended = self.cross_attn_d2e(energy_latent, driving_latent)
        
        combined = torch.cat([
            driving_latent, driving_attended,
            energy_latent, energy_attended
        ], dim=1)
        
        fused_latent = self.fusion(combined)
        return fused_latent

# ==================== 1. 加载数据 ====================
print(f"\n{'='*70}")
print("📂 Loading Data")
print("="*70)

features_df = pd.read_csv('./results/reloaded_full/trip_features_full.csv')
print(f"✅ Loaded: {features_df.shape}")

print(f"\n🔍 Data quality check:")
print(f"   Total trips: {len(features_df):,}")
print(f"   Vehicles: {features_df['vehicle_id'].nunique():,}")
print(f"   Avg trips/vehicle: {len(features_df)/features_df['vehicle_id'].nunique():.1f}")

# 数据清洗
print(f"\n🧹 Cleaning data...")
original_size = len(features_df)

features_df = features_df[
    (features_df['duration_minutes'] > 2) & (features_df['duration_minutes'] < 300) &
    (features_df['speed_mean'] >= 0) & (features_df['speed_mean'] < 150) &
    (features_df['distance_total'] >= 0) & (features_df['distance_total'] < 500) &
    (features_df['soc_drop_total'] >= 0) & (features_df['soc_drop_total'] < 100)
]

removed = original_size - len(features_df)
print(f"   Removed {removed:,} outliers ({removed/original_size*100:.1f}%)")
print(f"   Final dataset: {len(features_df):,} trips")

# ==================== 2. 准备双通道数据 ====================
print(f"\n{'='*70}")
print("🔧 Preparing Dual-Channel Features")
print("="*70)

driving_features = [
    'speed_mean', 'speed_max', 'speed_std', 'speed_median',
    'acc_mean', 'acc_std', 'harsh_accel', 'harsh_decel',
    'distance_total', 'duration_minutes', 'moving_ratio',
    'low_speed_ratio', 'medium_speed_ratio', 'high_speed_ratio'
]

energy_features = [
    'soc_drop_total', 'soc_mean', 'soc_std',
    'voltage_mean', 'voltage_std',
    'current_mean', 'current_max',
    'power_mean', 'power_max', 'power_std',
    'energy_consumption_total',
    'charging_ratio', 'regen_braking_ratio'
]

available_driving = [f for f in driving_features if f in features_df.columns]
available_energy = [f for f in energy_features if f in features_df.columns]

print(f"\n✅ Driving features ({len(available_driving)}):")
print(f"   {', '.join(available_driving)}")

print(f"\n✅ Energy features ({len(available_energy)}):")
print(f"   {', '.join(available_energy)}")

X_driving = np.nan_to_num(features_df[available_driving].values)
X_energy = np.nan_to_num(features_df[available_energy].values)

# 使用RobustScaler（对异常值更鲁棒）
print(f"\n📊 Normalizing with RobustScaler...")
scaler_driving = RobustScaler()
scaler_energy = RobustScaler()

X_driving = scaler_driving.fit_transform(X_driving)
X_energy = scaler_energy.fit_transform(X_energy)

# 裁剪极端值
X_driving = np.clip(X_driving, -5, 5)
X_energy = np.clip(X_energy, -5, 5)

print(f"✅ Driving channel: {X_driving.shape}")
print(f"✅ Energy channel: {X_energy.shape}")

# 检查数据质量
assert not np.isnan(X_driving).any(), "❌ Driving data contains NaN"
assert not np.isnan(X_energy).any(), "❌ Energy data contains NaN"
print(f"✅ Data is clean (no NaN/Inf)")

# ==================== 3. 构建和训练模型 ====================
print(f"\n{'='*70}")
print("🚀 Training Dual-Channel Model")
print("="*70)

device = CONFIG['device']
print(f"Device: {device}")

model = DualChannelAutoencoder(
    driving_dim=X_driving.shape[1],
    energy_dim=X_energy.shape[1],
    latent_dim=CONFIG['latent_dim']
)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

best_loss = float('inf')
patience = 0
history = {'loss': [], 'driving_loss': [], 'energy_loss': []}

print(f"\nTraining for {CONFIG['epochs']} epochs (batch size: {CONFIG['batch_size']})...")

for epoch in range(CONFIG['epochs']):
    model.train()
    epoch_losses = []
    epoch_driving_losses = []
    epoch_energy_losses = []
    
    indices = np.random.permutation(len(X_driving))
    
    # 进度条
    pbar = tqdm(range(0, len(X_driving), CONFIG['batch_size']),
                desc=f"Epoch {epoch+1:2d}/{CONFIG['epochs']}",
                leave=False)
    
    for i in pbar:
        batch_indices = indices[i:i+CONFIG['batch_size']]
        
        d_batch = torch.FloatTensor(X_driving[batch_indices]).to(device)
        e_batch = torch.FloatTensor(X_energy[batch_indices]).to(device)
        
        optimizer.zero_grad()
        
        d_recon, e_recon, _ = model(d_batch, e_batch)
        
        d_loss = F.mse_loss(d_recon, d_batch)
        e_loss = F.mse_loss(e_recon, e_batch)
        total_loss = d_loss + e_loss
        
        if torch.isnan(total_loss):
            continue
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        
        epoch_losses.append(total_loss.item())
        epoch_driving_losses.append(d_loss.item())
        epoch_energy_losses.append(e_loss.item())
        
        pbar.set_postfix({
            'loss': f'{total_loss.item():.4f}',
            'drv': f'{d_loss.item():.4f}',
            'eng': f'{e_loss.item():.4f}'
        })
    
    avg_loss = np.mean(epoch_losses)
    avg_d_loss = np.mean(epoch_driving_losses)
    avg_e_loss = np.mean(epoch_energy_losses)
    
    history['loss'].append(avg_loss)
    history['driving_loss'].append(avg_d_loss)
    history['energy_loss'].append(avg_e_loss)
    
    scheduler.step(avg_loss)
    
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d}/{CONFIG['epochs']} | "
              f"Loss: {avg_loss:.4f} | "
              f"Driving: {avg_d_loss:.4f} | "
              f"Energy: {avg_e_loss:.4f}")
    
    if avg_loss < best_loss:
        best_loss = avg_loss
        patience = 0
        torch.save(model.state_dict(), f"{CONFIG['output_dir']}/model_best.pth")
    else:
        patience += 1
        if patience >= 12:
            print(f"\n⚠️  Early stopping at epoch {epoch+1}")
            break

# 加载最佳模型
model.load_state_dict(torch.load(f"{CONFIG['output_dir']}/model_best.pth", weights_only=False))
print(f"\n✅ Training complete! Best loss: {best_loss:.4f}")

# 保存训练历史
history_df = pd.DataFrame(history)
history_df.to_csv(f"{CONFIG['output_dir']}/training_history.csv", index=False)

# ==================== 4. 提取潜在特征 ====================
print(f"\n{'='*70}")
print("🔍 Extracting Latent Features")
print("="*70)

model.eval()
latent_features = []

with torch.no_grad():
    for i in tqdm(range(0, len(X_driving), CONFIG['batch_size']), desc="Extracting"):
        d_batch = torch.FloatTensor(X_driving[i:i+CONFIG['batch_size']]).to(device)
        e_batch = torch.FloatTensor(X_energy[i:i+CONFIG['batch_size']]).to(device)
        
        latent = model.encode(d_batch, e_batch)
        latent_features.append(latent.cpu().numpy())

latent_features = np.vstack(latent_features)
print(f"✅ Extracted latent features: {latent_features.shape}")

# 保存
np.save(f"{CONFIG['output_dir']}/latent_features.npy", latent_features)

# ==================== 5. 聚类 ====================
print(f"\n{'='*70}")
print("🎯 K-Means Clustering")
print("="*70)

clustering_results = {}

for k in CONFIG['k_values']:
    print(f"\n{'='*50}")
    print(f"K = {k}")
    print("="*50)
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = kmeans.fit_predict(latent_features)
    
    # 评估指标
    sil = silhouette_score(latent_features, labels)
    ch = calinski_harabasz_score(latent_features, labels)
    db = davies_bouldin_score(latent_features, labels)
    
    print(f"📊 Metrics:")
    print(f"   Silhouette: {sil:.3f}")
    print(f"   CH Score: {ch:.2f}")
    print(f"   DB Score: {db:.3f}")
    
    # 簇分布
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    percentages = cluster_counts.values / len(labels) * 100
    cv = percentages.std() / percentages.mean()
    
    print(f"\n📈 Distribution:")
    for cid, count in cluster_counts.items():
        pct = count / len(labels) * 100
        print(f"   Cluster {cid}: {count:7,} ({pct:5.1f}%)")
    
    print(f"\n   CV: {cv:.3f} (lower = more balanced)")
    print(f"   Range: {percentages.min():.1f}% - {percentages.max():.1f}%")
    
    # 簇特征统计
    features_df[f'cluster_k{k}'] = labels
    
    print(f"\n🎯 Cluster Characteristics:")
    for cid in range(k):
        cluster_data = features_df[features_df[f'cluster_k{k}'] == cid]
        print(f"\n  Cluster {cid} ({len(cluster_data):,} trips, {len(cluster_data)/len(features_df)*100:.1f}%):")
        print(f"    Speed: {cluster_data['speed_mean'].mean():.1f} ± {cluster_data['speed_mean'].std():.1f} km/h")
        print(f"    Distance: {cluster_data['distance_total'].mean():.1f} ± {cluster_data['distance_total'].std():.1f} km")
        print(f"    Duration: {cluster_data['duration_minutes'].mean():.1f} ± {cluster_data['duration_minutes'].std():.1f} min")
        print(f"    SOC Drop: {cluster_data['soc_drop_total'].mean():.1f} ± {cluster_data['soc_drop_total'].std():.1f} %")
        print(f"    Moving: {cluster_data['moving_ratio'].mean()*100:.1f}%")
    
    # 保存聚类结果
    features_df[['trip_id', 'vehicle_id', f'cluster_k{k}']].to_csv(
        f"{CONFIG['output_dir']}/clustered_k{k}.csv", index=False
    )
    
    clustering_results[k] = {
        'labels': labels,
        'silhouette': sil,
        'ch_score': ch,
        'db_score': db,
        'cv': cv,
        'distribution': cluster_counts.to_dict()
    }

# 保存完整结果
features_df.to_csv(f"{CONFIG['output_dir']}/full_results.csv", index=False)

# ==================== 6. 可视化 ====================
print(f"\n{'='*70}")
print("📊 Creating Visualizations")
print("="*70)

# 6.1 训练历史
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

ax.plot(history['loss'], label='Total Loss', linewidth=2)
ax.plot(history['driving_loss'], label='Driving Loss', linewidth=2, alpha=0.7)
ax.plot(history['energy_loss'], label='Energy Loss', linewidth=2, alpha=0.7)

ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax.set_title('Training History', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{CONFIG['output_dir']}/training_history.png", dpi=300, bbox_inches='tight')
print(f"✅ Training history plot saved")

# 6.2 聚类结果对比（K=3,4,5）
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

for idx, k in enumerate(CONFIG['k_values']):
    labels = clustering_results[k]['labels']
    colors = sns.color_palette('Set2', k)
    
    # PCA降维到2D
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_features)
    
    # 散点图
    ax = axes[0, idx]
    for cid in range(k):
        mask = labels == cid
        ax.scatter(latent_2d[mask, 0], latent_2d[mask, 1],
                  alpha=0.4, s=5, c=[colors[cid]], label=f'C{cid}')
    
    ax.set_title(f'K={k} (Sil={clustering_results[k]["silhouette"]:.3f})',
                fontsize=13, fontweight='bold')
    ax.set_xlabel('PC1', fontsize=11)
    ax.set_ylabel('PC2', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    # 簇大小分布
    ax = axes[1, idx]
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    bars = ax.bar(cluster_counts.index, cluster_counts.values,
                 color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_title(f'K={k} (CV={clustering_results[k]["cv"]:.3f})',
                fontsize=13, fontweight='bold')
    ax.set_xlabel('Cluster', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    for bar, count in zip(bars, cluster_counts.values):
        pct = count / len(labels) * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
               f'{count:,}\n({pct:.1f}%)',
               ha='center', va='bottom', fontsize=9)

plt.suptitle('Dual-Channel Clustering Results', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{CONFIG['output_dir']}/clustering_comparison.png", dpi=300, bbox_inches='tight')
print(f"✅ Clustering comparison plot saved")

# 6.3 K=4的详细分析（推荐）
if 4 in CONFIG['k_values']:
    labels_k4 = clustering_results[4]['labels']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    colors_k4 = sns.color_palette('Set2', 4)
    
    # 速度 vs 距离
    ax = axes[0, 0]
    for cid in range(4):
        cluster_data = features_df[features_df['cluster_k4'] == cid]
        ax.scatter(cluster_data['speed_mean'], cluster_data['distance_total'],
                  alpha=0.4, s=8, c=[colors_k4[cid]], label=f'C{cid}')
    ax.set_xlabel('Speed (km/h)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Distance (km)', fontsize=12, fontweight='bold')
    ax.set_title('Speed vs Distance', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 100)
    
    # 速度分布
    ax = axes[0, 1]
    speed_data = [features_df[features_df['cluster_k4'] == c]['speed_mean'] for c in range(4)]
    parts = ax.violinplot(speed_data, positions=range(4), showmeans=True, showmedians=True)
    for pc, color in zip(parts['bodies'], colors_k4):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speed (km/h)', fontsize=12, fontweight='bold')
    ax.set_title('Speed Distribution', fontsize=14, fontweight='bold')
    ax.set_xticks(range(4))
    ax.grid(axis='y', alpha=0.3)
    
    # 持续时间分布
    ax = axes[0, 2]
    duration_data = [features_df[features_df['cluster_k4'] == c]['duration_minutes'] for c in range(4)]
    bp = ax.boxplot(duration_data, positions=range(4), patch_artist=True, showfliers=False)
    for patch, color in zip(bp['boxes'], colors_k4):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
    ax.set_ylabel('Duration (min)', fontsize=12, fontweight='bold')
    ax.set_title('Duration Distribution', fontsize=14, fontweight='bold')
    ax.set_xticks(range(4))
    ax.grid(axis='y', alpha=0.3)
    
    # SOC消耗
    ax = axes[1, 0]
    soc_data = [features_df[features_df['cluster_k4'] == c]['soc_drop_total'] for c in range(4)]
    bp2 = ax.boxplot(soc_data, positions=range(4), patch_artist=True, showfliers=False)
    for patch, color in zip(bp2['boxes'], colors_k4):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
    ax.set_ylabel('SOC Drop (%)', fontsize=12, fontweight='bold')
    ax.set_title('Energy Consumption', fontsize=14, fontweight='bold')
    ax.set_xticks(range(4))
    ax.grid(axis='y', alpha=0.3)
    
    # 移动比例
    ax = axes[1, 1]
    moving_data = [features_df[features_df['cluster_k4'] == c]['moving_ratio'] for c in range(4)]
    bp3 = ax.boxplot(moving_data, positions=range(4), patch_artist=True, showfliers=False)
    for patch, color in zip(bp3['boxes'], colors_k4):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
    ax.set_ylabel('Moving Ratio', fontsize=12, fontweight='bold')
    ax.set_title('Movement Activity', fontsize=14, fontweight='bold')
    ax.set_xticks(range(4))
    ax.grid(axis='y', alpha=0.3)
    
    # 簇大小
    ax = axes[1, 2]
    cluster_counts_k4 = pd.Series(labels_k4).value_counts().sort_index()
    bars = ax.bar(cluster_counts_k4.index, cluster_counts_k4.values,
                 color=colors_k4, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Cluster Sizes', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, count in zip(bars, cluster_counts_k4.values):
        pct = count / len(labels_k4) * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
               f'{count:,}\n({pct:.1f}%)',
               ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('Detailed Analysis (K=4)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{CONFIG['output_dir']}/detailed_analysis_k4.png", dpi=300, bbox_inches='tight')
    print(f"✅ Detailed K=4 analysis plot saved")

plt.close('all')

# ==================== 7. 总结报告 ====================
print(f"\n{'='*70}")
print("📋 Final Summary Report")
print("="*70)

print(f"\n🎯 Model Architecture:")
print(f"   Driving channel: {X_driving.shape[1]}D → 8D")
print(f"   Energy channel: {X_energy.shape[1]}D → 8D")
print(f"   Cross-attention: Bidirectional")
print(f"   Fusion method: Direct Concatenation (32D → 16D)")
print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")

print(f"\n📊 Training Results:")
print(f"   Best reconstruction loss: {best_loss:.4f}")
print(f"   Epochs trained: {len(history['loss'])}")
print(f"   Training samples: {len(X_driving):,}")

print(f"\n🎯 Clustering Results:")

# 创建对比表
comparison_data = []
for k in CONFIG['k_values']:
    res = clustering_results[k]
    comparison_data.append({
        'K': k,
        'Silhouette': f"{res['silhouette']:.3f}",
        'CH Score': f"{res['ch_score']:.1f}",
        'DB Score': f"{res['db_score']:.3f}",
        'CV': f"{res['cv']:.3f}",
        'Distribution': ', '.join([f"{v}({v/len(labels)*100:.1f}%)" 
                                  for v in res['distribution'].values()])
    })

comparison_df = pd.DataFrame(comparison_data)
print(f"\n{comparison_df.to_string(index=False)}")

# 推荐最佳K
best_k = min(CONFIG['k_values'], key=lambda k: clustering_results[k]['cv'])
print(f"\n💡 Recommended: K={best_k}")
print(f"   Reason: Most balanced distribution (CV={clustering_results[best_k]['cv']:.3f})")
print(f"   Silhouette: {clustering_results[best_k]['silhouette']:.3f}")

print(f"\n📁 Output Files:")
print(f"   Model: {CONFIG['output_dir']}/model_best.pth")
print(f"   Latent features: {CONFIG['output_dir']}/latent_features.npy")
print(f"   Full results: {CONFIG['output_dir']}/full_results.csv")
print(f"   Clustering (K={best_k}): {CONFIG['output_dir']}/clustered_k{best_k}.csv")
print(f"   Visualizations: {CONFIG['output_dir']}/*.png")

print("\n" + "="*70)
print("✅ Full Dual-Channel Pipeline Complete!")
print("="*70)

print(f"\n🎯 Next Steps:")
print(f"   1. Review visualizations in {CONFIG['output_dir']}/")
print(f"   2. Analyze cluster characteristics in full_results.csv")
print(f"   3. Run verification: python verify_clustering_quality_fixed.py")