"""
======================================================================
🎯 GRU Clustering K=4 - Cross-Attention Version
======================================================================
使用双向交叉注意力机制建模驾驶-能量的耦合关系
======================================================================
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from pathlib import Path
import json
import warnings
from tqdm import tqdm

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("🎯 GRU Clustering K=4 - Cross-Attention Version")
print("="*70)


# ==================== Dataset ====================
class VariableLengthDataset(Dataset):
    def __init__(self, driving_sequences, energy_sequences, lengths):
        self.driving = driving_sequences
        self.energy = energy_sequences
        self.lengths = lengths
    
    def __len__(self):
        return len(self.driving)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.driving[idx]),
            torch.FloatTensor(self.energy[idx]),
            self.lengths[idx]
        )


def collate_variable_length(batch):
    driving_seqs, energy_seqs, lengths = zip(*batch)
    lengths = torch.LongTensor(lengths)
    sorted_indices = torch.argsort(lengths, descending=True)
    
    lengths_sorted = lengths[sorted_indices]
    driving_sorted = [driving_seqs[i] for i in sorted_indices]
    energy_sorted = [energy_seqs[i] for i in sorted_indices]
    
    driving_padded = pad_sequence(driving_sorted, batch_first=True)
    energy_padded = pad_sequence(energy_sorted, batch_first=True)
    
    return driving_padded, energy_padded, lengths_sorted, sorted_indices


# ==================== GRU Encoder ====================
class GRUEncoderVarLen(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers,
                         batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        self.hidden_dim = hidden_dim
        
    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        _, h_n = self.gru(packed)
        return h_n[-1]


# ==================== Cross-Channel Attention ====================
class CrossChannelAttention(nn.Module):
    """
    交叉注意力机制：让一个通道关注另一个通道
    
    数学表达：
    Q = W_q * x1  (query from channel 1)
    K = W_k * x2  (key from channel 2)
    V = W_v * x2  (value from channel 2)
    
    Attention = softmax(Q * K^T / sqrt(d))
    Output = Attention * V
    """
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
    
    def forward(self, x1, x2):
        """
        x1: 查询通道 [batch, dim]
        x2: 被关注的通道 [batch, dim]
        返回: x1 经过 x2 增强后的特征 [batch, dim]
        """
        q = self.query(x1)  # [batch, dim]
        k = self.key(x2)    # [batch, dim]
        v = self.value(x2)  # [batch, dim]
        
        # 计算注意力分数
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [batch, batch]
        attn = torch.softmax(attn, dim=-1)
        
        # 加权求和
        out = torch.matmul(attn, v)  # [batch, dim]
        
        return out


# ==================== Dual-Channel GRU with Cross-Attention ====================
class DualChannelGRU_CrossAttn(nn.Module):
    """
    双通道GRU + 交叉注意力融合
    
    架构：
    1. 两个GRU编码器分别提取驾驶和能量的时序特征
    2. 交叉注意力：驾驶关注能量 + 能量关注驾驶
    3. 拼接原始特征和增强特征
    4. MLP融合
    """
    def __init__(self, driving_dim, energy_dim, latent_dim):
        super().__init__()
        
        print("\n🏗️  Building Dual-Channel GRU with Cross-Attention...")
        
        # 编码器
        self.driving_encoder = GRUEncoderVarLen(driving_dim, latent_dim)
        self.energy_encoder = GRUEncoderVarLen(energy_dim, latent_dim)
        
        # 交叉注意力（双向）
        self.cross_attn_d2e = CrossChannelAttention(latent_dim)  # 驾驶→能量
        self.cross_attn_e2d = CrossChannelAttention(latent_dim)  # 能量→驾驶
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 4, latent_dim * 3),  # 64 → 48
            nn.LayerNorm(latent_dim * 3),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(latent_dim * 3, latent_dim * 2),  # 48 → 32
            nn.LayerNorm(latent_dim * 2),
            nn.ReLU()
        )
        
        # 解码器
        self.decoder = nn.Linear(latent_dim * 2, driving_dim + energy_dim)
        
        # 计算参数量
        total_params = sum(p.numel() for p in self.parameters())
        print(f"   Total parameters: {total_params:,}")
        
        # 分解参数量
        encoder_params = sum(p.numel() for p in self.driving_encoder.parameters()) + \
                        sum(p.numel() for p in self.energy_encoder.parameters())
        attn_params = sum(p.numel() for p in self.cross_attn_d2e.parameters()) + \
                     sum(p.numel() for p in self.cross_attn_e2d.parameters())
        fusion_params = sum(p.numel() for p in self.fusion.parameters())
        
        print(f"   ├─ GRU Encoders: {encoder_params:,}")
        print(f"   ├─ Cross-Attention: {attn_params:,}")
        print(f"   └─ Fusion MLP: {fusion_params:,}")
        
    def forward(self, driving, energy, lengths):
        # 1. 编码
        driving_feat = self.driving_encoder(driving, lengths)  # [batch, 16]
        energy_feat = self.energy_encoder(energy, lengths)     # [batch, 16]
        
        # 2. 交叉注意力
        # 驾驶关注能量：驾驶行为如何受能量状态影响
        driving_attended = self.cross_attn_e2d(driving_feat, energy_feat)  # [batch, 16]
        
        # 能量关注驾驶：能量消耗如何受驾驶行为影响
        energy_attended = self.cross_attn_d2e(energy_feat, driving_feat)   # [batch, 16]
        
        # 3. 拼接所有特征
        combined = torch.cat([
            driving_feat,       # 原始驾驶特征
            driving_attended,   # 驾驶受能量影响的特征
            energy_feat,        # 原始能量特征
            energy_attended     # 能量受驾驶影响的特征
        ], dim=1)  # [batch, 64]
        
        # 4. MLP融合
        fused = self.fusion(combined)  # [batch, 32]
        
        # 5. 解码（用于训练）
        reconstructed = self.decoder(fused)  # [batch, 5]
        
        return reconstructed, fused  # 返回重构和融合特征


# ==================== Load Data ====================
def load_data(data_dir, max_samples=100000):
    print("\n" + "="*70)
    print("📂 Loading Data")
    print("="*70)
    
    data_path = Path(data_dir)
    
    driving = np.load(data_path / 'driving_sequences.npy', allow_pickle=True)
    energy = np.load(data_path / 'energy_sequences.npy', allow_pickle=True)
    lengths = np.load(data_path / 'seq_lengths.npy')
    
    valid_mask = (lengths >= 10) & (lengths <= 1000)
    driving = driving[valid_mask]
    energy = energy[valid_mask]
    lengths = lengths[valid_mask]
    
    if len(driving) > max_samples:
        indices = np.random.choice(len(driving), max_samples, replace=False)
        driving = driving[indices]
        energy = energy[indices]
        lengths = lengths[indices]
    
    print(f"✅ Data loaded:")
    print(f"   Samples: {len(driving):,}")
    print(f"   Driving features: {driving[0].shape[1]}D")
    print(f"   Energy features: {energy[0].shape[1]}D")
    
    return driving, energy, lengths


# ==================== Train Model ====================
def train_model(model, train_loader, model_path='./results/gru_model_k4_crossattn.pth', epochs=10):
    """训练模型"""
    
    # 检查是否已有模型
    if Path(model_path).exists():
        print(f"\n✅ Found trained model: {model_path}")
        print("   Loading model...")
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    
    print("\n" + "="*70)
    print("🚀 Training GRU Model with Cross-Attention")
    print("="*70)
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        epoch_loss = 0
        batch_count = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        
        for driving, energy, lengths, _ in pbar:
            driving = driving.to(device)
            energy = energy.to(device)
            
            optimizer.zero_grad()
            
            reconstructed, _ = model(driving, energy, lengths)
            
            # 构造目标
            targets = []
            for idx, length in enumerate(lengths):
                d_last = driving[idx, length-1, :]
                e_last = energy[idx, length-1, :]
                targets.append(torch.cat([d_last, e_last]))
            targets = torch.stack(targets)
            
            loss = criterion(reconstructed, targets)
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        if batch_count > 0:
            avg_loss = epoch_loss / batch_count
            best_loss = min(best_loss, avg_loss)
            
            if (epoch + 1) % 2 == 0:
                print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f} | Best: {best_loss:.4f}")
    
    print(f"\n✅ Training completed | Best loss: {best_loss:.4f}")
    
    torch.save(model.state_dict(), model_path)
    print(f"💾 Model saved: {model_path}")
    
    return model


# ==================== Extract Features ====================
def extract_features(model, loader):
    print("\n🔍 Extracting features with Cross-Attention...")
    
    model.eval()
    all_features = []
    all_indices = []
    
    with torch.no_grad():
        for driving, energy, lengths, sorted_idx in tqdm(loader, desc="Extracting", leave=False):
            driving = driving.to(device)
            energy = energy.to(device)
            _, feat = model(driving, energy, lengths)
            all_features.append(feat.cpu().numpy())
            all_indices.append(sorted_idx.numpy())
    
    features = np.vstack(all_features)
    indices = np.concatenate(all_indices)
    unsort_indices = np.argsort(indices)
    features = features[unsort_indices]
    
    print(f"✅ Features extracted: {features.shape}")
    return features


# ==================== Clustering ====================
def perform_clustering_k4(features):
    print("\n" + "="*70)
    print("🎯 K=4 Clustering")
    print("="*70)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=20)
    labels = kmeans.fit_predict(features)
    
    sil = silhouette_score(features, labels)
    unique, counts = np.unique(labels, return_counts=True)
    cv = np.std(counts) / np.mean(counts)
    
    print(f"\n✅ Clustering completed:")
    print(f"   Silhouette: {sil:.3f}")
    print(f"   CV: {cv:.3f}")
    print(f"\n   Distribution:")
    for cluster_id, count in zip(unique, counts):
        pct = count / len(labels) * 100
        print(f"      Cluster {cluster_id}: {count:6,} ({pct:5.1f}%)")
    
    return labels, kmeans, sil, cv


# ==================== Extract Cluster Stats ====================
def extract_cluster_features(driving_seqs, energy_seqs, labels):
    print("\n📊 Extracting cluster statistics...")
    
    cluster_stats = []
    
    for cluster_id in range(4):
        cluster_mask = (labels == cluster_id)
        cluster_driving = driving_seqs[cluster_mask]
        cluster_energy = energy_seqs[cluster_mask]
        
        stats = {}
        
        all_spd = np.concatenate([seq[:, 0] for seq in cluster_driving])
        all_acc = np.concatenate([seq[:, 1] for seq in cluster_driving])
        
        stats['Avg Speed'] = np.mean(all_spd)
        stats['Max Speed'] = np.percentile(all_spd, 95)
        stats['Avg Power'] = np.mean(np.abs(
            np.concatenate([seq[:, 1] * seq[:, 2] for seq in cluster_energy])
        ))
        stats['Trip Length'] = np.mean([len(seq) for seq in cluster_driving])
        
        cluster_stats.append(stats)
    
    return cluster_stats


# ==================== Plot Comparison ====================
def plot_comparison(cluster_stats_simple, cluster_stats_crossattn, 
                   sil_simple, sil_crossattn):
    """对比简单拼接 vs Cross-Attention"""
    print("\n📊 Generating comparison plot...")
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    features = ['Avg Speed', 'Max Speed', 'Avg Power', 'Trip Length']
    N = len(features)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    # 简单拼接版本
    ax = axes[0]
    ax = plt.subplot(121, projection='polar')
    
    for i, stats in enumerate(cluster_stats_simple):
        values = [stats[f] for f in features]
        # 归一化
        max_vals = [max([s[f] for s in cluster_stats_simple]) for f in features]
        values_norm = [v/m if m > 0 else 0 for v, m in zip(values, max_vals)]
        values_norm += values_norm[:1]
        
        ax.plot(angles, values_norm, 'o-', linewidth=3, 
               label=f'C{i}', color=colors[i], markersize=8)
        ax.fill(angles, values_norm, alpha=0.15, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_title(f'Simple Concatenation\nSil={sil_simple:.3f}', 
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.grid(True)
    
    # Cross-Attention版本
    ax = axes[1]
    ax = plt.subplot(122, projection='polar')
    
    for i, stats in enumerate(cluster_stats_crossattn):
        values = [stats[f] for f in features]
        max_vals = [max([s[f] for s in cluster_stats_crossattn]) for f in features]
        values_norm = [v/m if m > 0 else 0 for v, m in zip(values, max_vals)]
        values_norm += values_norm[:1]
        
        ax.plot(angles, values_norm, 'o-', linewidth=3, 
               label=f'C{i}', color=colors[i], markersize=8)
        ax.fill(angles, values_norm, alpha=0.15, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_title(f'Cross-Attention\nSil={sil_crossattn:.3f}', 
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.grid(True)
    
    plt.suptitle('Fusion Method Comparison', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('./results/fusion_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Comparison saved: ./results/fusion_comparison.png")


# ==================== Main ====================
def main():
    np.random.seed(42)
    torch.manual_seed(42)
    
    Path('./results').mkdir(exist_ok=True)
    
    print(f"\n🎯 K=4 Clustering with Cross-Attention")
    
    # 加载数据
    driving, energy, lengths = load_data('./results/temporal_soc_full', max_samples=100000)
    
    dataset = VariableLengthDataset(driving, energy, lengths)
    train_loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        collate_fn=collate_variable_length,
        num_workers=0
    )
    
    # 创建Cross-Attention模型
    model = DualChannelGRU_CrossAttn(
        driving_dim=driving[0].shape[1],
        energy_dim=energy[0].shape[1],
        latent_dim=16
    ).to(device)
    
    # 训练
    model = train_model(model, train_loader, epochs=10)
    
    # 提取特征
    features = extract_features(model, train_loader)
    
    # 聚类
    labels, kmeans, sil, cv = perform_clustering_k4(features)
    
    # 提取簇统计
    cluster_stats = extract_cluster_features(driving, energy, labels)
    
    # 保存
    np.save('./results/features_k4_crossattn.npy', features)
    np.save('./results/labels_k4_crossattn.npy', labels)
    
    # 保存特征表
    df = pd.DataFrame(cluster_stats)
    df.index = ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3']
    df.to_csv('./results/cluster_features_k4_crossattn.csv', encoding='utf-8-sig')
    
    print("\n" + "="*70)
    print("✅ Cross-Attention Version Completed!")
    print("="*70)
    print(f"\n📊 Performance:")
    print(f"   Silhouette: {sil:.3f}")
    print(f"   CV: {cv:.3f}")
    print(f"\n💡 Compare with simple concatenation version to see improvement!")


if __name__ == "__main__":
    main()