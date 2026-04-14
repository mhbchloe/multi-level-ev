"""
======================================================================
🎯 GRU Clustering K=4 - Cross-Attention Version (带反归一化)
======================================================================
聚类后自动检测并还原数据到真实物理单位，然后画图
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
import pickle

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("🎯 GRU Clustering K=4 - Cross-Attention (with Denormalization)")
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
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
    
    def forward(self, x1, x2):
        q = self.query(x1)
        k = self.key(x2)
        v = self.value(x2)
        
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        
        return out


# ==================== Dual-Channel GRU with Cross-Attention ====================
class DualChannelGRU_CrossAttn(nn.Module):
    def __init__(self, driving_dim, energy_dim, latent_dim):
        super().__init__()
        
        print("\n🏗️  Building Dual-Channel GRU with Cross-Attention...")
        
        self.driving_encoder = GRUEncoderVarLen(driving_dim, latent_dim)
        self.energy_encoder = GRUEncoderVarLen(energy_dim, latent_dim)
        
        self.cross_attn_d2e = CrossChannelAttention(latent_dim)
        self.cross_attn_e2d = CrossChannelAttention(latent_dim)
        
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 4, latent_dim * 3),
            nn.LayerNorm(latent_dim * 3),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(latent_dim * 3, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.ReLU()
        )
        
        self.decoder = nn.Linear(latent_dim * 2, driving_dim + energy_dim)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"   Total parameters: {total_params:,}")
        
    def forward(self, driving, energy, lengths):
        driving_feat = self.driving_encoder(driving, lengths)
        energy_feat = self.energy_encoder(energy, lengths)
        
        driving_attended = self.cross_attn_e2d(driving_feat, energy_feat)
        energy_attended = self.cross_attn_d2e(energy_feat, driving_feat)
        
        combined = torch.cat([
            driving_feat,
            driving_attended,
            energy_feat,
            energy_attended
        ], dim=1)
        
        fused = self.fusion(combined)
        reconstructed = self.decoder(fused)
        
        return reconstructed, fused


# ==================== 反归一化工具 ====================
class Denormalizer:
    """自动检测并反归一化数据"""
    
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.scalers = {}
        self.load_scalers()
        
    def load_scalers(self):
        """尝试加载保存的归一化参数"""
        scaler_path = self.data_dir / 'scalers.pkl'
        
        if scaler_path.exists():
            print("\n✅ Found saved scalers, loading...")
            with open(scaler_path, 'rb') as f:
                self.scalers = pickle.load(f)
            print(f"   Loaded scalers for: {list(self.scalers.keys())}")
        else:
            print("\n⚠️  No saved scalers found, will estimate from data")
            self.scalers = None
    
    def estimate_scalers_from_data(self, driving_seqs, energy_seqs):
        """从数据统计估计归一化参数"""
        print("\n🔍 Estimating normalization parameters from data...")
        
        # 采样一部分数据进行统计
        sample_size = min(1000, len(driving_seqs))
        
        # 驾驶数据统计
        driving_sample = np.vstack([driving_seqs[i] for i in range(sample_size)])
        
        # 能量数据统计
        energy_sample = np.vstack([energy_seqs[i] for i in range(sample_size)])
        
        # 检查数据范围
        print("\n📊 Data range analysis:")
        print(f"   Speed: [{driving_sample[:, 0].min():.4f}, {driving_sample[:, 0].max():.4f}]")
        print(f"   Accel: [{driving_sample[:, 1].min():.4f}, {driving_sample[:, 1].max():.4f}]")
        print(f"   SOC:   [{energy_sample[:, 0].min():.4f}, {energy_sample[:, 0].max():.4f}]")
        print(f"   Voltage: [{energy_sample[:, 1].min():.4f}, {energy_sample[:, 1].max():.4f}]")
        print(f"   Current: [{energy_sample[:, 2].min():.4f}, {energy_sample[:, 2].max():.4f}]")
        
        # 判断是否需要反归一化
        speed_max = driving_sample[:, 0].max()
        voltage_max = energy_sample[:, 1].max()
        
        self.needs_denorm = {
            'speed': speed_max < 10,  # 速度应该是几十km/h
            'voltage': voltage_max < 10,  # 电压应该是几百V
        }
        
        print("\n🔧 Denormalization strategy:")
        if self.needs_denorm['speed']:
            print("   Speed: Needs denormalization (likely normalized)")
            # 假设原始速度范围 0-120 km/h
            self.speed_scale = 120.0
            self.speed_offset = 0.0
        else:
            print("   Speed: No denormalization needed")
            self.speed_scale = 1.0
            self.speed_offset = 0.0
        
        if self.needs_denorm['voltage']:
            print("   Voltage: Needs denormalization (likely normalized)")
            # 假设原始电压范围 200-400V
            self.voltage_scale = 200.0
            self.voltage_offset = 300.0
        else:
            print("   Voltage: No denormalization needed")
            self.voltage_scale = 1.0
            self.voltage_offset = 0.0
        
        # Current通常也需要缩放
        if energy_sample[:, 2].max() < 10:
            print("   Current: Needs denormalization")
            self.current_scale = 200.0  # 假设最大电流200A
            self.current_offset = 0.0
        else:
            print("   Current: No denormalization needed")
            self.current_scale = 1.0
            self.current_offset = 0.0
    
    def denormalize_driving(self, driving_seq):
        """反归一化驾驶数据"""
        denorm = driving_seq.copy()
        
        # 速度
        denorm[:, 0] = driving_seq[:, 0] * self.speed_scale + self.speed_offset
        
        # 加速度 (通常也需要缩放)
        if hasattr(self, 'speed_scale'):
            denorm[:, 1] = driving_seq[:, 1] * (self.speed_scale / 10)  # 加速度缩放
        
        return denorm
    
    def denormalize_energy(self, energy_seq):
        """反归一化能量数据"""
        denorm = energy_seq.copy()
        
        # SOC通常是百分比，可能不需要反归一化
        # 但如果被归一化了，需要乘以100
        if energy_seq[:, 0].max() <= 1.0:
            denorm[:, 0] = energy_seq[:, 0] * 100.0
        
        # 电压
        denorm[:, 1] = energy_seq[:, 1] * self.voltage_scale + self.voltage_offset
        
        # 电流
        denorm[:, 2] = energy_seq[:, 2] * self.current_scale + self.current_offset
        
        return denorm


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


# ==================== Extract Cluster Stats (带反归一化) ====================
def extract_cluster_features_denorm(driving_seqs, energy_seqs, labels, denormalizer):
    print("\n" + "="*70)
    print("📊 Extracting Cluster Features (Denormalized)")
    print("="*70)
    
    cluster_stats = []
    
    for cluster_id in range(4):
        print(f"\n  Analyzing Cluster {cluster_id}...")
        
        cluster_mask = (labels == cluster_id)
        cluster_driving = driving_seqs[cluster_mask]
        cluster_energy = energy_seqs[cluster_mask]
        
        stats = {}
        
        # 反归一化所有数据
        driving_denorm_list = []
        energy_denorm_list = []
        
        for seq in cluster_driving:
            driving_denorm_list.append(denormalizer.denormalize_driving(seq))
        
        for seq in cluster_energy:
            energy_denorm_list.append(denormalizer.denormalize_energy(seq))
        
        # 合并
        all_spd = np.concatenate([seq[:, 0] for seq in driving_denorm_list])
        all_acc = np.concatenate([seq[:, 1] for seq in driving_denorm_list])
        
        all_v = np.concatenate([seq[:, 1] for seq in energy_denorm_list])
        all_i = np.concatenate([seq[:, 2] for seq in energy_denorm_list])
        
        # 计算统计特征（真实单位）
        stats['Avg Speed'] = np.mean(all_spd)
        stats['Max Speed'] = np.percentile(all_spd, 95)
        stats['Speed Std'] = np.std(all_spd)
        
        stats['Avg Accel'] = np.mean(np.abs(all_acc))
        stats['Accel Std'] = np.std(all_acc)
        
        # 功率 = 电压 × 电流 / 1000 (kW)
        stats['Avg Power'] = np.mean(np.abs(all_v * all_i)) / 1000
        
        # SOC下降率
        soc_drops = [seq[0, 0] - seq[-1, 0] for seq in energy_denorm_list if len(seq) > 1]
        stats['SOC Drop Rate'] = np.mean(soc_drops)
        
        # 行程长度
        stats['Trip Length'] = np.mean([len(seq) for seq in driving_denorm_list])
        
        cluster_stats.append(stats)
        
        # 打印真实单位值
        print(f"     Avg Speed: {stats['Avg Speed']:.2f} km/h")
        print(f"     Max Speed: {stats['Max Speed']:.2f} km/h")
        print(f"     Avg Power: {stats['Avg Power']:.2f} kW")
        print(f"     SOC Drop: {stats['SOC Drop Rate']:.2f} %/trip")
        print(f"     Trip Length: {stats['Trip Length']:.0f} points")
    
    return cluster_stats


# ==================== Main ====================
def main():
    np.random.seed(42)
    torch.manual_seed(42)
    
    Path('./results').mkdir(exist_ok=True)
    
    print(f"\n🎯 K=4 Clustering with Cross-Attention & Denormalization")
    
    # 加载数据
    data_dir = './results/temporal_soc_full'
    driving, energy, lengths = load_data(data_dir, max_samples=100000)
    
    # 创建反归一化器
    denormalizer = Denormalizer(data_dir)
    denormalizer.estimate_scalers_from_data(driving, energy)
    
    dataset = VariableLengthDataset(driving, energy, lengths)
    train_loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        collate_fn=collate_variable_length,
        num_workers=0
    )
    
    # 创建模型
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
    
    # 提取簇统计（带反归一化）
    cluster_stats = extract_cluster_features_denorm(driving, energy, labels, denormalizer)
    
    # 保存
    np.save('./results/features_k4_crossattn.npy', features)
    np.save('./results/labels_k4_crossattn.npy', labels)
    
    # 保存特征表（真实单位）
    df = pd.DataFrame(cluster_stats)
    df.index = ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3']
    df.to_csv('./results/cluster_features_k4_crossattn_denorm.csv', encoding='utf-8-sig')
    
    print("\n" + "="*70)
    print("📊 Denormalized Cluster Features")
    print("="*70)
    print(df.round(2))
    
    print("\n" + "="*70)
    print("✅ Cross-Attention Clustering with Denormalization Complete!")
    print("="*70)
    print(f"\n📊 Performance:")
    print(f"   Silhouette: {sil:.3f}")
    print(f"   CV: {cv:.3f}")
    print(f"\n📁 Output files:")
    print(f"   - cluster_features_k4_crossattn_denorm.csv (真实单位)")
    print(f"   - features_k4_crossattn.npy")
    print(f"   - labels_k4_crossattn.npy")
    print("="*70)


if __name__ == "__main__":
    main()