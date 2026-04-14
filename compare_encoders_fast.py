"""
======================================================================
🔥 编码器快速对比版（10-15分钟完成）
======================================================================
优化：
- 只用5万样本（从50万采样）
- 过滤超长/超短序列
- 5个epochs（够对比了）
- 增大batch_size
======================================================================
"""

import time
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from tqdm import tqdm

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print(f"使用设备: {device}")


# ==================== 数据集 ====================
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


# ==================== 加载与过滤 ====================
def load_and_filter_data(data_dir, config):
    """加载并过滤数据"""
    print("="*70)
    print("📂 加载数据")
    print("="*70)
    
    data_path = Path(data_dir)
    
    if not (data_path / 'driving_sequences.npy').exists():
        print("❌ 数据不存在")
        return None, None, None
    
    driving = np.load(data_path / 'driving_sequences.npy', allow_pickle=True)
    energy = np.load(data_path / 'energy_sequences.npy', allow_pickle=True)
    lengths = np.load(data_path / 'seq_lengths.npy')
    
    print(f"原始数据: {len(driving):,} 样本")
    
    # 过滤长度
    min_len = config['min_seq_len']
    max_len = config['max_seq_len']
    valid_mask = (lengths >= min_len) & (lengths <= max_len)
    
    driving = driving[valid_mask]
    energy = energy[valid_mask]
    lengths = lengths[valid_mask]
    
    print(f"长度过滤 ({min_len}-{max_len}): {len(driving):,} 样本")
    
    # 随机采样
    max_samples = config['max_samples']
    if len(driving) > max_samples:
        indices = np.random.choice(len(driving), max_samples, replace=False)
        driving = driving[indices]
        energy = energy[indices]
        lengths = lengths[indices]
        print(f"随机采样: {len(driving):,} 样本")
    
    print(f"\n✅ 最终数据:")
    print(f"   样本数: {len(driving):,}")
    print(f"   平均长度: {lengths.mean():.1f}")
    print(f"   最大长度: {lengths.max()}")
    print(f"   驾驶特征: {driving[0].shape[1]}D")
    print(f"   能量特征: {energy[0].shape[1]}D")
    
    return driving, energy, lengths


# ==================== 编码器 ====================
class GRUEncoderVarLen(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):  # ← 只用1层加速
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.hidden_dim = hidden_dim
        
    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        _, h_n = self.gru(packed)
        return h_n[-1]


class LSTMEncoderVarLen(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.hidden_dim = hidden_dim
        
    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        _, (h_n, _) = self.lstm(packed)
        return h_n[-1]


class BiGRUEncoderVarLen(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.bigru = nn.GRU(input_dim, hidden_dim // 2, num_layers,
                           batch_first=True, bidirectional=True)
        self.hidden_dim = hidden_dim
        
    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        _, h_n = self.bigru(packed)
        forward = h_n[-2]
        backward = h_n[-1]
        return torch.cat([forward, backward], dim=1)


class BiLSTMEncoderVarLen(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.bilstm = nn.LSTM(input_dim, hidden_dim // 2, num_layers,
                             batch_first=True, bidirectional=True)
        self.hidden_dim = hidden_dim
        
    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        _, (h_n, _) = self.bilstm(packed)
        forward = h_n[-2]
        backward = h_n[-1]
        return torch.cat([forward, backward], dim=1)


# ==================== 双通道模型 ====================
class DualChannelModelVarLen(nn.Module):
    def __init__(self, encoder_type, driving_dim, energy_dim, latent_dim):
        super().__init__()
        
        if encoder_type == 'GRU':
            self.driving_encoder = GRUEncoderVarLen(driving_dim, latent_dim)
            self.energy_encoder = GRUEncoderVarLen(energy_dim, latent_dim)
        elif encoder_type == 'LSTM':
            self.driving_encoder = LSTMEncoderVarLen(driving_dim, latent_dim)
            self.energy_encoder = LSTMEncoderVarLen(energy_dim, latent_dim)
        elif encoder_type == 'BiGRU':
            self.driving_encoder = BiGRUEncoderVarLen(driving_dim, latent_dim)
            self.energy_encoder = BiGRUEncoderVarLen(energy_dim, latent_dim)
        elif encoder_type == 'BiLSTM':
            self.driving_encoder = BiLSTMEncoderVarLen(driving_dim, latent_dim)
            self.energy_encoder = BiLSTMEncoderVarLen(energy_dim, latent_dim)
        else:
            raise ValueError(f"Unknown encoder: {encoder_type}")
        
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.decoder = nn.Linear(64, driving_dim + energy_dim)
        
    def forward(self, driving, energy, lengths):
        driving_feat = self.driving_encoder(driving, lengths)
        energy_feat = self.energy_encoder(energy, lengths)
        combined = torch.cat([driving_feat, energy_feat], dim=1)
        fused = self.fusion(combined)
        reconstructed = self.decoder(fused)
        return reconstructed, combined


# ==================== 训练（简化版）====================
def train_model_fast(model, train_loader, config):
    """快速训练"""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(config['epochs']):
        epoch_loss = 0
        batch_count = 0
        
        for driving, energy, lengths, _ in tqdm(train_loader, 
                                                 desc=f"Epoch {epoch+1}/{config['epochs']}", 
                                                 leave=False):
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
        
        if batch_count > 0:
            avg_loss = epoch_loss / batch_count
            best_loss = min(best_loss, avg_loss)
            print(f"Epoch {epoch+1}/{config['epochs']} | Loss: {avg_loss:.4f}")
    
    training_time = time.time() - start_time
    return best_loss, training_time


def extract_features_fast(model, loader):
    """快速提取特征"""
    model.eval()
    all_features = []
    all_indices = []
    
    with torch.no_grad():
        for driving, energy, lengths, sorted_idx in loader:
            driving = driving.to(device)
            energy = energy.to(device)
            _, feat = model(driving, energy, lengths)
            all_features.append(feat.cpu().numpy())
            all_indices.append(sorted_idx.numpy())
    
    features = np.vstack(all_features)
    indices = np.concatenate(all_indices)
    unsort_indices = np.argsort(indices)
    features = features[unsort_indices]
    
    return features


def clustering(features, k=4):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)  # n_init 20→10加速
    labels = kmeans.fit_predict(features)
    sil = silhouette_score(features, labels)
    unique, counts = np.unique(labels, return_counts=True)
    cv = np.std(counts) / np.mean(counts)
    distribution = dict(zip(unique.tolist(), counts.tolist()))
    return labels, sil, cv, distribution


# ==================== 主流程 ====================
class FastEncoderComparison:
    def __init__(self, config):
        self.config = config
        self.results = {}
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run_single_encoder(self, encoder_type, train_loader, driving_dim, energy_dim):
        print("\n" + "="*70)
        print(f"🧪 {encoder_type}")
        print("="*70)
        
        try:
            model = DualChannelModelVarLen(
                encoder_type, driving_dim, energy_dim, self.config['latent_dim']
            ).to(device)
            
            params = sum(p.numel() for p in model.parameters())
            print(f"参数: {params:,}")
            
            best_loss, train_time = train_model_fast(model, train_loader, self.config)
            print(f"✅ 损失: {best_loss:.4f}, 时间: {train_time:.1f}s")
            
            features = extract_features_fast(model, train_loader)
            labels, sil, cv, dist = clustering(features, self.config['k_value'])
            
            print(f"📊 Sil: {sil:.3f}, CV: {cv:.3f}")
            
            self.results[encoder_type] = {
                'encoder': encoder_type,
                'params': params,
                'best_loss': best_loss,
                'training_time': train_time,
                'silhouette': sil,
                'cv': cv,
                'distribution': dist
            }
            
            return True
        except Exception as e:
            print(f"❌ 失败: {e}")
            return False
    
    def run_all(self, driving_data, energy_data, lengths):
        dataset = VariableLengthDataset(driving_data, energy_data, lengths)
        train_loader = DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            collate_fn=collate_variable_length,
            num_workers=0
        )
        
        print(f"\n样本数: {len(dataset):,}")
        
        for encoder in ['GRU', 'LSTM', 'BiGRU', 'BiLSTM']:
            self.run_single_encoder(encoder, train_loader, 
                                   driving_data[0].shape[1], 
                                   energy_data[0].shape[1])
        
        if len(self.results) > 0:
            self.save_summary()
    
    def save_summary(self):
        summary = []
        for name, result in self.results.items():
            summary.append({
                '编码器': result['encoder'],
                '参数量': result['params'],
                '时间(s)': round(result['training_time'], 1),
                '损失': round(result['best_loss'], 4),
                'Sil': round(result['silhouette'], 3),
                'CV': round(result['cv'], 3)
            })
        
        df = pd.DataFrame(summary)
        df = df.sort_values('Sil', ascending=False)
        
        print("\n" + "="*70)
        print("📊 汇总")
        print("="*70)
        print(df.to_string(index=False))
        
        df.to_csv(self.output_dir / 'summary.csv', index=False, encoding='utf-8-sig')


# ==================== 主程序 ====================
if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    
    config = {
        'data_dir': './results/temporal_soc_full',
        'latent_dim': 16,
        'batch_size': 128,       # ← 加大
        'epochs': 5,             # ← 减少
        'lr': 0.001,
        'k_value': 4,
        'output_dir': './results/encoder_comparison_fast',
        
        # 数据过滤
        'max_samples': 50000,    # ← 只用5万
        'max_seq_len': 500,      # ← 最长500点
        'min_seq_len': 10
    }
    
    print("="*70)
    print("🚀 快速编码器对比（预计10-15分钟）")
    print("="*70)
    
    driving, energy, lengths = load_and_filter_data(config['data_dir'], config)
    
    if driving is not None:
        comparison = FastEncoderComparison(config)
        comparison.run_all(driving, energy, lengths)
        
        print("\n✅ 完成！")