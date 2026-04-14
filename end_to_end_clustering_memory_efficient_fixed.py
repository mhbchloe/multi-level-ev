"""
内存友好版本 - 修复inf/nan问题
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from pathlib import Path
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import gc

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("🚀 End-to-End Pipeline (Memory Efficient + Fixed)")
print("="*70)


# ==================== Step 1: 逐文件、逐车辆处理 ====================
def segment_by_soc_drop_streaming(csv_files, soc_drop_threshold=3.0, min_length=10, max_segments=50000):
    """
    流式处理：逐文件、逐车辆读取并分段
    """
    print("\n" + "="*70)
    print("📊 Step 1-2: Streaming Segmentation")
    print("="*70)
    
    segments = []
    total_records = 0
    total_vehicles = set()
    
    required_cols = ['vehicle_id', 'time', 'soc', 'spd', 'v', 'i', 'acc']
    
    for file in tqdm(csv_files, desc="Processing CSVs"):
        print(f"\n📂 {file.name}...")
        
        chunk_iter = pd.read_csv(file, usecols=required_cols, chunksize=1000000)
        
        for chunk_idx, chunk in enumerate(chunk_iter):
            print(f"   Chunk {chunk_idx+1}: {len(chunk):,} records", end=" → ")
            
            # 数据清洗
            chunk = chunk.dropna(subset=['soc', 'spd', 'v', 'i', 'acc'])
            
            # 过滤异常值
            chunk = chunk[
                (chunk['soc'] >= 0) & (chunk['soc'] <= 100) &
                (chunk['spd'] >= 0) & (chunk['spd'] <= 220) &
                (chunk['v'] > 0) & (chunk['v'] <= 1000) &
                (chunk['i'] >= -1000) & (chunk['i'] <= 1000) &
                (chunk['acc'] >= -10) & (chunk['acc'] <= 10)  # 加速度合理范围
            ]
            
            total_records += len(chunk)
            
            for vehicle_id in chunk['vehicle_id'].unique():
                total_vehicles.add(vehicle_id)
                
                vehicle_data = chunk[chunk['vehicle_id'] == vehicle_id].sort_values('time')
                
                if len(vehicle_data) < min_length:
                    continue
                
                soc_values = vehicle_data['soc'].values
                start_idx = 0
                
                while start_idx < len(vehicle_data):
                    soc_start = soc_values[start_idx]
                    
                    for end_idx in range(start_idx + 1, len(vehicle_data)):
                        soc_current = soc_values[end_idx]
                        soc_drop = soc_start - soc_current
                        
                        if soc_current > soc_start:
                            start_idx = end_idx
                            break
                        
                        if soc_drop >= soc_drop_threshold:
                            segment = vehicle_data.iloc[start_idx:end_idx+1][required_cols].copy()
                            
                            if len(segment) >= min_length:
                                segments.append(segment)
                                
                                if len(segments) >= max_segments:
                                    print(f"\n✅ Reached max: {max_segments:,} segments")
                                    return segments
                            
                            start_idx = end_idx + 1
                            break
                    else:
                        start_idx += 1
            
            print(f"Segments: {len(segments):,}")
            del chunk
            gc.collect()
    
    print(f"\n✅ Total: {len(segments):,} segments from {len(total_vehicles):,} vehicles")
    return segments


# ==================== Step 2: 特征提取 + 清洗 ====================
def extract_and_clean_features(segments):
    """
    提取特征并清理异常值
    """
    print("\n" + "="*70)
    print("🔍 Step 3: Extracting & Cleaning Features")
    print("="*70)
    
    driving_sequences = []
    energy_sequences = []
    valid_indices = []
    
    for idx, segment in enumerate(tqdm(segments, desc="Extracting")):
        # 驾驶通道
        driving_feat = segment[['spd', 'acc']].values.astype(np.float32)
        # 能量通道
        energy_feat = segment[['soc', 'v', 'i']].values.astype(np.float32)
        
        # 检查是否有inf/nan
        if (np.isfinite(driving_feat).all() and 
            np.isfinite(energy_feat).all()):
            
            # 再次检查范围
            if (driving_feat[:, 0].max() <= 220 and  # 速度
                np.abs(driving_feat[:, 1]).max() <= 10 and  # 加速度
                energy_feat[:, 0].max() <= 100 and  # SOC
                energy_feat[:, 1].max() <= 1000 and  # 电压
                np.abs(energy_feat[:, 2]).max() <= 1000):  # 电流
                
                driving_sequences.append(driving_feat)
                energy_sequences.append(energy_feat)
                valid_indices.append(idx)
    
    print(f"\n✅ Valid sequences: {len(driving_sequences):,} / {len(segments):,}")
    print(f"   Filtered out: {len(segments) - len(driving_sequences):,} sequences with inf/nan")
    
    return driving_sequences, energy_sequences


# ==================== Step 3: 归一化（增强版） ====================
def normalize_features(driving_seqs, energy_seqs):
    """
    归一化 - 带异常值检测
    """
    print("\n" + "="*70)
    print("🔧 Step 4: Normalizing (with outlier detection)")
    print("="*70)
    
    # 采样拟合scaler
    sample_size = min(10000, len(driving_seqs))
    sample_indices = np.random.choice(len(driving_seqs), sample_size, replace=False)
    
    driving_sample = np.vstack([driving_seqs[i] for i in sample_indices])
    energy_sample = np.vstack([energy_seqs[i] for i in sample_indices])
    
    print(f"\n   Scaler fitting on {sample_size:,} samples")
    print(f"   Driving sample shape: {driving_sample.shape}")
    print(f"   Energy sample shape: {energy_sample.shape}")
    
    # 检查拟合数据
    print(f"\n   Driving data check:")
    print(f"      Speed: [{driving_sample[:, 0].min():.2f}, {driving_sample[:, 0].max():.2f}]")
    print(f"      Accel: [{driving_sample[:, 1].min():.2f}, {driving_sample[:, 1].max():.2f}]")
    print(f"   Energy data check:")
    print(f"      SOC: [{energy_sample[:, 0].min():.2f}, {energy_sample[:, 0].max():.2f}]")
    print(f"      Voltage: [{energy_sample[:, 1].min():.2f}, {energy_sample[:, 1].max():.2f}]")
    print(f"      Current: [{energy_sample[:, 2].min():.2f}, {energy_sample[:, 2].max():.2f}]")
    
    # 拟合scaler
    driving_scaler = RobustScaler()
    energy_scaler = RobustScaler()
    
    driving_scaler.fit(driving_sample)
    energy_scaler.fit(energy_sample)
    
    print(f"\n   Scalers fitted successfully")
    
    # 归一化（带异常值处理）
    driving_normalized = []
    energy_normalized = []
    failed_count = 0
    
    for idx, (d_seq, e_seq) in enumerate(tqdm(zip(driving_seqs, energy_seqs), 
                                               total=len(driving_seqs), 
                                               desc="Normalizing")):
        try:
            # 再次检查输入
            if not (np.isfinite(d_seq).all() and np.isfinite(e_seq).all()):
                failed_count += 1
                continue
            
            # 转换
            d_norm = driving_scaler.transform(d_seq)
            e_norm = energy_scaler.transform(e_seq)
            
            # 检查输出
            if not (np.isfinite(d_norm).all() and np.isfinite(e_norm).all()):
                failed_count += 1
                continue
            
            # Clip极端值（避免后续计算溢出）
            d_norm = np.clip(d_norm, -10, 10)
            e_norm = np.clip(e_norm, -10, 10)
            
            driving_normalized.append(d_norm.astype(np.float32))
            energy_normalized.append(e_norm.astype(np.float32))
            
        except Exception as e:
            failed_count += 1
            if failed_count < 5:  # 只打印前几个错误
                print(f"\n   Warning: Failed to normalize sequence {idx}: {e}")
    
    print(f"\n✅ Normalized: {len(driving_normalized):,} sequences")
    if failed_count > 0:
        print(f"   ⚠️  Failed: {failed_count} sequences")
    
    # 保存scaler
    Path('./results').mkdir(exist_ok=True)
    import pickle
    with open('./results/scalers.pkl', 'wb') as f:
        pickle.dump({'driving': driving_scaler, 'energy': energy_scaler}, f)
    
    return driving_normalized, energy_normalized, {'driving': driving_scaler, 'energy': energy_scaler}


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


# ==================== GRU Model ====================
class GRUEncoderVarLen(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers,
                         batch_first=True, dropout=0.2 if num_layers > 1 else 0)
    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        _, h_n = self.gru(packed)
        return h_n[-1]


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
        return torch.matmul(attn, v)


class DualChannelGRU_CrossAttn(nn.Module):
    def __init__(self, driving_dim, energy_dim, latent_dim):
        super().__init__()
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
    
    def forward(self, driving, energy, lengths):
        driving_feat = self.driving_encoder(driving, lengths)
        energy_feat = self.energy_encoder(energy, lengths)
        driving_attended = self.cross_attn_e2d(driving_feat, energy_feat)
        energy_attended = self.cross_attn_d2e(energy_feat, driving_feat)
        combined = torch.cat([driving_feat, driving_attended, energy_feat, energy_attended], dim=1)
        fused = self.fusion(combined)
        reconstructed = self.decoder(fused)
        return reconstructed, fused


# ==================== Train ====================
def train_model(model, train_loader, epochs=10):
    print("\n" + "="*70)
    print("🚀 Step 5: Training")
    print("="*70)
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
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
                targets.append(torch.cat([driving[idx, length-1, :], energy[idx, length-1, :]]))
            targets = torch.stack(targets)
            
            loss = criterion(reconstructed, targets)
            if not (torch.isnan(loss) or torch.isinf(loss)):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                batch_count += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        if batch_count > 0:
            print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {epoch_loss/batch_count:.4f}")
    
    torch.save(model.state_dict(), './results/model.pth')
    print(f"✅ Model saved")
    return model


# ==================== Extract Features ====================
def extract_gru_features(model, loader):
    print("\n" + "="*70)
    print("🔍 Step 6: Extracting GRU Features")
    print("="*70)
    
    model.eval()
    all_features = []
    all_indices = []
    
    with torch.no_grad():
        for driving, energy, lengths, sorted_idx in tqdm(loader, desc="Extracting"):
            driving = driving.to(device)
            energy = energy.to(device)
            _, feat = model(driving, energy, lengths)
            all_features.append(feat.cpu().numpy())
            all_indices.append(sorted_idx.numpy())
    
    features = np.vstack(all_features)
    indices = np.concatenate(all_indices)
    features = features[np.argsort(indices)]
    
    print(f"✅ Extracted: {features.shape}")
    return features


# ==================== Clustering ====================
def perform_clustering(features, k=4):
    print("\n" + "="*70)
    print(f"🎯 Step 7: Clustering (K={k})")
    print("="*70)
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = kmeans.fit_predict(features)
    
    sil = silhouette_score(features, labels)
    unique, counts = np.unique(labels, return_counts=True)
    
    print(f"\n✅ Silhouette: {sil:.3f}")
    for cluster_id, count in zip(unique, counts):
        print(f"   Cluster {cluster_id}: {count:,} ({count/len(labels)*100:.1f}%)")
    
    return labels, sil


# ==================== Main ====================
def main():
    csv_files = sorted(Path('.').glob('*_processed.csv'))
    print(f"\nFound {len(csv_files)} CSV files")
    
    # Step 1-2: 分段
    segments = segment_by_soc_drop_streaming(csv_files, max_segments=50000)
    
    # Step 3: 特征提取 + 清洗
    driving_seqs, energy_seqs = extract_and_clean_features(segments)
    
    # Step 4: 归一化
    driving_norm, energy_norm, scalers = normalize_features(driving_seqs, energy_seqs)
    
    # 释放内存
    del driving_seqs, energy_seqs, segments
    gc.collect()
    
    # Step 5: 训练
    lengths = [len(s) for s in driving_norm]
    dataset = VariableLengthDataset(driving_norm, energy_norm, lengths)
    loader = DataLoader(dataset, batch_size=128, shuffle=True, collate_fn=collate_variable_length)
    
    model = DualChannelGRU_CrossAttn(2, 3, 16).to(device)
    model = train_model(model, loader, epochs=10)
    
    # Step 6: 提取特征
    features = extract_gru_features(model, loader)
    
    # Step 7: 聚类
    labels, sil = perform_clustering(features)
    
    # 保存
    np.save('./results/features.npy', features)
    np.save('./results/labels.npy', labels)
    
    print("\n" + "="*70)
    print("✅ Complete!")
    print("="*70)


if __name__ == "__main__":
    main()