"""
修正版：使用SOC 3%分割标准
与统计特征模型保持一致
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score
import os
from tqdm import tqdm
import glob

print("="*70)
print("🔥 Temporal Model with SOC-based Segmentation")
print("="*70)

# ==================== 配置 ====================
CONFIG = {
    'sample_vehicles': 2000,    # 采样2000辆车
    'min_seq_length': 5,        # 最小5个点
    'max_seq_length': 200,      # 增加到200（容纳更多数据）
    'soc_threshold': 3.0,       # ← SOC下降3%分割
    'driving_features': ['spd', 'acc'],
    'energy_features': ['soc', 'v', 'i'],
    'latent_dim': 16,
    'batch_size': 128,
    'epochs': 30,
    'lr': 0.001,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'k_value': 4,
    'output_dir': './results/temporal_soc'
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)

print(f"\n⚙️  Configuration:")
for key, value in CONFIG.items():
    print(f"   {key}: {value}")

# ==================== 核心：SOC分割函数 ====================
def segment_trips_by_soc(vehicle_data, vehicle_id, soc_threshold=3.0):
    """
    使用SOC下降百分比分割行程（与统计特征模型一致）
    """
    vehicle_data = vehicle_data.sort_values('datetime').reset_index(drop=True)
    vehicle_data['datetime'] = pd.to_datetime(vehicle_data['datetime'])
    
    if 'soc' not in vehicle_data.columns:
        return []
    
    trips = []
    trip_start_idx = 0
    start_soc = vehicle_data.loc[0, 'soc']
    
    for i in range(1, len(vehicle_data)):
        current_soc = vehicle_data.loc[i, 'soc']
        soc_drop = start_soc - current_soc
        
        # SOC下降超过阈值，分割行程
        if soc_drop >= soc_threshold:
            trip_data = vehicle_data.iloc[trip_start_idx:i+1]
            
            # 长度检查
            if len(trip_data) >= CONFIG['min_seq_length']:
                # 如果太长，分段保存
                if len(trip_data) > CONFIG['max_seq_length']:
                    # 分成多个max_length的片段
                    for j in range(0, len(trip_data), CONFIG['max_seq_length']):
                        segment = trip_data.iloc[j:j+CONFIG['max_seq_length']]
                        if len(segment) >= CONFIG['min_seq_length']:
                            trips.append({
                                'trip_id': f"{vehicle_id}_{len(trips)}",
                                'vehicle_id': vehicle_id,
                                'sequence': segment
                            })
                else:
                    trips.append({
                        'trip_id': f"{vehicle_id}_{len(trips)}",
                        'vehicle_id': vehicle_id,
                        'sequence': trip_data
                    })
            
            # 重置起点
            trip_start_idx = i + 1
            start_soc = current_soc
    
    # 最后一个行程
    trip_data = vehicle_data.iloc[trip_start_idx:]
    if len(trip_data) >= CONFIG['min_seq_length']:
        if len(trip_data) > CONFIG['max_seq_length']:
            for j in range(0, len(trip_data), CONFIG['max_seq_length']):
                segment = trip_data.iloc[j:j+CONFIG['max_seq_length']]
                if len(segment) >= CONFIG['min_seq_length']:
                    trips.append({
                        'trip_id': f"{vehicle_id}_{len(trips)}",
                        'vehicle_id': vehicle_id,
                        'sequence': segment
                    })
        else:
            trips.append({
                'trip_id': f"{vehicle_id}_{len(trips)}",
                'vehicle_id': vehicle_id,
                'sequence': trip_data
            })
    
    return trips

# ==================== 1. 数据加载 ====================
print(f"\n{'='*70}")
print("📂 Loading Data with SOC-based Segmentation")
print("="*70)

data_files = sorted(glob.glob('./*processed.csv'))
if not data_files:
    data_files = sorted(glob.glob('./data/*processed.csv'))

print(f"✅ Found {len(data_files)} files")

# 采样车辆
print(f"\n🎲 Sampling {CONFIG['sample_vehicles']} vehicles...")

all_vehicles = set()
for file in data_files[:2]:  # 从前2个文件采样
    df_sample = pd.read_csv(file, usecols=['vehicle_id'], nrows=500000)
    all_vehicles.update(df_sample['vehicle_id'].unique())

all_vehicles = list(all_vehicles)
sampled_vehicles = np.random.choice(all_vehicles, 
                                   min(CONFIG['sample_vehicles'], len(all_vehicles)),
                                   replace=False)

print(f"✅ Sampled {len(sampled_vehicles)} vehicles")

# 提取行程
print(f"\n🔍 Extracting trips using SOC 3% threshold...")

all_trips = []
chunk_size = 50000

for file_idx, file in enumerate(data_files):
    print(f"\n📄 File {file_idx+1}/{len(data_files)}: {os.path.basename(file)}")
    
    for chunk in pd.read_csv(file, chunksize=chunk_size):
        chunk = chunk[chunk['vehicle_id'].isin(sampled_vehicles)]
        
        if len(chunk) == 0:
            continue
        
        for vehicle_id, vehicle_data in chunk.groupby('vehicle_id'):
            trips = segment_trips_by_soc(vehicle_data, vehicle_id, 
                                        CONFIG['soc_threshold'])
            all_trips.extend(trips)
    
    print(f"   Total trips so far: {len(all_trips):,}")
    
    # 足够数据后停止
    if len(all_trips) >= 30000:
        print(f"✅ Collected {len(all_trips):,} trips, stopping")
        break

print(f"\n✅ Collected {len(all_trips):,} trip sequences")
print(f"   From {len(set([t['vehicle_id'] for t in all_trips]))} vehicles")

# 计算保留率
expected_trips = len(sampled_vehicles) * 19.6  # 平均每车19.6个行程
retention_rate = len(all_trips) / expected_trips if expected_trips > 0 else 0
print(f"\n📊 Data retention rate:")
print(f"   Expected trips (based on stats): ~{expected_trips:.0f}")
print(f"   Actual trips collected: {len(all_trips)}")
print(f"   Retention rate: {retention_rate*100:.1f}%")

# ==================== 2. 准备序列数据 ====================
print(f"\n{'='*70}")
print("🔧 Preparing Sequential Data")
print("="*70)

def prepare_sequence_data(trips, driving_features, energy_features, max_len):
    """准备序列数据"""
    driving_sequences = []
    energy_sequences = []
    trip_ids = []
    seq_lengths = []
    
    for trip in tqdm(trips, desc="Preparing sequences"):
        seq = trip['sequence']
        
        try:
            driving_seq = seq[driving_features].values
            energy_seq = seq[energy_features].values
        except KeyError:
            continue
        
        seq_len = len(driving_seq)
        
        # Padding
        if seq_len < max_len:
            pad_len = max_len - seq_len
            driving_seq = np.vstack([
                driving_seq,
                np.zeros((pad_len, len(driving_features)))
            ])
            energy_seq = np.vstack([
                energy_seq,
                np.zeros((pad_len, len(energy_features)))
            ])
        else:
            driving_seq = driving_seq[:max_len]
            energy_seq = energy_seq[:max_len]
            seq_len = max_len
        
        driving_sequences.append(driving_seq)
        energy_sequences.append(energy_seq)
        trip_ids.append(trip['trip_id'])
        seq_lengths.append(seq_len)
    
    return {
        'driving': np.array(driving_sequences, dtype=np.float32),
        'energy': np.array(energy_sequences, dtype=np.float32),
        'trip_ids': trip_ids,
        'lengths': np.array(seq_lengths)
    }

# 检查特征
sample_seq = all_trips[0]['sequence']
available_driving = [f for f in CONFIG['driving_features'] if f in sample_seq.columns]
available_energy = [f for f in CONFIG['energy_features'] if f in sample_seq.columns]

print(f"\nAvailable features:")
print(f"   Driving: {available_driving}")
print(f"   Energy: {available_energy}")

# 准备数据
data = prepare_sequence_data(all_trips, available_driving, available_energy,
                            CONFIG['max_seq_length'])

print(f"\n✅ Prepared sequential data:")
print(f"   Driving: {data['driving'].shape}")
print(f"   Energy: {data['energy'].shape}")

# 数据清洗
valid_mask = ~(np.isnan(data['driving']).any(axis=(1,2)) | 
               np.isnan(data['energy']).any(axis=(1,2)) |
               np.isinf(data['driving']).any(axis=(1,2)) |
               np.isinf(data['energy']).any(axis=(1,2)))

data['driving'] = data['driving'][valid_mask]
data['energy'] = data['energy'][valid_mask]
data['trip_ids'] = [tid for tid, v in zip(data['trip_ids'], valid_mask) if v]
data['lengths'] = data['lengths'][valid_mask]

print(f"   After cleaning: {data['driving'].shape[0]:,} trips")

# 计算最终保留率
final_retention = data['driving'].shape[0] / 179181 * (len(sampled_vehicles) / 9145) * 100
print(f"\n📊 Final data statistics:")
print(f"   Original (stats model): 179,181 trips")
print(f"   Temporal (SOC split): {data['driving'].shape[0]:,} trips")
print(f"   Expected retention: ~{(len(sampled_vehicles) / 9145)*100:.1f}% (due to sampling)")
print(f"   Actual retention: ~{final_retention:.1f}%")

# 归一化
print(f"\n📊 Normalizing...")

for i in range(data['driving'].shape[2]):
    scaler = RobustScaler()
    flat_data = data['driving'][:, :, i].reshape(-1, 1)
    normalized = scaler.fit_transform(flat_data)
    data['driving'][:, :, i] = normalized.reshape(data['driving'].shape[0], -1)

for i in range(data['energy'].shape[2]):
    scaler = RobustScaler()
    flat_data = data['energy'][:, :, i].reshape(-1, 1)
    normalized = scaler.fit_transform(flat_data)
    data['energy'][:, :, i] = normalized.reshape(data['energy'].shape[0], -1)

data['driving'] = np.clip(data['driving'], -5, 5)
data['energy'] = np.clip(data['energy'], -5, 5)

print(f"✅ Normalization complete")

# ==================== 3. 使用GRU训练（之前最好的时序模型）====================
print(f"\n{'='*70}")
print("🚀 Training GRU Model")
print("="*70)

# 复用之前的GRU模型定义
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

class GRUEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )
    
    def forward(self, x, lengths=None):
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            gru_out, hidden = self.gru(x)
            gru_out, _ = nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True)
        else:
            gru_out, hidden = self.gru(x)
        
        last_hidden = hidden[-1]
        latent = self.fc(last_hidden)
        return latent

class TemporalDualChannelGRU(nn.Module):
    def __init__(self, driving_dim, energy_dim, latent_dim=16, hidden_dim=32):
        super().__init__()
        
        self.driving_encoder = GRUEncoder(driving_dim, hidden_dim, latent_dim)
        self.energy_encoder = GRUEncoder(energy_dim, hidden_dim, latent_dim)
        
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
        
        self.driving_decoder = nn.Linear(latent_dim, driving_dim)
        self.energy_decoder = nn.Linear(latent_dim, energy_dim)
        
        print(f"   Total parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, driving_seq, energy_seq, lengths=None):
        driving_latent = self.driving_encoder(driving_seq, lengths)
        energy_latent = self.energy_encoder(energy_seq, lengths)
        
        driving_attended = self.cross_attn_e2d(driving_latent, energy_latent)
        energy_attended = self.cross_attn_d2e(energy_latent, driving_latent)
        
        combined = torch.cat([
            driving_latent, driving_attended,
            energy_latent, energy_attended
        ], dim=1)
        
        fused_latent = self.fusion(combined)
        
        driving_recon = self.driving_decoder(driving_latent)
        energy_recon = self.energy_decoder(energy_latent)
        
        return driving_recon, energy_recon, fused_latent
    
    def encode(self, driving_seq, energy_seq, lengths=None):
        driving_latent = self.driving_encoder(driving_seq, lengths)
        energy_latent = self.energy_encoder(energy_seq, lengths)
        driving_attended = self.cross_attn_e2d(driving_latent, energy_latent)
        energy_attended = self.cross_attn_d2e(energy_latent, driving_latent)
        combined = torch.cat([driving_latent, driving_attended, energy_latent, energy_attended], dim=1)
        return self.fusion(combined)

# 构建模型
model = TemporalDualChannelGRU(
    driving_dim=data['driving'].shape[2],
    energy_dim=data['energy'].shape[2],
    latent_dim=CONFIG['latent_dim'],
    hidden_dim=32
)
model.to(CONFIG['device'])

# 训练（简化版，复用之前的函数）
optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-5)

X_driving = data['driving']
X_energy = data['energy']
lengths = data['lengths']

best_loss = float('inf')

for epoch in range(CONFIG['epochs']):
    model.train()
    epoch_losses = []
    
    indices = np.random.permutation(len(X_driving))
    
    pbar = tqdm(range(0, len(X_driving), CONFIG['batch_size']),
               desc=f"Epoch {epoch+1:2d}/{CONFIG['epochs']}",
               leave=False)
    
    for i in pbar:
        batch_indices = indices[i:i+CONFIG['batch_size']]
        
        d_batch = torch.FloatTensor(X_driving[batch_indices]).to(CONFIG['device'])
        e_batch = torch.FloatTensor(X_energy[batch_indices]).to(CONFIG['device'])
        len_batch = torch.LongTensor(lengths[batch_indices])
        
        d_targets = []
        e_targets = []
        for idx, seq_len in enumerate(len_batch):
            d_targets.append(d_batch[idx, seq_len-1, :])
            e_targets.append(e_batch[idx, seq_len-1, :])
        
        d_targets = torch.stack(d_targets)
        e_targets = torch.stack(e_targets)
        
        optimizer.zero_grad()
        
        d_recon, e_recon, _ = model(d_batch, e_batch, len_batch)
        
        loss = F.mse_loss(d_recon, d_targets) + F.mse_loss(e_recon, e_targets)
        
        if torch.isnan(loss):
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        epoch_losses.append(loss.item())
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = np.mean(epoch_losses)
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:3d}/{CONFIG['epochs']} | Loss: {avg_loss:.4f}")
    
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), f"{CONFIG['output_dir']}/gru_soc_best.pth")

print(f"\n✅ Best loss: {best_loss:.4f}")

# ==================== 4. 提取特征并聚类 ====================
print(f"\n🔍 Extracting features...")

model.eval()
latent_features = []

with torch.no_grad():
    for i in range(0, len(X_driving), CONFIG['batch_size']):
        d_batch = torch.FloatTensor(X_driving[i:i+CONFIG['batch_size']]).to(CONFIG['device'])
        e_batch = torch.FloatTensor(X_energy[i:i+CONFIG['batch_size']]).to(CONFIG['device'])
        len_batch = torch.LongTensor(lengths[i:i+CONFIG['batch_size']])
        
        latent = model.encode(d_batch, e_batch, len_batch)
        latent_features.append(latent.cpu().numpy())

latent_features = np.vstack(latent_features)
print(f"✅ Extracted: {latent_features.shape}")

# 聚类
print(f"\n🎯 Clustering (K={CONFIG['k_value']})...")
kmeans = KMeans(n_clusters=CONFIG['k_value'], random_state=42, n_init=20)
labels = kmeans.fit_predict(latent_features)

sil = silhouette_score(latent_features, labels)
cluster_counts = pd.Series(labels).value_counts().sort_index()
percentages = cluster_counts.values / len(labels) * 100
cv = percentages.std() / percentages.mean()

print(f"\n📊 Results:")
print(f"   Samples: {len(labels):,}")
print(f"   Silhouette: {sil:.3f}")
print(f"   CV: {cv:.3f}")
print(f"\n   Distribution:")
for cid, count in cluster_counts.items():
    pct = count / len(labels) * 100
    print(f"      Cluster {cid}: {count:6,} ({pct:5.1f}%)")

# 保存
np.save(f"{CONFIG['output_dir']}/latent_features.npy", latent_features)
np.save(f"{CONFIG['output_dir']}/labels.npy", labels)

print("\n" + "="*70)
print("✅ SOC-based Temporal Model Complete!")
print("="*70)

print(f"\n📊 Comparison with previous results:")
print(f"   Time-based (30min): 3,558 trips, Sil=0.340, CV=0.939")
print(f"   SOC-based (3%):     {len(labels):,} trips, Sil={sil:.3f}, CV={cv:.3f}")
print(f"\n   Data retention improvement: {len(labels) / 3558 * 100:.0f}%")