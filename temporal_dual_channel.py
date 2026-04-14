"""
基于原始时序数据的双通道模型
输入：每个行程的时间序列（而非统计特征）
模型：LSTM / GRU / TCN
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
import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print("🔥 Temporal Dual-Channel Model")
print("="*70)

# ==================== 配置 ====================
CONFIG = {
    'sample_vehicles': 1000,  # 采样1000辆车（全量9170太大）
    'max_seq_length': 100,    # 每个行程最多100个时间步
    'driving_features': ['spd', 'acc'],  # 驾驶特征（时序）
    'energy_features': ['soc', 'v', 'i'],  # 能量特征（时序）
    'latent_dim': 16,
    'batch_size': 64,
    'epochs': 30,
    'lr': 0.001,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'k_value': 4,
    'output_dir': './results/temporal_model'
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)

print(f"\n⚙️  Configuration:")
for key, value in CONFIG.items():
    print(f"   {key}: {value}")

# ==================== 1. 数据加载和预处理 ====================
print(f"\n{'='*70}")
print("📂 Loading Original Time-Series Data")
print("="*70)

data_files = sorted(glob.glob('./*processed.csv'))
if not data_files:
    data_files = sorted(glob.glob('./data/*processed.csv'))

print(f"✅ Found {len(data_files)} files")

# 1.1 采样车辆
print(f"\n🎲 Sampling {CONFIG['sample_vehicles']} vehicles...")

all_vehicles = set()
for file in data_files[:1]:  # 先从第一个文件采样
    df_sample = pd.read_csv(file, usecols=['vehicle_id'], nrows=500000)
    all_vehicles.update(df_sample['vehicle_id'].unique())

all_vehicles = list(all_vehicles)
sampled_vehicles = np.random.choice(all_vehicles, 
                                   min(CONFIG['sample_vehicles'], len(all_vehicles)),
                                   replace=False)

print(f"✅ Sampled {len(sampled_vehicles)} vehicles")

# 1.2 提取行程时序数据
print(f"\n🔍 Extracting trip sequences...")

def segment_trips_temporal(vehicle_data, vehicle_id, time_gap_minutes=30):
    """
    分割行程，保留时序结构
    返回：每个行程的完整时间序列
    """
    vehicle_data = vehicle_data.sort_values('datetime').reset_index(drop=True)
    vehicle_data['datetime'] = pd.to_datetime(vehicle_data['datetime'])
    
    trips = []
    trip_start_idx = 0
    
    for i in range(1, len(vehicle_data)):
        time_gap = (vehicle_data.loc[i, 'datetime'] - 
                   vehicle_data.loc[i-1, 'datetime']).total_seconds() / 60
        
        if time_gap > time_gap_minutes:
            trip_data = vehicle_data.iloc[trip_start_idx:i]
            
            # 只保留长度合适的行程
            if len(trip_data) >= 10 and len(trip_data) <= CONFIG['max_seq_length']:
                trips.append({
                    'trip_id': f"{vehicle_id}_{len(trips)}",
                    'vehicle_id': vehicle_id,
                    'sequence': trip_data
                })
            
            trip_start_idx = i
    
    # 最后一个行程
    trip_data = vehicle_data.iloc[trip_start_idx:]
    if len(trip_data) >= 10 and len(trip_data) <= CONFIG['max_seq_length']:
        trips.append({
            'trip_id': f"{vehicle_id}_{len(trips)}",
            'vehicle_id': vehicle_id,
            'sequence': trip_data
        })
    
    return trips

# 收集所有行程序列
all_trips = []
chunk_size = 50000

print(f"Processing {len(sampled_vehicles)} vehicles from {len(data_files)} files...")

for file_idx, file in enumerate(data_files):
    print(f"\n📄 File {file_idx+1}/{len(data_files)}: {os.path.basename(file)}")
    
    for chunk in pd.read_csv(file, chunksize=chunk_size):
        # 只保留采样的车辆
        chunk = chunk[chunk['vehicle_id'].isin(sampled_vehicles)]
        
        if len(chunk) == 0:
            continue
        
        # 按车辆分组
        for vehicle_id, vehicle_data in chunk.groupby('vehicle_id'):
            trips = segment_trips_temporal(vehicle_data, vehicle_id)
            all_trips.extend(trips)
    
    print(f"   Total trips so far: {len(all_trips)}")
    
    # 如果已经有足够的行程，可以提前停止
    if len(all_trips) >= 10000:
        print(f"✅ Collected {len(all_trips)} trips, stopping early")
        break

print(f"\n✅ Collected {len(all_trips)} trip sequences")
print(f"   From {len(set([t['vehicle_id'] for t in all_trips]))} vehicles")

# 1.3 准备时序数据
print(f"\n🔧 Preparing sequential data...")

def prepare_sequence_data(trips, driving_features, energy_features, max_len):
    """
    将行程序列转换为张量
    返回：(N, max_len, feature_dim) 格式的数据
    """
    driving_sequences = []
    energy_sequences = []
    trip_ids = []
    seq_lengths = []
    
    for trip in tqdm(trips, desc="Preparing sequences"):
        seq = trip['sequence']
        
        # 提取特征
        try:
            driving_seq = seq[driving_features].values
            energy_seq = seq[energy_features].values
        except KeyError:
            continue
        
        # 记录原始长度
        seq_len = len(driving_seq)
        
        # Padding到max_len
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

# 检查特征是否存在
sample_seq = all_trips[0]['sequence']
available_driving = [f for f in CONFIG['driving_features'] if f in sample_seq.columns]
available_energy = [f for f in CONFIG['energy_features'] if f in sample_seq.columns]

print(f"\nAvailable features:")
print(f"   Driving: {available_driving}")
print(f"   Energy: {available_energy}")

if len(available_driving) == 0 or len(available_energy) == 0:
    print(f"\n❌ Required features not found in data!")
    print(f"   Available columns: {sample_seq.columns.tolist()}")
    exit(1)

# 准备数据
data = prepare_sequence_data(all_trips, available_driving, available_energy, 
                            CONFIG['max_seq_length'])

print(f"\n✅ Prepared sequential data:")
print(f"   Driving: {data['driving'].shape}")  # (N, max_len, driving_dim)
print(f"   Energy: {data['energy'].shape}")    # (N, max_len, energy_dim)

# 数据清洗：移除NaN/Inf
valid_mask = ~(np.isnan(data['driving']).any(axis=(1,2)) | 
               np.isnan(data['energy']).any(axis=(1,2)) |
               np.isinf(data['driving']).any(axis=(1,2)) |
               np.isinf(data['energy']).any(axis=(1,2)))

data['driving'] = data['driving'][valid_mask]
data['energy'] = data['energy'][valid_mask]
data['trip_ids'] = [tid for tid, v in zip(data['trip_ids'], valid_mask) if v]
data['lengths'] = data['lengths'][valid_mask]

print(f"   After cleaning: {data['driving'].shape[0]} trips")

# 归一化（每个特征独立归一化）
print(f"\n📊 Normalizing...")

for i in range(data['driving'].shape[2]):
    scaler = RobustScaler()
    # 展平，归一化，再reshape
    flat_data = data['driving'][:, :, i].reshape(-1, 1)
    normalized = scaler.fit_transform(flat_data)
    data['driving'][:, :, i] = normalized.reshape(data['driving'].shape[0], -1)

for i in range(data['energy'].shape[2]):
    scaler = RobustScaler()
    flat_data = data['energy'][:, :, i].reshape(-1, 1)
    normalized = scaler.fit_transform(flat_data)
    data['energy'][:, :, i] = normalized.reshape(data['energy'].shape[0], -1)

# 裁剪极端值
data['driving'] = np.clip(data['driving'], -5, 5)
data['energy'] = np.clip(data['energy'], -5, 5)

print(f"✅ Normalization complete")

# ==================== 2. 时序编码器模型 ====================
print(f"\n{'='*70}")
print("🏗️  Building Temporal Encoders")
print("="*70)

class LSTMEncoder(nn.Module):
    """LSTM编码器"""
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=2):
        super().__init__()
        self.name = "LSTM"
        
        self.lstm = nn.LSTM(
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
        # x: (batch, seq_len, input_dim)
        
        # LSTM处理
        if lengths is not None:
            # Pack padded sequence
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out, (hidden, cell) = self.lstm(x)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, (hidden, cell) = self.lstm(x)
        
        # 使用最后一个隐藏状态
        last_hidden = hidden[-1]  # (batch, hidden_dim)
        
        # 映射到潜在空间
        latent = self.fc(last_hidden)
        return latent

class GRUEncoder(nn.Module):
    """GRU编码器"""
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=2):
        super().__init__()
        self.name = "GRU"
        
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

class TCNEncoder(nn.Module):
    """Temporal Convolutional Network编码器"""
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.name = "TCN"
        
        # 1D卷积层（时间维度）
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim*2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim*2)
        
        self.conv3 = nn.Conv1d(hidden_dim*2, hidden_dim*4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(hidden_dim*4)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim*4, latent_dim),
            nn.ReLU()
        )
    
    def forward(self, x, lengths=None):
        # x: (batch, seq_len, input_dim)
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = self.global_pool(x).squeeze(-1)  # (batch, hidden_dim*4)
        
        latent = self.fc(x)
        return latent

# ==================== 3. 双通道时序模型 ====================
class TemporalDualChannelAE(nn.Module):
    """时序双通道自编码器"""
    def __init__(self, driving_dim, energy_dim, latent_dim=16, 
                 encoder_type='LSTM', hidden_dim=32):
        super().__init__()
        
        self.encoder_type = encoder_type
        
        # 选择编码器
        if encoder_type == 'LSTM':
            self.driving_encoder = LSTMEncoder(driving_dim, hidden_dim, latent_dim)
            self.energy_encoder = LSTMEncoder(energy_dim, hidden_dim, latent_dim)
        elif encoder_type == 'GRU':
            self.driving_encoder = GRUEncoder(driving_dim, hidden_dim, latent_dim)
            self.energy_encoder = GRUEncoder(energy_dim, hidden_dim, latent_dim)
        elif encoder_type == 'TCN':
            self.driving_encoder = TCNEncoder(driving_dim, hidden_dim, latent_dim)
            self.energy_encoder = TCNEncoder(energy_dim, hidden_dim, latent_dim)
        
        # 交叉注意力
        self.cross_attn_d2e = CrossChannelAttention(latent_dim)
        self.cross_attn_e2d = CrossChannelAttention(latent_dim)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 4, latent_dim * 3),
            nn.LayerNorm(latent_dim * 3),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(latent_dim * 3, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.ReLU()
        )
        
        # 解码器（简化，只重构最后时刻）
        self.driving_decoder = nn.Linear(latent_dim, driving_dim)
        self.energy_decoder = nn.Linear(latent_dim, energy_dim)
        
        print(f"\n   Encoder type: {encoder_type}")
        print(f"   Driving: {driving_dim}D × seq_len → {latent_dim}D")
        print(f"   Energy: {energy_dim}D × seq_len → {latent_dim}D")
        print(f"   Total parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, driving_seq, energy_seq, lengths=None):
        # 编码
        driving_latent = self.driving_encoder(driving_seq, lengths)
        energy_latent = self.energy_encoder(energy_seq, lengths)
        
        # 交叉注意力
        driving_attended = self.cross_attn_e2d(driving_latent, energy_latent)
        energy_attended = self.cross_attn_d2e(energy_latent, driving_latent)
        
        # 融合
        combined = torch.cat([
            driving_latent, driving_attended,
            energy_latent, energy_attended
        ], dim=1)
        
        fused_latent = self.fusion(combined)
        
        # 解码（重构最后时刻）
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

# ==================== 4. 训练 ====================
print(f"\n{'='*70}")
print("🚀 Training Temporal Models")
print("="*70)

def train_temporal_model(model, data, epochs, batch_size, lr, device):
    """训练时序模型"""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    X_driving = data['driving']
    X_energy = data['energy']
    lengths = data['lengths']
    
    best_loss = float('inf')
    history = []
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        
        indices = np.random.permutation(len(X_driving))
        
        pbar = tqdm(range(0, len(X_driving), batch_size),
                   desc=f"Epoch {epoch+1:2d}/{epochs}",
                   leave=False)
        
        for i in pbar:
            batch_indices = indices[i:i+batch_size]
            
            d_batch = torch.FloatTensor(X_driving[batch_indices]).to(device)
            e_batch = torch.FloatTensor(X_energy[batch_indices]).to(device)
            len_batch = torch.LongTensor(lengths[batch_indices])
            
            # 获取最后一个有效时刻的真实值
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
        history.append(avg_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
    
    return best_loss, history

def extract_temporal_features(model, data, batch_size, device):
    """提取时序潜在特征"""
    model.eval()
    latent_features = []
    
    X_driving = data['driving']
    X_energy = data['energy']
    lengths = data['lengths']
    
    with torch.no_grad():
        for i in range(0, len(X_driving), batch_size):
            d_batch = torch.FloatTensor(X_driving[i:i+batch_size]).to(device)
            e_batch = torch.FloatTensor(X_energy[i:i+batch_size]).to(device)
            len_batch = torch.LongTensor(lengths[i:i+batch_size])
            
            latent = model.encode(d_batch, e_batch, len_batch)
            latent_features.append(latent.cpu().numpy())
    
    return np.vstack(latent_features)

# 对比三种时序模型
temporal_models = ['LSTM', 'GRU', 'TCN']
temporal_results = []

for model_type in temporal_models:
    print(f"\n{'='*60}")
    print(f"Testing: {model_type}")
    print("="*60)
    
    # 构建模型
    model = TemporalDualChannelAE(
        driving_dim=data['driving'].shape[2],
        energy_dim=data['energy'].shape[2],
        latent_dim=CONFIG['latent_dim'],
        encoder_type=model_type,
        hidden_dim=32
    )
    
    # 训练
    print(f"\n🚀 Training...")
    best_loss, history = train_temporal_model(
        model, data,
        epochs=CONFIG['epochs'],
        batch_size=CONFIG['batch_size'],
        lr=CONFIG['lr'],
        device=CONFIG['device']
    )
    
    print(f"✅ Best loss: {best_loss:.4f}")
    
    # 提取特征
    print(f"🔍 Extracting features...")
    latent_features = extract_temporal_features(
        model, data, CONFIG['batch_size'], CONFIG['device']
    )
    
    # 聚类
    print(f"🎯 Clustering (K={CONFIG['k_value']})...")
    kmeans = KMeans(n_clusters=CONFIG['k_value'], random_state=42, n_init=20)
    labels = kmeans.fit_predict(latent_features)
    
    sil = silhouette_score(latent_features, labels)
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    percentages = cluster_counts.values / len(labels) * 100
    cv = percentages.std() / percentages.mean()
    
    # 保存
    torch.save(model.state_dict(), f"{CONFIG['output_dir']}/{model_type}_model.pth")
    np.save(f"{CONFIG['output_dir']}/{model_type}_latent.npy", latent_features)
    
    result = {
        'Model': model_type,
        'Parameters': sum(p.numel() for p in model.parameters()),
        'Best_Loss': best_loss,
        'Silhouette': sil,
        'CV': cv,
        'Distribution': cluster_counts.to_dict()
    }
    
    temporal_results.append(result)
    
    print(f"\n📊 Results:")
    print(f"   Silhouette: {sil:.3f}")
    print(f"   CV: {cv:.3f}")
    print(f"   Distribution: {cluster_counts.to_dict()}")

# ==================== 5. 生成报告 ====================
print(f"\n{'='*70}")
print("📊 Temporal Model Comparison")
print("="*70)

temporal_df = pd.DataFrame(temporal_results)
temporal_df = temporal_df.round(3)

print(f"\n{temporal_df.to_string(index=False)}")

temporal_df.to_csv(f"{CONFIG['output_dir']}/temporal_comparison.csv", index=False)

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Silhouette
ax = axes[0]
bars = ax.bar(temporal_df['Model'], temporal_df['Silhouette'],
             color=sns.color_palette('Set2', len(temporal_models)), alpha=0.8)
ax.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
ax.set_title('Clustering Quality', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, temporal_df['Silhouette']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
           f'{val:.3f}', ha='center', va='bottom', fontsize=11)

# CV
ax = axes[1]
bars = ax.bar(temporal_df['Model'], temporal_df['CV'],
             color=sns.color_palette('Set2', len(temporal_models)), alpha=0.8)
ax.set_ylabel('CV', fontsize=12, fontweight='bold')
ax.set_title('Distribution Balance', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, temporal_df['CV']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
           f'{val:.3f}', ha='center', va='bottom', fontsize=11)

# Loss
ax = axes[2]
bars = ax.bar(temporal_df['Model'], temporal_df['Best_Loss'],
             color=sns.color_palette('Set2', len(temporal_models)), alpha=0.8)
ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax.set_title('Training Quality', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, temporal_df['Best_Loss']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
           f'{val:.3f}', ha='center', va='bottom', fontsize=11)

plt.suptitle('Temporal Model Comparison', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{CONFIG['output_dir']}/temporal_comparison.png", dpi=300, bbox_inches='tight')

print(f"\n✅ Visualization saved")

# 推荐
best_sil = temporal_df.loc[temporal_df['Silhouette'].idxmax()]
best_cv = temporal_df.loc[temporal_df['CV'].idxmin()]

print(f"\n🏆 Recommendation:")
print(f"   Best clustering: {best_sil['Model']} (Silhouette={best_sil['Silhouette']:.3f})")
print(f"   Best balance: {best_cv['Model']} (CV={best_cv['CV']:.3f})")

print("\n" + "="*70)
print("✅ Temporal Model Experiment Complete!")
print("="*70)