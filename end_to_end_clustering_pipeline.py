"""
完整的端到端聚类分析Pipeline
1. 从CSV读取数据（根目录）
2. 按SOC下降≥3%分段（保留怠速）
3. 双通道GRU + Cross-Attention
4. K-means聚类
5. 物理意义分析
6. 综合可视化
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
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("🚀 End-to-End Clustering Pipeline")
print("="*70)


# ==================== Step 1: 加载和预处理数据 ====================
def load_and_preprocess_data(data_dir='.'):
    """
    从多个CSV文件加载数据并合并
    """
    print("\n" + "="*70)
    print("📂 Step 1: Loading and Preprocessing Data")
    print("="*70)
    
    # 修改：在根目录查找CSV文件
    csv_files = list(Path(data_dir).glob('*_processed.csv'))
    
    if len(csv_files) == 0:
        raise FileNotFoundError(f"No *_processed.csv files found in {data_dir}")
    
    print(f"\nFound {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"   - {f.name}")
    
    # 读取所有CSV
    dfs = []
    for file in tqdm(csv_files, desc="Loading CSVs"):
        try:
            df = pd.read_csv(file)
            dfs.append(df)
            print(f"   Loaded {file.name}: {len(df):,} records")
        except Exception as e:
            print(f"   ❌ Error loading {file.name}: {e}")
    
    if len(dfs) == 0:
        raise ValueError("No CSV files were successfully loaded")
    
    # 合并
    df_all = pd.concat(dfs, ignore_index=True)
    
    print(f"\n✅ Total loaded: {len(df_all):,} records")
    print(f"   Columns: {len(df_all.columns)}")
    
    # 检查必需的列
    required_cols = ['vehicle_id', 'time', 'soc', 'spd', 'v', 'i', 'acc']
    missing_cols = [col for col in required_cols if col not in df_all.columns]
    
    if missing_cols:
        print(f"\n⚠️  Missing columns: {missing_cols}")
        print(f"   Available columns: {df_all.columns.tolist()}")
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    if 'vehicle_id' in df_all.columns:
        print(f"   Vehicles: {df_all['vehicle_id'].nunique()}")
    
    # 数据清洗
    print("\n🔧 Cleaning data...")
    
    # 1. 去除缺失值
    before = len(df_all)
    df_all = df_all.dropna(subset=['soc', 'spd', 'v', 'i'])
    after = len(df_all)
    print(f"   Removed {before - after:,} rows with missing values")
    
    # 2. 按vehicle_id和时间排序
    df_all = df_all.sort_values(['vehicle_id', 'time'])
    
    # 3. 过滤异常值
    print(f"\n   Data ranges before filtering:")
    print(f"      SOC: [{df_all['soc'].min():.1f}, {df_all['soc'].max():.1f}]")
    print(f"      Speed: [{df_all['spd'].min():.1f}, {df_all['spd'].max():.1f}]")
    print(f"      Voltage: [{df_all['v'].min():.1f}, {df_all['v'].max():.1f}]")
    
    df_all = df_all[
        (df_all['soc'] >= 0) & (df_all['soc'] <= 100) &
        (df_all['spd'] >= 0) & (df_all['spd'] <= 220) &
        (df_all['v'] > 0) & (df_all['v'] <= 1000)
    ]
    
    print(f"   After filtering: {len(df_all):,} records")
    
    return df_all


# ==================== Step 2: 按SOC下降≥3%分段 ====================
def segment_by_soc_drop(df, soc_drop_threshold=3.0, min_length=10):
    """
    按SOC累计下降≥3%分段，保留怠速
    """
    print("\n" + "="*70)
    print("📊 Step 2: Segmenting Trips by SOC Drop ≥ 3%")
    print("="*70)
    
    segments = []
    
    vehicle_ids = df['vehicle_id'].unique()
    print(f"\nProcessing {len(vehicle_ids)} vehicles...")
    
    for vehicle_id in tqdm(vehicle_ids, desc="Segmenting"):
        vehicle_data = df[df['vehicle_id'] == vehicle_id].copy()
        vehicle_data = vehicle_data.sort_values('time')
        
        if len(vehicle_data) < min_length:
            continue
        
        soc_values = vehicle_data['soc'].values
        
        # 寻找SOC下降≥3%的片段
        start_idx = 0
        
        while start_idx < len(vehicle_data):
            soc_start = soc_values[start_idx]
            
            for end_idx in range(start_idx + 1, len(vehicle_data)):
                soc_current = soc_values[end_idx]
                soc_drop = soc_start - soc_current
                
                # 如果SOC上升（充电），重新开始
                if soc_current > soc_start:
                    start_idx = end_idx
                    break
                
                # 如果SOC下降≥3%，保存片段
                if soc_drop >= soc_drop_threshold:
                    segment = vehicle_data.iloc[start_idx:end_idx+1]
                    
                    if len(segment) >= min_length:
                        segments.append(segment)
                    
                    start_idx = end_idx + 1
                    break
            else:
                # 遍历完了，没找到
                start_idx += 1
    
    print(f"\n✅ Segmented into {len(segments):,} trips")
    
    if len(segments) == 0:
        raise ValueError("No segments found! Check SOC drop threshold and min_length")
    
    # 统计
    lengths = [len(seg) for seg in segments]
    print(f"   Length: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}")
    
    # SOC下降统计
    soc_drops = [seg['soc'].iloc[0] - seg['soc'].iloc[-1] for seg in segments if len(seg) > 1]
    print(f"   SOC drop: min={min(soc_drops):.1f}%, max={max(soc_drops):.1f}%, mean={np.mean(soc_drops):.1f}%")
    
    return segments


# ==================== Step 3: 特征提取 ====================
def extract_features(segments):
    """
    提取双通道特征：驾驶通道 + 能量通道
    """
    print("\n" + "="*70)
    print("🔍 Step 3: Extracting Features")
    print("="*70)
    
    driving_sequences = []
    energy_sequences = []
    
    for segment in tqdm(segments, desc="Extracting features"):
        # 驾驶通道特征：速度、加速度
        driving_feat = segment[['spd', 'acc']].values
        
        # 能量通道特征：SOC、电压、电流
        energy_feat = segment[['soc', 'v', 'i']].values
        
        driving_sequences.append(driving_feat)
        energy_sequences.append(energy_feat)
    
    print(f"\n✅ Extracted features from {len(driving_sequences):,} trips")
    print(f"   Driving features: 2D (speed, acceleration)")
    print(f"   Energy features: 3D (SOC, voltage, current)")
    
    return driving_sequences, energy_sequences


# ==================== Step 4: 特征归一化 ====================
def normalize_features(driving_seqs, energy_seqs):
    """
    使用RobustScaler归一化特征
    """
    print("\n" + "="*70)
    print("🔧 Step 4: Normalizing Features")
    print("="*70)
    
    # 合并所有数据用于拟合scaler
    all_driving = np.vstack(driving_seqs)
    all_energy = np.vstack(energy_seqs)
    
    print(f"\n   Total data points:")
    print(f"      Driving: {all_driving.shape}")
    print(f"      Energy: {all_energy.shape}")
    
    # 拟合scaler
    driving_scaler = RobustScaler()
    energy_scaler = RobustScaler()
    
    driving_scaler.fit(all_driving)
    energy_scaler.fit(all_energy)
    
    print(f"\n   Driving scaler parameters:")
    print(f"      Center: {driving_scaler.center_}")
    print(f"      Scale: {driving_scaler.scale_}")
    
    print(f"\n   Energy scaler parameters:")
    print(f"      Center: {energy_scaler.center_}")
    print(f"      Scale: {energy_scaler.scale_}")
    
    # 归一化每个序列
    driving_normalized = []
    energy_normalized = []
    
    for d_seq, e_seq in tqdm(zip(driving_seqs, energy_seqs), 
                             total=len(driving_seqs), 
                             desc="Normalizing"):
        driving_normalized.append(driving_scaler.transform(d_seq))
        energy_normalized.append(energy_scaler.transform(e_seq))
    
    print(f"\n✅ Normalized {len(driving_normalized):,} sequences")
    
    # 保存scaler用于反归一化
    Path('./results').mkdir(exist_ok=True)
    
    scalers = {
        'driving': driving_scaler,
        'energy': energy_scaler
    }
    
    import pickle
    with open('./results/scalers.pkl', 'wb') as f:
        pickle.dump(scalers, f)
    
    print(f"💾 Saved scalers to: ./results/scalers.pkl")
    
    return driving_normalized, energy_normalized, scalers


# ==================== Step 5: Dataset和DataLoader ====================
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


# ==================== Step 6: GRU + Cross-Attention模型 ====================
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
        
        out = torch.matmul(attn, v)
        return out


class DualChannelGRU_CrossAttn(nn.Module):
    def __init__(self, driving_dim, energy_dim, latent_dim):
        super().__init__()
        
        print(f"\n🏗️  Building Dual-Channel GRU with Cross-Attention...")
        print(f"   Driving dim: {driving_dim}, Energy dim: {energy_dim}, Latent dim: {latent_dim}")
        
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


# ==================== Step 7: 训练模型 ====================
def train_model(model, train_loader, epochs=10):
    print("\n" + "="*70)
    print("🚀 Step 5: Training GRU Model")
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
            
            print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f} | Best: {best_loss:.4f}")
    
    print(f"\n✅ Training completed | Best loss: {best_loss:.4f}")
    
    # 保存模型
    torch.save(model.state_dict(), './results/gru_crossattn_model.pth')
    print(f"💾 Model saved: ./results/gru_crossattn_model.pth")
    
    return model


# ==================== Step 8: 提取特征 ====================
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
    unsort_indices = np.argsort(indices)
    features = features[unsort_indices]
    
    print(f"✅ Features extracted: {features.shape}")
    
    return features


# ==================== Step 9: 聚类 ====================
def perform_clustering(features, k=4):
    print("\n" + "="*70)
    print(f"🎯 Step 7: K-means Clustering (K={k})")
    print("="*70)
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
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


# ==================== Step 10: 物理意义分析 ====================
def analyze_physical_meaning(segments, labels, scalers):
    print("\n" + "="*70)
    print("💡 Step 8: Physical Meaning Analysis")
    print("="*70)
    
    cluster_stats = []
    
    for cluster_id in range(4):
        print(f"\n  Analyzing Cluster {cluster_id}...")
        
        cluster_segments = [seg for seg, label in zip(segments, labels) if label == cluster_id]
        
        if len(cluster_segments) == 0:
            print(f"     ⚠️  No segments in cluster {cluster_id}")
            continue
        
        stats = {'cluster': cluster_id, 'count': len(cluster_segments)}
        
        # 合并所有片段的数据
        all_data = pd.concat(cluster_segments, ignore_index=True)
        
        # 速度特征（真实单位 km/h）
        stats['avg_speed'] = all_data['spd'].mean()
        stats['max_speed'] = all_data['spd'].quantile(0.95)
        stats['speed_std'] = all_data['spd'].std()
        stats['idle_ratio'] = (all_data['spd'] < 1).mean() * 100
        
        # 加速度特征
        if 'acc' in all_data.columns:
            stats['avg_accel'] = all_data['acc'].abs().mean()
            stats['accel_std'] = all_data['acc'].std()
        else:
            stats['avg_accel'] = 0
            stats['accel_std'] = 0
        
        # 能量特征（真实单位）
        stats['avg_soc'] = all_data['soc'].mean()
        stats['soc_drop'] = all_data.groupby(level=0)['soc'].apply(lambda x: x.iloc[0] - x.iloc[-1]).mean()
        
        # 功率（如果有power列）
        if 'power' in all_data.columns:
            stats['avg_power'] = all_data['power'].abs().mean()
        else:
            # 计算功率 = 电压 × 电流
            stats['avg_power'] = (all_data['v'] * all_data['i']).abs().mean() / 1000  # kW
        
        # 行程特征
        stats['avg_trip_length'] = np.mean([len(seg) for seg in cluster_segments])
        stats['avg_trip_duration'] = stats['avg_trip_length'] * 10  # 假设10秒采样间隔
        
        cluster_stats.append(stats)
        
        print(f"     Samples: {stats['count']:,}")
        print(f"     Avg Speed: {stats['avg_speed']:.2f} km/h")
        print(f"     Max Speed: {stats['max_speed']:.2f} km/h")
        print(f"     Idle Ratio: {stats['idle_ratio']:.1f}%")
        print(f"     Avg Power: {stats['avg_power']:.2f} kW")
        print(f"     SOC Drop: {stats['soc_drop']:.2f}%")
    
    df_stats = pd.DataFrame(cluster_stats)
    df_stats.to_csv('./results/cluster_statistics.csv', index=False, encoding='utf-8-sig')
    print(f"\n💾 Saved: ./results/cluster_statistics.csv")
    
    return df_stats


# ==================== Step 11: 综合可视化 ====================
def create_comprehensive_visualization(df_stats, features, labels):
    print("\n" + "="*70)
    print("🎨 Step 9: Creating Comprehensive Visualization")
    print("="*70)
    
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    colors = ['#3498DB', '#2ECC71', '#E74C3C', '#F39C12']
    
    # 1. PCA
    ax1 = fig.add_subplot(gs[0, 0])
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)
    
    for cluster_id in range(4):
        if cluster_id in labels:
            mask = labels == cluster_id
            ax1.scatter(features_2d[mask, 0], features_2d[mask, 1],
                       c=colors[cluster_id], label=f'C{cluster_id}',
                       alpha=0.6, s=20, edgecolors='none')
    
    ax1.set_xlabel('PC1', fontsize=11, fontweight='bold')
    ax1.set_ylabel('PC2', fontsize=11, fontweight='bold')
    ax1.set_title('PCA Space', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2-6. 特征对比
    feature_plots = [
        ('avg_speed', 'Average Speed (km/h)'),
        ('idle_ratio', 'Idle Ratio (%)'),
        ('avg_power', 'Average Power (kW)'),
        ('avg_trip_length', 'Trip Length (points)'),
        ('soc_drop', 'SOC Drop (%)'),
    ]
    
    for idx, (feat, ylabel) in enumerate(feature_plots, start=1):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        
        if feat not in df_stats.columns:
            ax.text(0.5, 0.5, f'{feat} not available', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        values = df_stats[feat].values
        x = df_stats['cluster'].values.astype(int)
        
        bars = ax.bar(x, values, color=[colors[i] for i in x], 
                      alpha=0.85, edgecolor='black', linewidth=2, width=0.65)
        
        ax.set_xlabel('Cluster', fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=10, fontweight='bold')
        ax.set_title(ylabel, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'C{i}' for i in x])
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, val, cid in zip(bars, values, x):
            height = bar.get_height()
            ax.text(cid, height, f'{val:.1f}', 
                   ha='center', va='bottom',
                   fontsize=9, fontweight='bold')
    
    plt.suptitle('Comprehensive Cluster Analysis (Real Physical Units)', 
                fontsize=18, fontweight='bold')
    plt.savefig('./results/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: ./results/comprehensive_analysis.png")


# ==================== Main Pipeline ====================
def main():
    # Step 1: 加载数据（根目录）
    df = load_and_preprocess_data('.')  # 修改为根目录
    
    # Step 2: 分段
    segments = segment_by_soc_drop(df, soc_drop_threshold=3.0, min_length=10)
    
    # Step 3: 特征提取
    driving_seqs, energy_seqs = extract_features(segments)
    
    # Step 4: 归一化
    driving_norm, energy_norm, scalers = normalize_features(driving_seqs, energy_seqs)
    
    # Step 5: 创建DataLoader
    lengths = [len(seq) for seq in driving_norm]
    dataset = VariableLengthDataset(driving_norm, energy_norm, lengths)
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True,
                             collate_fn=collate_variable_length, num_workers=0)
    
    # Step 6: 创建模型
    model = DualChannelGRU_CrossAttn(driving_dim=2, energy_dim=3, latent_dim=16).to(device)
    
    # Step 7: 训练
    model = train_model(model, train_loader, epochs=10)
    
    # Step 8: 提取GRU特征
    features = extract_gru_features(model, train_loader)
    
    # Step 9: 聚类
    labels, kmeans, sil, cv = perform_clustering(features, k=4)
    
    # Step 10: 物理意义分析
    df_stats = analyze_physical_meaning(segments, labels, scalers)
    
    # Step 11: 可视化
    create_comprehensive_visualization(df_stats, features, labels)
    
    # 保存
    np.save('./results/features.npy', features)
    np.save('./results/labels.npy', labels)
    
    print("\n" + "="*70)
    print("✅ Pipeline Complete!")
    print("="*70)
    print("\n📁 Generated files:")
    print("   ./results/scalers.pkl")
    print("   ./results/gru_crossattn_model.pth")
    print("   ./results/features.npy")
    print("   ./results/labels.npy")
    print("   ./results/cluster_statistics.csv")
    print("   ./results/comprehensive_analysis.png")
    print("="*70)


if __name__ == "__main__":
    main()