"""
对比5种融合方法：
1. 直接拼接（Concat）
2. 残差连接（Residual）
3. 加权拼接（Weighted）
4. 门控融合（Gated）
5. 注意力池化（Attention Pooling）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print("🔬 Fusion Method Comparison Experiment")
print("="*70)

# ==================== 配置 ====================
CONFIG = {
    'latent_dim': 8,
    'batch_size': 512,
    'epochs': 30,
    'lr': 0.0001,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'k_values': [3, 4, 5]
}

# ==================== 基础模块 ====================
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

# ==================== 方法1：直接拼接 ====================
class Method1_DirectConcat(nn.Module):
    """方法1：直接拼接4个向量 (32D→16D)"""
    def __init__(self, driving_dim, energy_dim, latent_dim=8):
        super().__init__()
        self.name = "Direct Concatenation"
        
        self.driving_encoder = nn.Sequential(
            nn.Linear(driving_dim, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.LayerNorm(32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, latent_dim), nn.ReLU()
        )
        
        self.energy_encoder = nn.Sequential(
            nn.Linear(energy_dim, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.LayerNorm(32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, latent_dim), nn.ReLU()
        )
        
        self.cross_attn_d2e = CrossChannelAttention(latent_dim)
        self.cross_attn_e2d = CrossChannelAttention(latent_dim)
        
        # 融合层：32D → 16D
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 4, latent_dim * 3),
            nn.LayerNorm(latent_dim * 3),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(latent_dim * 3, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.ReLU()
        )
        
        self.driving_decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, driving_dim)
        )
        
        self.energy_decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, energy_dim)
        )
    
    def forward(self, driving_features, energy_features):
        driving_latent = self.driving_encoder(driving_features)
        energy_latent = self.energy_encoder(energy_features)
        
        driving_attended = self.cross_attn_e2d(driving_latent, energy_latent)
        energy_attended = self.cross_attn_d2e(energy_latent, driving_latent)
        
        # 直接拼接4个向量
        combined = torch.cat([
            driving_latent, driving_attended,
            energy_latent, energy_attended
        ], dim=1)  # 32D
        
        fused_latent = self.fusion(combined)  # 16D
        
        driving_recon = self.driving_decoder(driving_latent)
        energy_recon = self.energy_decoder(energy_latent)
        
        return driving_recon, energy_recon, fused_latent
    
    def encode(self, driving_features, energy_features):
        driving_latent = self.driving_encoder(driving_features)
        energy_latent = self.energy_encoder(energy_features)
        driving_attended = self.cross_attn_e2d(driving_latent, energy_latent)
        energy_attended = self.cross_attn_d2e(energy_latent, driving_latent)
        combined = torch.cat([driving_latent, driving_attended, energy_latent, energy_attended], dim=1)
        return self.fusion(combined)

# ==================== 方法2：残差连接 ====================
class Method2_Residual(nn.Module):
    """方法2：残差连接 (16D)"""
    def __init__(self, driving_dim, energy_dim, latent_dim=8):
        super().__init__()
        self.name = "Residual Connection"
        
        self.driving_encoder = nn.Sequential(
            nn.Linear(driving_dim, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.LayerNorm(32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, latent_dim), nn.ReLU()
        )
        
        self.energy_encoder = nn.Sequential(
            nn.Linear(energy_dim, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.LayerNorm(32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, latent_dim), nn.ReLU()
        )
        
        self.cross_attn_d2e = CrossChannelAttention(latent_dim)
        self.cross_attn_e2d = CrossChannelAttention(latent_dim)
        
        # 融合层：16D → 16D
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.ReLU()
        )
        
        self.driving_decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, driving_dim)
        )
        
        self.energy_decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, energy_dim)
        )
    
    def forward(self, driving_features, energy_features):
        driving_latent = self.driving_encoder(driving_features)
        energy_latent = self.energy_encoder(energy_features)
        
        driving_attended = self.cross_attn_e2d(driving_latent, energy_latent)
        energy_attended = self.cross_attn_d2e(energy_latent, driving_latent)
        
        # 残差连接
        driving_fused = driving_latent + driving_attended
        energy_fused = energy_latent + energy_attended
        
        combined = torch.cat([driving_fused, energy_fused], dim=1)  # 16D
        fused_latent = self.fusion(combined)  # 16D
        
        driving_recon = self.driving_decoder(driving_latent)
        energy_recon = self.energy_decoder(energy_latent)
        
        return driving_recon, energy_recon, fused_latent
    
    def encode(self, driving_features, energy_features):
        driving_latent = self.driving_encoder(driving_features)
        energy_latent = self.energy_encoder(energy_features)
        driving_attended = self.cross_attn_e2d(driving_latent, energy_latent)
        energy_attended = self.cross_attn_d2e(energy_latent, driving_latent)
        driving_fused = driving_latent + driving_attended
        energy_fused = energy_latent + energy_attended
        combined = torch.cat([driving_fused, energy_fused], dim=1)
        return self.fusion(combined)

# ==================== 方法3：加权拼接 ====================
class Method3_Weighted(nn.Module):
    """方法3：可学习权重拼接 (16D)"""
    def __init__(self, driving_dim, energy_dim, latent_dim=8):
        super().__init__()
        self.name = "Weighted Fusion"
        
        self.driving_encoder = nn.Sequential(
            nn.Linear(driving_dim, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.LayerNorm(32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, latent_dim), nn.ReLU()
        )
        
        self.energy_encoder = nn.Sequential(
            nn.Linear(energy_dim, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.LayerNorm(32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, latent_dim), nn.ReLU()
        )
        
        self.cross_attn_d2e = CrossChannelAttention(latent_dim)
        self.cross_attn_e2d = CrossChannelAttention(latent_dim)
        
        # 可学习权重
        self.alpha_driving = nn.Parameter(torch.tensor(0.5))
        self.alpha_energy = nn.Parameter(torch.tensor(0.5))
        
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.ReLU()
        )
        
        self.driving_decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, driving_dim)
        )
        
        self.energy_decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, energy_dim)
        )
    
    def forward(self, driving_features, energy_features):
        driving_latent = self.driving_encoder(driving_features)
        energy_latent = self.energy_encoder(energy_features)
        
        driving_attended = self.cross_attn_e2d(driving_latent, energy_latent)
        energy_attended = self.cross_attn_d2e(energy_latent, driving_latent)
        
        # 加权融合
        alpha_d = torch.sigmoid(self.alpha_driving)
        alpha_e = torch.sigmoid(self.alpha_energy)
        
        driving_fused = alpha_d * driving_latent + (1 - alpha_d) * driving_attended
        energy_fused = alpha_e * energy_latent + (1 - alpha_e) * energy_attended
        
        combined = torch.cat([driving_fused, energy_fused], dim=1)  # 16D
        fused_latent = self.fusion(combined)
        
        driving_recon = self.driving_decoder(driving_latent)
        energy_recon = self.energy_decoder(energy_latent)
        
        return driving_recon, energy_recon, fused_latent
    
    def encode(self, driving_features, energy_features):
        driving_latent = self.driving_encoder(driving_features)
        energy_latent = self.energy_encoder(energy_features)
        driving_attended = self.cross_attn_e2d(driving_latent, energy_latent)
        energy_attended = self.cross_attn_d2e(energy_latent, driving_latent)
        alpha_d = torch.sigmoid(self.alpha_driving)
        alpha_e = torch.sigmoid(self.alpha_energy)
        driving_fused = alpha_d * driving_latent + (1 - alpha_d) * driving_attended
        energy_fused = alpha_e * energy_latent + (1 - alpha_e) * energy_attended
        combined = torch.cat([driving_fused, energy_fused], dim=1)
        return self.fusion(combined)

# ==================== 方法4：门控融合 ====================
class Method4_Gated(nn.Module):
    """方法4：门控机制融合 (16D)"""
    def __init__(self, driving_dim, energy_dim, latent_dim=8):
        super().__init__()
        self.name = "Gated Fusion"
        
        self.driving_encoder = nn.Sequential(
            nn.Linear(driving_dim, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.LayerNorm(32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, latent_dim), nn.ReLU()
        )
        
        self.energy_encoder = nn.Sequential(
            nn.Linear(energy_dim, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.LayerNorm(32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, latent_dim), nn.ReLU()
        )
        
        self.cross_attn_d2e = CrossChannelAttention(latent_dim)
        self.cross_attn_e2d = CrossChannelAttention(latent_dim)
        
        # 门控网络
        self.gate_driving = nn.Linear(latent_dim * 2, latent_dim)
        self.gate_energy = nn.Linear(latent_dim * 2, latent_dim)
        
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.ReLU()
        )
        
        self.driving_decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, driving_dim)
        )
        
        self.energy_decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, energy_dim)
        )
    
    def forward(self, driving_features, energy_features):
        driving_latent = self.driving_encoder(driving_features)
        energy_latent = self.energy_encoder(energy_features)
        
        driving_attended = self.cross_attn_e2d(driving_latent, energy_latent)
        energy_attended = self.cross_attn_d2e(energy_latent, driving_latent)
        
        # 门控融合
        gate_d = torch.sigmoid(self.gate_driving(
            torch.cat([driving_latent, driving_attended], dim=1)))
        driving_fused = gate_d * driving_latent + (1 - gate_d) * driving_attended
        
        gate_e = torch.sigmoid(self.gate_energy(
            torch.cat([energy_latent, energy_attended], dim=1)))
        energy_fused = gate_e * energy_latent + (1 - gate_e) * energy_attended
        
        combined = torch.cat([driving_fused, energy_fused], dim=1)  # 16D
        fused_latent = self.fusion(combined)
        
        driving_recon = self.driving_decoder(driving_latent)
        energy_recon = self.energy_decoder(energy_latent)
        
        return driving_recon, energy_recon, fused_latent
    
    def encode(self, driving_features, energy_features):
        driving_latent = self.driving_encoder(driving_features)
        energy_latent = self.energy_encoder(energy_features)
        driving_attended = self.cross_attn_e2d(driving_latent, energy_latent)
        energy_attended = self.cross_attn_d2e(energy_latent, driving_latent)
        gate_d = torch.sigmoid(self.gate_driving(torch.cat([driving_latent, driving_attended], dim=1)))
        driving_fused = gate_d * driving_latent + (1 - gate_d) * driving_attended
        gate_e = torch.sigmoid(self.gate_energy(torch.cat([energy_latent, energy_attended], dim=1)))
        energy_fused = gate_e * energy_latent + (1 - gate_e) * energy_attended
        combined = torch.cat([driving_fused, energy_fused], dim=1)
        return self.fusion(combined)

# ==================== 方法5：注意力池化 ====================
class Method5_AttentionPooling(nn.Module):
    """方法5：注意力池化 (8D)"""
    def __init__(self, driving_dim, energy_dim, latent_dim=8):
        super().__init__()
        self.name = "Attention Pooling"
        
        self.driving_encoder = nn.Sequential(
            nn.Linear(driving_dim, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.LayerNorm(32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, latent_dim), nn.ReLU()
        )
        
        self.energy_encoder = nn.Sequential(
            nn.Linear(energy_dim, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.LayerNorm(32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, latent_dim), nn.ReLU()
        )
        
        self.cross_attn_d2e = CrossChannelAttention(latent_dim)
        self.cross_attn_e2d = CrossChannelAttention(latent_dim)
        
        # 注意力池化
        self.attention_weights = nn.Linear(latent_dim, 1)
        
        # 简单融合
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.ReLU()
        )
        
        self.driving_decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, driving_dim)
        )
        
        self.energy_decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, energy_dim)
        )
    
    def forward(self, driving_features, energy_features):
        driving_latent = self.driving_encoder(driving_features)
        energy_latent = self.energy_encoder(energy_features)
        
        driving_attended = self.cross_attn_e2d(driving_latent, energy_latent)
        energy_attended = self.cross_attn_d2e(energy_latent, driving_latent)
        
        # 注意力池化
        stacked = torch.stack([
            driving_latent, driving_attended,
            energy_latent, energy_attended
        ], dim=1)  # (batch, 4, latent_dim)
        
        attn_scores = self.attention_weights(stacked)  # (batch, 4, 1)
        attn_weights = F.softmax(attn_scores, dim=1)
        
        pooled = (stacked * attn_weights).sum(dim=1)  # (batch, latent_dim)
        
        fused_latent = self.fusion(pooled)  # 8D → 16D
        
        driving_recon = self.driving_decoder(driving_latent)
        energy_recon = self.energy_decoder(energy_latent)
        
        return driving_recon, energy_recon, fused_latent
    
    def encode(self, driving_features, energy_features):
        driving_latent = self.driving_encoder(driving_features)
        energy_latent = self.energy_encoder(energy_features)
        driving_attended = self.cross_attn_e2d(driving_latent, energy_latent)
        energy_attended = self.cross_attn_d2e(energy_latent, driving_latent)
        stacked = torch.stack([driving_latent, driving_attended, energy_latent, energy_attended], dim=1)
        attn_scores = self.attention_weights(stacked)
        attn_weights = F.softmax(attn_scores, dim=1)
        pooled = (stacked * attn_weights).sum(dim=1)
        return self.fusion(pooled)

# ==================== 训练和评估函数 ====================
def train_model(model, X_driving, X_energy, epochs=30, batch_size=512, lr=0.0001):
    """训练单个模型"""
    device = CONFIG['device']
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    best_loss = float('inf')
    patience = 0
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        
        indices = np.random.permutation(len(X_driving))
        
        for i in range(0, len(X_driving), batch_size):
            batch_indices = indices[i:i+batch_size]
            
            d_batch = torch.FloatTensor(X_driving[batch_indices]).to(device)
            e_batch = torch.FloatTensor(X_energy[batch_indices]).to(device)
            
            optimizer.zero_grad()
            
            d_recon, e_recon, _ = model(d_batch, e_batch)
            
            loss = F.mse_loss(d_recon, d_batch) + F.mse_loss(e_recon, e_batch)
            
            if torch.isnan(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience = 0
        else:
            patience += 1
            if patience >= 10:
                break
    
    return best_loss

def extract_features(model, X_driving, X_energy, batch_size=512):
    """提取潜在特征"""
    device = CONFIG['device']
    model.eval()
    latent_features = []
    
    with torch.no_grad():
        for i in range(0, len(X_driving), batch_size):
            d_batch = torch.FloatTensor(X_driving[i:i+batch_size]).to(device)
            e_batch = torch.FloatTensor(X_energy[i:i+batch_size]).to(device)
            latent = model.encode(d_batch, e_batch)
            latent_features.append(latent.cpu().numpy())
    
    return np.vstack(latent_features)

def evaluate_clustering(latent_features, k_values=[3, 4, 5]):
    """评估聚类效果"""
    results = {}
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = kmeans.fit_predict(latent_features)
        
        sil = silhouette_score(latent_features, labels)
        ch = calinski_harabasz_score(latent_features, labels)
        db = davies_bouldin_score(latent_features, labels)
        
        # 簇分布
        cluster_counts = pd.Series(labels).value_counts().sort_index()
        percentages = cluster_counts.values / len(labels) * 100
        cv = percentages.std() / percentages.mean()
        
        results[k] = {
            'labels': labels,
            'silhouette': sil,
            'ch_score': ch,
            'db_score': db,
            'cv': cv,
            'distribution': cluster_counts.to_dict(),
            'min_pct': percentages.min(),
            'max_pct': percentages.max()
        }
    
    return results

# ==================== 加载数据 ====================
print(f"\n📂 Loading data...")

features_df = pd.read_csv('./results/reloaded_full/trip_features_full.csv')
print(f"✅ Loaded: {len(features_df)} trips")

# 清洗数据
features_df = features_df[
    (features_df['duration_minutes'] > 2) & (features_df['duration_minutes'] < 300) &
    (features_df['speed_mean'] >= 0) & (features_df['speed_mean'] < 150) &
    (features_df['distance_total'] >= 0) & (features_df['distance_total'] < 500)
]

print(f"✅ After cleaning: {len(features_df)} trips")

# 准备双通道数据
driving_features = [
    'speed_mean', 'speed_max', 'speed_std', 'speed_median',
    'acc_mean', 'acc_std', 'harsh_accel', 'harsh_decel',
    'distance_total', 'duration_minutes', 'moving_ratio',
    'low_speed_ratio', 'medium_speed_ratio', 'high_speed_ratio'
]

energy_features = [
    'soc_drop_total', 'soc_mean', 'soc_std',
    'voltage_mean', 'voltage_std', 'current_mean', 'current_max',
    'power_mean', 'power_max', 'power_std',
    'energy_consumption_total', 'charging_ratio', 'regen_braking_ratio'
]

available_driving = [f for f in driving_features if f in features_df.columns]
available_energy = [f for f in energy_features if f in features_df.columns]

X_driving = np.nan_to_num(features_df[available_driving].values)
X_energy = np.nan_to_num(features_df[available_energy].values)

scaler_d = RobustScaler()
scaler_e = RobustScaler()

X_driving = np.clip(scaler_d.fit_transform(X_driving), -5, 5)
X_energy = np.clip(scaler_e.fit_transform(X_energy), -5, 5)

print(f"✅ Prepared: Driving {X_driving.shape}, Energy {X_energy.shape}")

# ==================== 对比实验 ====================
print(f"\n{'='*70}")
print("🔬 Running Comparison Experiments")
print("="*70)

models = [
    Method1_DirectConcat(X_driving.shape[1], X_energy.shape[1], CONFIG['latent_dim']),
    Method2_Residual(X_driving.shape[1], X_energy.shape[1], CONFIG['latent_dim']),
    Method3_Weighted(X_driving.shape[1], X_energy.shape[1], CONFIG['latent_dim']),
    Method4_Gated(X_driving.shape[1], X_energy.shape[1], CONFIG['latent_dim']),
    Method5_AttentionPooling(X_driving.shape[1], X_energy.shape[1], CONFIG['latent_dim'])
]

comparison_results = []

for method_id, model in enumerate(models, 1):
    print(f"\n{'='*70}")
    print(f"Method {method_id}: {model.name}")
    print("="*70)
    
    # 训练
    print(f"\n🚀 Training...")
    best_loss = train_model(model, X_driving, X_energy, 
                           epochs=CONFIG['epochs'],
                           batch_size=CONFIG['batch_size'],
                           lr=CONFIG['lr'])
    
    print(f"✅ Best reconstruction loss: {best_loss:.4f}")
    
    # 提取特征
    print(f"🔍 Extracting features...")
    latent_features = extract_features(model, X_driving, X_energy, CONFIG['batch_size'])
    print(f"✅ Latent shape: {latent_features.shape}")
    
    # 聚类评估
    print(f"🎯 Clustering evaluation...")
    clustering_results = evaluate_clustering(latent_features, CONFIG['k_values'])
    
    # 记录结果
    for k in CONFIG['k_values']:
        res = clustering_results[k]
        
        comparison_results.append({
            'Method': model.name,
            'K': k,
            'Recon_Loss': best_loss,
            'Silhouette': res['silhouette'],
            'CH_Score': res['ch_score'],
            'DB_Score': res['db_score'],
            'CV': res['cv'],
            'Min_Cluster_%': res['min_pct'],
            'Max_Cluster_%': res['max_pct']
        })
        
        print(f"\n  K={k}:")
        print(f"    Silhouette: {res['silhouette']:.3f}")
        print(f"    CV: {res['cv']:.3f}")
        print(f"    Distribution: {res['distribution']}")
    
    # 保存模型和特征
    torch.save(model.state_dict(), f'./results/reloaded_full/method{method_id}_model.pth')
    np.save(f'./results/reloaded_full/method{method_id}_latent.npy', latent_features)
    
    # 保存聚类结果
    for k in CONFIG['k_values']:
        features_df[f'cluster_method{method_id}_k{k}'] = clustering_results[k]['labels']

# ==================== 生成对比报告 ====================
print(f"\n{'='*70}")
print("📊 Generating Comparison Report")
print("="*70)

comparison_df = pd.DataFrame(comparison_results)
comparison_df = comparison_df.round(3)

print(f"\n📋 Full Comparison Table:")
print(comparison_df.to_string(index=False))

# 保存表格
comparison_df.to_csv('./results/reloaded_full/fusion_method_comparison.csv', index=False)

# 找出最佳方法
print(f"\n{'='*70}")
print("🏆 Best Method Selection")
print("="*70)

for k in CONFIG['k_values']:
    k_results = comparison_df[comparison_df['K'] == k]
    
    print(f"\nK={k}:")
    
    # 按Silhouette排序
    best_sil = k_results.loc[k_results['Silhouette'].idxmax()]
    print(f"  Best Silhouette: {best_sil['Method']} ({best_sil['Silhouette']:.3f})")
    
    # 按CV排序（越小越好）
    best_cv = k_results.loc[k_results['CV'].idxmin()]
    print(f"  Best Distribution (lowest CV): {best_cv['Method']} (CV={best_cv['CV']:.3f})")
    
    # 综合评分（归一化后加权）
    k_results_norm = k_results.copy()
    k_results_norm['Sil_norm'] = k_results['Silhouette'] / k_results['Silhouette'].max()
    k_results_norm['CV_norm'] = k_results['CV'].min() / k_results['CV']
    k_results_norm['Score'] = k_results_norm['Sil_norm'] * 0.6 + k_results_norm['CV_norm'] * 0.4
    
    best_overall = k_results_norm.loc[k_results_norm['Score'].idxmax()]
    print(f"  Best Overall: {best_overall['Method']} (Score={best_overall['Score']:.3f})")

# ==================== 可视化对比 ====================
print(f"\n📊 Creating visualization...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

methods = comparison_df['Method'].unique()
colors = sns.color_palette('Set2', len(methods))

# 1. Silhouette对比
ax1 = axes[0, 0]
for k in CONFIG['k_values']:
    k_data = comparison_df[comparison_df['K'] == k]
    x = np.arange(len(methods))
    ax1.bar(x + (k-3)*0.25, k_data['Silhouette'], 0.25, 
           label=f'K={k}', alpha=0.8)

ax1.set_xlabel('Fusion Method', fontsize=12, fontweight='bold')
ax1.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
ax1.set_title('Clustering Quality (Silhouette)', fontsize=14, fontweight='bold')
ax1.set_xticks(np.arange(len(methods)))
ax1.set_xticklabels([m.split()[0] for m in methods], rotation=15, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# 2. CV对比（越小越好）
ax2 = axes[0, 1]
for k in CONFIG['k_values']:
    k_data = comparison_df[comparison_df['K'] == k]
    x = np.arange(len(methods))
    ax2.bar(x + (k-3)*0.25, k_data['CV'], 0.25,
           label=f'K={k}', alpha=0.8)

ax2.set_xlabel('Fusion Method', fontsize=12, fontweight='bold')
ax2.set_ylabel('Coefficient of Variation', fontsize=12, fontweight='bold')
ax2.set_title('Distribution Balance (lower is better)', fontsize=14, fontweight='bold')
ax2.set_xticks(np.arange(len(methods)))
ax2.set_xticklabels([m.split()[0] for m in methods], rotation=15, ha='right')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# 3. 重构损失对比
ax3 = axes[1, 0]
recon_losses = comparison_df.groupby('Method')['Recon_Loss'].first()
bars = ax3.bar(range(len(recon_losses)), recon_losses.values, 
              color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax3.set_xlabel('Fusion Method', fontsize=12, fontweight='bold')
ax3.set_ylabel('Reconstruction Loss', fontsize=12, fontweight='bold')
ax3.set_title('Training Quality', fontsize=14, fontweight='bold')
ax3.set_xticks(range(len(methods)))
ax3.set_xticklabels([m.split()[0] for m in methods], rotation=15, ha='right')
ax3.grid(axis='y', alpha=0.3)

for bar, loss in zip(bars, recon_losses.values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f'{loss:.4f}', ha='center', va='bottom', fontsize=10)

# 4. 综合对比（雷达图）
ax4 = fig.add_subplot(224, projection='polar')

# 为K=4绘制雷达图
k4_results = comparison_df[comparison_df['K'] == 4]

metrics = ['Silhouette', 'CV_inv', 'CH_norm']
k4_results['CV_inv'] = 1 / k4_results['CV']  # CV倒数（越大越好）
k4_results['CH_norm'] = k4_results['CH_Score'] / k4_results['CH_Score'].max()

angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]

for i, (idx, row) in enumerate(k4_results.iterrows()):
    values = [
        row['Silhouette'],
        row['CV_inv'] / k4_results['CV_inv'].max(),
        row['CH_norm']
    ]
    values += values[:1]
    
    ax4.plot(angles, values, 'o-', linewidth=2, 
            label=row['Method'].split()[0], color=colors[i])
    ax4.fill(angles, values, alpha=0.15, color=colors[i])

ax4.set_xticks(angles[:-1])
ax4.set_xticklabels(['Silhouette', 'Balance\n(1/CV)', 'CH Score'], fontsize=10)
ax4.set_ylim(0, 1)
ax4.set_title('Overall Performance (K=4)', fontsize=14, fontweight='bold', pad=20)
ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
ax4.grid(True, linestyle='--', alpha=0.5)

plt.suptitle('Fusion Method Comparison', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('./results/reloaded_full/fusion_method_comparison.png', dpi=300, bbox_inches='tight')
print(f"✅ Visualization saved")

# 保存完整结果
features_df.to_csv('./results/reloaded_full/all_methods_results.csv', index=False)

print("\n" + "="*70)
print("✅ Comparison Complete!")
print("="*70)
print(f"\n📁 Results saved to: ./results/reloaded_full/")
print(f"   - fusion_method_comparison.csv (table)")
print(f"   - fusion_method_comparison.png (visualization)")
print(f"   - all_methods_results.csv (full data with all clusterings)")