"""
对比不同编码器模型的聚类效果
基于当前的统计特征（141k × 14D驾驶，141k × 13D能量）

模型对比：
1. MLP（当前使用）
2. CNN-1D（捕捉特征间的局部关系）
3. Transformer-Mini（自注意力机制）
4. ResNet-1D（残差连接）
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
print("🔬 Encoder Model Comparison Experiment")
print("="*70)

# ==================== 配置 ====================
CONFIG = {
    'latent_dim': 8,
    'batch_size': 512,
    'epochs': 30,
    'lr': 0.0001,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'k_value': 4,  # 固定K=4进行对比
    'output_dir': './results/encoder_comparison'
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)

# ==================== 交叉注意力（共用）====================
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

# ==================== 编码器1：MLP（当前使用）====================
class MLPEncoder(nn.Module):
    """多层感知机编码器"""
    def __init__(self, input_dim, latent_dim=8):
        super().__init__()
        self.name = "MLP"
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
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
    
    def forward(self, x):
        return self.encoder(x)

# ==================== 编码器2：CNN-1D ====================
class CNN1DEncoder(nn.Module):
    """1D卷积编码器 - 捕捉特征间的局部关系"""
    def __init__(self, input_dim, latent_dim=8):
        super().__init__()
        self.name = "CNN-1D"
        
        # 把特征维度当作"序列长度"
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, latent_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        # x: (batch, input_dim)
        x = x.unsqueeze(1)  # (batch, 1, input_dim)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = self.global_pool(x)  # (batch, 64, 1)
        x = x.squeeze(-1)  # (batch, 64)
        
        x = self.fc(x)
        return x

# ==================== 编码器3：Transformer-Mini ====================
class TransformerEncoder(nn.Module):
    """轻量级Transformer编码器"""
    def __init__(self, input_dim, latent_dim=8):
        super().__init__()
        self.name = "Transformer"
        
        self.embedding_dim = 32
        
        # 线性投影
        self.input_proj = nn.Linear(input_dim, self.embedding_dim)
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1, input_dim, self.embedding_dim))
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=4,
            dim_feedforward=64,
            dropout=0.2,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(self.embedding_dim, latent_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        # x: (batch, input_dim)
        batch_size = x.size(0)
        
        # 把每个特征看作一个token
        x = x.unsqueeze(2)  # (batch, input_dim, 1)
        x = self.input_proj(x.transpose(1, 2))  # (batch, 1, embedding_dim)
        
        # 复制到input_dim个token
        x = x.repeat(1, x.size(1), 1)  # (batch, input_dim, embedding_dim)
        
        # 添加位置编码
        x = x + self.pos_encoding[:, :x.size(1), :]
        
        # Transformer
        x = self.transformer(x)  # (batch, input_dim, embedding_dim)
        
        # 平均池化
        x = x.mean(dim=1)  # (batch, embedding_dim)
        
        x = self.fc(x)
        return x

# ==================== 编码器4：ResNet-1D ====================
class ResBlock1D(nn.Module):
    """残差块"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class ResNet1DEncoder(nn.Module):
    """ResNet-1D编码器"""
    def __init__(self, input_dim, latent_dim=8):
        super().__init__()
        self.name = "ResNet-1D"
        
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        
        self.res_block1 = ResBlock1D(32)
        self.res_block2 = ResBlock1D(32)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(32, latent_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        # x: (batch, input_dim)
        x = x.unsqueeze(1)  # (batch, 1, input_dim)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res_block1(x)
        x = self.res_block2(x)
        
        x = self.global_pool(x).squeeze(-1)  # (batch, 32)
        x = self.fc(x)
        return x

# ==================== 双通道模型（可切换编码器）====================
class DualChannelAE(nn.Module):
    """双通道自编码器（可选择编码器类型）"""
    def __init__(self, driving_dim, energy_dim, latent_dim=8, encoder_type='MLP'):
        super().__init__()
        
        self.encoder_type = encoder_type
        
        # 选择编码器
        encoder_classes = {
            'MLP': MLPEncoder,
            'CNN': CNN1DEncoder,
            'Transformer': TransformerEncoder,
            'ResNet': ResNet1DEncoder
        }
        
        EncoderClass = encoder_classes[encoder_type]
        
        self.driving_encoder = EncoderClass(driving_dim, latent_dim)
        self.energy_encoder = EncoderClass(energy_dim, latent_dim)
        
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
        
        # 解码器（简单MLP）
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
        
        combined = torch.cat([
            driving_latent, driving_attended,
            energy_latent, energy_attended
        ], dim=1)
        
        fused_latent = self.fusion(combined)
        
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

# ==================== 训练和评估函数 ====================
def train_model(model, X_driving, X_energy, epochs, batch_size, lr, device):
    """训练模型"""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    best_loss = float('inf')
    patience = 0
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
            
            optimizer.zero_grad()
            
            d_recon, e_recon, _ = model(d_batch, e_batch)
            
            loss = F.mse_loss(d_recon, d_batch) + F.mse_loss(e_recon, e_batch)
            
            if torch.isnan(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            epoch_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = np.mean(epoch_losses)
        history.append(avg_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience = 0
        else:
            patience += 1
            if patience >= 10:
                break
    
    return best_loss, history

def extract_features(model, X_driving, X_energy, batch_size, device):
    """提取潜在特征"""
    model.eval()
    latent_features = []
    
    with torch.no_grad():
        for i in range(0, len(X_driving), batch_size):
            d_batch = torch.FloatTensor(X_driving[i:i+batch_size]).to(device)
            e_batch = torch.FloatTensor(X_energy[i:i+batch_size]).to(device)
            latent = model.encode(d_batch, e_batch)
            latent_features.append(latent.cpu().numpy())
    
    return np.vstack(latent_features)

def evaluate_clustering(latent_features, k, features_df):
    """评估聚类"""
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = kmeans.fit_predict(latent_features)
    
    sil = silhouette_score(latent_features, labels)
    ch = calinski_harabasz_score(latent_features, labels)
    db = davies_bouldin_score(latent_features, labels)
    
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    percentages = cluster_counts.values / len(labels) * 100
    cv = percentages.std() / percentages.mean()
    
    # 计算簇内速度方差（物理意义）
    features_df_temp = features_df.copy()
    features_df_temp['cluster'] = labels
    
    intra_speed_variance = 0
    for cid in range(k):
        cluster_data = features_df_temp[features_df_temp['cluster'] == cid]
        intra_speed_variance += cluster_data['speed_mean'].var()
    intra_speed_variance /= k
    
    return {
        'labels': labels,
        'silhouette': sil,
        'ch_score': ch,
        'db_score': db,
        'cv': cv,
        'distribution': cluster_counts.to_dict(),
        'intra_speed_var': intra_speed_variance
    }

# ==================== 主实验流程 ====================
print(f"\n📂 Loading data...")
features_df = pd.read_csv('./results/reloaded_full/trip_features_full.csv')

# 清洗
features_df = features_df[
    (features_df['duration_minutes'] > 2) & (features_df['duration_minutes'] < 300) &
    (features_df['speed_mean'] >= 0) & (features_df['speed_mean'] < 150) &
    (features_df['distance_total'] >= 0) & (features_df['distance_total'] < 500)
]

print(f"✅ Loaded {len(features_df):,} trips")

# 准备数据
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
print("🔬 Running Encoder Comparison")
print("="*70)

encoder_types = ['MLP', 'CNN', 'Transformer', 'ResNet']
results = []

for encoder_type in encoder_types:
    print(f"\n{'='*60}")
    print(f"Testing: {encoder_type} Encoder")
    print("="*60)
    
    # 构建模型
    model = DualChannelAE(
        driving_dim=X_driving.shape[1],
        energy_dim=X_energy.shape[1],
        latent_dim=CONFIG['latent_dim'],
        encoder_type=encoder_type
    )
    
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练
    print(f"\n🚀 Training...")
    best_loss, history = train_model(
        model, X_driving, X_energy,
        epochs=CONFIG['epochs'],
        batch_size=CONFIG['batch_size'],
        lr=CONFIG['lr'],
        device=CONFIG['device']
    )
    
    print(f"✅ Best loss: {best_loss:.4f}")
    
    # 提取特征
    print(f"🔍 Extracting features...")
    latent_features = extract_features(
        model, X_driving, X_energy,
        CONFIG['batch_size'], CONFIG['device']
    )
    
    # 聚类
    print(f"🎯 Clustering (K={CONFIG['k_value']})...")
    clustering_result = evaluate_clustering(
        latent_features, CONFIG['k_value'], features_df
    )
    
    # 保存
    torch.save(model.state_dict(), f"{CONFIG['output_dir']}/{encoder_type}_model.pth")
    np.save(f"{CONFIG['output_dir']}/{encoder_type}_latent.npy", latent_features)
    
    # 记录结果
    result = {
        'Encoder': encoder_type,
        'Parameters': sum(p.numel() for p in model.parameters()),
        'Best_Loss': best_loss,
        'Silhouette': clustering_result['silhouette'],
        'CH_Score': clustering_result['ch_score'],
        'DB_Score': clustering_result['db_score'],
        'CV': clustering_result['cv'],
        'Intra_Speed_Var': clustering_result['intra_speed_var'],
        'Distribution': clustering_result['distribution']
    }
    
    results.append(result)
    
    print(f"\n📊 Results:")
    print(f"   Silhouette: {clustering_result['silhouette']:.3f}")
    print(f"   CV: {clustering_result['cv']:.3f}")
    print(f"   Distribution: {clustering_result['distribution']}")

# ==================== 生成对比报告 ====================
print(f"\n{'='*70}")
print("📊 Comparison Report")
print("="*70)

comparison_df = pd.DataFrame(results)
comparison_df = comparison_df.round(3)

print(f"\n{comparison_df.to_string(index=False)}")

# 保存
comparison_df.to_csv(f"{CONFIG['output_dir']}/encoder_comparison.csv", index=False)

# 可视化
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. Silhouette对比
ax = axes[0, 0]
bars = ax.bar(comparison_df['Encoder'], comparison_df['Silhouette'], 
             color=sns.color_palette('Set2', len(encoder_types)), alpha=0.8)
ax.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
ax.set_title('Clustering Quality', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, comparison_df['Silhouette']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
           f'{val:.3f}', ha='center', va='bottom', fontsize=10)

# 2. CV对比
ax = axes[0, 1]
bars = ax.bar(comparison_df['Encoder'], comparison_df['CV'],
             color=sns.color_palette('Set2', len(encoder_types)), alpha=0.8)
ax.set_ylabel('Coefficient of Variation', fontsize=12, fontweight='bold')
ax.set_title('Distribution Balance (lower is better)', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, comparison_df['CV']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
           f'{val:.3f}', ha='center', va='bottom', fontsize=10)

# 3. 重构损失
ax = axes[0, 2]
bars = ax.bar(comparison_df['Encoder'], comparison_df['Best_Loss'],
             color=sns.color_palette('Set2', len(encoder_types)), alpha=0.8)
ax.set_ylabel('Reconstruction Loss', fontsize=12, fontweight='bold')
ax.set_title('Training Quality', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, comparison_df['Best_Loss']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
           f'{val:.3f}', ha='center', va='bottom', fontsize=10)

# 4. 参数量
ax = axes[1, 0]
bars = ax.bar(comparison_df['Encoder'], comparison_df['Parameters']/1000,
             color=sns.color_palette('Set2', len(encoder_types)), alpha=0.8)
ax.set_ylabel('Parameters (K)', fontsize=12, fontweight='bold')
ax.set_title('Model Complexity', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, comparison_df['Parameters']/1000):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
           f'{val:.1f}K', ha='center', va='bottom', fontsize=10)

# 5. 簇内速度方差
ax = axes[1, 1]
bars = ax.bar(comparison_df['Encoder'], comparison_df['Intra_Speed_Var'],
             color=sns.color_palette('Set2', len(encoder_types)), alpha=0.8)
ax.set_ylabel('Intra-cluster Speed Variance', fontsize=12, fontweight='bold')
ax.set_title('Cluster Tightness (lower is better)', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, comparison_df['Intra_Speed_Var']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
           f'{val:.1f}', ha='center', va='bottom', fontsize=10)

# 6. 综合得分雷达图
ax = axes[1, 2]
ax.remove()
ax = fig.add_subplot(2, 3, 6, projection='polar')

# 归一化指标
metrics = ['Silhouette', 'Balance\n(1/CV)', 'Quality\n(1/Loss)', 'Efficiency\n(1/Params)']
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]

for i, encoder in enumerate(encoder_types):
    row = comparison_df[comparison_df['Encoder'] == encoder].iloc[0]
    values = [
        row['Silhouette'],
        1 / row['CV'],
        1 / row['Best_Loss'],
        1 / (row['Parameters'] / 10000)
    ]
    # 归一化
    values = [v / comparison_df[[f.split('\n')[0] for f in metrics if '\n' not in f][j] if j < 1 else 
                                1/comparison_df['CV'] if j == 1 else
                                1/comparison_df['Best_Loss'] if j == 2 else
                                1/(comparison_df['Parameters']/10000)].max() 
             for j, v in enumerate(values)]
    
    values += values[:1]
    
    ax.plot(angles, values, 'o-', linewidth=2, label=encoder,
           color=sns.color_palette('Set2', len(encoder_types))[i])
    ax.fill(angles, values, alpha=0.15,
           color=sns.color_palette('Set2', len(encoder_types))[i])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics, fontsize=10)
ax.set_ylim(0, 1)
ax.set_title('Overall Performance', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
ax.grid(True)

plt.suptitle(f'Encoder Model Comparison (K={CONFIG["k_value"]})', 
            fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{CONFIG['output_dir']}/encoder_comparison.png", dpi=300, bbox_inches='tight')

print(f"\n✅ Visualization saved")

# ==================== 推荐 ====================
print(f"\n{'='*70}")
print("🏆 Recommendation")
print("="*70)

best_sil = comparison_df.loc[comparison_df['Silhouette'].idxmax()]
best_cv = comparison_df.loc[comparison_df['CV'].idxmin()]
best_loss = comparison_df.loc[comparison_df['Best_Loss'].idxmin()]

print(f"\n🥇 Best clustering quality: {best_sil['Encoder']} (Silhouette={best_sil['Silhouette']:.3f})")
print(f"🥈 Best distribution balance: {best_cv['Encoder']} (CV={best_cv['CV']:.3f})")
print(f"🥉 Best reconstruction: {best_loss['Encoder']} (Loss={best_loss['Best_Loss']:.3f})")

# 综合得分
comparison_df['Score'] = (
    comparison_df['Silhouette'] / comparison_df['Silhouette'].max() * 0.4 +
    (1/comparison_df['CV']) / (1/comparison_df['CV']).max() * 0.4 +
    (1/comparison_df['Best_Loss']) / (1/comparison_df['Best_Loss']).max() * 0.2
)

best_overall = comparison_df.loc[comparison_df['Score'].idxmax()]
print(f"\n🎯 Best overall: {best_overall['Encoder']} (Score={best_overall['Score']:.3f})")

print("\n" + "="*70)
print("✅ Encoder Comparison Complete!")
print("="*70)