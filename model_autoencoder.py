import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import os
import pickle

class DualChannelAutoencoder(nn.Module):
    """双通道自编码器"""
    
    def __init__(self, energy_dim, driving_dim, latent_dim=16):
        super().__init__()
        
        # 电量特征编码器
        self.energy_encoder = nn.Sequential(
            nn.Linear(energy_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        
        # 驾驶行为编码器
        self.driving_encoder = nn.Sequential(
            nn.Linear(driving_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU()
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, energy_dim + driving_dim)
        )
    
    def forward(self, energy_features, driving_features):
        # 分别编码
        energy_latent = self.energy_encoder(energy_features)
        driving_latent = self.driving_encoder(driving_features)
        
        # 融合
        fused = torch.cat([energy_latent, driving_latent], dim=1)
        latent = self.fusion(fused)
        
        # 解码
        reconstructed = self.decoder(latent)
        
        return reconstructed, latent, energy_latent, driving_latent


class EventDataset(Dataset):
    """事件数据集"""
    
    def __init__(self, energy_features, driving_features):
        self.energy_features = torch.FloatTensor(energy_features)
        self.driving_features = torch.FloatTensor(driving_features)
    
    def __len__(self):
        return len(self.energy_features)
    
    def __getitem__(self, idx):
        return self.energy_features[idx], self.driving_features[idx]


class AutoencoderTrainer:
    """自编码器训练器"""
    
    def __init__(self, energy_dim, driving_dim, latent_dim=16, n_clusters=5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  使用设备: {self.device}")
        
        self.model = DualChannelAutoencoder(energy_dim, driving_dim, latent_dim).to(self.device)
        self.latent_dim = latent_dim
        self.n_clusters = n_clusters
        
        self.energy_scaler = StandardScaler()
        self.driving_scaler = StandardScaler()
        
        self.history = {'train_loss': [], 'val_loss': []}
    
    def prepare_data(self, energy_df, driving_df, drop_cols=['event_id', 'vehicle_id']):
        """准备数据"""
        # 分离能量和驾驶特征
        energy_features = energy_df.drop(drop_cols, axis=1, errors='ignore')
        driving_features = driving_df.drop(drop_cols, axis=1, errors='ignore')
        
        # 处理异常值
        energy_features = energy_features.replace([np.inf, -np.inf], np.nan).fillna(0)
        driving_features = driving_features.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # 标准化
        energy_scaled = self.energy_scaler.fit_transform(energy_features)
        driving_scaled = self.driving_scaler.fit_transform(driving_features)
        
        return energy_scaled, driving_scaled
    
    def train(self, energy_df, driving_df, epochs=100, batch_size=32, lr=0.001, val_split=0.2):
        """训练模型"""
        print("\n" + "="*60)
        print("🚀 开始训练双通道自编码器")
        print("="*60)
        
        # 准备数据
        energy_scaled, driving_scaled = self.prepare_data(energy_df, driving_df)
        
        # 划分训练集和验证集
        n_val = int(len(energy_scaled) * val_split)
        indices = np.random.permutation(len(energy_scaled))
        
        train_indices = indices[n_val:]
        val_indices = indices[:n_val]
        
        train_dataset = EventDataset(energy_scaled[train_indices], driving_scaled[train_indices])
        val_dataset = EventDataset(energy_scaled[val_indices], driving_scaled[val_indices])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 优化器
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # 训练循环
        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0
        
        for epoch in range(epochs):
            # 训练
            self.model.train()
            train_loss = 0
            
            for energy_batch, driving_batch in train_loader:
                energy_batch = energy_batch.to(self.device)
                driving_batch = driving_batch.to(self.device)
                
                # 前向传播
                combined_batch = torch.cat([energy_batch, driving_batch], dim=1)
                reconstructed, latent, _, _ = self.model(energy_batch, driving_batch)
                
                # 计算损失
                loss = criterion(reconstructed, combined_batch)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # 验证
            self.model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for energy_batch, driving_batch in val_loader:
                    energy_batch = energy_batch.to(self.device)
                    driving_batch = driving_batch.to(self.device)
                    
                    combined_batch = torch.cat([energy_batch, driving_batch], dim=1)
                    reconstructed, latent, _, _ = self.model(energy_batch, driving_batch)
                    
                    loss = criterion(reconstructed, combined_batch)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            # 打印进度
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n⏹️  Early stopping at epoch {epoch+1}")
                    break
        
        print(f"\n✅ 训练完成！最佳验证损失: {best_val_loss:.4f}")
    
    def extract_features(self, energy_df, driving_df):
        """提取潜在特征"""
        self.model.eval()
        
        energy_scaled, driving_scaled = self.prepare_data(energy_df, driving_df)
        
        dataset = EventDataset(energy_scaled, driving_scaled)
        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        
        latent_features = []
        
        with torch.no_grad():
            for energy_batch, driving_batch in loader:
                energy_batch = energy_batch.to(self.device)
                driving_batch = driving_batch.to(self.device)
                
                _, latent, _, _ = self.model(energy_batch, driving_batch)
                latent_features.append(latent.cpu().numpy())
        
        latent_features = np.vstack(latent_features)
        
        return latent_features
    
    def cluster(self, latent_features):
        """在潜在空间聚类"""
        print("\n🔵 在潜在空间进行聚类...")
        
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(latent_features)
        
        # 评估
        metrics = {
            'silhouette': silhouette_score(latent_features, labels),
            'ch_score': calinski_harabasz_score(latent_features, labels),
            'db_score': davies_bouldin_score(latent_features, labels)
        }
        
        print(f"   Silhouette: {metrics['silhouette']:.3f}")
        print(f"   CH Score: {metrics['ch_score']:.2f}")
        print(f"   DB Score: {metrics['db_score']:.3f}")
        
        return labels, metrics
    
    def visualize_training(self, save_dir='./results/autoencoder'):
        """可视化训练过程"""
        os.makedirs(save_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Autoencoder Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{save_dir}/training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 训练曲线已保存至: {save_dir}/training_history.png")
    
    def save_model(self, save_dir='./results/autoencoder'):
        """保存模型"""
        os.makedirs(save_dir, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'energy_scaler': self.energy_scaler,
            'driving_scaler': self.driving_scaler,
            'history': self.history
        }, f'{save_dir}/autoencoder_model.pth')
        
        print(f"💾 模型已保存至: {save_dir}/autoencoder_model.pth")


# 使用示例
if __name__ == "__main__":
    # 读取特征
    energy_df = pd.read_csv('./results/features/energy_features.csv')
    driving_df = pd.read_csv('./results/features/driving_features.csv')
    
    print(f"📊 能量特征: {energy_df.shape}")
    print(f"📊 驾驶特征: {driving_df.shape}")
    
    # 创建训练器
    energy_dim = len(energy_df.columns) - 2  # 减去ID列
    driving_dim = len(driving_df.columns) - 2
    
    trainer = AutoencoderTrainer(energy_dim, driving_dim, latent_dim=16, n_clusters=5)
    
    # 训练
    trainer.train(energy_df, driving_df, epochs=100, batch_size=32, lr=0.001)
    
    # 提取特征
    latent_features = trainer.extract_features(energy_df, driving_df)
    print(f"\n📊 潜在特征形状: {latent_features.shape}")
    
    # 聚类
    labels, metrics = trainer.cluster(latent_features)
    
    # 保存结果
    results_df = energy_df[['event_id', 'vehicle_id']].copy()
    results_df['cluster'] = labels
    results_df.to_csv('./results/autoencoder/clustered_results.csv', index=False)
    
    # 可视化
    trainer.visualize_training()
    
    # 保存模型
    trainer.save_model()
    
    print("\n✅ 自编码器模型训练完成！")