import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # ⭐ 重要：添加这个导入
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import os

class DEC(nn.Module):
    """深度嵌入聚类"""
    
    def __init__(self, input_dim, n_clusters, hidden_dims=[128, 64, 32]):
        super().__init__()
        
        # 编码器
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = h_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # 聚类中心
        self.cluster_centers = nn.Parameter(torch.randn(n_clusters, hidden_dims[-1]))
        
        # 解码器（用于预训练）
        decoder_layers = []
        for h_dim in reversed(hidden_dims[:-1]):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU()
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        z = self.encoder(x)
        q = self._soft_assignment(z)
        recon = self.decoder(z)
        return z, q, recon
    
    def _soft_assignment(self, z):
        """计算软分配（Student-t分布）"""
        # 计算到每个聚类中心的距离
        dist = torch.sum((z.unsqueeze(1) - self.cluster_centers) ** 2, dim=2)
        
        # Student-t分布
        q = 1.0 / (1.0 + dist)
        q = q ** 2
        q = q / q.sum(dim=1, keepdim=True)
        
        return q


class DECDataset(Dataset):
    def __init__(self, data):
        self.data = torch.FloatTensor(data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class DECTrainer:
    """DEC训练器"""
    
    def __init__(self, input_dim, n_clusters=5, hidden_dims=[128, 64, 32]):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  使用设备: {self.device}")
        
        self.model = DEC(input_dim, n_clusters, hidden_dims).to(self.device)
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        
        self.history = {'pretrain_loss': [], 'cluster_loss': []}
    
    def target_distribution(self, q):
        """计算目标分布"""
        p = q ** 2 / q.sum(dim=0)
        p = p / p.sum(dim=1, keepdim=True)
        return p
    
    def prepare_data(self, features_df, drop_cols=['event_id', 'vehicle_id']):
        """准备数据"""
        X = features_df.drop(drop_cols, axis=1, errors='ignore')
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled
    
    def pretrain(self, X_scaled, epochs=50, batch_size=32, lr=0.001):
        """预训练自编码器"""
        print("\n🔧 步骤1: 预训练自编码器...")
        
        dataset = DECDataset(X_scaled)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            
            for batch in loader:
                batch = batch.to(self.device)
                
                z, q, recon = self.model(batch)
                loss = criterion(recon, batch)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            epoch_loss /= len(loader)
            self.history['pretrain_loss'].append(epoch_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f}")
        
        print("✅ 预训练完成！")
    
    def initialize_clusters(self, X_scaled):
        """初始化聚类中心"""
        print("\n🔧 步骤2: 初始化聚类中心...")
        
        self.model.eval()
        
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            z, _, _ = self.model(X_tensor)
            z_np = z.cpu().numpy()
        
        # 使用K-Means初始化聚类中心
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        kmeans.fit(z_np)
        
        # 更新模型的聚类中心
        self.model.cluster_centers.data = torch.FloatTensor(kmeans.cluster_centers_).to(self.device)
        
        print("✅ 聚类中心初始化完成！")
    
    def train(self, features_df, pretrain_epochs=50, train_epochs=100, batch_size=32, lr=0.001):
        """训练DEC"""
        print("\n" + "="*60)
        print("🚀 开始训练DEC")
        print("="*60)
        
        # 准备数据
        X_scaled = self.prepare_data(features_df)
        
        # 预训练
        self.pretrain(X_scaled, epochs=pretrain_epochs, batch_size=batch_size, lr=lr)
        
        # 初始化聚类中心
        self.initialize_clusters(X_scaled)
        
        # 聚类训练
        print("\n🔧 步骤3: 聚类训练...")
        
        dataset = DECDataset(X_scaled)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr * 0.1)
        
        for epoch in range(train_epochs):
            self.model.eval()
            
            # 计算目标分布
            q_list = []
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(self.device)
                    _, q, _ = self.model(batch)
                    q_list.append(q)
            
            q_all = torch.cat(q_list, dim=0)
            p_all = self.target_distribution(q_all)
            
            # 更新模型
            self.model.train()
            epoch_loss = 0
            
            idx = 0
            for batch in loader:
                batch_size_actual = batch.size(0)
                batch = batch.to(self.device)
                
                _, q, _ = self.model(batch)
                p = p_all[idx:idx+batch_size_actual]
                
                # KL散度损失 - ⭐ 这里使用了 F.kl_div
                loss = F.kl_div(q.log(), p, reduction='batchmean')
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                idx += batch_size_actual
            
            epoch_loss /= len(loader)
            self.history['cluster_loss'].append(epoch_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch [{epoch+1}/{train_epochs}] - Loss: {epoch_loss:.4f}")
        
        print("\n✅ DEC训练完成！")
    
    def predict(self, features_df):
        """预测聚类标签"""
        self.model.eval()
        
        X_scaled = self.prepare_data(features_df)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            z, q, _ = self.model(X_tensor)
            labels = torch.argmax(q, dim=1).cpu().numpy()
            z_np = z.cpu().numpy()
        
        # 评估
        metrics = {
            'silhouette': silhouette_score(z_np, labels),
            'ch_score': calinski_harabasz_score(z_np, labels),
            'db_score': davies_bouldin_score(z_np, labels)
        }
        
        print(f"\n📊 聚类效果:")
        print(f"   Silhouette: {metrics['silhouette']:.3f}")
        print(f"   CH Score: {metrics['ch_score']:.2f}")
        print(f"   DB Score: {metrics['db_score']:.3f}")
        
        return labels, z_np, metrics
    
    def visualize_training(self, save_dir='./results/dec'):
        """可视化训练过程"""
        os.makedirs(save_dir, exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # 预训练损失
        axes[0].plot(self.history['pretrain_loss'], linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Autoencoder Pretraining Loss')
        axes[0].grid(True, alpha=0.3)
        
        # 聚类训练损失
        axes[1].plot(self.history['cluster_loss'], linewidth=2, color='orange')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('KL Divergence')
        axes[1].set_title('DEC Clustering Loss')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 训练曲线已保存至: {save_dir}/training_history.png")
    
    def save_model(self, save_dir='./results/dec'):
        """保存模型"""
        os.makedirs(save_dir, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'history': self.history
        }, f'{save_dir}/dec_model.pth')
        
        print(f"💾 模型已保存至: {save_dir}/dec_model.pth")


# 使用示例
if __name__ == "__main__":
    # 读取特征
    features_df = pd.read_csv('./results/features/combined_features.csv')
    
    print(f"📊 特征形状: {features_df.shape}")
    
    # 创建训练器
    input_dim = len(features_df.columns) - 2
    trainer = DECTrainer(input_dim, n_clusters=5, hidden_dims=[128, 64, 32])
    
    # 训练
    trainer.train(features_df, pretrain_epochs=50, train_epochs=100, batch_size=32, lr=0.001)
    
    # 预测
    labels, features, metrics = trainer.predict(features_df)
    
    # 保存结果
    results_df = features_df[['event_id', 'vehicle_id']].copy()
    results_df['cluster'] = labels
    results_df.to_csv('./results/dec/clustered_results.csv', index=False)
    
    # 可视化
    trainer.visualize_training()
    
    # 保存模型
    trainer.save_model()
    
    print("\n✅ DEC模型训练完成！")