import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import os

class ContrastiveEncoder(nn.Module):
    """对比学习编码器"""
    
    def __init__(self, input_dim, hidden_dim=128, output_dim=64):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # 投影头
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return F.normalize(z, dim=1), h


class ContrastiveDataset(Dataset):
    """对比学习数据集"""
    
    def __init__(self, data, noise_level=0.1):
        self.data = torch.FloatTensor(data)
        self.noise_level = noise_level
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        
        # 数据增强：添加高斯噪声
        x1 = x + torch.randn_like(x) * self.noise_level
        x2 = x + torch.randn_like(x) * self.noise_level
        
        return x1, x2


class ContrastiveTrainer:
    """对比学习训练器"""
    
    def __init__(self, input_dim, hidden_dim=128, output_dim=64, n_clusters=5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  使用设备: {self.device}")
        
        self.model = ContrastiveEncoder(input_dim, hidden_dim, output_dim).to(self.device)
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        
        self.history = {'train_loss': [], 'val_loss': []}
    
    def nt_xent_loss(self, z1, z2, temperature=0.5):
        """NT-Xent损失函数（SimCLR）"""
        batch_size = z1.shape[0]
        
        # 合并两个视图
        z = torch.cat([z1, z2], dim=0)  # (2*batch, dim)
        
        # 计算相似度矩阵
        sim_matrix = torch.mm(z, z.T) / temperature  # (2*batch, 2*batch)
        
        # 创建mask：排除自身和对角线
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=self.device)
        sim_matrix.masked_fill_(mask, -9e15)
        
        # 正样本对的索引
        pos_indices = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=self.device),
            torch.arange(batch_size, device=self.device)
        ])
        
        # 计算损失
        exp_sim = torch.exp(sim_matrix)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        loss = -log_prob[range(2 * batch_size), pos_indices].mean()
        
        return loss
    
    def prepare_data(self, features_df, drop_cols=['event_id', 'vehicle_id']):
        """准备数据"""
        X = features_df.drop(drop_cols, axis=1, errors='ignore')
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled
    
    def train(self, features_df, epochs=100, batch_size=32, lr=0.001, val_split=0.2, temperature=0.5):
        """训练模型"""
        print("\n" + "="*60)
        print("🚀 开始训练对比学习模型")
        print("="*60)
        
        # 准备数据
        X_scaled = self.prepare_data(features_df)
        
        # 划分训练集和验证集
        n_val = int(len(X_scaled) * val_split)
        indices = np.random.permutation(len(X_scaled))
        
        train_indices = indices[n_val:]
        val_indices = indices[:n_val]
        
        train_dataset = ContrastiveDataset(X_scaled[train_indices], noise_level=0.1)
        val_dataset = ContrastiveDataset(X_scaled[val_indices], noise_level=0.1)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 优化器
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # 训练循环
        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0
        
        for epoch in range(epochs):
            # 训练
            self.model.train()
            train_loss = 0
            
            for x1, x2 in train_loader:
                x1, x2 = x1.to(self.device), x2.to(self.device)
                
                # 前向传播
                z1, _ = self.model(x1)
                z2, _ = self.model(x2)
                
                # 计算损失
                loss = self.nt_xent_loss(z1, z2, temperature)
                
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
                for x1, x2 in val_loader:
                    x1, x2 = x1.to(self.device), x2.to(self.device)
                    z1, _ = self.model(x1)
                    z2, _ = self.model(x2)
                    loss = self.nt_xent_loss(z1, z2, temperature)
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
    
    def extract_features(self, features_df):
        """提取特征"""
        self.model.eval()
        
        X_scaled = self.prepare_data(features_df)
        
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            _, h = self.model(X_tensor)
            features = h.cpu().numpy()
        
        return features
    
    def cluster(self, features):
        """聚类"""
        print("\n🔵 在特征空间进行聚类...")
        
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)
        
        # 评估
        metrics = {
            'silhouette': silhouette_score(features, labels),
            'ch_score': calinski_harabasz_score(features, labels),
            'db_score': davies_bouldin_score(features, labels)
        }
        
        print(f"   Silhouette: {metrics['silhouette']:.3f}")
        print(f"   CH Score: {metrics['ch_score']:.2f}")
        print(f"   DB Score: {metrics['db_score']:.3f}")
        
        return labels, metrics
    
    def visualize_training(self, save_dir='./results/contrastive'):
        """可视化训练过程"""
        os.makedirs(save_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['train_loss'], label='Train Loss', linewidth=2)
        plt.plot(self.history['val_loss'], label='Val Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Contrastive Learning Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{save_dir}/training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 训练曲线已保存至: {save_dir}/training_history.png")
    
    def save_model(self, save_dir='./results/contrastive'):
        """保存模型"""
        os.makedirs(save_dir, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'history': self.history
        }, f'{save_dir}/contrastive_model.pth')
        
        print(f"💾 模型已保存至: {save_dir}/contrastive_model.pth")


# 使用示例
if __name__ == "__main__":
    # 读取特征
    features_df = pd.read_csv('./results/features/combined_features.csv')
    
    print(f"📊 特征形状: {features_df.shape}")
    
    # 创建训练器
    input_dim = len(features_df.columns) - 2
    trainer = ContrastiveTrainer(input_dim, hidden_dim=128, output_dim=64, n_clusters=5)
    
    # 训练
    trainer.train(features_df, epochs=100, batch_size=32, lr=0.001, temperature=0.5)
    
    # 提取特征
    features = trainer.extract_features(features_df)
    print(f"\n📊 特征形状: {features.shape}")
    
    # 聚类
    labels, metrics = trainer.cluster(features)
    
    # 保存结果
    results_df = features_df[['event_id', 'vehicle_id']].copy()
    results_df['cluster'] = labels
    results_df.to_csv('./results/contrastive/clustered_results.csv', index=False)
    
    # 可视化
    trainer.visualize_training()
    
    # 保存模型
    trainer.save_model()
    
    print("\n✅ 对比学习模型训练完成！")