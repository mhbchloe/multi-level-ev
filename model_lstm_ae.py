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
from tqdm import tqdm

class LSTMAutoencoder(nn.Module):
    """LSTM自编码器 - 捕捉时序依赖"""
    
    def __init__(self, input_dim, hidden_dim=64, latent_dim=16, num_layers=2):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        
        # 编码器
        self.encoder_lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)
        
        # 解码器
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            hidden_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        self.output_fc = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        batch_size, seq_len, _ = x.size()
        
        # 编码
        _, (hidden, cell) = self.encoder_lstm(x)
        # hidden shape: (num_layers, batch, hidden_dim)
        
        # 取最后一层的hidden state
        latent = self.encoder_fc(hidden[-1])
        # latent shape: (batch, latent_dim)
        
        # 解码
        hidden_dec = self.decoder_fc(latent)
        # hidden_dec shape: (batch, hidden_dim)
        
        # 重复hidden_dec以匹配序列长度
        hidden_dec = hidden_dec.unsqueeze(1).repeat(1, seq_len, 1)
        # hidden_dec shape: (batch, seq_len, hidden_dim)
        
        decoded, _ = self.decoder_lstm(hidden_dec)
        # decoded shape: (batch, seq_len, hidden_dim)
        
        output = self.output_fc(decoded)
        # output shape: (batch, seq_len, input_dim)
        
        return output, latent


class SequenceDataset(Dataset):
    """时序数据集"""
    
    def __init__(self, sequences):
        self.sequences = torch.FloatTensor(sequences)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]


class LSTMAETrainer:
    """LSTM自编码器训练器"""
    
    def __init__(self, input_dim, hidden_dim=64, latent_dim=16, num_layers=2, n_clusters=5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  使用设备: {self.device}")
        
        self.model = LSTMAutoencoder(input_dim, hidden_dim, latent_dim, num_layers).to(self.device)
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        
        self.history = {'train_loss': [], 'val_loss': []}
        
        print(f"📐 模型参数: input_dim={input_dim}, hidden_dim={hidden_dim}, latent_dim={latent_dim}")
    
    def prepare_sequences(self, events, max_seq_len=100):
        """
        将事件转换为固定长度的序列
        
        参数:
            events: 事件列表
            max_seq_len: 最大序列长度
        """
        print("\n🔧 准备时序数据...")
        
        sequences = []
        valid_events = []
        
        for event in tqdm(events, desc="处理事件"):
            event_data = event['data']
            
            # 选择关键特征
            feature_cols = ['spd', 'acc', 'soc', 'v', 'i', 'power']
            available_cols = [col for col in feature_cols if col in event_data.columns]
            
            if len(available_cols) == 0:
                continue
            
            seq = event_data[available_cols].values
            
            # 处理异常值
            seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 填充或截断到固定长度
            if len(seq) < max_seq_len:
                # 填充
                padding = np.zeros((max_seq_len - len(seq), seq.shape[1]))
                seq = np.vstack([seq, padding])
            else:
                # 截断
                seq = seq[:max_seq_len]
            
            sequences.append(seq)
            valid_events.append(event)
        
        sequences = np.array(sequences)
        
        # 标准化（对每个特征维度）
        n_samples, seq_len, n_features = sequences.shape
        sequences_flat = sequences.reshape(-1, n_features)
        sequences_flat = self.scaler.fit_transform(sequences_flat)
        sequences = sequences_flat.reshape(n_samples, seq_len, n_features)
        
        print(f"✅ 序列数据形状: {sequences.shape}")
        
        return sequences, valid_events
    
    def train(self, sequences, epochs=50, batch_size=32, lr=0.001, val_split=0.2):
        """训练模型"""
        print("\n" + "="*60)
        print("🚀 开始训练LSTM自编码器")
        print("="*60)
        
        # 划分训练集和验证集
        n_val = int(len(sequences) * val_split)
        indices = np.random.permutation(len(sequences))
        
        train_indices = indices[n_val:]
        val_indices = indices[:n_val]
        
        train_dataset = SequenceDataset(sequences[train_indices])
        val_dataset = SequenceDataset(sequences[val_indices])
        
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
            
            for batch in train_loader:
                batch = batch.to(self.device)
                
                # 前向传播
                reconstructed, latent = self.model(batch)
                loss = criterion(reconstructed, batch)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # 验证
            self.model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
                    reconstructed, latent = self.model(batch)
                    loss = criterion(reconstructed, batch)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            # 打印进度
            if (epoch + 1) % 5 == 0:
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
    
    def extract_features(self, sequences):
        """提取潜在特征"""
        self.model.eval()
        
        dataset = SequenceDataset(sequences)
        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        
        latent_features = []
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                _, latent = self.model(batch)
                latent_features.append(latent.cpu().numpy())
        
        latent_features = np.vstack(latent_features)
        
        return latent_features
    
    def cluster(self, latent_features):
        """聚类"""
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
    
    def visualize_training(self, save_dir='./results/lstm_ae'):
        """可视化训练过程"""
        os.makedirs(save_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['train_loss'], label='Train Loss', linewidth=2)
        plt.plot(self.history['val_loss'], label='Val Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('LSTM-AE Training History', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{save_dir}/training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 训练曲线已保存至: {save_dir}/training_history.png")
    
    def save_model(self, save_dir='./results/lstm_ae'):
        """保存模型"""
        os.makedirs(save_dir, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'history': self.history
        }, f'{save_dir}/lstm_ae_model.pth')
        
        print(f"💾 模型已保存至: {save_dir}/lstm_ae_model.pth")


# 使用示例
if __name__ == "__main__":
    import pickle
    
    # 读取事件数据
    with open('./results/events/events.pkl', 'rb') as f:
        events = pickle.load(f)
    
    print(f"📊 事件数量: {len(events)}")
    
    # 创建训练器
    trainer = LSTMAETrainer(
        input_dim=6,  # spd, acc, soc, v, i, power
        hidden_dim=64,
        latent_dim=16,
        num_layers=2,
        n_clusters=5
    )
    
    # 准备序列数据
    sequences, valid_events = trainer.prepare_sequences(events, max_seq_len=100)
    
    # 训练
    trainer.train(sequences, epochs=50, batch_size=32, lr=0.001)
    
    # 提取特征
    latent_features = trainer.extract_features(sequences)
    print(f"\n📊 潜在特征形状: {latent_features.shape}")
    
    # 聚类
    labels, metrics = trainer.cluster(latent_features)
    
    # 保存结果
    results_df = pd.DataFrame({
        'event_id': [e['event_id'] for e in valid_events],
        'vehicle_id': [e['vehicle_id'] for e in valid_events],
        'cluster': labels
    })
    results_df.to_csv('./results/lstm_ae/clustered_results.csv', index=False)
    
    # 可视化
    trainer.visualize_training()
    
    # 保存模型
    trainer.save_model()
    
    print("\n✅ LSTM-AE模型训练完成！")