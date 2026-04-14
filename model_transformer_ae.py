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
import math
from tqdm import tqdm

class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerAutoencoder(nn.Module):
    """Transformer自编码器 - 最流行的时序模型"""
    
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, latent_dim=16):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.latent_dim = latent_dim
        
        # 输入投影
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 编码到潜在空间
        self.to_latent = nn.Sequential(
            nn.Linear(d_model, latent_dim),
            nn.ReLU()
        )
        
        # 解码器
        self.from_latent = nn.Linear(latent_dim, d_model)
        
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        
        # 输出投影
        self.output_projection = nn.Linear(d_model, input_dim)
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        batch_size, seq_len, _ = x.size()
        
        # 编码
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        x = self.pos_encoder(x)
        encoded = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        
        # 取平均作为序列表示
        seq_repr = encoded.mean(dim=1)  # (batch, d_model)
        latent = self.to_latent(seq_repr)  # (batch, latent_dim)
        
        # 解码
        decoded_repr = self.from_latent(latent)  # (batch, d_model)
        decoded_repr = decoded_repr.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, d_model)
        
        decoded = self.transformer_decoder(decoded_repr)
        output = self.output_projection(decoded)  # (batch, seq_len, input_dim)
        
        return output, latent


class SequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = torch.FloatTensor(sequences)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]


class TransformerAETrainer:
    """Transformer自编码器训练器"""
    
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, latent_dim=16, n_clusters=5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  使用设备: {self.device}")
        
        self.model = TransformerAutoencoder(
            input_dim, d_model, nhead, num_layers, latent_dim
        ).to(self.device)
        
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.history = {'train_loss': [], 'val_loss': []}
        
        print(f"📐 Transformer参数: d_model={d_model}, heads={nhead}, layers={num_layers}")
    
    def prepare_sequences(self, events, max_seq_len=100):
        """准备时序数据"""
        print("\n🔧 准备时序数据...")
        
        sequences = []
        valid_events = []
        
        for event in tqdm(events, desc="处理事件"):
            event_data = event['data']
            
            feature_cols = ['spd', 'acc', 'soc', 'v', 'i', 'power']
            available_cols = [col for col in feature_cols if col in event_data.columns]
            
            if len(available_cols) == 0:
                continue
            
            seq = event_data[available_cols].values
            seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
            
            if len(seq) < max_seq_len:
                padding = np.zeros((max_seq_len - len(seq), seq.shape[1]))
                seq = np.vstack([seq, padding])
            else:
                seq = seq[:max_seq_len]
            
            sequences.append(seq)
            valid_events.append(event)
        
        sequences = np.array(sequences)
        
        # 标准化
        n_samples, seq_len, n_features = sequences.shape
        sequences_flat = sequences.reshape(-1, n_features)
        sequences_flat = self.scaler.fit_transform(sequences_flat)
        sequences = sequences_flat.reshape(n_samples, seq_len, n_features)
        
        print(f"✅ 序列数据形状: {sequences.shape}")
        
        return sequences, valid_events
    
    def train(self, sequences, epochs=50, batch_size=32, lr=0.001, val_split=0.2):
        """训练模型"""
        print("\n" + "="*60)
        print("🚀 开始训练Transformer自编码器")
        print("="*60)
        
        # 划分数据
        n_val = int(len(sequences) * val_split)
        indices = np.random.permutation(len(sequences))
        
        train_indices = indices[n_val:]
        val_indices = indices[:n_val]
        
        train_dataset = SequenceDataset(sequences[train_indices])
        val_dataset = SequenceDataset(sequences[val_indices])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0
        
        for epoch in range(epochs):
            # 训练
            self.model.train()
            train_loss = 0
            
            for batch in train_loader:
                batch = batch.to(self.device)
                
                reconstructed, latent = self.model(batch)
                loss = criterion(reconstructed, batch)
                
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
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - Train: {train_loss:.4f}, Val: {val_loss:.4f}")
            
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
        
        return np.vstack(latent_features)
    
    def cluster(self, latent_features):
        """聚类"""
        print("\n🔵 在潜在空间进行聚类...")
        
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(latent_features)
        
        metrics = {
            'silhouette': silhouette_score(latent_features, labels),
            'ch_score': calinski_harabasz_score(latent_features, labels),
            'db_score': davies_bouldin_score(latent_features, labels)
        }
        
        print(f"   Silhouette: {metrics['silhouette']:.3f}")
        print(f"   CH Score: {metrics['ch_score']:.2f}")
        print(f"   DB Score: {metrics['db_score']:.3f}")
        
        return labels, metrics
    
    def visualize_training(self, save_dir='./results/transformer_ae'):
        """可视化训练过程"""
        os.makedirs(save_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['train_loss'], label='Train Loss', linewidth=2)
        plt.plot(self.history['val_loss'], label='Val Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Transformer-AE Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{save_dir}/training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 训练曲线已保存至: {save_dir}/training_history.png")
    
    def save_model(self, save_dir='./results/transformer_ae'):
        """保存模型"""
        os.makedirs(save_dir, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'history': self.history
        }, f'{save_dir}/transformer_ae_model.pth')
        
        print(f"💾 模型已保存至: {save_dir}/transformer_ae_model.pth")


# 使用示例
if __name__ == "__main__":
    import pickle
    
    with open('./results/events/events.pkl', 'rb') as f:
        events = pickle.load(f)
    
    print(f"📊 事件数量: {len(events)}")
    
    trainer = TransformerAETrainer(
        input_dim=6,
        d_model=64,
        nhead=4,
        num_layers=2,
        latent_dim=16,
        n_clusters=5
    )
    
    sequences, valid_events = trainer.prepare_sequences(events, max_seq_len=100)
    trainer.train(sequences, epochs=30, batch_size=32)
    
    latent_features = trainer.extract_features(sequences)
    labels, metrics = trainer.cluster(latent_features)
    
    results_df = pd.DataFrame({
        'event_id': [e['event_id'] for e in valid_events],
        'vehicle_id': [e['vehicle_id'] for e in valid_events],
        'cluster': labels
    })
    results_df.to_csv('./results/transformer_ae/clustered_results.csv', index=False)
    
    trainer.visualize_training()
    trainer.save_model()
    
    print("\n✅ Transformer-AE训练完成！")