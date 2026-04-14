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

class TemporalBlock(nn.Module):
    """TCN的基本块"""
    
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
        # ⭐ 添加：计算输出长度用于裁剪
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
    
    def forward(self, x):
        original_len = x.size(2)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        # ⭐ 关键修复：裁剪到原始长度
        if out.size(2) != original_len:
            out = out[:, :, :original_len]
        
        res = x if self.downsample is None else self.downsample(x)
        
        # ⭐ 确保残差连接的尺寸匹配
        if res.size(2) != out.size(2):
            res = res[:, :, :out.size(2)]
        
        return self.relu(out + res)


class TCNAutoencoder(nn.Module):
    """TCN自编码器（修复版）"""
    
    def __init__(self, input_dim, num_channels=[64, 64, 32], kernel_size=3, latent_dim=16, dropout=0.2):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # 编码器
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            # ⭐ 使用因果填充（causal padding）
            padding = (kernel_size - 1) * dilation_size
            
            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size,
                stride=1, dilation=dilation_size,
                padding=padding,
                dropout=dropout
            ))
        
        self.encoder = nn.Sequential(*layers)
        
        # 到潜在空间
        self.to_latent = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(num_channels[-1], latent_dim)
        )
        
        # 解码器
        self.from_latent = nn.Linear(latent_dim, num_channels[-1])
        
        decoder_layers = []
        for i in range(num_levels-1, -1, -1):
            dilation_size = 2 ** i
            in_channels = num_channels[i]
            out_channels = num_channels[i-1] if i > 0 else input_dim
            
            padding = (kernel_size - 1) * dilation_size
            
            decoder_layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size,
                stride=1, dilation=dilation_size,
                padding=padding,
                dropout=dropout
            ))
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        batch_size, seq_len, _ = x.size()
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        
        # 编码
        encoded = self.encoder(x)
        latent = self.to_latent(encoded)
        
        # 解码
        decoded = self.from_latent(latent)
        decoded = decoded.unsqueeze(2).repeat(1, 1, seq_len)
        
        output = self.decoder(decoded)
        
        # ⭐ 确保输出长度与输入相同
        if output.size(2) != seq_len:
            output = output[:, :, :seq_len]
        
        output = output.transpose(1, 2)  # (batch, seq_len, input_dim)
        
        return output, latent


class SequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = torch.FloatTensor(sequences)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]


class TCNAETrainer:
    """TCN自编码器训练器（修复版）"""
    
    def __init__(self, input_dim, num_channels=[64, 64, 32], latent_dim=16, n_clusters=5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  使用设备: {self.device}")
        
        self.model = TCNAutoencoder(input_dim, num_channels, latent_dim=latent_dim).to(self.device)
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.history = {'train_loss': [], 'val_loss': []}
        
        print(f"📐 TCN参数: channels={num_channels}, latent_dim={latent_dim}")
    
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
            
            # ⭐ 严格控制序列长度
            if len(seq) < max_seq_len:
                padding = np.zeros((max_seq_len - len(seq), seq.shape[1]))
                seq = np.vstack([seq, padding])
            elif len(seq) > max_seq_len:
                seq = seq[:max_seq_len]
            
            # ⭐ 再次确认长度
            assert seq.shape[0] == max_seq_len, f"序列长度错误: {seq.shape[0]} != {max_seq_len}"
            
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
        print("🚀 开始训练TCN自编码器")
        print("="*60)
        
        n_val = int(len(sequences) * val_split)
        indices = np.random.permutation(len(sequences))
        
        train_dataset = SequenceDataset(sequences[indices[n_val:]])
        val_dataset = SequenceDataset(sequences[indices[:n_val]])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            
            for batch in train_loader:
                batch = batch.to(self.device)
                
                reconstructed, latent = self.model(batch)
                
                # ⭐ 确保形状匹配
                if reconstructed.shape != batch.shape:
                    print(f"警告: 形状不匹配 {reconstructed.shape} vs {batch.shape}")
                    min_len = min(reconstructed.size(1), batch.size(1))
                    reconstructed = reconstructed[:, :min_len, :]
                    batch = batch[:, :min_len, :]
                
                loss = criterion(reconstructed, batch)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            self.model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
                    reconstructed, latent = self.model(batch)
                    
                    if reconstructed.shape != batch.shape:
                        min_len = min(reconstructed.size(1), batch.size(1))
                        reconstructed = reconstructed[:, :min_len, :]
                        batch = batch[:, :min_len, :]
                    
                    loss = criterion(reconstructed, batch)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - Train: {train_loss:.4f}, Val: {val_loss:.4f}")
            
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
        """提取特征"""
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
        print("\n🔵 聚类...")
        
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
    
    def visualize_training(self, save_dir='./results/tcn_ae'):
        """可视化"""
        os.makedirs(save_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['train_loss'], label='Train Loss', linewidth=2)
        plt.plot(self.history['val_loss'], label='Val Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('TCN-AE Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{save_dir}/training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 训练曲线已保存")
    
    def save_model(self, save_dir='./results/tcn_ae'):
        """保存模型"""
        os.makedirs(save_dir, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'history': self.history
        }, f'{save_dir}/tcn_ae_model.pth')
        
        print(f"💾 模型已保存")


if __name__ == "__main__":
    import pickle
    
    with open('./results/events/events.pkl', 'rb') as f:
        events = pickle.load(f)
    
    trainer = TCNAETrainer(input_dim=6, num_channels=[64, 64, 32], latent_dim=16, n_clusters=5)
    sequences, valid_events = trainer.prepare_sequences(events, max_seq_len=100)
    trainer.train(sequences, epochs=30, batch_size=32)
    
    latent_features = trainer.extract_features(sequences)
    labels, metrics = trainer.cluster(latent_features)
    
    results_df = pd.DataFrame({
        'event_id': [e['event_id'] for e in valid_events],
        'vehicle_id': [e['vehicle_id'] for e in valid_events],
        'cluster': labels
    })
    results_df.to_csv('./results/tcn_ae/clustered_results.csv', index=False)
    
    trainer.visualize_training()
    trainer.save_model()
    
    print("\n✅ TCN-AE训练完成！")