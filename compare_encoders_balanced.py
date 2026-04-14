"""
======================================================================
🔥 编码器对比 - 平衡版（方案A）
======================================================================
优化配置：
- 15万样本（原50万的30%）
- 最长1000点（保留更多长途行程）
- 10个epochs（充分训练）
- 预计时间：30-40分钟
- 预期效果：Sil 0.40+
======================================================================
"""

import time
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from tqdm import tqdm

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print(f"🖥️  使用设备: {device}")


# ==================== 数据集 ====================
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
    """处理变长序列的collate函数"""
    driving_seqs, energy_seqs, lengths = zip(*batch)
    lengths = torch.LongTensor(lengths)
    sorted_indices = torch.argsort(lengths, descending=True)
    
    lengths_sorted = lengths[sorted_indices]
    driving_sorted = [driving_seqs[i] for i in sorted_indices]
    energy_sorted = [energy_seqs[i] for i in sorted_indices]
    
    driving_padded = pad_sequence(driving_sorted, batch_first=True)
    energy_padded = pad_sequence(energy_sorted, batch_first=True)
    
    return driving_padded, energy_padded, lengths_sorted, sorted_indices


# ==================== 加载与过滤数据 ====================
def load_and_filter_data(data_dir, config):
    """加载并过滤数据"""
    print("="*70)
    print("📂 加载数据（方案A：平衡版）")
    print("="*70)
    
    data_path = Path(data_dir)
    
    if not (data_path / 'driving_sequences.npy').exists():
        print("❌ 数据不存在，请先运行 prepare_full_data_soc_only.py")
        return None, None, None
    
    # 加载
    driving = np.load(data_path / 'driving_sequences.npy', allow_pickle=True)
    energy = np.load(data_path / 'energy_sequences.npy', allow_pickle=True)
    lengths = np.load(data_path / 'seq_lengths.npy')
    
    print(f"原始数据: {len(driving):,} 样本")
    print(f"   长度范围: {lengths.min()} - {lengths.max()}")
    print(f"   平均长度: {lengths.mean():.1f}")
    
    # 1. 过滤长度
    min_len = config['min_seq_len']
    max_len = config['max_seq_len']
    valid_mask = (lengths >= min_len) & (lengths <= max_len)
    
    driving = driving[valid_mask]
    energy = energy[valid_mask]
    lengths = lengths[valid_mask]
    
    filtered_out = np.sum(~valid_mask)
    print(f"\n🔧 长度过滤 ({min_len}-{max_len}):")
    print(f"   保留: {len(driving):,} 样本")
    print(f"   过滤: {filtered_out:,} 样本 ({filtered_out/(filtered_out+len(driving))*100:.1f}%)")
    
    # 2. 随机采样
    max_samples = config['max_samples']
    if len(driving) > max_samples:
        print(f"\n🎲 随机采样到 {max_samples:,} 样本...")
        indices = np.random.choice(len(driving), max_samples, replace=False)
        driving = driving[indices]
        energy = energy[indices]
        lengths = lengths[indices]
    else:
        print(f"\n✅ 数据量 {len(driving):,} 已小于 {max_samples:,}，无需采样")
    
    print(f"\n✅ 最终数据:")
    print(f"   样本数: {len(driving):,}")
    print(f"   平均长度: {lengths.mean():.1f}")
    print(f"   中位数长度: {np.median(lengths):.0f}")
    print(f"   最大长度: {lengths.max()}")
    print(f"   P95长度: {np.percentile(lengths, 95):.0f}")
    print(f"   驾驶特征: {driving[0].shape[1]}D")
    print(f"   能量特征: {energy[0].shape[1]}D")
    print("="*70)
    
    return driving, energy, lengths


# ==================== 编码器实现 ====================
class GRUEncoderVarLen(nn.Module):
    """支持变长的GRU编码器"""
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers,
                         batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        self.hidden_dim = hidden_dim
        
    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        _, h_n = self.gru(packed)
        return h_n[-1]


class LSTMEncoderVarLen(nn.Module):
    """支持变长的LSTM编码器"""
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                           batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        self.hidden_dim = hidden_dim
        
    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        _, (h_n, _) = self.lstm(packed)
        return h_n[-1]


class BiGRUEncoderVarLen(nn.Module):
    """支持变长的双向GRU"""
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.bigru = nn.GRU(input_dim, hidden_dim // 2, num_layers,
                           batch_first=True, bidirectional=True,
                           dropout=0.2 if num_layers > 1 else 0)
        self.hidden_dim = hidden_dim
        
    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        _, h_n = self.bigru(packed)
        forward = h_n[-2]
        backward = h_n[-1]
        return torch.cat([forward, backward], dim=1)


class BiLSTMEncoderVarLen(nn.Module):
    """支持变长的双向LSTM"""
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.bilstm = nn.LSTM(input_dim, hidden_dim // 2, num_layers,
                             batch_first=True, bidirectional=True,
                             dropout=0.2 if num_layers > 1 else 0)
        self.hidden_dim = hidden_dim
        
    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        _, (h_n, _) = self.bilstm(packed)
        forward = h_n[-2]
        backward = h_n[-1]
        return torch.cat([forward, backward], dim=1)


# ==================== 双通道模型 ====================
class DualChannelModelVarLen(nn.Module):
    def __init__(self, encoder_type, driving_dim, energy_dim, latent_dim):
        super().__init__()
        
        # 创建编码器
        if encoder_type == 'GRU':
            self.driving_encoder = GRUEncoderVarLen(driving_dim, latent_dim)
            self.energy_encoder = GRUEncoderVarLen(energy_dim, latent_dim)
        elif encoder_type == 'LSTM':
            self.driving_encoder = LSTMEncoderVarLen(driving_dim, latent_dim)
            self.energy_encoder = LSTMEncoderVarLen(energy_dim, latent_dim)
        elif encoder_type == 'BiGRU':
            self.driving_encoder = BiGRUEncoderVarLen(driving_dim, latent_dim)
            self.energy_encoder = BiGRUEncoderVarLen(energy_dim, latent_dim)
        elif encoder_type == 'BiLSTM':
            self.driving_encoder = BiLSTMEncoderVarLen(driving_dim, latent_dim)
            self.energy_encoder = BiLSTMEncoderVarLen(energy_dim, latent_dim)
        else:
            raise ValueError(f"Unknown encoder: {encoder_type}")
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 解码器
        self.decoder = nn.Linear(64, driving_dim + energy_dim)
        
    def forward(self, driving, energy, lengths):
        driving_feat = self.driving_encoder(driving, lengths)
        energy_feat = self.energy_encoder(energy, lengths)
        combined = torch.cat([driving_feat, energy_feat], dim=1)
        fused = self.fusion(combined)
        reconstructed = self.decoder(fused)
        return reconstructed, combined


# ==================== 训练函数 ====================
def train_model_balanced(model, train_loader, config):
    """平衡版训练（10 epochs）"""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    start_time = time.time()
    
    print(f"\n🚀 开始训练 ({config['epochs']} epochs)...")
    
    for epoch in range(config['epochs']):
        epoch_loss = 0
        batch_count = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:2d}/{config['epochs']}", leave=False)
        
        for driving, energy, lengths, _ in pbar:
            driving = driving.to(device)
            energy = energy.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            reconstructed, _ = model(driving, energy, lengths)
            
            # 构造目标
            targets = []
            for idx, length in enumerate(lengths):
                d_last = driving[idx, length-1, :]
                e_last = energy[idx, length-1, :]
                targets.append(torch.cat([d_last, e_last]))
            targets = torch.stack(targets)
            
            # 计算损失
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
            
            # 每2个epoch打印一次
            if (epoch + 1) % 2 == 0:
                print(f"Epoch {epoch+1:2d}/{config['epochs']} | Loss: {avg_loss:.4f} | Best: {best_loss:.4f}")
    
    training_time = time.time() - start_time
    print(f"✅ 训练完成 | 最佳损失: {best_loss:.4f} | 用时: {training_time:.1f}s")
    
    return best_loss, training_time


def extract_features_varlen(model, loader):
    """提取特征"""
    model.eval()
    all_features = []
    all_indices = []
    
    with torch.no_grad():
        for driving, energy, lengths, sorted_idx in tqdm(loader, desc="提取特征", leave=False):
            driving = driving.to(device)
            energy = energy.to(device)
            _, feat = model(driving, energy, lengths)
            all_features.append(feat.cpu().numpy())
            all_indices.append(sorted_idx.numpy())
    
    features = np.vstack(all_features)
    indices = np.concatenate(all_indices)
    
    # 恢复原始顺序
    unsort_indices = np.argsort(indices)
    features = features[unsort_indices]
    
    return features


def clustering(features, k=4):
    """K-means聚类"""
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = kmeans.fit_predict(features)
    
    sil = silhouette_score(features, labels)
    unique, counts = np.unique(labels, return_counts=True)
    cv = np.std(counts) / np.mean(counts)
    distribution = dict(zip(unique.tolist(), counts.tolist()))
    
    return labels, sil, cv, distribution


# ==================== 主对比流程 ====================
class BalancedEncoderComparison:
    def __init__(self, config):
        self.config = config
        self.results = {}
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run_single_encoder(self, encoder_type, train_loader, driving_dim, energy_dim):
        """测试单个编码器"""
        print("\n" + "="*70)
        print(f"🧪 测试编码器: {encoder_type}")
        print("="*70)
        
        try:
            # 创建模型
            model = DualChannelModelVarLen(
                encoder_type, driving_dim, energy_dim, self.config['latent_dim']
            ).to(device)
            
            params = sum(p.numel() for p in model.parameters())
            print(f"参数量: {params:,}")
            
            # 训练
            best_loss, train_time = train_model_balanced(model, train_loader, self.config)
            
            # 提取特征
            print("\n🔍 提取特征...")
            features = extract_features_varlen(model, train_loader)
            
            # 聚类
            print("🎯 聚类评估...")
            labels, sil, cv, dist = clustering(features, self.config['k_value'])
            
            print(f"\n📊 聚类结果:")
            print(f"   轮廓系数: {sil:.3f}")
            print(f"   CV: {cv:.3f}")
            print(f"   分布: {dist}")
            
            # 保存结果
            self.results[encoder_type] = {
                'encoder': encoder_type,
                'params': params,
                'best_loss': best_loss,
                'training_time': train_time,
                'silhouette': sil,
                'cv': cv,
                'distribution': dist
            }
            
            # 保存文件
            torch.save(model.state_dict(), self.output_dir / f'{encoder_type}_model.pth')
            np.save(self.output_dir / f'{encoder_type}_features.npy', features)
            np.save(self.output_dir / f'{encoder_type}_labels.npy', labels)
            
            return True
            
        except Exception as e:
            print(f"\n❌ {encoder_type} 失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_all_encoders(self, driving_data, energy_data, lengths):
        """运行所有编码器对比"""
        print("="*70)
        print("🔥 开始编码器架构对比（方案A：平衡版）")
        print("="*70)
        
        # 创建数据加载器
        dataset = VariableLengthDataset(driving_data, energy_data, lengths)
        train_loader = DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            collate_fn=collate_variable_length,
            num_workers=0
        )
        
        driving_dim = driving_data[0].shape[1]
        energy_dim = energy_data[0].shape[1]
        
        print(f"\n数据信息:")
        print(f"   样本数: {len(dataset):,}")
        print(f"   驾驶特征: {driving_dim}D")
        print(f"   能量特征: {energy_dim}D")
        print(f"   Batch size: {self.config['batch_size']}")
        print(f"   Epochs: {self.config['epochs']}")
        
        # 测试所有编码器
        encoders = ['GRU', 'LSTM', 'BiGRU', 'BiLSTM']
        
        total_start = time.time()
        
        for encoder in encoders:
            self.run_single_encoder(encoder, train_loader, driving_dim, energy_dim)
        
        total_time = time.time() - total_start
        
        # 生成报告
        if len(self.results) > 0:
            self.save_summary()
            self.visualize_comparison()
            
            print(f"\n⏱️  总用时: {total_time/60:.1f} 分钟")
        else:
            print("\n⚠️  所有编码器都失败了")
        
        print("\n" + "="*70)
        print("✅ 编码器对比完成！")
        print(f"📁 结果保存在: {self.output_dir}")
        print("="*70)
    
    def save_summary(self):
        """保存汇总结果"""
        summary = []
        for name, result in self.results.items():
            summary.append({
                '编码器': result['encoder'],
                '参数量': result['params'],
                '训练时间(s)': round(result['training_time'], 1),
                '最佳损失': round(result['best_loss'], 4),
                '轮廓系数': round(result['silhouette'], 3),
                'CV': round(result['cv'], 3)
            })
        
        df = pd.DataFrame(summary)
        df = df.sort_values('轮廓系数', ascending=False)
        
        print("\n" + "="*70)
        print("📊 编码器对比汇总（方案A）")
        print("="*70)
        print(df.to_string(index=False))
        print("="*70)
        
        # 保存CSV
        df.to_csv(self.output_dir / 'summary.csv', index=False, encoding='utf-8-sig')
        
        # 保存JSON
        with open(self.output_dir / 'results.json', 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        # 对比之前最佳结果
        print(f"\n📈 与之前最佳结果对比:")
        best_encoder = df.iloc[0]
        print(f"   当前最佳: {best_encoder['编码器']}")
        print(f"   轮廓系数: {best_encoder['轮廓系数']:.3f}")
        print(f"   之前GRU(SOC): 0.438")
        
        if best_encoder['轮廓系数'] >= 0.40:
            improvement = (best_encoder['轮廓系数'] - 0.325) / 0.325 * 100
            print(f"   ✅ 相比快速版提升: {improvement:+.1f}%")
        
    def visualize_comparison(self):
        """生成对比可视化"""
        if len(self.results) == 0:
            return
        
        print("\n📊 生成可视化...")
        
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        encoders = list(self.results.keys())
        colors = plt.cm.Set2(np.linspace(0, 1, len(encoders)))
        
        # 1. 轮廓系数对比
        ax = fig.add_subplot(gs[0, 0])
        sils = [r['silhouette'] for r in self.results.values()]
        bars = ax.bar(encoders, sils, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax.set_ylabel('轮廓系数', fontsize=12, fontweight='bold')
        ax.set_title('聚类质量对比', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=0.438, color='r', linestyle='--', linewidth=2, label='之前最佳(GRU)')
        ax.legend()
        
        for bar, val in zip(bars, sils):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # 2. CV对比
        ax = fig.add_subplot(gs[0, 1])
        cvs = [r['cv'] for r in self.results.values()]
        bars = ax.bar(encoders, cvs, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax.set_ylabel('变异系数 (CV)', fontsize=12, fontweight='bold')
        ax.set_title('聚类平衡性对比', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=0.539, color='r', linestyle='--', linewidth=2, label='之前最佳')
        ax.legend()
        
        for bar, val in zip(bars, cvs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # 3. 参数量对比
        ax = fig.add_subplot(gs[0, 2])
        params = [r['params'] / 1000 for r in self.results.values()]
        bars = ax.bar(encoders, params, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax.set_ylabel('参数量 (K)', fontsize=12, fontweight='bold')
        ax.set_title('模型复杂度', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, params):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}K', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 4. 训练时间对比
        ax = fig.add_subplot(gs[1, 0])
        times = [r['training_time'] / 60 for r in self.results.values()]  # 转为分钟
        bars = ax.bar(encoders, times, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax.set_ylabel('训练时间 (分钟)', fontsize=12, fontweight='bold')
        ax.set_title('训练效率', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}min', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 5. 损失对比
        ax = fig.add_subplot(gs[1, 1])
        losses = [r['best_loss'] for r in self.results.values()]
        bars = ax.bar(encoders, losses, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax.set_ylabel('最佳损失', fontsize=12, fontweight='bold')
        ax.set_title('重构质量', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, losses):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 6. 综合排名
        ax = fig.add_subplot(gs[1, 2])
        ax.axis('off')
        
        # 计算综合得分
        rankings = []
        sils_norm = np.array(sils) / max(sils)
        cvs_norm = 1 - np.array(cvs) / max(cvs)
        times_norm = 1 - np.array(times) / max(times)
        
        for idx, encoder in enumerate(encoders):
            score = (sils_norm[idx] * 0.5 + cvs_norm[idx] * 0.3 + times_norm[idx] * 0.2)
            rankings.append({
                'encoder': encoder,
                'score': score,
                'sil': sils[idx],
                'cv': cvs[idx]
            })
        
        rankings = sorted(rankings, key=lambda x: x['score'], reverse=True)
        
        rank_text = "🏆 综合排名\n\n"
        rank_text += f"{'排名':<8}{'编码器':<12}{'轮廓系数':<12}{'CV':<10}\n"
        rank_text += "─" * 45 + "\n"
        
        for rank, item in enumerate(rankings, 1):
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
            rank_text += f"{medal:<8}{item['encoder']:<12}{item['sil']:.3f}{' '*8}{item['cv']:.3f}\n"
        
        ax.text(0.1, 0.5, rank_text, fontsize=11, family='monospace',
               verticalalignment='center',
               bbox=dict(boxstyle='round,pad=1', facecolor='lightgreen', alpha=0.3,
                        edgecolor='green', linewidth=2))
        
        plt.suptitle('编码器架构对比 - 方案A（平衡版）', fontsize=16, fontweight='bold')
        plt.savefig(self.output_dir / 'comparison_balanced.png',
                   dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ 可视化保存至: {self.output_dir / 'comparison_balanced.png'}")


# ==================== 主程序 ====================
if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # 方案A配置：平衡版
    config = {
        'data_dir': './results/temporal_soc_full',
        'latent_dim': 16,
        'batch_size': 128,
        'epochs': 10,              # ← 10个epoch
        'lr': 0.001,
        'k_value': 4,
        'output_dir': './results/encoder_comparison_balanced',
        
        # 数据过滤配置
        'max_samples': 150000,     # ← 15万样本
        'max_seq_len': 1000,       # ← 最长1000点
        'min_seq_len': 10
    }
    
    print("="*70)
    print("🚀 编码器对比 - 方案A（平衡版）")
    print("="*70)
    print("\n配置:")
    print(f"   样本数: 最多 {config['max_samples']:,}")
    print(f"   序列长度: {config['min_seq_len']}-{config['max_seq_len']}")
    print(f"   训练轮数: {config['epochs']}")
    print(f"   Batch size: {config['batch_size']}")
    print(f"\n⏱️  预计时间: 30-40分钟")
    print(f"🎯 预期效果: Sil 0.40+\n")
    
    # 加载并过滤数据
    driving, energy, lengths = load_and_filter_data(config['data_dir'], config)
    
    if driving is not None:
        # 运行对比
        comparison = BalancedEncoderComparison(config)
        comparison.run_all_encoders(driving, energy, lengths)