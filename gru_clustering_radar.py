"""
======================================================================
🎯 GRU聚类分析 + 雷达图可视化
======================================================================
1. 确定最优K值
2. GRU特征提取 + K-means聚类
3. 分析每个簇的驾驶特征
4. 绘制雷达图对比
======================================================================
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

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("🎯 GRU聚类分析 + 雷达图可视化")
print("="*70)


# ==================== 数据加载 ====================
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


# ==================== GRU模型定义 ====================
class GRUEncoderVarLen(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers,
                         batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        self.hidden_dim = hidden_dim
        
    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        _, h_n = self.gru(packed)
        return h_n[-1]


class DualChannelGRU(nn.Module):
    def __init__(self, driving_dim, energy_dim, latent_dim):
        super().__init__()
        
        self.driving_encoder = GRUEncoderVarLen(driving_dim, latent_dim)
        self.energy_encoder = GRUEncoderVarLen(energy_dim, latent_dim)
        
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.decoder = nn.Linear(64, driving_dim + energy_dim)
        
    def forward(self, driving, energy, lengths):
        driving_feat = self.driving_encoder(driving, lengths)
        energy_feat = self.energy_encoder(energy, lengths)
        combined = torch.cat([driving_feat, energy_feat], dim=1)
        fused = self.fusion(combined)
        reconstructed = self.decoder(fused)
        return reconstructed, combined


# ==================== 加载数据 ====================
def load_data(data_dir, max_samples=50000):
    """加载数据"""
    print("\n" + "="*70)
    print("📂 加载数据")
    print("="*70)
    
    data_path = Path(data_dir)
    
    driving = np.load(data_path / 'driving_sequences.npy', allow_pickle=True)
    energy = np.load(data_path / 'energy_sequences.npy', allow_pickle=True)
    lengths = np.load(data_path / 'seq_lengths.npy')
    
    # 过滤长度
    valid_mask = (lengths >= 10) & (lengths <= 1000)
    driving = driving[valid_mask]
    energy = energy[valid_mask]
    lengths = lengths[valid_mask]
    
    # 采样
    if len(driving) > max_samples:
        indices = np.random.choice(len(driving), max_samples, replace=False)
        driving = driving[indices]
        energy = energy[indices]
        lengths = lengths[indices]
    
    print(f"✅ 数据加载完成:")
    print(f"   样本数: {len(driving):,}")
    print(f"   驾驶特征: {driving[0].shape[1]}D")
    print(f"   能量特征: {energy[0].shape[1]}D")
    
    return driving, energy, lengths


# ==================== 训练GRU模型 ====================
def train_gru(model, train_loader, epochs=10, lr=0.001):
    """训练GRU模型"""
    print("\n" + "="*70)
    print("🚀 训练GRU模型")
    print("="*70)
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        epoch_loss = 0
        batch_count = 0
        
        for driving, energy, lengths, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
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
        
        if batch_count > 0:
            avg_loss = epoch_loss / batch_count
            best_loss = min(best_loss, avg_loss)
            
            if (epoch + 1) % 2 == 0:
                print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f} | Best: {best_loss:.4f}")
    
    print(f"\n✅ 训练完成 | 最佳损失: {best_loss:.4f}")
    return model


# ==================== 提取特征 ====================
def extract_features(model, loader):
    """提取GRU特征"""
    print("\n🔍 提取特征...")
    
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
    unsort_indices = np.argsort(indices)
    features = features[unsort_indices]
    
    print(f"✅ 特征提取完成: {features.shape}")
    return features


# ==================== 确定最优K值 ====================
def find_optimal_k(features, k_range=(2, 10)):
    """使用肘部法则和轮廓系数确定最优K值"""
    print("\n" + "="*70)
    print("🔍 确定最优聚类数K")
    print("="*70)
    
    inertias = []
    silhouettes = []
    davies_bouldins = []
    
    k_values = range(k_range[0], k_range[1] + 1)
    
    for k in tqdm(k_values, desc="测试不同K值"):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = kmeans.fit_predict(features)
        
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(features, labels))
        davies_bouldins.append(davies_bouldin_score(features, labels))
    
    # 绘制评估指标
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. 肘部法则
    ax = axes[0]
    ax.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('聚类数 K', fontsize=12, fontweight='bold')
    ax.set_ylabel('Inertia（越小越好）', fontsize=12, fontweight='bold')
    ax.set_title('肘部法则', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # 2. 轮廓系数
    ax = axes[1]
    ax.plot(k_values, silhouettes, 'go-', linewidth=2, markersize=8)
    ax.set_xlabel('聚类数 K', fontsize=12, fontweight='bold')
    ax.set_ylabel('轮廓系数（越大越好）', fontsize=12, fontweight='bold')
    ax.set_title('轮廓系数', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    
    best_k_sil = k_values[np.argmax(silhouettes)]
    ax.axvline(x=best_k_sil, color='r', linestyle='--', linewidth=2, label=f'最优K={best_k_sil}')
    ax.legend()
    
    # 3. Davies-Bouldin指数
    ax = axes[2]
    ax.plot(k_values, davies_bouldins, 'ro-', linewidth=2, markersize=8)
    ax.set_xlabel('聚类数 K', fontsize=12, fontweight='bold')
    ax.set_ylabel('Davies-Bouldin指数（越小越好）', fontsize=12, fontweight='bold')
    ax.set_title('Davies-Bouldin指数', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    
    best_k_db = k_values[np.argmin(davies_bouldins)]
    ax.axvline(x=best_k_db, color='r', linestyle='--', linewidth=2, label=f'最优K={best_k_db}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('./results/optimal_k_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✅ K值分析图保存至: ./results/optimal_k_analysis.png")
    
    # 打印结果
    print(f"\n📊 K值评估结果:")
    print(f"{'K':<5}{'Inertia':<15}{'Silhouette':<15}{'Davies-Bouldin':<15}")
    print("-" * 50)
    for i, k in enumerate(k_values):
        print(f"{k:<5}{inertias[i]:<15.2f}{silhouettes[i]:<15.3f}{davies_bouldins[i]:<15.3f}")
    
    print(f"\n🎯 推荐:")
    print(f"   轮廓系数最优: K = {best_k_sil}")
    print(f"   Davies-Bouldin最优: K = {best_k_db}")
    
    # 综合推荐
    recommended_k = best_k_sil  # 优先用轮廓系数
    print(f"\n✅ 综合推荐: K = {recommended_k}")
    
    return recommended_k, {
        'k_values': list(k_values),
        'inertias': inertias,
        'silhouettes': silhouettes,
        'davies_bouldins': davies_bouldins,
        'recommended_k': recommended_k
    }


# ==================== K-means聚类 ====================
def perform_clustering(features, k):
    """执行K-means聚类"""
    print(f"\n🎯 使用K={k}进行聚类...")
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = kmeans.fit_predict(features)
    
    sil = silhouette_score(features, labels)
    unique, counts = np.unique(labels, return_counts=True)
    cv = np.std(counts) / np.mean(counts)
    
    print(f"✅ 聚类完成:")
    print(f"   轮廓系数: {sil:.3f}")
    print(f"   CV: {cv:.3f}")
    print(f"\n   聚类分布:")
    for cluster_id, count in zip(unique, counts):
        pct = count / len(labels) * 100
        print(f"      簇 {cluster_id}: {count:6,} ({pct:5.1f}%)")
    
    return labels, kmeans


# ==================== 提取每个簇的特征统计 ====================
def extract_cluster_features(driving_seqs, energy_seqs, labels, k):
    """提取每个簇的原始特征统计"""
    print("\n" + "="*70)
    print("📊 提取每个簇的特征统计")
    print("="*70)
    
    cluster_stats = []
    
    for cluster_id in range(k):
        print(f"\n分析簇 {cluster_id}...")
        
        # 找到属于该簇的序列
        cluster_mask = (labels == cluster_id)
        cluster_driving = driving_seqs[cluster_mask]
        cluster_energy = energy_seqs[cluster_mask]
        
        # 提取统计特征
        stats = {}
        
        # 驾驶特征统计（假设：[spd, acc]）
        all_spd = np.concatenate([seq[:, 0] for seq in cluster_driving])
        all_acc = np.concatenate([seq[:, 1] for seq in cluster_driving])
        
        stats['spd_mean'] = np.mean(all_spd)
        stats['spd_max'] = np.percentile(all_spd, 95)  # 用P95代替max避免极值
        stats['spd_std'] = np.std(all_spd)
        stats['idle_ratio'] = np.mean(all_spd < 1)
        stats['high_speed_ratio'] = np.mean(all_spd > 60)
        
        stats['acc_mean'] = np.abs(np.mean(all_acc))  # 加速度绝对值
        stats['acc_std'] = np.std(all_acc)
        stats['harsh_accel'] = np.mean(all_acc > 1.5)
        stats['harsh_decel'] = np.mean(all_acc < -1.5)
        
        # 能量特征统计（假设：[soc, v, i]）
        all_soc = np.concatenate([seq[:, 0] for seq in cluster_energy])
        all_v = np.concatenate([seq[:, 1] for seq in cluster_energy])
        all_i = np.concatenate([seq[:, 2] for seq in cluster_energy])
        
        stats['soc_mean'] = np.mean(all_soc)
        stats['soc_drop_rate'] = np.mean([seq[0, 0] - seq[-1, 0] for seq in cluster_energy if len(seq) > 1])
        
        stats['voltage_mean'] = np.mean(all_v)
        stats['current_mean'] = np.abs(np.mean(all_i))
        stats['power_mean'] = np.mean(np.abs(all_v * all_i))
        
        # 行程特征
        stats['trip_length'] = np.mean([len(seq) for seq in cluster_driving])
        
        cluster_stats.append(stats)
        
        print(f"   速度均值: {stats['spd_mean']:.1f} km/h")
        print(f"   怠速比例: {stats['idle_ratio']*100:.1f}%")
        print(f"   高速比例: {stats['high_speed_ratio']*100:.1f}%")
        print(f"   加速度标准差: {stats['acc_std']:.2f}")
    
    return cluster_stats


# ==================== 绘制雷达图 ====================
def plot_radar_chart(cluster_stats, k, output_path='./results/cluster_radar.png'):
    """绘制雷达图对比不同簇"""
    print("\n" + "="*70)
    print("🎨 绘制雷达图")
    print("="*70)
    
    # 选择要展示的特征
    features_to_plot = [
        'spd_mean',          # 平均速度
        'spd_std',           # 速度波动
        'high_speed_ratio',  # 高速占比
        'idle_ratio',        # 怠速占比
        'acc_std',           # 加速度波动
        'harsh_accel',       # 急加速
        'harsh_decel',       # 急减速
        'soc_drop_rate',     # SOC下降率
        'power_mean',        # 平均功率
        'trip_length'        # 行程长度
    ]
    
    # 中文标签
    feature_labels = [
        '平均速度',
        '速度波动',
        '高速占比',
        '怠速占比',
        '加速度波动',
        '急加速比例',
        '急减速比例',
        'SOC下降率',
        '平均功率',
        '行程长度'
    ]
    
    # 提取数据
    data = []
    for stats in cluster_stats:
        data.append([stats[f] for f in features_to_plot])
    
    data = np.array(data)
    
    # 归一化到0-1（每个特征单独归一化）
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data.T).T
    
    # 绘制
    N = len(features_to_plot)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
    
    colors = plt.cm.Set3(np.linspace(0, 1, k))
    
    for cluster_id in range(k):
        values = data_normalized[cluster_id].tolist()
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2.5, 
               label=f'簇 {cluster_id}', color=colors[cluster_id], markersize=8)
        ax.fill(angles, values, alpha=0.15, color=colors[cluster_id])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7, linewidth=1.5)
    
    ax.set_title(f'驾驶行为聚类特征对比（K={k}）', 
                fontsize=16, fontweight='bold', pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.15), 
             fontsize=12, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ 雷达图保存至: {output_path}")
    
    # 额外：绘制每个特征的柱状图对比
    plot_feature_bars(cluster_stats, k, features_to_plot, feature_labels)


def plot_feature_bars(cluster_stats, k, features, labels):
    """绘制特征柱状图对比"""
    print("\n📊 绘制特征柱状图...")
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    axes = axes.flatten()
    
    colors = plt.cm.Set3(np.linspace(0, 1, k))
    
    for idx, (feature, label) in enumerate(zip(features[:10], labels[:10])):
        ax = axes[idx]
        
        values = [stats[feature] for stats in cluster_stats]
        x = np.arange(k)
        
        bars = ax.bar(x, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        ax.set_xlabel('簇编号', fontsize=11, fontweight='bold')
        ax.set_ylabel(label, fontsize=11, fontweight='bold')
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'簇{i}' for i in range(k)])
        ax.grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 隐藏多余的子图
    for idx in range(len(features), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'聚类特征详细对比（K={k}）', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('./results/cluster_features_bars.png', dpi=300, bbox_inches='tight')
    print(f"✅ 柱状图保存至: ./results/cluster_features_bars.png")


# ==================== 聚类解释 ====================
def interpret_clusters(cluster_stats, k):
    """解释每个簇的特征"""
    print("\n" + "="*70)
    print("💡 聚类解释")
    print("="*70)
    
    for cluster_id, stats in enumerate(cluster_stats):
        print(f"\n📍 簇 {cluster_id}:")
        
        # 速度特征
        if stats['spd_mean'] > 40:
            print("   🚗 高速驾驶：平均速度较高")
        elif stats['spd_mean'] < 20:
            print("   🚦 城市驾驶：平均速度较低")
        else:
            print("   🛣️  混合驾驶：中等速度")
        
        # 怠速
        if stats['idle_ratio'] > 0.3:
            print("   ⏸️  频繁停车：高怠速比例")
        
        # 加速行为
        if stats['harsh_accel'] > 0.1 or stats['harsh_decel'] > 0.1:
            print("   ⚡ 激进驾驶：急加减速频繁")
        elif stats['acc_std'] < 0.5:
            print("   🍃 平稳驾驶：加速度变化小")
        
        # 高速
        if stats['high_speed_ratio'] > 0.3:
            print("   🛣️  高速路段：高速占比大")
        
        # 能耗
        if stats['soc_drop_rate'] > 2.0:
            print("   🔋 高能耗：SOC下降快")
        elif stats['soc_drop_rate'] < 1.0:
            print("   🔋 低能耗：SOC下降慢")
        
        # 行程长度
        if stats['trip_length'] > 100:
            print("   📏 长途行程：序列较长")
        elif stats['trip_length'] < 30:
            print("   📏 短途行程：序列较短")


# ==================== 主流程 ====================
def main():
    np.random.seed(42)
    torch.manual_seed(42)
    
    Path('./results').mkdir(exist_ok=True)
    
    # 1. 加载数据
    driving, energy, lengths = load_data('./results/temporal_soc_full', max_samples=50000)
    
    # 2. 创建数据加载器
    dataset = VariableLengthDataset(driving, energy, lengths)
    train_loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        collate_fn=collate_variable_length,
        num_workers=0
    )
    
    # 3. 创建并训练GRU模型
    model = DualChannelGRU(
        driving_dim=driving[0].shape[1],
        energy_dim=energy[0].shape[1],
        latent_dim=16
    ).to(device)
    
    model = train_gru(model, train_loader, epochs=10)
    
    # 4. 提取特征
    features = extract_features(model, train_loader)
    
    # 5. 确定最优K值
    optimal_k, k_analysis = find_optimal_k(features, k_range=(2, 8))
    
    # 6. 使用最优K值进行聚类
    labels, kmeans = perform_clustering(features, optimal_k)
    
    # 7. 提取每个簇的特征统计
    cluster_stats = extract_cluster_features(driving, energy, labels, optimal_k)
    
    # 8. 绘制雷达图
    plot_radar_chart(cluster_stats, optimal_k)
    
    # 9. 解释聚类
    interpret_clusters(cluster_stats, optimal_k)
    
    # 10. 保存结果
    results = {
        'optimal_k': optimal_k,
        'k_analysis': k_analysis,
        'cluster_stats': [{k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                          for k, v in stats.items()} for stats in cluster_stats],
        'silhouette': float(silhouette_score(features, labels)),
        'cv': float(np.std(np.bincount(labels)) / np.mean(np.bincount(labels)))
    }
    
    with open('./results/clustering_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*70)
    print("✅ 完成！所有结果保存在 ./results/")
    print("="*70)


if __name__ == "__main__":
    main()