"""
======================================================================
🎯 GRU聚类分析 - 固定K=4版本
======================================================================
专门生成K=4的聚类结果和雷达图
所有输出文件带 _k4 后缀，不会覆盖之前的结果
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
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("🎯 GRU聚类分析 - K=4 专用版")
print("="*70)


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
    driving_seqs, energy_seqs, lengths = zip(*batch)
    lengths = torch.LongTensor(lengths)
    sorted_indices = torch.argsort(lengths, descending=True)
    
    lengths_sorted = lengths[sorted_indices]
    driving_sorted = [driving_seqs[i] for i in sorted_indices]
    energy_sorted = [energy_seqs[i] for i in sorted_indices]
    
    driving_padded = pad_sequence(driving_sorted, batch_first=True)
    energy_padded = pad_sequence(energy_sorted, batch_first=True)
    
    return driving_padded, energy_padded, lengths_sorted, sorted_indices


# ==================== GRU模型 ====================
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
def load_data(data_dir, max_samples=100000):
    """加载数据（增加样本���到10万）"""
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


# ==================== 训练或加载模型 ====================
def train_or_load_gru(model, train_loader, model_path='./results/gru_model_k4.pth', epochs=10):
    """训练GRU或加载已有模型"""
    
    # 尝试加载已有模型
    if Path(model_path).exists():
        print(f"\n✅ 发现已训练模型: {model_path}")
        print("   加载模型...")
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    
    # 否则训练新模型
    print("\n" + "="*70)
    print("🚀 训练GRU模型")
    print("="*70)
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
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
    
    # 保存模型
    torch.save(model.state_dict(), model_path)
    print(f"💾 模型已保存: {model_path}")
    
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


# ==================== K=4聚类 ====================
def perform_clustering_k4(features):
    """使用K=4进行聚类"""
    print("\n" + "="*70)
    print("🎯 K=4 聚类")
    print("="*70)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=20)
    labels = kmeans.fit_predict(features)
    
    sil = silhouette_score(features, labels)
    unique, counts = np.unique(labels, return_counts=True)
    cv = np.std(counts) / np.mean(counts)
    
    print(f"\n✅ 聚类完成:")
    print(f"   轮廓系数: {sil:.3f}")
    print(f"   CV: {cv:.3f}")
    print(f"\n   聚类分布:")
    for cluster_id, count in zip(unique, counts):
        pct = count / len(labels) * 100
        print(f"      簇 {cluster_id}: {count:6,} ({pct:5.1f}%)")
    
    return labels, kmeans, sil, cv


# ==================== 提取簇特征 ====================
def extract_cluster_features(driving_seqs, energy_seqs, labels):
    """提取每个簇的特征统计"""
    print("\n" + "="*70)
    print("📊 提取簇特征统计")
    print("="*70)
    
    cluster_stats = []
    
    for cluster_id in range(4):
        print(f"\n分析簇 {cluster_id}...")
        
        cluster_mask = (labels == cluster_id)
        cluster_driving = driving_seqs[cluster_mask]
        cluster_energy = energy_seqs[cluster_mask]
        
        stats = {}
        
        # 驾驶特征
        all_spd = np.concatenate([seq[:, 0] for seq in cluster_driving])
        all_acc = np.concatenate([seq[:, 1] for seq in cluster_driving])
        
        stats['平均速度'] = np.mean(all_spd)
        stats['最大速度'] = np.percentile(all_spd, 95)
        stats['速度波动'] = np.std(all_spd)
        stats['怠速占比'] = np.mean(all_spd < 1) * 100
        stats['高速占比'] = np.mean(all_spd > 60) * 100
        
        stats['加速度均值'] = np.abs(np.mean(all_acc))
        stats['加速度波动'] = np.std(all_acc)
        stats['急加速占比'] = np.mean(all_acc > 1.5) * 100
        stats['急减速占比'] = np.mean(all_acc < -1.5) * 100
        
        # 能量特征
        all_soc = np.concatenate([seq[:, 0] for seq in cluster_energy])
        all_v = np.concatenate([seq[:, 1] for seq in cluster_energy])
        all_i = np.concatenate([seq[:, 2] for seq in cluster_energy])
        
        stats['SOC均值'] = np.mean(all_soc)
        stats['SOC下降率'] = np.mean([seq[0, 0] - seq[-1, 0] for seq in cluster_energy if len(seq) > 1])
        
        stats['电压均值'] = np.mean(all_v)
        stats['电流均值'] = np.abs(np.mean(all_i))
        stats['功率均值'] = np.mean(np.abs(all_v * all_i))
        
        stats['行程长度'] = np.mean([len(seq) for seq in cluster_driving])
        
        cluster_stats.append(stats)
        
        # 打印关键特征
        print(f"   平均速度: {stats['平均速度']:.1f} km/h")
        print(f"   怠速占比: {stats['怠��占比']:.1f}%")
        print(f"   高速占比: {stats['高速占比']:.1f}%")
        print(f"   加速度波动: {stats['加速度波动']:.2f}")
    
    return cluster_stats


# ==================== 绘制雷达图（K=4专用）====================
def plot_radar_k4(cluster_stats):
    """绘制K=4雷达图"""
    print("\n" + "="*70)
    print("🎨 绘制K=4雷达图")
    print("="*70)
    
    # 选择展示的特征
    feature_keys = [
        '平均速度',
        '速度波动',
        '高速占比',
        '怠速占比',
        '加速度波动',
        '急加速占比',
        '急减速占比',
        'SOC下降率',
        '功率均值',
        '行程长度'
    ]
    
    # 提取数据
    data = []
    for stats in cluster_stats:
        data.append([stats[key] for key in feature_keys])
    
    data = np.array(data)
    
    # 归一化
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data.T).T
    
    # 绘制
    N = len(feature_keys)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(14, 14), subplot_kw=dict(projection='polar'))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    labels_cn = ['簇0', '簇1', '簇2', '簇3']
    
    for cluster_id in range(4):
        values = data_normalized[cluster_id].tolist()
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=3, 
               label=labels_cn[cluster_id], color=colors[cluster_id], 
               markersize=10)
        ax.fill(angles, values, alpha=0.20, color=colors[cluster_id])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_keys, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=11, color='gray')
    ax.grid(True, linestyle='--', alpha=0.7, linewidth=1.5)
    
    ax.set_title('驾驶行为聚类特征对比（K=4）', 
                fontsize=18, fontweight='bold', pad=35)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), 
             fontsize=13, framealpha=0.95, shadow=True)
    
    plt.tight_layout()
    plt.savefig('./results/cluster_radar_k4.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ 雷达图保存至: ./results/cluster_radar_k4.png")
    
    return data, data_normalized


# ==================== 绘制特征热力图 ====================
def plot_heatmap_k4(cluster_stats):
    """绘制特征热力图"""
    print("\n📊 绘制特征热力图...")
    
    feature_keys = [
        '平均速度', '速度波动', '高速占比', '怠速占比',
        '加速度波动', '急加速占比', '急减速占比',
        'SOC下降率', '功率均值', '行程长度'
    ]
    
    data = []
    for stats in cluster_stats:
        data.append([stats[key] for key in feature_keys])
    
    df = pd.DataFrame(data, columns=feature_keys, index=['簇0', '簇1', '簇2', '簇3'])
    
    # 归一化
    df_normalized = (df - df.min()) / (df.max() - df.min())
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.heatmap(df_normalized, annot=True, fmt='.2f', cmap='YlOrRd', 
               linewidths=2, cbar_kws={'label': '归一化值'}, 
               ax=ax, vmin=0, vmax=1)
    
    ax.set_title('聚类特征热力图（K=4）', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('聚类簇', fontsize=13, fontweight='bold')
    ax.set_xlabel('特征维度', fontsize=13, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('./results/cluster_heatmap_k4.png', dpi=300, bbox_inches='tight')
    print(f"✅ 热力图保存至: ./results/cluster_heatmap_k4.png")


# ==================== 绘制特征柱状图 ====================
def plot_feature_bars_k4(cluster_stats):
    """绘制特征柱状图对比"""
    print("\n📊 绘制特征柱状图...")
    
    feature_keys = [
        '平均速度', '速度波动', '高速占比', '怠速占比',
        '加速度波动', '急加速占比', '急减速占比',
        'SOC下降率', '功率均值', '行程长度'
    ]
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    axes = axes.flatten()
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    for idx, feature in enumerate(feature_keys):
        ax = axes[idx]
        
        values = [stats[feature] for stats in cluster_stats]
        x = np.arange(4)
        
        bars = ax.bar(x, values, color=colors, alpha=0.85, edgecolor='black', linewidth=2)
        
        ax.set_xlabel('聚类簇', fontsize=12, fontweight='bold')
        ax.set_ylabel(feature, fontsize=12, fontweight='bold')
        ax.set_title(feature, fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['簇0', '簇1', '簇2', '簇3'])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # 添加数值标签
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}', ha='center', va='bottom', 
                   fontsize=10, fontweight='bold')
    
    # 隐藏多余的子图
    for idx in range(len(feature_keys), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('聚类特征详细对比（K=4）', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig('./results/cluster_features_bars_k4.png', dpi=300, bbox_inches='tight')
    print(f"✅ 柱状图保存至: ./results/cluster_features_bars_k4.png")


# ==================== 聚类解释 ====================
def interpret_clusters_k4(cluster_stats):
    """解释每个簇的特征"""
    print("\n" + "="*70)
    print("💡 聚类解释（K=4）")
    print("="*70)
    
    interpretations = []
    
    for cluster_id, stats in enumerate(cluster_stats):
        print(f"\n📍 簇 {cluster_id}:")
        interpretation = []
        
        # 速度特征
        if stats['平均速度'] > 50:
            msg = "   🚗 高速驾驶：平均速度高（{:.1f} km/h）".format(stats['平均速度'])
            print(msg)
            interpretation.append("高速驾驶")
        elif stats['平均速度'] < 25:
            msg = "   🚦 城市拥堵：平均速度低（{:.1f} km/h）".format(stats['平均速度'])
            print(msg)
            interpretation.append("城市拥堵")
        else:
            msg = "   🛣️  混合路况：中等速度（{:.1f} km/h）".format(stats['平均速度'])
            print(msg)
            interpretation.append("混合路况")
        
        # 怠速
        if stats['怠速占比'] > 30:
            msg = "   ⏸️  频繁停车：怠速占比 {:.1f}%".format(stats['怠速占比'])
            print(msg)
            interpretation.append("频繁停车")
        
        # 高速
        if stats['高速占比'] > 30:
            msg = "   🛣️  高速路段：高速占比 {:.1f}%".format(stats['高速占比'])
            print(msg)
            interpretation.append("高速路段")
        
        # 加速行为
        if stats['急加速占比'] > 8 or stats['急减速占比'] > 8:
            msg = "   ⚡ 激进驾驶：急加速 {:.1f}%, 急减速 {:.1f}%".format(
                stats['急加速占比'], stats['急减速占比'])
            print(msg)
            interpretation.append("激进驾驶")
        elif stats['加速度波动'] < 0.6:
            msg = "   🍃 平稳驾驶：加速度波动小（{:.2f}）".format(stats['加速度波动'])
            print(msg)
            interpretation.append("平稳驾驶")
        
        # 能耗
        if stats['SOC下降率'] > 2.5:
            msg = "   🔋 高能耗：SOC下降率 {:.2f}%/段".format(stats['SOC下降率'])
            print(msg)
            interpretation.append("高能耗")
        elif stats['SOC下降率'] < 1.5:
            msg = "   🔋 节能驾驶：SOC下降率 {:.2f}%/段".format(stats['SOC下降率'])
            print(msg)
            interpretation.append("节能驾驶")
        
        # 行程长度
        if stats['行程长度'] > 120:
            msg = "   📏 长途行程：平均长度 {:.0f} 点".format(stats['行程长度'])
            print(msg)
            interpretation.append("长途")
        elif stats['行程长度'] < 50:
            msg = "   📏 短途行程：平均长度 {:.0f} 点".format(stats['行程长度'])
            print(msg)
            interpretation.append("短途")
        
        interpretations.append(' + '.join(interpretation))
    
    return interpretations


# ==================== 生成详细报告 ====================
def generate_report_k4(cluster_stats, labels, sil, cv, interpretations):
    """生成详细报告"""
    print("\n📄 生成详细报告...")
    
    report = {
        'k': 4,
        'silhouette': float(sil),
        'cv': float(cv),
        'sample_count': len(labels),
        'cluster_distribution': {
            f'cluster_{i}': int(np.sum(labels == i)) for i in range(4)
        },
        'cluster_interpretations': {
            f'cluster_{i}': interpretations[i] for i in range(4)
        },
        'cluster_features': {}
    }
    
    for i, stats in enumerate(cluster_stats):
        report['cluster_features'][f'cluster_{i}'] = {
            k: float(v) for k, v in stats.items()
        }
    
    # 保存JSON
    with open('./results/clustering_report_k4.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # 保存CSV
    df = pd.DataFrame(cluster_stats)
    df.index = ['簇0', '簇1', '簇2', '簇3']
    df.to_csv('./results/cluster_features_k4.csv', encoding='utf-8-sig')
    
    print(f"✅ 报告已保存:")
    print(f"   JSON: ./results/clustering_report_k4.json")
    print(f"   CSV:  ./results/cluster_features_k4.csv")


# ==================== 主流程 ====================
def main():
    np.random.seed(42)
    torch.manual_seed(42)
    
    Path('./results').mkdir(exist_ok=True)
    
    print(f"\n🎯 固定K=4聚类分析")
    print(f"   所有输出文件带 _k4 后缀\n")
    
    # 1. 加载数据
    driving, energy, lengths = load_data('./results/temporal_soc_full', max_samples=100000)
    
    # 2. 创建数据加载器
    dataset = VariableLengthDataset(driving, energy, lengths)
    train_loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        collate_fn=collate_variable_length,
        num_workers=0
    )
    
    # 3. 创建并训练/加载GRU模型
    model = DualChannelGRU(
        driving_dim=driving[0].shape[1],
        energy_dim=energy[0].shape[1],
        latent_dim=16
    ).to(device)
    
    model = train_or_load_gru(model, train_loader, epochs=10)
    
    # 4. 提取特征
    features = extract_features(model, train_loader)
    
    # 5. K=4聚类
    labels, kmeans, sil, cv = perform_clustering_k4(features)
    
    # 6. 提取簇特征
    cluster_stats = extract_cluster_features(driving, energy, labels)
    
    # 7. 绘制雷达图
    plot_radar_k4(cluster_stats)
    
    # 8. 绘制热力图
    plot_heatmap_k4(cluster_stats)
    
    # 9. 绘制柱状图
    plot_feature_bars_k4(cluster_stats)
    
    # 10. 聚类解释
    interpretations = interpret_clusters_k4(cluster_stats)
    
    # 11. 生成报告
    generate_report_k4(cluster_stats, labels, sil, cv, interpretations)
    
    # 12. 保存特征和标签
    np.save('./results/features_k4.npy', features)
    np.save('./results/labels_k4.npy', labels)
    
    print("\n" + "="*70)
    print("✅ K=4 聚类分析完成！")
    print("="*70)
    print("\n📁 生成的文件（所有带_k4后缀）:")
    print("   - cluster_radar_k4.png        (雷达图)")
    print("   - cluster_heatmap_k4.png      (热力图)")
    print("   - cluster_features_bars_k4.png (柱状图)")
    print("   - clustering_report_k4.json   (详细报告)")
    print("   - cluster_features_k4.csv     (特征表格)")
    print("   - features_k4.npy             (提取的特征)")
    print("   - labels_k4.npy               (聚类标签)")
    print("   - gru_model_k4.pth            (训练的模型)")
    print("\n   不会覆盖之前的任何文件！")
    print("="*70)


if __name__ == "__main__":
    main()