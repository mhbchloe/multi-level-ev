"""
======================================================================
🎯 GRU Clustering Analysis - K=4 Version (English)
======================================================================
Generate K=4 clustering results and radar charts
All output files with _k4 suffix, won't overwrite previous results
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

# Use default font (no Chinese)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("🎯 GRU Clustering Analysis - K=4 Version")
print("="*70)


# ==================== Dataset ====================
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


# ==================== GRU Model ====================
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


# ==================== Load Data ====================
def load_data(data_dir, max_samples=100000):
    """Load data"""
    print("\n" + "="*70)
    print("📂 Loading Data")
    print("="*70)
    
    data_path = Path(data_dir)
    
    driving = np.load(data_path / 'driving_sequences.npy', allow_pickle=True)
    energy = np.load(data_path / 'energy_sequences.npy', allow_pickle=True)
    lengths = np.load(data_path / 'seq_lengths.npy')
    
    # Filter by length
    valid_mask = (lengths >= 10) & (lengths <= 1000)
    driving = driving[valid_mask]
    energy = energy[valid_mask]
    lengths = lengths[valid_mask]
    
    # Sampling
    if len(driving) > max_samples:
        indices = np.random.choice(len(driving), max_samples, replace=False)
        driving = driving[indices]
        energy = energy[indices]
        lengths = lengths[indices]
    
    print(f"✅ Data loaded:")
    print(f"   Samples: {len(driving):,}")
    print(f"   Driving features: {driving[0].shape[1]}D")
    print(f"   Energy features: {energy[0].shape[1]}D")
    
    return driving, energy, lengths


# ==================== Train or Load Model ====================
def train_or_load_gru(model, train_loader, model_path='./results/gru_model_k4.pth', epochs=10):
    """Train GRU or load existing model"""
    
    # Try to load existing model
    if Path(model_path).exists():
        print(f"\n✅ Found trained model: {model_path}")
        print("   Loading model...")
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    
    # Otherwise train new model
    print("\n" + "="*70)
    print("🚀 Training GRU Model")
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
    
    print(f"\n✅ Training completed | Best loss: {best_loss:.4f}")
    
    # Save model
    torch.save(model.state_dict(), model_path)
    print(f"💾 Model saved: {model_path}")
    
    return model


# ==================== Extract Features ====================
def extract_features(model, loader):
    """Extract GRU features"""
    print("\n🔍 Extracting features...")
    
    model.eval()
    all_features = []
    all_indices = []
    
    with torch.no_grad():
        for driving, energy, lengths, sorted_idx in tqdm(loader, desc="Extracting", leave=False):
            driving = driving.to(device)
            energy = energy.to(device)
            _, feat = model(driving, energy, lengths)
            all_features.append(feat.cpu().numpy())
            all_indices.append(sorted_idx.numpy())
    
    features = np.vstack(all_features)
    indices = np.concatenate(all_indices)
    unsort_indices = np.argsort(indices)
    features = features[unsort_indices]
    
    print(f"✅ Features extracted: {features.shape}")
    return features


# ==================== K=4 Clustering ====================
def perform_clustering_k4(features):
    """Perform K=4 clustering"""
    print("\n" + "="*70)
    print("🎯 K=4 Clustering")
    print("="*70)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=20)
    labels = kmeans.fit_predict(features)
    
    sil = silhouette_score(features, labels)
    unique, counts = np.unique(labels, return_counts=True)
    cv = np.std(counts) / np.mean(counts)
    
    print(f"\n✅ Clustering completed:")
    print(f"   Silhouette: {sil:.3f}")
    print(f"   CV: {cv:.3f}")
    print(f"\n   Cluster distribution:")
    for cluster_id, count in zip(unique, counts):
        pct = count / len(labels) * 100
        print(f"      Cluster {cluster_id}: {count:6,} ({pct:5.1f}%)")
    
    return labels, kmeans, sil, cv


# ==================== Extract Cluster Features ====================
def extract_cluster_features(driving_seqs, energy_seqs, labels):
    """Extract feature statistics for each cluster"""
    print("\n" + "="*70)
    print("📊 Extracting Cluster Features")
    print("="*70)
    
    cluster_stats = []
    
    for cluster_id in range(4):
        print(f"\nAnalyzing Cluster {cluster_id}...")
        
        cluster_mask = (labels == cluster_id)
        cluster_driving = driving_seqs[cluster_mask]
        cluster_energy = energy_seqs[cluster_mask]
        
        stats = {}
        
        # Driving features
        all_spd = np.concatenate([seq[:, 0] for seq in cluster_driving])
        all_acc = np.concatenate([seq[:, 1] for seq in cluster_driving])
        
        stats['Avg Speed'] = np.mean(all_spd)
        stats['Max Speed'] = np.percentile(all_spd, 95)
        stats['Speed Std'] = np.std(all_spd)
        stats['Idle Ratio'] = np.mean(all_spd < 1) * 100
        stats['High Speed Ratio'] = np.mean(all_spd > 60) * 100
        
        stats['Avg Accel'] = np.abs(np.mean(all_acc))
        stats['Accel Std'] = np.std(all_acc)
        stats['Harsh Accel %'] = np.mean(all_acc > 1.5) * 100
        stats['Harsh Decel %'] = np.mean(all_acc < -1.5) * 100
        
        # Energy features
        all_soc = np.concatenate([seq[:, 0] for seq in cluster_energy])
        all_v = np.concatenate([seq[:, 1] for seq in cluster_energy])
        all_i = np.concatenate([seq[:, 2] for seq in cluster_energy])
        
        stats['Avg SOC'] = np.mean(all_soc)
        stats['SOC Drop Rate'] = np.mean([seq[0, 0] - seq[-1, 0] for seq in cluster_energy if len(seq) > 1])
        
        stats['Avg Voltage'] = np.mean(all_v)
        stats['Avg Current'] = np.abs(np.mean(all_i))
        stats['Avg Power'] = np.mean(np.abs(all_v * all_i))
        
        stats['Trip Length'] = np.mean([len(seq) for seq in cluster_driving])
        
        cluster_stats.append(stats)
        
        # Print key features
        print(f"   Avg Speed: {stats['Avg Speed']:.1f} km/h")
        print(f"   Idle Ratio: {stats['Idle Ratio']:.1f}%")
        print(f"   High Speed Ratio: {stats['High Speed Ratio']:.1f}%")
        print(f"   Accel Std: {stats['Accel Std']:.2f}")
    
    return cluster_stats


# ==================== Plot Radar Chart (K=4) ====================
def plot_radar_k4(cluster_stats):
    """Plot K=4 radar chart"""
    print("\n" + "="*70)
    print("🎨 Plotting K=4 Radar Chart")
    print("="*70)
    
    # Select features to display
    feature_keys = [
        'Avg Speed',
        'Speed Std',
        'High Speed Ratio',
        'Idle Ratio',
        'Accel Std',
        'Harsh Accel %',
        'Harsh Decel %',
        'SOC Drop Rate',
        'Avg Power',
        'Trip Length'
    ]
    
    # Extract data
    data = []
    for stats in cluster_stats:
        data.append([stats[key] for key in feature_keys])
    
    data = np.array(data)
    
    # Normalize
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data.T).T
    
    # Plot
    N = len(feature_keys)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(14, 14), subplot_kw=dict(projection='polar'))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    labels = ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3']
    
    for cluster_id in range(4):
        values = data_normalized[cluster_id].tolist()
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=3, 
               label=labels[cluster_id], color=colors[cluster_id], 
               markersize=10)
        ax.fill(angles, values, alpha=0.20, color=colors[cluster_id])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_keys, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=11, color='gray')
    ax.grid(True, linestyle='--', alpha=0.7, linewidth=1.5)
    
    ax.set_title('Driving Behavior Clustering Feature Comparison (K=4)', 
                fontsize=18, fontweight='bold', pad=35)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), 
             fontsize=13, framealpha=0.95, shadow=True)
    
    plt.tight_layout()
    plt.savefig('./results/cluster_radar_k4.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Radar chart saved: ./results/cluster_radar_k4.png")
    
    return data, data_normalized


# ==================== Plot Heatmap ====================
def plot_heatmap_k4(cluster_stats):
    """Plot feature heatmap"""
    print("\n📊 Plotting feature heatmap...")
    
    feature_keys = [
        'Avg Speed', 'Speed Std', 'High Speed Ratio', 'Idle Ratio',
        'Accel Std', 'Harsh Accel %', 'Harsh Decel %',
        'SOC Drop Rate', 'Avg Power', 'Trip Length'
    ]
    
    data = []
    for stats in cluster_stats:
        data.append([stats[key] for key in feature_keys])
    
    df = pd.DataFrame(data, columns=feature_keys, 
                     index=['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3'])
    
    # Normalize
    df_normalized = (df - df.min()) / (df.max() - df.min())
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.heatmap(df_normalized, annot=True, fmt='.2f', cmap='YlOrRd', 
               linewidths=2, cbar_kws={'label': 'Normalized Value'}, 
               ax=ax, vmin=0, vmax=1)
    
    ax.set_title('Cluster Feature Heatmap (K=4)', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Cluster', fontsize=13, fontweight='bold')
    ax.set_xlabel('Feature Dimension', fontsize=13, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('./results/cluster_heatmap_k4.png', dpi=300, bbox_inches='tight')
    print(f"✅ Heatmap saved: ./results/cluster_heatmap_k4.png")


# ==================== Plot Feature Bars ====================
def plot_feature_bars_k4(cluster_stats):
    """Plot feature bar charts"""
    print("\n📊 Plotting feature bar charts...")
    
    feature_keys = [
        'Avg Speed', 'Speed Std', 'High Speed Ratio', 'Idle Ratio',
        'Accel Std', 'Harsh Accel %', 'Harsh Decel %',
        'SOC Drop Rate', 'Avg Power', 'Trip Length'
    ]
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    axes = axes.flatten()
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    for idx, feature in enumerate(feature_keys):
        ax = axes[idx]
        
        values = [stats[feature] for stats in cluster_stats]
        x = np.arange(4)
        
        bars = ax.bar(x, values, color=colors, alpha=0.85, edgecolor='black', linewidth=2)
        
        ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
        ax.set_ylabel(feature, fontsize=12, fontweight='bold')
        ax.set_title(feature, fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['C0', 'C1', 'C2', 'C3'])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}', ha='center', va='bottom', 
                   fontsize=10, fontweight='bold')
    
    # Hide extra subplots
    for idx in range(len(feature_keys), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Cluster Feature Detailed Comparison (K=4)', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig('./results/cluster_features_bars_k4.png', dpi=300, bbox_inches='tight')
    print(f"✅ Bar chart saved: ./results/cluster_features_bars_k4.png")


# ==================== Interpret Clusters ====================
def interpret_clusters_k4(cluster_stats):
    """Interpret each cluster's characteristics"""
    print("\n" + "="*70)
    print("💡 Cluster Interpretation (K=4)")
    print("="*70)
    
    interpretations = []
    
    for cluster_id, stats in enumerate(cluster_stats):
        print(f"\n📍 Cluster {cluster_id}:")
        interpretation = []
        
        # Speed characteristics
        if stats['Avg Speed'] > 50:
            msg = f"   🚗 Highway Driving: High avg speed ({stats['Avg Speed']:.1f} km/h)"
            print(msg)
            interpretation.append("Highway")
        elif stats['Avg Speed'] < 25:
            msg = f"   🚦 Urban Congestion: Low avg speed ({stats['Avg Speed']:.1f} km/h)"
            print(msg)
            interpretation.append("Urban")
        else:
            msg = f"   🛣️  Mixed Conditions: Medium speed ({stats['Avg Speed']:.1f} km/h)"
            print(msg)
            interpretation.append("Mixed")
        
        # Idle
        if stats['Idle Ratio'] > 30:
            msg = f"   ⏸️  Frequent Stops: {stats['Idle Ratio']:.1f}% idle"
            print(msg)
            interpretation.append("Frequent Stops")
        
        # High speed
        if stats['High Speed Ratio'] > 30:
            msg = f"   🛣️  Highway Segments: {stats['High Speed Ratio']:.1f}% high speed"
            print(msg)
            interpretation.append("Highway")
        
        # Acceleration behavior
        if stats['Harsh Accel %'] > 8 or stats['Harsh Decel %'] > 8:
            msg = f"   ⚡ Aggressive: {stats['Harsh Accel %']:.1f}% accel, {stats['Harsh Decel %']:.1f}% decel"
            print(msg)
            interpretation.append("Aggressive")
        elif stats['Accel Std'] < 0.6:
            msg = f"   🍃 Smooth Driving: Low accel variation ({stats['Accel Std']:.2f})"
            print(msg)
            interpretation.append("Smooth")
        
        # Energy consumption
        if stats['SOC Drop Rate'] > 2.5:
            msg = f"   🔋 High Consumption: {stats['SOC Drop Rate']:.2f}%/trip"
            print(msg)
            interpretation.append("High Energy")
        elif stats['SOC Drop Rate'] < 1.5:
            msg = f"   🔋 Eco Driving: {stats['SOC Drop Rate']:.2f}%/trip"
            print(msg)
            interpretation.append("Eco")
        
        # Trip length
        if stats['Trip Length'] > 120:
            msg = f"   📏 Long Trips: Avg {stats['Trip Length']:.0f} points"
            print(msg)
            interpretation.append("Long")
        elif stats['Trip Length'] < 50:
            msg = f"   📏 Short Trips: Avg {stats['Trip Length']:.0f} points"
            print(msg)
            interpretation.append("Short")
        
        interpretations.append(' + '.join(interpretation))
    
    return interpretations


# ==================== Generate Report ====================
def generate_report_k4(cluster_stats, labels, sil, cv, interpretations):
    """Generate detailed report"""
    print("\n📄 Generating detailed report...")
    
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
    
    # Save JSON
    with open('./results/clustering_report_k4.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Save CSV
    df = pd.DataFrame(cluster_stats)
    df.index = ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3']
    df.to_csv('./results/cluster_features_k4.csv', encoding='utf-8-sig')
    
    print(f"✅ Report saved:")
    print(f"   JSON: ./results/clustering_report_k4.json")
    print(f"   CSV:  ./results/cluster_features_k4.csv")


# ==================== Main ====================
def main():
    np.random.seed(42)
    torch.manual_seed(42)
    
    Path('./results').mkdir(exist_ok=True)
    
    print(f"\n🎯 Fixed K=4 Clustering Analysis")
    print(f"   All output files with _k4 suffix\n")
    
    # 1. Load data
    driving, energy, lengths = load_data('./results/temporal_soc_full', max_samples=100000)
    
    # 2. Create data loader
    dataset = VariableLengthDataset(driving, energy, lengths)
    train_loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        collate_fn=collate_variable_length,
        num_workers=0
    )
    
    # 3. Create and train/load GRU model
    model = DualChannelGRU(
        driving_dim=driving[0].shape[1],
        energy_dim=energy[0].shape[1],
        latent_dim=16
    ).to(device)
    
    model = train_or_load_gru(model, train_loader, epochs=10)
    
    # 4. Extract features
    features = extract_features(model, train_loader)
    
    # 5. K=4 clustering
    labels, kmeans, sil, cv = perform_clustering_k4(features)
    
    # 6. Extract cluster features
    cluster_stats = extract_cluster_features(driving, energy, labels)
    
    # 7. Plot radar chart
    plot_radar_k4(cluster_stats)
    
    # 8. Plot heatmap
    plot_heatmap_k4(cluster_stats)
    
    # 9. Plot bar charts
    plot_feature_bars_k4(cluster_stats)
    
    # 10. Interpret clusters
    interpretations = interpret_clusters_k4(cluster_stats)
    
    # 11. Generate report
    generate_report_k4(cluster_stats, labels, sil, cv, interpretations)
    
    # 12. Save features and labels
    np.save('./results/features_k4.npy', features)
    np.save('./results/labels_k4.npy', labels)
    
    print("\n" + "="*70)
    print("✅ K=4 Clustering Analysis Completed!")
    print("="*70)
    print("\n📁 Generated files (all with _k4 suffix):")
    print("   - cluster_radar_k4.png        (Radar Chart)")
    print("   - cluster_heatmap_k4.png      (Heatmap)")
    print("   - cluster_features_bars_k4.png (Bar Charts)")
    print("   - clustering_report_k4.json   (Detailed Report)")
    print("   - cluster_features_k4.csv     (Feature Table)")
    print("   - features_k4.npy             (Extracted Features)")
    print("   - labels_k4.npy               (Cluster Labels)")
    print("   - gru_model_k4.pth            (Trained Model)")
    print("\n   Won't overwrite any previous files!")
    print("="*70)


if __name__ == "__main__":
    main()