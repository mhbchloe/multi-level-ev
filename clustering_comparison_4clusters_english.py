"""
Clustering Algorithm Comparison: K-means vs GMM vs Deep Embedded Clustering
4 Clusters, English Version
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score, 
    davies_bouldin_score, 
    calinski_harabasz_score,
    normalized_mutual_info_score
)
import time
import warnings
warnings.filterwarnings('ignore')

# Deep Clustering
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not installed, skipping Deep Clustering")

# English font settings
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

print("="*70)
print("🔬 Clustering Algorithm Comparison (4 Clusters)")
print("="*70)

# ============ 1. Data Preparation ============
print("\n📂 Loading data...")

df = pd.read_csv('./results/event_table.csv')
print(f"✅ Loaded {len(df):,} discharge events")

# Feature selection
feature_cols = [
    'speed_mean', 'speed_std', 'speed_max', 'speed_median', 'speed_p95',
    'idle_ratio', 'low_speed_ratio', 'medium_speed_ratio', 'high_speed_ratio',
    'accel_mean', 'accel_abs_mean', 'accel_std',
    'accel_positive_ratio', 'accel_negative_ratio',
    'harsh_accel_count', 'harsh_brake_count',
    'soc_drop', 'soc_mean', 'soc_std',
    'power_mean', 'power_std', 'power_p95',
    'energy_consumption_kwh', 'efficiency_kwh_per_km', 'efficiency_soc_per_km',
    'duration_seconds', 'distance_km', 'speed_change_rate',
    'idle_count', 'num_points'
]

available_features = [col for col in feature_cols if col in df.columns]
X = df[available_features].copy()
X = X.replace([np.inf, -np.inf], np.nan).dropna()

print(f"✅ Features: {len(available_features)}, Samples: {len(X):,}")

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============ 2. Algorithm Implementation ============

# === 2.1 K-means ===
class KMeansClusterer:
    def __init__(self, n_clusters=4, random_state=42):
        self.name = "K-means"
        self.model = KMeans(
            n_clusters=n_clusters,
            init='k-means++',
            n_init=10,
            random_state=random_state
        )
        self.n_clusters = n_clusters
    
    def fit_predict(self, X):
        start_time = time.time()
        labels = self.model.fit_predict(X)
        train_time = time.time() - start_time
        return labels, train_time
    
    def get_probabilities(self, X):
        labels = self.model.predict(X)
        n_samples = len(labels)
        probs = np.zeros((n_samples, self.n_clusters))
        probs[np.arange(n_samples), labels] = 1.0
        return probs

# === 2.2 GMM ===
class GMMClusterer:
    def __init__(self, n_clusters=4, random_state=42):
        self.name = "GMM"
        self.n_clusters = n_clusters
        self.model = GaussianMixture(
            n_components=n_clusters,
            covariance_type='full',
            n_init=10,
            random_state=random_state
        )
    
    def fit_predict(self, X):
        start_time = time.time()
        self.model.fit(X)
        labels = self.model.predict(X)
        train_time = time.time() - start_time
        return labels, train_time
    
    def get_probabilities(self, X):
        return self.model.predict_proba(X)

# === 2.3 Deep Embedded Clustering ===
if TORCH_AVAILABLE:
    class AutoEncoder(nn.Module):
        def __init__(self, input_dim, encoding_dim=10):
            super(AutoEncoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, encoding_dim)
            )
            self.decoder = nn.Sequential(
                nn.Linear(encoding_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, input_dim)
            )
        
        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return encoded, decoded
    
    class DeepClusterer:
        def __init__(self, n_clusters=4, encoding_dim=10, random_state=42):
            self.name = "Deep Clustering"
            self.n_clusters = n_clusters
            self.encoding_dim = encoding_dim
            torch.manual_seed(random_state)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        def pretrain_autoencoder(self, X, epochs=50, batch_size=256):
            input_dim = X.shape[1]
            self.autoencoder = AutoEncoder(input_dim, self.encoding_dim).to(self.device)
            
            dataset = TensorDataset(
                torch.FloatTensor(X),
                torch.FloatTensor(X)
            )
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            optimizer = optim.Adam(self.autoencoder.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            self.autoencoder.train()
            for epoch in range(epochs):
                total_loss = 0
                for batch_x, batch_y in loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    encoded, decoded = self.autoencoder(batch_x)
                    loss = criterion(decoded, batch_y)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                if (epoch + 1) % 10 == 0:
                    print(f"      Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")
        
        def fit_predict(self, X, pretrain_epochs=50):
            start_time = time.time()
            
            print(f"   {self.name}: Pretraining autoencoder...")
            self.pretrain_autoencoder(X, epochs=pretrain_epochs)
            
            print(f"   {self.name}: Extracting embeddings...")
            self.autoencoder.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.device)
                embeddings, _ = self.autoencoder(X_tensor)
                embeddings = embeddings.cpu().numpy()
            
            print(f"   {self.name}: Clustering in embedding space...")
            self.kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
            labels = self.kmeans.fit_predict(embeddings)
            
            train_time = time.time() - start_time
            self.embeddings = embeddings
            return labels, train_time
        
        def get_probabilities(self, X):
            distances = np.linalg.norm(
                self.embeddings[:, np.newaxis] - self.kmeans.cluster_centers_, 
                axis=2
            )
            probs = np.exp(-distances) / np.exp(-distances).sum(axis=1, keepdims=True)
            return probs

# ============ 3. Run Experiments ============
print("\n🔬 Running experiments...")

algorithms = [
    KMeansClusterer(n_clusters=4),
    GMMClusterer(n_clusters=4),
]

if TORCH_AVAILABLE:
    algorithms.append(DeepClusterer(n_clusters=4, encoding_dim=10))

results = {}

for algo in algorithms:
    print(f"\n{'='*70}")
    print(f"🔧 {algo.name}")
    print(f"{'='*70}")
    
    if hasattr(algo, 'pretrain_autoencoder'):
        labels, train_time = algo.fit_predict(X_scaled, pretrain_epochs=30)
    else:
        labels, train_time = algo.fit_predict(X_scaled)
    
    # Evaluation metrics
    sil_score = silhouette_score(X_scaled, labels)
    db_score = davies_bouldin_score(X_scaled, labels)
    ch_score = calinski_harabasz_score(X_scaled, labels)
    
    # Cluster distribution
    unique, counts = np.unique(labels, return_counts=True)
    cluster_dist = dict(zip(unique, counts))
    
    results[algo.name] = {
        'labels': labels,
        'train_time': train_time,
        'silhouette': sil_score,
        'davies_bouldin': db_score,
        'calinski_harabasz': ch_score,
        'cluster_distribution': cluster_dist,
        'probabilities': algo.get_probabilities(X_scaled)
    }
    
    print(f"\n📊 Results:")
    print(f"   Training Time: {train_time:.2f}s")
    print(f"   Silhouette Score: {sil_score:.4f}")
    print(f"   Davies-Bouldin Index: {db_score:.4f}")
    print(f"   Calinski-Harabasz Score: {ch_score:.1f}")
    print(f"\n   Cluster Distribution:")
    for c, count in cluster_dist.items():
        print(f"      Cluster {c}: {count:,} ({count/len(labels)*100:.1f}%)")

# ============ 4. Stability Test ============
print(f"\n{'='*70}")
print(f"🔄 Stability Test (10 runs)")
print(f"{'='*70}")

stability_results = {algo.name: [] for algo in algorithms}

for run in range(10):
    for algo in algorithms:
        if algo.name == "Deep Clustering":
            algo_run = DeepClusterer(n_clusters=4, random_state=run)
            labels, _ = algo_run.fit_predict(X_scaled, pretrain_epochs=20)
        else:
            algo_run = type(algo)(n_clusters=4, random_state=run)
            labels, _ = algo_run.fit_predict(X_scaled)
        
        stability_results[algo.name].append(labels)

nmi_scores = {}
for algo_name, labels_list in stability_results.items():
    nmi = [normalized_mutual_info_score(labels_list[0], labels_list[i]) 
           for i in range(1, len(labels_list))]
    nmi_scores[algo_name] = np.mean(nmi)
    print(f"   {algo_name}: NMI = {np.mean(nmi):.4f} (±{np.std(nmi):.4f})")

# ============ 5. Visualization ============
print(f"\n📈 Generating visualizations...")

fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.3)

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 4-cluster color palette
colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']  # Blue, Green, Red, Orange
cluster_names = ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3']

# === Row 1: PCA Visualization ===
for idx, (algo_name, result) in enumerate(results.items()):
    ax = fig.add_subplot(gs[0, idx])
    labels = result['labels']
    
    for i in range(4):
        mask = labels == i
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   c=colors[i], label=cluster_names[i], alpha=0.5, s=12)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=10)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=10)
    ax.set_title(f'{algo_name}\n(Silhouette: {result["silhouette"]:.3f})', 
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(alpha=0.3)

# === Row 2: Evaluation Metrics ===
algo_names = list(results.keys())

# Metric 1: Silhouette Score
ax = fig.add_subplot(gs[1, 0])
sil_scores = [results[name]['silhouette'] for name in algo_names]
bars = ax.bar(range(len(algo_names)), sil_scores, 
              color=colors[:len(algo_names)], alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_xticks(range(len(algo_names)))
ax.set_xticklabels(algo_names, rotation=15, ha='right', fontsize=9)
ax.set_ylabel('Silhouette Score', fontsize=10, fontweight='bold')
ax.set_title('Silhouette Score (Higher is Better)', fontsize=11, fontweight='bold')
ax.grid(alpha=0.3, axis='y')
ax.set_ylim([0, max(sil_scores) * 1.2])

for i, (bar, score) in enumerate(zip(bars, sil_scores)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
            f'{score:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Metric 2: Davies-Bouldin Index
ax = fig.add_subplot(gs[1, 1])
db_scores = [results[name]['davies_bouldin'] for name in algo_names]
bars = ax.bar(range(len(algo_names)), db_scores, 
              color=colors[:len(algo_names)], alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_xticks(range(len(algo_names)))
ax.set_xticklabels(algo_names, rotation=15, ha='right', fontsize=9)
ax.set_ylabel('Davies-Bouldin Index', fontsize=10, fontweight='bold')
ax.set_title('Davies-Bouldin Index (Lower is Better)', fontsize=11, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

for i, (bar, score) in enumerate(zip(bars, db_scores)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
            f'{score:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Metric 3: Training Time
ax = fig.add_subplot(gs[1, 2])
train_times = [results[name]['train_time'] for name in algo_names]
bars = ax.bar(range(len(algo_names)), train_times, 
              color=colors[:len(algo_names)], alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_xticks(range(len(algo_names)))
ax.set_xticklabels(algo_names, rotation=15, ha='right', fontsize=9)
ax.set_ylabel('Training Time (seconds)', fontsize=10, fontweight='bold')
ax.set_title('Computational Efficiency', fontsize=11, fontweight='bold')
ax.set_yscale('log')
ax.grid(alpha=0.3, axis='y', which='both')

for i, (bar, time_val) in enumerate(zip(bars, train_times)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.3, 
            f'{time_val:.1f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Metric 4: Stability (NMI)
ax = fig.add_subplot(gs[1, 3])
nmi_values = [nmi_scores[name] for name in algo_names]
bars = ax.bar(range(len(algo_names)), nmi_values, 
              color=colors[:len(algo_names)], alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_xticks(range(len(algo_names)))
ax.set_xticklabels(algo_names, rotation=15, ha='right', fontsize=9)
ax.set_ylabel('NMI (Normalized Mutual Info)', fontsize=10, fontweight='bold')
ax.set_title('Stability (Higher is Better)', fontsize=11, fontweight='bold')
ax.set_ylim([0, 1.1])
ax.grid(alpha=0.3, axis='y')

for i, (bar, nmi) in enumerate(zip(bars, nmi_values)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03, 
            f'{nmi:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# === Row 3: Cluster Distribution ===
for idx, (algo_name, result) in enumerate(results.items()):
    if idx >= 3:
        break
    
    ax = fig.add_subplot(gs[2, idx])
    
    cluster_dist = result['cluster_distribution']
    clusters = sorted(cluster_dist.keys())
    counts = [cluster_dist[c] for c in clusters]
    percentages = [count/sum(counts)*100 for count in counts]
    
    bars = ax.bar(clusters, percentages, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Cluster ID', fontsize=10, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=10, fontweight='bold')
    ax.set_title(f'{algo_name} - Cluster Distribution', fontsize=11, fontweight='bold')
    ax.set_xticks(clusters)
    ax.set_xticklabels([f'C{c}' for c in clusters])
    ax.grid(alpha=0.3, axis='y')
    
    for bar, pct, count in zip(bars, percentages, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{pct:.1f}%\n({count:,})', ha='center', va='bottom', fontsize=8)

# === Row 3, Col 4: Summary Table ===
ax = fig.add_subplot(gs[2, 3])
ax.axis('off')

comparison_data = []
for name in algo_names:
    comparison_data.append([
        name,
        f"{results[name]['silhouette']:.3f}",
        f"{results[name]['davies_bouldin']:.3f}",
        f"{nmi_scores[name]:.3f}",
        f"{results[name]['train_time']:.1f}s"
    ])

table = ax.table(
    cellText=comparison_data,
    colLabels=['Algorithm', 'Silhouette↑', 'DB Index↓', 'Stability↑', 'Time'],
    cellLoc='center',
    loc='center',
    colWidths=[0.35, 0.15, 0.15, 0.15, 0.2]
)

table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 2)

for i in range(5):
    table[(0, i)].set_facecolor('#34495e')
    table[(0, i)].set_text_props(weight='bold', color='white')

for i in range(1, len(comparison_data) + 1):
    for j in range(5):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#ecf0f1')

ax.set_title('Quantitative Comparison', fontsize=12, fontweight='bold', pad=20)

# === Row 4: Probability Distribution (Soft Clustering) ===
for idx, (algo_name, result) in enumerate(results.items()):
    if idx >= 3:
        break
    
    ax = fig.add_subplot(gs[3, idx])
    probs = result['probabilities']
    
    for i in range(4):
        prob_i = probs[:, i]
        ax.hist(prob_i, bins=50, alpha=0.6, label=cluster_names[i], color=colors[i])
    
    ax.set_xlabel('Membership Probability', fontsize=10, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax.set_title(f'{algo_name} - Probability Distribution', fontsize=11, fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis='y')

# === Row 4, Col 4: Cluster Interpretation ===
ax = fig.add_subplot(gs[3, 3])
ax.axis('off')

# Analyze cluster characteristics for the best algorithm
best_algo = max(results.items(), key=lambda x: x[1]['silhouette'])
best_labels = best_algo[1]['labels']

cluster_features = []
for i in range(4):
    cluster_data = df.iloc[X.index[best_labels == i]]
    
    avg_speed = cluster_data['speed_mean'].mean() if 'speed_mean' in cluster_data.columns else 0
    avg_harsh = (cluster_data.get('harsh_accel_count', pd.Series([0])).mean() + 
                 cluster_data.get('harsh_brake_count', pd.Series([0])).mean())
    avg_soc_drop = cluster_data.get('soc_drop', pd.Series([0])).mean()
    
    cluster_features.append([
        f'Cluster {i}',
        f'{avg_speed:.1f} km/h',
        f'{avg_harsh:.2f}',
        f'{avg_soc_drop:.1f}%'
    ])

feature_table = ax.table(
    cellText=cluster_features,
    colLabels=['Cluster', 'Avg Speed', 'Harsh Events', 'SOC Drop'],
    cellLoc='center',
    loc='center',
    colWidths=[0.2, 0.25, 0.25, 0.3]
)

feature_table.auto_set_font_size(False)
feature_table.set_fontsize(8)
feature_table.scale(1, 2)

for i in range(4):
    feature_table[(0, i)].set_facecolor('#34495e')
    feature_table[(0, i)].set_text_props(weight='bold', color='white')

for i in range(1, 5):
    feature_table[(i, 0)].set_facecolor(colors[i-1])
    feature_table[(i, 0)].set_text_props(weight='bold', color='white')

ax.set_title(f'Cluster Characteristics ({best_algo[0]})', fontsize=12, fontweight='bold', pad=20)

plt.suptitle('Clustering Algorithm Comparison: K-means vs GMM vs Deep Clustering (4 Clusters)', 
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig('./results/clustering_comparison_4clusters_english.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: clustering_comparison_4clusters_english.png")

# ============ 6. Detailed Report ============
print(f"\n💾 Saving detailed report...")

report = []
report.append("="*70)
report.append("Clustering Algorithm Comparison Report (4 Clusters)")
report.append("="*70)
report.append(f"\nDataset Information:")
report.append(f"  Samples: {len(X):,}")
report.append(f"  Features: {len(available_features)}")
report.append(f"  Number of Clusters: 4")

for algo_name, result in results.items():
    report.append(f"\n{'-'*70}")
    report.append(f"Algorithm: {algo_name}")
    report.append(f"{'-'*70}")
    report.append(f"  Training Time: {result['train_time']:.2f} seconds")
    report.append(f"  Silhouette Score: {result['silhouette']:.4f}")
    report.append(f"  Davies-Bouldin Index: {result['davies_bouldin']:.4f}")
    report.append(f"  Calinski-Harabasz Score: {result['calinski_harabasz']:.1f}")
    report.append(f"  Stability (NMI): {nmi_scores[algo_name]:.4f}")
    report.append(f"\n  Cluster Distribution:")
    for c, count in result['cluster_distribution'].items():
        report.append(f"    Cluster {c}: {count:,} ({count/len(X)*100:.1f}%)")

report.append(f"\n{'='*70}")
report.append(f"Algorithm Selection Recommendation:")
report.append(f"{'='*70}")

best_sil = max(results.items(), key=lambda x: x[1]['silhouette'])
fastest = min(results.items(), key=lambda x: x[1]['train_time'])
most_stable = max(nmi_scores.items(), key=lambda x: x[1])

report.append(f"  Best Clustering Quality: {best_sil[0]} (Silhouette: {best_sil[1]['silhouette']:.4f})")
report.append(f"  Fastest Training: {fastest[0]} ({fastest[1]['train_time']:.2f}s)")
report.append(f"  Most Stable: {most_stable[0]} (NMI: {most_stable[1]:.4f})")

report.append(f"\n{'='*70}")
report.append(f"Cluster Interpretation (Based on {best_sil[0]}):")
report.append(f"{'='*70}")

for i in range(4):
    cluster_data = df.iloc[X.index[best_labels == i]]
    
    report.append(f"\nCluster {i} (n={len(cluster_data):,}):")
    
    if 'speed_mean' in cluster_data.columns:
        report.append(f"  Average Speed: {cluster_data['speed_mean'].mean():.2f} km/h")
    if 'harsh_accel_count' in cluster_data.columns:
        report.append(f"  Harsh Acceleration: {cluster_data['harsh_accel_count'].mean():.2f}")
    if 'harsh_brake_count' in cluster_data.columns:
        report.append(f"  Harsh Braking: {cluster_data['harsh_brake_count'].mean():.2f}")
    if 'soc_drop' in cluster_data.columns:
        report.append(f"  SOC Drop: {cluster_data['soc_drop'].mean():.2f}%")
    if 'efficiency_kwh_per_km' in cluster_data.columns:
        report.append(f"  Energy Efficiency: {cluster_data['efficiency_kwh_per_km'].mean():.4f} kWh/km")

report_text = "\n".join(report)
print(report_text)

with open('./results/clustering_comparison_4clusters_report.txt', 'w', encoding='utf-8') as f:
    f.write(report_text)

print(f"\n💾 Saved: clustering_comparison_4clusters_report.txt")

print(f"\n{'='*70}")
print(f"✅ Comparison Complete!")
print(f"{'='*70}")
print(f"\n📁 Generated Files:")
print(f"   1. clustering_comparison_4clusters_english.png")
print(f"   2. clustering_comparison_4clusters_report.txt")
print(f"\n💡 Recommendation:")
if best_sil[0] == "K-means":
    print(f"   K-means performs best on your data. Use it for final analysis.")
elif best_sil[0] == "GMM":
    print(f"   GMM shows better performance. Consider using soft clustering results.")
else:
    print(f"   {best_sil[0]} achieves best quality but requires more computation.")
print(f"{'='*70}")