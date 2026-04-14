import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

class BaselineModels:
    """传统聚类算法基线模型"""
    
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.models = {}
        self.labels = {}
        self.metrics = {}
    
    def prepare_data(self, features_df, drop_cols=['event_id', 'vehicle_id']):
        """准备数据"""
        X = features_df.drop(drop_cols, axis=1, errors='ignore')
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, X.columns
    
    def train_kmeans(self, X):
        """K-Means聚类"""
        print("🔵 训练 K-Means...")
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        self.models['kmeans'] = kmeans
        self.labels['kmeans'] = labels
        
        metrics = self.evaluate_clustering(X, labels)
        self.metrics['kmeans'] = metrics
        
        print(f"   ✅ Silhouette: {metrics['silhouette']:.3f}")
        return labels, metrics
    
    def train_dbscan(self, X, eps=0.5, min_samples=5):
        """DBSCAN聚类"""
        print("🟢 训练 DBSCAN...")
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        self.models['dbscan'] = dbscan
        self.labels['dbscan'] = labels
        
        print(f"   发现 {n_clusters} 个簇, {n_noise} 个噪声点")
        
        if n_clusters > 1:
            # 过滤噪声点
            mask = labels != -1
            metrics = self.evaluate_clustering(X[mask], labels[mask])
            self.metrics['dbscan'] = metrics
            print(f"   ✅ Silhouette: {metrics['silhouette']:.3f}")
        else:
            self.metrics['dbscan'] = {'silhouette': -1, 'ch_score': 0, 'db_score': 999}
        
        return labels, self.metrics['dbscan']
    
    def train_gmm(self, X):
        """高斯混合模型"""
        print("🟣 训练 GMM...")
        gmm = GaussianMixture(n_components=self.n_clusters, random_state=42)
        labels = gmm.fit_predict(X)
        
        self.models['gmm'] = gmm
        self.labels['gmm'] = labels
        
        metrics = self.evaluate_clustering(X, labels)
        self.metrics['gmm'] = metrics
        
        print(f"   ✅ Silhouette: {metrics['silhouette']:.3f}")
        return labels, metrics
    
    def train_hierarchical(self, X):
        """层次聚类"""
        print("🟠 训练 Hierarchical Clustering...")
        hierarchical = AgglomerativeClustering(n_clusters=self.n_clusters)
        labels = hierarchical.fit_predict(X)
        
        self.models['hierarchical'] = hierarchical
        self.labels['hierarchical'] = labels
        
        metrics = self.evaluate_clustering(X, labels)
        self.metrics['hierarchical'] = metrics
        
        print(f"   ✅ Silhouette: {metrics['silhouette']:.3f}")
        return labels, metrics
    
    def evaluate_clustering(self, X, labels):
        """评估聚类效果"""
        if len(set(labels)) < 2:
            return {'silhouette': -1, 'ch_score': 0, 'db_score': 999}
        
        try:
            metrics = {
                'silhouette': silhouette_score(X, labels),
                'ch_score': calinski_harabasz_score(X, labels),
                'db_score': davies_bouldin_score(X, labels)
            }
        except:
            metrics = {'silhouette': -1, 'ch_score': 0, 'db_score': 999}
        
        return metrics
    
    def train_all(self, features_df):
        """训练所有基线模型"""
        print("\n" + "="*60)
        print("🚀 开始训练基线模型")
        print("="*60)
        
        X_scaled, feature_names = self.prepare_data(features_df)
        
        # 训练所有模型
        self.train_kmeans(X_scaled)
        self.train_dbscan(X_scaled, eps=0.5, min_samples=5)
        self.train_gmm(X_scaled)
        self.train_hierarchical(X_scaled)
        
        # 打印对比
        print("\n" + "="*60)
        print("📊 基线模型评估结果")
        print("="*60)
        
        results = []
        for name, metrics in self.metrics.items():
            results.append({
                'Model': name,
                'Silhouette': metrics['silhouette'],
                'CH_Score': metrics['ch_score'],
                'DB_Score': metrics['db_score']
            })
        
        results_df = pd.DataFrame(results)
        print(results_df.to_string(index=False))
        
        return results_df
    
    def visualize_clusters(self, features_df, save_dir='./results/baseline'):
        """可视化聚类结果"""
        os.makedirs(save_dir, exist_ok=True)
        
        X_scaled, feature_names = self.prepare_data(features_df)
        
        # PCA降维到2D
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('基线模型聚类结果对比', fontsize=16)
        
        models_list = ['kmeans', 'dbscan', 'gmm', 'hierarchical']
        
        for idx, model_name in enumerate(models_list):
            ax = axes[idx // 2, idx % 2]
            
            if model_name in self.labels:
                labels = self.labels[model_name]
                scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], 
                                    c=labels, cmap='viridis', alpha=0.6, s=30)
                
                metrics = self.metrics[model_name]
                title = f"{model_name.upper()}\n"
                title += f"Silhouette: {metrics['silhouette']:.3f}, "
                title += f"DB: {metrics['db_score']:.3f}"
                
                ax.set_title(title)
                ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
                ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
                plt.colorbar(scatter, ax=ax, label='Cluster')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/baseline_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 可视化已保存至: {save_dir}/baseline_comparison.png")
        plt.close()
    
    def save_models(self, save_dir='./results/baseline'):
        """保存模型"""
        os.makedirs(save_dir, exist_ok=True)
        
        with open(f'{save_dir}/baseline_models.pkl', 'wb') as f:
            pickle.dump({
                'models': self.models,
                'labels': self.labels,
                'metrics': self.metrics,
                'scaler': self.scaler
            }, f)
        
        print(f"💾 模型已保存至: {save_dir}/baseline_models.pkl")


# 使用示例
if __name__ == "__main__":
    # 读取特征
    features_df = pd.read_csv('./results/features/combined_features.csv')
    
    print(f"📊 数据形状: {features_df.shape}")
    
    # 训练基线模型
    baseline = BaselineModels(n_clusters=5)
    results_df = baseline.train_all(features_df)
    
    # 可视化
    baseline.visualize_clusters(features_df)
    
    # 保存
    baseline.save_models()
    results_df.to_csv('./results/baseline/baseline_metrics.csv', index=False)
    
    print("\n✅ 基线模型训练完成！")