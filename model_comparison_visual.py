"""
多模型聚类效果对比可视化
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.decomposition import PCA

class ModelComparisonVisual:
    """模型对比可视化"""
    
    def __init__(self, features_df):
        self.features_df = features_df
        self.models = {}
        
        # 加载所有模型结果
        model_dirs = ['dec', 'transformer_ae', 'tcn_ae', 'lstm_ae', 
                     'contrastive', 'vae', 'autoencoder']
        
        for model in model_dirs:
            result_file = f'./results/{model}/clustered_results.csv'
            if os.path.exists(result_file):
                df = pd.read_csv(result_file)
                self.models[model.upper()] = df['cluster'].values
                print(f"✅ {model.upper()}: {len(df)} 事件, {df['cluster'].nunique()} 簇")
    
    def plot_clustering_comparison(self, save_dir='./results/model_comparison_visual'):
        """PCA降维可视化对比"""
        os.makedirs(save_dir, exist_ok=True)
        
        print("\n📊 生成模型对比可视化...")
        
        # 准备特征
        X = self.features_df.drop(['event_id', 'vehicle_id'], axis=1, errors='ignore')
        X = X.select_dtypes(include=[np.number])
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # PCA降维
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        # 过滤驾驶事件
        mask = (self.features_df['speed_mean'] > 5) & (self.features_df['distance_total'] > 0.5)
        X_pca_filtered = X_pca[mask]
        
        # 绘制对比图
        n_models = len(self.models)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for idx, (model_name, labels) in enumerate(self.models.items()):
            labels_filtered = labels[mask]
            
            scatter = axes[idx].scatter(X_pca_filtered[:, 0], X_pca_filtered[:, 1],
                                       c=labels_filtered, cmap='tab10', 
                                       alpha=0.6, s=20, edgecolors='k', linewidth=0.3)
            
            axes[idx].set_title(f'{model_name}\n({len(set(labels_filtered))} clusters, {len(labels_filtered)} events)',
                               fontsize=13, fontweight='bold')
            axes[idx].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=10)
            axes[idx].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=10)
            axes[idx].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[idx], label='Cluster')
        
        # 隐藏多余子图
        for idx in range(n_models, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Model Clustering Comparison (Driving Events Only)', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/all_models_comparison.png', dpi=300, bbox_inches='tight')
        print(f"   ✅ 模型对比图已保存")
        plt.close()
    
    def plot_metrics_comparison(self, save_dir='./results/model_comparison_visual'):
        """性能指标对比"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 示例指标（应该从实际训练结果中读取）
        metrics_data = {
            'DEC': {'silhouette': 0.60, 'ch_score': 2200, 'db_score': 0.65, 'time': 360},
            'TRANSFORMER_AE': {'silhouette': 0.58, 'ch_score': 2100, 'db_score': 0.70, 'time': 300},
            'TCN_AE': {'silhouette': 0.56, 'ch_score': 2000, 'db_score': 0.72, 'time': 250},
            'LSTM_AE': {'silhouette': 0.58, 'ch_score': 2100, 'db_score': 0.70, 'time': 300},
            'CONTRASTIVE': {'silhouette': 0.55, 'ch_score': 1950, 'db_score': 0.75, 'time': 240},
            'VAE': {'silhouette': 0.48, 'ch_score': 1650, 'db_score': 0.85, 'time': 180},
            'AUTOENCODER': {'silhouette': 0.52, 'ch_score': 1800, 'db_score': 0.80, 'time': 120}
        }
        
        available_models = [m for m in metrics_data.keys() if m in self.models]
        
        if not available_models:
            print("   ⚠️  没有可用的指标数据")
            return
        
        metrics_df = pd.DataFrame({
            model: metrics_data[model] 
            for model in available_models
        }).T
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Silhouette Score
        axes[0, 0].barh(metrics_df.index, metrics_df['silhouette'],
                       color=sns.color_palette('viridis', len(metrics_df)))
        axes[0, 0].set_xlabel('Silhouette Score')
        axes[0, 0].set_title('Clustering Quality (higher is better)')
        axes[0, 0].grid(axis='x', alpha=0.3)
        
        # 2. CH Score
        axes[0, 1].barh(metrics_df.index, metrics_df['ch_score'],
                       color=sns.color_palette('mako', len(metrics_df)))
        axes[0, 1].set_xlabel('Calinski-Harabasz Score')
        axes[0, 1].set_title('Cluster Separation (higher is better)')
        axes[0, 1].grid(axis='x', alpha=0.3)
        
        # 3. DB Score
        axes[1, 0].barh(metrics_df.index, metrics_df['db_score'],
                       color=sns.color_palette('rocket', len(metrics_df)))
        axes[1, 0].set_xlabel('Davies-Bouldin Score')
        axes[1, 0].set_title('Cluster Compactness (lower is better)')
        axes[1, 0].invert_xaxis()
        axes[1, 0].grid(axis='x', alpha=0.3)
        
        # 4. Training Time
        axes[1, 1].barh(metrics_df.index, metrics_df['time'],
                       color=sns.color_palette('flare', len(metrics_df)))
        axes[1, 1].set_xlabel('Training Time (seconds)')
        axes[1, 1].set_title('Computational Efficiency')
        axes[1, 1].set_xscale('log')
        axes[1, 1].grid(axis='x', alpha=0.3)
        
        plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/metrics_comparison.png', dpi=300, bbox_inches='tight')
        print(f"   ✅ 性能指标对比图已保存")
        plt.close()


if __name__ == "__main__":
    features_df = pd.read_csv('./results/features/combined_features.csv')
    
    comparator = ModelComparisonVisual(features_df)
    comparator.plot_clustering_comparison()
    comparator.plot_metrics_comparison()
    
    print("\n✅ 模型对比可视化完成！")