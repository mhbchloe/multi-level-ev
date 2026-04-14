"""
完整模型对比分析
包含：性能指标对比、可视化对比、统计分析
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import os
import pickle
import json

class CompleteModelComparison:
    """完整模型对比系统"""
    
    def __init__(self, results_dir='./results'):
        self.results_dir = results_dir
        self.features_df = None
        self.models_data = {}
        self.metrics_summary = {}
        
    def load_all_models(self):
        """加载所有模型结果"""
        print("\n" + "="*70)
        print("📂 加载模型结果")
        print("="*70)
        
        # 读取特征
        features_path = f'{self.results_dir}/features/combined_features.csv'
        if not os.path.exists(features_path):
            print(f"❌ 错误: 找不到特征文件 {features_path}")
            return False
        
        self.features_df = pd.read_csv(features_path)
        print(f"✅ 特征数据: {self.features_df.shape}")
        
        # 模型列表
        model_dirs = {
            'LSTM-AE': 'lstm_ae',
            'GRU-AE': 'gru_ae',
            'Attention-LSTM': 'attention_lstm',
            'Transformer-AE': 'transformer_ae',
            'TCN-AE': 'tcn_ae',
            'Autoencoder': 'autoencoder',
            'VAE': 'vae',
            'Contrastive': 'contrastive',
            'DEC': 'dec',
            'K-Means': 'baseline'
        }
        
        # 加载每个模型
        for model_name, dir_name in model_dirs.items():
            result_file = f'{self.results_dir}/{dir_name}/clustered_results.csv'
            
            if os.path.exists(result_file):
                try:
                    df = pd.read_csv(result_file)
                    
                    # 计算评估指标
                    merged = self.features_df.merge(df, on=['event_id', 'vehicle_id'])
                    X = merged.drop(['event_id', 'vehicle_id', 'cluster'], axis=1, errors='ignore')
                    X = X.select_dtypes(include=[np.number])
                    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
                    
                    labels = df['cluster'].values
                    
                    if len(set(labels)) > 1:  # 至少2个簇
                        metrics = {
                            'silhouette': silhouette_score(X, labels),
                            'ch_score': calinski_harabasz_score(X, labels),
                            'db_score': davies_bouldin_score(X, labels),
                            'n_clusters': len(set(labels)),
                            'n_events': len(df)
                        }
                    else:
                        metrics = {
                            'silhouette': 0,
                            'ch_score': 0,
                            'db_score': 999,
                            'n_clusters': len(set(labels)),
                            'n_events': len(df)
                        }
                    
                    self.models_data[model_name] = {
                        'labels': labels,
                        'data': df,
                        'metrics': metrics
                    }
                    
                    print(f"✅ {model_name}: {len(df)} 事件, {metrics['n_clusters']} 簇, "
                          f"Sil={metrics['silhouette']:.3f}")
                    
                except Exception as e:
                    print(f"⚠️  {model_name}: 加载失败 - {e}")
            else:
                print(f"⚠️  {model_name}: 结果文件不存在")
        
        print(f"\n✅ 总共加载 {len(self.models_data)} 个模型")
        return len(self.models_data) > 0
    
    def generate_metrics_comparison(self, save_dir='./results/comparison'):
        """生成性能指标对比"""
        os.makedirs(save_dir, exist_ok=True)
        
        print("\n📊 生成性能指标对比...")
        
        # 整理数据
        metrics_data = []
        for model_name, data in self.models_data.items():
            metrics = data['metrics']
            metrics_data.append({
                'Model': model_name,
                'Silhouette': metrics.get('silhouette', 0),
                'CH Score': metrics.get('ch_score', 0),
                'DB Score': metrics.get('db_score', 999),
                'Clusters': metrics.get('n_clusters', 0),
                'Events': metrics.get('n_events', 0)
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        
        # 计算排���
        metrics_df['Sil_Rank'] = metrics_df['Silhouette'].rank(ascending=False)
        metrics_df['CH_Rank'] = metrics_df['CH Score'].rank(ascending=False)
        metrics_df['DB_Rank'] = metrics_df['DB Score'].rank(ascending=True)
        metrics_df['Overall_Rank'] = (metrics_df['Sil_Rank'] + metrics_df['CH_Rank'] + metrics_df['DB_Rank']) / 3
        
        metrics_df = metrics_df.sort_values('Overall_Rank')
        
        # 保存
        metrics_df.to_csv(f'{save_dir}/metrics_comparison.csv', index=False)
        
        # 可视化
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Silhouette Score
        axes[0, 0].barh(metrics_df['Model'], metrics_df['Silhouette'],
                       color=sns.color_palette('viridis', len(metrics_df)))
        axes[0, 0].set_xlabel('Silhouette Score (higher is better)', fontsize=11)
        axes[0, 0].set_title('Clustering Quality - Silhouette Score', fontsize=13, fontweight='bold')
        axes[0, 0].axvline(0.5, color='red', linestyle='--', alpha=0.5, label='Excellent')
        axes[0, 0].legend()
        axes[0, 0].grid(axis='x', alpha=0.3)
        
        # 添加数值标签
        for i, v in enumerate(metrics_df['Silhouette']):
            axes[0, 0].text(v, i, f' {v:.3f}', va='center', fontsize=9)
        
        # 2. CH Score
        axes[0, 1].barh(metrics_df['Model'], metrics_df['CH Score'],
                       color=sns.color_palette('mako', len(metrics_df)))
        axes[0, 1].set_xlabel('Calinski-Harabasz Score (higher is better)', fontsize=11)
        axes[0, 1].set_title('Cluster Separation - CH Score', fontsize=13, fontweight='bold')
        axes[0, 1].grid(axis='x', alpha=0.3)
        
        # 3. DB Score
        axes[1, 0].barh(metrics_df['Model'], metrics_df['DB Score'],
                       color=sns.color_palette('rocket', len(metrics_df)))
        axes[1, 0].set_xlabel('Davies-Bouldin Score (lower is better)', fontsize=11)
        axes[1, 0].set_title('Cluster Compactness - DB Score', fontsize=13, fontweight='bold')
        axes[1, 0].invert_xaxis()
        axes[1, 0].grid(axis='x', alpha=0.3)
        
        # 4. 综合排名
        axes[1, 1].barh(metrics_df['Model'], metrics_df['Overall_Rank'],
                       color=sns.color_palette('flare', len(metrics_df)))
        axes[1, 1].set_xlabel('Overall Rank (lower is better)', fontsize=11)
        axes[1, 1].set_title('Overall Performance Ranking', fontsize=13, fontweight='bold')
        axes[1, 1].invert_xaxis()
        axes[1, 1].grid(axis='x', alpha=0.3)
        
        plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/metrics_comparison.png', dpi=300, bbox_inches='tight')
        print(f"   ✅ 性能对比图已保存")
        plt.close()
        
        # 打印排名
        print("\n🏆 模型排名:")
        print(metrics_df[['Model', 'Silhouette', 'CH Score', 'DB Score', 'Overall_Rank']].to_string(index=False))
        
        return metrics_df
    
    def visualize_clustering_comparison(self, save_dir='./results/comparison'):
        """可视化聚类效果对比"""
        os.makedirs(save_dir, exist_ok=True)
        
        print("\n🎨 生成聚类可视化对比...")
        
        # 准备数据
        X = self.features_df.drop(['event_id', 'vehicle_id'], axis=1, errors='ignore')
        X = X.select_dtypes(include=[np.number])
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # PCA降维
        print("   降维中 (PCA)...")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        # 选择要可视化的模型（最多9个）
        models_to_plot = list(self.models_data.keys())[:9]
        
        n_models = len(models_to_plot)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for idx, model_name in enumerate(models_to_plot):
            labels = self.models_data[model_name]['labels']
            
            if len(labels) != len(X_pca):
                print(f"   ⚠️  {model_name}: 标签数量不匹配，跳过")
                continue
            
            scatter = axes[idx].scatter(X_pca[:, 0], X_pca[:, 1],
                                       c=labels, cmap='tab10',
                                       alpha=0.6, s=20, edgecolors='k', linewidth=0.3)
            
            metrics = self.models_data[model_name]['metrics']
            title = f"{model_name}\n"
            title += f"({metrics.get('n_clusters', '?')} clusters, "
            title += f"Sil={metrics.get('silhouette', 0):.3f})"
            
            axes[idx].set_title(title, fontsize=11, fontweight='bold')
            axes[idx].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=9)
            axes[idx].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=9)
            axes[idx].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[idx], label='Cluster')
        
        # 隐藏多余子图
        for idx in range(n_models, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Clustering Results Comparison (PCA Visualization)',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/clustering_comparison_pca.png', dpi=300, bbox_inches='tight')
        print(f"   ✅ PCA可视化已保存")
        plt.close()
        
        # t-SNE可视化（使用子集）
        if len(X_pca) > 1000:
            print("   降维中 (t-SNE, 使用1000样本)...")
            sample_idx = np.random.choice(len(X_pca), 1000, replace=False)
            X_tsne_input = X.iloc[sample_idx]
        else:
            print("   降维中 (t-SNE)...")
            sample_idx = np.arange(len(X_pca))
            X_tsne_input = X
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(X_tsne_input)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for idx, model_name in enumerate(models_to_plot):
            labels = self.models_data[model_name]['labels'][sample_idx]
            
            scatter = axes[idx].scatter(X_tsne[:, 0], X_tsne[:, 1],
                                       c=labels, cmap='tab10',
                                       alpha=0.6, s=20, edgecolors='k', linewidth=0.3)
            
            axes[idx].set_title(model_name, fontsize=11, fontweight='bold')
            axes[idx].set_xlabel('t-SNE 1', fontsize=9)
            axes[idx].set_ylabel('t-SNE 2', fontsize=9)
            axes[idx].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[idx], label='Cluster')
        
        for idx in range(n_models, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Clustering Results Comparison (t-SNE Visualization)',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/clustering_comparison_tsne.png', dpi=300, bbox_inches='tight')
        print(f"   ✅ t-SNE可视化已保存")
        plt.close()
    
    def generate_comparison_report(self, metrics_df, save_dir='./results/comparison'):
        """生成对比报告（Markdown）"""
        os.makedirs(save_dir, exist_ok=True)
        
        print("\n📝 生成对比报告...")
        
        report = "# Electric Vehicle Driving Behavior Clustering - Model Comparison Report\n\n"
        report += f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        report += "## 1. Performance Ranking\n\n"
        report += metrics_df[['Model', 'Silhouette', 'CH Score', 'DB Score', 'Clusters', 'Overall_Rank']].to_markdown(index=False)
        
        report += "\n\n## 2. Best Model\n\n"
        best_model = metrics_df.iloc[0]
        report += f"**Recommended Model: {best_model['Model']}**\n\n"
        report += f"- **Silhouette Score**: {best_model['Silhouette']:.3f}\n"
        report += f"- **CH Score**: {best_model['CH Score']:.2f}\n"
        report += f"- **DB Score**: {best_model['DB Score']:.3f}\n"
        report += f"- **Number of Clusters**: {int(best_model['Clusters'])}\n"
        report += f"- **Total Events**: {int(best_model['Events'])}\n"
        
        report += "\n## 3. Model Categories\n\n"
        report += "### Traditional Methods\n"
        report += "- K-Means: Fast, interpretable baseline\n\n"
        
        report += "### Deep Learning (Non-Temporal)\n"
        report += "- Autoencoder, VAE: Learn abstract representations\n"
        report += "- Contrastive Learning: SOTA for representation learning\n"
        report += "- DEC: End-to-end clustering optimization\n\n"
        
        report += "### Deep Learning (Temporal)\n"
        report += "- **LSTM-AE**: Capture long-term dependencies\n"
        report += "- **GRU-AE**: Lighter than LSTM, faster training\n"
        report += "- **Attention-LSTM**: Focus on important time steps\n"
        report += "- **Transformer-AE**: State-of-the-art for sequences\n"
        report += "- **TCN-AE**: Efficient temporal convolutions\n\n"
        
        report += "## 4. Key Findings\n\n"
        
        # 分析时序vs非时序
        temporal_models = [m for m in metrics_df['Model'] if any(x in m for x in ['LSTM', 'GRU', 'Attention', 'Transformer', 'TCN'])]
        if temporal_models:
            temporal_avg = metrics_df[metrics_df['Model'].isin(temporal_models)]['Silhouette'].mean()
            report += f"- **Temporal Models** average Silhouette: {temporal_avg:.3f}\n"
        
        non_temporal = [m for m in metrics_df['Model'] if m not in temporal_models]
        if non_temporal:
            non_temporal_avg = metrics_df[metrics_df['Model'].isin(non_temporal)]['Silhouette'].mean()
            report += f"- **Non-Temporal Models** average Silhouette: {non_temporal_avg:.3f}\n"
        
        report += f"\n## 5. Recommendations\n\n"
        report += f"1. **For Production**: Use {best_model['Model']} (best performance)\n"
        report += f"2. **For Speed**: Use K-Means (fastest)\n"
        report += f"3. **For Interpretability**: Use traditional methods with feature analysis\n"
        report += f"4. **For Research**: Compare top 3 models in detail\n"
        
        # 保存
        with open(f'{save_dir}/comparison_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"   ✅ 报告已保存: {save_dir}/comparison_report.md")
        
        return report
    
    def run_complete_comparison(self):
        """运行完整对比流程"""
        print("\n" + "="*70)
        print("🔬 完整模型对比分析")
        print("="*70)
        
        # 1. 加载模型
        if not self.load_all_models():
            print("❌ 没有可用的模型结果")
            return
        
        # 2. 性能指标对比
        metrics_df = self.generate_metrics_comparison()
        
        # 3. 聚类可视化对比
        self.visualize_clustering_comparison()
        
        # 4. 生成报告
        self.generate_comparison_report(metrics_df)
        
        print("\n" + "="*70)
        print("✅ 模型对比分析完成！")
        print("="*70)
        print(f"📁 结果保存在: ./results/comparison/")
        print("   - metrics_comparison.csv (性能指标)")
        print("   - metrics_comparison.png (性能对比图)")
        print("   - clustering_comparison_pca.png (PCA可视化)")
        print("   - clustering_comparison_tsne.png (t-SNE可视化)")
        print("   - comparison_report.md (对比报告)")
        
        return metrics_df


if __name__ == "__main__":
    comparator = CompleteModelComparison(results_dir='./results')
    metrics_df = comparator.run_complete_comparison()
    
    if metrics_df is not None:
        print(f"\n🏆 最佳模型: {metrics_df.iloc[0]['Model']}")
        print(f"📊 Silhouette Score: {metrics_df.iloc[0]['Silhouette']:.3f}")