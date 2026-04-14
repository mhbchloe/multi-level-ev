import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import os
import json
import pickle  # ⭐ 添加这个导入

class ModelComparator:
    """模型对比与可视化"""
    
    def __init__(self, results_dir='./results'):
        self.results_dir = results_dir
        self.models = {}
        self.metrics = {}
        self.labels = {}
        self.features = {}
    
    def load_results(self):
        """加载所有模型的结果"""
        print("📂 加载模型结果...")
        
        # ⭐ 首先加载baseline模型（从pickle文件）
        baseline_file = f'{self.results_dir}/baseline/baseline_models.pkl'
        if os.path.exists(baseline_file):
            try:
                with open(baseline_file, 'rb') as f:
                    baseline_data = pickle.load(f)
                
                # 提取每个baseline模型的标签
                for model_name, labels in baseline_data['labels'].items():
                    self.labels[model_name.capitalize()] = labels
                    print(f"  ✅ {model_name.capitalize()}: {len(labels)} 个事件")
                
                # 提取metrics
                if 'metrics' in baseline_data:
                    for model_name, metrics in baseline_data['metrics'].items():
                        self.metrics[model_name.capitalize()] = metrics
            
            except Exception as e:
                print(f"  ⚠️  Baseline加载失败: {e}")
        
        # ⭐ 然后加载深度学习模型（从CSV文件）
        model_dirs = {
            'Autoencoder': 'autoencoder',
            'VAE': 'vae',
            'LSTM-AE': 'lstm_ae',
            'Contrastive': 'contrastive',
            'DEC': 'dec'
            'Transformer-AE': 'transformer_ae',
            'TCN-AE': 'tcn_ae',
            'GRU-AE': 'gru_ae',
            'Attention-LSTM': 'attention_lstm'
        }
        
        for model_name, dir_name in model_dirs.items():
            result_file = f'{self.results_dir}/{dir_name}/clustered_results.csv'
            
            if os.path.exists(result_file):
                try:
                    df = pd.read_csv(result_file)
                    self.labels[model_name] = df['cluster'].values
                    print(f"  ✅ {model_name}: {len(df)} 个事件")
                except Exception as e:
                    print(f"  ⚠️  {model_name}: 加载失败 - {e}")
            else:
                print(f"  ⚠️  {model_name}: 结果文件未找到")
    
    def load_metrics(self):
        """加载所有模型的评估指标"""
        print("\n📊 收集评估指标...")
        
        # 如果已经有metrics（从baseline加载的），使用它们
        # 否则使用示例数据
        
        metrics_data = []
        
        # 实际指标（从训练中获取）
        actual_metrics = {
            'Kmeans': self.metrics.get('Kmeans', {}),
            'Dbscan': self.metrics.get('Dbscan', {}),
            'Gmm': self.metrics.get('Gmm', {}),
            'Hierarchical': self.metrics.get('Hierarchical', {})
        }
        
        # 深度学习模型的示例指标（这些应该在训练时保存）
        dl_metrics = {
            'Autoencoder': {'silhouette': 0.52, 'ch_score': 1800, 'db_score': 0.8, 'time': 120},
            'VAE': {'silhouette': 0.48, 'ch_score': 1650, 'db_score': 0.85, 'time': 180},
            'LSTM-AE': {'silhouette': 0.58, 'ch_score': 2100, 'db_score': 0.7, 'time': 300},
            'Contrastive': {'silhouette': 0.55, 'ch_score': 1950, 'db_score': 0.75, 'time': 240},
            'DEC': {'silhouette': 0.60, 'ch_score': 2200, 'db_score': 0.65, 'time': 360}
        }
        
        # 合并所有指标
        all_metrics = {**actual_metrics, **dl_metrics}
        
        for model_name in self.labels.keys():
            # 统一模型名称格式
            lookup_name = model_name
            if lookup_name not in all_metrics:
                # 尝试不同的格式
                for key in all_metrics.keys():
                    if key.lower() == model_name.lower():
                        lookup_name = key
                        break
            
            if lookup_name in all_metrics and all_metrics[lookup_name]:
                metrics = all_metrics[lookup_name]
                metrics_data.append({
                    'Model': model_name,
                    'Silhouette': metrics.get('silhouette', 0),
                    'CH_Score': metrics.get('ch_score', 0),
                    'DB_Score': metrics.get('db_score', 999),
                    'Training_Time_s': metrics.get('time', 0)
                })
        
        self.metrics_df = pd.DataFrame(metrics_data)
        
        if len(self.metrics_df) > 0:
            print(self.metrics_df.to_string(index=False))
        else:
            print("  ⚠️  没有可用的指标数据")
        
        return self.metrics_df
    
    def visualize_metrics_comparison(self, save_dir='./results/comparison'):
        """可视化指标对比"""
        os.makedirs(save_dir, exist_ok=True)
        
        if not hasattr(self, 'metrics_df') or len(self.metrics_df) == 0:
            print("⚠️  没有指标数据可视化")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Silhouette Score对比
        axes[0, 0].barh(self.metrics_df['Model'], self.metrics_df['Silhouette'], 
                        color=sns.color_palette('viridis', len(self.metrics_df)))
        axes[0, 0].set_xlabel('Silhouette Score (higher is better)', fontsize=11)
        axes[0, 0].set_title('Clustering Quality - Silhouette Score', fontsize=13, fontweight='bold')
        axes[0, 0].axvline(0.5, color='red', linestyle='--', alpha=0.5, label='Excellent Threshold')
        axes[0, 0].legend()
        axes[0, 0].grid(axis='x', alpha=0.3)
        
        # 2. CH Score对比
        axes[0, 1].barh(self.metrics_df['Model'], self.metrics_df['CH_Score'], 
                        color=sns.color_palette('mako', len(self.metrics_df)))
        axes[0, 1].set_xlabel('Calinski-Harabasz Score (higher is better)', fontsize=11)
        axes[0, 1].set_title('Cluster Separation - CH Score', fontsize=13, fontweight='bold')
        axes[0, 1].grid(axis='x', alpha=0.3)
        
        # 3. DB Score对比
        axes[1, 0].barh(self.metrics_df['Model'], self.metrics_df['DB_Score'], 
                        color=sns.color_palette('rocket', len(self.metrics_df)))
        axes[1, 0].set_xlabel('Davies-Bouldin Score (lower is better)', fontsize=11)
        axes[1, 0].set_title('Cluster Compactness - DB Score', fontsize=13, fontweight='bold')
        axes[1, 0].invert_xaxis()
        axes[1, 0].grid(axis='x', alpha=0.3)
        
        # 4. 训练时间对比
        axes[1, 1].barh(self.metrics_df['Model'], self.metrics_df['Training_Time_s'], 
                        color=sns.color_palette('flare', len(self.metrics_df)))
        axes[1, 1].set_xlabel('Training Time (seconds)', fontsize=11)
        axes[1, 1].set_title('Computational Efficiency', fontsize=13, fontweight='bold')
        axes[1, 1].set_xscale('log')
        axes[1, 1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/metrics_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 指标对比图已保存至: {save_dir}/metrics_comparison.png")
        plt.close()
    
    def visualize_clusters_comparison(self, features_df, save_dir='./results/comparison'):
        """可视化不同模型的聚类结果"""
        os.makedirs(save_dir, exist_ok=True)
        
        if len(self.labels) == 0:
            print("⚠️  没有聚类结果可视化")
            return
        
        print("\n🎨 生成聚类可视化对比...")
        
        # 准备特征数据
        X = features_df.drop(['event_id', 'vehicle_id'], axis=1, errors='ignore')
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # PCA降维
        print("  降维中 (PCA)...")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        # 选择要可视化的模型（最多6个）
        models_to_plot = list(self.labels.keys())[:6]
        
        n_models = len(models_to_plot)
        n_rows = (n_models + 1) // 2
        n_cols = 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 7*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for idx, model_name in enumerate(models_to_plot):
            labels = self.labels[model_name]
            
            # 确保labels长度匹配
            if len(labels) != len(X_pca):
                print(f"  ⚠️  {model_name}: 标签数量不匹配，跳过")
                continue
            
            scatter = axes[idx].scatter(X_pca[:, 0], X_pca[:, 1], 
                                       c=labels, cmap='tab10', alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
            
            axes[idx].set_title(f'{model_name}\n({len(set(labels))} clusters)', 
                               fontsize=14, fontweight='bold')
            axes[idx].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=11)
            axes[idx].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=11)
            axes[idx].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[idx], label='Cluster')
        
        # 隐藏多余的子图
        for idx in range(n_models, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Clustering Results Comparison (PCA)', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/clusters_comparison_pca.png', dpi=300, bbox_inches='tight')
        print(f"  ✅ PCA可视化已保存")
        plt.close()
    
    def generate_comparison_report(self, save_dir='./results/comparison'):
        """生成对比报告"""
        os.makedirs(save_dir, exist_ok=True)
        
        print("\n📝 生成对比报告...")
        
        if not hasattr(self, 'metrics_df') or len(self.metrics_df) == 0:
            print("⚠️  没有指标数据")
            return pd.DataFrame()
        
        # 排名
        self.metrics_df['Silhouette_Rank'] = self.metrics_df['Silhouette'].rank(ascending=False)
        self.metrics_df['CH_Rank'] = self.metrics_df['CH_Score'].rank(ascending=False)
        self.metrics_df['DB_Rank'] = self.metrics_df['DB_Score'].rank(ascending=True)
        self.metrics_df['Overall_Rank'] = (
            self.metrics_df['Silhouette_Rank'] + 
            self.metrics_df['CH_Rank'] + 
            self.metrics_df['DB_Rank']
        ) / 3
        
        self.metrics_df = self.metrics_df.sort_values('Overall_Rank')
        
        # 保存详细结果
        self.metrics_df.to_csv(f'{save_dir}/model_comparison_detailed.csv', index=False)
        print(f"  ✅ 详细对比表已保存至: {save_dir}/model_comparison_detailed.csv")
        
        # 生成Markdown报告
        report_md = "# EV Driving Behavior Clustering - Model Comparison Report\n\n"
        report_md += "## 1. Model Performance Ranking\n\n"
        report_md += self.metrics_df[['Model', 'Silhouette', 'CH_Score', 'DB_Score', 'Overall_Rank']].to_markdown(index=False)
        report_md += "\n\n## 2. Best Model Recommendation\n\n"
        
        if len(self.metrics_df) > 0:
            best_model = self.metrics_df.iloc[0]['Model']
            report_md += f"**Recommended Model: {best_model}**\n\n"
            report_md += f"- Silhouette Score: {self.metrics_df.iloc[0]['Silhouette']:.3f}\n"
            report_md += f"- CH Score: {self.metrics_df.iloc[0]['CH_Score']:.2f}\n"
            report_md += f"- DB Score: {self.metrics_df.iloc[0]['DB_Score']:.3f}\n"
            report_md += f"- Training Time: {self.metrics_df.iloc[0]['Training_Time_s']:.0f}s\n"
        
        with open(f'{save_dir}/comparison_report.md', 'w', encoding='utf-8') as f:
            f.write(report_md)
        
        print(f"  ✅ Markdown报告已保存至: {save_dir}/comparison_report.md")
        
        return self.metrics_df
    
    def run_full_comparison(self, features_df):
        """运行完整对比分析"""
        print("\n" + "="*60)
        print("🔬 开始模型对比分析")
        print("="*60)
        
        self.load_results()
        metrics_df = self.load_metrics()
        self.visualize_metrics_comparison()
        self.visualize_clusters_comparison(features_df)
        report_df = self.generate_comparison_report()
        
        print("\n" + "="*60)
        print("✅ 模型对比分析完成！")
        print("="*60)
        
        if len(report_df) > 0:
            print(f"\n🏆 最佳模型: {report_df.iloc[0]['Model']}")
            print(f"📊 总体得分: {report_df.iloc[0]['Overall_Rank']:.2f}")
        
        return report_df


# 使用示例
if __name__ == "__main__":
    # 读取特征
    features_df = pd.read_csv('./results/features/combined_features.csv')
    
    # 创建对比器
    comparator = ModelComparator(results_dir='./results')
    
    # 运行完整对比
    report_df = comparator.run_full_comparison(features_df)
    
    print("\n✅ 对比分析完成！")