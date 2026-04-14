"""
智能模型对比与推荐系统
自动分析、对比、解释并给出最佳方案
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import os
import pickle

# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class IntelligentModelComparator:
    """智能模型对比系统"""
    
    def __init__(self, results_dir='./results'):
        self.results_dir = results_dir
        self.features_df = None
        self.models_data = {}
        self.cluster_interpretations = {}
        
    def load_all_data(self):
        """加载所有数据"""
        print("\n" + "="*70)
        print("📂 加载数据")
        print("="*70)
        
        # 1. 读取特征
        self.features_df = pd.read_csv(f'{self.results_dir}/features/combined_features.csv')
        print(f"✅ 特征数据: {self.features_df.shape}")
        
        # 2. 加载所有模型
        print("\n📦 加载模型结果...")
        
        # Baseline模型
        baseline_pkl = f'{self.results_dir}/baseline/baseline_models.pkl'
        if os.path.exists(baseline_pkl):
            with open(baseline_pkl, 'rb') as f:
                baseline_data = pickle.load(f)
            
            for key in ['kmeans', 'gmm', 'hierarchical']:
                if key in baseline_data.get('labels', {}):
                    self.models_data[key.upper()] = {
                        'labels': baseline_data['labels'][key],
                        'type': 'traditional',
                        'channel': 'single'
                    }
        
        # 深度学习模型配置
        dl_models = {
            'LSTM-AE': {'type': 'temporal', 'channel': 'single', 'dir': 'lstm_ae'},
            'GRU-AE': {'type': 'temporal', 'channel': 'single', 'dir': 'gru_ae'},
            'Attention-LSTM': {'type': 'temporal', 'channel': 'single', 'dir': 'attention_lstm'},
            'Transformer-AE': {'type': 'temporal', 'channel': 'single', 'dir': 'transformer_ae'},
            'TCN-AE': {'type': 'temporal', 'channel': 'single', 'dir': 'tcn_ae'},
            'Autoencoder': {'type': 'non-temporal', 'channel': 'dual', 'dir': 'autoencoder'},
            'VAE': {'type': 'non-temporal', 'channel': 'single', 'dir': 'vae'},
            'Contrastive': {'type': 'non-temporal', 'channel': 'single', 'dir': 'contrastive'},
            'DEC': {'type': 'non-temporal', 'channel': 'single', 'dir': 'dec'}
        }
        
        for model_name, config in dl_models.items():
            result_file = f"{self.results_dir}/{config['dir']}/clustered_results.csv"
            if os.path.exists(result_file):
                df = pd.read_csv(result_file)
                self.models_data[model_name] = {
                    'labels': df['cluster'].values,
                    'type': config['type'],
                    'channel': config['channel']
                }
        
        print(f"✅ 加载了 {len(self.models_data)} 个模型")
        
    def evaluate_models(self):
        """评估所有模型"""
        print("\n" + "="*70)
        print("📊 评估模型性能")
        print("="*70)
        
        # 准备特征
        X = self.features_df.drop(['event_id', 'vehicle_id'], axis=1, errors='ignore')
        X = X.select_dtypes(include=[np.number])
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        results = []
        
        for model_name, data in self.models_data.items():
            labels = data['labels']
            
            if len(set(labels)) > 1:
                try:
                    metrics = {
                        'Model': model_name,
                        'Type': data['type'],
                        'Channel': data['channel'],
                        'Silhouette': silhouette_score(X, labels),
                        'CH_Score': calinski_harabasz_score(X, labels),
                        'DB_Score': davies_bouldin_score(X, labels),
                        'N_Clusters': len(set(labels)),
                        'N_Events': len(labels)
                    }
                except:
                    metrics = {
                        'Model': model_name,
                        'Type': data['type'],
                        'Channel': data['channel'],
                        'Silhouette': 0,
                        'CH_Score': 0,
                        'DB_Score': 999,
                        'N_Clusters': len(set(labels)),
                        'N_Events': len(labels)
                    }
            else:
                metrics = {
                    'Model': model_name,
                    'Type': data['type'],
                    'Channel': data['channel'],
                    'Silhouette': 0,
                    'CH_Score': 0,
                    'DB_Score': 999,
                    'N_Clusters': len(set(labels)),
                    'N_Events': len(labels)
                }
            
            results.append(metrics)
            print(f"  {model_name:20s} | Sil={metrics['Silhouette']:.3f} | CH={metrics['CH_Score']:7.1f} | DB={metrics['DB_Score']:.3f}")
        
        self.metrics_df = pd.DataFrame(results)
        
        # 计算综合排名
        self.metrics_df['Sil_Rank'] = self.metrics_df['Silhouette'].rank(ascending=False)
        self.metrics_df['CH_Rank'] = self.metrics_df['CH_Score'].rank(ascending=False)
        self.metrics_df['DB_Rank'] = self.metrics_df['DB_Score'].rank(ascending=True)
        self.metrics_df['Overall_Score'] = (
            self.metrics_df['Silhouette'] * 0.5 +  # Silhouette权重50%
            (self.metrics_df['CH_Score'] / self.metrics_df['CH_Score'].max()) * 0.3 +  # CH权重30%
            (1 - self.metrics_df['DB_Score'] / self.metrics_df['DB_Score'].max()) * 0.2  # DB权重20%
        )
        self.metrics_df = self.metrics_df.sort_values('Overall_Score', ascending=False)
        
        return self.metrics_df
    
    def interpret_clusters(self, model_name):
        """解释聚类结果（物理意义）"""
        print(f"\n🔍 解释 {model_name} 的聚类结果...")
        
        labels = self.models_data[model_name]['labels']
        
        # 合并数据
        data = self.features_df.copy()
        data['cluster'] = labels
        
        # 过滤驾驶事件
        data = data[
            (data['speed_mean'] > 5) &
            (data['distance_total'] > 0.5) &
            (data['moving_ratio'] > 0.3)
        ]
        
        if len(data) == 0:
            print("  ⚠️  没有有效驾驶事件")
            return {}
        
        # 分析每个簇
        cluster_profiles = {}
        
        for cluster_id in sorted(data['cluster'].unique()):
            cluster_data = data[data['cluster'] == cluster_id]
            
            profile = {
                'count': len(cluster_data),
                'speed_mean': cluster_data['speed_mean'].mean(),
                'speed_std': cluster_data['speed_std'].mean(),
                'acc_std': cluster_data['acc_std'].mean(),
                'harsh_accel': cluster_data['harsh_accel'].mean(),
                'harsh_decel': cluster_data['harsh_decel'].mean(),
                'power_mean': cluster_data['power_mean'].mean(),
                'soc_drop': abs(cluster_data['soc_drop_total'].mean()),
                'distance': cluster_data['distance_total'].mean(),
                'efficiency': cluster_data.get('efficiency_mean', pd.Series([0])).mean()
            }
            
            # 自动命名
            name = self._auto_name_cluster(profile)
            profile['name'] = name
            
            cluster_profiles[cluster_id] = profile
            
            print(f"\n  簇 {cluster_id}: {name}")
            print(f"    事件数: {profile['count']}")
            print(f"    平均速度: {profile['speed_mean']:.1f} km/h")
            print(f"    急加速: {profile['harsh_accel']:.1f} 次")
            print(f"    急减速: {profile['harsh_decel']:.1f} 次")
            print(f"    平均功率: {profile['power_mean']:.1f} kW")
        
        self.cluster_interpretations[model_name] = cluster_profiles
        return cluster_profiles
    
    def _auto_name_cluster(self, profile):
        """自动命名簇（基于物理特征）"""
        speed = profile['speed_mean']
        harsh_accel = profile['harsh_accel']
        harsh_decel = profile['harsh_decel']
        power = profile['power_mean']
        
        # 命名逻辑
        if harsh_accel > 5 or harsh_decel > 5:
            return "🔴 激进驾驶 (Aggressive)"
        elif speed < 20:
            return "🟡 城市拥堵 (Urban Congestion)"
        elif 20 <= speed < 40:
            return "🟠 城市道路 (Urban Driving)"
        elif 40 <= speed < 70:
            return "🟢 郊区道路 (Suburban)"
        elif speed >= 70:
            return "🔵 高速巡航 (Highway Cruise)"
        elif power < -10:
            return "🟣 经济驾驶 (Eco Driving)"
        else:
            return "⚪ 平稳驾驶 (Smooth Driving)"
    
    def compare_architectures(self):
        """对比不同架构"""
        print("\n" + "="*70)
        print("🏗️  架构对比分析")
        print("="*70)
        
        # 按类型分组
        temporal = self.metrics_df[self.metrics_df['Type'] == 'temporal']
        non_temporal = self.metrics_df[self.metrics_df['Type'] == 'non-temporal']
        traditional = self.metrics_df[self.metrics_df['Type'] == 'traditional']
        
        # 按通道分组
        dual_channel = self.metrics_df[self.metrics_df['Channel'] == 'dual']
        single_channel = self.metrics_df[self.metrics_df['Channel'] == 'single']
        
        print("\n📊 按模型类型:")
        print(f"  时序模型平均分: {temporal['Overall_Score'].mean():.3f}")
        print(f"  非时序模型平均分: {non_temporal['Overall_Score'].mean():.3f}")
        print(f"  传统模型平均分: {traditional['Overall_Score'].mean():.3f}")
        
        print("\n📊 按通道类型:")
        print(f"  双通道模型平均分: {dual_channel['Overall_Score'].mean():.3f}")
        print(f"  单通道模型平均分: {single_channel['Overall_Score'].mean():.3f}")
        
        # 最佳模型
        best_temporal = temporal.iloc[0] if len(temporal) > 0 else None
        best_non_temporal = non_temporal.iloc[0] if len(non_temporal) > 0 else None
        
        return {
            'best_temporal': best_temporal,
            'best_non_temporal': best_non_temporal,
            'dual_channel_score': dual_channel['Overall_Score'].mean() if len(dual_channel) > 0 else 0,
            'single_channel_score': single_channel['Overall_Score'].mean()
        }
    
    def visualize_comparison(self, save_dir='./results/intelligent_analysis'):
        """生成对比可视化"""
        os.makedirs(save_dir, exist_ok=True)
        
        print("\n🎨 生成对比可视化...")
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. 综合性能对比
        ax1 = fig.add_subplot(gs[0, :2])
        colors = ['#2ecc71' if i == 0 else '#3498db' if i < 3 else '#95a5a6' 
                 for i in range(len(self.metrics_df))]
        bars = ax1.barh(self.metrics_df['Model'], self.metrics_df['Overall_Score'], color=colors)
        ax1.set_xlabel('Overall Score (higher is better)', fontsize=12, fontweight='bold')
        ax1.set_title('🏆 Model Performance Ranking', fontsize=14, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # 添加分数标签
        for i, (bar, score) in enumerate(zip(bars, self.metrics_df['Overall_Score'])):
            ax1.text(score, bar.get_y() + bar.get_height()/2, 
                    f' {score:.3f}', va='center', fontsize=9, fontweight='bold')
        
        # 2. 类型对比
        ax2 = fig.add_subplot(gs[0, 2])
        type_scores = self.metrics_df.groupby('Type')['Overall_Score'].mean().sort_values(ascending=False)
        colors_type = ['#e74c3c', '#3498db', '#95a5a6']
        ax2.bar(range(len(type_scores)), type_scores.values, color=colors_type)
        ax2.set_xticks(range(len(type_scores)))
        ax2.set_xticklabels(type_scores.index, rotation=45)
        ax2.set_ylabel('Average Score')
        ax2.set_title('📊 Model Type Comparison', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. Silhouette对比
        ax3 = fig.add_subplot(gs[1, 0])
        top5 = self.metrics_df.head(5)
        ax3.barh(top5['Model'], top5['Silhouette'], color=sns.color_palette('RdYlGn', 5))
        ax3.set_xlabel('Silhouette Score')
        ax3.set_title('🎯 Top 5 - Silhouette', fontsize=11, fontweight='bold')
        ax3.grid(axis='x', alpha=0.3)
        
        # 4. CH Score对比
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.barh(top5['Model'], top5['CH_Score'], color=sns.color_palette('viridis', 5))
        ax4.set_xlabel('CH Score')
        ax4.set_title('🎯 Top 5 - CH Score', fontsize=11, fontweight='bold')
        ax4.grid(axis='x', alpha=0.3)
        
        # 5. DB Score对比
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.barh(top5['Model'], top5['DB_Score'], color=sns.color_palette('rocket', 5))
        ax5.set_xlabel('DB Score (lower is better)')
        ax5.set_title('🎯 Top 5 - DB Score', fontsize=11, fontweight='bold')
        ax5.invert_xaxis()
        ax5.grid(axis='x', alpha=0.3)
        
        # 6. 通道类型对比
        ax6 = fig.add_subplot(gs[2, 0])
        channel_scores = self.metrics_df.groupby('Channel')['Overall_Score'].mean()
        colors_ch = ['#e67e22', '#3498db']
        ax6.bar(range(len(channel_scores)), channel_scores.values, color=colors_ch)
        ax6.set_xticks(range(len(channel_scores)))
        ax6.set_xticklabels(['Dual Channel', 'Single Channel'])
        ax6.set_ylabel('Average Score')
        ax6.set_title('🔀 Channel Comparison', fontsize=12, fontweight='bold')
        ax6.grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for i, v in enumerate(channel_scores.values):
            ax6.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 7. 散点图：性能 vs 簇数
        ax7 = fig.add_subplot(gs[2, 1])
        scatter = ax7.scatter(self.metrics_df['N_Clusters'], 
                             self.metrics_df['Overall_Score'],
                             c=self.metrics_df['Overall_Score'],
                             s=100, cmap='RdYlGn', alpha=0.6, edgecolors='k')
        ax7.set_xlabel('Number of Clusters')
        ax7.set_ylabel('Overall Score')
        ax7.set_title('📈 Performance vs Cluster Count', fontsize=11, fontweight='bold')
        ax7.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax7, label='Score')
        
        # 8. 推荐标注
        ax8 = fig.add_subplot(gs[2, 2])
        ax8.axis('off')
        
        best = self.metrics_df.iloc[0]
        recommendation_text = f"""
🏆 RECOMMENDED MODEL
{'='*25}

Model: {best['Model']}
Type: {best['Type'].upper()}
Channel: {best['Channel'].upper()}

PERFORMANCE:
• Silhouette: {best['Silhouette']:.3f}
• CH Score: {best['CH_Score']:.1f}
• DB Score: {best['DB_Score']:.3f}
• Overall: {best['Overall_Score']:.3f}

Clusters: {int(best['N_Clusters'])}
Events: {int(best['N_Events'])}
"""
        ax8.text(0.1, 0.5, recommendation_text, 
                fontsize=10, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3),
                verticalalignment='center')
        
        plt.suptitle('Intelligent Model Comparison & Analysis', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        plt.savefig(f'{save_dir}/complete_comparison.png', dpi=300, bbox_inches='tight')
        print(f"  ✅ 完整对比图: {save_dir}/complete_comparison.png")
        plt.close()
    
    def generate_recommendation_report(self, save_dir='./results/intelligent_analysis'):
        """生成推荐报告"""
        os.makedirs(save_dir, exist_ok=True)
        
        print("\n📝 生成推荐报告...")
        
        best = self.metrics_df.iloc[0]
        arch_comparison = self.compare_architectures()
        
        # 解释最佳模型
        best_interpretation = self.interpret_clusters(best['Model'])
        
        report = f"""
# 🎯 电动车驾驶行为聚类 - 智能分析报告

生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. 最佳模型推荐

### 🏆 推荐模型: **{best['Model']}**

**为什么选择这个模型？**

1. **性能最优**
   - Silhouette Score: **{best['Silhouette']:.3f}** (越高越好，>0.5为优秀)
   - CH Score: **{best['CH_Score']:.2f}** (簇间分离度)
   - DB Score: **{best['DB_Score']:.3f}** (簇内紧密度，���低越好)
   - 综合得分: **{best['Overall_Score']:.3f}**

2. **模型特点**
   - 类型: **{best['Type']}**
   - 通道: **{best['Channel']}**
   - 簇数: **{int(best['N_Clusters'])}**

---

## 2. 架构对比结论

### 时序 vs 非时序
"""
        
        if arch_comparison['best_temporal'] is not None:
            report += f"""
**时序模型** (LSTM, GRU, Transformer等)
- 最佳: {arch_comparison['best_temporal']['Model']}
- 得分: {arch_comparison['best_temporal']['Overall_Score']:.3f}
- ✅ 优势: 能捕捉驾驶行为的时间依赖性
"""
        
        if arch_comparison['best_non_temporal'] is not None:
            report += f"""
**非时序模型** (Autoencoder, VAE等)
- 最佳: {arch_comparison['best_non_temporal']['Model']}
- 得分: {arch_comparison['best_non_temporal']['Overall_Score']:.3f}
- ✅ 优势: 训练速度快，计算效率高
"""
        
        report += f"""
### 双通道 vs 单通道

**双通道模型** (分别处理能量和驾驶特征)
- 平均得分: {arch_comparison['dual_channel_score']:.3f}
- ✅ 优势: 保留特征的物理意义，可解释性强
- ❌ 劣势: 结构复杂，需要更多训练时间

**单通道模型** (统一处理所有特征)
- 平均得分: {arch_comparison['single_channel_score']:.3f}
- ✅ 优势: 结构简单，端到端优化
- ❌ 劣势: 特征混合，可能丢失物理意义

**结论**: {'双通道更优' if arch_comparison['dual_channel_score'] > arch_comparison['single_channel_score'] else '单通道更优'}

---

## 3. 聚类结果解释（物理意义）

使用 **{best['Model']}** 的聚类结果：

"""
        
        # 添加簇解释
        if best['Model'] in self.cluster_interpretations:
            for cluster_id, profile in self.cluster_interpretations[best['Model']].items():
                report += f"""
### {profile['name']}
- **事件数**: {profile['count']} ({profile['count']/sum([p['count'] for p in self.cluster_interpretations[best['Model']].values()])*100:.1f}%)
- **平均速度**: {profile['speed_mean']:.1f} km/h
- **速度稳定性**: {'稳定' if profile['speed_std'] < 10 else '波动大'}
- **加速激进度**: {'激进' if profile['harsh_accel'] > 3 else '温和'}
- **制动激进度**: {'激进' if profile['harsh_decel'] > 3 else '温和'}
- **平均功率**: {profile['power_mean']:.1f} kW
- **能耗**: {profile['soc_drop']:.1f}% SOC

**特征**: 这是典型的{'高速' if profile['speed_mean'] > 60 else '城市'}驾驶场景，
驾驶风格{'激进' if profile['harsh_accel'] > 3 or profile['harsh_decel'] > 3 else '平稳'}。
"""
        
        report += f"""
---

## 4. 完整排名

| 排名 | 模型 | 类型 | 通道 | Silhouette | CH Score | DB Score | 综合得分 |
|------|------|------|------|-----------|----------|----------|----------|
"""
        
        for idx, row in self.metrics_df.head(10).iterrows():
            report += f"| {idx+1} | {row['Model']} | {row['Type']} | {row['Channel']} | {row['Silhouette']:.3f} | {row['CH_Score']:.1f} | {row['DB_Score']:.3f} | {row['Overall_Score']:.3f} |\n"
        
        report += f"""
---

## 5. 论文建议

### 方法论
1. **特征工程**: 分别提取能量特征（SOC、电压、电流、功率）和驾驶特征（速度、加速度）
2. **模型选择**: 使用 **{best['Model']}** 进行聚类
3. **评估指标**: Silhouette + CH + DB综合评估

### 实验设计
1. 对比传统方法（K-Means, GMM）与深度学习方法
2. 对比时序模型与非时序模型
3. 对比单通道与双通道架构

### 结果讨论
- {'时序模型' if best['Type'] == 'temporal' else '非时序模型'}在本数据集上效果最好
- 聚类结果具有明确的物理意义
- 可用于驾驶行为分析、能耗预测、安全评估

---

## 6. 实际应用建议

### 生产环境
- **推荐**: {best['Model']}
- **原因**: 性能最优，聚类质量高

### 实时应用
- **推荐**: {'K-Means或GRU-AE' if best['Type'] == 'temporal' else 'K-Means'}
- **原因**: 计算速度快

### 研究分析
- **推荐**: 对比Top 3模型
- **原因**: 全面评估，发现规律

---

*报告生成完成*
"""
        
        # 保存报告
        with open(f'{save_dir}/intelligent_recommendation.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        # 保存详细数据
        self.metrics_df.to_csv(f'{save_dir}/detailed_comparison.csv', index=False)
        
        print(f"  ✅ 推荐报告: {save_dir}/intelligent_recommendation.md")
        print(f"  ✅ 详细数据: {save_dir}/detailed_comparison.csv")
        
        return report
    
    def run_complete_analysis(self):
        """运行完整分析"""
        print("\n" + "="*70)
        print("🧠 智能模型对比与推荐系统")
        print("="*70)
        
        # 1. 加载数据
        self.load_all_data()
        
        # 2. 评估模型
        self.evaluate_models()
        
        # 3. 架构对比
        arch_results = self.compare_architectures()
        
        # 4. 可视化
        self.visualize_comparison()
        
        # 5. 生成报告
        report = self.generate_recommendation_report()
        
        # 6. 打印关键结论
        print("\n" + "="*70)
        print("✅ 分析完成！关键结论：")
        print("="*70)
        
        best = self.metrics_df.iloc[0]
        print(f"\n🏆 最佳模型: {best['Model']}")
        print(f"📊 综合得分: {best['Overall_Score']:.3f}")
        print(f"🏗️  模型类型: {best['Type']}")
        print(f"🔀 通道类型: {best['Channel']}")
        print(f"📈 Silhouette: {best['Silhouette']:.3f}")
        
        print(f"\n💡 建议:")
        print(f"   1. 论文使用: {best['Model']} (性能最优)")
        print(f"   2. 生产环境: {best['Model']} 或 K-Means (根据需求)")
        print(f"   3. {'双通道' if arch_results['dual_channel_score'] > arch_results['single_channel_score'] else '单通道'}架构在本数据集上更有效")
        
        print(f"\n📁 结果位置:")
        print(f"   - ./results/intelligent_analysis/complete_comparison.png")
        print(f"   - ./results/intelligent_analysis/intelligent_recommendation.md")
        
        return best, arch_results


if __name__ == "__main__":
    analyzer = IntelligentModelComparator()
    best_model, arch_results = analyzer.run_complete_analysis()
    
    print("\n" + "="*70)
    print("🎉 智能分析完成！")
    print("="*70)