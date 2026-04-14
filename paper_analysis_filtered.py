"""
论文分析（最终修复版）- 自动适配特征
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class FinalPaperAnalysis:
    """最终修复版论文分析"""
    
    def __init__(self, features_df, results_df, model_name='K-Means'):
        self.model_name = model_name
        
        # 合并数据
        self.data = features_df.merge(results_df, on=['event_id', 'vehicle_id'])
        
        print(f"📊 数据: {len(self.data)} 个事件")
        print(f"   簇数: {self.data['cluster'].nunique()}")
        print(f"   簇分布: {self.data['cluster'].value_counts().sort_index().to_dict()}")
        
        # 只保留数值列
        self.numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        self.numeric_cols = [col for col in self.numeric_cols if col not in ['event_id', 'cluster']]
    
    def generate_all_plots(self, save_dir='./results/paper_final'):
        """生成所有图表"""
        os.makedirs(save_dir, exist_ok=True)
        
        print("\n" + "="*70)
        print(f"📊 开始论文分析 - {self.model_name}")
        print("="*70)
        
        # 1. 速度分析
        self._plot_speed(save_dir)
        
        # 2. 加速度分析
        self._plot_acceleration(save_dir)
        
        # 3. 能耗分析
        self._plot_energy(save_dir)
        
        # 4. 综合对比
        self._plot_comprehensive(save_dir)
        
        # 5. 统计表
        self._generate_table(save_dir)
        
        print("\n✅ 所有分析完成！")
    
    def _plot_speed(self, save_dir):
        """速度分析"""
        print("\n🚗 速度分析...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 平均速度箱线图
        if 'speed_mean' in self.data.columns:
            self.data.boxplot(column='speed_mean', by='cluster', ax=axes[0, 0])
            axes[0, 0].set_title('Average Speed by Cluster')
            axes[0, 0].set_xlabel('Cluster')
            axes[0, 0].set_ylabel('Speed (km/h)')
            axes[0, 0].get_figure().suptitle('')
        
        # 2. 速度分布
        if 'speed_mean' in self.data.columns:
            for cluster_id in sorted(self.data['cluster'].unique()):
                cluster_data = self.data[self.data['cluster'] == cluster_id]
                axes[0, 1].hist(cluster_data['speed_mean'], alpha=0.5,
                               label=f'Cluster {cluster_id}', bins=20)
            axes[0, 1].set_xlabel('Speed (km/h)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Speed Distribution')
            axes[0, 1].legend()
            axes[0, 1].grid(alpha=0.3)
        
        # 3. 速度稳定性
        if 'speed_std' in self.data.columns:
            speed_std = self.data.groupby('cluster')['speed_std'].mean().sort_index()
            axes[1, 0].bar(speed_std.index, speed_std.values,
                          color=sns.color_palette('RdYlGn_r', len(speed_std)))
            axes[1, 0].set_xlabel('Cluster')
            axes[1, 0].set_ylabel('Speed Std Dev')
            axes[1, 0].set_title('Speed Stability')
            axes[1, 0].grid(axis='y', alpha=0.3)
        
        # 4. 最大速度
        if 'speed_max' in self.data.columns:
            speed_max = self.data.groupby('cluster')['speed_max'].mean().sort_index()
            axes[1, 1].bar(speed_max.index, speed_max.values,
                          color=sns.color_palette('Blues', len(speed_max)))
            axes[1, 1].set_xlabel('Cluster')
            axes[1, 1].set_ylabel('Max Speed (km/h)')
            axes[1, 1].set_title('Maximum Speed')
            axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/01_speed_analysis.png', dpi=300, bbox_inches='tight')
        print("   ✅ 速度分析完成")
        plt.close()
    
    def _plot_acceleration(self, save_dir):
        """加速度分析"""
        print("\n⚡ 加速度分析...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 急加速/急减速
        if 'harsh_accel' in self.data.columns and 'harsh_decel' in self.data.columns:
            harsh_stats = self.data.groupby('cluster')[['harsh_accel', 'harsh_decel']].mean()
            
            x = np.arange(len(harsh_stats))
            width = 0.35
            axes[0, 0].bar(x - width/2, harsh_stats['harsh_accel'], width,
                          label='Harsh Accel', color='#FF6B6B')
            axes[0, 0].bar(x + width/2, harsh_stats['harsh_decel'], width,
                          label='Harsh Decel', color='#4ECDC4')
            axes[0, 0].set_xlabel('Cluster')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].set_title('Harsh Events')
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(harsh_stats.index)
            axes[0, 0].legend()
            axes[0, 0].grid(axis='y', alpha=0.3)
        
        # 2. 加速度标准差
        if 'acc_std' in self.data.columns:
            acc_std = self.data.groupby('cluster')['acc_std'].mean().sort_index()
            axes[0, 1].bar(acc_std.index, acc_std.values,
                          color=sns.color_palette('Reds', len(acc_std)))
            axes[0, 1].set_xlabel('Cluster')
            axes[0, 1].set_ylabel('Acc Std (m/s²)')
            axes[0, 1].set_title('Driving Aggressiveness')
            axes[0, 1].grid(axis='y', alpha=0.3)
        
        # 3. 平均加速度
        if 'acc_mean' in self.data.columns:
            self.data.boxplot(column='acc_mean', by='cluster', ax=axes[1, 0])
            axes[1, 0].set_title('Average Acceleration')
            axes[1, 0].set_xlabel('Cluster')
            axes[1, 0].set_ylabel('Acceleration (m/s²)')
            axes[1, 0].get_figure().suptitle('')
        
        # 4. 加速度范围
        if 'acc_max' in self.data.columns and 'acc_min' in self.data.columns:
            self.data['acc_range'] = self.data['acc_max'] - self.data['acc_min']
            acc_range = self.data.groupby('cluster')['acc_range'].mean().sort_index()
            axes[1, 1].bar(acc_range.index, acc_range.values,
                          color=sns.color_palette('Oranges', len(acc_range)))
            axes[1, 1].set_xlabel('Cluster')
            axes[1, 1].set_ylabel('Acc Range (m/s²)')
            axes[1, 1].set_title('Acceleration Range')
            axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/02_acceleration_analysis.png', dpi=300, bbox_inches='tight')
        print("   ✅ 加速度分析完成")
        plt.close()
    
    def _plot_energy(self, save_dir):
        """能耗分析"""
        print("\n🔋 能耗分析...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 平均功率
        if 'power_mean' in self.data.columns:
            power_mean = self.data.groupby('cluster')['power_mean'].mean().sort_index()
            axes[0, 0].bar(power_mean.index, power_mean.values,
                          color=sns.color_palette('YlOrRd', len(power_mean)))
            axes[0, 0].set_xlabel('Cluster')
            axes[0, 0].set_ylabel('Power (kW)')
            axes[0, 0].set_title('Average Power')
            axes[0, 0].grid(axis='y', alpha=0.3)
        
        # 2. SOC下降
        if 'soc_drop_total' in self.data.columns:
            self.data['soc_drop_abs'] = abs(self.data['soc_drop_total'])
            soc_drop = self.data.groupby('cluster')['soc_drop_abs'].mean().sort_index()
            axes[0, 1].bar(soc_drop.index, soc_drop.values,
                          color=sns.color_palette('Greens_r', len(soc_drop)))
            axes[0, 1].set_xlabel('Cluster')
            axes[0, 1].set_ylabel('SOC Drop (%)')
            axes[0, 1].set_title('Energy Consumption')
            axes[0, 1].grid(axis='y', alpha=0.3)
        
        # 3. 功率 vs 速度
        if 'power_mean' in self.data.columns and 'speed_mean' in self.data.columns:
            for cluster_id in sorted(self.data['cluster'].unique()):
                cluster_data = self.data[self.data['cluster'] == cluster_id]
                axes[1, 0].scatter(cluster_data['speed_mean'], cluster_data['power_mean'],
                                  alpha=0.6, label=f'Cluster {cluster_id}', s=30)
            axes[1, 0].set_xlabel('Speed (km/h)')
            axes[1, 0].set_ylabel('Power (kW)')
            axes[1, 0].set_title('Power vs Speed')
            axes[1, 0].legend()
            axes[1, 0].grid(alpha=0.3)
        
        # 4. 行驶距离
        if 'distance_total' in self.data.columns:
            self.data.boxplot(column='distance_total', by='cluster', ax=axes[1, 1])
            axes[1, 1].set_title('Trip Distance')
            axes[1, 1].set_xlabel('Cluster')
            axes[1, 1].set_ylabel('Distance (km)')
            axes[1, 1].get_figure().suptitle('')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/03_energy_analysis.png', dpi=300, bbox_inches='tight')
        print("   ✅ 能耗分析完成")
        plt.close()
    
    def _plot_comprehensive(self, save_dir):
        """综合对比"""
        print("\n📊 综合对比...")
        
        key_features = [
            ('speed_mean', 'Avg Speed (km/h)'),
            ('speed_max', 'Max Speed (km/h)'),
            ('acc_std', 'Acc Std (m/s²)'),
            ('harsh_accel', 'Harsh Accel'),
            ('harsh_decel', 'Harsh Decel'),
            ('power_mean', 'Avg Power (kW)'),
            ('distance_total', 'Distance (km)'),
            ('duration_minutes', 'Duration (min)')
        ]
        
        available = [(f, t) for f, t in key_features if f in self.data.columns]
        
        n_features = len(available)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for idx, (feature, title) in enumerate(available):
            cluster_means = self.data.groupby('cluster')[feature].mean().sort_index()
            axes[idx].bar(cluster_means.index, cluster_means.values,
                         color=sns.color_palette('Set2', len(cluster_means)))
            axes[idx].set_xlabel('Cluster')
            axes[idx].set_ylabel(title)
            axes[idx].set_title(title, fontweight='bold')
            axes[idx].grid(axis='y', alpha=0.3)
        
        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'{self.model_name} - Comprehensive Comparison',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/04_comprehensive_comparison.png', dpi=300, bbox_inches='tight')
        print("   ✅ 综合对比完成")
        plt.close()
    
    def _generate_table(self, save_dir):
        """生成统计表"""
        print("\n📋 生成统计表...")
        
        key_features = ['speed_mean', 'acc_std', 'harsh_accel', 'harsh_decel',
                       'power_mean', 'distance_total', 'duration_minutes']
        
        available = [f for f in key_features if f in self.data.columns]
        
        summary = self.data.groupby('cluster')[available].agg(['mean', 'std', 'count'])
        summary.to_csv(f'{save_dir}/cluster_statistics.csv')
        
        print("   ✅ 统计表已保存")
        print("\n簇统计:")
        print(self.data.groupby('cluster')[available[:4]].mean().to_string())


# 主程序
if __name__ == "__main__":
    # 读取重新聚类的结果
    features_df = pd.read_csv('./results/features/combined_features.csv')
    results_df = pd.read_csv('./results/recluster/clustered_results.csv')
    
    analyzer = FinalPaperAnalysis(features_df, results_df, model_name='K-Means')
    analyzer.generate_all_plots()
    
    print("\n✅ 论文分析完成！查看: ./results/paper_final/")