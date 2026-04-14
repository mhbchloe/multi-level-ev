"""
论文级别的聚类结果分析与可视化
包含：速度分析、加速度分析、能耗分析、时间分析等
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats

# 设置中文字体（根据系统选择）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用于正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用于正常显示负号

class PaperAnalysis:
    """论文级别的聚类分析"""
    
    def __init__(self, features_df, results_df, model_name='Best Model'):
        """
        参数:
            features_df: 特征数据
            results_df: 聚类结果（包含event_id, vehicle_id, cluster）
            model_name: 模型名称
        """
        self.model_name = model_name
        
        # 合并数据
        self.data = features_df.merge(results_df, on=['event_id', 'vehicle_id'])
        
        # 只保留数值列
        self.numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        self.numeric_cols = [col for col in self.numeric_cols if col not in ['event_id', 'cluster']]
        
        print(f"📊 数据加载完成:")
        print(f"   事件数: {len(self.data)}")
        print(f"   簇数: {self.data['cluster'].nunique()}")
        print(f"   特征数: {len(self.numeric_cols)}")
    
    def analyze_speed(self, save_dir='./results/paper_analysis'):
        """速度特性分析"""
        os.makedirs(save_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("🚗 速度特性分析")
        print("="*60)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 平均速度箱线图
        if 'speed_mean' in self.data.columns:
            self.data.boxplot(column='speed_mean', by='cluster', ax=axes[0, 0])
            axes[0, 0].set_title('Average Speed by Cluster')
            axes[0, 0].set_xlabel('Cluster')
            axes[0, 0].set_ylabel('Speed (km/h)')
            axes[0, 0].get_figure().suptitle('')
            
            # 打印统计信息
            print("\n平均速度统计:")
            speed_stats = self.data.groupby('cluster')['speed_mean'].describe()
            print(speed_stats)
        
        # 2. 最大速度分布
        if 'speed_max' in self.data.columns:
            for cluster_id in sorted(self.data['cluster'].unique()):
                cluster_data = self.data[self.data['cluster'] == cluster_id]
                axes[0, 1].hist(cluster_data['speed_max'], alpha=0.5, 
                               label=f'Cluster {cluster_id}', bins=20)
            axes[0, 1].set_xlabel('Max Speed (km/h)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Max Speed Distribution')
            axes[0, 1].legend()
            axes[0, 1].grid(alpha=0.3)
        
        # 3. 速度标准差（驾驶稳定性）
        if 'speed_std' in self.data.columns:
            cluster_means = self.data.groupby('cluster')['speed_std'].mean().sort_index()
            axes[0, 2].bar(cluster_means.index, cluster_means.values, 
                          color=sns.color_palette('Set2', len(cluster_means)))
            axes[0, 2].set_xlabel('Cluster')
            axes[0, 2].set_ylabel('Speed Std Dev (km/h)')
            axes[0, 2].set_title('Speed Stability (lower = more stable)')
            axes[0, 2].grid(axis='y', alpha=0.3)
        
        # 4. 低速/中速/高速比例
        if all(col in self.data.columns for col in ['low_speed_ratio', 'medium_speed_ratio', 'high_speed_ratio']):
            cluster_speed_ratios = self.data.groupby('cluster')[
                ['low_speed_ratio', 'medium_speed_ratio', 'high_speed_ratio']
            ].mean()
            
            cluster_speed_ratios.plot(kind='bar', stacked=True, ax=axes[1, 0])
            axes[1, 0].set_xlabel('Cluster')
            axes[1, 0].set_ylabel('Ratio')
            axes[1, 0].set_title('Speed Category Distribution')
            axes[1, 0].legend(['Low (<40)', 'Medium (40-80)', 'High (>80)'])
            axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=0)
        
        # 5. 速度变化率
        if 'speed_change_rate_mean' in self.data.columns:
            self.data.boxplot(column='speed_change_rate_mean', by='cluster', ax=axes[1, 1])
            axes[1, 1].set_title('Speed Change Rate')
            axes[1, 1].set_xlabel('Cluster')
            axes[1, 1].set_ylabel('Change Rate')
            axes[1, 1].get_figure().suptitle('')
        
        # 6. 速度中位数 vs 平均速度
        if 'speed_mean' in self.data.columns and 'speed_median' in self.data.columns:
            for cluster_id in sorted(self.data['cluster'].unique()):
                cluster_data = self.data[self.data['cluster'] == cluster_id]
                axes[1, 2].scatter(cluster_data['speed_mean'], cluster_data['speed_median'],
                                  alpha=0.5, label=f'Cluster {cluster_id}', s=20)
            axes[1, 2].plot([0, 120], [0, 120], 'k--', alpha=0.3)
            axes[1, 2].set_xlabel('Mean Speed (km/h)')
            axes[1, 2].set_ylabel('Median Speed (km/h)')
            axes[1, 2].set_title('Speed Mean vs Median')
            axes[1, 2].legend()
            axes[1, 2].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/speed_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 速度分析图已保存: {save_dir}/speed_analysis.png")
        plt.close()
    
    def analyze_acceleration(self, save_dir='./results/paper_analysis'):
        """加速度特性分析"""
        os.makedirs(save_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("⚡ 加速度特性分析")
        print("="*60)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 平均加速度
        if 'acc_mean' in self.data.columns:
            self.data.boxplot(column='acc_mean', by='cluster', ax=axes[0, 0])
            axes[0, 0].set_title('Average Acceleration by Cluster')
            axes[0, 0].set_xlabel('Cluster')
            axes[0, 0].set_ylabel('Acceleration (m/s²)')
            axes[0, 0].get_figure().suptitle('')
            
            print("\n平均加速度统计:")
            acc_stats = self.data.groupby('cluster')['acc_mean'].describe()
            print(acc_stats)
        
        # 2. 加速度标准差（激进程度）
        if 'acc_std' in self.data.columns:
            cluster_means = self.data.groupby('cluster')['acc_std'].mean().sort_index()
            axes[0, 1].bar(cluster_means.index, cluster_means.values,
                          color=sns.color_palette('Reds', len(cluster_means)))
            axes[0, 1].set_xlabel('Cluster')
            axes[0, 1].set_ylabel('Acceleration Std Dev (m/s²)')
            axes[0, 1].set_title('Driving Aggressiveness')
            axes[0, 1].grid(axis='y', alpha=0.3)
        
        # 3. 急加速次数
        if 'harsh_accel' in self.data.columns:
            cluster_means = self.data.groupby('cluster')['harsh_accel'].mean().sort_index()
            axes[0, 2].bar(cluster_means.index, cluster_means.values,
                          color=sns.color_palette('Oranges', len(cluster_means)))
            axes[0, 2].set_xlabel('Cluster')
            axes[0, 2].set_ylabel('Harsh Accel Count')
            axes[0, 2].set_title('Harsh Acceleration Events')
            axes[0, 2].grid(axis='y', alpha=0.3)
            
            print("\n急加速统计:")
            print(self.data.groupby('cluster')['harsh_accel'].describe())
        
        # 4. 急减速次数
        if 'harsh_decel' in self.data.columns:
            cluster_means = self.data.groupby('cluster')['harsh_decel'].mean().sort_index()
            axes[1, 0].bar(cluster_means.index, cluster_means.values,
                          color=sns.color_palette('Blues', len(cluster_means)))
            axes[1, 0].set_xlabel('Cluster')
            axes[1, 0].set_ylabel('Harsh Decel Count')
            axes[1, 0].set_title('Harsh Deceleration Events')
            axes[1, 0].grid(axis='y', alpha=0.3)
        
        # 5. 急加速 vs 急减速散点图
        if 'harsh_accel' in self.data.columns and 'harsh_decel' in self.data.columns:
            for cluster_id in sorted(self.data['cluster'].unique()):
                cluster_data = self.data[self.data['cluster'] == cluster_id]
                axes[1, 1].scatter(cluster_data['harsh_accel'], cluster_data['harsh_decel'],
                                  alpha=0.5, label=f'Cluster {cluster_id}', s=20)
            axes[1, 1].set_xlabel('Harsh Accel Count')
            axes[1, 1].set_ylabel('Harsh Decel Count')
            axes[1, 1].set_title('Harsh Accel vs Decel')
            axes[1, 1].legend()
            axes[1, 1].grid(alpha=0.3)
        
        # 6. 加速度范围（max - min）
        if 'acc_max' in self.data.columns and 'acc_min' in self.data.columns:
            self.data['acc_range'] = self.data['acc_max'] - self.data['acc_min']
            self.data.boxplot(column='acc_range', by='cluster', ax=axes[1, 2])
            axes[1, 2].set_title('Acceleration Range')
            axes[1, 2].set_xlabel('Cluster')
            axes[1, 2].set_ylabel('Range (m/s²)')
            axes[1, 2].get_figure().suptitle('')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/acceleration_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 加速度分析图已保存: {save_dir}/acceleration_analysis.png")
        plt.close()
    
    def analyze_energy(self, save_dir='./results/paper_analysis'):
        """能耗特性分析"""
        os.makedirs(save_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("🔋 能耗特性分析")
        print("="*60)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. SOC下降
        if 'soc_drop_total' in self.data.columns:
            self.data.boxplot(column='soc_drop_total', by='cluster', ax=axes[0, 0])
            axes[0, 0].set_title('SOC Drop by Cluster')
            axes[0, 0].set_xlabel('Cluster')
            axes[0, 0].set_ylabel('SOC Drop (%)')
            axes[0, 0].get_figure().suptitle('')
            
            print("\nSOC下降统计:")
            print(self.data.groupby('cluster')['soc_drop_total'].describe())
        
        # 2. 平均功率
        if 'power_mean' in self.data.columns:
            cluster_means = self.data.groupby('cluster')['power_mean'].mean().sort_index()
            axes[0, 1].bar(cluster_means.index, cluster_means.values,
                          color=sns.color_palette('Greens', len(cluster_means)))
            axes[0, 1].set_xlabel('Cluster')
            axes[0, 1].set_ylabel('Power (kW)')
            axes[0, 1].set_title('Average Power Consumption')
            axes[0, 1].grid(axis='y', alpha=0.3)
        
        # 3. 能效（Wh/km）
        if 'efficiency_mean' in self.data.columns:
            # 过滤极端值
            eff_data = self.data[self.data['efficiency_mean'] < self.data['efficiency_mean'].quantile(0.95)]
            eff_data.boxplot(column='efficiency_mean', by='cluster', ax=axes[0, 2])
            axes[0, 2].set_title('Energy Efficiency')
            axes[0, 2].set_xlabel('Cluster')
            axes[0, 2].set_ylabel('Wh/km')
            axes[0, 2].get_figure().suptitle('')
        
        # 4. 再生制动比例
        if 'regen_braking_ratio' in self.data.columns:
            cluster_means = self.data.groupby('cluster')['regen_braking_ratio'].mean().sort_index()
            axes[1, 0].bar(cluster_means.index, cluster_means.values * 100,
                          color=sns.color_palette('YlGn', len(cluster_means)))
            axes[1, 0].set_xlabel('Cluster')
            axes[1, 0].set_ylabel('Regen Braking (%)')
            axes[1, 0].set_title('Regenerative Braking Usage')
            axes[1, 0].grid(axis='y', alpha=0.3)
        
        # 5. 功率 vs 速度
        if 'power_mean' in self.data.columns and 'speed_mean' in self.data.columns:
            for cluster_id in sorted(self.data['cluster'].unique()):
                cluster_data = self.data[self.data['cluster'] == cluster_id]
                axes[1, 1].scatter(cluster_data['speed_mean'], cluster_data['power_mean'],
                                  alpha=0.5, label=f'Cluster {cluster_id}', s=20)
            axes[1, 1].set_xlabel('Speed (km/h)')
            axes[1, 1].set_ylabel('Power (kW)')
            axes[1, 1].set_title('Power vs Speed')
            axes[1, 1].legend()
            axes[1, 1].grid(alpha=0.3)
        
        # 6. 能耗总量
        if 'energy_consumption_total' in self.data.columns:
            cluster_means = self.data.groupby('cluster')['energy_consumption_total'].mean().sort_index()
            axes[1, 2].bar(cluster_means.index, cluster_means.values,
                          color=sns.color_palette('Purples', len(cluster_means)))
            axes[1, 2].set_xlabel('Cluster')
            axes[1, 2].set_ylabel('Energy (kWh)')
            axes[1, 2].set_title('Total Energy Consumption')
            axes[1, 2].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/energy_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 能耗分析图已保存: {save_dir}/energy_analysis.png")
        plt.close()
    
    def analyze_time_distance(self, save_dir='./results/paper_analysis'):
        """时间与距离分析"""
        os.makedirs(save_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("⏱️  时间与距离分析")
        print("="*60)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 行驶距离
        if 'distance_total' in self.data.columns:
            self.data.boxplot(column='distance_total', by='cluster', ax=axes[0, 0])
            axes[0, 0].set_title('Trip Distance by Cluster')
            axes[0, 0].set_xlabel('Cluster')
            axes[0, 0].set_ylabel('Distance (km)')
            axes[0, 0].get_figure().suptitle('')
            
            print("\n行驶距离统计:")
            print(self.data.groupby('cluster')['distance_total'].describe())
        
        # 2. 持续时间
        if 'duration_minutes' in self.data.columns:
            self.data.boxplot(column='duration_minutes', by='cluster', ax=axes[0, 1])
            axes[0, 1].set_title('Trip Duration by Cluster')
            axes[0, 1].set_xlabel('Cluster')
            axes[0, 1].set_ylabel('Duration (minutes)')
            axes[0, 1].get_figure().suptitle('')
        
        # 3. 距离 vs 时间
        if 'distance_total' in self.data.columns and 'duration_minutes' in self.data.columns:
            for cluster_id in sorted(self.data['cluster'].unique()):
                cluster_data = self.data[self.data['cluster'] == cluster_id]
                axes[1, 0].scatter(cluster_data['duration_minutes'], cluster_data['distance_total'],
                                  alpha=0.5, label=f'Cluster {cluster_id}', s=20)
            axes[1, 0].set_xlabel('Duration (minutes)')
            axes[1, 0].set_ylabel('Distance (km)')
            axes[1, 0].set_title('Distance vs Duration')
            axes[1, 0].legend()
            axes[1, 0].grid(alpha=0.3)
        
        # 4. 停车次数
        if 'stop_count' in self.data.columns:
            cluster_means = self.data.groupby('cluster')['stop_count'].mean().sort_index()
            axes[1, 1].bar(cluster_means.index, cluster_means.values,
                          color=sns.color_palette('Set3', len(cluster_means)))
            axes[1, 1].set_xlabel('Cluster')
            axes[1, 1].set_ylabel('Stop Count')
            axes[1, 1].set_title('Average Stop Count')
            axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/time_distance_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 时间距离分析图已保存: {save_dir}/time_distance_analysis.png")
        plt.close()
    
    def generate_summary_table(self, save_dir='./results/paper_analysis'):
        """生成汇总统计表（用于论文）"""
        os.makedirs(save_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("📋 生成汇总统计表")
        print("="*60)
        
        key_features = [
            'speed_mean', 'speed_max', 'speed_std',
            'acc_mean', 'harsh_accel', 'harsh_decel',
            'power_mean', 'soc_drop_total',
            'distance_total', 'duration_minutes'
        ]
        
        available_features = [f for f in key_features if f in self.data.columns]
        
        summary = self.data.groupby('cluster')[available_features].agg(['mean', 'std', 'median'])
        
        # 保存为CSV
        summary.to_csv(f'{save_dir}/cluster_summary_statistics.csv')
        print(f"✅ 汇总统计表已保存: {save_dir}/cluster_summary_statistics.csv")
        
        # 打印到控制台
        print("\n各簇关键特征统计:")
        print(summary.to_string())
        
        return summary
    
    def run_full_analysis(self):
        """运行完整分析"""
        print("\n" + "="*70)
        print(f"📊 开始论文级别分析 - {self.model_name}")
        print("="*70)
        
        save_dir = './results/paper_analysis'
        os.makedirs(save_dir, exist_ok=True)
        
        # 运行所有分析
        self.analyze_speed(save_dir)
        self.analyze_acceleration(save_dir)
        self.analyze_energy(save_dir)
        self.analyze_time_distance(save_dir)
        summary = self.generate_summary_table(save_dir)
        
        print("\n" + "="*70)
        print("✅ 所有分析完成！")
        print("="*70)
        print(f"📁 结果保存在: {save_dir}/")
        print("   - speed_analysis.png")
        print("   - acceleration_analysis.png")
        print("   - energy_analysis.png")
        print("   - time_distance_analysis.png")
        print("   - cluster_summary_statistics.csv")
        
        return summary


# 使用示例
if __name__ == "__main__":
    # 读取数据
    features_df = pd.read_csv('./results/features/combined_features.csv')
    
    # 读取最佳模型的聚类结果（这里用DEC作为例子）
    results_df = pd.read_csv('./results/dec/clustered_results.csv')
    
    # 创建分析器
    analyzer = PaperAnalysis(features_df, results_df, model_name='DEC')
    
    # 运行完整分析
    summary = analyzer.run_full_analysis()
    
    print("\n🎉 论文分析完成！")