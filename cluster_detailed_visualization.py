"""
超详细聚类可视化分析系统
使用最佳模型（Transformer-AE）的聚类结果
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from scipy import stats
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams['figure.facecolor'] = 'white'

class DetailedClusterVisualizer:
    """详细聚类可视化分析器"""
    
    def __init__(self, model_name='Transformer-AE'):
        self.model_name = model_name
        self.model_dir = model_name.lower().replace('-', '_')
        self.colors = sns.color_palette('Set2', 10)
        
    def load_data(self):
        """加载所有数据"""
        print("\n" + "="*70)
        print(f"📂 加载 {self.model_name} 数据")
        print("="*70)
        
        # 1. 加载聚类结果
        results_file = f'./results/{self.model_dir}/clustered_results.csv'
        if not os.path.exists(results_file):
            print(f"❌ 找不到结果文件: {results_file}")
            return False
        
        self.results_df = pd.read_csv(results_file)
        print(f"✅ 聚类结果: {len(self.results_df)} 事件, {self.results_df['cluster'].nunique()} 簇")
        
        # 2. 加载特征
        self.features_df = pd.read_csv('./results/features/combined_features.csv')
        print(f"✅ 特征数据: {self.features_df.shape}")
        
        # 3. 加载原始事件数据
        events_file = './results/events/events.pkl'
        if os.path.exists(events_file):
            with open(events_file, 'rb') as f:
                self.events = pickle.load(f)
            print(f"✅ 原始事件: {len(self.events)} 个")
        else:
            self.events = None
            print("⚠️  原始事件数据不存在")
        
        # 4. 合并数据
        self.data = self.features_df.merge(self.results_df, on=['event_id', 'vehicle_id'])
        
        # 5. 过滤驾驶事件
        self.driving_data = self.data[
            (self.data['speed_mean'] > 5) &
            (self.data['distance_total'] > 0.5) &
            (self.data['moving_ratio'] > 0.3)
        ].copy()
        
        print(f"✅ 驾驶事件: {len(self.driving_data)} ({len(self.driving_data)/len(self.data)*100:.1f}%)")
        print(f"   簇分布: {self.driving_data['cluster'].value_counts().sort_index().to_dict()}")
        
        return True
    
    def plot_overview(self, save_dir):
        """总览图：簇分布和基本统计"""
        print("\n📊 生成总览图...")
        
        fig = plt.figure(figsize=(20, 10))
        gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. 簇大小分布（饼图）
        ax1 = fig.add_subplot(gs[0, 0])
        cluster_counts = self.driving_data['cluster'].value_counts().sort_index()
        colors_pie = [self.colors[i % len(self.colors)] for i in range(len(cluster_counts))]
        wedges, texts, autotexts = ax1.pie(cluster_counts.values, 
                                           labels=[f'簇 {i}' for i in cluster_counts.index],
                                           autopct='%1.1f%%',
                                           colors=colors_pie,
                                           startangle=90)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        ax1.set_title('簇分布', fontsize=14, fontweight='bold')
        
        # 2. 簇大小柱状图
        ax2 = fig.add_subplot(gs[0, 1])
        bars = ax2.bar(cluster_counts.index, cluster_counts.values, color=colors_pie)
        ax2.set_xlabel('簇 ID', fontsize=11)
        ax2.set_ylabel('事件数', fontsize=11)
        ax2.set_title('各簇事件数量', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        for i, (bar, count) in enumerate(zip(bars, cluster_counts.values)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # 3. 速度分布对比（小提琴图）
        ax3 = fig.add_subplot(gs[0, 2:])
        cluster_order = sorted(self.driving_data['cluster'].unique())
        violin_parts = ax3.violinplot([self.driving_data[self.driving_data['cluster']==c]['speed_mean'].values 
                                       for c in cluster_order],
                                      positions=cluster_order,
                                      showmeans=True,
                                      showmedians=True)
        
        for pc, color in zip(violin_parts['bodies'], [self.colors[i % len(self.colors)] for i in cluster_order]):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        ax3.set_xlabel('簇 ID', fontsize=11)
        ax3.set_ylabel('平均速度 (km/h)', fontsize=11)
        ax3.set_title('各簇速度分布（小提琴图）', fontsize=12, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. 关键特征热力图
        ax4 = fig.add_subplot(gs[1, :2])
        key_features = ['speed_mean', 'speed_max', 'acc_std', 'harsh_accel', 
                       'harsh_decel', 'power_mean', 'distance_total', 'duration_minutes']
        available_features = [f for f in key_features if f in self.driving_data.columns]
        
        cluster_profiles = self.driving_data.groupby('cluster')[available_features].mean()
        
        # 标准化
        cluster_profiles_norm = (cluster_profiles - cluster_profiles.min()) / (cluster_profiles.max() - cluster_profiles.min())
        
        im = ax4.imshow(cluster_profiles_norm.T, cmap='RdYlGn', aspect='auto')
        ax4.set_xticks(range(len(cluster_profiles)))
        ax4.set_xticklabels([f'簇 {i}' for i in cluster_profiles.index])
        ax4.set_yticks(range(len(available_features)))
        ax4.set_yticklabels([f.replace('_', ' ').title() for f in available_features])
        ax4.set_title('各簇特征热力图（标准化）', fontsize=12, fontweight='bold')
        
        # 添加数值标签
        for i in range(len(available_features)):
            for j in range(len(cluster_profiles)):
                text = ax4.text(j, i, f'{cluster_profiles_norm.iloc[j, i]:.2f}',
                               ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im, ax=ax4, label='标准化值')
        
        # 5. 簇命名和特征摘要
        ax5 = fig.add_subplot(gs[1, 2:])
        ax5.axis('off')
        
        summary_text = "各簇特征摘要\n" + "="*50 + "\n\n"
        
        for cluster_id in sorted(self.driving_data['cluster'].unique()):
            cluster_data = self.driving_data[self.driving_data['cluster'] == cluster_id]
            
            name = self._auto_name_cluster(cluster_data)
            
            summary_text += f"簇 {cluster_id}: {name}\n"
            summary_text += f"  事件数: {len(cluster_data)}\n"
            summary_text += f"  平均速度: {cluster_data['speed_mean'].mean():.1f} km/h\n"
            summary_text += f"  急加速: {cluster_data['harsh_accel'].mean():.1f} 次\n"
            summary_text += f"  平均功率: {cluster_data['power_mean'].mean():.1f} kW\n"
            summary_text += f"  行驶距离: {cluster_data['distance_total'].mean():.1f} km\n\n"
        
        ax5.text(0.05, 0.95, summary_text, 
                transform=ax5.transAxes,
                fontsize=10,
                verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.suptitle(f'{self.model_name} 聚类结果总览', fontsize=16, fontweight='bold')
        plt.savefig(f'{save_dir}/01_overview.png', dpi=300, bbox_inches='tight')
        print(f"  ✅ 总览图: 01_overview.png")
        plt.close()
    
    def plot_speed_analysis(self, save_dir):
        """速度详细分析"""
        print("\n🚗 生成速度分析图...")
        
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. 平均速度箱线图
        ax1 = fig.add_subplot(gs[0, 0])
        bp = ax1.boxplot([self.driving_data[self.driving_data['cluster']==c]['speed_mean'].values 
                          for c in sorted(self.driving_data['cluster'].unique())],
                         labels=[f'簇{c}' for c in sorted(self.driving_data['cluster'].unique())],
                         patch_artist=True)
        for patch, color in zip(bp['boxes'], self.colors):
            patch.set_facecolor(color)
        ax1.set_ylabel('平均速度 (km/h)', fontsize=11)
        ax1.set_title('平均速度分布', fontsize=12, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. 最大速度对比
        ax2 = fig.add_subplot(gs[0, 1])
        speed_max = self.driving_data.groupby('cluster')['speed_max'].mean().sort_index()
        bars = ax2.bar(speed_max.index, speed_max.values, 
                      color=[self.colors[i % len(self.colors)] for i in range(len(speed_max))])
        ax2.set_xlabel('簇 ID', fontsize=11)
        ax2.set_ylabel('最大速度 (km/h)', fontsize=11)
        ax2.set_title('各���最大速度', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, speed_max.values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. 速度稳定性（标准差）
        ax3 = fig.add_subplot(gs[0, 2])
        speed_std = self.driving_data.groupby('cluster')['speed_std'].mean().sort_index()
        bars = ax3.bar(speed_std.index, speed_std.values,
                      color=sns.color_palette('RdYlGn_r', len(speed_std)))
        ax3.set_xlabel('簇 ID', fontsize=11)
        ax3.set_ylabel('速度标准差 (km/h)', fontsize=11)
        ax3.set_title('速度稳定性（越低越稳定）', fontsize=12, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. 速度范围（min-max）
        ax4 = fig.add_subplot(gs[1, 0])
        cluster_ids = sorted(self.driving_data['cluster'].unique())
        speed_ranges = []
        for c in cluster_ids:
            cluster_data = self.driving_data[self.driving_data['cluster'] == c]
            speed_ranges.append([
                cluster_data['speed_mean'].min(),
                cluster_data['speed_mean'].quantile(0.25),
                cluster_data['speed_mean'].median(),
                cluster_data['speed_mean'].quantile(0.75),
                cluster_data['speed_mean'].max()
            ])
        
        for i, (c, ranges) in enumerate(zip(cluster_ids, speed_ranges)):
            color = self.colors[i % len(self.colors)]
            ax4.plot([i, i], [ranges[0], ranges[4]], 'k-', linewidth=2)
            ax4.plot([i-0.2, i+0.2], [ranges[0], ranges[0]], 'k-', linewidth=2)
            ax4.plot([i-0.2, i+0.2], [ranges[4], ranges[4]], 'k-', linewidth=2)
            rect = plt.Rectangle((i-0.3, ranges[1]), 0.6, ranges[3]-ranges[1],
                                facecolor=color, alpha=0.7, edgecolor='black')
            ax4.add_patch(rect)
            ax4.plot(i, ranges[2], 'wo', markersize=8, markeredgecolor='black', markeredgewidth=2)
        
        ax4.set_xticks(range(len(cluster_ids)))
        ax4.set_xticklabels([f'簇{c}' for c in cluster_ids])
        ax4.set_ylabel('速度 (km/h)', fontsize=11)
        ax4.set_title('速度范围（箱线图）', fontsize=12, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)
        
        # 5. 速度类别分布（低中高速）
        ax5 = fig.add_subplot(gs[1, 1])
        if all(col in self.driving_data.columns for col in ['low_speed_ratio', 'medium_speed_ratio', 'high_speed_ratio']):
            speed_categories = self.driving_data.groupby('cluster')[
                ['low_speed_ratio', 'medium_speed_ratio', 'high_speed_ratio']
            ].mean() * 100
            
            x = range(len(speed_categories))
            width = 0.6
            
            ax5.bar(x, speed_categories['low_speed_ratio'], width, 
                   label='低速 (<40 km/h)', color='#FFA07A')
            ax5.bar(x, speed_categories['medium_speed_ratio'], width,
                   bottom=speed_categories['low_speed_ratio'],
                   label='中速 (40-80 km/h)', color='#87CEEB')
            ax5.bar(x, speed_categories['high_speed_ratio'], width,
                   bottom=speed_categories['low_speed_ratio'] + speed_categories['medium_speed_ratio'],
                   label='高速 (>80 km/h)', color='#90EE90')
            
            ax5.set_xticks(x)
            ax5.set_xticklabels([f'簇{c}' for c in speed_categories.index])
            ax5.set_ylabel('比例 (%)', fontsize=11)
            ax5.set_title('速度类别分布', fontsize=12, fontweight='bold')
            ax5.legend()
            ax5.grid(axis='y', alpha=0.3)
        
        # 6. 速度直方图对比
        ax6 = fig.add_subplot(gs[1, 2])
        for c in sorted(self.driving_data['cluster'].unique()):
            cluster_data = self.driving_data[self.driving_data['cluster'] == c]
            ax6.hist(cluster_data['speed_mean'], bins=20, alpha=0.5,
                    label=f'簇{c}', color=self.colors[c % len(self.colors)])
        ax6.set_xlabel('平均速度 (km/h)', fontsize=11)
        ax6.set_ylabel('频数', fontsize=11)
        ax6.set_title('速度分布直方图', fontsize=12, fontweight='bold')
        ax6.legend()
        ax6.grid(alpha=0.3)
        
        # 7-9. 速度时间序列（抽样）
        if self.events is not None:
            for idx, cluster_id in enumerate(sorted(self.driving_data['cluster'].unique())[:3]):
                ax = fig.add_subplot(gs[2, idx])
                
                cluster_event_ids = self.driving_data[self.driving_data['cluster'] == cluster_id]['event_id'].values[:3]
                
                for event_id in cluster_event_ids:
                    event = next((e for e in self.events if e['event_id'] == event_id), None)
                    if event is not None and 'spd' in event['data'].columns:
                        speed_data = event['data']['spd'].values[:100]
                        ax.plot(speed_data, alpha=0.7, linewidth=1)
                
                ax.set_xlabel('时间步', fontsize=10)
                ax.set_ylabel('速度 (km/h)', fontsize=10)
                ax.set_title(f'簇{cluster_id} 速度时间序列（样例）', fontsize=11, fontweight='bold')
                ax.grid(alpha=0.3)
        
        plt.suptitle(f'{self.model_name} - 速度特性详细分析', fontsize=16, fontweight='bold')
        plt.savefig(f'{save_dir}/02_speed_analysis.png', dpi=300, bbox_inches='tight')
        print(f"  ✅ 速度分析图: 02_speed_analysis.png")
        plt.close()
    
    def plot_energy_analysis(self, save_dir):
        """能耗详细分析"""
        print("\n🔋 生成能耗分析图...")
        
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. 平均功率对比
        ax1 = fig.add_subplot(gs[0, 0])
        power_mean = self.driving_data.groupby('cluster')['power_mean'].mean().sort_index()
        bars = ax1.bar(power_mean.index, power_mean.values,
                      color=sns.color_palette('YlOrRd', len(power_mean)))
        ax1.set_xlabel('簇 ID', fontsize=11)
        ax1.set_ylabel('平均功率 (kW)', fontsize=11)
        ax1.set_title('各簇平均功率', fontsize=12, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, power_mean.values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. SOC下降（能耗）
        ax2 = fig.add_subplot(gs[0, 1])
        self.driving_data['soc_drop_abs'] = abs(self.driving_data['soc_drop_total'])
        soc_drop = self.driving_data.groupby('cluster')['soc_drop_abs'].mean().sort_index()
        bars = ax2.bar(soc_drop.index, soc_drop.values,
                      color=sns.color_palette('Greens_r', len(soc_drop)))
        ax2.set_xlabel('簇 ID', fontsize=11)
        ax2.set_ylabel('SOC 下降 (%)', fontsize=11)
        ax2.set_title('能量消耗（SOC）', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, soc_drop.values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. 功率箱线图
        ax3 = fig.add_subplot(gs[0, 2])
        bp = ax3.boxplot([self.driving_data[self.driving_data['cluster']==c]['power_mean'].values 
                         for c in sorted(self.driving_data['cluster'].unique())],
                        labels=[f'簇{c}' for c in sorted(self.driving_data['cluster'].unique())],
                        patch_artist=True)
        for patch, color in zip(bp['boxes'], self.colors):
            patch.set_facecolor(color)
        ax3.set_ylabel('功率 (kW)', fontsize=11)
        ax3.set_title('功率分布', fontsize=12, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. 功率 vs 速度散点图
        ax4 = fig.add_subplot(gs[1, 0])
        for c in sorted(self.driving_data['cluster'].unique()):
            cluster_data = self.driving_data[self.driving_data['cluster'] == c]
            ax4.scatter(cluster_data['speed_mean'], cluster_data['power_mean'],
                       alpha=0.6, s=30, label=f'簇{c}', 
                       color=self.colors[c % len(self.colors)])
        ax4.set_xlabel('平均速度 (km/h)', fontsize=11)
        ax4.set_ylabel('平均功率 (kW)', fontsize=11)
        ax4.set_title('功率 vs 速度关系', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(alpha=0.3)
        
        # 5. 能效（如果有）
        ax5 = fig.add_subplot(gs[1, 1])
        if 'efficiency_mean' in self.driving_data.columns:
            eff_data = self.driving_data[self.driving_data['efficiency_mean'] < 
                                        self.driving_data['efficiency_mean'].quantile(0.95)]
            bp = ax5.boxplot([eff_data[eff_data['cluster']==c]['efficiency_mean'].values 
                             for c in sorted(eff_data['cluster'].unique())],
                            labels=[f'簇{c}' for c in sorted(eff_data['cluster'].unique())],
                            patch_artist=True)
            for patch, color in zip(bp['boxes'], self.colors):
                patch.set_facecolor(color)
            ax5.set_ylabel('能效 (Wh/km)', fontsize=11)
            ax5.set_title('能效分布', fontsize=12, fontweight='bold')
            ax5.grid(axis='y', alpha=0.3)
        
        # 6. 用电时长
        ax6 = fig.add_subplot(gs[1, 2])
        duration = self.driving_data.groupby('cluster')['duration_minutes'].mean().sort_index()
        bars = ax6.bar(duration.index, duration.values,
                      color=sns.color_palette('Purples', len(duration)))
        ax6.set_xlabel('簇 ID', fontsize=11)
        ax6.set_ylabel('平均时长 (分钟)', fontsize=11)
        ax6.set_title('用电时长', fontsize=12, fontweight='bold')
        ax6.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, duration.values):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 7. 能耗 vs 距离
        ax7 = fig.add_subplot(gs[2, 0])
        for c in sorted(self.driving_data['cluster'].unique()):
            cluster_data = self.driving_data[self.driving_data['cluster'] == c]
            ax7.scatter(cluster_data['distance_total'], cluster_data['soc_drop_abs'],
                       alpha=0.6, s=30, label=f'簇{c}',
                       color=self.colors[c % len(self.colors)])
        ax7.set_xlabel('行驶距离 (km)', fontsize=11)
        ax7.set_ylabel('SOC下降 (%)', fontsize=11)
        ax7.set_title('能耗 vs 距离', fontsize=12, fontweight='bold')
        ax7.legend()
        ax7.grid(alpha=0.3)
        
        # 8. 电压电流（如果有）
        ax8 = fig.add_subplot(gs[2, 1])
        if 'voltage_mean' in self.driving_data.columns and 'current_mean' in self.driving_data.columns:
            voltage = self.driving_data.groupby('cluster')['voltage_mean'].mean()
            current = self.driving_data.groupby('cluster')['current_mean'].mean()
            
            x = np.arange(len(voltage))
            width = 0.35
            
            ax8_twin = ax8.twinx()
            bars1 = ax8.bar(x - width/2, voltage.values, width, label='电压', color='orange')
            bars2 = ax8_twin.bar(x + width/2, current.values, width, label='电流', color='blue')
            
            ax8.set_xlabel('簇 ID', fontsize=11)
            ax8.set_ylabel('电压 (V)', fontsize=11, color='orange')
            ax8_twin.set_ylabel('电流 (A)', fontsize=11, color='blue')
            ax8.set_title('电压电流对比', fontsize=12, fontweight='bold')
            ax8.set_xticks(x)
            ax8.set_xticklabels([f'簇{c}' for c in voltage.index])
            ax8.tick_params(axis='y', labelcolor='orange')
            ax8_twin.tick_params(axis='y', labelcolor='blue')
            ax8.grid(alpha=0.3)
        
        # 9. 总能耗对比
        ax9 = fig.add_subplot(gs[2, 2])
        if 'energy_consumption_total' in self.driving_data.columns:
            energy_total = self.driving_data.groupby('cluster')['energy_consumption_total'].mean().sort_index()
            bars = ax9.bar(energy_total.index, abs(energy_total.values),
                          color=sns.color_palette('Reds', len(energy_total)))
            ax9.set_xlabel('簇 ID', fontsize=11)
            ax9.set_ylabel('总能耗 (kWh)', fontsize=11)
            ax9.set_title('总能量消耗', fontsize=12, fontweight='bold')
            ax9.grid(axis='y', alpha=0.3)
        
        plt.suptitle(f'{self.model_name} - 能耗特性详细分析', fontsize=16, fontweight='bold')
        plt.savefig(f'{save_dir}/03_energy_analysis.png', dpi=300, bbox_inches='tight')
        print(f"  ✅ 能耗分析图: 03_energy_analysis.png")
        plt.close()
    
    def plot_driving_behavior(self, save_dir):
        """驾驶行为详细分析"""
        print("\n⚡ 生成驾驶行为分析图...")
        
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. 急加速/急减速对比
        ax1 = fig.add_subplot(gs[0, 0])
        harsh_stats = self.driving_data.groupby('cluster')[['harsh_accel', 'harsh_decel']].mean()
        
        x = np.arange(len(harsh_stats))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, harsh_stats['harsh_accel'], width,
                       label='急加速', color='#FF6B6B')
        bars2 = ax1.bar(x + width/2, harsh_stats['harsh_decel'], width,
                       label='急减速', color='#4ECDC4')
        
        ax1.set_xlabel('簇 ID', fontsize=11)
        ax1.set_ylabel('次数', fontsize=11)
        ax1.set_title('急加速/急减速对比', fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'簇{c}' for c in harsh_stats.index])
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. 加速度标准差（激进程度）
        ax2 = fig.add_subplot(gs[0, 1])
        acc_std = self.driving_data.groupby('cluster')['acc_std'].mean().sort_index()
        bars = ax2.bar(acc_std.index, acc_std.values,
                      color=sns.color_palette('Reds', len(acc_std)))
        ax2.set_xlabel('簇 ID', fontsize=11)
        ax2.set_ylabel('加速度标准差 (m/s²)', fontsize=11)
        ax2.set_title('驾驶激进程度', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, acc_std.values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. 急加速 vs 急减速散点图
        ax3 = fig.add_subplot(gs[0, 2])
        for c in sorted(self.driving_data['cluster'].unique()):
            cluster_data = self.driving_data[self.driving_data['cluster'] == c]
            ax3.scatter(cluster_data['harsh_accel'], cluster_data['harsh_decel'],
                       alpha=0.6, s=50, label=f'簇{c}',
                       color=self.colors[c % len(self.colors)])
        ax3.set_xlabel('急加速次数', fontsize=11)
        ax3.set_ylabel('急减速次数', fontsize=11)
        ax3.set_title('激进驾驶模式', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # 4. 行驶距离分布
        ax4 = fig.add_subplot(gs[1, 0])
        bp = ax4.boxplot([self.driving_data[self.driving_data['cluster']==c]['distance_total'].values 
                         for c in sorted(self.driving_data['cluster'].unique())],
                        labels=[f'簇{c}' for c in sorted(self.driving_data['cluster'].unique())],
                        patch_artist=True)
        for patch, color in zip(bp['boxes'], self.colors):
            patch.set_facecolor(color)
        ax4.set_ylabel('行驶距离 (km)', fontsize=11)
        ax4.set_title('行驶距离分布', fontsize=12, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)
        
        # 5. 停车次数
        ax5 = fig.add_subplot(gs[1, 1])
        if 'stop_count' in self.driving_data.columns:
            stop_count = self.driving_data.groupby('cluster')['stop_count'].mean().sort_index()
            bars = ax5.bar(stop_count.index, stop_count.values,
                          color=sns.color_palette('Blues', len(stop_count)))
            ax5.set_xlabel('簇 ID', fontsize=11)
            ax5.set_ylabel('平均停车次数', fontsize=11)
            ax5.set_title('停车频率', fontsize=12, fontweight='bold')
            ax5.grid(axis='y', alpha=0.3)
        
        # 6. 移动比例
        ax6 = fig.add_subplot(gs[1, 2])
        moving_ratio = self.driving_data.groupby('cluster')['moving_ratio'].mean().sort_index() * 100
        bars = ax6.bar(moving_ratio.index, moving_ratio.values,
                      color=sns.color_palette('Greens', len(moving_ratio)))
        ax6.set_xlabel('簇 ID', fontsize=11)
        ax6.set_ylabel('移动比例 (%)', fontsize=11)
        ax6.set_title('移动时间占比', fontsize=12, fontweight='bold')
        ax6.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, moving_ratio.values):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 7-9. 加速度时间序列（抽样）
        if self.events is not None:
            for idx, cluster_id in enumerate(sorted(self.driving_data['cluster'].unique())[:3]):
                ax = fig.add_subplot(gs[2, idx])
                
                cluster_event_ids = self.driving_data[self.driving_data['cluster'] == cluster_id]['event_id'].values[:3]
                
                for event_id in cluster_event_ids:
                    event = next((e for e in self.events if e['event_id'] == event_id), None)
                    if event is not None and 'acc' in event['data'].columns:
                        acc_data = event['data']['acc'].values[:100]
                        ax.plot(acc_data, alpha=0.7, linewidth=1)
                
                ax.set_xlabel('时间步', fontsize=10)
                ax.set_ylabel('加速度 (m/s²)', fontsize=10)
                ax.set_title(f'簇{cluster_id} 加速度时间序列', fontsize=11, fontweight='bold')
                ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
                ax.grid(alpha=0.3)
        
        plt.suptitle(f'{self.model_name} - 驾驶行为详细分析', fontsize=16, fontweight='bold')
        plt.savefig(f'{save_dir}/04_driving_behavior.png', dpi=300, bbox_inches='tight')
        print(f"  ✅ 驾驶行为图: 04_driving_behavior.png")
        plt.close()
    
    def plot_time_series_detail(self, save_dir):
        """时间序列详细分析（如果有原始事件数据）"""
        if self.events is None:
            print("\n⚠️  跳过时间序列分析（无原始数据）")
            return
        
        print("\n📈 生成时间序列详细分析...")
        
        for cluster_id in sorted(self.driving_data['cluster'].unique()):
            fig = plt.figure(figsize=(20, 12))
            gs = GridSpec(4, 1, figure=fig, hspace=0.3)
            
            # 获取该簇的事件
            cluster_event_ids = self.driving_data[self.driving_data['cluster'] == cluster_id]['event_id'].values[:5]
            
            # 1. 速度
            ax1 = fig.add_subplot(gs[0, 0])
            for event_id in cluster_event_ids:
                event = next((e for e in self.events if e['event_id'] == event_id), None)
                if event is not None and 'spd' in event['data'].columns:
                    speed_data = event['data']['spd'].values[:200]
                    ax1.plot(speed_data, alpha=0.7, linewidth=1.5)
            ax1.set_ylabel('速度 (km/h)', fontsize=11)
            ax1.set_title(f'簇 {cluster_id} - 速度时间序列', fontsize=12, fontweight='bold')
            ax1.grid(alpha=0.3)
            
            # 2. 加速度
            ax2 = fig.add_subplot(gs[1, 0])
            for event_id in cluster_event_ids:
                event = next((e for e in self.events if e['event_id'] == event_id), None)
                if event is not None and 'acc' in event['data'].columns:
                    acc_data = event['data']['acc'].values[:200]
                    ax2.plot(acc_data, alpha=0.7, linewidth=1.5)
            ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax2.set_ylabel('加速度 (m/s²)', fontsize=11)
            ax2.set_title('加速度时间序列', fontsize=12, fontweight='bold')
            ax2.grid(alpha=0.3)
            
            # 3. SOC
            ax3 = fig.add_subplot(gs[2, 0])
            for event_id in cluster_event_ids:
                event = next((e for e in self.events if e['event_id'] == event_id), None)
                if event is not None and 'soc' in event['data'].columns:
                    soc_data = event['data']['soc'].values[:200]
                    ax3.plot(soc_data, alpha=0.7, linewidth=1.5)
            ax3.set_ylabel('SOC (%)', fontsize=11)
            ax3.set_title('SOC变化', fontsize=12, fontweight='bold')
            ax3.grid(alpha=0.3)
            
            # 4. 功率
            ax4 = fig.add_subplot(gs[3, 0])
            for event_id in cluster_event_ids:
                event = next((e for e in self.events if e['event_id'] == event_id), None)
                if event is not None and 'power' in event['data'].columns:
                    power_data = event['data']['power'].values[:200]
                    ax4.plot(power_data, alpha=0.7, linewidth=1.5)
            ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax4.set_xlabel('时间步', fontsize=11)
            ax4.set_ylabel('功率 (kW)', fontsize=11)
            ax4.set_title('功率时间序列', fontsize=12, fontweight='bold')
            ax4.grid(alpha=0.3)
            
            plt.suptitle(f'簇 {cluster_id} 时间序列详细分析（5个样例事件）', 
                        fontsize=16, fontweight='bold')
            plt.savefig(f'{save_dir}/05_timeseries_cluster_{cluster_id}.png', 
                       dpi=300, bbox_inches='tight')
            print(f"  ✅ 簇{cluster_id}时间序列: 05_timeseries_cluster_{cluster_id}.png")
            plt.close()
    
    def plot_overview(self, save_dir):
        """总览图：簇分布和基本统计（优化版）"""
        print("\n📊 生成总览图...")
        
        fig = plt.figure(figsize=(20, 10))
        gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. 簇大小分布（饼图）
        ax1 = fig.add_subplot(gs[0, 0])
        cluster_counts = self.driving_data['cluster'].value_counts().sort_index()
        colors_pie = [self.colors[i % len(self.colors)] for i in range(len(cluster_counts))]
        wedges, texts, autotexts = ax1.pie(cluster_counts.values, 
                                        labels=[f'簇 {i}' for i in cluster_counts.index],
                                        autopct='%1.1f%%',
                                        colors=colors_pie,
                                        startangle=90)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
        ax1.set_title('簇分布', fontsize=14, fontweight='bold')
        
        # 2. 簇大小柱状图（加对数刻度选项）
        ax2 = fig.add_subplot(gs[0, 1])
        bars = ax2.bar(cluster_counts.index, cluster_counts.values, color=colors_pie)
        ax2.set_xlabel('簇 ID', fontsize=11)
        ax2.set_ylabel('事件数', fontsize=11)
        ax2.set_title('各簇事件数量', fontsize=12, fontweight='bold')
        
        # ⭐ 如果分布差异太大，使用对数刻度
        if cluster_counts.max() / cluster_counts.min() > 10:
            ax2.set_yscale('log')
            ax2.set_ylabel('事件数 (对数刻度)', fontsize=11)
        
        ax2.grid(axis='y', alpha=0.3)
        
        for i, (bar, count) in enumerate(zip(bars, cluster_counts.values)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # 3. 速度分布对比（小提琴图）
        ax3 = fig.add_subplot(gs[0, 2:])
        cluster_order = sorted(self.driving_data['cluster'].unique())
        violin_parts = ax3.violinplot([self.driving_data[self.driving_data['cluster']==c]['speed_mean'].values 
                                    for c in cluster_order],
                                    positions=cluster_order,
                                    showmeans=True,
                                    showmedians=True)
        
        for pc, color in zip(violin_parts['bodies'], [self.colors[i % len(self.colors)] for i in cluster_order]):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        ax3.set_xlabel('簇 ID', fontsize=11)
        ax3.set_ylabel('平均速度 (km/h)', fontsize=11)
        ax3.set_title('各簇速度分布（小提琴图）', fontsize=12, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. 关键特征热力图
        ax4 = fig.add_subplot(gs[1, :2])
        key_features = ['speed_mean', 'speed_max', 'acc_std', 'harsh_accel', 
                    'harsh_decel', 'power_mean', 'distance_total', 'duration_minutes']
        available_features = [f for f in key_features if f in self.driving_data.columns]
        
        cluster_profiles = self.driving_data.groupby('cluster')[available_features].mean()
        
        # 标准化
        cluster_profiles_norm = (cluster_profiles - cluster_profiles.min()) / (cluster_profiles.max() - cluster_profiles.min())
        
        im = ax4.imshow(cluster_profiles_norm.T, cmap='RdYlGn', aspect='auto')
        ax4.set_xticks(range(len(cluster_profiles)))
        ax4.set_xticklabels([f'簇 {i}' for i in cluster_profiles.index])
        ax4.set_yticks(range(len(available_features)))
        ax4.set_yticklabels([f.replace('_', ' ').title() for f in available_features], fontsize=10)
        ax4.set_title('各簇特征热力图（标准化）', fontsize=12, fontweight='bold')
        
        # 添加数值标签
        for i in range(len(available_features)):
            for j in range(len(cluster_profiles)):
                text = ax4.text(j, i, f'{cluster_profiles_norm.iloc[j, i]:.2f}',
                            ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im, ax=ax4, label='标准化值')
        
        # 5. 簇命名和特征摘要（优化版）
        ax5 = fig.add_subplot(gs[1, 2:])
        ax5.axis('off')
        
        # ⭐ 优化文本格式
        summary_lines = []
        summary_lines.append("各簇特征摘要")
        summary_lines.append("="*60)
        summary_lines.append("")
        
        for cluster_id in sorted(self.driving_data['cluster'].unique()):
            cluster_data = self.driving_data[self.driving_data['cluster'] == cluster_id]
            name = self._auto_name_cluster(cluster_data)
            pct = len(cluster_data) / len(self.driving_data) * 100
            
            summary_lines.append(f"簇 {cluster_id}: {name}")
            summary_lines.append(f"  事件数: {len(cluster_data)} ({pct:.1f}%)")
            summary_lines.append(f"  速度: {cluster_data['speed_mean'].mean():.1f} km/h")
            summary_lines.append(f"  急加速: {cluster_data['harsh_accel'].mean():.1f} 次")
            summary_lines.append(f"  急减速: {cluster_data['harsh_decel'].mean():.1f} 次")
            summary_lines.append(f"  功率: {cluster_data['power_mean'].mean():.1f} kW")
            summary_lines.append(f"  距离: {cluster_data['distance_total'].mean():.1f} km")
            summary_lines.append("")  # 空行分隔
        
        summary_text = "\n".join(summary_lines)
        
        ax5.text(0.05, 0.98, summary_text, 
                transform=ax5.transAxes,
                fontsize=9,
                verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.2))
        
        plt.suptitle(f'{self.model_name} 聚类结果总览', fontsize=16, fontweight='bold')
        plt.savefig(f'{save_dir}/01_overview.png', dpi=300, bbox_inches='tight')
        print(f"  ✅ 总览图: 01_overview.png")
        plt.close()
    
    def generate_statistics_report(self, save_dir):
        """生成统计报告"""
        print("\n�� 生成统计报告...")
        
        # 计算详细统计
        key_features = ['speed_mean', 'speed_max', 'speed_std', 'acc_std', 
                       'harsh_accel', 'harsh_decel', 'power_mean', 'soc_drop_abs',
                       'distance_total', 'duration_minutes', 'moving_ratio']
        
        available = [f for f in key_features if f in self.driving_data.columns]
        
        summary_stats = self.driving_data.groupby('cluster')[available].agg(['mean', 'std', 'median', 'min', 'max'])
        summary_stats.to_csv(f'{save_dir}/cluster_detailed_statistics.csv')
        
        print(f"  ✅ 详细统计表: cluster_detailed_statistics.csv")
        
        # 生成Markdown报告
        report = f"# {self.model_name} 聚类结果详细报告\n\n"
        report += f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        report += "## 1. 总体概况\n\n"
        report += f"- 总事件数: {len(self.data)}\n"
        report += f"- 驾驶事件数: {len(self.driving_data)}\n"
        report += f"- 簇数量: {self.driving_data['cluster'].nunique()}\n\n"
        
        report += "## 2. 各簇详细特征\n\n"
        
        for cluster_id in sorted(self.driving_data['cluster'].unique()):
            cluster_data = self.driving_data[self.driving_data['cluster'] == cluster_id]
            name = self._auto_name_cluster(cluster_data)
            
            report += f"### 簇 {cluster_id}: {name}\n\n"
            report += f"- **事件数**: {len(cluster_data)} ({len(cluster_data)/len(self.driving_data)*100:.1f}%)\n"
            report += f"- **平均速度**: {cluster_data['speed_mean'].mean():.1f} km/h (±{cluster_data['speed_mean'].std():.1f})\n"
            report += f"- **最大速度**: {cluster_data['speed_max'].mean():.1f} km/h\n"
            report += f"- **速度稳定性**: {cluster_data['speed_std'].mean():.2f} km/h\n"
            report += f"- **急加速**: {cluster_data['harsh_accel'].mean():.1f} 次\n"
            report += f"- **急减速**: {cluster_data['harsh_decel'].mean():.1f} 次\n"
            report += f"- **平均功率**: {cluster_data['power_mean'].mean():.1f} kW\n"
            report += f"- **能耗(SOC)**: {cluster_data['soc_drop_abs'].mean():.1f}%\n"
            report += f"- **行驶距离**: {cluster_data['distance_total'].mean():.2f} km\n"
            report += f"- **行驶时长**: {cluster_data['duration_minutes'].mean():.1f} 分钟\n\n"
        
        with open(f'{save_dir}/cluster_detailed_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"  ✅ Markdown报告: cluster_detailed_report.md")
    
    def _auto_name_cluster(self, cluster_data):
        """自动命名簇"""
        speed = cluster_data['speed_mean'].mean()
        harsh_accel = cluster_data['harsh_accel'].mean()
        harsh_decel = cluster_data['harsh_decel'].mean()
        
        if harsh_accel > 5 or harsh_decel > 5:
            return "🔴 激进驾驶"
        elif speed < 20:
            return "🟡 城市拥堵"
        elif 20 <= speed < 40:
            return "🟠 城市道路"
        elif 40 <= speed < 70:
            return "🟢 郊区道路"
        elif speed >= 70:
            return "🔵 高速巡航"
        else:
            return "⚪ 平稳驾驶"
    
    def run_complete_visualization(self):
        """运行完整可视化"""
        print("\n" + "="*70)
        print(f"🎨 {self.model_name} 详细可视化分析")
        print("="*70)
        
        if not self.load_data():
            return
        
        save_dir = f'./results/detailed_visualization/{self.model_dir}'
        os.makedirs(save_dir, exist_ok=True)
        
        # 生成所有图表
        self.plot_overview(save_dir)
        self.plot_speed_analysis(save_dir)
        self.plot_energy_analysis(save_dir)
        self.plot_driving_behavior(save_dir)
        self.plot_time_series_detail(save_dir)
        self.plot_comprehensive_summary(save_dir)
        self.generate_statistics_report(save_dir)
        
        print("\n" + "="*70)
        print("✅ 详细可视化完成！")
        print("="*70)
        print(f"📁 所有结果保存在: {save_dir}/")
        print("\n生成的文件:")
        print("  01_overview.png - 总览（簇分布、特征热力图）")
        print("  02_speed_analysis.png - 速度详细分析")
        print("  03_energy_analysis.png - 能耗详细分析")
        print("  04_driving_behavior.png - 驾驶行为分析")
        print("  05_timeseries_cluster_X.png - 各簇时间序列")
        print("  06_comprehensive_summary.png - 所有指标综合对比")
        print("  cluster_detailed_statistics.csv - 详细统计表")
        print("  cluster_detailed_report.md - Markdown报告")


if __name__ == "__main__":
    # 使用最佳模型
    visualizer = DetailedClusterVisualizer(model_name='Transformer-AE')
    visualizer.run_complete_visualization()
    
    print("\n🎉 可视化完成！")