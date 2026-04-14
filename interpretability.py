import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import os

class ClusterInterpreter:
    """聚类结果可解释性分析（修复版）"""
    
    def __init__(self, features_df, cluster_labels, model_name='Unknown'):
        self.features_df = features_df.copy()
        self.cluster_labels = cluster_labels
        self.model_name = model_name
        
        # 添加聚类标签
        self.features_df['cluster'] = cluster_labels
    
    def cluster_profiles(self):
        """生成每个簇的特征剖面"""
        print("\n" + "="*60)
        print(f"📊 {self.model_name} - 簇特征剖面分析")
        print("="*60)
        
        # ⭐ 只选择数值列
        numeric_cols = self.features_df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in ['event_id', 'cluster']]
        
        if len(numeric_cols) == 0:
            print("⚠️  没有数值特征可分析")
            return pd.DataFrame()
        
        # 计算每个簇的统计特征
        profiles = self.features_df.groupby('cluster')[numeric_cols].mean()
        
        print("\n各簇平均特征值（前10个特征）:")
        print(profiles.iloc[:, :10].to_string())
        
        return profiles
    
    def name_clusters(self):
        """根据特征自动命名簇"""
        print("\n🏷️  自动簇命名...")
        
        # ⭐ 只使用数值列计算均值
        numeric_cols = self.features_df.select_dtypes(include=[np.number]).columns
        profiles = self.features_df.groupby('cluster')[numeric_cols].mean()
        
        cluster_names = {}
        
        for cluster_id in profiles.index:
            profile = profiles.loc[cluster_id]
            
            # 根据特征命名
            speed_mean = profile.get('speed_mean', 0)
            harsh_accel = profile.get('harsh_accel', 0)
            harsh_decel = profile.get('harsh_decel', 0)
            power_mean = profile.get('power_mean', 0)
            
            # 命名逻辑
            if harsh_accel > profiles['harsh_accel'].mean() * 1.5:
                name = "🔴 激进驾驶"
            elif speed_mean < 30:
                name = "🟡 城市拥堵"
            elif speed_mean > 80:
                name = "🟢 高速巡航"
            elif power_mean > profiles['power_mean'].mean() * 1.2:
                name = "🔵 高能耗驾驶"
            else:
                name = "⚪ 平稳驾驶"
            
            cluster_names[cluster_id] = name
            
            print(f"  簇 {cluster_id}: {name}")
            print(f"    - 平均速度: {speed_mean:.1f} km/h")
            print(f"    - 急加速: {harsh_accel:.1f} 次")
            print(f"    - 急减速: {harsh_decel:.1f} 次")
            print(f"    - 平均功率: {power_mean:.2f} kW")
        
        self.cluster_names = cluster_names
        return cluster_names
    
    # ... 其他方法保持不变 ...
    
    def run_full_interpretation(self):
        """运行完整可解释性分析"""
        print("\n" + "="*60)
        print(f"🔍 开始可解释性分析 - {self.model_name}")
        print("="*60)
        
        try:
            self.cluster_profiles()
        except Exception as e:
            print(f"⚠️  簇剖面分析失败: {e}")
        
        try:
            self.name_clusters()
        except Exception as e:
            print(f"⚠️  簇命名失败: {e}")
        
        try:
            self.feature_importance()
        except Exception as e:
            print(f"⚠️  特征重要性分析失败: {e}")
        
        try:
            self.visualize_cluster_distribution()
        except Exception as e:
            print(f"⚠️  簇分布可视化失败: {e}")
        
        try:
            self.visualize_feature_importance()
        except Exception as e:
            print(f"⚠️  特征重要性可视化失败: {e}")
        
        try:
            self.visualize_cluster_heatmap()
        except Exception as e:
            print(f"⚠️  热力图可视化失败: {e}")
        
        try:
            self.generate_interpretation_report()
        except Exception as e:
            print(f"⚠️  报告生成失败: {e}")
        
        print("\n✅ 可解释性分析完成！")