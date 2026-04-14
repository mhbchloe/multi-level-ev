import pandas as pd
import numpy as np
from typing import Dict, List
from tqdm import tqdm

class FeatureEngineer:
    """双通道特征工程"""
    
    def __init__(self, events: List[Dict]):
        self.events = events
        self.energy_features_list = []
        self.driving_features_list = []
    
    def extract_energy_features(self, event_data: pd.DataFrame) -> Dict:
        """
        电量特征通道
        """
        features = {}
        
        # 基础电量特征
        features['soc_drop_total'] = event_data['soc'].iloc[0] - event_data['soc'].iloc[-1]
        features['soc_mean'] = event_data['soc'].mean()
        features['soc_std'] = event_data['soc'].std()
        features['soc_min'] = event_data['soc'].min()
        features['soc_range'] = event_data['soc'].max() - event_data['soc'].min()
        
        # 能耗特征
        if 'energy_consumption' in event_data.columns:
            features['energy_consumption_total'] = event_data['energy_consumption'].sum()
            features['energy_consumption_mean'] = event_data['energy_consumption'].mean()
            features['energy_consumption_std'] = event_data['energy_consumption'].std()
        else:
            features['energy_consumption_total'] = 0
            features['energy_consumption_mean'] = 0
            features['energy_consumption_std'] = 0
        
        # 能效特征
        if 'efficiency_wh_per_km' in event_data.columns:
            eff_clean = event_data['efficiency_wh_per_km'].replace([np.inf, -np.inf], np.nan).dropna()
            features['efficiency_mean'] = eff_clean.mean() if len(eff_clean) > 0 else 0
            features['efficiency_std'] = eff_clean.std() if len(eff_clean) > 0 else 0
            features['efficiency_median'] = eff_clean.median() if len(eff_clean) > 0 else 0
        else:
            features['efficiency_mean'] = 0
            features['efficiency_std'] = 0
            features['efficiency_median'] = 0
        
        # 充放电特征
        if 'is_charging' in event_data.columns:
            features['charging_ratio'] = (event_data['is_charging'] == 1).sum() / len(event_data)
        else:
            features['charging_ratio'] = 0
        
        if 'is_discharging' in event_data.columns:
            features['discharging_ratio'] = (event_data['is_discharging'] == 1).sum() / len(event_data)
        else:
            features['discharging_ratio'] = 0
        
        if 'is_regenerative_braking' in event_data.columns:
            features['regen_braking_ratio'] = (event_data['is_regenerative_braking'] == 1).sum() / len(event_data)
            features['regen_braking_count'] = (event_data['is_regenerative_braking'] == 1).sum()
        else:
            features['regen_braking_ratio'] = 0
            features['regen_braking_count'] = 0
        
        # 电压电流特征
        features['voltage_mean'] = event_data['v'].mean()
        features['voltage_std'] = event_data['v'].std()
        features['voltage_max'] = event_data['v'].max()
        features['voltage_min'] = event_data['v'].min()
        
        features['current_mean'] = event_data['i'].mean()
        features['current_std'] = event_data['i'].std()
        features['current_max'] = event_data['i'].max()
        
        # 功率特征
        if 'power' in event_data.columns:
            features['power_mean'] = event_data['power'].mean()
            features['power_std'] = event_data['power'].std()
            features['power_max'] = event_data['power'].max()
            features['power_min'] = event_data['power'].min()
        else:
            features['power_mean'] = 0
            features['power_std'] = 0
            features['power_max'] = 0
            features['power_min'] = 0
        
        # SOC变化率
        if 'soc_rate' in event_data.columns:
            features['soc_rate_mean'] = event_data['soc_rate'].mean()
            features['soc_rate_std'] = event_data['soc_rate'].std()
        else:
            features['soc_rate_mean'] = 0
            features['soc_rate_std'] = 0
        
        return features
    
    def extract_driving_features(self, event_data: pd.DataFrame) -> Dict:
        """
        驾驶行为特征通道
        """
        features = {}
        
        # 速度特征
        features['speed_mean'] = event_data['spd'].mean()
        features['speed_std'] = event_data['spd'].std()
        features['speed_max'] = event_data['spd'].max()
        features['speed_min'] = event_data['spd'].min()
        features['speed_median'] = event_data['spd'].median()
        features['speed_cv'] = event_data['spd'].std() / (event_data['spd'].mean() + 1e-6)  # 变异系数
        
        # 速度分布特征（低速、中速、高速比例）
        features['low_speed_ratio'] = ((event_data['spd'] > 0) & (event_data['spd'] <= 40)).sum() / len(event_data)
        features['medium_speed_ratio'] = ((event_data['spd'] > 40) & (event_data['spd'] <= 80)).sum() / len(event_data)
        features['high_speed_ratio'] = (event_data['spd'] > 80).sum() / len(event_data)
        
        # 加速度特征
        if 'acc' in event_data.columns:
            features['acc_mean'] = event_data['acc'].mean()
            features['acc_std'] = event_data['acc'].std()
            features['acc_max'] = event_data['acc'].max()
            features['acc_min'] = event_data['acc'].min()
            features['acc_abs_mean'] = event_data['acc'].abs().mean()
            
            # 急加速/急减速
            features['harsh_accel_count'] = (event_data['acc'] > 2).sum()
            features['harsh_decel_count'] = (event_data['acc'] < -2).sum()
            features['harsh_accel_ratio'] = features['harsh_accel_count'] / len(event_data)
            features['harsh_decel_ratio'] = features['harsh_decel_count'] / len(event_data)
            
            # 加速度变化（急动度 jerk）
            if len(event_data) > 1:
                jerk = event_data['acc'].diff().abs()
                features['jerk_mean'] = jerk.mean()
                features['jerk_std'] = jerk.std()
            else:
                features['jerk_mean'] = 0
                features['jerk_std'] = 0
        else:
            for key in ['acc_mean', 'acc_std', 'acc_max', 'acc_min', 'acc_abs_mean',
                       'harsh_accel_count', 'harsh_decel_count', 'harsh_accel_ratio',
                       'harsh_decel_ratio', 'jerk_mean', 'jerk_std']:
                features[key] = 0
        
        # 运动状态特征
        if 'is_moving' in event_data.columns:
            features['moving_ratio'] = (event_data['is_moving'] == 1).sum() / len(event_data)
        else:
            features['moving_ratio'] = (event_data['spd'] > 0).sum() / len(event_data)
        
        if 'kinematic_state' in event_data.columns:
            features['idle_ratio'] = (event_data['kinematic_state'] == '静止').sum() / len(event_data)
            features['uniform_ratio'] = (event_data['kinematic_state'] == '匀速').sum() / len(event_data)
            features['accel_ratio'] = (event_data['kinematic_state'] == '加速').sum() / len(event_data)
            features['decel_ratio'] = (event_data['kinematic_state'] == '减速').sum() / len(event_data)
        else:
            features['idle_ratio'] = (event_data['spd'] == 0).sum() / len(event_data)
            features['uniform_ratio'] = 0
            features['accel_ratio'] = 0
            features['decel_ratio'] = 0
        
        # 速度变化特征
        if 'spd_change_rate' in event_data.columns:
            features['speed_change_rate_mean'] = event_data['spd_change_rate'].mean()
            features['speed_change_rate_std'] = event_data['spd_change_rate'].std()
        else:
            features['speed_change_rate_mean'] = 0
            features['speed_change_rate_std'] = 0
        
        # 驾驶模式变化
        if 'driving_mode' in event_data.columns:
            features['driving_mode_changes'] = (event_data['driving_mode'].diff() != 0).sum()
        else:
            features['driving_mode_changes'] = 0
        
        if 've_s_changed' in event_data.columns:
            features['ve_s_changes'] = event_data['ve_s_changed'].sum()
        else:
            features['ve_s_changes'] = 0
        
        # 行程特征
        if 'distance_km' in event_data.columns:
            features['distance_total'] = event_data['distance_km'].sum()
        else:
            features['distance_total'] = 0
        
        if 'time_diff' in event_data.columns:
            features['duration_seconds'] = event_data['time_diff'].sum()
            features['duration_minutes'] = features['duration_seconds'] / 60
        else:
            features['duration_seconds'] = 0
            features['duration_minutes'] = 0
        
        # 轨迹特征（转弯相关）
        if 'heading_change' in event_data.columns:
            heading_clean = event_data['heading_change'].replace([np.inf, -np.inf], np.nan).dropna()
            features['heading_change_mean'] = heading_clean.abs().mean() if len(heading_clean) > 0 else 0
            features['heading_change_std'] = heading_clean.abs().std() if len(heading_clean) > 0 else 0
            features['sharp_turn_count'] = (np.abs(heading_clean) > 45).sum() if len(heading_clean) > 0 else 0
            features['sharp_turn_ratio'] = features['sharp_turn_count'] / len(event_data)
        else:
            features['heading_change_mean'] = 0
            features['heading_change_std'] = 0
            features['sharp_turn_count'] = 0
            features['sharp_turn_ratio'] = 0
        
        # 停车次数（速度从>0降到0）
        if len(event_data) > 1:
            speed_zero = (event_data['spd'] == 0).astype(int)
            features['stop_count'] = (speed_zero.diff() == 1).sum()
        else:
            features['stop_count'] = 0
        
        return features
    
    def extract_all_features(self) -> tuple:
        """
        提取所有事件的特征
        
        返回:
            (energy_features_df, driving_features_df, combined_features_df)
        """
        print("🔧 开始特征提取...")
        
        self.energy_features_list = []
        self.driving_features_list = []
        event_ids = []
        vehicle_ids = []
        
        for event in tqdm(self.events, desc="提取特征"):
            event_data = event['data']
            
            # 提取特征
            energy_feat = self.extract_energy_features(event_data)
            driving_feat = self.extract_driving_features(event_data)
            
            self.energy_features_list.append(energy_feat)
            self.driving_features_list.append(driving_feat)
            event_ids.append(event['event_id'])
            vehicle_ids.append(event['vehicle_id'])
        
        # 转换为DataFrame
        energy_df = pd.DataFrame(self.energy_features_list)
        driving_df = pd.DataFrame(self.driving_features_list)
        
        # 添加ID列
        energy_df.insert(0, 'event_id', event_ids)
        energy_df.insert(1, 'vehicle_id', vehicle_ids)
        driving_df.insert(0, 'event_id', event_ids)
        driving_df.insert(1, 'vehicle_id', vehicle_ids)
        
        # 合并特征
        combined_df = pd.concat([energy_df, driving_df.drop(['event_id', 'vehicle_id'], axis=1)], axis=1)
        
        print(f"✅ 特征提取完成！")
        print(f"   电量特征维度: {energy_df.shape[1] - 2}")  # 减去ID列
        print(f"   驾驶特征维度: {driving_df.shape[1] - 2}")
        print(f"   总特征维度: {combined_df.shape[1] - 2}")
        
        return energy_df, driving_df, combined_df
    
    def handle_missing_and_inf(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值和无穷值"""
        df_clean = df.copy()
        
        # 替换无穷值为NaN
        df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # 填充NaN为0（或使用中位数）
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(0)
        
        return df_clean
    
    def save_features(self, energy_df: pd.DataFrame, driving_df: pd.DataFrame, 
                     combined_df: pd.DataFrame, output_dir: str = './features'):
        """保存特征到文件"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        energy_df.to_csv(f'{output_dir}/energy_features.csv', index=False)
        driving_df.to_csv(f'{output_dir}/driving_features.csv', index=False)
        combined_df.to_csv(f'{output_dir}/combined_features.csv', index=False)
        
        print(f"\n💾 特征已保存至: {output_dir}/")


# 使用示例
if __name__ == "__main__":
    # 假设已经有events（从上一步获得）
    import pickle
    
    # 读取事件（或从上一步直接传入）
    # with open('events.pkl', 'rb') as f:
    #     events = pickle.load(f)
    
    # 创建特征工程器
    engineer = FeatureEngineer(events)
    
    # 提取特征
    energy_df, driving_df, combined_df = engineer.extract_all_features()
    
    # 处理异常值
    energy_df_clean = engineer.handle_missing_and_inf(energy_df)
    driving_df_clean = engineer.handle_missing_and_inf(driving_df)
    combined_df_clean = engineer.handle_missing_and_inf(combined_df)
    
    # 保存特征
    engineer.save_features(energy_df_clean, driving_df_clean, combined_df_clean)
    
    print("\n✅ 特征工程完成！")