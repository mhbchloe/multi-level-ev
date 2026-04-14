"""
完整预处理流程：数据加载 → 事件切分 → 特征提取
"""
import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm
import pickle
import gc

class CompletePreprocessing:
    """完整预处理管道"""
    
    def __init__(self, data_dir='./', test_mode=True):
        self.data_dir = data_dir
        self.test_mode = test_mode
        self.sample_ratio = 0.1 if test_mode else None
        
    def load_data(self):
        """加载数据"""
        print("\n" + "="*70)
        print("📥 步骤1: 数据加载")
        print("="*70)
        
        csv_files = sorted(glob.glob(os.path.join(self.data_dir, '*_processed.csv')))
        print(f"找到 {len(csv_files)} 个文件")
        
        essential_cols = [
            'vehicle_id', 'datetime', 'soc', 'v', 'i', 'power',
            'is_charging', 'spd', 'acc',
            'lat', 'lon', 'distance_km',
            'is_moving', 'energy_consumption'
        ]
        
        dtype_dict = {
            'vehicle_id': 'category',
            'spd': 'float32', 'v': 'float32', 'i': 'float32',
            'soc': 'float32', 'power': 'float32',
            'is_moving': 'uint8', 'is_charging': 'uint8'
        }
        
        all_chunks = []
        for file in csv_files:
            print(f"📖 读取: {os.path.basename(file)}")
            sample = pd.read_csv(file, nrows=1)
            available_cols = [col for col in essential_cols if col in sample.columns]
            
            reader = pd.read_csv(
                file,
                chunksize=100000,
                usecols=available_cols,
                dtype={k: v for k, v in dtype_dict.items() if k in available_cols},
                low_memory=False
            )
            
            for chunk in reader:
                if self.sample_ratio:
                    chunk = chunk.sample(frac=self.sample_ratio, random_state=42)
                all_chunks.append(chunk)
            gc.collect()
        
        self.df = pd.concat(all_chunks, ignore_index=True)
        del all_chunks
        gc.collect()
        
        print(f"✅ 加载完成: {len(self.df):,} 行")
        return self.df
    
    def extract_events(self, soc_threshold=3.0, min_event_length=5):
        """切分事件"""
        print("\n" + "="*70)
        print("✂️  步骤2: 事件切分")
        print("="*70)
        
        self.df['datetime'] = pd.to_datetime(self.df['datetime'])
        events = []
        
        for vehicle_id in tqdm(self.df['vehicle_id'].unique(), desc="处理车辆"):
            vehicle_data = self.df[self.df['vehicle_id'] == vehicle_id].sort_values('datetime').reset_index(drop=True)
            
            if len(vehicle_data) < min_event_length:
                continue
            
            start_idx = 0
            start_soc = vehicle_data.iloc[0]['soc']
            
            for i in range(1, len(vehicle_data)):
                current_soc = vehicle_data.iloc[i]['soc']
                
                if start_soc - current_soc >= soc_threshold:
                    event_data = vehicle_data.iloc[start_idx:i]
                    
                    if len(event_data) >= min_event_length:
                        # 只保留驾驶事件
                        if event_data.get('is_charging', pd.Series([0])).mean() < 0.5:
                            events.append({
                                'event_id': len(events),
                                'vehicle_id': vehicle_id,
                                'data': event_data
                            })
                    
                    start_idx = i
                    start_soc = current_soc
        
        self.events = events
        print(f"✅ 提取事件: {len(events)} 个")
        
        # 保存
        os.makedirs('./results/events', exist_ok=True)
        with open('./results/events/events.pkl', 'wb') as f:
            pickle.dump(events, f)
        
        return events
    
    def extract_features(self):
        """特征提取"""
        print("\n" + "="*70)
        print("🔧 步骤3: 特征提取")
        print("="*70)
        
        energy_features = []
        driving_features = []
        event_ids = []
        vehicle_ids = []
        
        for event in tqdm(self.events, desc="提取特征"):
            ed = event['data']
            
            # 电量特征
            energy_feat = {
                'soc_drop_total': ed['soc'].iloc[0] - ed['soc'].iloc[-1] if 'soc' in ed.columns else 0,
                'soc_mean': ed['soc'].mean() if 'soc' in ed.columns else 0,
                'soc_std': ed['soc'].std() if 'soc' in ed.columns else 0,
                'voltage_mean': ed['v'].mean() if 'v' in ed.columns else 0,
                'voltage_std': ed['v'].std() if 'v' in ed.columns else 0,
                'current_mean': ed['i'].mean() if 'i' in ed.columns else 0,
                'current_max': ed['i'].max() if 'i' in ed.columns else 0,
                'power_mean': ed['power'].mean() if 'power' in ed.columns else 0,
                'power_max': ed['power'].max() if 'power' in ed.columns else 0,
                'power_std': ed['power'].std() if 'power' in ed.columns else 0,
            }
            
            if 'energy_consumption' in ed.columns:
                energy_feat['energy_consumption_total'] = ed['energy_consumption'].sum()
                energy_feat['efficiency_mean'] = ed['energy_consumption'].mean()
            else:
                energy_feat['energy_consumption_total'] = 0
                energy_feat['efficiency_mean'] = 0
            
            energy_feat['charging_ratio'] = ed.get('is_charging', pd.Series([0])).mean()
            energy_feat['regen_braking_ratio'] = 0  # 占位
            
            # 驾驶特征
            driving_feat = {
                'speed_mean': ed['spd'].mean() if 'spd' in ed.columns else 0,
                'speed_max': ed['spd'].max() if 'spd' in ed.columns else 0,
                'speed_std': ed['spd'].std() if 'spd' in ed.columns else 0,
                'speed_median': ed['spd'].median() if 'spd' in ed.columns else 0,
                'speed_cv': ed['spd'].std() / (ed['spd'].mean() + 1e-6) if 'spd' in ed.columns else 0,
            }
            
            if 'spd' in ed.columns:
                driving_feat['low_speed_ratio'] = ((ed['spd'] > 0) & (ed['spd'] <= 40)).sum() / len(ed)
                driving_feat['medium_speed_ratio'] = ((ed['spd'] > 40) & (ed['spd'] <= 80)).sum() / len(ed)
                driving_feat['high_speed_ratio'] = (ed['spd'] > 80).sum() / len(ed)
            else:
                driving_feat['low_speed_ratio'] = 0
                driving_feat['medium_speed_ratio'] = 0
                driving_feat['high_speed_ratio'] = 0
            
            if 'acc' in ed.columns:
                driving_feat['acc_mean'] = ed['acc'].mean()
                driving_feat['acc_std'] = ed['acc'].std()
                driving_feat['acc_max'] = ed['acc'].max()
                driving_feat['acc_min'] = ed['acc'].min()
                driving_feat['harsh_accel'] = (ed['acc'] > 2).sum()
                driving_feat['harsh_decel'] = (ed['acc'] < -2).sum()
            else:
                for key in ['acc_mean', 'acc_std', 'acc_max', 'acc_min', 'harsh_accel', 'harsh_decel']:
                    driving_feat[key] = 0
            
            driving_feat['moving_ratio'] = ed.get('is_moving', pd.Series([0])).mean()
            driving_feat['idle_ratio'] = 1 - driving_feat['moving_ratio']
            
            driving_feat['distance_total'] = ed['distance_km'].sum() if 'distance_km' in ed.columns else 0
            
            if 'datetime' in ed.columns:
                duration = (ed['datetime'].iloc[-1] - ed['datetime'].iloc[0]).total_seconds() / 60
                driving_feat['duration_minutes'] = duration
            else:
                driving_feat['duration_minutes'] = 0
            
            driving_feat['heading_change_mean'] = 0  # 占位
            driving_feat['sharp_turn_count'] = 0
            driving_feat['stop_count'] = 0
            
            energy_features.append(energy_feat)
            driving_features.append(driving_feat)
            event_ids.append(event['event_id'])
            vehicle_ids.append(event['vehicle_id'])
        
        # 转为DataFrame
        energy_df = pd.DataFrame(energy_features)
        driving_df = pd.DataFrame(driving_features)
        
        energy_df.insert(0, 'event_id', event_ids)
        energy_df.insert(1, 'vehicle_id', vehicle_ids)
        driving_df.insert(0, 'event_id', event_ids)
        driving_df.insert(1, 'vehicle_id', vehicle_ids)
        
        combined_df = pd.concat([energy_df, driving_df.drop(['event_id', 'vehicle_id'], axis=1)], axis=1)
        
        # 清理
        energy_df = energy_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        driving_df = driving_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        combined_df = combined_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # 保存
        os.makedirs('./results/features', exist_ok=True)
        energy_df.to_csv('./results/features/energy_features.csv', index=False)
        driving_df.to_csv('./results/features/driving_features.csv', index=False)
        combined_df.to_csv('./results/features/combined_features.csv', index=False)
        
        print(f"✅ 特征提取完成: {combined_df.shape}")
        
        return energy_df, driving_df, combined_df
    
    def run(self):
        """运行完整流程"""
        print("\n" + "="*70)
        print("🚀 完整预处理流程")
        print("="*70)
        print(f"模式: {'测试模式 (10%数据)' if self.test_mode else '生产模式 (全部数据)'}")
        
        self.load_data()
        self.extract_events()
        energy_df, driving_df, combined_df = self.extract_features()
        
        print("\n" + "="*70)
        print("✅ 预处理完成！")
        print("="*70)
        print(f"📁 结果保存在: ./results/")
        
        return energy_df, driving_df, combined_df


if __name__ == "__main__":
    import sys
    
    test_mode = True  # 改为False使用全部数据
    
    preprocessor = CompletePreprocessing(data_dir='./', test_mode=test_mode)
    energy_df, driving_df, combined_df = preprocessor.run()
    
    print("\n下一步: python 03_train_all.py")