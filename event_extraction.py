import pandas as pd
import numpy as np
from typing import List, Dict
from tqdm import tqdm

class EventExtractor:
    """基于SOC下降的事件切分器"""
    
    def __init__(self, df: pd.DataFrame, soc_drop_threshold: float = 3.0):
        """
        参数:
            df: 原始数据
            soc_drop_threshold: SOC下降阈值（%），默认3%
        """
        self.df = df.copy()
        self.df['datetime'] = pd.to_datetime(self.df['datetime'])
        self.soc_drop_threshold = soc_drop_threshold
        self.events = []
    
    def extract_events(self, min_event_length: int = 5, max_event_length: int = 1000) -> List[Dict]:
        """
        提取事件
        
        参数:
            min_event_length: 最小事件长度（数据点数）
            max_event_length: 最大事件长度（数据点数）
        
        返回:
            events: 事件列表
        """
        print("🔪 开始事件切分...")
        print(f"   SOC下降阈值: {self.soc_drop_threshold}%")
        print(f"   最小事件长度: {min_event_length} 个数据点")
        
        self.events = []
        vehicle_ids = self.df['vehicle_id'].unique()
        
        for vehicle_id in tqdm(vehicle_ids, desc="处理车辆"):
            vehicle_data = self.df[self.df['vehicle_id'] == vehicle_id].sort_values('datetime').reset_index(drop=True)
            
            if len(vehicle_data) < min_event_length:
                continue
            
            start_idx = 0
            start_soc = vehicle_data.iloc[0]['soc']
            start_charging = vehicle_data.iloc[0].get('is_charging', 0)
            
            for i in range(1, len(vehicle_data)):
                current_soc = vehicle_data.iloc[i]['soc']
                current_charging = vehicle_data.iloc[i].get('is_charging', 0)
                
                # 切分条件：
                # 1. SOC下降达到阈值
                # 2. 充电状态切换
                # 3. 达到最大事件长度
                soc_dropped = start_soc - current_soc >= self.soc_drop_threshold
                charging_changed = current_charging != start_charging
                max_length_reached = (i - start_idx) >= max_event_length
                
                if soc_dropped or charging_changed or max_length_reached:
                    event_data = vehicle_data.iloc[start_idx:i].copy()
                    
                    # 过滤太短的事件
                    if len(event_data) >= min_event_length:
                        event_info = self._create_event_info(vehicle_id, event_data, len(self.events))
                        self.events.append(event_info)
                    
                    # 重置起始点
                    start_idx = i
                    start_soc = current_soc
                    start_charging = current_charging
            
            # 处理最后一个事件
            if start_idx < len(vehicle_data) - 1:
                event_data = vehicle_data.iloc[start_idx:].copy()
                if len(event_data) >= min_event_length:
                    event_info = self._create_event_info(vehicle_id, event_data, len(self.events))
                    self.events.append(event_info)
        
        print(f"✅ 事件切分完成！共提取 {len(self.events)} 个事件")
        self._print_statistics()
        
        return self.events
    
    def _create_event_info(self, vehicle_id: str, event_data: pd.DataFrame, event_id: int) -> Dict:
        """创建事件信息字典"""
        return {
            'event_id': event_id,
            'vehicle_id': vehicle_id,
            'data': event_data,
            'start_time': event_data.iloc[0]['datetime'],
            'end_time': event_data.iloc[-1]['datetime'],
            'duration_minutes': (event_data.iloc[-1]['datetime'] - event_data.iloc[0]['datetime']).total_seconds() / 60,
            'soc_start': event_data.iloc[0]['soc'],
            'soc_end': event_data.iloc[-1]['soc'],
            'soc_drop': event_data.iloc[0]['soc'] - event_data.iloc[-1]['soc'],
            'distance_km': event_data['distance_km'].sum() if 'distance_km' in event_data.columns else 0,
            'num_points': len(event_data),
            'is_charging_event': event_data['is_charging'].mean() > 0.5 if 'is_charging' in event_data.columns else False
        }
    
    def _print_statistics(self):
        """打印事件统计信息"""
        if not self.events:
            return
        
        print("\n" + "=" * 50)
        print("📊 事件统计信息")
        print("=" * 50)
        
        durations = [e['duration_minutes'] for e in self.events]
        soc_drops = [e['soc_drop'] for e in self.events]
        distances = [e['distance_km'] for e in self.events]
        num_points = [e['num_points'] for e in self.events]
        
        print(f"  事件总数: {len(self.events)}")
        print(f"  平均持续时间: {np.mean(durations):.2f} 分钟")
        print(f"  平均SOC下降: {np.mean(soc_drops):.2f}%")
        print(f"  平均行驶距离: {np.mean(distances):.2f} km")
        print(f"  平均数据点数: {np.mean(num_points):.0f}")
        print(f"  充电事件数: {sum([e['is_charging_event'] for e in self.events])}")
    
    def save_events(self, output_dir: str = './events'):
        """保存事件到文件"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存事件元数据
        metadata = []
        for event in self.events:
            metadata.append({
                'event_id': event['event_id'],
                'vehicle_id': event['vehicle_id'],
                'start_time': event['start_time'],
                'end_time': event['end_time'],
                'duration_minutes': event['duration_minutes'],
                'soc_start': event['soc_start'],
                'soc_end': event['soc_end'],
                'soc_drop': event['soc_drop'],
                'distance_km': event['distance_km'],
                'num_points': event['num_points'],
                'is_charging_event': event['is_charging_event']
            })
        
        metadata_df = pd.DataFrame(metadata)
        metadata_df.to_csv(f'{output_dir}/event_metadata.csv', index=False)
        print(f"\n💾 事件元数据已保存至: {output_dir}/event_metadata.csv")
        
        # 保存每个事件的详细数据（可选，数据量大时慎用）
        # for event in self.events[:10]:  # 只保存前10个作为示例
        #     event['data'].to_csv(f"{output_dir}/event_{event['event_id']}.csv", index=False)
        
        return metadata_df
    
    def filter_events(self, 
                      min_duration: float = None,
                      max_duration: float = None,
                      min_distance: float = None,
                      exclude_charging: bool = True) -> List[Dict]:
        """
        过滤事件
        
        参数:
            min_duration: 最小持续时间（分钟）
            max_duration: 最大持续时间（分钟）
            min_distance: 最小行驶距离（km）
            exclude_charging: 是否排除充电事件
        """
        filtered_events = self.events.copy()
        
        if exclude_charging:
            filtered_events = [e for e in filtered_events if not e['is_charging_event']]
        
        if min_duration is not None:
            filtered_events = [e for e in filtered_events if e['duration_minutes'] >= min_duration]
        
        if max_duration is not None:
            filtered_events = [e for e in filtered_events if e['duration_minutes'] <= max_duration]
        
        if min_distance is not None:
            filtered_events = [e for e in filtered_events if e['distance_km'] >= min_distance]
        
        print(f"\n🔍 过滤后事件数: {len(filtered_events)} (原始: {len(self.events)})")
        
        return filtered_events


# 使用示例
if __name__ == "__main__":
    # 读取数据
    df = pd.read_csv('your_data.csv')
    
    # 创建事件提取器
    extractor = EventExtractor(df, soc_drop_threshold=3.0)
    
    # 提取事件
    events = extractor.extract_events(min_event_length=5, max_event_length=1000)
    
    # 保存事件
    metadata_df = extractor.save_events(output_dir='./events')
    
    # 过滤事件（排除充电事件，只保留驾驶事件）
    driving_events = extractor.filter_events(
        min_duration=5,      # 至少5分钟
        min_distance=1,      # 至少1公里
        exclude_charging=True
    )
    
    print(f"\n✅ 最终用于分析的驾驶事件数: {len(driving_events)}")