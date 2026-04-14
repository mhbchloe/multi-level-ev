import pandas as pd
import numpy as np
import glob
import os
from typing import List, Optional
from tqdm import tqdm
import gc

class LargeDataLoader:
    """大文件内存优化加载器"""
    
    def __init__(self, data_dir: str = './'):
        self.data_dir = data_dir
        self.csv_files = sorted(glob.glob(os.path.join(data_dir, '*.csv')))
        print(f"📂 找到 {len(self.csv_files)} 个CSV文件")
        for f in self.csv_files:
            size_mb = os.path.getsize(f) / (1024 * 1024)
            print(f"   - {os.path.basename(f)}: {size_mb:.2f} MB")
    
    def optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        优化数据类型，减少内存占用
        可以减少50-80%的内存使用
        """
        print("🔧 优化数据类型...")
        
        # 整数类型优化
        int_cols = df.select_dtypes(include=['int64']).columns
        for col in int_cols:
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_min >= 0:  # 无符号整数
                if col_max < 255:
                    df[col] = df[col].astype('uint8')
                elif col_max < 65535:
                    df[col] = df[col].astype('uint16')
                elif col_max < 4294967295:
                    df[col] = df[col].astype('uint32')
            else:  # 有符号整数
                if col_min > -128 and col_max < 127:
                    df[col] = df[col].astype('int8')
                elif col_min > -32768 and col_max < 32767:
                    df[col] = df[col].astype('int16')
                elif col_min > -2147483648 and col_max < 2147483647:
                    df[col] = df[col].astype('int32')
        
        # 浮点类型优化
        float_cols = df.select_dtypes(include=['float64']).columns
        for col in float_cols:
            df[col] = df[col].astype('float32')
        
        # 对象类型（字符串）优化为category
        obj_cols = df.select_dtypes(include=['object']).columns
        for col in obj_cols:
            if col not in ['datetime']:  # 排除datetime列
                num_unique = df[col].nunique()
                num_total = len(df[col])
                if num_unique / num_total < 0.5:  # 如果唯一值<50%，转为category
                    df[col] = df[col].astype('category')
        
        return df
    
    def load_single_file_chunked(self, 
                                  file_path: str, 
                                  chunksize: int = 100000,
                                  usecols: Optional[List[str]] = None,
                                  sample_ratio: Optional[float] = None) -> pd.DataFrame:
        """
        分块加载单个大文件
        
        参数:
            file_path: 文件路径
            chunksize: 每次读取的行数
            usecols: 只读取指定的列（减少内存）
            sample_ratio: 采样比例（0-1），用于快速测试
        """
        print(f"\n📖 加载文件: {os.path.basename(file_path)}")
        
        chunks = []
        total_rows = 0
        
        # 第一遍：确定数据类型
        sample_df = pd.read_csv(file_path, nrows=1000, usecols=usecols)
        
        # 定义需要的列和类型
        if usecols is None:
            usecols = sample_df.columns.tolist()
        
        # 优化dtype
        dtype_dict = {}
        for col in sample_df.columns:
            if col in ['vehicle_id', 'kinematic_state', 'driving_mode', 've_s', 'ch_s']:
                dtype_dict[col] = 'category'
            elif col in ['spd', 'v', 'i', 'soc', 'hv', 'lv', 'lat', 'lon', 
                        'acc', 'distance_km', 'power', 'efficiency_wh_per_km']:
                dtype_dict[col] = 'float32'
            elif col in ['is_moving', 'is_charging', 'is_discharging', 
                        'is_weekend', 've_s_changed', 'ch_s_changed']:
                dtype_dict[col] = 'uint8'
        
        # 分块读取
        reader = pd.read_csv(file_path, 
                            chunksize=chunksize, 
                            usecols=usecols,
                            dtype=dtype_dict,
                            low_memory=False)
        
        for chunk in tqdm(reader, desc="读取分块"):
            # 采样（如果需要）
            if sample_ratio is not None and sample_ratio < 1.0:
                chunk = chunk.sample(frac=sample_ratio, random_state=42)
            
            chunks.append(chunk)
            total_rows += len(chunk)
            
            # 定期清理内存
            if len(chunks) % 10 == 0:
                gc.collect()
        
        print(f"   ✅ 读取完成: {total_rows:,} 行")
        
        # 合并所有chunks
        df = pd.concat(chunks, ignore_index=True)
        del chunks
        gc.collect()
        
        return df
    
    def load_all_files(self, 
                       chunksize: int = 100000,
                       usecols: Optional[List[str]] = None,
                       sample_ratio: Optional[float] = None,
                       file_pattern: str = '*.csv') -> pd.DataFrame:
        """
        加载所有CSV文件并合并
        
        参数:
            chunksize: 每次读取的行数
            usecols: 只读取指定的列
            sample_ratio: 采样比例（建议先用0.1测试）
            file_pattern: 文件匹配模式
        """
        print("=" * 60)
        print("🚀 开始加载多个大文件...")
        print("=" * 60)
        
        # 重新查找文件
        csv_files = sorted(glob.glob(os.path.join(self.data_dir, file_pattern)))
        
        if not csv_files:
            raise FileNotFoundError(f"未找到匹配的文件: {file_pattern}")
        
        all_dfs = []
        
        for file_path in csv_files:
            df = self.load_single_file_chunked(
                file_path, 
                chunksize=chunksize,
                usecols=usecols,
                sample_ratio=sample_ratio
            )
            all_dfs.append(df)
            
            # 显示内存使用
            mem_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)
            print(f"   内存占用: {mem_usage:.2f} MB")
        
        # 合并所有文件
        print("\n🔗 合并所有数据...")
        combined_df = pd.concat(all_dfs, ignore_index=True)
        del all_dfs
        gc.collect()
        
        print(f"✅ 总数据量: {len(combined_df):,} 行 x {len(combined_df.columns)} 列")
        total_mem = combined_df.memory_usage(deep=True).sum() / (1024 * 1024)
        print(f"✅ 总内存占用: {total_mem:.2f} MB")
        
        return combined_df
    
    def get_essential_columns(self) -> List[str]:
        """
        返回必需的列（用于减少内存）
        """
        essential_cols = [
            # 必需列
            'vehicle_id', 'datetime', 'time',
            
            # 电量相关
            'soc', 'v', 'i', 'power', 
            'is_charging', 'is_discharging',
            'energy_consumption', 'efficiency_wh_per_km',
            
            # 驾驶行为
            'spd', 'acc', 'acc_smooth',
            'kinematic_state', 'is_moving',
            'driving_mode',
            
            # 位置
            'lat', 'lon', 'distance_km', 'heading', 'heading_change',
            
            # 时间特征（可选）
            'hour', 'day_of_week', 'is_weekend',
            
            # 制动
            'is_regenerative_braking',
            
            # 状态变化
            've_s', 'ch_s', 've_s_changed', 'ch_s_changed',
            
            # 其他
            'time_diff', 'soc_change', 'soc_rate'
        ]
        
        return essential_cols
    
    def load_optimized(self, 
                       test_mode: bool = True,
                       sample_ratio: float = 0.1) -> pd.DataFrame:
        """
        一键优化加载（推荐使用）
        
        参数:
            test_mode: 是否测试模式（只加载一小部分数据）
            sample_ratio: 采样比例
        """
        if test_mode:
            print("⚠️  测试模式：只加载10%数据")
            sample_ratio = 0.1
        else:
            print("🔥 生产模式：加载全部数据")
            sample_ratio = None
        
        # 只读取必要的列
        essential_cols = self.get_essential_columns()
        
        df = self.load_all_files(
            chunksize=100000,
            usecols=essential_cols,
            sample_ratio=sample_ratio,
            file_pattern='*_processed.csv'  # 只读取processed文件
        )
        
        # 进一步优化数据类型
        df = self.optimize_dtypes(df)
        
        return df


# 使用示例
if __name__ == "__main__":
    import time
    
    # 创建加载器
    loader = LargeDataLoader(data_dir='./')
    
    # ========== 方式1: 测试模式（快速验证） ==========
    print("\n" + "="*60)
    print("方式1: 测试模式（10%数据）")
    print("="*60)
    start_time = time.time()
    
    df_test = loader.load_optimized(test_mode=True, sample_ratio=0.1)
    
    print(f"\n⏱️  加载耗时: {time.time() - start_time:.2f} 秒")
    print(df_test.info(memory_usage='deep'))
    print(df_test.head())
    
    # ========== 方式2: 生产模式（全部数据） ==========
    # 注意：全部加载可能需要32GB+ 内存
    # print("\n" + "="*60)
    # print("方式2: 生产模式（全部数据）")
    # print("="*60)
    # start_time = time.time()
    # 
    # df_full = loader.load_optimized(test_mode=False)
    # 
    # print(f"\n⏱️  加载耗时: {time.time() - start_time:.2f} 秒")
    
    # ========== 方式3: 只加载指定文件 ==========
    print("\n" + "="*60)
    print("方式3: 只加载一个文件")
    print("="*60)
    
    single_df = loader.load_single_file_chunked(
        file_path='./20250701_processed.csv',
        chunksize=100000,
        usecols=loader.get_essential_columns(),
        sample_ratio=0.1  # 10%采样
    )
    
    print(f"数据形状: {single_df.shape}")