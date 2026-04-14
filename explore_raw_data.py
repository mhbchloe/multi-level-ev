"""
探索原始数据：查看7个CSV文件的结构和内容
"""

import pandas as pd
import os
import glob

print("="*70)
print("📂 Exploring Raw Data Files")
print("="*70)

# 找到所有的原始CSV文件
raw_data_dir = "./"
csv_files = sorted(glob.glob(os.path.join(raw_data_dir, "2025070*.csv")))

print(f"\n🔍 Found {len(csv_files)} raw data files:")
for f in csv_files:
    print(f"   - {os.path.basename(f)}")

# 逐个加载并展示
for csv_file in csv_files:
    filename = os.path.basename(csv_file)
    print(f"\n{'='*70}")
    print(f"📋 File: {filename}")
    print("="*70)
    
    df = pd.read_csv(csv_file)
    
    print(f"\n�� Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\n📋 Columns: {df.columns.tolist()}")
    
    print(f"\n🔍 Data Types:")
    print(df.dtypes)
    
    print(f"\n📈 First 5 rows:")
    print(df.head())
    
    print(f"\n📊 Statistical Summary:")
    print(df.describe())
    
    print(f"\n🔍 Missing Values:")
    print(df.isnull().sum())
    
    print(f"\n🔍 Unique Values (for key columns):")
    for col in df.columns:
        if df[col].dtype == 'object':  # 字符串列
            nunique = df[col].nunique()
            if nunique <= 20:
                print(f"   {col}: {nunique} unique values")
                print(f"      {df[col].unique()[:10]}")
        else:
            print(f"   {col}: {df[col].nunique()} unique values")

print(f"\n{'='*70}")
print("✅ Exploration Complete!")
print("="*70)