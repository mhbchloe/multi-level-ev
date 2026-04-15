"""
诊断脚本：查看processed文件实际有哪些列
"""

import pandas as pd
import glob

processed_files = sorted(glob.glob('./data_20250*_processed.csv'))
if not processed_files:
    processed_files = sorted(glob.glob('./data_processed_one_month/data_*_processed.csv'))
if not processed_files:
    processed_files = sorted(glob.glob('./**/data_*_processed.csv', recursive=True))

if not processed_files:
    print("❌ No processed files found!")
    exit()

print("="*70)
print("🔍 Checking Processed File Columns")
print("="*70)

csv_file = processed_files[0]
print(f"\nReading: {csv_file}\n")

df = pd.read_csv(csv_file, nrows=100, on_bad_lines='skip')

print(f"Total columns: {len(df.columns)}\n")
print("Available columns:")
for i, col in enumerate(df.columns, 1):
    print(f"   {i:2d}. {col}")

print(f"\n{'─'*70}")
print("Sample data (first 3 rows):")
print(df.head(3))

print(f"\n{'─'*70}")
print("Data types:")
print(df.dtypes)