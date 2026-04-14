"""
检查时间戳格式
"""
import pandas as pd

df = pd.read_csv('./results/event_table.csv')

print("时间戳样本：")
print(df[['start_time', 'end_time']].head(10))
print(f"\nstart_time类型: {df['start_time'].dtype}")
print(f"start_time示例: {df['start_time'].iloc[0]}")