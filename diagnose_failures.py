"""
诊断模型训练失败的原因
"""
import os
import sys

print("="*70)
print("🔍 诊断模型训练失败原因")
print("="*70)

# ==================== 1. 检查Python和依赖 ====================
print("\n📦 1. 检查依赖包...")

required_packages = {
    'torch': 'PyTorch',
    'sklearn': 'scikit-learn',
    'pandas': 'Pandas',
    'numpy': 'NumPy'
}

missing_packages = []
for package, name in required_packages.items():
    try:
        __import__(package)
        print(f"  ✅ {name}: 已安装")
    except ImportError:
        print(f"  ❌ {name}: 未安装")
        missing_packages.append(package)

if missing_packages:
    print(f"\n⚠️  缺少依赖包: {missing_packages}")
    print("请运行: pip install " + " ".join(missing_packages))

# ==================== 2. 检查PyTorch ====================
print("\n🔥 2. 检查PyTorch...")

try:
    import torch
    print(f"  ✅ PyTorch版本: {torch.__version__}")
    print(f"  CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"  使用CPU模式")
    
    # 测试PyTorch基本功能
    x = torch.randn(2, 3)
    print(f"  ✅ PyTorch基本功能正常")
    
except Exception as e:
    print(f"  ❌ PyTorch错误: {e}")

# ==================== 3. 检查数据文件 ====================
print("\n📂 3. 检查数据文件...")

required_files = {
    'energy_features': './results/features/energy_features.csv',
    'driving_features': './results/features/driving_features.csv',
    'combined_features': './results/features/combined_features.csv',
    'events': './results/events/events.pkl'
}

missing_files = []
for name, path in required_files.items():
    if os.path.exists(path):
        import os
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  ✅ {name}: {size_mb:.2f} MB")
    else:
        print(f"  ❌ {name}: 未找到")
        missing_files.append(name)

if missing_files:
    print(f"\n⚠️  缺少数据文件: {missing_files}")

# ==================== 4. 检查模型文件 ====================
print("\n📄 4. 检查模型文件...")

model_files = [
    '4_model_baseline.py',
    '5_model_autoencoder.py',
    '6_model_vae.py',
    '7_model_lstm_ae.py',
    '8_model_contrastive.py',
    '9_model_dec.py'
]

for file in model_files:
    if os.path.exists(file):
        print(f"  ✅ {file}")
    else:
        print(f"  ❌ {file}: 未找到")

# ==================== 5. 测试简单模型训练 ====================
print("\n🧪 5. 测试简单模型训练...")

try:
    import torch
    import torch.nn as nn
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    
    # 读取特征
    features_df = pd.read_csv('./results/features/combined_features.csv')
    X = features_df.drop(['event_id', 'vehicle_id'], axis=1, errors='ignore')
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"  ✅ 数据加载成功: {X_scaled.shape}")
    
    # 创建简单神经网络
    class SimpleNet(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.fc = nn.Linear(input_dim, 32)
        
        def forward(self, x):
            return self.fc(x)
    
    model = SimpleNet(X_scaled.shape[1])
    X_tensor = torch.FloatTensor(X_scaled[:10])
    output = model(X_tensor)
    
    print(f"  ✅ 神经网络测试成功")
    print(f"  输入形状: {X_tensor.shape}")
    print(f"  输出形状: {output.shape}")
    
except Exception as e:
    print(f"  ❌ 测试失败: {e}")
    import traceback
    print("\n完整错误信息:")
    traceback.print_exc()

# ==================== 6. 检查train_all_models.py ====================
print("\n📜 6. 检查训练脚本...")

if os.path.exists('train_all_models.py'):
    with open('train_all_models.py', 'r', encoding='utf-8') as f:
        content = f.read()
        
    # 检查导入语句
    if 'from model_baseline' in content or 'from 4_model_baseline' in content:
        print("  ⚠️  导入方式可能有问题")
        print("  Python模块名不能以数字开头")
    else:
        print("  ✅ 导入语句正常")
else:
    print("  ❌ train_all_models.py 未找到")

# ==================== 诊断结果 ====================
print("\n" + "="*70)
print("📊 诊断结果总结")
print("="*70)

if missing_packages:
    print("\n🔴 主要问题: 缺少依赖包")
    print(f"   解决方案: pip install {' '.join(missing_packages)}")
elif missing_files:
    print("\n🔴 主要问题: 缺少数据文件")
    print("   解决方案: 先运行 python run_full_pipeline.py")
else:
    print("\n🟡 可能的问题:")
    print("   1. 模块导入错误（文件名以数字开头）")
    print("   2. 训练过程中出现运行时错误")
    print("   3. 内存不足")
    
    print("\n💡 建议的解决方案:")
    print("   运行下面的修复脚本")

print("\n" + "="*70)