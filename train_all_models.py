"""
一键训练所有模型并生成完整报告（修复版 - 自动创建目录）
"""
import os
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime

# 导入所有模型
sys.path.append(os.path.dirname(__file__))

def main():
    print("="*70)
    print("🚀 电动车驾驶行为聚类 - 完整模型训练流程")
    print("="*70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    # ==================== 创建所有需要的目录 ====================
    print("\n📁 创建结果目录...")
    
    result_dirs = [
        './results/baseline',
        './results/autoencoder',
        './results/vae',
        './results/lstm_ae',
        './results/contrastive',
        './results/dec',
        './results/comparison',
        './results/interpretability'
    ]
    
    for dir_path in result_dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"  ✅ {dir_path}")
    
    # ==================== 检查数据 ====================
    print("\n📂 步骤0: 检查数据文件...")
    
    required_files = [
        './results/features/energy_features.csv',
        './results/features/driving_features.csv',
        './results/features/combined_features.csv'
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            print(f"❌ 错误: 找不到文件 {file}")
            print("请先运行数据预处理流程: python run_full_pipeline.py")
            return
    
    print("✅ 所有数据文件就绪")
    
    # 读取特征
    energy_df = pd.read_csv('./results/features/energy_features.csv')
    driving_df = pd.read_csv('./results/features/driving_features.csv')
    combined_df = pd.read_csv('./results/features/combined_features.csv')
    
    print(f"   能量特征: {energy_df.shape}")
    print(f"   驾驶特征: {driving_df.shape}")
    print(f"   合并特征: {combined_df.shape}")
    
    # 统计训练结果
    models_trained = []
    models_failed = []
    
    # ==================== 1. 基线模型 ====================
    print("\n" + "="*70)
    print("📌 步骤1: 训练基线模型 (K-Means, DBSCAN, GMM, Hierarchical)")
    print("="*70)
    
    try:
        from model_baseline import BaselineModels
        
        baseline = BaselineModels(n_clusters=5)
        baseline.train_all(combined_df)
        baseline.visualize_clusters(combined_df)
        baseline.save_models()
        
        models_trained.append('baseline')
        print("✅ 基线模型训练完成")
    except Exception as e:
        models_failed.append(('baseline', str(e)))
        print(f"❌ 基线模型训练失败: {e}")
        import traceback
        traceback.print_exc()
    
    # ==================== 2. 自编码器 ====================
    print("\n" + "="*70)
    print("📌 步骤2: 训练双通道自编码器")
    print("="*70)
    
    try:
        from model_autoencoder import AutoencoderTrainer
        
        energy_dim = len(energy_df.columns) - 2
        driving_dim = len(driving_df.columns) - 2
        
        ae_trainer = AutoencoderTrainer(energy_dim, driving_dim, latent_dim=16, n_clusters=5)
        ae_trainer.train(energy_df, driving_df, epochs=50, batch_size=32)
        
        latent_features = ae_trainer.extract_features(energy_df, driving_df)
        labels, metrics = ae_trainer.cluster(latent_features)
        
        # 确保目录存在（双保险）
        os.makedirs('./results/autoencoder', exist_ok=True)
        
        # 保存结果
        results_df = energy_df[['event_id', 'vehicle_id']].copy()
        results_df['cluster'] = labels
        results_df.to_csv('./results/autoencoder/clustered_results.csv', index=False)
        
        ae_trainer.visualize_training()
        ae_trainer.save_model()
        
        models_trained.append('autoencoder')
        print("✅ 自编码器训练完成")
    except Exception as e:
        models_failed.append(('autoencoder', str(e)))
        print(f"❌ 自编码器训练失败: {e}")
        import traceback
        traceback.print_exc()
    
    # ==================== 3. VAE ====================
    print("\n" + "="*70)
    print("📌 步骤3: 训练VAE")
    print("="*70)
    
    try:
        from model_vae import VAETrainer
        
        input_dim = len(combined_df.columns) - 2
        vae_trainer = VAETrainer(input_dim, latent_dim=16, n_clusters=5)
        vae_trainer.train(combined_df, epochs=50, batch_size=32)
        
        latent_features = vae_trainer.extract_features(combined_df)
        labels, metrics = vae_trainer.cluster(latent_features)
        
        # 确保目录存在
        os.makedirs('./results/vae', exist_ok=True)
        
        results_df = combined_df[['event_id', 'vehicle_id']].copy()
        results_df['cluster'] = labels
        results_df.to_csv('./results/vae/clustered_results.csv', index=False)
        
        vae_trainer.visualize_training()
        vae_trainer.save_model()
        
        models_trained.append('vae')
        print("✅ VAE训练完成")
    except Exception as e:
        models_failed.append(('vae', str(e)))
        print(f"❌ VAE训练失败: {e}")
        import traceback
        traceback.print_exc()
    
    # ==================== 4. LSTM-AE ====================
    print("\n" + "="*70)
    print("📌 步骤4: 训练LSTM-AE (推荐)")
    print("="*70)
    
    try:
        import pickle
        from model_lstm_ae import LSTMAETrainer
        
        # 检查事件文件是否存在
        if not os.path.exists('./results/events/events.pkl'):
            raise FileNotFoundError("事件文件不存在，请先运行 run_full_pipeline.py")
        
        # 读取事件数据
        with open('./results/events/events.pkl', 'rb') as f:
            events = pickle.load(f)
        
        lstm_trainer = LSTMAETrainer(input_dim=6, hidden_dim=64, latent_dim=16, n_clusters=5)
        sequences, valid_events = lstm_trainer.prepare_sequences(events, max_seq_len=100)
        
        lstm_trainer.train(sequences, epochs=30, batch_size=32)
        
        latent_features = lstm_trainer.extract_features(sequences)
        labels, metrics = lstm_trainer.cluster(latent_features)
        
        # 确保目录存在
        os.makedirs('./results/lstm_ae', exist_ok=True)
        
        results_df = pd.DataFrame({
            'event_id': [e['event_id'] for e in valid_events],
            'vehicle_id': [e['vehicle_id'] for e in valid_events],
            'cluster': labels
        })
        results_df.to_csv('./results/lstm_ae/clustered_results.csv', index=False)
        
        lstm_trainer.visualize_training()
        lstm_trainer.save_model()
        
        models_trained.append('lstm_ae')
        print("✅ LSTM-AE训练完成")
    except Exception as e:
        models_failed.append(('lstm_ae', str(e)))
        print(f"❌ LSTM-AE训练失败: {e}")
        import traceback
        traceback.print_exc()
    
    # ==================== 5. 对比学习 ====================
    print("\n" + "="*70)
    print("📌 步骤5: 训练对比学习模型")
    print("="*70)
    
    try:
        from model_contrastive import ContrastiveTrainer
        
        input_dim = len(combined_df.columns) - 2
        cont_trainer = ContrastiveTrainer(input_dim, hidden_dim=128, output_dim=64, n_clusters=5)
        cont_trainer.train(combined_df, epochs=50, batch_size=32)
        
        features = cont_trainer.extract_features(combined_df)
        labels, metrics = cont_trainer.cluster(features)
        
        # 确保目录存在
        os.makedirs('./results/contrastive', exist_ok=True)
        
        results_df = combined_df[['event_id', 'vehicle_id']].copy()
        results_df['cluster'] = labels
        results_df.to_csv('./results/contrastive/clustered_results.csv', index=False)
        
        cont_trainer.visualize_training()
        cont_trainer.save_model()
        
        models_trained.append('contrastive')
        print("✅ 对比学习模型训练完成")
    except Exception as e:
        models_failed.append(('contrastive', str(e)))
        print(f"❌ 对比学习模型训练失败: {e}")
        import traceback
        traceback.print_exc()
    
    # ==================== 6. DEC ====================
    print("\n" + "="*70)
    print("📌 步骤6: 训练DEC")
    print("="*70)
    
    try:
        from model_dec import DECTrainer
        
        input_dim = len(combined_df.columns) - 2
        dec_trainer = DECTrainer(input_dim, n_clusters=5)
        dec_trainer.train(combined_df, pretrain_epochs=30, train_epochs=50, batch_size=32)
        
        labels, features, metrics = dec_trainer.predict(combined_df)
        
        # 确保目录存在
        os.makedirs('./results/dec', exist_ok=True)
        
        results_df = combined_df[['event_id', 'vehicle_id']].copy()
        results_df['cluster'] = labels
        results_df.to_csv('./results/dec/clustered_results.csv', index=False)
        
        dec_trainer.visualize_training()
        dec_trainer.save_model()
        
        models_trained.append('dec')
        print("✅ DEC训练完成")
    except Exception as e:
        models_failed.append(('dec', str(e)))
        print(f"❌ DEC训练失败: {e}")
        import traceback
        traceback.print_exc()
    # 在 main() 函数中，DEC训练后添加：

    # ==================== 新增：Transformer-AE ====================
    print("\n" + "="*70)
    print("📌 步骤7: 训练Transformer-AE (Attention机制)")
    print("="*70)
    
    try:
        import pickle
        from model_transformer_ae import TransformerAETrainer
        
        with open('./results/events/events.pkl', 'rb') as f:
            events = pickle.load(f)
        
        os.makedirs('./results/transformer_ae', exist_ok=True)
        
        transformer_trainer = TransformerAETrainer(
            input_dim=6,
            d_model=64,
            nhead=4,
            num_layers=2,
            latent_dim=16,
            n_clusters=5
        )
        
        sequences, valid_events = transformer_trainer.prepare_sequences(events, max_seq_len=100)
        transformer_trainer.train(sequences, epochs=30, batch_size=32)
        
        latent_features = transformer_trainer.extract_features(sequences)
        labels, metrics = transformer_trainer.cluster(latent_features)
        
        results_df = pd.DataFrame({
            'event_id': [e['event_id'] for e in valid_events],
            'vehicle_id': [e['vehicle_id'] for e in valid_events],
            'cluster': labels
        })
        results_df.to_csv('./results/transformer_ae/clustered_results.csv', index=False)
        
        transformer_trainer.visualize_training()
        transformer_trainer.save_model()
        
        models_trained.append('transformer_ae')
        print("✅ Transformer-AE训练完成")
    except Exception as e:
        models_failed.append(('transformer_ae', str(e)))
        print(f"❌ Transformer-AE训练失败: {e}")
        import traceback
        traceback.print_exc()
    
    # ==================== 新增：TCN-AE ====================
    print("\n" + "="*70)
    print("📌 步骤8: 训练TCN-AE (时序卷积)")
    print("="*70)
    
    try:
        from model_tcn_ae import TCNAETrainer
        
        os.makedirs('./results/tcn_ae', exist_ok=True)
        
        tcn_trainer = TCNAETrainer(
            input_dim=6,
            num_channels=[64, 64, 32],
            latent_dim=16,
            n_clusters=5
        )
        
        sequences, valid_events = tcn_trainer.prepare_sequences(events, max_seq_len=100)
        tcn_trainer.train(sequences, epochs=30, batch_size=32)
        
        latent_features = tcn_trainer.extract_features(sequences)
        labels, metrics = tcn_trainer.cluster(latent_features)
        
        results_df = pd.DataFrame({
            'event_id': [e['event_id'] for e in valid_events],
            'vehicle_id': [e['vehicle_id'] for e in valid_events],
            'cluster': labels
        })
        results_df.to_csv('./results/tcn_ae/clustered_results.csv', index=False)
        
        tcn_trainer.visualize_training()
        tcn_trainer.save_model()
        
        models_trained.append('tcn_ae')
        print("✅ TCN-AE训练完成")
    except Exception as e:
        models_failed.append(('tcn_ae', str(e)))
        print(f"❌ TCN-AE训练失败: {e}")
        import traceback
        traceback.print_exc()
    # ==================== 7. 模型对比 ====================
    print("\n" + "="*70)
    print("📌 步骤7: 模型对比分析")
    print("="*70)
    
    try:
        from model_comparison import ModelComparator
        
        # 确保目录存在
        os.makedirs('./results/comparison', exist_ok=True)
        
        comparator = ModelComparator(results_dir='./results')
        report_df = comparator.run_full_comparison(combined_df)
        
        print("✅ 模型对比完成")
    except Exception as e:
        print(f"❌ 模型对比失败: {e}")
        import traceback
        traceback.print_exc()
    
    # ==================== 8. 可解释性分析 ====================
    print("\n" + "="*70)
    print("📌 步骤8: 可解释性分析 (最佳模型)")
    print("="*70)
    
    try:
        from interpretability import ClusterInterpreter
        
        # 确保目录存在
        os.makedirs('./results/interpretability', exist_ok=True)
        
        # 找到可用的最佳模型
        best_model_file = None
        for model_name in ['lstm_ae', 'dec', 'contrastive', 'vae', 'autoencoder']:
            result_file = f'./results/{model_name}/clustered_results.csv'
            if os.path.exists(result_file):
                best_model_file = result_file
                best_model_name = model_name.upper()
                break
        
        if best_model_file is None:
            # 使用baseline的K-Means结果
            import pickle
            with open('./results/baseline/baseline_models.pkl', 'rb') as f:
                baseline_data = pickle.load(f)
                labels = baseline_data['labels']['kmeans']
                best_model_name = 'K-Means'
        else:
            best_results = pd.read_csv(best_model_file)
            labels = best_results['cluster'].values
        
        interpreter = ClusterInterpreter(combined_df, labels, model_name=best_model_name)
        interpreter.run_full_interpretation()
        
        print("✅ 可解释性分析完成")
    except Exception as e:
        print(f"❌ 可解释性分析失败: {e}")
        import traceback
        traceback.print_exc()
    
    # ==================== 总结 ====================
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("🎉 训练流程完成！")
    print("="*70)
    print(f"⏱️  总耗时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
    print(f"✅ 成功: {len(models_trained)}/{len(models_trained) + len(models_failed)} 个模型")
    print(f"❌ 失败: {len(models_failed)}/{len(models_trained) + len(models_failed)} 个模型")
    
    if models_trained:
        print(f"\n✅ 成功的模型: {', '.join(models_trained)}")
    
    if models_failed:
        print(f"\n❌ 失败的模型:")
        for model_name, error in models_failed:
            print(f"   - {model_name}: {error[:80]}...")
    
    print(f"\n📁 结果目录: ./results/")
    
    if os.path.exists('./results/comparison/comparison_report.md'):
        print(f"📊 查看对比报告: ./results/comparison/comparison_report.md")
    
    if os.path.exists('./results/interpretability'):
        print(f"🔍 查看可解释性报告: ./results/interpretability/")
    
    print("="*70)


if __name__ == "__main__":
    main()