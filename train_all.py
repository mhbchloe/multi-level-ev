"""
一键训练所有模型（修复版）
"""
import os
import time
import pickle

print("="*70)
print("🚀 一键训练所有模型")
print("="*70)

# 检查预处理是否完成
if not os.path.exists('./results/events/events.pkl'):
    print("❌ 错误: 请先运行 01_preprocessing.py")
    exit(1)

if not os.path.exists('./results/features/combined_features.csv'):
    print("❌ 错误: 特征文件不存在")
    exit(1)

# 读取数据
import pandas as pd

with open('./results/events/events.pkl', 'rb') as f:
    events = pickle.load(f)

energy_df = pd.read_csv('./results/features/energy_features.csv')
driving_df = pd.read_csv('./results/features/driving_features.csv')
combined_df = pd.read_csv('./results/features/combined_features.csv')

print(f"✅ 数据就绪: {len(events)} 事件, {combined_df.shape} 特征")

models_trained = []
models_failed = []

# ==================== 1. LSTM-AE ====================
print("\n" + "="*70)
print("📌 训练 LSTM-AE")
print("="*70)

try:
    from model_lstm_ae import LSTMAETrainer
    
    trainer = LSTMAETrainer(
        input_dim=6,
        hidden_dim=64,
        latent_dim=16,
        num_layers=2,
        n_clusters=5
    )
    
    sequences, valid_events = trainer.prepare_sequences(events, max_seq_len=100)
    trainer.train(sequences, epochs=30, batch_size=32)
    
    latent_features = trainer.extract_features(sequences)
    labels, metrics = trainer.cluster(latent_features)
    
    results_df = pd.DataFrame({
        'event_id': [e['event_id'] for e in valid_events],
        'vehicle_id': [e['vehicle_id'] for e in valid_events],
        'cluster': labels
    })
    
    os.makedirs('./results/lstm_ae', exist_ok=True)
    results_df.to_csv('./results/lstm_ae/clustered_results.csv', index=False)
    
    trainer.visualize_training()
    trainer.save_model()
    
    models_trained.append('LSTM-AE')
    print(f"✅ LSTM-AE 完成")
    
except Exception as e:
    models_failed.append(('LSTM-AE', str(e)))
    print(f"❌ LSTM-AE 失败: {e}")
    import traceback
    traceback.print_exc()

# ==================== 2. GRU-AE ====================
print("\n" + "="*70)
print("📌 训练 GRU-AE")
print("="*70)

try:
    from model_gru_ae import GRUAETrainer
    
    trainer = GRUAETrainer(
        input_dim=6,
        hidden_dim=64,
        latent_dim=16,
        num_layers=2,
        n_clusters=5
    )
    
    sequences, valid_events = trainer.prepare_sequences(events, max_seq_len=100)
    trainer.train(sequences, epochs=30, batch_size=32)
    
    latent_features = trainer.extract_features(sequences)
    labels, metrics = trainer.cluster(latent_features)
    
    results_df = pd.DataFrame({
        'event_id': [e['event_id'] for e in valid_events],
        'vehicle_id': [e['vehicle_id'] for e in valid_events],
        'cluster': labels
    })
    
    os.makedirs('./results/gru_ae', exist_ok=True)
    results_df.to_csv('./results/gru_ae/clustered_results.csv', index=False)
    
    trainer.visualize_training()
    trainer.save_model()
    
    models_trained.append('GRU-AE')
    print(f"✅ GRU-AE 完成")
    
except Exception as e:
    models_failed.append(('GRU-AE', str(e)))
    print(f"❌ GRU-AE 失败: {e}")
    import traceback
    traceback.print_exc()

# ==================== 3. Attention-LSTM ====================
print("\n" + "="*70)
print("📌 训练 Attention-LSTM")
print("="*70)

try:
    from model_attention_lstm import AttentionLSTMTrainer
    
    trainer = AttentionLSTMTrainer(
        input_dim=6,
        hidden_dim=64,
        latent_dim=16,
        num_layers=2,
        n_clusters=5
    )
    
    sequences, valid_events = trainer.prepare_sequences(events, max_seq_len=100)
    trainer.train(sequences, epochs=30, batch_size=32)
    
    latent_features = trainer.extract_features(sequences)
    labels, metrics = trainer.cluster(latent_features)
    
    results_df = pd.DataFrame({
        'event_id': [e['event_id'] for e in valid_events],
        'vehicle_id': [e['vehicle_id'] for e in valid_events],
        'cluster': labels
    })
    
    os.makedirs('./results/attention_lstm', exist_ok=True)
    results_df.to_csv('./results/attention_lstm/clustered_results.csv', index=False)
    
    trainer.visualize_training()
    trainer.visualize_attention(sample_idx=0)
    trainer.save_model()
    
    models_trained.append('Attention-LSTM')
    print(f"✅ Attention-LSTM 完成")
    
except Exception as e:
    models_failed.append(('Attention-LSTM', str(e)))
    print(f"❌ Attention-LSTM 失败: {e}")
    import traceback
    traceback.print_exc()

# ==================== 4. Transformer-AE ====================
print("\n" + "="*70)
print("📌 训练 Transformer-AE")
print("="*70)

try:
    from model_transformer_ae import TransformerAETrainer
    
    # ⭐ 注意：Transformer使用不同的参数
    trainer = TransformerAETrainer(
        input_dim=6,
        d_model=64,      # 使用d_model而不是hidden_dim
        nhead=4,         # 注意力头数
        num_layers=2,
        latent_dim=16,
        n_clusters=5
    )
    
    sequences, valid_events = trainer.prepare_sequences(events, max_seq_len=100)
    trainer.train(sequences, epochs=30, batch_size=32)
    
    latent_features = trainer.extract_features(sequences)
    labels, metrics = trainer.cluster(latent_features)
    
    results_df = pd.DataFrame({
        'event_id': [e['event_id'] for e in valid_events],
        'vehicle_id': [e['vehicle_id'] for e in valid_events],
        'cluster': labels
    })
    
    os.makedirs('./results/transformer_ae', exist_ok=True)
    results_df.to_csv('./results/transformer_ae/clustered_results.csv', index=False)
    
    trainer.visualize_training()
    trainer.save_model()
    
    models_trained.append('Transformer-AE')
    print(f"✅ Transformer-AE 完成")
    
except Exception as e:
    models_failed.append(('Transformer-AE', str(e)))
    print(f"❌ Transformer-AE 失败: {e}")
    import traceback
    traceback.print_exc()

# ==================== 5. TCN-AE ====================
print("\n" + "="*70)
print("📌 训练 TCN-AE")
print("="*70)

try:
    from model_tcn_ae import TCNAETrainer
    
    # ⭐ 注意：TCN使用不同的参数
    trainer = TCNAETrainer(
        input_dim=6,
        num_channels=[64, 64, 32],  # 使用num_channels而不是hidden_dim
        latent_dim=16,
        n_clusters=5
    )
    
    sequences, valid_events = trainer.prepare_sequences(events, max_seq_len=100)
    trainer.train(sequences, epochs=30, batch_size=32)
    
    latent_features = trainer.extract_features(sequences)
    labels, metrics = trainer.cluster(latent_features)
    
    results_df = pd.DataFrame({
        'event_id': [e['event_id'] for e in valid_events],
        'vehicle_id': [e['vehicle_id'] for e in valid_events],
        'cluster': labels
    })
    
    os.makedirs('./results/tcn_ae', exist_ok=True)
    results_df.to_csv('./results/tcn_ae/clustered_results.csv', index=False)
    
    trainer.visualize_training()
    trainer.save_model()
    
    models_trained.append('TCN-AE')
    print(f"✅ TCN-AE 完成")
    
except Exception as e:
    models_failed.append(('TCN-AE', str(e)))
    print(f"❌ TCN-AE 失败: {e}")
    import traceback
    traceback.print_exc()

# ==================== 6. 其他深度学习模型 ====================

# Autoencoder
print("\n" + "="*70)
print("📌 训练 Autoencoder")
print("="*70)

try:
    from model_autoencoder import AutoencoderTrainer
    
    energy_dim = len(energy_df.columns) - 2
    driving_dim = len(driving_df.columns) - 2
    
    ae_trainer = AutoencoderTrainer(energy_dim, driving_dim, latent_dim=16, n_clusters=5)
    ae_trainer.train(energy_df, driving_df, epochs=30, batch_size=32)
    
    latent_features = ae_trainer.extract_features(energy_df, driving_df)
    labels, metrics = ae_trainer.cluster(latent_features)
    
    results_df = energy_df[['event_id', 'vehicle_id']].copy()
    results_df['cluster'] = labels
    os.makedirs('./results/autoencoder', exist_ok=True)
    results_df.to_csv('./results/autoencoder/clustered_results.csv', index=False)
    
    ae_trainer.visualize_training()
    ae_trainer.save_model()
    
    models_trained.append('Autoencoder')
    print(f"✅ Autoencoder 完成")
    
except Exception as e:
    models_failed.append(('Autoencoder', str(e)))
    print(f"❌ Autoencoder 失败: {e}")

# VAE
print("\n" + "="*70)
print("📌 训练 VAE")
print("="*70)

try:
    from model_vae import VAETrainer
    
    input_dim = len(combined_df.columns) - 2
    vae_trainer = VAETrainer(input_dim, latent_dim=16, n_clusters=5)
    vae_trainer.train(combined_df, epochs=30, batch_size=32)
    
    latent_features = vae_trainer.extract_features(combined_df)
    labels, metrics = vae_trainer.cluster(latent_features)
    
    results_df = combined_df[['event_id', 'vehicle_id']].copy()
    results_df['cluster'] = labels
    os.makedirs('./results/vae', exist_ok=True)
    results_df.to_csv('./results/vae/clustered_results.csv', index=False)
    
    vae_trainer.visualize_training()
    vae_trainer.save_model()
    
    models_trained.append('VAE')
    print(f"✅ VAE 完成")
    
except Exception as e:
    models_failed.append(('VAE', str(e)))
    print(f"❌ VAE 失败: {e}")

# Contrastive
print("\n" + "="*70)
print("📌 训练 Contrastive Learning")
print("="*70)

try:
    from model_contrastive import ContrastiveTrainer
    
    input_dim = len(combined_df.columns) - 2
    cont_trainer = ContrastiveTrainer(input_dim, hidden_dim=128, output_dim=64, n_clusters=5)
    cont_trainer.train(combined_df, epochs=30, batch_size=32)
    
    features = cont_trainer.extract_features(combined_df)
    labels, metrics = cont_trainer.cluster(features)
    
    results_df = combined_df[['event_id', 'vehicle_id']].copy()
    results_df['cluster'] = labels
    os.makedirs('./results/contrastive', exist_ok=True)
    results_df.to_csv('./results/contrastive/clustered_results.csv', index=False)
    
    cont_trainer.visualize_training()
    cont_trainer.save_model()
    
    models_trained.append('Contrastive')
    print(f"✅ Contrastive Learning 完成")
    
except Exception as e:
    models_failed.append(('Contrastive', str(e)))
    print(f"❌ Contrastive Learning 失败: {e}")

# DEC
print("\n" + "="*70)
print("📌 训练 DEC")
print("="*70)

try:
    from model_dec import DECTrainer
    
    input_dim = len(combined_df.columns) - 2
    dec_trainer = DECTrainer(input_dim, n_clusters=5)
    dec_trainer.train(combined_df, pretrain_epochs=20, train_epochs=30, batch_size=32)
    
    labels, features, metrics = dec_trainer.predict(combined_df)
    
    results_df = combined_df[['event_id', 'vehicle_id']].copy()
    results_df['cluster'] = labels
    os.makedirs('./results/dec', exist_ok=True)
    results_df.to_csv('./results/dec/clustered_results.csv', index=False)
    
    dec_trainer.visualize_training()
    dec_trainer.save_model()
    
    models_trained.append('DEC')
    print(f"✅ DEC 完成")
    
except Exception as e:
    models_failed.append(('DEC', str(e)))
    print(f"❌ DEC 失败: {e}")

# ==================== 总结 ====================
print("\n" + "="*70)
print("📊 训练总结")
print("="*70)
print(f"✅ 成功: {len(models_trained)}/{len(models_trained) + len(models_failed)}")
print(f"❌ 失败: {len(models_failed)}/{len(models_trained) + len(models_failed)}")

if models_trained:
    print(f"\n✅ 成功的模型: {', '.join(models_trained)}")

if models_failed:
    print(f"\n❌ 失败的模型:")
    for model_name, error in models_failed:
        print(f"   - {model_name}: {error[:80]}...")

print("\n下一步: python 04_compare_models.py")
print("="*70)