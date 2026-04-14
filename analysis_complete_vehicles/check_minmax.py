import h5py
import numpy as np

h5_path = './analysis_complete_vehicles/results/dual_channel_dataset.h5'

with h5py.File(h5_path, 'r') as f:
    print("driving_min:", f['driving_min'][:])
    print("driving_max:", f['driving_max'][:])
    print("energy_min:", f['energy_min'][:])
    print("energy_max:", f['energy_max'][:])

    # 看一下前几个样本的实际值范围
    offsets = f['offsets'][:]
    drv = f['driving_packed'][:1000]
    eng = f['energy_packed'][:1000]
    print(f"\ndriving_packed 前1000步:")
    print(f"  col0 (speed):   min={drv[:,0].min():.4f}, max={drv[:,0].max():.4f}")
    print(f"  col1 (acc):     min={drv[:,1].min():.4f}, max={drv[:,1].max():.4f}")
    print(f"  col2 (heading): min={drv[:,2].min():.4f}, max={drv[:,2].max():.4f}")
    print(f"\nenergy_packed 前1000步:")
    print(f"  col0 (soc):     min={eng[:,0].min():.4f}, max={eng[:,0].max():.4f}")
    print(f"  col1 (voltage): min={eng[:,1].min():.4f}, max={eng[:,1].max():.4f}")
    print(f"  col2 (current): min={eng[:,2].min():.4f}, max={eng[:,2].max():.4f}")
    print(f"  col3 (power):   min={eng[:,3].min():.4f}, max={eng[:,3].max():.4f}")