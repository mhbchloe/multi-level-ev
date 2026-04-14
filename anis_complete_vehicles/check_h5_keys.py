import h5py

h5_path = './analysis_complete_vehicles/results/dual_channel_dataset.h5'

with h5py.File(h5_path, 'r') as f:
    print("Top-level keys:", list(f.keys()))
    print()
    for key in f.keys():
        item = f[key]
        if hasattr(item, 'shape'):
            print(f"  {key}: shape={item.shape}, dtype={item.dtype}")
        else:
            print(f"  {key}: (group)")
        # 查看 attrs
        if len(item.attrs) > 0:
            for attr_name, attr_val in item.attrs.items():
                print(f"    attr[{attr_name}] = {attr_val}")

    # 顶层 attrs
    if len(f.attrs) > 0:
        print("\nFile attrs:")
        for attr_name, attr_val in f.attrs.items():
            print(f"  {attr_name} = {attr_val}")