import numpy as np
import sys

# Path to y_train.npy (adjust as needed)
experiment_name = 'ecg_hybrid_net_exp5_maxpool'  # e.g., 'ecg_hybrid_net'
y_train_path = f'output/{experiment_name}/data/y_train.npy'

try:
    y_train = np.load(y_train_path, allow_pickle=True)
    sample_count = len(y_train)  # or y_train.shape[0]
    print(f"Number of samples in {y_train_path}: {sample_count}")
except FileNotFoundError:
    print(f"File not found: {y_train_path}")
    sys.exit(1)
