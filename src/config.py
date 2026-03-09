# src/config.py
"""Shared configuration, seed, and device setup."""

import os
import random
import numpy as np
import torch
import yaml

# -------------------- Load params from params.yaml --------------------
_params_path = os.path.join(os.path.dirname(__file__), '..', 'params.yaml')
with open(_params_path, 'r') as f:
    _params = yaml.safe_load(f)


# -------------------- Reproducibility --------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(_params.get('seed', 42))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------- Configuration --------------------
USE_PRETRAINED = _params.get('use_pretrained', True)

config = {
    'dataset': 'CIFAR10',
    'image_size': _params.get('image_size', 32),
    'channels': _params.get('channels', 3),
    # Training (only used if USE_PRETRAINED=False)
    'train_timesteps': _params.get('train_timesteps', 1000),
    'sampling_timesteps': _params.get('sampling_timesteps', 50),
    'batch_size': _params.get('batch_size', 32),
    'epochs': _params.get('epochs', 500),
    'lr': _params.get('lr', 1e-4),
    # Custom UNet architecture (ignored when pretrained)
    'base_channels': _params.get('base_channels', 64),
    'channel_mults': _params.get('channel_mults', [1, 2, 2, 4]),
    'num_res_blocks': _params.get('num_res_blocks', 2),
    # Quantization
    'quant_layers': _params.get('quant_layers', ['mid', 'down', 'up', 'conv']),
    'ptq_calib_steps': _params.get('ptq_calib_steps', 10),
    'qat_epochs': _params.get('qat_epochs', 20),
    'qat_lr': _params.get('qat_lr', 1e-5),
    # Evaluation
    'fid_num_samples': _params.get('fid_num_samples', 5000),
    'fid_batch_size': _params.get('fid_batch_size', 64),
}

# Derived paths
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

if __name__ == '__main__':
    print(f"Device: {device}")
    print(f"USE_PRETRAINED: {USE_PRETRAINED}")
    print(f"Config: {config}")
