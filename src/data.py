# src/data.py
"""Download and prepare the CIFAR-10 dataset."""

import os
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from src.config import config, DATA_DIR


def get_transform():
    """Return the training transform for CIFAR-10."""
    return transforms.Compose([
        transforms.Resize(config['image_size']),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def get_dataset():
    """Download CIFAR-10 and return the dataset object."""
    return torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=get_transform()
    )


def get_dataloader(dataset=None):
    """Return a DataLoader for the CIFAR-10 training set."""
    if dataset is None:
        dataset = get_dataset()
    return DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )


if __name__ == '__main__':
    # Entry point for the DVC 'prepare_data' stage
    dataset = get_dataset()
    print(f"Dataset: CIFAR10, {len(dataset)} images")

    # Write sentinel flag so DVC can track completion
    flag_path = os.path.join(DATA_DIR, 'data_ready.flag')
    with open(flag_path, 'w') as f:
        f.write(f"CIFAR10 downloaded. {len(dataset)} images.\n")
    print(f"Flag written to {flag_path}")
