#!/usr/bin/env python3
"""Utilities for downloading MNIST and building PyTorch data loaders.

The official MNIST training split is loaded first and then divided into a
smaller training set and a validation set. The official MNIST test split is
kept separate and is not used in that split operation.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def build_mnist_transforms(augmentation_config=None):
    """Create separate transforms for training and evaluation datasets."""
    if augmentation_config is None:
        augmentation_config = {}

    use_augmentation = augmentation_config.get("enabled", False)
    rotation_degrees = augmentation_config.get("rotation_degrees", 0.0)
    translate = augmentation_config.get("translate", (0.0, 0.0))
    scale = augmentation_config.get("scale", (1.0, 1.0))

    train_transforms = []
    if use_augmentation:
        train_transforms.append(
            transforms.RandomAffine(
                degrees=rotation_degrees,
                translate=translate,
                scale=scale,
            )
        )

    train_transforms.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    evaluation_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    return transforms.Compose(train_transforms), evaluation_transforms


def get_mnist_loaders(
    batch_size=64,
    data_dir="./data",
    validation_ratio=0.1,
    seed=42,
    augmentation_config=None,
):
    """Create training, validation, and test data loaders for MNIST.

    The official MNIST ``train=True`` dataset is loaded as ``full_train_set``
    and then split into ``train_set`` and ``validation_set``. The official
    MNIST ``train=False`` dataset becomes ``test_set`` and stays untouched.

    Args:
        batch_size: Number of samples to include in each batch.
        data_dir: Directory where the dataset will be stored.
        validation_ratio: Fraction of the training set used for validation.
        seed: Random seed for the train/validation split.
        augmentation_config: Optional augmentation settings for train only.

    Returns:
        A tuple containing the training, validation, and test
        ``DataLoader`` instances.
    """
    root = Path(data_dir)
    train_transform, evaluation_transform = build_mnist_transforms(
        augmentation_config=augmentation_config
    )

    full_train_set = datasets.MNIST(
        root=root,
        train=True,
        download=True,
        transform=evaluation_transform,
    )
    augmented_train_set = datasets.MNIST(
        root=root,
        train=True,
        download=True,
        transform=train_transform,
    )
    test_set = datasets.MNIST(
        root=root,
        train=False,
        download=True,
        transform=evaluation_transform,
    )

    # Split only the official training data into train/validation.
    validation_size = int(len(full_train_set) * validation_ratio)
    train_size = len(full_train_set) - validation_size
    generator = torch.Generator().manual_seed(seed)
    train_set, validation_set = random_split(
        full_train_set,
        [train_size, validation_size],
        generator=generator,
    )
    augmented_train_set = torch.utils.data.Subset(
        augmented_train_set,
        train_set.indices,
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": 0,
        "pin_memory": torch.cuda.is_available(),
    }

    train_loader = DataLoader(augmented_train_set, shuffle=True, **loader_kwargs)
    validation_loader = DataLoader(validation_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)

    return train_loader, validation_loader, test_loader


if __name__ == "__main__":
    train_loader, validation_loader, test_loader = get_mnist_loaders()
    images, labels = next(iter(train_loader))
    print(f"Train batch image shape: {images.shape}")
    print(f"Train batch label shape: {labels.shape}")
    print(f"Number of train batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(validation_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
