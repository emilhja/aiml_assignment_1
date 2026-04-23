#!/usr/bin/env python3
"""Helpers for selecting and describing PyTorch compute devices."""
import torch


def get_device(prefer_cuda=True):
    """Return CUDA device 0 when available and requested.

    Falls back to CPU otherwise.
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def describe_device(device):
    """Return a short human-readable description of the device.

    Includes GPU name and CUDA version when running on CUDA.
    """
    if device.type == "cuda":
        gpu_index = device.index if device.index is not None else 0
        return (
            f"Using CUDA GPU {gpu_index}: "
            f"{torch.cuda.get_device_name(gpu_index)} "
            f"(CUDA {torch.version.cuda})"
        )
    return "Using CPU because CUDA is not available."


def list_cuda_gpus():
    """List available CUDA GPUs with index and device name.

    Returns an empty list if CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        return []
    return [
        {
            "index": index,
            "name": torch.cuda.get_device_name(index),
        }
        for index in range(torch.cuda.device_count())
    ]


if __name__ == "__main__":
    device = get_device(prefer_cuda=True)
    print(describe_device(device))

    gpus = list_cuda_gpus()
    if gpus:
        print(f"Number of GPUs available: {len(gpus)}")
        for gpu in gpus:
            print(f"GPU index {gpu['index']}: {gpu['name']}")
    else:
        print("CUDA is not available.")
