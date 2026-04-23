#!/usr/bin/env python3
"""Inspect and summarize saved PyTorch checkpoint files."""

import argparse
from pathlib import Path

import torch


def inspect_checkpoint(checkpoint_path):
    """Load a checkpoint file and print its stored training metadata."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Epoch: {checkpoint.get('epoch')}")
    train_loss = checkpoint.get("train_loss")
    train_accuracy = checkpoint.get("train_accuracy")
    val_loss = checkpoint.get("val_loss")
    val_accuracy = checkpoint.get("val_accuracy")

    if train_loss is not None:
        print(f"Train loss: {train_loss:.4f}")
    if train_accuracy is not None:
        print(f"Train accuracy: {train_accuracy:.2%}")
    if val_loss is not None:
        print(f"Validation loss: {val_loss:.4f}")
    if val_accuracy is not None:
        print(f"Validation accuracy: {val_accuracy:.2%}")

    config = checkpoint.get("config")
    if config is not None:
        print("Config:")
        for key, value in config.items():
            print(f"  {key}: {value}")

    print("Stored keys:")
    for key in checkpoint.keys():
        print(f"  - {key}")


def main():
    """Parse CLI arguments and inspect the requested checkpoint file."""
    parser = argparse.ArgumentParser(
        description="Inspect a saved PyTorch checkpoint."
    )
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default="outputs/Part2",
        help="Path to the .pt checkpoint file.",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "best_model.pt"
    inspect_checkpoint(checkpoint_path)


if __name__ == "__main__":
    main()
