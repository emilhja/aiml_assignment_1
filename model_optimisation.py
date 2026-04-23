#!/usr/bin/env python3
"""Utilities for saving checkpoints and training history artifacts."""

import json
from pathlib import Path

import torch


class CheckpointManager:
    """Manage best-model, periodic checkpoint, and history persistence."""

    def __init__(self, output_dir, checkpoint_interval=5):
        """Prepare output paths and checkpoint-saving policy."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_interval = checkpoint_interval
        self.best_loss = float("inf")
        self.best_path = self.output_dir / "best_model.pt"
        self.history_path = self.output_dir / "training_history.json"
        self.config_path = self.output_dir / "config.json"
        self.summary_path = self.output_dir / "summary.json"

    def _build_payload(
        self,
        epoch,
        model,
        optimizer,
        config,
        train_loss,
        train_accuracy,
        val_loss,
        val_accuracy,
    ):
        """Build the checkpoint payload saved to disk."""
        return {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }

    def save_best(
        self,
        epoch,
        model,
        optimizer,
        config,
        train_loss,
        train_accuracy,
        val_loss,
        val_accuracy,
    ):
        """Save a checkpoint when the validation loss improves."""
        if val_loss >= self.best_loss:
            return None

        self.best_loss = val_loss
        payload = self._build_payload(
            epoch,
            model,
            optimizer,
            config,
            train_loss,
            train_accuracy,
            val_loss,
            val_accuracy,
        )
        torch.save(payload, self.best_path)
        return self.best_path

    def save_periodic(
        self,
        epoch,
        model,
        optimizer,
        config,
        train_loss,
        train_accuracy,
        val_loss,
        val_accuracy,
    ):
        """Save a checkpoint at the configured epoch interval."""
        if epoch % self.checkpoint_interval != 0:
            return None

        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch:02d}.pt"
        payload = self._build_payload(
            epoch,
            model,
            optimizer,
            config,
            train_loss,
            train_accuracy,
            val_loss,
            val_accuracy,
        )
        torch.save(payload, checkpoint_path)
        return checkpoint_path

    def save_history(self, history):
        """Write training history to JSON and return its path."""
        serializable_history = {
            key: [float(value) for value in values]
            for key, values in history.items()
        }
        self.history_path.write_text(
            json.dumps(serializable_history, indent=2),
            encoding="utf-8",
        )
        return self.history_path

    def save_config(self, config):
        """Write experiment configuration to JSON and return its path."""
        self.config_path.write_text(
            json.dumps(config, indent=2),
            encoding="utf-8",
        )
        return self.config_path

    def save_summary(self, summary):
        """Write final experiment summary to JSON and return its path."""
        self.summary_path.write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        return self.summary_path
