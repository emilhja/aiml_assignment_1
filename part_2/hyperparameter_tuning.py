#!/usr/bin/env python3
"""Run a small hyperparameter tuning sweep for MNIST CNN experiments."""

import argparse
import json
from datetime import datetime
from pathlib import Path

try:
    from .main import OUTPUT_ROOT, run_experiment
    from .notebook_templates import create_tuning_notebook
    from .notebook_utils import execute_notebook
except ImportError:
    if __package__:
        raise
    from main import OUTPUT_ROOT, run_experiment
    from notebook_templates import create_tuning_notebook
    from notebook_utils import execute_notebook


def build_parser():
    """Create CLI arguments for tuning runs."""
    parser = argparse.ArgumentParser(
        description="Train a list of hyperparameter settings and compare the results."
    )
    parser.add_argument("--config-path", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=7)
    parser.add_argument("--checkpoint-interval", type=int, default=5)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    return parser


def build_default_search_space():
    """Return a compact list of hand-picked hyperparameter combinations."""
    return [
        {
            "run_name": "tune_01_baseline",
            "description": "Baseline 2-conv CNN with default learning rate and no extra regularization.",
            "model_name": "cnn_medium",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "l1_lambda": 0.0,
            "input_noise_std": 0.0,
            "adam_betas": (0.9, 0.999),
            "adam_eps": 1e-8,
            "model_overrides": None,
            "augmentation_config": {"enabled": False, "rotation_degrees": 0.0, "translate": (0.0, 0.0), "scale": (1.0, 1.0)},
        },
        {
            "run_name": "tune_02_dropout",
            "description": "Tests whether dropout alone improves generalization relative to the baseline.",
            "model_name": "cnn_dropout",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "l1_lambda": 0.0,
            "input_noise_std": 0.0,
            "adam_betas": (0.9, 0.999),
            "adam_eps": 1e-8,
            "model_overrides": None,
            "augmentation_config": {"enabled": False, "rotation_degrees": 0.0, "translate": (0.0, 0.0), "scale": (1.0, 1.0)},
        },
        {
            "run_name": "tune_03_weight_decay",
            "description": "Tests L2-style regularization through Adam weight decay on the baseline architecture.",
            "model_name": "cnn_medium",
            "learning_rate": 0.001,
            "weight_decay": 1e-4,
            "l1_lambda": 0.0,
            "input_noise_std": 0.0,
            "adam_betas": (0.9, 0.999),
            "adam_eps": 1e-8,
            "model_overrides": None,
            "augmentation_config": {"enabled": False, "rotation_degrees": 0.0, "translate": (0.0, 0.0), "scale": (1.0, 1.0)},
        },
        {
            "run_name": "tune_04_batchnorm",
            "description": "Tests whether batch normalization helps optimization and validation performance.",
            "model_name": "cnn_batchnorm",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "l1_lambda": 0.0,
            "input_noise_std": 0.0,
            "adam_betas": (0.9, 0.999),
            "adam_eps": 1e-8,
            "model_overrides": None,
            "augmentation_config": {"enabled": False, "rotation_degrees": 0.0, "translate": (0.0, 0.0), "scale": (1.0, 1.0)},
        },
        {
            "run_name": "tune_05_noise",
            "description": "Adds Gaussian input noise to test whether noise injection stabilizes training.",
            "model_name": "cnn_batchnorm",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "l1_lambda": 0.0,
            "input_noise_std": 0.05,
            "adam_betas": (0.9, 0.999),
            "adam_eps": 1e-8,
            "model_overrides": None,
            "augmentation_config": {"enabled": False, "rotation_degrees": 0.0, "translate": (0.0, 0.0), "scale": (1.0, 1.0)},
        },
        {
            "run_name": "tune_06_deep_balanced",
            "description": "Tests a deeper 3-conv CNN while keeping capacity in a moderate range.",
            "model_name": "cnn_deep_balanced",
            "learning_rate": 0.001,
            "weight_decay": 1e-4,
            "l1_lambda": 0.0,
            "input_noise_std": 0.0,
            "adam_betas": (0.9, 0.999),
            "adam_eps": 1e-8,
            "model_overrides": None,
            "augmentation_config": {"enabled": False, "rotation_degrees": 0.0, "translate": (0.0, 0.0), "scale": (1.0, 1.0)},
        },
        {
            "run_name": "tune_07_augmented",
            "description": "Tests whether data augmentation improves a strong batch-normalized 3-conv model.",
            "model_name": "cnn_batchnorm",
            "learning_rate": 0.001,
            "weight_decay": 1e-4,
            "l1_lambda": 0.0,
            "input_noise_std": 0.0,
            "adam_betas": (0.9, 0.999),
            "adam_eps": 1e-8,
            "model_overrides": None,
            "augmentation_config": {"enabled": True, "rotation_degrees": 10.0, "translate": (0.1, 0.1), "scale": (0.95, 1.05)},
        },
        {
            "run_name": "tune_08_low_lr",
            "description": "Tests whether a lower learning rate helps the regularized deep model converge better.",
            "model_name": "cnn_regularized",
            "learning_rate": 0.0005,
            "weight_decay": 1e-4,
            "l1_lambda": 0.0,
            "input_noise_std": 0.05,
            "adam_betas": (0.9, 0.999),
            "adam_eps": 1e-8,
            "model_overrides": None,
            "augmentation_config": {"enabled": False, "rotation_degrees": 0.0, "translate": (0.0, 0.0), "scale": (1.0, 1.0)},
        },
        {
            "run_name": "tune_09_custom_kernel",
            "description": "Tests a custom 5x5 kernel setup with narrower channels and batch normalization.",
            "model_name": "cnn_medium",
            "learning_rate": 0.001,
            "weight_decay": 1e-4,
            "l1_lambda": 0.0,
            "input_noise_std": 0.0,
            "adam_betas": (0.9, 0.999),
            "adam_eps": 1e-8,
            "model_overrides": {
                "model_name": "cnn_medium_custom_kernel",
                "conv_channels": [24, 48],
                "kernel_size": 5,
                "classifier_hidden_size": 128,
                "activation": "ReLU",
                "dropout": 0.1,
                "batch_norm": True,
            },
            "augmentation_config": {"enabled": False, "rotation_degrees": 0.0, "translate": (0.0, 0.0), "scale": (1.0, 1.0)},
        },
        {
            "run_name": "tune_10_deep_augmented",
            "description": "Combines deeper architecture, stronger regularization, and augmentation in one run.",
            "model_name": "cnn_regularized",
            "learning_rate": 0.0007,
            "weight_decay": 2e-4,
            "l1_lambda": 0.0,
            "input_noise_std": 0.03,
            "adam_betas": (0.9, 0.999),
            "adam_eps": 1e-8,
            "model_overrides": None,
            "augmentation_config": {"enabled": True, "rotation_degrees": 12.0, "translate": (0.1, 0.1), "scale": (0.9, 1.1)},
        },
        {
            "run_name": "tune_11_hidden_512",
            "description": "Tests whether a much larger classifier hidden layer improves accuracy enough to justify the extra parameters.",
            "model_name": "cnn_medium",
            "learning_rate": 0.001,
            "weight_decay": 1e-4,
            "l1_lambda": 0.0,
            "input_noise_std": 0.0,
            "adam_betas": (0.9, 0.999),
            "adam_eps": 1e-8,
            "model_overrides": {
                "model_name": "cnn_medium_hidden_512",
                "conv_channels": [32, 64],
                "classifier_hidden_size": 512,
                "activation": "LeakyReLU",
                "dropout": 0.0,
                "batch_norm": False,
            },
            "augmentation_config": {"enabled": False, "rotation_degrees": 0.0, "translate": (0.0, 0.0), "scale": (1.0, 1.0)},
        },
        {
            "run_name": "tune_12_hidden_64",
            "description": "Tests whether a smaller classifier hidden layer can reduce model size with limited accuracy loss.",
            "model_name": "cnn_medium",
            "learning_rate": 0.001,
            "weight_decay": 1e-4,
            "l1_lambda": 0.0,
            "input_noise_std": 0.0,
            "adam_betas": (0.9, 0.999),
            "adam_eps": 1e-8,
            "model_overrides": {
                "model_name": "cnn_medium_hidden_64",
                "conv_channels": [32, 64],
                "classifier_hidden_size": 64,
                "activation": "LeakyReLU",
                "dropout": 0.0,
                "batch_norm": False,
            },
            "augmentation_config": {"enabled": False, "rotation_degrees": 0.0, "translate": (0.0, 0.0), "scale": (1.0, 1.0)},
        },
        {
            "run_name": "tune_13_leaky_deep_bn",
            "description": "Tests LeakyReLU instead of ReLU in the deeper batch-normalized architecture.",
            "model_name": "cnn_batchnorm",
            "learning_rate": 0.001,
            "weight_decay": 1e-4,
            "l1_lambda": 0.0,
            "input_noise_std": 0.0,
            "adam_betas": (0.9, 0.999),
            "adam_eps": 1e-8,
            "model_overrides": {
                "model_name": "cnn_batchnorm_leaky",
                "conv_channels": [32, 64, 64],
                "classifier_hidden_size": 256,
                "activation": "LeakyReLU",
                "dropout": 0.1,
                "batch_norm": True,
            },
            "augmentation_config": {"enabled": False, "rotation_degrees": 0.0, "translate": (0.0, 0.0), "scale": (1.0, 1.0)},
        },
        {
            "run_name": "tune_14_wide_3conv",
            "description": "Tests a wider 3-conv architecture with moderate regularization and small input noise.",
            "model_name": "cnn_regularized",
            "learning_rate": 0.001,
            "weight_decay": 1e-4,
            "l1_lambda": 0.0,
            "input_noise_std": 0.02,
            "adam_betas": (0.9, 0.999),
            "adam_eps": 1e-8,
            "model_overrides": {
                "model_name": "cnn_wide_3conv",
                "conv_channels": [48, 96, 128],
                "classifier_hidden_size": 256,
                "activation": "ReLU",
                "dropout": 0.2,
                "batch_norm": True,
            },
            "augmentation_config": {"enabled": False, "rotation_degrees": 0.0, "translate": (0.0, 0.0), "scale": (1.0, 1.0)},
        },
        {
            "run_name": "tune_15_small_kernel_3conv",
            "description": "Tests a smaller 3-conv network with fewer channels but still using batch normalization and regularization.",
            "model_name": "cnn_regularized",
            "learning_rate": 0.001,
            "weight_decay": 1e-4,
            "l1_lambda": 1e-6,
            "input_noise_std": 0.02,
            "adam_betas": (0.9, 0.999),
            "adam_eps": 1e-8,
            "model_overrides": {
                "model_name": "cnn_small_kernel_3conv",
                "conv_channels": [24, 48, 64],
                "kernel_size": 3,
                "classifier_hidden_size": 192,
                "activation": "ReLU",
                "dropout": 0.15,
                "batch_norm": True,
            },
            "augmentation_config": {"enabled": False, "rotation_degrees": 0.0, "translate": (0.0, 0.0), "scale": (1.0, 1.0)},
        },
        {
            "run_name": "tune_16_adam_beta_variant",
            "description": "Tests whether changing Adam beta values improves training dynamics.",
            "model_name": "cnn_batchnorm",
            "learning_rate": 0.001,
            "weight_decay": 1e-4,
            "l1_lambda": 0.0,
            "input_noise_std": 0.0,
            "adam_betas": (0.85, 0.995),
            "adam_eps": 1e-8,
            "model_overrides": None,
            "augmentation_config": {"enabled": False, "rotation_degrees": 0.0, "translate": (0.0, 0.0), "scale": (1.0, 1.0)},
        },
        {
            "run_name": "tune_17_adam_eps_variant",
            "description": "Tests whether a larger Adam epsilon improves optimization stability.",
            "model_name": "cnn_batchnorm",
            "learning_rate": 0.001,
            "weight_decay": 1e-4,
            "l1_lambda": 0.0,
            "input_noise_std": 0.0,
            "adam_betas": (0.9, 0.999),
            "adam_eps": 1e-7,
            "model_overrides": None,
            "augmentation_config": {"enabled": False, "rotation_degrees": 0.0, "translate": (0.0, 0.0), "scale": (1.0, 1.0)},
        },
        {
            "run_name": "tune_18_l1_regularized",
            "description": "Tests explicit L1 regularization added to a batch-normalized 3-conv model.",
            "model_name": "cnn_batchnorm",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "l1_lambda": 1e-6,
            "input_noise_std": 0.0,
            "adam_betas": (0.9, 0.999),
            "adam_eps": 1e-8,
            "model_overrides": None,
            "augmentation_config": {"enabled": False, "rotation_degrees": 0.0, "translate": (0.0, 0.0), "scale": (1.0, 1.0)},
        },
    ]


def load_search_space(config_path):
    """Load a JSON list of tuning specifications or fall back to defaults."""
    if config_path is None:
        return build_default_search_space()
    return json.loads(Path(config_path).read_text(encoding="utf-8"))


def main():
    """Run the full tuning sweep."""
    args = build_parser().parse_args()
    run_specs = load_search_space(args.config_path)
    tuning_timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    tuning_root = OUTPUT_ROOT / f"hyperparameter_tuning_{tuning_timestamp}"
    print(f"Saving tuning runs under: {tuning_root}")

    for spec in run_specs:
        output_dir = tuning_root / spec["run_name"]
        print(f"\n=== {spec['run_name']} ===")
        run_experiment(
            model_name=spec["model_name"],
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=spec["learning_rate"],
            output_dir=output_dir,
            checkpoint_interval=args.checkpoint_interval,
            validation_ratio=args.validation_ratio,
            seed=args.seed,
            early_stopping_patience=args.early_stopping_patience,
            augmentation_config=spec.get("augmentation_config"),
            weight_decay=spec.get("weight_decay", 0.0),
            l1_lambda=spec.get("l1_lambda", 0.0),
            input_noise_std=spec.get("input_noise_std", 0.0),
            model_overrides=spec.get("model_overrides"),
            adam_betas=tuple(spec.get("adam_betas", (0.9, 0.999))),
            adam_eps=spec.get("adam_eps", 1e-8),
        )

    notebook_path = create_tuning_notebook(tuning_root, run_specs)
    execute_notebook(notebook_path)
    print(f"\nSaved executed tuning notebook to: {notebook_path}")


if __name__ == "__main__":
    main()
