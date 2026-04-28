#!/usr/bin/env python3
"""Compare several regularization setups for the MNIST CNN."""

import argparse
from datetime import datetime

try:
    from .main import OUTPUT_ROOT, run_experiment
    from .notebook_templates import (
        create_regularization_comparison_notebook as create_comparison_notebook,
    )
    from .notebook_utils import execute_notebook, load_json
except ImportError:
    if __package__:
        raise
    from main import OUTPUT_ROOT, run_experiment
    from notebook_templates import (
        create_regularization_comparison_notebook as create_comparison_notebook,
    )
    from notebook_utils import execute_notebook, load_json


def build_parser():
    """Create a CLI for comparing regularization strategies."""
    parser = argparse.ArgumentParser(
        description="Compare dropout, batch norm, weight decay, and noise injection."
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--checkpoint-interval", type=int, default=5)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    return parser


def build_run_settings():
    """Return a fixed set of regularization experiments."""
    return [
        {
            "run_name": "baseline",
            "model_name": "cnn_medium",
            "weight_decay": 0.0,
            "l1_lambda": 0.0,
            "input_noise_std": 0.0,
        },
        {
            "run_name": "dropout_only",
            "model_name": "cnn_dropout",
            "weight_decay": 0.0,
            "l1_lambda": 0.0,
            "input_noise_std": 0.0,
        },
        {
            "run_name": "batchnorm_only",
            "model_name": "cnn_batchnorm",
            "weight_decay": 0.0,
            "l1_lambda": 0.0,
            "input_noise_std": 0.0,
        },
        {
            "run_name": "weight_decay_only",
            "model_name": "cnn_medium",
            "weight_decay": 1e-4,
            "l1_lambda": 0.0,
            "input_noise_std": 0.0,
        },
        {
            "run_name": "l1_only",
            "model_name": "cnn_medium",
            "weight_decay": 0.0,
            "l1_lambda": 1e-6,
            "input_noise_std": 0.0,
        },
        {
            "run_name": "combined_regularization",
            "model_name": "cnn_regularized",
            "weight_decay": 1e-4,
            "l1_lambda": 1e-6,
            "input_noise_std": 0.05,
        },
    ]


def main():
    """Run the predefined regularization experiments."""
    args = build_parser().parse_args()
    comparison_timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    comparison_root = OUTPUT_ROOT / f"regularization_comparison_{comparison_timestamp}"
    print(f"Saving regularization comparison runs under: {comparison_root}")

    summaries = []
    for spec in build_run_settings():
        output_dir = comparison_root / spec["run_name"]
        print(f"\n=== {spec['run_name']} ===")
        run_experiment(
            model_name=spec["model_name"],
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            output_dir=output_dir,
            checkpoint_interval=args.checkpoint_interval,
            validation_ratio=args.validation_ratio,
            seed=args.seed,
            early_stopping_patience=args.early_stopping_patience,
            weight_decay=spec["weight_decay"],
            l1_lambda=spec["l1_lambda"],
            input_noise_std=spec["input_noise_std"],
        )
        summary_path = output_dir / "summary.json"
        summary = load_json(summary_path)
        summaries.append((spec["run_name"], summary_path, summary))

    notebook_path = create_comparison_notebook(comparison_root, summaries)
    execute_notebook(notebook_path)
    print(f"\nSaved executed comparison notebook to: {notebook_path}")


if __name__ == "__main__":
    main()
