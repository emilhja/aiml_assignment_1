#!/usr/bin/env python3
"""Run the MNIST training pipeline with and without augmentation."""

import argparse
from datetime import datetime

try:
    from .main import OUTPUT_ROOT, run_experiment
    from .notebook_templates import (
        create_augmentation_comparison_notebook as create_comparison_notebook,
    )
    from .notebook_utils import execute_notebook, load_json
except ImportError:
    if __package__:
        raise
    from main import OUTPUT_ROOT, run_experiment
    from notebook_templates import (
        create_augmentation_comparison_notebook as create_comparison_notebook,
    )
    from notebook_utils import execute_notebook, load_json


def build_parser():
    """Create a CLI for comparing baseline and augmented training runs."""
    parser = argparse.ArgumentParser(
        description="Compare MNIST training with and without data augmentation."
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--checkpoint-interval", type=int, default=5)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--rotation-degrees", type=float, default=10.0)
    parser.add_argument(
        "--translate-x",
        type=float,
        default=0.1,
        help="Maximum horizontal translation fraction for RandomAffine.",
    )
    parser.add_argument(
        "--translate-y",
        type=float,
        default=0.1,
        help="Maximum vertical translation fraction for RandomAffine.",
    )
    parser.add_argument(
        "--scale-min",
        type=float,
        default=0.95,
        help="Minimum scale factor for RandomAffine.",
    )
    parser.add_argument(
        "--scale-max",
        type=float,
        default=1.05,
        help="Maximum scale factor for RandomAffine.",
    )
    return parser


def build_run_settings(args):
    """Return the paired experiment settings used for augmentation A/B tests."""
    baseline_config = {
        "enabled": False,
        "rotation_degrees": 0.0,
        "translate": (0.0, 0.0),
        "scale": (1.0, 1.0),
    }
    augmented_config = {
        "enabled": True,
        "rotation_degrees": args.rotation_degrees,
        "translate": (args.translate_x, args.translate_y),
        "scale": (args.scale_min, args.scale_max),
    }
    return [
        ("without_augmentation", baseline_config),
        ("with_augmentation", augmented_config),
    ]


def main():
    """Run the same experiment twice so augmentation can be compared directly."""
    args = build_parser().parse_args()
    comparison_timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    comparison_root = OUTPUT_ROOT / f"augmentation_comparison_{comparison_timestamp}"

    print(f"Saving augmentation comparison runs under: {comparison_root}")

    summaries = []
    for run_name, augmentation_config in build_run_settings(args):
        output_dir = comparison_root / run_name
        print(f"\n=== {run_name} ===")
        print(f"Output directory: {output_dir}")
        print(f"Augmentation config: {augmentation_config}")

        run_experiment(
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            output_dir=output_dir,
            checkpoint_interval=args.checkpoint_interval,
            validation_ratio=args.validation_ratio,
            seed=args.seed,
            early_stopping_patience=args.early_stopping_patience,
            augmentation_config=augmentation_config,
        )
        summary_path = output_dir / "summary.json"
        summary = load_json(summary_path)
        summaries.append((run_name, summary_path, summary))

    print("\nFinished augmentation comparison.")
    print(
        "Inspect the two run folders and the shared experiments.db in the "
        "comparison directory."
    )
    print("\nSummary comparison:")
    for run_name, summary_path, summary in summaries:
        print(
            f"{run_name}: "
            f"best_epoch={summary['best_epoch']} | "
            f"best_val_loss={summary['best_validation_loss']:.4f} | "
            f"best_val_acc={summary['best_validation_accuracy']:.2%} | "
            f"test_acc={summary['final_test_accuracy']:.2%} | "
            f"total_time={summary['total_training_time_seconds']:.2f}s | "
            f"summary={summary_path}"
        )
    notebook_path = create_comparison_notebook(comparison_root, summaries)
    execute_notebook(notebook_path)
    print(f"\nSaved executed comparison notebook to: {notebook_path}")


if __name__ == "__main__":
    main()
