#!/usr/bin/env python3
"""Compare several MNIST CNN variants and generate a report notebook."""

import argparse
from datetime import datetime

try:
    from .main import AVAILABLE_MODELS, OUTPUT_ROOT, run_experiment
    from .notebook_templates import (
        create_cnn_comparison_notebook as create_comparison_notebook,
    )
    from .notebook_utils import execute_notebook, load_json
except ImportError:
    if __package__:
        raise
    from main import AVAILABLE_MODELS, OUTPUT_ROOT, run_experiment
    from notebook_templates import (
        create_cnn_comparison_notebook as create_comparison_notebook,
    )
    from notebook_utils import execute_notebook, load_json


DEFAULT_MODELS = (
    "cnn_small",
    "cnn_medium",
    "cnn_dropout",
    "cnn_deep_balanced",
    "cnn_deep_wide",
)


def build_parser():
    """Create a CLI for running a CNN architecture comparison."""
    parser = argparse.ArgumentParser(
        description="Compare multiple MNIST CNN variants.",
        epilog=(
            "Example: python part_2/cnn_comparison.py --epochs 10 "
            "--models cnn_small cnn_medium cnn_dropout"
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="Model names to compare.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--checkpoint-interval", type=int, default=5)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    return parser


def validate_models(model_names):
    """Ensure only supported model names are requested."""
    unknown_models = [name for name in model_names if name not in AVAILABLE_MODELS]
    if unknown_models:
        raise ValueError(
            "Unknown model(s): "
            + ", ".join(unknown_models)
            + f". Available models: {', '.join(AVAILABLE_MODELS)}"
        )


def main():
    """Run several CNN variants and summarize the results."""
    args = build_parser().parse_args()
    validate_models(args.models)
    comparison_timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    comparison_root = OUTPUT_ROOT / f"cnn_comparison_{comparison_timestamp}"

    print(f"Saving CNN comparison runs under: {comparison_root}")

    summaries = []
    for model_name in args.models:
        output_dir = comparison_root / model_name
        print(f"\n=== {model_name} ===")
        print(f"Output directory: {output_dir}")
        run_experiment(
            model_name=model_name,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            output_dir=output_dir,
            checkpoint_interval=args.checkpoint_interval,
            validation_ratio=args.validation_ratio,
            seed=args.seed,
            early_stopping_patience=args.early_stopping_patience,
        )
        summary_path = output_dir / "summary.json"
        summary = load_json(summary_path)
        summaries.append((model_name, summary_path, summary))

    print("\nFinished CNN comparison.")
    print("\nSummary comparison:")
    for model_name, summary_path, summary in summaries:
        print(
            f"{model_name}: "
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
