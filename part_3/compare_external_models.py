#!/usr/bin/env python3
"""Compare Part 3 Oxford Pet models and generate a runnable notebook report."""

import argparse
import json
import sys
from argparse import Namespace
from datetime import datetime
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from part_3.notebook_templates import create_external_model_comparison_notebook
from part_3.notebook_utils import execute_notebook
from part_3.part3_finetuning_external_models import (
    AVAILABLE_MODELS,
    run_experiment,
)

DEFAULT_MODELS = (
    "scratch_cnn",
    "deeper_cnn",
    "resnet18_transfer",
    "resnet50_transfer",
    "mobilenet_v3_transfer",
)
TRANSFER_MODELS = {
    "mobilenet_v3_transfer",
    "resnet18_transfer",
    "resnet50_transfer",
}


def build_parser():
    """Create a CLI for Part 3 model comparison runs."""
    parser = argparse.ArgumentParser(
        description="Compare scratch and transfer models on Oxford-IIIT Pet.",
        epilog=(
            "Example: .\\venv\\Scripts\\python.exe part_3\\compare_external_models.py "
            "--epochs-head 1 --epochs-finetune 1 --models resnet18_transfer mobilenet_v3_transfer"
        ),
    )
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--dataset-root", type=str, default="data")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--epochs-head",
        type=int,
        default=None,
        help="Head-training epochs. Defaults depend on the selected model.",
    )
    parser.add_argument(
        "--epochs-finetune",
        type=int,
        default=None,
        help="Fine-tuning epochs. Defaults depend on the selected model.",
    )
    parser.add_argument(
        "--deeper-cnn-epochs",
        type=int,
        default=80,
        help=(
            "Total epochs for deeper_cnn when --epochs-head is not explicitly set. "
            "Defaults to 60 for the comparison run."
        ),
    )
    parser.add_argument(
        "--scratch-cnn-epochs",
        type=int,
        default=50,
        help=(
            "Total epochs for scratch_cnn when epoch stage overrides are not "
            "explicitly set. Defaults to 50 for the comparison run."
        ),
    )
    parser.add_argument(
        "--transfer-epochs",
        type=int,
        default=25,
        help=(
            "Total epochs for transfer models when epoch stage overrides are not "
            "explicitly set. Defaults to 25 for the comparison run."
        ),
    )
    parser.add_argument("--learning-rate-head", type=float, default=1e-3)
    parser.add_argument("--learning-rate-finetune", type=float, default=1e-4)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser


def validate_models(model_names):
    """Ensure all requested models are supported."""
    unknown_models = [name for name in model_names if name not in AVAILABLE_MODELS]
    if unknown_models:
        raise ValueError(
            "Unknown model(s): "
            + ", ".join(unknown_models)
            + f". Available models: {', '.join(AVAILABLE_MODELS)}"
        )


def split_total_epochs(total_epochs, first_stage_epochs=3):
    """Split a total epoch budget into first-stage and second-stage epochs."""
    if total_epochs <= 0:
        raise ValueError("Epoch counts must be positive.")
    first_stage = min(first_stage_epochs, total_epochs)
    second_stage = total_epochs - first_stage
    return first_stage, second_stage


def resolve_comparison_epochs(model_name, args):
    """Resolve comparison-specific epoch defaults for one model."""
    if model_name == "deeper_cnn":
        default_head, default_finetune = args.deeper_cnn_epochs, 0
    elif model_name == "scratch_cnn":
        default_head, default_finetune = split_total_epochs(args.scratch_cnn_epochs)
    elif model_name in TRANSFER_MODELS:
        default_head, default_finetune = split_total_epochs(args.transfer_epochs)
    else:
        default_head, default_finetune = None, None

    epochs_head = default_head if args.epochs_head is None else args.epochs_head
    epochs_finetune = (
        default_finetune
        if args.epochs_finetune is None
        else args.epochs_finetune
    )
    return epochs_head, epochs_finetune


def main():
    """Run the requested Part 3 models and create a comparison notebook."""
    args = build_parser().parse_args()
    validate_models(args.models)
    comparison_timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    comparison_root = (
        CURRENT_DIR
        / "part_3"
        / "outputs"
        / f"external_model_comparison_{comparison_timestamp}"
    )

    print(f"Saving Part 3 comparison runs under: {comparison_root}")

    summaries = []
    for model_name in args.models:
        output_dir = comparison_root / model_name
        print(f"\n=== {model_name} ===")
        print(f"Output directory: {output_dir}")
        epochs_head, epochs_finetune = resolve_comparison_epochs(model_name, args)
        print(f"Epoch schedule: head={epochs_head}, finetune={epochs_finetune}")

        run_args = Namespace(
            model=model_name,
            dataset_root=args.dataset_root,
            batch_size=args.batch_size,
            epochs_head=epochs_head,
            epochs_finetune=epochs_finetune,
            learning_rate_head=args.learning_rate_head,
            learning_rate_finetune=args.learning_rate_finetune,
            validation_ratio=args.validation_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            output_dir=str(output_dir),
            checkpoint_interval=args.checkpoint_interval,
            num_workers=args.num_workers,
        )
        run_experiment(run_args)
        summary_path = output_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summaries.append((model_name, summary_path, summary))

    print("\nFinished Part 3 comparison.")
    for model_name, summary_path, summary in summaries:
        print(
            f"{model_name}: "
            f"best_epoch={summary['best_epoch']} | "
            f"best_val_acc={summary['best_validation_accuracy']:.2%} | "
            f"test_acc={summary['final_test_accuracy']:.2%} | "
            f"macro_f1={summary['final_test_macro_f1']:.4f} | "
            f"time={summary['total_training_time_seconds']:.2f}s | "
            f"test_eval_time={summary.get('test_evaluation_time_seconds', float('nan')):.2f}s | "
            f"summary={summary_path}"
        )

    notebook_path = create_external_model_comparison_notebook(comparison_root, summaries)
    execute_notebook(notebook_path, timeout=1800)
    print(f"\nSaved executed comparison notebook to: {notebook_path}")


if __name__ == "__main__":
    main()
