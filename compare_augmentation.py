#!/usr/bin/env python3
"""Run the MNIST training pipeline with and without augmentation."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from main import CURRENT_DIR, run_experiment


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


def _markdown_cell(source):
    """Build a markdown notebook cell."""
    return nbformat.v4.new_markdown_cell(source=source)


def _code_cell(source):
    """Build a code notebook cell."""
    return nbformat.v4.new_code_cell(source=source)


def create_comparison_notebook(comparison_root, summaries):
    """Create a notebook that compares augmented and non-augmented runs."""
    comparison_root = Path(comparison_root)
    notebook_path = comparison_root / "comparison_report.ipynb"

    run_paths = {
        run_name: str(summary_path.parent)
        for run_name, summary_path, _summary in summaries
    }

    notebook = nbformat.v4.new_notebook()
    notebook.cells = [
        _markdown_cell(
            "# Augmentation Comparison Report\n\n"
            "This notebook compares the `without_augmentation` and "
            "`with_augmentation` runs saved in this folder."
        ),
        _code_cell(
            "import json\n"
            "from pathlib import Path\n"
            "from IPython.display import Image, display\n"
            "\n"
            f'COMPARISON_DIR = Path(r"{comparison_root}")\n'
            f'WITHOUT_RUN_DIR = Path(r"{run_paths["without_augmentation"]}")\n'
            f'WITH_RUN_DIR = Path(r"{run_paths["with_augmentation"]}")\n'
            "\n"
            "def load_json(path):\n"
            "    return json.loads(Path(path).read_text(encoding='utf-8'))\n"
            "\n"
            "without_config = load_json(WITHOUT_RUN_DIR / 'config.json')\n"
            "with_config = load_json(WITH_RUN_DIR / 'config.json')\n"
            "without_summary = load_json(WITHOUT_RUN_DIR / 'summary.json')\n"
            "with_summary = load_json(WITH_RUN_DIR / 'summary.json')\n"
            "without_history = load_json(WITHOUT_RUN_DIR / 'training_history.json')\n"
            "with_history = load_json(WITH_RUN_DIR / 'training_history.json')\n"
            "print(f'Comparison folder: {COMPARISON_DIR}')"
        ),
        _markdown_cell("## Summary Comparison"),
        _code_cell(
            "comparison = {\n"
            "    'without_augmentation': {\n"
            "        'best_epoch': without_summary['best_epoch'],\n"
            "        'best_validation_loss': without_summary['best_validation_loss'],\n"
            "        'best_validation_accuracy': without_summary['best_validation_accuracy'],\n"
            "        'final_test_accuracy': without_summary['final_test_accuracy'],\n"
            "        'total_training_time_seconds': without_summary['total_training_time_seconds'],\n"
            "    },\n"
            "    'with_augmentation': {\n"
            "        'best_epoch': with_summary['best_epoch'],\n"
            "        'best_validation_loss': with_summary['best_validation_loss'],\n"
            "        'best_validation_accuracy': with_summary['best_validation_accuracy'],\n"
            "        'final_test_accuracy': with_summary['final_test_accuracy'],\n"
            "        'total_training_time_seconds': with_summary['total_training_time_seconds'],\n"
            "    },\n"
            "}\n"
            "comparison"
        ),
        _markdown_cell("## Augmentation Settings"),
        _code_cell(
            "{\n"
            "    'without_augmentation': without_config.get('augmentation_config'),\n"
            "    'with_augmentation': with_config.get('augmentation_config'),\n"
            "}"
        ),
        _markdown_cell("## Saved Plots"),
        _code_cell(
            "plot_files = [\n"
            "    'loss_curve.png',\n"
            "    'accuracy_curve.png',\n"
            "    'confusion_matrix.png',\n"
            "    'correct_predictions.png',\n"
            "    'incorrect_predictions.png',\n"
            "]\n"
            "\n"
            "for plot_name in plot_files:\n"
            "    print(f'\\n### {plot_name} - without_augmentation')\n"
            "    display(Image(filename=str(WITHOUT_RUN_DIR / plot_name)))\n"
            "    print(f'### {plot_name} - with_augmentation')\n"
            "    display(Image(filename=str(WITH_RUN_DIR / plot_name)))\n"
        ),
        _markdown_cell("## Training History"),
        _code_cell(
            "{\n"
            "    'without_augmentation': without_history,\n"
            "    'with_augmentation': with_history,\n"
            "}"
        ),
        _markdown_cell("## Final Remark"),
        _code_cell(
            "test_acc_gap = (\n"
            "    without_summary['final_test_accuracy']\n"
            "    - with_summary['final_test_accuracy']\n"
            ")\n"
            "val_acc_gap = (\n"
            "    without_summary['best_validation_accuracy']\n"
            "    - with_summary['best_validation_accuracy']\n"
            ")\n"
            "val_loss_gap = (\n"
            "    with_summary['best_validation_loss']\n"
            "    - without_summary['best_validation_loss']\n"
            ")\n"
            "time_gap = (\n"
            "    with_summary['total_training_time_seconds']\n"
            "    - without_summary['total_training_time_seconds']\n"
            ")\n"
            "\n"
            "if test_acc_gap > 0:\n"
            "    winner = 'without_augmentation'\n"
            "elif test_acc_gap < 0:\n"
            "    winner = 'with_augmentation'\n"
            "else:\n"
            "    winner = 'tie'\n"
            "\n"
            "loss_winner = (\n"
            "    'without_augmentation'\n"
            "    if val_loss_gap > 0\n"
            "    else 'with_augmentation'\n"
            "    if val_loss_gap < 0\n"
            "    else 'neither run'\n"
            ")\n"
            "time_winner = (\n"
            "    'with_augmentation'\n"
            "    if time_gap > 0\n"
            "    else 'without_augmentation'\n"
            "    if time_gap < 0\n"
            "    else 'same runtime'\n"
            ")\n"
            "\n"
            "remark_lines = [\n"
            "    f\"Overall better run: {winner}\",\n"
            "    f\"Test accuracy difference: {abs(test_acc_gap):.2%}\",\n"
            "    f\"Best validation accuracy difference: {abs(val_acc_gap):.2%}\",\n"
            "    f\"Best validation loss better for: {loss_winner}\",\n"
            "    f\"Validation loss difference: {abs(val_loss_gap):.4f}\",\n"
            "    f\"Training time difference: {abs(time_gap):.2f}s\",\n"
            "    f\"Slower run: {time_winner}\",\n"
            "]\n"
            "print('\\n'.join(remark_lines))\n"
            "remark_lines"
        ),
    ]
    notebook.metadata["language_info"] = {"name": "python"}
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }

    notebook_path.write_text(
        nbformat.writes(notebook, version=4),
        encoding="utf-8",
    )
    return notebook_path


def execute_notebook(notebook_path, timeout=600):
    """Execute a notebook in place so outputs are saved."""
    notebook_path = Path(notebook_path)
    notebook = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(notebook_path.parent)}},
    )
    client.execute()
    notebook_path.write_text(nbformat.writes(notebook, version=4), encoding="utf-8")
    return notebook_path


def main():
    """Run the same experiment twice so augmentation can be compared directly."""
    args = build_parser().parse_args()
    comparison_timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    comparison_root = (
        CURRENT_DIR
        / "outputs"
        / "Part2"
        / f"augmentation_comparison_{comparison_timestamp}"
    )

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
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
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
