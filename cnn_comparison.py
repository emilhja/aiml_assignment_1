#!/usr/bin/env python3
"""Compare several MNIST CNN variants and generate a report notebook."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from main import AVAILABLE_MODELS, CURRENT_DIR, run_experiment


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
            "Example: python cnn_comparison.py --epochs 10 "
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


def _markdown_cell(source):
    """Build a markdown notebook cell."""
    return nbformat.v4.new_markdown_cell(source=source)


def _code_cell(source):
    """Build a code notebook cell."""
    return nbformat.v4.new_code_cell(source=source)


def create_comparison_notebook(comparison_root, summaries):
    """Create a notebook that compares the selected CNN runs."""
    comparison_root = Path(comparison_root)
    notebook_path = comparison_root / "comparison_report.ipynb"
    run_paths = {
        run_name: str(summary_path.parent)
        for run_name, summary_path, _summary in summaries
    }
    ordered_run_names = [run_name for run_name, _summary_path, _summary in summaries]

    notebook = nbformat.v4.new_notebook()
    notebook.cells = [
        _markdown_cell(
            "# CNN Comparison Report\n\n"
            "This notebook compares several CNN variants trained on MNIST."
        ),
        _code_cell(
            "import json\n"
            "from pathlib import Path\n"
            "from IPython.display import Image, display\n"
            "\n"
            f'COMPARISON_DIR = Path(r"{comparison_root}")\n'
            f"RUN_NAMES = {ordered_run_names!r}\n"
            f"RUN_DIRS = {run_paths!r}\n"
            "\n"
            "def load_json(path):\n"
            "    return json.loads(Path(path).read_text(encoding='utf-8'))\n"
        ),
        _markdown_cell("## Summary Table"),
        _code_cell(
            "rows = []\n"
            "for run_name in RUN_NAMES:\n"
            "    run_dir = Path(RUN_DIRS[run_name])\n"
            "    summary = load_json(run_dir / 'summary.json')\n"
            "    config = load_json(run_dir / 'config.json')\n"
            "    rows.append({\n"
            "        'model': run_name,\n"
            "        'best_epoch': summary['best_epoch'],\n"
            "        'best_val_loss': summary['best_validation_loss'],\n"
            "        'best_val_acc': summary['best_validation_accuracy'],\n"
            "        'test_acc': summary['final_test_accuracy'],\n"
            "        'total_time_s': summary['total_training_time_seconds'],\n"
            "        'conv_channels': config.get('conv_channels'),\n"
            "        'num_conv_layers': config.get('num_conv_layers', 0),\n"
            "        'trainable_parameters': config.get('trainable_parameters'),\n"
            "        'dropout': config.get('dropout', 0.0),\n"
            "        'activation': config.get('activation'),\n"
            "    })\n"
            "rows"
        ),
        _markdown_cell(
            "## Parameter-Aware Comparison\n\n"
            "When evaluating architecture changes, models with very different parameter "
            "counts should not be treated as a clean apples-to-apples comparison. "
            "Use the table below to compare depth changes while keeping parameter "
            "counts reasonably close."
        ),
        _code_cell(
            "sorted_rows = sorted(rows, key=lambda row: row['trainable_parameters'] or 0)\n"
            "sorted_rows"
        ),
        _markdown_cell("## Saved Plots"),
        _code_cell(
            "plot_files = [\n"
            "    'loss_curve.png',\n"
            "    'accuracy_curve.png',\n"
            "    'confusion_matrix.png',\n"
            "]\n"
            "\n"
            "for run_name in RUN_NAMES:\n"
            "    run_dir = Path(RUN_DIRS[run_name])\n"
            "    print(f'\\n## {run_name}')\n"
            "    for plot_name in plot_files:\n"
            "        print(plot_name)\n"
            "        display(Image(filename=str(run_dir / plot_name)))\n"
        ),
        _markdown_cell(
            "## First Vs Later Convolution Filters\n\n"
            "The first convolution layer usually learns simple local patterns such as "
            "edges, stroke directions, and small blobs. Later convolution layers usually "
            "combine those earlier responses into more structured digit parts, for example "
            "curves, corners, loops, and stroke combinations that are useful for whole-digit "
            "recognition."
        ),
        _code_cell(
            "filter_plot_files = [\n"
            "    'conv_filters_first.png',\n"
            "    'conv_filters_last.png',\n"
            "]\n"
            "\n"
            "for run_name in RUN_NAMES:\n"
            "    run_dir = Path(RUN_DIRS[run_name])\n"
            "    print(f'\\n## {run_name}')\n"
            "    for plot_name in filter_plot_files:\n"
            "        plot_path = run_dir / plot_name\n"
            "        print(plot_name)\n"
            "        if plot_path.exists():\n"
            "            display(Image(filename=str(plot_path)))\n"
            "        else:\n"
            "            print('Missing:', plot_path)\n"
        ),
        _markdown_cell("## Architecture Notes"),
        _code_cell(
            "for row in rows:\n"
            "    print(\n"
            "        f\"{row['model']}: {row['num_conv_layers']} conv layers | \"\n"
            "        f\"channels={row['conv_channels']} | \"\n"
            "        f\"params={row['trainable_parameters']:,} | \"\n"
            "        f\"activation={row['activation']} | dropout={row['dropout']}\"\n"
            "    )\n"
            "\n"
            "print('\\nInterpretation:')\n"
            "print('- More convolution layers increase representational depth.')\n"
            "print('- First-layer filters tend to be edge or stroke detectors.')\n"
            "print('- Later-layer filters represent combinations of earlier patterns.')\n"
            "print('- Compare accuracy together with parameter count and training time.')\n"
        ),
        _markdown_cell("## Final Remark"),
        _code_cell(
            "best_by_test = None\n"
            "best_test_acc = -1.0\n"
            "fastest_model = None\n"
            "fastest_time = None\n"
            "\n"
            "for run_name in RUN_NAMES:\n"
            "    run_dir = Path(RUN_DIRS[run_name])\n"
            "    summary = load_json(run_dir / 'summary.json')\n"
            "    test_acc = summary['final_test_accuracy']\n"
            "    total_time = summary['total_training_time_seconds']\n"
            "    if test_acc > best_test_acc:\n"
            "        best_test_acc = test_acc\n"
            "        best_by_test = run_name\n"
            "    if fastest_time is None or total_time < fastest_time:\n"
            "        fastest_time = total_time\n"
            "        fastest_model = run_name\n"
            "\n"
            "print(f'Best test accuracy: {best_by_test} ({best_test_acc:.2%})')\n"
            "print(f'Fastest training run: {fastest_model} ({fastest_time:.2f}s)')\n"
        ),
        _markdown_cell("## Recommended Model"),
        _code_cell(
            "rows = []\n"
            "for run_name in RUN_NAMES:\n"
            "    run_dir = Path(RUN_DIRS[run_name])\n"
            "    summary = load_json(run_dir / 'summary.json')\n"
            "    rows.append({\n"
            "        'model': run_name,\n"
            "        'test_acc': summary['final_test_accuracy'],\n"
            "        'time_s': summary['total_training_time_seconds'],\n"
            "        'params': load_json(run_dir / 'config.json').get('trainable_parameters', 0),\n"
            "    })\n"
            "\n"
            "best_test_acc = max(row['test_acc'] for row in rows)\n"
            "fastest_time = min(row['time_s'] for row in rows)\n"
            "\n"
            "for row in rows:\n"
            "    row['acc_gap'] = best_test_acc - row['test_acc']\n"
            "    row['time_ratio'] = row['time_s'] / fastest_time if fastest_time > 0 else 1.0\n"
            "    row['param_ratio'] = row['params'] / min(candidate['params'] for candidate in rows if candidate['params'] > 0)\n"
            "\n"
            "preferred_order = {\n"
            "    'cnn_medium': 0,\n"
            "    'cnn_deep_balanced': 1,\n"
            "    'cnn_deep_wide': 2,\n"
            "    'cnn_small': 3,\n"
            "    'cnn_dropout': 4,\n"
            "    'mlp': 5,\n"
            "}\n"
            "recommended = min(\n"
            "    rows,\n"
            "    key=lambda row: (\n"
            "        row['acc_gap'] > 0.002,\n"
            "        row['acc_gap'],\n"
            "        row['time_ratio'] > 1.25,\n"
            "        row['param_ratio'] > 1.8,\n"
            "        row['time_s'],\n"
            "        preferred_order.get(row['model'], 99),\n"
            "    ),\n"
            ")\n"
            "\n"
            "recommendation = (\n"
            "    f\"Among the tested CNN variants, {recommended['model']} gave the best balance \"\n"
            "    f\"between simplicity, training time, and accuracy, so it was selected \"\n"
            "    f\"as the final architecture.\"\n"
            ")\n"
            "print(recommendation)\n"
            "recommendation\n"
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
    """Run several CNN variants and summarize the results."""
    args = build_parser().parse_args()
    validate_models(args.models)
    comparison_timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    comparison_root = (
        CURRENT_DIR
        / "outputs"
        / "Part2"
        / f"cnn_comparison_{comparison_timestamp}"
    )

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
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
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
