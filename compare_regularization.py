#!/usr/bin/env python3
"""Compare several regularization setups for the MNIST CNN."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from main import CURRENT_DIR, run_experiment


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


def _markdown_cell(source):
    """Build a markdown notebook cell."""
    return nbformat.v4.new_markdown_cell(source=source)


def _code_cell(source):
    """Build a code notebook cell."""
    return nbformat.v4.new_code_cell(source=source)


def create_comparison_notebook(comparison_root, summaries):
    """Create a notebook that compares regularization runs."""
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
            "# Regularization Comparison Report\n\n"
            "This notebook compares several regularization strategies on MNIST."
        ),
        _code_cell(
            "import json\n"
            "from pathlib import Path\n"
            "import pandas as pd\n"
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
            "    config = load_json(run_dir / 'config.json')\n"
            "    summary = load_json(run_dir / 'summary.json')\n"
            "    rows.append({\n"
            "        'run': run_name,\n"
            "        'model': config['model_name'],\n"
            "        'dropout': config.get('dropout', 0.0),\n"
            "        'batch_norm': config.get('batch_norm', False),\n"
            "        'weight_decay': config.get('weight_decay', 0.0),\n"
            "        'l1_lambda': config.get('l1_lambda', 0.0),\n"
            "        'input_noise_std': config.get('input_noise_std', 0.0),\n"
            "        'test_acc': summary['final_test_accuracy'],\n"
            "        'best_val_acc': summary['best_validation_accuracy'],\n"
            "        'best_val_loss': summary['best_validation_loss'],\n"
            "        'time_s': summary['total_training_time_seconds'],\n"
            "    })\n"
            "rows"
        ),
        _markdown_cell("## Pandas Overview"),
        _code_cell(
            "df = pd.DataFrame(rows)\n"
            "display_columns = [\n"
            "    'run',\n"
            "    'model',\n"
            "    'dropout',\n"
            "    'batch_norm',\n"
            "    'weight_decay',\n"
            "    'l1_lambda',\n"
            "    'input_noise_std',\n"
            "    'best_val_loss',\n"
            "    'best_val_acc',\n"
            "    'test_acc',\n"
            "    'time_s',\n"
            "]\n"
            "df[display_columns].sort_values(by='test_acc', ascending=False).reset_index(drop=True)"
        ),
        _markdown_cell(
            "## Interpretation\n\n"
            "Regularization methods mainly help by reducing overfitting and making "
            "optimization more stable. Dropout randomly removes features during training, "
            "weight decay discourages very large weights, batch normalization stabilizes "
            "activations across mini-batches, and input noise injection makes the model "
            "less sensitive to small perturbations."
        ),
        _code_cell(
            "best_run = max(rows, key=lambda row: row['test_acc'])\n"
            "fastest_run = min(rows, key=lambda row: row['time_s'])\n"
            "{'best_test_run': best_run, 'fastest_run': fastest_run}"
        ),
        _markdown_cell("## Saved Plots"),
        _code_cell(
            "plot_files = ['loss_curve.png', 'accuracy_curve.png', 'confusion_matrix.png']\n"
            "for run_name in RUN_NAMES:\n"
            "    run_dir = Path(RUN_DIRS[run_name])\n"
            "    print(f'\\n## {run_name}')\n"
            "    for plot_name in plot_files:\n"
            "        display(Image(filename=str(run_dir / plot_name)))\n"
        ),
        _markdown_cell(
            "## Final Summary\n\n"
            "This section is generated from the saved metrics for the current comparison run."
        ),
        _code_cell(
            "best_test = max(rows, key=lambda row: row['test_acc'])\n"
            "best_val_loss = min(rows, key=lambda row: row['best_val_loss'])\n"
            "fastest = min(rows, key=lambda row: row['time_s'])\n"
            "baseline = next((row for row in rows if row['run'] == 'baseline'), None)\n"
            "\n"
            "lines = [\n"
            "    'English summary:',\n"
            "    (\n"
            "        f\"Best test accuracy: {best_test['run']} ({best_test['test_acc']:.2%}). \"\n"
            "        f\"Best validation loss: {best_val_loss['run']} ({best_val_loss['best_val_loss']:.4f}). \"\n"
            "        f\"Fastest run: {fastest['run']} ({fastest['time_s']:.2f}s).\"\n"
            "    ),\n"
            "]\n"
            "\n"
            "if baseline is not None and best_test['run'] != 'baseline':\n"
            "    gap = best_test['test_acc'] - baseline['test_acc']\n"
            "    lines.append(\n"
            "        f\"Compared with the baseline, the best-test run changes accuracy by {gap:+.2%}.\"\n"
            "    )\n"
            "elif baseline is not None:\n"
            "    lines.append('The baseline run remained the strongest model on the test set in this comparison.')\n"
            "\n"
            "lines.extend([\n"
            "    '',\n"
            "    'Kort svensk sammanfattning:',\n"
            "    (\n"
            "        f\"Högst test accuracy: {best_test['run']} ({best_test['test_acc']:.2%}). \"\n"
            "        f\"Bäst validation loss: {best_val_loss['run']} ({best_val_loss['best_val_loss']:.4f}). \"\n"
            "        f\"Snabbast körning: {fastest['run']} ({fastest['time_s']:.2f}s).\"\n"
            "    ),\n"
            "])\n"
            "\n"
            "if baseline is not None and best_test['run'] != 'baseline':\n"
            "    gap = best_test['test_acc'] - baseline['test_acc']\n"
            "    lines.append(\n"
            "        f\"Jämfört med baseline ändras test accuracy med {gap:+.2%}.\"\n"
            "    )\n"
            "elif baseline is not None:\n"
            "    lines.append('Baseline var fortfarande starkast på testmängden i denna jämförelse.')\n"
            "\n"
            "print('\\n'.join(lines))\n"
            "lines"
        ),
    ]
    notebook.metadata["language_info"] = {"name": "python"}
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook_path.write_text(nbformat.writes(notebook, version=4), encoding="utf-8")
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
    """Run the predefined regularization experiments."""
    args = build_parser().parse_args()
    comparison_timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    comparison_root = (
        CURRENT_DIR / "outputs" / "Part2" / f"regularization_comparison_{comparison_timestamp}"
    )
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
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summaries.append((spec["run_name"], summary_path, summary))

    notebook_path = create_comparison_notebook(comparison_root, summaries)
    execute_notebook(notebook_path)
    print(f"\nSaved executed comparison notebook to: {notebook_path}")


if __name__ == "__main__":
    main()
