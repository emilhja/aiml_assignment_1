#!/usr/bin/env python3
"""Compare Part 3 Oxford Pet models and generate a runnable notebook report."""

import argparse
import json
import sys
from argparse import Namespace
from datetime import datetime
from pathlib import Path

import nbformat
from nbclient import NotebookClient

CURRENT_DIR = Path(__file__).resolve().parent.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from Part3.part3_finetuning_external_models import (
    AVAILABLE_MODELS,
    run_experiment,
)

DEFAULT_MODELS = (
    "scratch_cnn",
    "resnet18_transfer",
    "resnet50_transfer",
    "mobilenet_v3_transfer",
)


def build_parser():
    """Create a CLI for Part 3 model comparison runs."""
    parser = argparse.ArgumentParser(
        description="Compare scratch and transfer models on Oxford-IIIT Pet.",
        epilog=(
            "Example: .\\venv\\Scripts\\python.exe Part3\\compare_external_models.py "
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


def _markdown_cell(source):
    """Build a markdown notebook cell."""
    return nbformat.v4.new_markdown_cell(source=source)


def _code_cell(source):
    """Build a code notebook cell."""
    return nbformat.v4.new_code_cell(source=source)


def create_comparison_notebook(comparison_root, summaries):
    """Create a runnable notebook that compares multiple Part 3 runs."""
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
            "# Part 3 External Model Comparison\n\n"
            "This notebook compares the scratch CNN baseline against the transfer-learning runs."
        ),
        _code_cell(
            "import json\n"
            "from pathlib import Path\n"
            "\n"
            "import matplotlib.pyplot as plt\n"
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
            "        'model': run_name,\n"
            "        'backbone': config.get('backbone_name'),\n"
            "        'transfer_learning': config.get('transfer_learning'),\n"
            "        'weights': config.get('weights_name'),\n"
            "        'best_epoch': summary['best_epoch'],\n"
            "        'best_stage': summary['best_stage'],\n"
            "        'best_val_loss': summary['best_validation_loss'],\n"
            "        'best_val_acc': summary['best_validation_accuracy'],\n"
            "        'test_loss': summary['final_test_loss'],\n"
            "        'test_acc': summary['final_test_accuracy'],\n"
            "        'macro_f1': summary['final_test_macro_f1'],\n"
            "        'time_s': summary['total_training_time_seconds'],\n"
            "        'trainable_parameters': summary['trainable_parameters'],\n"
            "        'total_parameters': summary['total_parameters'],\n"
            "    })\n"
            "comparison_df = pd.DataFrame(rows).sort_values(by='test_acc', ascending=False).reset_index(drop=True)\n"
            "comparison_df"
        ),
        _markdown_cell("## Accuracy Vs Runtime"),
        _code_cell(
            "ax = comparison_df.plot.scatter(x='time_s', y='test_acc', s=120, figsize=(7, 5))\n"
            "for _, row in comparison_df.iterrows():\n"
            "    ax.annotate(row['model'], (row['time_s'], row['test_acc']))\n"
            "ax.grid(True, alpha=0.3)\n"
            "ax.set_title('Accuracy vs Runtime')\n"
            "plt.show()"
        ),
        _markdown_cell("## Parameter-Aware View"),
        _code_cell(
            "comparison_df[['model', 'test_acc', 'macro_f1', 'time_s', 'trainable_parameters', 'total_parameters']]"
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
            "for run_name in RUN_NAMES:\n"
            "    run_dir = Path(RUN_DIRS[run_name])\n"
            "    print(f'\\n## {run_name}')\n"
            "    for plot_name in plot_files:\n"
            "        plot_path = run_dir / plot_name\n"
            "        print(plot_name)\n"
            "        if plot_path.exists():\n"
            "            display(Image(filename=str(plot_path)))\n"
            "        else:\n"
            "            print('Missing:', plot_path)\n"
        ),
        _markdown_cell("## Per-Run Epoch Histories"),
        _code_cell(
            "history_frames = []\n"
            "for run_name in RUN_NAMES:\n"
            "    run_dir = Path(RUN_DIRS[run_name])\n"
            "    history_df = pd.DataFrame(load_json(run_dir / 'training_history.json'))\n"
            "    history_df['epoch'] = range(1, len(history_df) + 1)\n"
            "    history_df['model'] = run_name\n"
            "    history_frames.append(history_df)\n"
            "history_all = pd.concat(history_frames, ignore_index=True)\n"
            "history_all.head()"
        ),
        _code_cell(
            "for metric in ['val_accuracy', 'val_loss']:\n"
            "    plt.figure(figsize=(8, 4))\n"
            "    for run_name in RUN_NAMES:\n"
            "        subset = history_all[history_all['model'] == run_name]\n"
            "        plt.plot(subset['epoch'], subset[metric], marker='o', label=run_name)\n"
            "    plt.title(metric)\n"
            "    plt.xlabel('epoch')\n"
            "    plt.grid(True, alpha=0.3)\n"
            "    plt.legend()\n"
            "    plt.tight_layout()\n"
            "    plt.show()"
        ),
        _markdown_cell("## Final Remark"),
        _code_cell(
            "best_run = comparison_df.iloc[0]\n"
            "fastest_run = comparison_df.sort_values(by='time_s', ascending=True).iloc[0]\n"
            "print(f\"Best test accuracy: {best_run['model']} ({best_run['test_acc']:.2%})\")\n"
            "print(f\"Fastest run: {fastest_run['model']} ({fastest_run['time_s']:.2f}s)\")"
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


def execute_notebook(notebook_path, timeout=1800):
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
    """Run the requested Part 3 models and create a comparison notebook."""
    args = build_parser().parse_args()
    validate_models(args.models)
    comparison_timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    comparison_root = (
        CURRENT_DIR
        / "outputs"
        / "Part3"
        / f"external_model_comparison_{comparison_timestamp}"
    )

    print(f"Saving Part3 comparison runs under: {comparison_root}")

    summaries = []
    for model_name in args.models:
        output_dir = comparison_root / model_name
        print(f"\n=== {model_name} ===")
        print(f"Output directory: {output_dir}")
        run_args = Namespace(
            model=model_name,
            dataset_root=args.dataset_root,
            batch_size=args.batch_size,
            epochs_head=args.epochs_head,
            epochs_finetune=args.epochs_finetune,
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

    print("\nFinished Part3 comparison.")
    for model_name, summary_path, summary in summaries:
        print(
            f"{model_name}: "
            f"best_epoch={summary['best_epoch']} | "
            f"best_val_acc={summary['best_validation_accuracy']:.2%} | "
            f"test_acc={summary['final_test_accuracy']:.2%} | "
            f"macro_f1={summary['final_test_macro_f1']:.4f} | "
            f"time={summary['total_training_time_seconds']:.2f}s | "
            f"summary={summary_path}"
        )

    notebook_path = create_comparison_notebook(comparison_root, summaries)
    execute_notebook(notebook_path)
    print(f"\nSaved executed comparison notebook to: {notebook_path}")


if __name__ == "__main__":
    main()
