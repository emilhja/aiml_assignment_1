"""Notebook builders for Part 3 experiment reports."""

from pathlib import Path

try:
    from .notebook_utils import code_cell, markdown_cell, write_notebook
except ImportError:  # pragma: no cover - supports direct script execution
    from notebook_utils import code_cell, markdown_cell, write_notebook


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def create_experiment_report_notebook(run_dir):
    """Create an editable notebook report for a Part 3 run."""
    run_dir = Path(run_dir)
    notebook_path = run_dir / "report.ipynb"

    return write_notebook(
        notebook_path,
        [
            markdown_cell(
                "# Part 3 Transfer Learning Report\n\n"
                f"Run folder: `{run_dir.name}`\n\n"
                "This notebook loads the saved run artifacts, rebuilds pandas tables from the "
                "JSON files, and includes cells you can adjust and rerun."
            ),
            code_cell(
                "import json\n"
                "import sys\n"
                "from pathlib import Path\n"
                "\n"
                "import matplotlib.pyplot as plt\n"
                "import pandas as pd\n"
                "import torch\n"
                "from IPython.display import Image, display\n"
                "\n"
                f'PROJECT_ROOT = Path(r"{PROJECT_ROOT}")\n'
                f'RUN_DIR = Path(r"{run_dir}")\n'
                "CONFIG_PATH = RUN_DIR / 'config.json'\n"
                "SUMMARY_PATH = RUN_DIR / 'summary.json'\n"
                "HISTORY_PATH = RUN_DIR / 'training_history.json'\n"
                "\n"
                "if str(PROJECT_ROOT) not in sys.path:\n"
                "    sys.path.insert(0, str(PROJECT_ROOT))\n"
                "\n"
                "from Part3.part3_finetuning_external_models import build_data_loaders, denormalize_image\n"
                "\n"
                "def load_json(path):\n"
                "    return json.loads(Path(path).read_text(encoding='utf-8'))\n"
                "\n"
                "config = load_json(CONFIG_PATH)\n"
                "summary = load_json(SUMMARY_PATH)\n"
                "history = load_json(HISTORY_PATH)\n"
                "print(f'Loaded run: {RUN_DIR.name}')"
            ),
            markdown_cell("## Config And Summary"),
            code_cell(
                "config_df = pd.DataFrame([\n"
                "    {'field': key, 'value': value}\n"
                "    for key, value in config.items()\n"
                "])\n"
                "summary_df = pd.DataFrame([\n"
                "    {'metric': key, 'value': value}\n"
                "    for key, value in summary.items()\n"
                "])\n"
                "config_df"
            ),
            code_cell("summary_df"),
            markdown_cell("## Epoch History"),
            code_cell(
                "history_df = pd.DataFrame(history)\n"
                "history_df.index = history_df.index + 1\n"
                "history_df.index.name = 'epoch'\n"
                "history_df"
            ),
            markdown_cell("## Metric Curves From JSON"),
            code_cell(
                "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
                "history_df[['train_loss', 'val_loss']].plot(ax=axes[0], marker='o', title='Loss')\n"
                "history_df[['train_accuracy', 'val_accuracy']].plot(ax=axes[1], marker='o', title='Accuracy')\n"
                "axes[0].grid(True, alpha=0.3)\n"
                "axes[1].grid(True, alpha=0.3)\n"
                "plt.tight_layout()\n"
                "plt.show()"
            ),
            markdown_cell("## Rebuild The Dataset"),
            code_cell(
                "data_bundle = build_data_loaders(\n"
                "    dataset_root=Path(config['dataset_root']),\n"
                "    batch_size=config['batch_size'],\n"
                "    validation_ratio=config['validation_ratio'],\n"
                "    test_ratio=config['test_ratio'],\n"
                "    seed=config['seed'],\n"
                "    num_workers=config.get('num_workers', 0),\n"
                ")\n"
                "pd.DataFrame([\n"
                "    {'split': 'train', 'samples': data_bundle['train_size']},\n"
                "    {'split': 'validation', 'samples': data_bundle['val_size']},\n"
                "    {'split': 'test', 'samples': data_bundle['test_size']},\n"
                "])"
            ),
            markdown_cell("## Preview A Training Batch"),
            code_cell(
                "images, labels = next(iter(data_bundle['train_loader']))\n"
                "fig, axes = plt.subplots(2, 4, figsize=(12, 6))\n"
                "axes = axes.flatten()\n"
                "for axis in axes:\n"
                "    axis.axis('off')\n"
                "for index in range(min(8, len(images))):\n"
                "    axes[index].imshow(denormalize_image(images[index]))\n"
                "    axes[index].set_title(data_bundle['class_names'][int(labels[index])])\n"
                "plt.tight_layout()\n"
                "plt.show()"
            ),
            markdown_cell("## Saved Artifacts"),
            code_cell(
                "plot_files = [\n"
                "    'loss_curve.png',\n"
                "    'accuracy_curve.png',\n"
                "    'confusion_matrix.png',\n"
                "    'correct_predictions.png',\n"
                "    'incorrect_predictions.png',\n"
                "]\n"
                "for plot_name in plot_files:\n"
                "    plot_path = RUN_DIR / plot_name\n"
                "    print(f'\\n### {plot_name}')\n"
                "    if plot_path.exists():\n"
                "        display(Image(filename=str(plot_path)))\n"
                "    else:\n"
                "        print('Missing:', plot_path)"
            ),
            markdown_cell("## Re-Run This Experiment"),
            code_cell(
                "from argparse import Namespace\n"
                "from Part3.part3_finetuning_external_models import run_experiment\n"
                "\n"
                "# Uncomment to rerun into a new folder.\n"
                "# rerun_args = Namespace(\n"
                "#     model=config['model_name'],\n"
                "#     dataset_root=config['dataset_root'],\n"
                "#     batch_size=config['batch_size'],\n"
                "#     epochs_head=config['epochs_head'],\n"
                "#     epochs_finetune=config['epochs_finetune'],\n"
                "#     learning_rate_head=config['learning_rate_head'],\n"
                "#     learning_rate_finetune=config['learning_rate_finetune'],\n"
                "#     validation_ratio=config['validation_ratio'],\n"
                "#     test_ratio=config['test_ratio'],\n"
                "#     seed=config['seed'],\n"
                "#     output_dir=str(RUN_DIR.parent / f'{RUN_DIR.name}_rerun'),\n"
                "#     checkpoint_interval=config['checkpoint_interval'],\n"
                "#     num_workers=config.get('num_workers', 0),\n"
                "# )\n"
                "# run_experiment(rerun_args)"
            ),
        ],
    )


def create_external_model_comparison_notebook(comparison_root, summaries):
    """Create a runnable notebook that compares multiple Part 3 runs."""
    comparison_root = Path(comparison_root)
    notebook_path = comparison_root / "comparison_report.ipynb"
    run_paths = {
        run_name: str(summary_path.parent)
        for run_name, summary_path, _summary in summaries
    }
    ordered_run_names = [run_name for run_name, _summary_path, _summary in summaries]

    return write_notebook(
        notebook_path,
        [
            markdown_cell(
                "# Part 3 External Model Comparison\n\n"
                "This notebook compares the scratch CNN baseline against the transfer-learning runs."
            ),
            code_cell(
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
            markdown_cell("## Summary Table"),
            code_cell(
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
                "        'test_eval_time_s': summary.get('test_evaluation_time_seconds'),\n"
                "        'test_total_time_s': summary.get('final_test_total_time_seconds'),\n"
                "        'test_images_per_s': summary.get('test_evaluation_images_per_second'),\n"
                "        'test_ms_per_image': summary.get('test_evaluation_time_per_image_ms'),\n"
                "        'trainable_parameters': summary['trainable_parameters'],\n"
                "        'total_parameters': summary['total_parameters'],\n"
                "    })\n"
                "comparison_df = pd.DataFrame(rows).sort_values(by='test_acc', ascending=False).reset_index(drop=True)\n"
                "comparison_df"
            ),
            markdown_cell("## Accuracy Vs Runtime"),
            code_cell(
                "ax = comparison_df.plot.scatter(x='time_s', y='test_acc', s=120, figsize=(7, 5))\n"
                "for _, row in comparison_df.iterrows():\n"
                "    ax.annotate(row['model'], (row['time_s'], row['test_acc']))\n"
                "ax.grid(True, alpha=0.3)\n"
                "ax.set_title('Accuracy vs Runtime')\n"
                "plt.show()"
            ),
            markdown_cell("## Parameter-Aware View"),
            code_cell(
                "comparison_df[['model', 'test_acc', 'macro_f1', 'time_s', 'test_eval_time_s', 'test_total_time_s', 'test_images_per_s', 'test_ms_per_image', 'trainable_parameters', 'total_parameters']]"
            ),
            markdown_cell("## Saved Plots"),
            code_cell(
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
            markdown_cell("## Per-Run Epoch Histories"),
            code_cell(
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
            code_cell(
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
            markdown_cell("## Final Remark"),
            code_cell(
                "best_run = comparison_df.iloc[0]\n"
                "fastest_run = comparison_df.sort_values(by='time_s', ascending=True).iloc[0]\n"
                "timed_test_df = comparison_df.dropna(subset=['test_eval_time_s'])\n"
                "print(f\"Best test accuracy: {best_run['model']} ({best_run['test_acc']:.2%})\")\n"
                "print(f\"Fastest training run: {fastest_run['model']} ({fastest_run['time_s']:.2f}s)\")\n"
                "if not timed_test_df.empty:\n"
                "    fastest_test_run = timed_test_df.sort_values(by='test_eval_time_s', ascending=True).iloc[0]\n"
                "    print(f\"Fastest test evaluation: {fastest_test_run['model']} ({fastest_test_run['test_eval_time_s']:.2f}s)\")\n"
                "else:\n"
                "    print('Fastest test evaluation: unavailable in these summaries')"
            ),
        ],
    )


create_comparison_notebook = create_external_model_comparison_notebook
