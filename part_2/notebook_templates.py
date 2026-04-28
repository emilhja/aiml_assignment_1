"""Notebook builders for Part 2 experiment reports."""

from pathlib import Path

try:
    from .notebook_utils import (
        code_cell,
        load_json,
        markdown_cell,
        summarize_run_paths,
        write_notebook,
    )
except ImportError:
    if __package__:
        raise
    from notebook_utils import (
        code_cell,
        load_json,
        markdown_cell,
        summarize_run_paths,
        write_notebook,
    )


def create_experiment_report_notebook(run_dir):
    """Create a runnable notebook report for one experiment run."""
    run_dir = Path(run_dir)
    project_root = Path(__file__).resolve().parent.parent
    config_path = run_dir / "config.json"
    summary_path = run_dir / "summary.json"
    notebook_path = run_dir / "report.ipynb"

    load_json(config_path)
    load_json(summary_path)

    return write_notebook(
        notebook_path,
        [
            markdown_cell(
                f"# Experiment Report\n\n"
                f"Run folder: `{run_dir.name}`\n\n"
                f"This notebook reproduces the configuration and inspects the "
                f"saved artifacts for this experiment."
            ),
            code_cell(
                "import sys\n"
                "from pathlib import Path\n"
                "import json\n"
                "import matplotlib.pyplot as plt\n"
                "from IPython.display import Image, display\n"
                "\n"
                f'PROJECT_ROOT = Path(r"{project_root}")\n'
                f'RUN_DIR = Path(r"{run_dir}")\n'
                f'CONFIG_PATH = RUN_DIR / "config.json"\n'
                f'SUMMARY_PATH = RUN_DIR / "summary.json"\n'
                f'HISTORY_PATH = RUN_DIR / "training_history.json"\n'
                "\n"
                "if str(PROJECT_ROOT) not in sys.path:\n"
                "    sys.path.insert(0, str(PROJECT_ROOT))\n"
                "\n"
                'config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))\n'
                'summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))\n'
                'history = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))\n'
                'print(f"Loaded run: {RUN_DIR.name}")'
            ),
            markdown_cell("## Training Config"),
            code_cell("config"),
            markdown_cell("## Final Summary"),
            code_cell("summary"),
            markdown_cell("## Re-run This Experiment"),
            code_cell(
                "from part_2.main import run_experiment\n"
                "\n"
                "# Uncomment to re-run this exact experiment into a separate folder.\n"
                "# rerun_output_dir = RUN_DIR.parent / f\"{RUN_DIR.name}_rerun\"\n"
                "# run_experiment(\n"
                "#     batch_size=config['batch_size'],\n"
                "#     epochs=config['epochs'],\n"
                "#     learning_rate=config['learning_rate'],\n"
                "#     output_dir=rerun_output_dir,\n"
                "#     checkpoint_interval=config['checkpoint_interval'],\n"
                "#     validation_ratio=config['validation_ratio'],\n"
                "#     seed=config['seed'],\n"
                "#     early_stopping_patience=config['early_stopping_patience'],\n"
                "#     augmentation_config=config.get('augmentation_config'),\n"
                "# )"
            ),
            markdown_cell("## Saved Plots"),
            code_cell(
                "plot_files = [\n"
                "    'loss_curve.png',\n"
                "    'accuracy_curve.png',\n"
                "    'correct_predictions.png',\n"
                "    'incorrect_predictions.png',\n"
                "    'confusion_matrix.png',\n"
                "    'conv_filters_first.png',\n"
                "    'conv_filters_last.png',\n"
                "]\n"
                "for plot_name in plot_files:\n"
                "    plot_path = RUN_DIR / plot_name\n"
                "    print(f'\\n### {plot_name}')\n"
                "    if plot_path.exists():\n"
                "        display(Image(filename=str(plot_path)))\n"
                "    else:\n"
                "        print('Missing:', plot_path)"
            ),
            markdown_cell("## Training History"),
            code_cell("history"),
        ],
    )


def create_augmentation_comparison_notebook(comparison_root, summaries):
    """Create a notebook that compares augmented and non-augmented runs."""
    comparison_root = Path(comparison_root)
    notebook_path = comparison_root / "comparison_report.ipynb"
    run_paths = {
        run_name: str(summary_path.parent)
        for run_name, summary_path, _summary in summaries
    }

    return write_notebook(
        notebook_path,
        [
            markdown_cell(
                "# Augmentation Comparison Report\n\n"
                "This notebook compares the `without_augmentation` and "
                "`with_augmentation` runs saved in this folder."
            ),
            code_cell(
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
            markdown_cell("## Summary Comparison"),
            code_cell(
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
            markdown_cell("## Augmentation Settings"),
            code_cell(
                "{\n"
                "    'without_augmentation': without_config.get('augmentation_config'),\n"
                "    'with_augmentation': with_config.get('augmentation_config'),\n"
                "}"
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
                "\n"
                "for plot_name in plot_files:\n"
                "    print(f'\\n### {plot_name} - without_augmentation')\n"
                "    display(Image(filename=str(WITHOUT_RUN_DIR / plot_name)))\n"
                "    print(f'### {plot_name} - with_augmentation')\n"
                "    display(Image(filename=str(WITH_RUN_DIR / plot_name)))\n"
            ),
            markdown_cell("## Training History"),
            code_cell(
                "{\n"
                "    'without_augmentation': without_history,\n"
                "    'with_augmentation': with_history,\n"
                "}"
            ),
            markdown_cell("## Final Remark"),
            code_cell(
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
        ],
    )


def create_regularization_comparison_notebook(comparison_root, summaries):
    """Create a notebook that compares regularization runs."""
    comparison_root = Path(comparison_root)
    notebook_path = comparison_root / "comparison_report.ipynb"
    run_paths, ordered_run_names = summarize_run_paths(summaries)

    return write_notebook(
        notebook_path,
        [
            markdown_cell(
                "# Regularization Comparison Report\n\n"
                "This notebook compares several regularization strategies on MNIST."
            ),
            code_cell(
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
            markdown_cell("## Summary Table"),
            code_cell(
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
            markdown_cell("## Pandas Overview"),
            code_cell(
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
            markdown_cell(
                "## Interpretation\n\n"
                "Regularization methods mainly help by reducing overfitting and making "
                "optimization more stable. Dropout randomly removes features during training, "
                "weight decay discourages very large weights, batch normalization stabilizes "
                "activations across mini-batches, and input noise injection makes the model "
                "less sensitive to small perturbations."
            ),
            code_cell(
                "best_run = max(rows, key=lambda row: row['test_acc'])\n"
                "fastest_run = min(rows, key=lambda row: row['time_s'])\n"
                "{'best_test_run': best_run, 'fastest_run': fastest_run}"
            ),
            markdown_cell("## Saved Plots"),
            code_cell(
                "plot_files = ['loss_curve.png', 'accuracy_curve.png', 'confusion_matrix.png']\n"
                "for run_name in RUN_NAMES:\n"
                "    run_dir = Path(RUN_DIRS[run_name])\n"
                "    print(f'\\n## {run_name}')\n"
                "    for plot_name in plot_files:\n"
                "        display(Image(filename=str(run_dir / plot_name)))\n"
            ),
            markdown_cell(
                "## Final Summary\n\n"
                "This section is generated from the saved metrics for the current comparison run."
            ),
            code_cell(
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
                "print('\\n'.join(lines))\n"
                "lines"
            ),
        ],
    )


def create_cnn_comparison_notebook(comparison_root, summaries):
    """Create a notebook that compares the selected CNN runs."""
    comparison_root = Path(comparison_root)
    notebook_path = comparison_root / "comparison_report.ipynb"
    run_paths, ordered_run_names = summarize_run_paths(summaries)

    return write_notebook(
        notebook_path,
        [
            markdown_cell(
                "# CNN Comparison Report\n\n"
                "This notebook compares several CNN variants trained on MNIST."
            ),
            code_cell(
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
            markdown_cell("## Summary Table"),
            code_cell(
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
            markdown_cell(
                "## Parameter-Aware Comparison\n\n"
                "When evaluating architecture changes, models with very different parameter "
                "counts should not be treated as a clean apples-to-apples comparison. "
                "Use the table below to compare depth changes while keeping parameter "
                "counts reasonably close."
            ),
            code_cell(
                "sorted_rows = sorted(rows, key=lambda row: row['trainable_parameters'] or 0)\n"
                "sorted_rows"
            ),
            markdown_cell("## Saved Plots"),
            code_cell(
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
            markdown_cell(
                "## First Vs Later Convolution Filters\n\n"
                "The first convolution layer usually learns simple local patterns such as "
                "edges, stroke directions, and small blobs. Later convolution layers usually "
                "combine those earlier responses into more structured digit parts, for example "
                "curves, corners, loops, and stroke combinations that are useful for whole-digit "
                "recognition."
            ),
            code_cell(
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
            markdown_cell("## Architecture Notes"),
            code_cell(
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
            markdown_cell("## Final Remark"),
            code_cell(
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
            markdown_cell("## Recommended Model"),
            code_cell(
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
        ],
    )


def create_tuning_notebook(tuning_root, run_specs):
    """Create a notebook that summarizes the tuning sweep."""
    tuning_root = Path(tuning_root)
    notebook_path = tuning_root / "tuning_report.ipynb"
    run_dirs = {spec["run_name"]: str(tuning_root / spec["run_name"]) for spec in run_specs}
    ordered_run_names = [spec["run_name"] for spec in run_specs]

    return write_notebook(
        notebook_path,
        [
            markdown_cell(
                "# Hyperparameter Tuning Report\n\n"
                "This notebook compares all tuning runs saved in this folder."
            ),
            code_cell(
                "import json\n"
                "from pathlib import Path\n"
                "import pandas as pd\n"
                "\n"
                f'ROOT_DIR = Path(r"{tuning_root}")\n'
                f"RUN_NAMES = {ordered_run_names!r}\n"
                f"RUN_DIRS = {run_dirs!r}\n"
                f"RUN_SPECS = {run_specs!r}\n"
                "\n"
                "def load_json(path):\n"
                "    return json.loads(Path(path).read_text(encoding='utf-8'))\n"
            ),
            markdown_cell(
                "## What Each Test Tries To Measure\n\n"
                "This section explains the purpose of each tuning run before comparing the metrics."
            ),
            code_cell(
                "descriptions = [\n"
                "    {\n"
                "        'run': spec['run_name'],\n"
                "        'description': spec.get('description', ''),\n"
                "    }\n"
                "    for spec in RUN_SPECS\n"
                "]\n"
                "pd.DataFrame(descriptions)"
            ),
            markdown_cell("## Summary Table"),
            code_cell(
                "rows = []\n"
                "for run_name in RUN_NAMES:\n"
                "    run_dir = Path(RUN_DIRS[run_name])\n"
                "    config = load_json(run_dir / 'config.json')\n"
                "    summary = load_json(run_dir / 'summary.json')\n"
                "    spec = next(spec for spec in RUN_SPECS if spec['run_name'] == run_name)\n"
                "    rows.append({\n"
                "        'run': run_name,\n"
                "        'description': spec.get('description', ''),\n"
                "        'model': config['model_name'],\n"
                "        'conv_channels': config.get('conv_channels'),\n"
                "        'activation': config.get('activation'),\n"
                "        'dropout': config.get('dropout', 0.0),\n"
                "        'batch_norm': config.get('batch_norm', False),\n"
                "        'weight_decay': config.get('weight_decay', 0.0),\n"
                "        'l1_lambda': config.get('l1_lambda', 0.0),\n"
                "        'input_noise_std': config.get('input_noise_std', 0.0),\n"
                "        'learning_rate': config.get('learning_rate'),\n"
                "        'adam_beta1': config.get('adam_beta1'),\n"
                "        'adam_beta2': config.get('adam_beta2'),\n"
                "        'adam_eps': config.get('adam_eps'),\n"
                "        'augmentation_enabled': config.get('augmentation_enabled'),\n"
                "        'test_acc': summary['final_test_accuracy'],\n"
                "        'best_val_acc': summary['best_validation_accuracy'],\n"
                "        'best_val_loss': summary['best_validation_loss'],\n"
                "        'time_s': summary['total_training_time_seconds'],\n"
                "        'params': config.get('trainable_parameters', 0),\n"
                "    })\n"
                "sorted(rows, key=lambda row: row['test_acc'], reverse=True)"
            ),
            markdown_cell("## Pandas Overview"),
            code_cell(
                "df = pd.DataFrame(rows)\n"
                "display_columns = [\n"
                "    'run',\n"
                "    'description',\n"
                "    'model',\n"
                "    'conv_channels',\n"
                "    'activation',\n"
                "    'dropout',\n"
                "    'batch_norm',\n"
                "    'weight_decay',\n"
                "    'l1_lambda',\n"
                "    'input_noise_std',\n"
                "    'learning_rate',\n"
                "    'adam_beta1',\n"
                "    'adam_beta2',\n"
                "    'adam_eps',\n"
                "    'augmentation_enabled',\n"
                "    'params',\n"
                "    'best_val_loss',\n"
                "    'best_val_acc',\n"
                "    'test_acc',\n"
                "    'time_s',\n"
                "]\n"
                "df[display_columns].sort_values(by='test_acc', ascending=False).reset_index(drop=True)"
            ),
            markdown_cell(
                "## Tuning Takeaway\n\n"
                "Hyperparameter tuning means trying many plausible settings, then selecting "
                "the best trade-off between validation behavior, test accuracy, runtime, and "
                "model size."
            ),
            code_cell(
                "best_by_test = max(rows, key=lambda row: row['test_acc'])\n"
                "best_by_val_loss = min(rows, key=lambda row: row['best_val_loss'])\n"
                "{'best_by_test': best_by_test, 'best_by_val_loss': best_by_val_loss}"
            ),
            markdown_cell(
                "## Final Summary\n\n"
                "This section is generated from the saved metrics for the current tuning sweep."
            ),
            code_cell(
                "best_test = max(rows, key=lambda row: row['test_acc'])\n"
                "best_val_acc = max(rows, key=lambda row: row['best_val_acc'])\n"
                "best_val_loss = min(rows, key=lambda row: row['best_val_loss'])\n"
                "fastest = min(rows, key=lambda row: row['time_s'])\n"
                "smallest = min((row for row in rows if row['params'] > 0), key=lambda row: row['params'])\n"
                "\n"
                "lines = [\n"
                "    'English summary:',\n"
                "    (\n"
                "        f\"Best test run: {best_test['run']} ({best_test['test_acc']:.2%}) using {best_test['model']}. \"\n"
                "        f\"Best validation accuracy: {best_val_acc['run']} ({best_val_acc['best_val_acc']:.2%}). \"\n"
                "        f\"Best validation loss: {best_val_loss['run']} ({best_val_loss['best_val_loss']:.4f}).\"\n"
                "    ),\n"
                "    (\n"
                "        f\"Fastest run: {fastest['run']} ({fastest['time_s']:.2f}s). \"\n"
                "        f\"Smallest model: {smallest['run']} ({smallest['params']:,} trainable parameters).\"\n"
                "    ),\n"
                "]\n"
                "\n"
                "print('\\n'.join(lines))\n"
                "lines"
            ),
        ],
    )
