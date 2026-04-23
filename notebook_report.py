#!/usr/bin/env python3
"""Generate and execute per-run Jupyter notebook reports."""

import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def _markdown_cell(source):
    """Build a markdown notebook cell."""
    return nbformat.v4.new_markdown_cell(source=source)


def _code_cell(source):
    """Build a code notebook cell."""
    return nbformat.v4.new_code_cell(source=source)


def create_report_notebook(run_dir):
    """Create a runnable notebook report for an experiment run."""
    run_dir = Path(run_dir)
    project_root = Path(__file__).resolve().parent
    config_path = run_dir / "config.json"
    summary_path = run_dir / "summary.json"
    history_path = run_dir / "training_history.json"
    notebook_path = run_dir / "report.ipynb"

    config = json.loads(config_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    notebook = nbformat.v4.new_notebook()
    notebook.cells = [
        _markdown_cell(
            f"# Experiment Report\n\n"
            f"Run folder: `{run_dir.name}`\n\n"
            f"This notebook reproduces the configuration and inspects the "
            f"saved artifacts for this experiment."
        ),
        _code_cell(
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
        _markdown_cell("## Training Config"),
        _code_cell("config"),
        _markdown_cell("## Final Summary"),
        _code_cell("summary"),
        _markdown_cell("## Re-run This Experiment"),
        _code_cell(
            "from main import run_experiment\n"
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
        _markdown_cell("## Saved Plots"),
        _code_cell(
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
        _markdown_cell("## Training History"),
        _code_cell("history"),
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


def execute_report_notebook(notebook_path, timeout=600):
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
