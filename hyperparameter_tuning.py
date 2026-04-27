#!/usr/bin/env python3
"""Run a small hyperparameter tuning sweep for MNIST CNN experiments."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from main import CURRENT_DIR, run_experiment


def build_parser():
    """Create CLI arguments for tuning runs."""
    parser = argparse.ArgumentParser(
        description="Train a list of hyperparameter settings and compare the results."
    )
    parser.add_argument("--config-path", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=7)
    parser.add_argument("--checkpoint-interval", type=int, default=5)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    return parser


def build_default_search_space():
    """Return a compact list of hand-picked hyperparameter combinations."""
    return [
        {
            "run_name": "tune_01_baseline",
            "description": "Baseline 2-conv CNN with default learning rate and no extra regularization.",
            "model_name": "cnn_medium",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "l1_lambda": 0.0,
            "input_noise_std": 0.0,
            "adam_betas": (0.9, 0.999),
            "adam_eps": 1e-8,
            "model_overrides": None,
            "augmentation_config": {"enabled": False, "rotation_degrees": 0.0, "translate": (0.0, 0.0), "scale": (1.0, 1.0)},
        },
        {
            "run_name": "tune_02_dropout",
            "description": "Tests whether dropout alone improves generalization relative to the baseline.",
            "model_name": "cnn_dropout",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "l1_lambda": 0.0,
            "input_noise_std": 0.0,
            "adam_betas": (0.9, 0.999),
            "adam_eps": 1e-8,
            "model_overrides": None,
            "augmentation_config": {"enabled": False, "rotation_degrees": 0.0, "translate": (0.0, 0.0), "scale": (1.0, 1.0)},
        },
        {
            "run_name": "tune_03_weight_decay",
            "description": "Tests L2-style regularization through Adam weight decay on the baseline architecture.",
            "model_name": "cnn_medium",
            "learning_rate": 0.001,
            "weight_decay": 1e-4,
            "l1_lambda": 0.0,
            "input_noise_std": 0.0,
            "adam_betas": (0.9, 0.999),
            "adam_eps": 1e-8,
            "model_overrides": None,
            "augmentation_config": {"enabled": False, "rotation_degrees": 0.0, "translate": (0.0, 0.0), "scale": (1.0, 1.0)},
        },
        {
            "run_name": "tune_04_batchnorm",
            "description": "Tests whether batch normalization helps optimization and validation performance.",
            "model_name": "cnn_batchnorm",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "l1_lambda": 0.0,
            "input_noise_std": 0.0,
            "adam_betas": (0.9, 0.999),
            "adam_eps": 1e-8,
            "model_overrides": None,
            "augmentation_config": {"enabled": False, "rotation_degrees": 0.0, "translate": (0.0, 0.0), "scale": (1.0, 1.0)},
        },
        {
            "run_name": "tune_05_noise",
            "description": "Adds Gaussian input noise to test whether noise injection stabilizes training.",
            "model_name": "cnn_batchnorm",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "l1_lambda": 0.0,
            "input_noise_std": 0.05,
            "adam_betas": (0.9, 0.999),
            "adam_eps": 1e-8,
            "model_overrides": None,
            "augmentation_config": {"enabled": False, "rotation_degrees": 0.0, "translate": (0.0, 0.0), "scale": (1.0, 1.0)},
        },
        {
            "run_name": "tune_06_deep_balanced",
            "description": "Tests a deeper 3-conv CNN while keeping capacity in a moderate range.",
            "model_name": "cnn_deep_balanced",
            "learning_rate": 0.001,
            "weight_decay": 1e-4,
            "l1_lambda": 0.0,
            "input_noise_std": 0.0,
            "adam_betas": (0.9, 0.999),
            "adam_eps": 1e-8,
            "model_overrides": None,
            "augmentation_config": {"enabled": False, "rotation_degrees": 0.0, "translate": (0.0, 0.0), "scale": (1.0, 1.0)},
        },
        {
            "run_name": "tune_07_augmented",
            "description": "Tests whether data augmentation improves a strong batch-normalized 3-conv model.",
            "model_name": "cnn_batchnorm",
            "learning_rate": 0.001,
            "weight_decay": 1e-4,
            "l1_lambda": 0.0,
            "input_noise_std": 0.0,
            "adam_betas": (0.9, 0.999),
            "adam_eps": 1e-8,
            "model_overrides": None,
            "augmentation_config": {"enabled": True, "rotation_degrees": 10.0, "translate": (0.1, 0.1), "scale": (0.95, 1.05)},
        },
        {
            "run_name": "tune_08_low_lr",
            "description": "Tests whether a lower learning rate helps the regularized deep model converge better.",
            "model_name": "cnn_regularized",
            "learning_rate": 0.0005,
            "weight_decay": 1e-4,
            "l1_lambda": 0.0,
            "input_noise_std": 0.05,
            "adam_betas": (0.9, 0.999),
            "adam_eps": 1e-8,
            "model_overrides": None,
            "augmentation_config": {"enabled": False, "rotation_degrees": 0.0, "translate": (0.0, 0.0), "scale": (1.0, 1.0)},
        },
        {
            "run_name": "tune_09_custom_kernel",
            "description": "Tests a custom 5x5 kernel setup with narrower channels and batch normalization.",
            "model_name": "cnn_medium",
            "learning_rate": 0.001,
            "weight_decay": 1e-4,
            "l1_lambda": 0.0,
            "input_noise_std": 0.0,
            "adam_betas": (0.9, 0.999),
            "adam_eps": 1e-8,
            "model_overrides": {
                "model_name": "cnn_medium_custom_kernel",
                "conv_channels": [24, 48],
                "kernel_size": 5,
                "classifier_hidden_size": 128,
                "activation": "ReLU",
                "dropout": 0.1,
                "batch_norm": True,
            },
            "augmentation_config": {"enabled": False, "rotation_degrees": 0.0, "translate": (0.0, 0.0), "scale": (1.0, 1.0)},
        },
        {
            "run_name": "tune_10_deep_augmented",
            "description": "Combines deeper architecture, stronger regularization, and augmentation in one run.",
            "model_name": "cnn_regularized",
            "learning_rate": 0.0007,
            "weight_decay": 2e-4,
            "l1_lambda": 0.0,
            "input_noise_std": 0.03,
            "adam_betas": (0.9, 0.999),
            "adam_eps": 1e-8,
            "model_overrides": None,
            "augmentation_config": {"enabled": True, "rotation_degrees": 12.0, "translate": (0.1, 0.1), "scale": (0.9, 1.1)},
        },
        {
            "run_name": "tune_11_hidden_512",
            "description": "Tests whether a much larger classifier hidden layer improves accuracy enough to justify the extra parameters.",
            "model_name": "cnn_medium",
            "learning_rate": 0.001,
            "weight_decay": 1e-4,
            "l1_lambda": 0.0,
            "input_noise_std": 0.0,
            "adam_betas": (0.9, 0.999),
            "adam_eps": 1e-8,
            "model_overrides": {
                "model_name": "cnn_medium_hidden_512",
                "conv_channels": [32, 64],
                "classifier_hidden_size": 512,
                "activation": "LeakyReLU",
                "dropout": 0.0,
                "batch_norm": False,
            },
            "augmentation_config": {"enabled": False, "rotation_degrees": 0.0, "translate": (0.0, 0.0), "scale": (1.0, 1.0)},
        },
        {
            "run_name": "tune_12_hidden_64",
            "description": "Tests whether a smaller classifier hidden layer can reduce model size with limited accuracy loss.",
            "model_name": "cnn_medium",
            "learning_rate": 0.001,
            "weight_decay": 1e-4,
            "l1_lambda": 0.0,
            "input_noise_std": 0.0,
            "adam_betas": (0.9, 0.999),
            "adam_eps": 1e-8,
            "model_overrides": {
                "model_name": "cnn_medium_hidden_64",
                "conv_channels": [32, 64],
                "classifier_hidden_size": 64,
                "activation": "LeakyReLU",
                "dropout": 0.0,
                "batch_norm": False,
            },
            "augmentation_config": {"enabled": False, "rotation_degrees": 0.0, "translate": (0.0, 0.0), "scale": (1.0, 1.0)},
        },
        {
            "run_name": "tune_13_leaky_deep_bn",
            "description": "Tests LeakyReLU instead of ReLU in the deeper batch-normalized architecture.",
            "model_name": "cnn_batchnorm",
            "learning_rate": 0.001,
            "weight_decay": 1e-4,
            "l1_lambda": 0.0,
            "input_noise_std": 0.0,
            "adam_betas": (0.9, 0.999),
            "adam_eps": 1e-8,
            "model_overrides": {
                "model_name": "cnn_batchnorm_leaky",
                "conv_channels": [32, 64, 64],
                "classifier_hidden_size": 256,
                "activation": "LeakyReLU",
                "dropout": 0.1,
                "batch_norm": True,
            },
            "augmentation_config": {"enabled": False, "rotation_degrees": 0.0, "translate": (0.0, 0.0), "scale": (1.0, 1.0)},
        },
        {
            "run_name": "tune_14_wide_3conv",
            "description": "Tests a wider 3-conv architecture with moderate regularization and small input noise.",
            "model_name": "cnn_regularized",
            "learning_rate": 0.001,
            "weight_decay": 1e-4,
            "l1_lambda": 0.0,
            "input_noise_std": 0.02,
            "adam_betas": (0.9, 0.999),
            "adam_eps": 1e-8,
            "model_overrides": {
                "model_name": "cnn_wide_3conv",
                "conv_channels": [48, 96, 128],
                "classifier_hidden_size": 256,
                "activation": "ReLU",
                "dropout": 0.2,
                "batch_norm": True,
            },
            "augmentation_config": {"enabled": False, "rotation_degrees": 0.0, "translate": (0.0, 0.0), "scale": (1.0, 1.0)},
        },
        {
            "run_name": "tune_15_small_kernel_3conv",
            "description": "Tests a smaller 3-conv network with fewer channels but still using batch normalization and regularization.",
            "model_name": "cnn_regularized",
            "learning_rate": 0.001,
            "weight_decay": 1e-4,
            "l1_lambda": 1e-6,
            "input_noise_std": 0.02,
            "adam_betas": (0.9, 0.999),
            "adam_eps": 1e-8,
            "model_overrides": {
                "model_name": "cnn_small_kernel_3conv",
                "conv_channels": [24, 48, 64],
                "kernel_size": 3,
                "classifier_hidden_size": 192,
                "activation": "ReLU",
                "dropout": 0.15,
                "batch_norm": True,
            },
            "augmentation_config": {"enabled": False, "rotation_degrees": 0.0, "translate": (0.0, 0.0), "scale": (1.0, 1.0)},
        },
        {
            "run_name": "tune_16_adam_beta_variant",
            "description": "Tests whether changing Adam beta values improves training dynamics.",
            "model_name": "cnn_batchnorm",
            "learning_rate": 0.001,
            "weight_decay": 1e-4,
            "l1_lambda": 0.0,
            "input_noise_std": 0.0,
            "adam_betas": (0.85, 0.995),
            "adam_eps": 1e-8,
            "model_overrides": None,
            "augmentation_config": {"enabled": False, "rotation_degrees": 0.0, "translate": (0.0, 0.0), "scale": (1.0, 1.0)},
        },
        {
            "run_name": "tune_17_adam_eps_variant",
            "description": "Tests whether a larger Adam epsilon improves optimization stability.",
            "model_name": "cnn_batchnorm",
            "learning_rate": 0.001,
            "weight_decay": 1e-4,
            "l1_lambda": 0.0,
            "input_noise_std": 0.0,
            "adam_betas": (0.9, 0.999),
            "adam_eps": 1e-7,
            "model_overrides": None,
            "augmentation_config": {"enabled": False, "rotation_degrees": 0.0, "translate": (0.0, 0.0), "scale": (1.0, 1.0)},
        },
        {
            "run_name": "tune_18_l1_regularized",
            "description": "Tests explicit L1 regularization added to a batch-normalized 3-conv model.",
            "model_name": "cnn_batchnorm",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "l1_lambda": 1e-6,
            "input_noise_std": 0.0,
            "adam_betas": (0.9, 0.999),
            "adam_eps": 1e-8,
            "model_overrides": None,
            "augmentation_config": {"enabled": False, "rotation_degrees": 0.0, "translate": (0.0, 0.0), "scale": (1.0, 1.0)},
        },
    ]


def load_search_space(config_path):
    """Load a JSON list of tuning specifications or fall back to defaults."""
    if config_path is None:
        return build_default_search_space()
    return json.loads(Path(config_path).read_text(encoding="utf-8"))


def _markdown_cell(source):
    """Build a markdown notebook cell."""
    return nbformat.v4.new_markdown_cell(source=source)


def _code_cell(source):
    """Build a code notebook cell."""
    return nbformat.v4.new_code_cell(source=source)


def create_tuning_notebook(tuning_root, run_specs):
    """Create a notebook that summarizes the tuning sweep."""
    tuning_root = Path(tuning_root)
    notebook_path = tuning_root / "tuning_report.ipynb"
    run_dirs = {spec["run_name"]: str(tuning_root / spec["run_name"]) for spec in run_specs}
    ordered_run_names = [spec["run_name"] for spec in run_specs]

    notebook = nbformat.v4.new_notebook()
    notebook.cells = [
        _markdown_cell(
            "# Hyperparameter Tuning Report\n\n"
            "This notebook compares all tuning runs saved in this folder."
        ),
        _code_cell(
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
        _markdown_cell(
            "## What Each Test Tries To Measure\n\n"
            "This section explains the purpose of each tuning run before comparing the metrics."
        ),
        _code_cell(
            "descriptions = [\n"
            "    {\n"
            "        'run': spec['run_name'],\n"
            "        'description': spec.get('description', ''),\n"
            "    }\n"
            "    for spec in RUN_SPECS\n"
            "]\n"
            "pd.DataFrame(descriptions)"
        ),
        _markdown_cell("## Summary Table"),
        _code_cell(
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
        _markdown_cell("## Pandas Overview"),
        _code_cell(
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
        _markdown_cell(
            "## Tuning Takeaway\n\n"
            "Hyperparameter tuning means trying many plausible settings, then selecting "
            "the best trade-off between validation behavior, test accuracy, runtime, and "
            "model size."
        ),
        _code_cell(
            "best_by_test = max(rows, key=lambda row: row['test_acc'])\n"
            "best_by_val_loss = min(rows, key=lambda row: row['best_val_loss'])\n"
            "{'best_by_test': best_by_test, 'best_by_val_loss': best_by_val_loss}"
        ),
        _markdown_cell(
            "## Final Summary\n\n"
            "This section is generated from the saved metrics for the current tuning sweep."
        ),
        _code_cell(
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
            "    '',\n"
            "    'Kort svensk sammanfattning:',\n"
            "    (\n"
            "        f\"Bästa testkörning: {best_test['run']} ({best_test['test_acc']:.2%}) med modellen {best_test['model']}. \"\n"
            "        f\"Bäst validation accuracy: {best_val_acc['run']} ({best_val_acc['best_val_acc']:.2%}). \"\n"
            "        f\"Bäst validation loss: {best_val_loss['run']} ({best_val_loss['best_val_loss']:.4f}).\"\n"
            "    ),\n"
            "    (\n"
            "        f\"Snabbaste körning: {fastest['run']} ({fastest['time_s']:.2f}s). \"\n"
            "        f\"Minsta modellen: {smallest['run']} ({smallest['params']:,} träningsbara parametrar).\"\n"
            "    ),\n"
            "]\n"
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
    """Run the full tuning sweep."""
    args = build_parser().parse_args()
    run_specs = load_search_space(args.config_path)
    tuning_timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    tuning_root = CURRENT_DIR / "outputs" / "Part2" / f"hyperparameter_tuning_{tuning_timestamp}"
    print(f"Saving tuning runs under: {tuning_root}")

    for spec in run_specs:
        output_dir = tuning_root / spec["run_name"]
        print(f"\n=== {spec['run_name']} ===")
        run_experiment(
            model_name=spec["model_name"],
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=spec["learning_rate"],
            output_dir=output_dir,
            checkpoint_interval=args.checkpoint_interval,
            validation_ratio=args.validation_ratio,
            seed=args.seed,
            early_stopping_patience=args.early_stopping_patience,
            augmentation_config=spec.get("augmentation_config"),
            weight_decay=spec.get("weight_decay", 0.0),
            l1_lambda=spec.get("l1_lambda", 0.0),
            input_noise_std=spec.get("input_noise_std", 0.0),
            model_overrides=spec.get("model_overrides"),
            adam_betas=tuple(spec.get("adam_betas", (0.9, 0.999))),
            adam_eps=spec.get("adam_eps", 1e-8),
        )

    notebook_path = create_tuning_notebook(tuning_root, run_specs)
    execute_notebook(notebook_path)
    print(f"\nSaved executed tuning notebook to: {notebook_path}")


if __name__ == "__main__":
    main()
