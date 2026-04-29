# Assignment 1 AI1 in classic ML

PyTorch experiments for the assignment. The repository is split into three parts:

- `part_1`: Introductory NumPy/PyTorch exercises and an MNIST training script.
- `part_2`: MNIST CNN experiments, regularization, augmentation, tuning, checkpoints, plots, SQLite experiment logging, and generated notebook reports.
- `part_3`: Oxford-IIIT Pet cat-vs-dog experiments with CNNs and transfer-learning models.

## Repository Layout

```text
part_1/
  A_B/                         Basic NumPy/PyTorch scripts
  C_D/                         MNIST training script
part_2/
  main.py                      Main MNIST CNN training entry point
  compare_augmentation.py      Augmentation comparison
  compare_regularization.py    Regularization comparison
  cnn_comparison.py            CNN architecture comparison
  hyperparameter_tuning.py     Hyperparameter testing
  notebook_*.py                Notebook report generation helpers
part_3/
  part3_finetuning_external_models.py
  improve_scractch_cnn.py
  compare_external_models.py
  notebook_*.py                Part 3 notebook report helpers
docs/
  torch_gpu_install.md         GPU install notes
  tested_parameters.md         Tested Part 2 settings
```

## Setup

Create and activate a virtual environment from the repository root.

PowerShell:

```powershell
cd C:\Users\emil_\vscode\Assignment1
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio matplotlib numpy pandas nbformat nbclient ipython
```

Git Bash:

```bash
cd /c/Users/emil_/vscode/Assignment1
python -m venv venv
source venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio matplotlib numpy pandas nbformat nbclient ipython
```

For CUDA-specific PyTorch installation notes, see `docs/torch_gpu_install.md`.

## Running Part 1

PowerShell:

```powershell
.\venv\Scripts\python.exe part_1\A_B\A1_class.py
.\venv\Scripts\python.exe part_1\A_B\A1_class_numpy.py
.\venv\Scripts\python.exe part_1\A_B\B.py
.\venv\Scripts\python.exe part_1\C_D\C.py
```

Git Bash:

```bash
./venv/Scripts/python.exe part_1/A_B/A1_class.py
./venv/Scripts/python.exe part_1/A_B/A1_class_numpy.py
./venv/Scripts/python.exe part_1/A_B/B.py
./venv/Scripts/python.exe part_1/C_D/C.py
```

## Running Part 2

Run one MNIST CNN experiment:

```bash
./venv/Scripts/python.exe part_2/main.py --model cnn_medium --epochs 5
```

Useful comparison scripts:

```bash
./venv/Scripts/python.exe part_2/compare_augmentation.py --epochs 5
./venv/Scripts/python.exe part_2/compare_regularization.py --epochs 5
./venv/Scripts/python.exe part_2/cnn_comparison.py --epochs 5
./venv/Scripts/python.exe part_2/hyperparameter_tuning.py --epochs 7
```

Part 2 writes artifacts under `part_2/outputs/`, including checkpoints, plots, `summary.json`, `training_history.json`, SQLite experiment logs, and generated `.ipynb` reports.

Inspect saved experiments:

```bash
./venv/Scripts/python.exe part_2/list_experiments.py
./venv/Scripts/python.exe part_2/inspect_checkpoint.py --run-dir part_2/outputs/<run-folder>
```

## Running Part 3

Run a single Oxford-IIIT Pet model:

```bash
./venv/Scripts/python.exe part_3/part3_finetuning_external_models.py --model resnet18_transfer
```

Available models:

```text
scratch_cnn
deeper_cnn
resnet18_transfer
resnet50_transfer
mobilenet_v3_transfer
```

Run improved scratch CNN variants:

```bash
./venv/Scripts/python.exe part_3/improve_scractch_cnn.py --variants baseline_v2 deeper_cnn residual_cnn --epochs 25
```

Run the external model comparison. This script runs the selected models one after another and then creates a comparison notebook:

```bash
./venv/Scripts/python.exe part_3/compare_external_models.py
```

Quick Part 3 test run:

```bash
./venv/Scripts/python.exe part_3/compare_external_models.py \
  --models resnet18_transfer mobilenet_v3_transfer \
  --epochs-head 1 \
  --epochs-finetune 1 \
  --test-ratio 0.1
```

Part 3 writes artifacts under `part_3/outputs/`, including model checkpoints, plots, JSON summaries, and executed notebook reports.

## Output Files

Most runs save:

- `config.json`: exact command/configuration used for the run.
- `summary.json`: final metrics and timing.
- `training_history.json`: per-epoch metrics.
- `best_model.pt`: best checkpoint.
- `checkpoint_epoch_*.pt`: periodic checkpoints.
- `loss_curve.png`, `accuracy_curve.png`, `confusion_matrix.png`: visual diagnostics.
- `report.ipynb` or `comparison_report.ipynb`: executed notebook report.

The `data/`, `outputs/`, `part_2/outputs/`, and `part_3/outputs/` folders are generated artifacts and are ignored by git.

## Notes

- Run commands from the repository root so imports resolve correctly.
- Dataset downloads are handled by `torchvision` and may take time on first run.
- Notebook reports require `nbformat`, `nbclient`, `ipython`, `pandas`, and a working Python kernel.
- GPU usage is automatic when PyTorch detects CUDA. Use `part_2/torch_gpu.py` to check the active device:

```bash
./venv/Scripts/python.exe part_2/torch_gpu.py
```
## Summary and conclusions

Part 1
Was implemented and tested in its differents forms. Not much to say

Part 2
The most comprehensisive task. Here a notebook creater is added for easier testing and follow up.
The python scripts can be run with specific settings:
e.g. ./venv/Scripts/python.exe part_2/cnn_comparison.py --epochs 5

Part 3
Here transferred learning was tested with some different models.
I also tried with an own CNN but it was not even close to matching the pretrained models