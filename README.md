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
  selected_outputs/            Curated executed notebooks for review
part_3/
  part3_finetuning_external_models.py
  improve_scractch_cnn.py
  compare_external_models.py
  notebook_*.py                Part 3 notebook report helpers
  selected_outputs/            Curated executed notebooks for review
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

Run the best Part 2 CNN architecture adapted to Oxford-IIIT Pet:

```bash
./venv/Scripts/python.exe part_3/part3_finetuning_external_models.py --model part2_cnn_deep_wide
```

Available models:

```text
part2_cnn_deep_wide
scratch_cnn
deeper_cnn
resnet18_transfer
resnet50_transfer
mobilenet_v3_transfer
```

`part2_cnn_deep_wide` is adapted from the best Part 2 CNN architecture, `cnn_deep_wide`: three convolution layers with channels `[32, 64, 128]`, a dense hidden layer of 256 units, and ReLU activations. In Part 3 it uses RGB input, two output classes, and adaptive pooling for the larger pet images.

Run improved scratch CNN variants:

```bash
./venv/Scripts/python.exe part_3/improve_scractch_cnn.py --variants baseline_v2 deeper_cnn residual_cnn --epochs 25
```

Run only the Part 2 CNN adaptation through the comparison script:

```bash
./venv/Scripts/python.exe part_3/compare_external_models.py --models part2_cnn_deep_wide
```

Run the full comparison set:

```bash
./venv/Scripts/python.exe part_3/compare_external_models.py \
  --models part2_cnn_deep_wide scratch_cnn deeper_cnn resnet18_transfer resnet50_transfer mobilenet_v3_transfer
```

Quick Part 3 test run:

```bash
./venv/Scripts/python.exe part_3/compare_external_models.py \
  --models part2_cnn_deep_wide \
  --epochs-head 1 \
  --epochs-finetune 1 \
  --test-ratio 0.1
```

Part 3 writes artifacts under `part_3/outputs/`, including model checkpoints, plots, JSON summaries, and executed notebook reports.

## Selected Outputs

The full `outputs/` folders are generated artifacts and are ignored by git. A smaller review set is included under `selected_outputs/`:

Part 2:

- `part_2/selected_outputs/augmentation_comparison_2026-04-28_154632.ipynb`
- `part_2/selected_outputs/cnn_comparison_2026-04-28_154826.ipynb`
- `part_2/selected_outputs/combined_regularization.ipynb`
- `part_2/selected_outputs/hyperparameter_tuning_2026-04-28_155935.ipynb`
- `part_2/selected_outputs/regularization_comparison_2026-04-28_155225.ipynb`

Part 3:

- `part_3/selected_outputs/external_model_comparison_2026-04-29_082032.ipynb`
- `part_3/selected_outputs/part2_cnn_deep_wide_2.ipynb`

These notebooks contain executed cells and saved plots/tables for the main comparisons used in the summaries.

## Output Files

Most runs save:

- `config.json`: exact command/configuration used for the run.
- `summary.json`: final metrics and timing.
- `training_history.json`: per-epoch metrics.
- `best_model.pt`: best checkpoint.
- `checkpoint_epoch_*.pt`: periodic checkpoints.
- `loss_curve.png`, `accuracy_curve.png`, `confusion_matrix.png`: visual diagnostics.
- `report.ipynb` or `comparison_report.ipynb`: executed notebook report.

The `data/`, `outputs/`, `part_2/outputs/`, and `part_3/outputs/` folders are generated artifacts and are ignored by git. The curated `part_2/selected_outputs/` and `part_3/selected_outputs/` notebooks are kept separately for assignment review.

## Notes

- Run commands from the repository root so imports resolve correctly.
- Dataset downloads are handled by `torchvision` and may take time on first run.
- Notebook reports require `nbformat`, `nbclient`, `ipython`, `pandas`, and a working Python kernel.
- GPU usage is automatic when PyTorch detects CUDA. Use `part_2/torch_gpu.py` to check the active device:

```bash
./venv/Scripts/python.exe part_2/torch_gpu.py
```
## Summary and conclusions

Part 1 implements and tests the introductory NumPy and PyTorch exercises, including the first MNIST training script.

Part 2 is the main MNIST CNN experiment section. It compares different CNN architectures, augmentation variants, regularization configurations, and hyperparameter configurations. The best architecture comparison result was `cnn_deep_wide` with 99.26% test accuracy. The best tuning result was `tune_07_augmented`, which reached 99.39% test accuracy with about 206K trainable parameters and less than 90 seconds of training time.

The Part 2 results suggest that deeper convolutional feature extraction helped more than simply widening the classifier, which is reasonable for a 10-class digit task. Augmentation gave a small gain in this setup, while regularization methods such as dropout, batch normalization, weight decay, L1 regularization, and combined configurations were tested to compare their effect on overfitting and generalization.

Part 3 moves from MNIST to Oxford-IIIT Pet binary cat-vs-dog classification. It compares scratch CNN models with transfer-learning models: MobileNetV3-Small, ResNet18, and ResNet50. The same idea as Part 2 is also tested with `part2_cnn_deep_wide`, adapted from grayscale MNIST input to RGB pet images and two output classes.

Transfer learning performed best on the smaller pet dataset. In the saved full comparison, the scratch CNN reached 77.08% test accuracy, the deeper CNN reached 94.00%, MobileNetV3 reached 97.36%, ResNet18 reached 98.99%, and ResNet50 reached 99.62%. This supports the conclusion that ImageNet-pretrained features give a strong starting point compared with training all weights from random initialization.
