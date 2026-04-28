# Tested Parameters and Model Settings

This document summarizes the main parameters and architectural choices that are currently implemented and have been tested to improve the model.

It is intended as a compact reference for the report and for rerunning experiments.

## 1. Model Families

The project currently includes the following model families in [main.py](C:\Users\emil_\vscode\Assignment1\part_2\main.py):

| Model name | Type | Main idea |
| --- | --- | --- |
| `mlp` | Fully connected network | Baseline non-CNN model |
| `cnn_small` | 2-layer CNN | Smaller convolutional baseline |
| `cnn_medium` | 2-layer CNN | Stronger 2-layer CNN baseline |
| `cnn_dropout` | 2-layer CNN | CNN with dropout regularization |
| `cnn_deep_balanced` | 3-layer CNN | Deeper CNN with balanced width |
| `cnn_deep_wide` | 3-layer CNN | Deeper and wider CNN |
| `cnn_batchnorm` | 3-layer CNN | Deeper CNN with batch normalization |
| `cnn_regularized` | 3-layer CNN | Deeper CNN with batch norm and dropout |

## 2. Architecture Parameters

These parameters define the model architecture itself.

| Parameter | Meaning | Implemented in | Tested values / examples |
| --- | --- | --- | --- |
| `conv_channels` | Number of convolution filters per conv layer | `part_2/main.py`, `part_2/hyperparameter_tuning.py` | `[16, 32]`, `[24, 48]`, `[24, 48, 64]`, `[32, 64]`, `[32, 64, 64]`, `[32, 64, 128]`, `[48, 96, 128]` |
| `num_conv_layers` | Number of convolution layers | derived from `conv_channels` | `2`, `3` |
| `kernel_size` | Convolution kernel size | `part_2/main.py`, `part_2/hyperparameter_tuning.py` | `3`, `5` |
| `pool_kernel_size` | Max-pooling kernel / stride | `part_2/main.py` | `2` |
| `hidden_dim` | Size of dense classifier hidden layer | `part_2/main.py`, `part_2/hyperparameter_tuning.py` | preset-dependent, plus explicit tests with `64` and `512` |
| `activation` | Activation function | `part_2/main.py`, `part_2/hyperparameter_tuning.py` | `ReLU`, `LeakyReLU` |
| `dropout` | Dropout probability in classifier | `part_2/main.py`, `part_2/compare_regularization.py`, `part_2/hyperparameter_tuning.py` | `0.0`, `0.1`, `0.15`, `0.2`, `0.25`, `0.3` |
| `batch_norm` | Batch normalization after convolution | `part_2/main.py`, `part_2/compare_regularization.py`, `part_2/hyperparameter_tuning.py` | `False`, `True` |

## 3. Optimization Hyperparameters

These parameters control training and optimizer behavior.

| Parameter | Meaning | Tested values / examples |
| --- | --- | --- |
| `learning_rate` | Step size for Adam | `0.001`, `0.0007`, `0.0005` |
| `adam_beta1` | Adam first-moment coefficient | `0.9`, `0.85` |
| `adam_beta2` | Adam second-moment coefficient | `0.999`, `0.995` |
| `adam_eps` | Adam numerical stability term | `1e-8`, `1e-7` |
| `epochs` | Number of training epochs | user-controlled at runtime, commonly `7` in comparison runs |
| `batch_size` | Mini-batch size | runtime training parameter in `part_2/main.py` |

## 4. Regularization Methods

The codebase now includes several regularization methods.

| Method | Parameter | Status | Tested values / examples |
| --- | --- | --- | --- |
| Dropout | `dropout` | Implemented and tested | `0.1`, `0.15`, `0.2`, `0.25`, `0.3` |
| L2 regularization | `weight_decay` | Implemented and tested | `0.0`, `1e-4`, `2e-4` |
| L1 regularization | `l1_lambda` | Implemented and tested | `0.0`, `1e-6` |
| Input noise injection | `input_noise_std` | Implemented and tested | `0.0`, `0.02`, `0.03`, `0.05` |
| Batch normalization | `batch_norm` | Implemented and tested | `False`, `True` |
| Data augmentation | `augmentation_config` | Implemented and tested | see Section 5 |

## 5. Data Augmentation Parameters

Data augmentation is configured through `augmentation_config`.

| Parameter | Meaning | Tested values / examples |
| --- | --- | --- |
| `enabled` | Turns augmentation on/off | `False`, `True` |
| `rotation_degrees` | Random rotation range | `0.0`, `10.0`, `12.0` |
| `translate` | Random translation range | `(0.0, 0.0)`, `(0.1, 0.1)` |
| `scale` | Random scale range | `(1.0, 1.0)`, `(0.95, 1.05)`, `(0.9, 1.1)` |

## 6. What Has Been Tested in Practice

### CNN architecture comparisons

The architecture comparison work has tested:

- smaller vs larger 2-layer CNNs
- 2-layer vs 3-layer CNNs
- balanced vs wider 3-layer CNNs
- different activation functions
- dropout vs no dropout
- batch normalization vs no batch normalization

Main script:

- [cnn_comparison.py](C:\Users\emil_\vscode\Assignment1\part_2\cnn_comparison.py)

### Regularization comparison

The regularization comparison has tested these focused setups:

| Run | Main change |
| --- | --- |
| `baseline` | No extra regularization |
| `dropout_only` | Dropout |
| `batchnorm_only` | Batch normalization |
| `weight_decay_only` | L2 regularization |
| `l1_only` | L1 regularization |
| `combined_regularization` | Dropout + batch norm + L2 + L1 + noise |

Main script:

- [compare_regularization.py](C:\Users\emil_\vscode\Assignment1\part_2\compare_regularization.py)

### Hyperparameter tuning sweep

The hyperparameter tuning sweep has tested combinations of:

- model architecture
- number of convolution layers
- convolution width (`conv_channels`)
- kernel size
- hidden layer size
- activation function
- dropout level
- batch normalization
- L1 regularization
- L2 regularization
- input noise
- learning rate
- Adam `beta1`, `beta2`, and `eps`
- data augmentation settings

Main script:

- [hyperparameter_tuning.py](C:\Users\emil_\vscode\Assignment1\part_2\hyperparameter_tuning.py)

## 7. Parameters Explicitly Varied in the Tuning Runs

Below is a compact overview of the main tuning dimensions that have been explored.

| Category | Examples of tested settings |
| --- | --- |
| CNN depth | `2 conv`, `3 conv` |
| CNN width | `[16, 32]`, `[32, 64]`, `[32, 64, 64]`, `[32, 64, 128]`, `[48, 96, 128]` |
| Kernel size | `3x3`, `5x5` |
| Dense hidden size | `64`, default preset sizes, `512` |
| Activation | `ReLU`, `LeakyReLU` |
| Dropout | `0.0`, `0.1`, `0.15`, `0.2`, `0.25`, `0.3` |
| Batch norm | on / off |
| L2 | `0.0`, `1e-4`, `2e-4` |
| L1 | `0.0`, `1e-6` |
| Input noise | `0.0`, `0.02`, `0.03`, `0.05` |
| Learning rate | `0.001`, `0.0007`, `0.0005` |
| Adam betas | `(0.9, 0.999)`, `(0.85, 0.995)` |
| Adam eps | `1e-8`, `1e-7` |
| Augmentation | none, mild augmentation, stronger augmentation |

## 8. Short Report-Friendly Summary

To improve the MNIST classifier, we tested both architectural choices and training hyperparameters. On the architecture side, we compared several CNN variants that differed in depth, width, activation function, kernel size, hidden-layer size, dropout, and batch normalization. On the optimization side, we varied learning rate and Adam settings. We also evaluated several regularization methods, including dropout, batch normalization, L2 regularization (`weight_decay`), L1 regularization (`l1_lambda`), input noise injection, and data augmentation with rotation, translation, and scaling. The final tuning sweep therefore covered both CNN design choices and training hyperparameters rather than changing only one setting at a time.
