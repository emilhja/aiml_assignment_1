## Part 2 - Summary and Conclusions

MNIST CNN experiments: architecture comparison, data augmentation, regularization, and hyperparameter tuning.

---
**Experiments run:**
- 5 CNN architectures
- 2 augmentation variants (with vs. without)
- 6 regularization configurations
- 18 hyperparameter tuning configurations

**CNN description and results:**

| Model | Params | Test Acc | Architecture |
|---|---|---|---|
| cnn_small | 105K | 98.80% | 2-conv [16→32], dense 64, ReLU |
| cnn_medium | 421K | 99.11% | 2-conv [32→64], dense 128, LeakyReLU |
| cnn_dropout | 421K | 99.03% | 2-conv [32→64], dense 128, ReLU, dropout=0.3 |
| cnn_deep_balanced | 356K | 99.14% | 3-conv [32→64→64], dense 512, ReLU |
| cnn_deep_wide | 390K | **99.26%** ← best | 3-conv [32→64→128], dense 256, ReLU |

Looking only at the architecture comparison, `cnn_deep_wide` won, but `cnn_small` was still surprisingly strong considering its size.

### MNIST CNN — What We Learned

**Architecture:**
- 3-conv layers > 2-conv layers performed better
- For this setup, channel depth mattered more than classifier width.
  "Channel depth" means how many filters each convolution layer uses, for example 32→64→128. That helped more than only making the final classifier wider, which makes sense for a 10-class task.
- Max-pooling after each convolution layer provides translation invariance, which is important for digit recognition.
  Without pooling: digit shifted 3px right → completely different activation pattern → more difficult prediction.
  With pooling, the model keeps more focus on the important local features.

**Regularisation:**
- Dropout was most effective for MNIST in these runs. It reduces reliance on single neurons and helped generalization.
- MNIST probably needs less regularisation than real-world datasets due to that the data is quite simple and we also have many pictures to train on.

**Augmentation:**
- Small but consistent gain (+0.10%). Probably more impactful on other datasets?
- should probably not transform too much since this can make it more difficult to recognise the correct one. I.e. if you rotate a nr 6, it will eventually be identified as a nr 9

**Hyperparameter tuning:**
- Augmentation + BatchNorm + light Dropout was a good combo
- LR=1e-3 (Adam default) is right for low amount of epochs (here 7 epochs), logically a lower LR needs more epochs to be useful.
- Best accuracy achieved: **99.39%** with 206K parameters and <90s training time.

**Best model for deployment:** 
More testing could be done, but these parameters were the best in my testing:
`tune_07_augmented` or `tune_14_wide_3conv`  
Both achieve top accuracy with compact models (206K–451K params) and fast inference.

### Selected notebooks available

The selected review notebooks are saved in `part_2/selected_outputs/`:

- `augmentation_comparison_2026-04-28_154632.ipynb`
- `cnn_comparison_2026-04-28_154826.ipynb`
- `combined_regularization.ipynb`
- `hyperparameter_tuning_2026-04-28_155935.ipynb`
- `regularization_comparison_2026-04-28_155225.ipynb`

These notebooks include executed cells, plots, and tables for the results summarized above.
