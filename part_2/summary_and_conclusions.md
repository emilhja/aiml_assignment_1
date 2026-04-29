## Part 2 - Summary and Conclusions

MNIST CNN experiments: architecture comparison, data augmentation, regularization, and hyperparameter tuning.

All results loaded from the `outputs/` 
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
When only looking at above: Deep wide won but I think cnn_small was suprisingly good considering its size.

### MNIST CNN — What We Learned

**Architecture:**
- 3-conv layers > 2-conv layers performed better
- For this setup channel depth mattered more than classifier width.
  i.e. "Channel depth" how many filters per conv layer (e.g. 32→64→128) was helping more vs
  classifier width. Which makes sense since we only had 10 outputs.
- Max-pooling after each conv provides translation invariance — critical for digit recognition.
  Without pooling: digit shifted 3px right → completely different activation pattern → more difficult prediction.
  With pooling this improves and at the same time focus on the important.

**Regularisation:**
- Dropout was most effective for MNIST. Forces more neurons to activate itself, this was quite important.
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