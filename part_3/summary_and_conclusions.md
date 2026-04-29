# Part 3 — Summary and conclusions

**Dataset:** Oxford-IIIT Pet (cat vs. dog binary classification)  
**Models compared:** Part2 CNN Deep Wide, Scratch CNN, Deeper CNN, ResNet18, ResNet50, MobileNetV3-Small  
**Key question:** Does transfer learning outperform training from scratch on a small dataset?

---
**Dataset split:**
- Train: 3,312 images
- Validation: 368 images
- Test: 3,669 images

**Training regime for transfer learning:**  
Head-only (3 epochs, LR=1e-3) → Finetune (22 epochs, LR=1e-4)

**Part 2 CNN baseline adaptation:**  
`part2_cnn_deep_wide` is the best Part 2 `cnn_deep_wide` architecture adapted from MNIST to the Oxford-IIIT Pet RGB images. The important conversion is in the first convolution: the Part 2 MNIST model used one grayscale input channel (`Conv2d(1, 32, ...)`), while the Part 3 version uses three RGB input channels (`Conv2d(3, 32, ...)`). So it was not only "one more channel"; it was expanded from 1 channel to 3 channels while keeping the same wide convolution stack `[32 → 64 → 128]` and 256-unit classifier style.

**Part 2 CNN baseline result:**  
`part2_cnn_deep_wide`: best_epoch=33, best_val_acc=92.12%, test_acc=87.19%, macro_f1=0.8542, time=1027.16s, test_eval_time=1.43s.

**Best result:** ResNet50 transfer — **99.62% test accuracy**

### Model Overview Analysis

**Dramatic accuracy gap between scratch and transfer learning:**

| Model | Test Acc | Params | Train Time |
|---|---|---|---|
| Part2 CNN Deep Wide | 87.19% | 389K | 17.1 min |
| Scratch CNN | 77.08% | 422K | 21.4 min |
| Deeper CNN  | 94.00% | 3.5M | 41.2 min |
| MobileNetV3 | 97.36% | 1.5M | 9.3 min |
| ResNet18    | 98.99% | 11.2M | 10.7 min |
| ResNet50    | **99.62%** | 23.5M | 13.6 min |

**Transfer learning wins decisively**
Achieving near-perfect accuracy in 13 minutes, while the scratch CNN only reaches 77% after 21 minutes. 
The key advantage: ImageNet-pretrained weights encode already have been trained on large amounts of data and learned features, which gives a head start.

**MobileNetV3 efficiency:** 1.5M params → 97.36% accuracy in 9.3 minutes. This is good speed and accuracy for such a small model

**Key takeaways:**
1. Transfer learning was very powerful and much more manual work would be needed to train a model close to that.
2. The adapted Part 2 CNN shows that the MNIST architecture could transfer structurally to RGB images after changing the input channel count, but its 87.19% test accuracy still stayed clearly below the stronger scratch CNN and pretrained models.
3. Amount of parmaters is not everything. Deeper CNN had 3M parametres compared to scratch CNN and MobilnetV3, but in the end MobileNetV3 was still better. Amount of parmeters is not the main priority.
