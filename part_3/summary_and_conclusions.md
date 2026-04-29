# Part 3 — Summary and conclusions

**Dataset:** Oxford-IIIT Pet (cat vs. dog binary classification)  
**Models compared:** Scratch CNN, Deeper CNN, ResNet18, ResNet50, MobileNetV3-Small  
**Key question:** Does transfer learning outperform training from scratch on a small dataset?

---
**Dataset split:**
- Train: 3,312 images
- Validation: 368 images
- Test: 3,669 images

**Training regime for transfer learning:**  
Head-only (3 epochs, LR=1e-3) → Finetune (22 epochs, LR=1e-4)

**Best result:** ResNet50 transfer — **99.62% test accuracy**

### Model Overview Analysis

**Dramatic accuracy gap between scratch and transfer learning:**

| Model | Test Acc | Params | Train Time |
|---|---|---|---|
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
2. Amount of parmaters is not everything. Deeper CNN had 3M parametres compared to scratch CNN and MobilnetV3, but in the end MobileNetV3 was still better. Amount of parmeters is not the main priority.