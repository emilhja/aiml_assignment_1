 Verdict: Part 3 is technically fulfilled, but I would fix a few report/clarity issues before submitting.

  Evidence:

  - External dataset: uses Oxford-IIIT Pet, mapped from 37 breeds to cat/dog in part_3/part3_finetuning_external_models.py:168 and loaded via
    OxfordIIITPet at part_3/part3_finetuning_external_models.py:359.
  - Classic learning: ScratchPetCNN and DeeperScratchPetCNN are custom models with random initialization, no pretrained weights, at part_3/
    part3_finetuning_external_models.py:217 and part_3/part3_finetuning_external_models.py:276.
  - Transfer learning: ResNet18/ResNet50/MobileNetV3 load pretrained torchvision weights and replace the classifier at part_3/
    part3_finetuning_external_models.py:445. ResNet50 specifically uses ResNet50_Weights.DEFAULT at part_3/
    part3_finetuning_external_models.py:481.
  - Freeze/fine-tune behavior exists: pretrained models freeze everything except the head first, then unfreeze for fine-tuning in part_3/
    part3_finetuning_external_models.py:522.
  - Data augmentation is correctly train-only: random flip/affine/color jitter in part_3/part3_finetuning_external_models.py:302, while eval
    uses resize/tensor/normalize only.
  - Completed results exist in part_3/outputs/external_model_comparison_2026-04-28_162516.

  Main saved results from that run:

  - scratch_cnn: 50 epochs, test acc 75.47%, macro-F1 0.6965
  - deeper_cnn: 80 epochs, test acc 93.65%, macro-F1 0.9269
  - mobilenet_v3_transfer: 25 epochs, test acc 97.36%, macro-F1 0.9695
  - resnet18_transfer: 25 epochs, test acc 98.72%, macro-F1 0.9852
  - resnet50_transfer: 25 epochs, test acc 99.81%, macro-F1 0.9978

  Issues to fix before submission:

  - part_3/outputs/part3_summary_deep_dive.ipynb has an inconsistency: one markdown cell says MobileNetV3 used 15 epochs and deeper_cnn used
    45, but the cited JSON summaries show MobileNetV3 used 25 and deeper_cnn used 80.
  - There is a newer partial comparison folder, external_model_comparison_2026-04-28_201309, with only three models. Make clear which run is
    the final reported run, preferably external_model_comparison_2026-04-28_162516.
  - Some notebook report text sounds polished/generic, especially sections like “Elevator Pitch”, “The Data Story”, and “Main Conclusions”.
    If this is your actual report text, rewrite it yourself from your own understanding. Do not ask AI to rewrite it.

  Syntax check passed: python -m compileall part_3.

  10 questions you should be able to answer:

  1. Why is Oxford-IIIT Pet a valid “own/external” dataset for Part 3?
  2. How are breed labels converted into binary cat/dog labels?
  3. Why is ImageNet normalization used for both scratch and transfer models?
  4. What exactly is random-initialized in scratch_cnn?
  5. What does the replaced ResNet50 fc layer do?
  6. Why freeze the backbone during the first transfer-learning stage?
  7. Why can ImageNet features help with cat/dog classification?
  8. Why is macro-F1 much worse than accuracy for the original scratch CNN?
  9. Why is the official test split only used after validation selection?
  10. Why might ResNet50 outperform MobileNetV3, and why might MobileNetV3 still be preferable?
