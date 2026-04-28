"""Data augmentation notes.

# 🛟 Exempel på transform för träningsdatasettet (via DataLoader:n för träningsdata). Observera att man inte vill ha random, jitter och sådant för test-data, endast för träningsdata:
# Compared to current implementation this is more relevant for color pictures, but it is still a good example of how to do data augmentation. You can adjust the parameters as needed.

```python
# TRAINING DATA TRANSFORM
train_transform = transforms.Compose([
    transforms.RandomAffine(degrees=degrees, translate=translate, scale=scale),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(saturation=saturation),  # Saturation to given range
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```
"""
