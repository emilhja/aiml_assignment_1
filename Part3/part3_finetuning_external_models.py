#!/usr/bin/env python3
"""Train Oxford-IIIT Pet cat-vs-dog models with scratch and transfer setups."""

import argparse
import json
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import nbformat
import numpy as np
import torch
from matplotlib import colors
from nbclient import NotebookClient
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
    mobilenet_v3_small,
    resnet18,
    resnet50,
)

CURRENT_DIR = Path(__file__).resolve().parent.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from model_optimisation import CheckpointManager
from torch_gpu import describe_device, get_device

AVAILABLE_MODELS = (
    "scratch_cnn",
    "deeper_cnn",
    "resnet18_transfer",
    "resnet50_transfer",
    "mobilenet_v3_transfer",
)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _markdown_cell(source):
    """Build a markdown notebook cell."""
    return nbformat.v4.new_markdown_cell(source=source)


def _code_cell(source):
    """Build a code notebook cell."""
    return nbformat.v4.new_code_cell(source=source)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train scratch and transfer models on Oxford-IIIT Pet cat-vs-dog.",
        epilog=(
            "Example: .\\venv\\Scripts\\python.exe "
            "Part3\\part3_finetuning_external_models.py "
            "--model resnet18_transfer --epochs-head 2 --epochs-finetune 2"
        ),
    )
    parser.add_argument(
        "--model",
        choices=AVAILABLE_MODELS,
        default="resnet18_transfer",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="data",
        help="Dataset root that will contain torchvision's oxford-iiit-pet folder.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--epochs-head",
        type=int,
        default=None,
        help="Head-training epochs. Defaults depend on the selected model.",
    )
    parser.add_argument(
        "--epochs-finetune",
        type=int,
        default=None,
        help="Fine-tuning epochs. Defaults depend on the selected model.",
    )
    parser.add_argument("--learning-rate-head", type=float, default=1e-3)
    parser.add_argument("--learning-rate-finetune", type=float, default=1e-4)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=1.0,
        help="Use a fraction of the official test split. Useful for smoke tests.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional explicit output directory for this run.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1,
        help="Save periodic checkpoints every N epochs.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count. Keep 0 for maximum Windows compatibility.",
    )
    return parser.parse_args()


def seed_everything(seed):
    """Set reproducible random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_path(path_str):
    """Resolve a CLI path relative to the repository root."""
    path = Path(path_str)
    if not path.is_absolute():
        path = CURRENT_DIR / path
    return path


def build_run_output_dir(output_root, model_name):
    """Build a timestamped output directory."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return Path(output_root) / f"run_{timestamp}_{model_name}"


def get_git_revision(project_root):
    """Return git commit metadata when available."""
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(project_root), "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return {"git_commit": commit, "git_is_dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"git_commit": None, "git_is_dirty": None}


class OxfordPetLabelAdapter:
    """Map Oxford-IIIT Pet targets to a future-proof label representation."""

    def __init__(self, breed_names):
        self.label_mode = "species"
        self.class_names = ["cat", "dog"]
        self.breed_names = [name.lower() for name in breed_names]

    def encode(self, target):
        """Return the encoded class index plus metadata."""
        breed_index, binary_index = target
        label = int(binary_index)
        if label not in (0, 1):
            raise ValueError(f"Expected binary label in {{0, 1}}, got {label}")
        return label, {
            "species_name": self.class_names[label],
            "breed_index": int(breed_index),
            "breed_name": self.breed_names[int(breed_index)],
        }


class OxfordPetClassificationDataset(Dataset):
    """Dataset wrapper that exposes mapped labels while keeping breed metadata."""

    def __init__(self, root, split, transform, label_adapter, download):
        self.base_dataset = OxfordIIITPet(
            root=root,
            split=split,
            target_types=["category", "binary-category"],
            transform=transform,
            download=download,
        )
        self.label_adapter = label_adapter

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        image, target = self.base_dataset[index]
        label, _metadata = self.label_adapter.encode(target)
        return image, label

    def get_metadata(self, index):
        """Return decoded metadata for a sample index."""
        _image, target = self.base_dataset[index]
        _label, metadata = self.label_adapter.encode(target)
        return metadata


class ScratchPetCNN(nn.Module):
    """Compact CNN baseline for 224x224 RGB pet images."""

    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, images):
        features = self.features(images)
        return self.classifier(features)


class ConvBlock(nn.Module):
    """Two-convolution block used by the stronger scratch CNN."""

    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, images):
        return self.block(images)


class DeeperScratchPetCNN(nn.Module):
    """Higher-capacity scratch CNN that outperformed the original baseline."""

    def __init__(self, num_classes=2, channels=(32, 64, 128, 256, 384), classifier_width=256):
        super().__init__()
        blocks = []
        in_channels = 3
        for index, out_channels in enumerate(channels):
            dropout = 0.0 if index < 2 else 0.1
            blocks.append(ConvBlock(in_channels, out_channels, dropout=dropout))
            in_channels = out_channels
        self.features = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d((1, 1)))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(channels[-1], classifier_width),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(classifier_width, num_classes),
        )

    def forward(self, images):
        features = self.features(images)
        return self.classifier(features)


def build_transforms():
    """Create train/validation transforms."""
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=8,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
            ),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.02,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return train_transform, evaluation_transform


def split_indices(num_samples, validation_ratio, seed):
    """Split a dataset into train and validation indices."""
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("--validation-ratio must be between 0 and 1.")
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(num_samples, generator=generator).tolist()
    val_count = max(1, int(round(num_samples * validation_ratio)))
    val_indices = permutation[:val_count]
    train_indices = permutation[val_count:]
    if not train_indices:
        raise ValueError("Validation split consumed the entire trainval set.")
    return train_indices, val_indices


def select_fractional_subset(num_samples, ratio, seed):
    """Select a deterministic subset of indices."""
    if not 0.0 < ratio <= 1.0:
        raise ValueError("--test-ratio must be in the interval (0, 1].")
    if ratio == 1.0:
        return list(range(num_samples))
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(num_samples, generator=generator).tolist()
    count = max(1, int(round(num_samples * ratio)))
    return permutation[:count]


def build_data_loaders(dataset_root, batch_size, validation_ratio, test_ratio, seed, num_workers):
    """Build train, validation, and test loaders plus dataset metadata."""
    train_transform, evaluation_transform = build_transforms()

    reference_dataset = OxfordIIITPet(
        root=dataset_root,
        split="trainval",
        target_types=["category", "binary-category"],
        download=True,
    )
    label_adapter = OxfordPetLabelAdapter(reference_dataset.classes)
    train_dataset_full = OxfordPetClassificationDataset(
        root=dataset_root,
        split="trainval",
        transform=train_transform,
        label_adapter=label_adapter,
        download=False,
    )
    eval_dataset_full = OxfordPetClassificationDataset(
        root=dataset_root,
        split="trainval",
        transform=evaluation_transform,
        label_adapter=label_adapter,
        download=False,
    )
    test_dataset_full = OxfordPetClassificationDataset(
        root=dataset_root,
        split="test",
        transform=evaluation_transform,
        label_adapter=label_adapter,
        download=False,
    )

    train_indices, val_indices = split_indices(
        num_samples=len(train_dataset_full),
        validation_ratio=validation_ratio,
        seed=seed,
    )
    test_indices = select_fractional_subset(
        num_samples=len(test_dataset_full),
        ratio=test_ratio,
        seed=seed,
    )

    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(eval_dataset_full, val_indices)
    test_dataset = Subset(test_dataset_full, test_indices)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "class_names": label_adapter.class_names,
        "breed_names": label_adapter.breed_names,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
        "train_transform": train_transform,
        "evaluation_transform": evaluation_transform,
        "label_mode": label_adapter.label_mode,
    }


def build_model(model_name):
    """Create the requested model and return stage metadata."""
    if model_name == "scratch_cnn":
        model = ScratchPetCNN(num_classes=2)
        return model, {
            "weights_name": None,
            "transfer_learning": False,
            "backbone_name": "scratch_cnn",
        }

    if model_name == "deeper_cnn":
        model = DeeperScratchPetCNN(num_classes=2)
        return model, {
            "weights_name": None,
            "transfer_learning": False,
            "backbone_name": "deeper_cnn",
            "optimizer_name": "AdamW",
            "scheduler_name": "cosine",
            "weight_decay": 1e-4,
            "label_smoothing": 0.05,
            "use_class_weights": True,
            "variant_notes": "Higher-capacity CNN with class-weighted loss.",
        }

    if model_name == "resnet18_transfer":
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 2)
        return model, {
            "weights_name": str(weights),
            "transfer_learning": True,
            "backbone_name": "resnet18",
        }

    if model_name == "resnet50_transfer":
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 2)
        return model, {
            "weights_name": str(weights),
            "transfer_learning": True,
            "backbone_name": "resnet50",
        }

    if model_name == "mobilenet_v3_transfer":
        weights = MobileNet_V3_Small_Weights.DEFAULT
        model = mobilenet_v3_small(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, 2)
        return model, {
            "weights_name": str(weights),
            "transfer_learning": True,
            "backbone_name": "mobilenet_v3_small",
        }

    raise ValueError(
        f"Unknown model '{model_name}'. Available models: {', '.join(AVAILABLE_MODELS)}"
    )


def get_head_parameters(model, model_name):
    """Return the classifier parameters for head-only optimization."""
    if model_name == "scratch_cnn":
        return list(model.parameters())
    if model_name == "deeper_cnn":
        return list(model.parameters())
    if model_name == "resnet18_transfer":
        return list(model.fc.parameters())
    if model_name == "resnet50_transfer":
        return list(model.fc.parameters())
    if model_name == "mobilenet_v3_transfer":
        return list(model.classifier.parameters())
    raise ValueError(f"Unsupported model '{model_name}'")


def set_trainable_parameters(model, model_name, stage_name):
    """Freeze or unfreeze trainable parameters for the current stage."""
    if model_name in {"scratch_cnn", "deeper_cnn"}:
        for parameter in model.parameters():
            parameter.requires_grad = True
        return

    if stage_name == "head":
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in get_head_parameters(model, model_name):
            parameter.requires_grad = True
        return

    for parameter in model.parameters():
        parameter.requires_grad = True


def count_parameters(model):
    """Return total and trainable parameter counts."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable


def compute_class_weights(data_loader, num_classes):
    """Compute inverse-frequency weights from the training split."""
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for _images, labels in data_loader:
        counts += torch.bincount(labels, minlength=num_classes).to(torch.float32)
    weights = counts.sum() / torch.clamp(counts, min=1.0)
    return weights / weights.sum() * len(weights)


def get_training_defaults(model_name):
    """Return model-specific optimizer and loss defaults."""
    defaults = {
        "optimizer_name": "Adam",
        "scheduler_name": "none",
        "weight_decay": 0.0,
        "label_smoothing": 0.0,
        "use_class_weights": False,
    }
    if model_name == "deeper_cnn":
        defaults.update(
            {
                "optimizer_name": "AdamW",
                "scheduler_name": "cosine",
                "weight_decay": 1e-4,
                "label_smoothing": 0.05,
                "use_class_weights": True,
            }
        )
    return defaults


def train_one_epoch(model, data_loader, loss_fn, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


def evaluate(model, data_loader, loss_fn, device):
    """Evaluate loss and accuracy."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = loss_fn(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


def collect_predictions(model, data_loader, device):
    """Collect test tensors and predictions."""
    model.eval()
    all_images = []
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in data_loader:
            logits = model(images.to(device, non_blocking=True))
            predictions = logits.argmax(dim=1).cpu()
            all_images.append(images.cpu())
            all_labels.append(labels.cpu())
            all_predictions.append(predictions)

    return (
        torch.cat(all_images, dim=0),
        torch.cat(all_labels, dim=0),
        torch.cat(all_predictions, dim=0),
    )


def denormalize_image(image_tensor):
    """Convert a normalized tensor back to displayable RGB."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    image = image_tensor.cpu() * std + mean
    image = image.clamp(0.0, 1.0)
    return image.permute(1, 2, 0).numpy()


def save_prediction_examples(
    images,
    labels,
    predictions,
    class_names,
    output_path,
    title,
    select_correct,
    num_images=10,
):
    """Save example predictions for correct or incorrect classifications."""
    matches = predictions.eq(labels)
    selected_indices = torch.nonzero(
        matches if select_correct else ~matches,
        as_tuple=False,
    ).flatten()

    if len(selected_indices) == 0:
        fig, axis = plt.subplots(figsize=(8, 3))
        axis.text(
            0.5,
            0.5,
            "No examples found for this category.",
            ha="center",
            va="center",
            fontsize=12,
        )
        axis.axis("off")
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return

    selected_indices = selected_indices[:num_images]
    columns = 5
    rows = int(np.ceil(len(selected_indices) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(14, max(3, rows * 2.8)))
    axes = np.atleast_1d(axes).flatten()

    for axis in axes:
        axis.axis("off")

    for plot_index, sample_index in enumerate(selected_indices.tolist()):
        axis = axes[plot_index]
        image = denormalize_image(images[sample_index])
        true_label = labels[sample_index].item()
        predicted_label = predictions[sample_index].item()
        is_correct = predicted_label == true_label

        axis.imshow(image)
        axis.set_title(
            f"Pred: {class_names[predicted_label]}\nTrue: {class_names[true_label]}",
            color="green" if is_correct else "red",
            fontsize=9,
        )
        axis.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_confusion_matrix(labels, predictions, num_classes):
    """Build a count confusion matrix with rows=true and cols=predicted."""
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.float32)
    for true_label, predicted_label in zip(labels, predictions):
        confusion[true_label.long(), predicted_label.long()] += 1
    return confusion


def plot_confusion_matrix(labels, predictions, class_names, output_path):
    """Save a normalized confusion matrix plot."""
    confusion = build_confusion_matrix(labels, predictions, len(class_names))
    row_totals = confusion.sum(dim=1, keepdim=True).clamp_min(1.0)
    confusion_percent = (confusion / row_totals) * 100.0
    matrix = confusion_percent.numpy()

    fig, axis = plt.subplots(figsize=(7, 6))
    norm = colors.TwoSlopeNorm(vmin=0.0, vcenter=50.0, vmax=100.0)
    image = axis.imshow(matrix, cmap="cividis", norm=norm)
    axis.set_title("Oxford Pet Test Confusion Matrix (%)")
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_xticks(range(len(class_names)))
    axis.set_yticks(range(len(class_names)))
    axis.set_xticklabels(class_names)
    axis.set_yticklabels(class_names)

    for row in range(len(class_names)):
        for column in range(len(class_names)):
            value = matrix[row, column]
            red, green, blue, _ = image.cmap(image.norm(value))
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            axis.text(
                column,
                row,
                f"{value:.1f}",
                ha="center",
                va="center",
                color="black" if luminance > 0.5 else "white",
                fontsize=10,
            )

    fig.colorbar(image, ax=axis, label="Percent")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def compute_macro_f1(labels, predictions, num_classes):
    """Compute macro F1 without external dependencies."""
    scores = []
    for class_index in range(num_classes):
        true_positive = (
            (predictions == class_index) & (labels == class_index)
        ).sum().item()
        false_positive = (
            (predictions == class_index) & (labels != class_index)
        ).sum().item()
        false_negative = (
            (predictions != class_index) & (labels == class_index)
        ).sum().item()

        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        precision = (
            true_positive / precision_denominator
            if precision_denominator > 0
            else 0.0
        )
        recall = (
            true_positive / recall_denominator
            if recall_denominator > 0
            else 0.0
        )
        if precision + recall == 0.0:
            scores.append(0.0)
        else:
            scores.append((2.0 * precision * recall) / (precision + recall))
    return float(sum(scores) / len(scores))


def plot_loss_curves(history, output_path):
    """Plot train and validation loss."""
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], marker="o", label="Train loss")
    plt.plot(epochs, history["val_loss"], marker="s", label="Validation loss")
    plt.title("Oxford Pet Loss Per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(list(epochs))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_accuracy_curves(history, output_path):
    """Plot train and validation accuracy."""
    epochs = range(1, len(history["train_accuracy"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_accuracy"], marker="o", label="Train accuracy")
    plt.plot(epochs, history["val_accuracy"], marker="s", label="Validation accuracy")
    plt.title("Oxford Pet Accuracy Per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.xticks(list(epochs))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_history_json(history, output_path):
    """Write training history to JSON with mixed numeric and string fields."""
    serializable_history = {}
    for key, values in history.items():
        if key == "stage":
            serializable_history[key] = [str(value) for value in values]
        else:
            serializable_history[key] = [float(value) for value in values]
    output_path.write_text(
        json.dumps(serializable_history, indent=2),
        encoding="utf-8",
    )
    return output_path


def resolve_epoch_schedule(model_name, epochs_head, epochs_finetune):
    """Resolve model-specific default epoch schedules while preserving overrides."""
    default_schedules = {
        "scratch_cnn": (3, 12),
        "deeper_cnn": (45, 0),
        "mobilenet_v3_transfer": (3, 12),
        "resnet18_transfer": (3, 22),
        "resnet50_transfer": (3, 22),
    }
    default_head, default_finetune = default_schedules[model_name]
    resolved_head = default_head if epochs_head is None else epochs_head
    resolved_finetune = default_finetune if epochs_finetune is None else epochs_finetune
    return resolved_head, resolved_finetune


def create_report_notebook(run_dir):
    """Create an editable notebook report for a Part 3 run."""
    run_dir = Path(run_dir)
    notebook_path = run_dir / "report.ipynb"

    notebook = nbformat.v4.new_notebook()
    notebook.cells = [
        _markdown_cell(
            "# Part 3 Transfer Learning Report\n\n"
            f"Run folder: `{run_dir.name}`\n\n"
            "This notebook loads the saved run artifacts, rebuilds pandas tables from the "
            "JSON files, and includes cells you can adjust and rerun."
        ),
        _code_cell(
            "import json\n"
            "import sys\n"
            "from pathlib import Path\n"
            "\n"
            "import matplotlib.pyplot as plt\n"
            "import pandas as pd\n"
            "import torch\n"
            "from IPython.display import Image, display\n"
            "\n"
            f'PROJECT_ROOT = Path(r"{CURRENT_DIR}")\n'
            f'RUN_DIR = Path(r"{run_dir}")\n'
            "CONFIG_PATH = RUN_DIR / 'config.json'\n"
            "SUMMARY_PATH = RUN_DIR / 'summary.json'\n"
            "HISTORY_PATH = RUN_DIR / 'training_history.json'\n"
            "\n"
            "if str(PROJECT_ROOT) not in sys.path:\n"
            "    sys.path.insert(0, str(PROJECT_ROOT))\n"
            "\n"
            "from Part3.part3_finetuning_external_models import build_data_loaders, denormalize_image\n"
            "\n"
            "def load_json(path):\n"
            "    return json.loads(Path(path).read_text(encoding='utf-8'))\n"
            "\n"
            "config = load_json(CONFIG_PATH)\n"
            "summary = load_json(SUMMARY_PATH)\n"
            "history = load_json(HISTORY_PATH)\n"
            "print(f'Loaded run: {RUN_DIR.name}')"
        ),
        _markdown_cell("## Config And Summary"),
        _code_cell(
            "config_df = pd.DataFrame([\n"
            "    {'field': key, 'value': value}\n"
            "    for key, value in config.items()\n"
            "])\n"
            "summary_df = pd.DataFrame([\n"
            "    {'metric': key, 'value': value}\n"
            "    for key, value in summary.items()\n"
            "])\n"
            "config_df"
        ),
        _code_cell("summary_df"),
        _markdown_cell("## Epoch History"),
        _code_cell(
            "history_df = pd.DataFrame(history)\n"
            "history_df.index = history_df.index + 1\n"
            "history_df.index.name = 'epoch'\n"
            "history_df"
        ),
        _markdown_cell("## Metric Curves From JSON"),
        _code_cell(
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
            "history_df[['train_loss', 'val_loss']].plot(ax=axes[0], marker='o', title='Loss')\n"
            "history_df[['train_accuracy', 'val_accuracy']].plot(ax=axes[1], marker='o', title='Accuracy')\n"
            "axes[0].grid(True, alpha=0.3)\n"
            "axes[1].grid(True, alpha=0.3)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        _markdown_cell("## Rebuild The Dataset"),
        _code_cell(
            "data_bundle = build_data_loaders(\n"
            "    dataset_root=Path(config['dataset_root']),\n"
            "    batch_size=config['batch_size'],\n"
            "    validation_ratio=config['validation_ratio'],\n"
            "    test_ratio=config['test_ratio'],\n"
            "    seed=config['seed'],\n"
            "    num_workers=config.get('num_workers', 0),\n"
            ")\n"
            "pd.DataFrame([\n"
            "    {'split': 'train', 'samples': data_bundle['train_size']},\n"
            "    {'split': 'validation', 'samples': data_bundle['val_size']},\n"
            "    {'split': 'test', 'samples': data_bundle['test_size']},\n"
            "])"
        ),
        _markdown_cell("## Preview A Training Batch"),
        _code_cell(
            "images, labels = next(iter(data_bundle['train_loader']))\n"
            "fig, axes = plt.subplots(2, 4, figsize=(12, 6))\n"
            "axes = axes.flatten()\n"
            "for axis in axes:\n"
            "    axis.axis('off')\n"
            "for index in range(min(8, len(images))):\n"
            "    axes[index].imshow(denormalize_image(images[index]))\n"
            "    axes[index].set_title(data_bundle['class_names'][int(labels[index])])\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        _markdown_cell("## Saved Artifacts"),
        _code_cell(
            "plot_files = [\n"
            "    'loss_curve.png',\n"
            "    'accuracy_curve.png',\n"
            "    'confusion_matrix.png',\n"
            "    'correct_predictions.png',\n"
            "    'incorrect_predictions.png',\n"
            "]\n"
            "for plot_name in plot_files:\n"
            "    plot_path = RUN_DIR / plot_name\n"
            "    print(f'\\n### {plot_name}')\n"
            "    if plot_path.exists():\n"
            "        display(Image(filename=str(plot_path)))\n"
            "    else:\n"
            "        print('Missing:', plot_path)"
        ),
        _markdown_cell("## Re-Run This Experiment"),
        _code_cell(
            "from argparse import Namespace\n"
            "from Part3.part3_finetuning_external_models import run_experiment\n"
            "\n"
            "# Uncomment to rerun into a new folder.\n"
            "# rerun_args = Namespace(\n"
            "#     model=config['model_name'],\n"
            "#     dataset_root=config['dataset_root'],\n"
            "#     batch_size=config['batch_size'],\n"
            "#     epochs_head=config['epochs_head'],\n"
            "#     epochs_finetune=config['epochs_finetune'],\n"
            "#     learning_rate_head=config['learning_rate_head'],\n"
            "#     learning_rate_finetune=config['learning_rate_finetune'],\n"
            "#     validation_ratio=config['validation_ratio'],\n"
            "#     test_ratio=config['test_ratio'],\n"
            "#     seed=config['seed'],\n"
            "#     output_dir=str(RUN_DIR.parent / f'{RUN_DIR.name}_rerun'),\n"
            "#     checkpoint_interval=config['checkpoint_interval'],\n"
            "#     num_workers=config.get('num_workers', 0),\n"
            "# )\n"
            "# run_experiment(rerun_args)"
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


def execute_report_notebook(notebook_path, timeout=1200):
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


def build_stage_plan(model_name, epochs_head, epochs_finetune, lr_head, lr_finetune):
    """Build the ordered training stages for the selected model."""
    stages = []
    if model_name == "deeper_cnn":
        if epochs_head <= 0:
            raise ValueError("deeper_cnn requires a positive --epochs-head value.")
        return [
            {
                "name": "scratch_improved",
                "epochs": epochs_head,
                "learning_rate": lr_head,
            }
        ]
    if epochs_head > 0:
        stages.append(
            {
                "name": "head" if model_name != "scratch_cnn" else "scratch_stage_1",
                "epochs": epochs_head,
                "learning_rate": lr_head,
            }
        )
    if epochs_finetune > 0:
        stages.append(
            {
                "name": "finetune" if model_name != "scratch_cnn" else "scratch_stage_2",
                "epochs": epochs_finetune,
                "learning_rate": lr_finetune,
            }
        )
    if not stages:
        raise ValueError("At least one of --epochs-head or --epochs-finetune must be positive.")
    return stages


def train_model(
    model,
    model_name,
    stage_plan,
    train_loader,
    val_loader,
    device,
    checkpoint_manager,
    config,
    class_weights=None,
):
    """Run staged training with checkpointing and history tracking."""
    class_weights = None if class_weights is None else class_weights.to(device)
    loss_fn = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=config.get("label_smoothing", 0.0),
    )
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "epoch_duration_seconds": [],
        "learning_rate": [],
        "stage": [],
    }
    best_epoch = 0
    best_validation_accuracy = float("-inf")
    best_validation_loss = float("inf")
    time_to_best_model_seconds = None
    run_start_time = perf_counter()
    global_epoch = 0
    total_planned_epochs = sum(stage["epochs"] for stage in stage_plan)

    for stage in stage_plan:
        stage_name = stage["name"]
        epochs = stage["epochs"]
        if epochs <= 0:
            continue

        if model_name in {"scratch_cnn", "deeper_cnn"}:
            current_stage_name = stage_name
        elif stage_name == "head":
            current_stage_name = "head"
        else:
            current_stage_name = "finetune"

        set_trainable_parameters(model, model_name, current_stage_name)
        trainable_parameters_for_stage = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        if config.get("optimizer_name") == "AdamW":
            optimizer = torch.optim.AdamW(
                trainable_parameters_for_stage,
                lr=stage["learning_rate"],
                weight_decay=config.get("weight_decay", 0.0),
            )
        else:
            optimizer = torch.optim.Adam(
                trainable_parameters_for_stage,
                lr=stage["learning_rate"],
                weight_decay=config.get("weight_decay", 0.0),
            )
        scheduler = None
        if config.get("scheduler_name") == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, total_planned_epochs),
            )
        _total_parameters, trainable_parameters = count_parameters(model)
        print(
            f"Starting stage '{stage_name}' with lr={stage['learning_rate']:.6f} "
            f"and trainable_parameters={trainable_parameters}"
        )

        for stage_epoch in range(1, epochs + 1):
            global_epoch += 1
            epoch_start_time = perf_counter()
            train_loss, train_accuracy = train_one_epoch(
                model=model,
                data_loader=train_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=device,
            )
            val_loss, val_accuracy = evaluate(
                model=model,
                data_loader=val_loader,
                loss_fn=loss_fn,
                device=device,
            )
            epoch_duration_seconds = perf_counter() - epoch_start_time

            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(train_accuracy)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_accuracy)
            history["epoch_duration_seconds"].append(epoch_duration_seconds)
            history["learning_rate"].append(stage["learning_rate"])
            history["stage"].append(stage_name)

            print(
                f"Epoch {global_epoch:02d} "
                f"(stage={stage_name} step={stage_epoch}/{epochs}) | "
                f"train_loss={train_loss:.4f} | train_acc={train_accuracy:.2%} | "
                f"val_loss={val_loss:.4f} | val_acc={val_accuracy:.2%} | "
                f"epoch_time={epoch_duration_seconds:.2f}s"
            )

            is_better_accuracy = val_accuracy > best_validation_accuracy
            is_tied_accuracy = abs(val_accuracy - best_validation_accuracy) < 1e-12
            is_better_loss_tiebreak = val_loss < best_validation_loss
            if is_better_accuracy or (is_tied_accuracy and is_better_loss_tiebreak):
                best_validation_accuracy = val_accuracy
                best_validation_loss = val_loss
                payload = checkpoint_manager._build_payload(
                    global_epoch,
                    model,
                    optimizer,
                    config,
                    train_loss,
                    train_accuracy,
                    val_loss,
                    val_accuracy,
                )
                torch.save(payload, checkpoint_manager.best_path)
                best_path = checkpoint_manager.best_path
            else:
                best_path = None

            checkpoint_path = checkpoint_manager.save_periodic(
                global_epoch,
                model,
                optimizer,
                config,
                train_loss,
                train_accuracy,
                val_loss,
                val_accuracy,
            )
            if best_path is not None:
                best_epoch = global_epoch
                time_to_best_model_seconds = perf_counter() - run_start_time
                print(f"Saved new best model to: {best_path}")
            if checkpoint_path is not None:
                print(f"Saved checkpoint to: {checkpoint_path}")
            if scheduler is not None:
                scheduler.step()

    total_training_time_seconds = perf_counter() - run_start_time
    return (
        history,
        best_epoch,
        best_validation_accuracy,
        best_validation_loss,
        time_to_best_model_seconds,
        total_training_time_seconds,
    )


def run_experiment(args):
    """Run a full Part 3 experiment."""
    seed_everything(args.seed)
    run_started_at = datetime.now().isoformat(timespec="seconds")
    dataset_root = resolve_path(args.dataset_root)
    device = get_device(prefer_cuda=True)
    print(describe_device(device))

    data_bundle = build_data_loaders(
        dataset_root=dataset_root,
        batch_size=args.batch_size,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    model, model_metadata = build_model(args.model)
    model = model.to(device)
    training_defaults = get_training_defaults(args.model)
    epochs_head, epochs_finetune = resolve_epoch_schedule(
        model_name=args.model,
        epochs_head=args.epochs_head,
        epochs_finetune=args.epochs_finetune,
    )

    if args.output_dir is None:
        output_root = CURRENT_DIR / "outputs" / "Part3"
        output_path = build_run_output_dir(output_root, args.model)
    else:
        output_path = resolve_path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    checkpoint_manager = CheckpointManager(
        output_dir=output_path,
        checkpoint_interval=args.checkpoint_interval,
    )
    stage_plan = build_stage_plan(
        model_name=args.model,
        epochs_head=epochs_head,
        epochs_finetune=epochs_finetune,
        lr_head=args.learning_rate_head,
        lr_finetune=args.learning_rate_finetune,
    )
    total_parameters, initial_trainable_parameters = count_parameters(model)
    class_weights = None
    if training_defaults["use_class_weights"]:
        class_weights = compute_class_weights(
            data_bundle["train_loader"],
            num_classes=len(data_bundle["class_names"]),
        )
    config = {
        "task_name": "oxford_pet_cat_vs_dog",
        "label_mode": data_bundle["label_mode"],
        "class_names": data_bundle["class_names"],
        "breed_count": len(data_bundle["breed_names"]),
        "breed_names_preview": data_bundle["breed_names"][:10],
        "model_name": args.model,
        "batch_size": args.batch_size,
        "epochs_head": epochs_head,
        "epochs_finetune": epochs_finetune,
        "learning_rate_head": args.learning_rate_head,
        "learning_rate_finetune": args.learning_rate_finetune,
        "validation_ratio": args.validation_ratio,
        "test_ratio": args.test_ratio,
        "seed": args.seed,
        "checkpoint_interval": args.checkpoint_interval,
        "num_workers": args.num_workers,
        "dataset_root": str(dataset_root),
        "output_dir": str(output_path),
        "device": str(device),
        "image_size": [224, 224],
        "normalization_mean": list(IMAGENET_MEAN),
        "normalization_std": list(IMAGENET_STD),
        "train_transform": (
            "Resize(224x224) -> RandomHorizontalFlip -> RandomAffine -> "
            "ColorJitter -> ToTensor -> Normalize(ImageNet)"
        ),
        "eval_transform": "Resize(224x224) -> ToTensor -> Normalize(ImageNet)",
        "train_size": data_bundle["train_size"],
        "validation_size": data_bundle["val_size"],
        "test_size": data_bundle["test_size"],
        "total_parameters": total_parameters,
        "initial_trainable_parameters": initial_trainable_parameters,
        "stage_plan": stage_plan,
        "run_started_at": run_started_at,
        "optimizer_name": training_defaults["optimizer_name"],
        "scheduler_name": training_defaults["scheduler_name"],
        "weight_decay": training_defaults["weight_decay"],
        "label_smoothing": training_defaults["label_smoothing"],
        "use_class_weights": training_defaults["use_class_weights"],
        "class_weights": (
            None if class_weights is None else [float(value) for value in class_weights.tolist()]
        ),
    }
    config.update(model_metadata)
    config.update(get_git_revision(CURRENT_DIR))
    config_path = checkpoint_manager.save_config(config)

    print(f"Train batches: {len(data_bundle['train_loader'])}")
    print(f"Validation batches: {len(data_bundle['val_loader'])}")
    print(f"Test batches: {len(data_bundle['test_loader'])}")
    print(f"Saved run config to: {config_path}")

    (
        history,
        best_epoch,
        best_validation_accuracy,
        best_validation_loss,
        time_to_best_model_seconds,
        total_training_time_seconds,
    ) = train_model(
        model=model,
        model_name=args.model,
        stage_plan=stage_plan,
        train_loader=data_bundle["train_loader"],
        val_loader=data_bundle["val_loader"],
        device=device,
        checkpoint_manager=checkpoint_manager,
        config=config,
        class_weights=class_weights,
    )

    plot_loss_curves(history, output_path / "loss_curve.png")
    plot_accuracy_curves(history, output_path / "accuracy_curve.png")
    history_path = save_history_json(history, output_path / "training_history.json")

    best_checkpoint = torch.load(
        checkpoint_manager.best_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(best_checkpoint["model_state_dict"])
    loss_fn = nn.CrossEntropyLoss()
    test_loss, test_accuracy = evaluate(
        model=model,
        data_loader=data_bundle["test_loader"],
        loss_fn=loss_fn,
        device=device,
    )
    test_images, test_labels, test_predictions = collect_predictions(
        model=model,
        data_loader=data_bundle["test_loader"],
        device=device,
    )
    macro_f1 = compute_macro_f1(
        labels=test_labels,
        predictions=test_predictions,
        num_classes=len(data_bundle["class_names"]),
    )
    confusion = build_confusion_matrix(
        labels=test_labels,
        predictions=test_predictions,
        num_classes=len(data_bundle["class_names"]),
    )

    plot_confusion_matrix(
        labels=test_labels,
        predictions=test_predictions,
        class_names=data_bundle["class_names"],
        output_path=output_path / "confusion_matrix.png",
    )
    save_prediction_examples(
        images=test_images,
        labels=test_labels,
        predictions=test_predictions,
        class_names=data_bundle["class_names"],
        output_path=output_path / "correct_predictions.png",
        title="Correctly Classified Oxford Pet Test Examples",
        select_correct=True,
    )
    save_prediction_examples(
        images=test_images,
        labels=test_labels,
        predictions=test_predictions,
        class_names=data_bundle["class_names"],
        output_path=output_path / "incorrect_predictions.png",
        title="Incorrectly Classified Oxford Pet Test Examples",
        select_correct=False,
    )

    final_total_parameters, final_trainable_parameters = count_parameters(model)
    best_validation_loss_overall = min(history["val_loss"])
    best_epoch_by_loss = history["val_loss"].index(best_validation_loss_overall) + 1
    summary = {
        "best_epoch": best_epoch,
        "best_stage": history["stage"][best_epoch - 1],
        "selection_metric": "validation_accuracy",
        "best_validation_accuracy": best_validation_accuracy,
        "best_validation_loss_at_best_epoch": best_validation_loss,
        "best_epoch_by_loss": best_epoch_by_loss,
        "best_validation_loss": best_validation_loss_overall,
        "time_to_best_model_seconds": time_to_best_model_seconds,
        "total_training_time_seconds": total_training_time_seconds,
        "average_epoch_time_seconds": (
            sum(history["epoch_duration_seconds"]) / len(history["epoch_duration_seconds"])
        ),
        "final_test_loss": test_loss,
        "final_test_accuracy": test_accuracy,
        "final_test_macro_f1": macro_f1,
        "stopped_early": False,
        "epochs_completed": len(history["train_loss"]),
        "best_model_path": str(checkpoint_manager.best_path),
        "trainable_parameters": final_trainable_parameters,
        "total_parameters": final_total_parameters,
        "dataset_sizes": {
            "train": data_bundle["train_size"],
            "validation": data_bundle["val_size"],
            "test": data_bundle["test_size"],
        },
        "confusion_matrix_counts": confusion.int().tolist(),
    }
    summary_path = checkpoint_manager.save_summary(summary)
    report_notebook_path = create_report_notebook(output_path)
    execute_report_notebook(report_notebook_path)

    print(f"Saved training history to: {history_path}")
    print(f"Saved experiment summary to: {summary_path}")
    print(f"Saved executed notebook report to: {report_notebook_path}")
    print(f"Final test_loss={test_loss:.4f} | test_acc={test_accuracy:.2%} | macro_f1={macro_f1:.4f}")
    return summary_path


def main():
    """Entry point."""
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
