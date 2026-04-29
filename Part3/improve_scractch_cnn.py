#!/usr/bin/env python3
"""Run stronger scratch-only CNN experiments for Part 3."""

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import torch
from torch import nn

CURRENT_DIR = Path(__file__).resolve().parent.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from part_2.model_optimisation import CheckpointManager
from Part3.notebook_report import create_report_notebook, execute_report_notebook
from Part3.part3_finetuning_external_models import (
    build_confusion_matrix,
    build_data_loaders,
    collect_predictions,
    compute_macro_f1,
    evaluate,
    get_git_revision,
    get_device,
    plot_accuracy_curves,
    plot_confusion_matrix,
    plot_loss_curves,
    resolve_path,
    save_history_json,
    save_prediction_examples,
    seed_everything,
    seconds_per_image,
    milliseconds_per_image,
    images_per_second,
    synchronize_device,
    train_one_epoch,
)
from part_2.torch_gpu import describe_device

AVAILABLE_VARIANTS = (
    "baseline_v2",
    "deeper_cnn",
    "residual_cnn",
)


class ConvBlock(nn.Module):
    """A small conv block with two convolutions before downsampling."""

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

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    """A basic residual block for a scratch CNN."""

    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        out = out + identity
        return self.relu(out)


class ImprovedScratchCNN(nn.Module):
    """A stronger plain CNN baseline with double-conv blocks."""

    def __init__(self, channels=(32, 64, 128, 256, 384), classifier_width=256):
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
            nn.Linear(classifier_width, 2),
        )

    def forward(self, images):
        return self.classifier(self.features(images))


class ResidualScratchCNN(nn.Module):
    """A small residual CNN trained from scratch."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layers = nn.Sequential(
            ResidualBlock(32, 32, stride=1),
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 64, stride=1, dropout=0.05),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128, stride=1, dropout=0.1),
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256, stride=1, dropout=0.1),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.35),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, 2),
        )

    def forward(self, images):
        x = self.stem(images)
        x = self.layers(x)
        x = self.pool(x)
        return self.classifier(x)


def build_parser():
    """Create a CLI for scratch-improvement experiments."""
    parser = argparse.ArgumentParser(
        description="Run improved scratch CNN experiments on Oxford-IIIT Pet.",
        epilog=(
            "Example: .\\venv\\Scripts\\python.exe Part3\\improve_scractch_cnn.py "
            "--variants baseline_v2 deeper_cnn residual_cnn --epochs 25"
        ),
    )
    parser.add_argument("--variants", nargs="+", default=list(AVAILABLE_VARIANTS))
    parser.add_argument("--dataset-root", type=str, default="data")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--scheduler", choices=("cosine", "plateau", "none"), default="cosine")
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--output-root", type=str, default=None)
    return parser


def validate_variants(variants):
    unknown = [variant for variant in variants if variant not in AVAILABLE_VARIANTS]
    if unknown:
        raise ValueError(
            f"Unknown variant(s): {', '.join(unknown)}. "
            f"Available: {', '.join(AVAILABLE_VARIANTS)}"
        )


def count_parameters(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable


def compute_class_weights(train_loader):
    """Compute inverse-frequency class weights from the training split."""
    counts = torch.zeros(2, dtype=torch.float32)
    for _images, labels in train_loader:
        bincount = torch.bincount(labels, minlength=2).to(torch.float32)
        counts += bincount
    weights = counts.sum() / torch.clamp(counts, min=1.0)
    return weights / weights.sum() * len(weights)


def build_variant_config(variant_name):
    """Return the model and optimization defaults for a variant."""
    if variant_name == "baseline_v2":
        model = ImprovedScratchCNN(channels=(32, 64, 128, 256), classifier_width=128)
        return {
            "model": model,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "weight_decay_multiplier": 1.0,
            "use_class_weights": False,
            "notes": "Deeper plain CNN with two convolutions per block.",
        }

    if variant_name == "deeper_cnn":
        model = ImprovedScratchCNN(channels=(32, 64, 128, 256, 384), classifier_width=256)
        return {
            "model": model,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "weight_decay_multiplier": 1.0,
            "use_class_weights": True,
            "notes": "Higher-capacity CNN with class-weighted loss.",
        }

    if variant_name == "residual_cnn":
        model = ResidualScratchCNN()
        return {
            "model": model,
            "optimizer": "adamw",
            "scheduler": "plateau",
            "weight_decay_multiplier": 0.5,
            "use_class_weights": True,
            "notes": "Scratch residual CNN with skip connections.",
        }

    raise ValueError(f"Unsupported variant '{variant_name}'")


def build_optimizer(model, optimizer_name, learning_rate, weight_decay):
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            nesterov=True,
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer '{optimizer_name}'")


def build_scheduler(optimizer, scheduler_name, epochs):
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if scheduler_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
        )
    if scheduler_name == "none":
        return None
    raise ValueError(f"Unsupported scheduler '{scheduler_name}'")


def make_output_root(output_root):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    if output_root is None:
        return CURRENT_DIR / "Part3" / "outputs" / f"scratch_improvement_{timestamp}"
    return resolve_path(output_root)


def train_variant(
    model,
    variant_name,
    train_loader,
    val_loader,
    device,
    epochs,
    learning_rate,
    weight_decay,
    scheduler_name,
    checkpoint_manager,
    config,
    label_smoothing,
    class_weights=None,
):
    """Train one scratch variant and persist best checkpoints."""
    weights_tensor = None if class_weights is None else class_weights.to(device)
    loss_fn = nn.CrossEntropyLoss(
        weight=weights_tensor,
        label_smoothing=label_smoothing,
    )
    optimizer = build_optimizer(model, config["optimizer_name"], learning_rate, weight_decay)
    scheduler = build_scheduler(optimizer, scheduler_name, epochs)
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
    run_start = perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start = perf_counter()
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
        epoch_duration = perf_counter() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["epoch_duration_seconds"].append(epoch_duration)
        history["learning_rate"].append(current_lr)
        history["stage"].append("scratch_improved")

        print(
            f"{variant_name} | epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} | train_acc={train_accuracy:.2%} | "
            f"val_loss={val_loss:.4f} | val_acc={val_accuracy:.2%} | "
            f"lr={current_lr:.6f} | epoch_time={epoch_duration:.2f}s"
        )

        is_better_accuracy = val_accuracy > best_validation_accuracy
        is_tied_accuracy = math.isclose(val_accuracy, best_validation_accuracy, rel_tol=0.0, abs_tol=1e-12)
        is_better_loss_tiebreak = val_loss < best_validation_loss
        if is_better_accuracy or (is_tied_accuracy and is_better_loss_tiebreak):
            best_validation_accuracy = val_accuracy
            best_validation_loss = val_loss
            payload = checkpoint_manager._build_payload(
                epoch,
                model,
                optimizer,
                config,
                train_loss,
                train_accuracy,
                val_loss,
                val_accuracy,
            )
            torch.save(payload, checkpoint_manager.best_path)
            best_epoch = epoch
            time_to_best_model_seconds = perf_counter() - run_start
            print(f"Saved new best model to: {checkpoint_manager.best_path}")

        checkpoint_path = checkpoint_manager.save_periodic(
            epoch,
            model,
            optimizer,
            config,
            train_loss,
            train_accuracy,
            val_loss,
            val_accuracy,
        )
        if checkpoint_path is not None:
            print(f"Saved checkpoint to: {checkpoint_path}")

        if scheduler is not None:
            if scheduler_name == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

    total_training_time_seconds = perf_counter() - run_start
    return (
        history,
        best_epoch,
        best_validation_accuracy,
        best_validation_loss,
        time_to_best_model_seconds,
        total_training_time_seconds,
    )


def run_variant(args, variant_name, output_root):
    """Run one improved scratch experiment."""
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
    variant = build_variant_config(variant_name)
    model = variant["model"].to(device)
    total_parameters, trainable_parameters = count_parameters(model)

    class_weights = None
    if variant["use_class_weights"]:
        class_weights = compute_class_weights(data_bundle["train_loader"])

    output_dir = output_root / variant_name
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_manager = CheckpointManager(
        output_dir=output_dir,
        checkpoint_interval=args.checkpoint_interval,
    )
    effective_scheduler = args.scheduler if args.scheduler != "none" else variant["scheduler"]
    if variant["scheduler"] == "none":
        effective_scheduler = args.scheduler

    config = {
        "task_name": "oxford_pet_cat_vs_dog",
        "label_mode": data_bundle["label_mode"],
        "class_names": data_bundle["class_names"],
        "breed_count": len(data_bundle["breed_names"]),
        "breed_names_preview": data_bundle["breed_names"][:10],
        "model_name": "scratch_cnn_improved",
        "backbone_name": variant_name,
        "transfer_learning": False,
        "weights_name": None,
        "variant_name": variant_name,
        "variant_notes": variant["notes"],
        "batch_size": args.batch_size,
        "epochs_head": args.epochs,
        "epochs_finetune": 0,
        "learning_rate_head": args.learning_rate,
        "learning_rate_finetune": 0.0,
        "validation_ratio": args.validation_ratio,
        "test_ratio": args.test_ratio,
        "seed": args.seed,
        "checkpoint_interval": args.checkpoint_interval,
        "num_workers": args.num_workers,
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "device": str(device),
        "image_size": [224, 224],
        "normalization_mean": [0.485, 0.456, 0.406],
        "normalization_std": [0.229, 0.224, 0.225],
        "train_transform": (
            "Resize(224x224) -> RandomHorizontalFlip -> RandomAffine -> "
            "ColorJitter -> ToTensor -> Normalize(ImageNet)"
        ),
        "eval_transform": "Resize(224x224) -> ToTensor -> Normalize(ImageNet)",
        "train_size": data_bundle["train_size"],
        "validation_size": data_bundle["val_size"],
        "test_size": data_bundle["test_size"],
        "total_parameters": total_parameters,
        "initial_trainable_parameters": trainable_parameters,
        "stage_plan": [
            {
                "name": "scratch_improved",
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
            }
        ],
        "run_started_at": run_started_at,
        "optimizer_name": variant["optimizer"],
        "scheduler_name": effective_scheduler,
        "weight_decay": args.weight_decay * variant["weight_decay_multiplier"],
        "label_smoothing": args.label_smoothing,
        "use_class_weights": variant["use_class_weights"],
        "class_weights": None if class_weights is None else [float(x) for x in class_weights.tolist()],
    }
    config.update(get_git_revision(CURRENT_DIR))
    checkpoint_manager.save_config(config)

    (
        history,
        best_epoch,
        best_validation_accuracy,
        best_validation_loss,
        time_to_best_model_seconds,
        total_training_time_seconds,
    ) = train_variant(
        model=model,
        variant_name=variant_name,
        train_loader=data_bundle["train_loader"],
        val_loader=data_bundle["val_loader"],
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=config["weight_decay"],
        scheduler_name=effective_scheduler,
        checkpoint_manager=checkpoint_manager,
        config=config,
        label_smoothing=args.label_smoothing,
        class_weights=class_weights,
    )

    plot_loss_curves(history, output_dir / "loss_curve.png")
    plot_accuracy_curves(history, output_dir / "accuracy_curve.png")
    save_history_json(history, output_dir / "training_history.json")

    best_checkpoint = torch.load(
        checkpoint_manager.best_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_loss_fn = nn.CrossEntropyLoss()
    synchronize_device(device)
    test_eval_start = perf_counter()
    test_loss, test_accuracy = evaluate(
        model=model,
        data_loader=data_bundle["test_loader"],
        loss_fn=test_loss_fn,
        device=device,
    )
    synchronize_device(device)
    test_evaluation_time_seconds = perf_counter() - test_eval_start

    synchronize_device(device)
    prediction_collection_start = perf_counter()
    test_images, test_labels, test_predictions = collect_predictions(
        model=model,
        data_loader=data_bundle["test_loader"],
        device=device,
    )
    synchronize_device(device)
    test_prediction_collection_time_seconds = (
        perf_counter() - prediction_collection_start
    )
    final_test_total_time_seconds = (
        test_evaluation_time_seconds + test_prediction_collection_time_seconds
    )
    confusion = build_confusion_matrix(
        labels=test_labels,
        predictions=test_predictions,
        num_classes=len(data_bundle["class_names"]),
    )
    macro_f1 = compute_macro_f1(
        labels=test_labels,
        predictions=test_predictions,
        num_classes=len(data_bundle["class_names"]),
    )
    plot_confusion_matrix(
        labels=test_labels,
        predictions=test_predictions,
        class_names=data_bundle["class_names"],
        output_path=output_dir / "confusion_matrix.png",
    )
    save_prediction_examples(
        images=test_images,
        labels=test_labels,
        predictions=test_predictions,
        class_names=data_bundle["class_names"],
        output_path=output_dir / "correct_predictions.png",
        title=f"Correct Predictions: {variant_name}",
        select_correct=True,
    )
    save_prediction_examples(
        images=test_images,
        labels=test_labels,
        predictions=test_predictions,
        class_names=data_bundle["class_names"],
        output_path=output_dir / "incorrect_predictions.png",
        title=f"Incorrect Predictions: {variant_name}",
        select_correct=False,
    )

    summary = {
        "best_epoch": best_epoch,
        "best_stage": "scratch_improved",
        "selection_metric": "validation_accuracy",
        "best_validation_accuracy": best_validation_accuracy,
        "best_validation_loss_at_best_epoch": best_validation_loss,
        "best_epoch_by_loss": history["val_loss"].index(min(history["val_loss"])) + 1,
        "best_validation_loss": min(history["val_loss"]),
        "time_to_best_model_seconds": time_to_best_model_seconds,
        "total_training_time_seconds": total_training_time_seconds,
        "average_epoch_time_seconds": (
            sum(history["epoch_duration_seconds"]) / len(history["epoch_duration_seconds"])
        ),
        "final_test_loss": test_loss,
        "final_test_accuracy": test_accuracy,
        "final_test_macro_f1": macro_f1,
        "test_evaluation_time_seconds": test_evaluation_time_seconds,
        "test_evaluation_images_per_second": images_per_second(
            data_bundle["test_size"],
            test_evaluation_time_seconds,
        ),
        "test_evaluation_time_per_image_seconds": seconds_per_image(
            test_evaluation_time_seconds,
            data_bundle["test_size"],
        ),
        "test_evaluation_time_per_image_ms": milliseconds_per_image(
            test_evaluation_time_seconds,
            data_bundle["test_size"],
        ),
        "test_prediction_collection_time_seconds": (
            test_prediction_collection_time_seconds
        ),
        "final_test_total_time_seconds": final_test_total_time_seconds,
        "stopped_early": False,
        "epochs_completed": len(history["train_loss"]),
        "best_model_path": str(checkpoint_manager.best_path),
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "dataset_sizes": {
            "train": data_bundle["train_size"],
            "validation": data_bundle["val_size"],
            "test": data_bundle["test_size"],
        },
        "confusion_matrix_counts": confusion.int().tolist(),
    }
    checkpoint_manager.save_summary(summary)
    report_notebook_path = create_report_notebook(output_dir)
    execute_report_notebook(report_notebook_path)

    print(
        f"{variant_name} complete | test_loss={test_loss:.4f} | "
        f"test_acc={test_accuracy:.2%} | macro_f1={macro_f1:.4f} | "
        f"test_eval_time={test_evaluation_time_seconds:.2f}s | "
        f"test_total_time={final_test_total_time_seconds:.2f}s"
    )
    return summary


def save_comparison(output_root, results):
    comparison_path = output_root / "comparison_summary.json"
    comparison_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    return comparison_path


def main():
    args = build_parser().parse_args()
    validate_variants(args.variants)
    seed_everything(args.seed)
    output_root = make_output_root(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    results = {}
    for variant_name in args.variants:
        print(f"\n=== Running {variant_name} ===")
        results[variant_name] = run_variant(args, variant_name, output_root)

    comparison_path = save_comparison(output_root, results)
    print(f"\nSaved scratch improvement summary to: {comparison_path}")
    for variant_name, summary in sorted(
        results.items(),
        key=lambda item: item[1]["final_test_accuracy"],
        reverse=True,
    ):
        print(
            f"{variant_name}: "
            f"test_acc={summary['final_test_accuracy']:.2%} | "
            f"macro_f1={summary['final_test_macro_f1']:.4f} | "
            f"best_val_acc={summary['best_validation_accuracy']:.2%}"
        )


if __name__ == "__main__":
    main()
