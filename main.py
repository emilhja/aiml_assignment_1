#!/usr/bin/env python3
"""Train an MNIST classification model and save experiment artifacts."""

import argparse
from datetime import datetime
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colors
from torch import nn

from experiment_db import ExperimentDB
from mnist_loader import get_mnist_loaders
from model_optimisation import CheckpointManager
from notebook_report import create_report_notebook, execute_report_notebook
from torch_gpu import describe_device, get_device

CURRENT_DIR = Path(__file__).resolve().parent
AVAILABLE_MODELS = ("mlp", "cnn_small", "cnn_medium", "cnn_dropout")


class MNISTPerceptron(nn.Module):
    """Simple multilayer perceptron for MNIST digit classification."""

    def __init__(self, hidden_size_1=128, hidden_size_2=64):
        """Initialize the network architecture and hidden layer sizes."""
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_size_1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_size_2, 10),
        )

    def forward(self, x):
        """Run a forward pass on a batch of images."""
        return self.network(x)


class MNISTCNN(nn.Module):
    """Convolutional neural network for MNIST digit classification."""

    def __init__(self):
        """Initialize convolutional feature extraction and classification layers."""
        super().__init__()
        self.features = nn.Sequential(
            # Input: 1 x 28 x 28 -> Output: 32 x 28 x 28
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            # Output: 32 x 14 x 14
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Output: 64 x 14 x 14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            # Output: 64 x 7 x 7
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # 64 x 7 x 7 = 3136 features per image
            nn.Linear(64 * 7 * 7, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        """Run a forward pass on a batch of images."""
        x = self.features(x)
        return self.classifier(x)


class ConfigurableMNISTCNN(nn.Module):
    """Configurable CNN used for architecture comparisons on MNIST."""

    def __init__(self, conv_channels, classifier_hidden_size=128, dropout=0.0):
        """Initialize a CNN with repeated conv-relu-pool blocks."""
        super().__init__()
        feature_layers = []
        in_channels = 1
        spatial_size = 28

        for out_channels in conv_channels:
            feature_layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                ]
            )
            in_channels = out_channels
            spatial_size //= 2

        classifier_layers = [
            nn.Flatten(),
            nn.Linear(in_channels * spatial_size * spatial_size, classifier_hidden_size),
            nn.ReLU(),
        ]
        if dropout > 0.0:
            classifier_layers.append(nn.Dropout(dropout))
        classifier_layers.append(nn.Linear(classifier_hidden_size, 10))

        self.features = nn.Sequential(*feature_layers)
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x):
        """Run a forward pass on a batch of images."""
        x = self.features(x)
        return self.classifier(x)


def build_model(model_name):
    """Create the requested MNIST model by name."""
    if model_name == "mlp":
        return MNISTPerceptron()
    if model_name == "cnn_small":
        return ConfigurableMNISTCNN(conv_channels=[16, 32], classifier_hidden_size=64)
    if model_name == "cnn_medium":
        return MNISTCNN()
    if model_name == "cnn_dropout":
        return ConfigurableMNISTCNN(
            conv_channels=[32, 64],
            classifier_hidden_size=128,
            dropout=0.3,
        )
    raise ValueError(
        f"Unknown model '{model_name}'. Available models: {', '.join(AVAILABLE_MODELS)}"
    )


def build_model_config(model_name):
    """Return serializable metadata for the selected architecture."""
    if model_name == "mlp":
        return {
            "model_name": "MNISTPerceptron",
            "hidden_size_1": 128,
            "hidden_size_2": 64,
        }
    if model_name == "cnn_small":
        return {
            "model_name": "cnn_small",
            "conv_channels": [16, 32],
            "kernel_size": 3,
            "pool_kernel_size": 2,
            "classifier_hidden_size": 64,
            "activation": "ReLU",
            "dropout": 0.0,
        }
    if model_name == "cnn_medium":
        return {
            "model_name": "cnn_medium",
            "conv_channels": [32, 64],
            "kernel_size": 3,
            "pool_kernel_size": 2,
            "classifier_hidden_size": 128,
            "activation": "LeakyReLU",
            "dropout": 0.0,
        }
    if model_name == "cnn_dropout":
        return {
            "model_name": "cnn_dropout",
            "conv_channels": [32, 64],
            "kernel_size": 3,
            "pool_kernel_size": 2,
            "classifier_hidden_size": 128,
            "activation": "ReLU",
            "dropout": 0.3,
        }
    raise ValueError(
        f"Unknown model '{model_name}'. Available models: {', '.join(AVAILABLE_MODELS)}"
    )


def train_one_epoch(model, data_loader, loss_fn, optimizer, device):
    """Train the model for one epoch and return loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return running_loss / total, correct / total


def evaluate(model, data_loader, loss_fn, device):
    """Evaluate the model on a dataset and return loss and accuracy."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = loss_fn(logits, labels)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += batch_size

    return running_loss / total, correct / total


def plot_loss_curves(history, output_path):
    """Plot and save training and validation loss curves across epochs."""
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], marker="o", label="Train loss")
    plt.plot(epochs, history["val_loss"], marker="s", label="Validation loss")
    plt.title("MNIST Loss Per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(list(epochs))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_accuracy_curves(history, output_path):
    """Plot and save training and validation accuracy curves across epochs."""
    epochs = range(1, len(history["train_accuracy"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_accuracy"], marker="o", label="Train accuracy")
    plt.plot(
        epochs,
        history["val_accuracy"],
        marker="s",
        label="Validation accuracy",
    )
    plt.title("MNIST Accuracy Per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.xticks(list(epochs))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def format_elapsed_time(seconds):
    """Return elapsed time in hh:mm:ss format."""
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def collect_predictions(model, data_loader, device):
    """Collect test images, labels, and predictions for post-run analysis."""
    model.eval()
    all_images = []
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in data_loader:
            logits = model(images.to(device))
            predictions = logits.argmax(dim=1).cpu()
            all_images.append(images.cpu())
            all_labels.append(labels.cpu())
            all_predictions.append(predictions)

    return (
        torch.cat(all_images, dim=0),
        torch.cat(all_labels, dim=0),
        torch.cat(all_predictions, dim=0),
    )


def save_prediction_examples(
    images,
    labels,
    predictions,
    output_path,
    title,
    select_correct,
    num_images=10,
):
    """Save either correctly or incorrectly classified examples."""
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
    fig, axes = plt.subplots(rows, columns, figsize=(12, max(3, rows * 2.5)))
    axes = np.atleast_1d(axes).flatten()

    for axis in axes:
        axis.axis("off")

    for plot_index, sample_index in enumerate(selected_indices.tolist()):
        axis = axes[plot_index]
        image = images[sample_index].squeeze(0).cpu()
        true_label = labels[sample_index].item()
        predicted_label = predictions[sample_index].item()
        is_correct = predicted_label == true_label

        axis.imshow(image, cmap="gray")
        axis.set_title(
            f"Pred: {predicted_label}\nTrue: {true_label}",
            color="green" if is_correct else "red",
            fontsize=10,
        )
        axis.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(labels, predictions, output_path):
    """Plot a normalized confusion matrix as percentages."""
    num_classes = 10
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.float32)

    for true_label, predicted_label in zip(labels, predictions):
        confusion[predicted_label.long(), true_label.long()] += 1

    column_totals = confusion.sum(dim=0, keepdim=True).clamp_min(1.0)
    confusion_percent = (confusion / column_totals) * 100.0
    matrix = confusion_percent.numpy()

    fig, axis = plt.subplots(figsize=(9, 7))
    # Allocate more of the palette to the 90-100% range so near-perfect cells
    # remain visually distinct instead of collapsing into the same dark blue.
    norm = colors.TwoSlopeNorm(vmin=0.0, vcenter=90.0, vmax=100.0)
    image = axis.imshow(matrix, cmap="cividis", norm=norm)
    axis.set_title("Test Confusion Matrix (%)")
    axis.set_xlabel("True class")
    axis.set_ylabel("Predicted class")
    axis.set_xticks(range(num_classes))
    axis.set_yticks(range(num_classes))

    for row in range(num_classes):
        for column in range(num_classes):
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
                fontsize=8,
            )

    fig.colorbar(image, ax=axis, label="Percent")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_run_output_dir(base_dir, model_name):
    """Create a timestamped output directory for a single experiment run."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return base_dir / f"run_{timestamp}_{model_name}"


def run_experiment(
    model_name="cnn_medium",
    batch_size=64,
    epochs=5,
    learning_rate=0.001,
    output_dir=None,
    checkpoint_interval=5,
    validation_ratio=0.1,
    seed=42,
    early_stopping_patience=5,
    augmentation_config=None,
):
    """Run training, evaluation, checkpointing, and artifact generation."""
    torch.manual_seed(seed)
    run_started_at = datetime.now().isoformat(timespec="seconds")
    device = get_device(prefer_cuda=True)
    print(describe_device(device))
    if augmentation_config is None:
        augmentation_config = {
            "enabled": False,
            "rotation_degrees": 0.0,
            "translate": (0.0, 0.0),
            "scale": (1.0, 1.0),
        }

    train_loader, val_loader, test_loader = get_mnist_loaders(
        batch_size=batch_size,
        data_dir=CURRENT_DIR / "data",
        validation_ratio=validation_ratio,
        seed=seed,
        augmentation_config=augmentation_config,
    )

    model = build_model(model_name).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if output_dir is None:
        output_root = CURRENT_DIR / "outputs" / "Part2"
        output_path = build_run_output_dir(output_root, model_name)
    else:
        output_path = Path(output_dir)
        if not output_path.is_absolute():
            output_path = CURRENT_DIR / output_path

    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_manager = CheckpointManager(
        output_dir=output_path,
        checkpoint_interval=checkpoint_interval,
    )
    db_path = output_path.parent / "experiments.db"
    config = {
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "checkpoint_interval": checkpoint_interval,
        "validation_ratio": validation_ratio,
        "seed": seed,
        "early_stopping_patience": early_stopping_patience,
        "optimizer_name": "Adam",
        "weight_decay": 0.0,
        "augmentation_enabled": augmentation_config["enabled"],
        "augmentation_description": (
            "rotation_degrees="
            f"{augmentation_config['rotation_degrees']}, "
            f"translate={augmentation_config['translate']}, "
            f"scale={augmentation_config['scale']}"
        ),
        "augmentation_config": augmentation_config,
        "device": str(device),
        "output_dir": str(output_path),
    }
    config.update(build_model_config(model_name))
    config_path = checkpoint_manager.save_config(config)
    experiment_db = ExperimentDB(db_path)
    run_id = experiment_db.create_run(
        run_name=output_path.name,
        started_at=run_started_at,
        config=config,
    )

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "epoch_duration_seconds": [],
    }
    best_epoch = 0
    time_to_best_model_seconds = None
    epochs_without_improvement = 0
    run_start_time = perf_counter()

    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches:       {len(test_loader)}")
    print(f"Model device:  {next(model.parameters()).device}")
    print(f"Training config: {config}")
    print(f"Saved run config to: {config_path}")
    print(f"Experiment database: {db_path}")

    for epoch in range(1, epochs + 1):
        epoch_start_time = perf_counter()
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device
        )
        val_loss, val_accuracy = evaluate(model, val_loader, loss_fn, device)
        epoch_duration_seconds = perf_counter() - epoch_start_time

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["epoch_duration_seconds"].append(epoch_duration_seconds)
        elapsed_since_start = perf_counter() - run_start_time

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} | train_acc={train_accuracy:.2%} | "
            f"val_loss={val_loss:.4f} | val_acc={val_accuracy:.2%} | "
            f"epoch_time={epoch_duration_seconds:.2f}s | "
            f"elapsed={format_elapsed_time(elapsed_since_start)}"
        )

        best_path = checkpoint_manager.save_best(
            epoch,
            model,
            optimizer,
            config,
            train_loss,
            train_accuracy,
            val_loss,
            val_accuracy,
        )
        if best_path is not None:
            best_epoch = epoch
            time_to_best_model_seconds = perf_counter() - run_start_time
            epochs_without_improvement = 0
            print(f"Saved new best model to: {best_path}")
        else:
            epochs_without_improvement += 1

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

        experiment_db.log_epoch(
            run_id=run_id,
            epoch=epoch,
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
            epoch_duration_seconds=epoch_duration_seconds,
            is_best_epoch=best_path is not None,
            checkpoint_saved=checkpoint_path is not None,
        )

        if epochs_without_improvement >= early_stopping_patience:
            print(
                "Early stopping triggered because validation loss "
                f"did not improve for {early_stopping_patience} epochs."
            )
            break

    loss_curve_path = output_path / "loss_curve.png"
    accuracy_curve_path = output_path / "accuracy_curve.png"
    test_predictions_path = output_path / "test_predictions.png"
    plot_loss_curves(history, loss_curve_path)
    plot_accuracy_curves(history, accuracy_curve_path)

    best_checkpoint = torch.load(checkpoint_manager.best_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_loss, test_accuracy = evaluate(model, test_loader, loss_fn, device)
    total_training_time_seconds = perf_counter() - run_start_time
    test_images, test_labels, test_predictions = collect_predictions(
        model,
        test_loader,
        device,
    )
    save_prediction_examples(
        test_images,
        test_labels,
        test_predictions,
        output_path / "correct_predictions.png",
        title="Correctly Classified Test Examples",
        select_correct=True,
    )
    save_prediction_examples(
        test_images,
        test_labels,
        test_predictions,
        output_path / "incorrect_predictions.png",
        title="Incorrectly Classified Test Examples",
        select_correct=False,
    )
    save_prediction_examples(
        test_images,
        test_labels,
        test_predictions,
        test_predictions_path,
        title="Test Prediction Examples",
        select_correct=True,
    )
    plot_confusion_matrix(
        test_labels,
        test_predictions,
        output_path / "confusion_matrix.png",
    )
    history_path = checkpoint_manager.save_history(history)
    best_validation_accuracy = history["val_accuracy"][best_epoch - 1]
    summary = {
        "best_epoch": best_epoch,
        "best_validation_loss": checkpoint_manager.best_loss,
        "best_validation_accuracy": best_validation_accuracy,
        "time_to_best_model_seconds": time_to_best_model_seconds,
        "total_training_time_seconds": total_training_time_seconds,
        "average_epoch_time_seconds": (
            sum(history["epoch_duration_seconds"])
            / len(history["epoch_duration_seconds"])
        ),
        "final_test_loss": test_loss,
        "final_test_accuracy": test_accuracy,
        "stopped_early": len(history["train_loss"]) < epochs,
        "epochs_completed": len(history["train_loss"]),
        "best_model_path": str(checkpoint_manager.best_path),
    }
    summary_path = checkpoint_manager.save_summary(summary)
    experiment_db.finalize_run(run_id, summary)
    experiment_db.close()
    report_notebook_path = create_report_notebook(output_path)
    execute_report_notebook(report_notebook_path)

    print(f"Saved loss curve to: {loss_curve_path}")
    print(f"Saved accuracy curve to: {accuracy_curve_path}")
    print(f"Saved test predictions to: {test_predictions_path}")
    print(f"Saved correct predictions to: {output_path / 'correct_predictions.png'}")
    print(f"Saved incorrect predictions to: {output_path / 'incorrect_predictions.png'}")
    print(f"Saved confusion matrix to: {output_path / 'confusion_matrix.png'}")
    print(f"Saved training history to: {history_path}")
    print(f"Saved experiment summary to: {summary_path}")
    print(f"Saved executed notebook report to: {report_notebook_path}")
    print(
        f"Timing: total={total_training_time_seconds:.2f}s | "
        f"time_to_best={time_to_best_model_seconds:.2f}s"
    )
    print(f"Final test_loss={test_loss:.4f} | test_acc={test_accuracy:.2%}")

    return model, history, loss_curve_path, test_predictions_path


def parse_args():
    """Parse command-line arguments for experiment configuration."""
    parser = argparse.ArgumentParser(
        description="Train the MNIST CNN and save experiment artifacts.",
        epilog=(
            "Example: python main.py --epochs 10 --batch-size 128 "
            "--learning-rate 0.0005 --model cnn_dropout"
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="cnn_medium",
        choices=AVAILABLE_MODELS,
        help="Model architecture to train.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Mini-batch size used for training and evaluation.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate for the Adam optimizer.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory for this run.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5,
        help="Save a periodic checkpoint every N epochs.",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.1,
        help="Fraction of the training set reserved for validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5,
        help="Stop after N epochs without validation loss improvement.",
    )
    return parser.parse_args()


def main():
    """Run the default MNIST training experiment."""
    args = parse_args()
    run_experiment(
        model_name=args.model,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        checkpoint_interval=args.checkpoint_interval,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
        early_stopping_patience=args.early_stopping_patience,
    )


if __name__ == "__main__":
    main()
