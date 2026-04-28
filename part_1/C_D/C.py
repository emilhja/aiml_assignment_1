#!/usr/bin/env python3

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import torch
from torch import nn

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
OUTPUT_ROOT = CURRENT_DIR / "outputs"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from part_2.mnist_loader import get_mnist_loaders
from part_2.model_optimisation import CheckpointManager
from part_2.torch_gpu import describe_device, get_device


class MNISTPerceptron(nn.Module):
    def __init__(self, hidden_size_1=128, hidden_size_2=64):
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
        return self.network(x)


def run_epoch(model, data_loader, loss_fn, device, optimizer=None):
    is_training = optimizer is not None
    model.train(is_training)
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(is_training):
        for images, labels in data_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if is_training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, labels)
            if is_training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += batch_size

    return running_loss / total, correct / total


def plot_loss_curves(history, output_path):
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
    epochs = range(1, len(history["train_accuracy"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_accuracy"], marker="o", label="Train accuracy")
    plt.plot(epochs, history["val_accuracy"], marker="s", label="Validation accuracy")
    plt.title("MNIST Accuracy Per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.xticks(list(epochs))
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def show_test_predictions(model, data_loader, device, output_path, num_images=10):
    model.eval()

    with torch.no_grad():
        images, labels = next(iter(data_loader))
        images = images[:num_images]
        labels = labels[:num_images]
        logits = model(images.to(device))
        predictions = logits.argmax(dim=1).cpu()

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()

    for index, axis in enumerate(axes):
        image = images[index].squeeze(0).cpu()
        true_label = labels[index].item()
        predicted_label = predictions[index].item()
        is_correct = predicted_label == true_label

        axis.imshow(image, cmap="gray")
        axis.set_title(
            f"Pred: {predicted_label}\nTrue: {true_label}",
            color="green" if is_correct else "red",
            fontsize=10,
        )
        axis.axis("off")

    fig.suptitle("Test Set: 10 MNIST Images and Model Predictions")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def resolve_output_path(output_dir):
    if output_dir is None:
        return OUTPUT_ROOT

    output_path = Path(output_dir)
    if output_path.is_absolute():
        return output_path

    if output_path.parts and output_path.parts[0] == OUTPUT_ROOT.name:
        return CURRENT_DIR / output_path

    return OUTPUT_ROOT / output_path


def run_experiment(
    batch_size=64,
    epochs=5,
    learning_rate=0.001,
    output_dir=None,
    checkpoint_interval=5,
):
    device = get_device(prefer_cuda=True)
    print(describe_device(device))

    train_loader, validation_loader, test_loader = get_mnist_loaders(
        batch_size=batch_size,
        data_dir=PROJECT_ROOT / "data",
    )

    model = MNISTPerceptron().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    output_path = resolve_output_path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_manager = CheckpointManager(
        output_dir=output_path,
        checkpoint_interval=checkpoint_interval,
    )
    config = {
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "checkpoint_interval": checkpoint_interval,
        "model_name": "MNISTPerceptron",
        "validation_set": "10% split from MNIST train split",
        "test_set": "MNIST test split",
    }
    checkpoint_manager.save_config(config)

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }

    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(validation_loader)}")
    print(f"Test batches:  {len(test_loader)}")
    print(f"Model device:  {next(model.parameters()).device}")

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, loss_fn, device, optimizer
        )
        val_loss, val_accuracy = run_epoch(
            model, validation_loader, loss_fn, device
        )
        test_loss, test_accuracy = run_epoch(
            model, test_loader, loss_fn, device
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["test_loss"].append(test_loss)
        history["test_accuracy"].append(test_accuracy)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} | train_acc={train_accuracy:.2%} | "
            f"val_loss={val_loss:.4f} | val_acc={val_accuracy:.2%} | "
            f"test_loss={test_loss:.4f} | test_acc={test_accuracy:.2%}"
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
            print(f"Saved new best model to: {best_path}")

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

    loss_curve_path = output_path / "loss_curve.png"
    accuracy_curve_path = output_path / "accuracy_curve.png"
    test_predictions_path = output_path / "test_predictions.png"
    plot_loss_curves(history, loss_curve_path)
    plot_accuracy_curves(history, accuracy_curve_path)
    show_test_predictions(model, test_loader, device, test_predictions_path)
    history_path = checkpoint_manager.save_history(history)

    print(f"Saved loss curve to: {loss_curve_path}")
    print(f"Saved accuracy curve to: {accuracy_curve_path}")
    print(f"Saved test predictions to: {test_predictions_path}")
    print(f"Saved training history to: {history_path}")

    return model, history, loss_curve_path, accuracy_curve_path, test_predictions_path


def main():
    run_experiment()


if __name__ == "__main__":
    main()
