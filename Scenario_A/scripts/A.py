import csv
import os
import time
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import datasets
from torchvision.models import resnet18, ResNet18_Weights

from preprocessing import get_tinyimagenet_transforms


# =========================
# Configuration
# =========================

DATA_ROOT = Path("tiny-imagenet-200")

BATCH_SIZE = 128
EPOCHS = 2
LR = 3e-4
NUM_CLASSES = 200

NUM_WORKERS = 4
IMAGE_SIZE = 224

# EARLY_STOPPING_PATIENCE = 5  # Disabled: fixed 2-epoch run
MIN_DELTA = 0.0001
WEIGHT_DECAY = 1e-4

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = OUTPUT_DIR / "resnet18_tinyimagenet_best.pth"
RESULTS_CSV_PATH = OUTPUT_DIR / "training_results.csv"

# Databricks MLflow tracking settings.
# Before running this script on your private server, set:
#   export DATABRICKS_HOST="https://your-workspace-url"
#   export DATABRICKS_TOKEN="your-databricks-token"
#
# You can override the experiment name without editing this file:
#   export MLFLOW_EXPERIMENT_NAME="/Users/your.email@databricks.com/resnet18-tinyimagenet"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "databricks")
MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "/Users/1749412@uab.cat/resnet18-tinyimagenet",
)
MLFLOW_RUN_NAME = os.getenv("MLFLOW_RUN_NAME", "resnet18-tinyimagenet-private-server")


# =========================
# MLflow setup
# =========================

def configure_mlflow():
    """Configure MLflow to log this private-server training run to Databricks."""
    if MLFLOW_TRACKING_URI == "databricks":
        missing_env_vars = [
            name
            for name in ["DATABRICKS_HOST", "DATABRICKS_TOKEN"]
            if not os.getenv(name)
        ]
        if missing_env_vars:
            raise EnvironmentError(
                "Missing Databricks authentication environment variables: "
                f"{', '.join(missing_env_vars)}.\n"
                "Set them before running, for example:\n"
                "  export DATABRICKS_HOST='https://your-workspace-url'\n"
                "  export DATABRICKS_TOKEN='your-databricks-token'"
            )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


# =========================
# Data loading
# =========================

def get_loaders(device):
    train_tfms, val_tfms = get_tinyimagenet_transforms(image_size=IMAGE_SIZE)

    train_ds = datasets.ImageFolder(DATA_ROOT / "train", transform=train_tfms)
    val_ds = datasets.ImageFolder(DATA_ROOT / "val", transform=val_tfms)

    print(f"Training images: {len(train_ds)}")
    print(f"Validation images: {len(val_ds)}")
    print(f"Classes: {len(train_ds.classes)}")

    use_cuda = device.type == "cuda"

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=use_cuda,
        persistent_workers=True if NUM_WORKERS > 0 else False,
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=use_cuda,
        persistent_workers=True if NUM_WORKERS > 0 else False,
    )

    return train_loader, val_loader, train_ds.classes


# =========================
# Model
# =========================

def build_model():
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # Replace ImageNet's 1000-class final layer with Tiny ImageNet's 200 classes.
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    return model


# =========================
# CSV logging
# =========================

def initialize_csv_log(csv_path):
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy",
            "training_time_seconds",
            "training_throughput_samples_per_sec",
            "avg_data_loading_latency_seconds_per_batch",
            "avg_compute_time_seconds_per_batch",
            "validation_time_seconds",
            "validation_throughput_samples_per_sec",
            "best_val_accuracy",
            "epochs_without_improvement",
            "checkpoint_saved",
        ])


def append_epoch_to_csv(
    csv_path,
    epoch,
    train_metrics,
    val_metrics,
    best_val_acc,
    epochs_without_improvement,
    checkpoint_saved,
):
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch,
            train_metrics["loss"],
            train_metrics["accuracy"],
            val_metrics["loss"],
            val_metrics["accuracy"],
            train_metrics["epoch_time"],
            train_metrics["throughput"],
            train_metrics["avg_data_wait_time"],
            train_metrics["avg_compute_time"],
            val_metrics["eval_time"],
            val_metrics["throughput"],
            best_val_acc,
            epochs_without_improvement,
            checkpoint_saved,
        ])


# =========================
# Training
# =========================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    epoch_start_time = time.perf_counter()

    total_data_wait_time = 0.0
    total_compute_time = 0.0

    end_of_previous_batch = time.perf_counter()

    for images, labels in tqdm(loader, desc="Training"):
        # Measures how long the training loop waited for the next batch.
        data_wait_time = time.perf_counter() - end_of_previous_batch
        total_data_wait_time += data_wait_time

        batch_compute_start = time.perf_counter()

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        # Synchronize CUDA so timing is accurate.
        if device.type == "cuda":
            torch.cuda.synchronize()

        batch_compute_time = time.perf_counter() - batch_compute_start
        total_compute_time += batch_compute_time

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        end_of_previous_batch = time.perf_counter()

    epoch_time = time.perf_counter() - epoch_start_time

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    throughput = total / epoch_time
    avg_data_wait_time = total_data_wait_time / len(loader)
    avg_compute_time = total_compute_time / len(loader)

    return {
        "loss": epoch_loss,
        "accuracy": epoch_acc,
        "epoch_time": epoch_time,
        "throughput": throughput,
        "avg_data_wait_time": avg_data_wait_time,
        "avg_compute_time": avg_compute_time,
        "total_data_wait_time": total_data_wait_time,
        "total_compute_time": total_compute_time,
    }


# =========================
# Validation
# =========================

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    eval_start_time = time.perf_counter()

    for images, labels in tqdm(loader, desc="Validation"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        if device.type == "cuda":
            torch.cuda.synchronize()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    eval_time = time.perf_counter() - eval_start_time

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    throughput = total / eval_time

    return {
        "loss": epoch_loss,
        "accuracy": epoch_acc,
        "eval_time": eval_time,
        "throughput": throughput,
    }


# =========================
# Main
# =========================

def main():
    configure_mlflow()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: cuda")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using device: cpu")
        print("CUDA is not available. Training will run on CPU.")

    with mlflow.start_run(run_name=MLFLOW_RUN_NAME):
        mlflow.log_params({
            "model": "resnet18",
            "pretrained_weights": "ResNet18_Weights.DEFAULT",
            "dataset": "tiny-imagenet-200",
            "num_classes": NUM_CLASSES,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LR,
            "optimizer": "AdamW",
            "weight_decay": WEIGHT_DECAY,
            "loss": "CrossEntropyLoss",
            "label_smoothing": 0.1,
            "scheduler": "CosineAnnealingLR",
            "scheduler_t_max": EPOCHS,
            "scheduler_eta_min": 1e-6,
            "num_workers": NUM_WORKERS,
            "image_size": IMAGE_SIZE,
            "early_stopping_patience": "disabled",
            "min_delta": MIN_DELTA,
            "device": device.type,
            "data_root": str(DATA_ROOT),
            "private_server_training": True,
        })

        if device.type == "cuda":
            mlflow.log_param("gpu_name", torch.cuda.get_device_name(0))

        train_loader, val_loader, class_names = get_loaders(device)
        mlflow.log_params({
            "training_images": len(train_loader.dataset),
            "validation_images": len(val_loader.dataset),
            "num_dataset_classes": len(class_names),
        })

        model = build_model().to(device)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=EPOCHS,
            eta_min=1e-6,
        )

        best_val_acc = 0.0
        # epochs_without_improvement = 0  # Disabled: early stopping is not used

        initialize_csv_log(RESULTS_CSV_PATH)
        print(f"CSV results will be saved to: {RESULTS_CSV_PATH}")
        print(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
        print(f"MLflow experiment: {MLFLOW_EXPERIMENT_NAME}")

        for epoch in range(EPOCHS):
            print(f"\nEpoch {epoch + 1}/{EPOCHS}")

            train_metrics = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
            )

            val_metrics = evaluate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
            )

            train_loss = train_metrics["loss"]
            train_acc = train_metrics["accuracy"]

            val_loss = val_metrics["loss"]
            val_acc = val_metrics["accuracy"]
            current_lr = optimizer.param_groups[0]["lr"]

            print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")
            print(f"Val loss:   {val_loss:.4f} | Val acc:   {val_acc:.4f}")

            print(f"Training time per epoch: {train_metrics['epoch_time']:.2f} seconds")
            print(f"Training throughput: {train_metrics['throughput']:.2f} samples/sec")
            print(f"Average data loading latency: {train_metrics['avg_data_wait_time']:.4f} seconds/batch")
            print(f"Average compute time: {train_metrics['avg_compute_time']:.4f} seconds/batch")

            print(f"Validation time: {val_metrics['eval_time']:.2f} seconds")
            print(f"Validation throughput: {val_metrics['throughput']:.2f} samples/sec")
            print(f"Learning rate: {current_lr:.8f}")

            checkpoint_saved = False
            improved = val_acc > best_val_acc + MIN_DELTA

            if improved:
                best_val_acc = val_acc
                # epochs_without_improvement = 0  # Disabled: early stopping is not used

                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "class_names": class_names,
                    "val_acc": best_val_acc,
                    "epoch": epoch + 1,
                }

                torch.save(checkpoint, CHECKPOINT_PATH)
                checkpoint_saved = True

                print(f"Validation accuracy improved. Saved best model to {CHECKPOINT_PATH}")
            else:
                # epochs_without_improvement += 1  # Disabled: early stopping is not used
                # print(
                #     f"No validation improvement for "
                #     f"{epochs_without_improvement}/{EARLY_STOPPING_PATIENCE} epochs."
                # )
                print("No validation improvement. Early stopping is disabled for this fixed 2-epoch run.")

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "training_time_seconds": train_metrics["epoch_time"],
                    "training_throughput_samples_per_sec": train_metrics["throughput"],
                    "avg_data_loading_latency_seconds_per_batch": train_metrics["avg_data_wait_time"],
                    "avg_compute_time_seconds_per_batch": train_metrics["avg_compute_time"],
                    "validation_time_seconds": val_metrics["eval_time"],
                    "validation_throughput_samples_per_sec": val_metrics["throughput"],
                    "best_val_accuracy": best_val_acc,
                    "epochs_without_improvement": 0,  # Disabled: early stopping is not used
                    "checkpoint_saved": int(checkpoint_saved),
                    "learning_rate": current_lr,
                },
                step=epoch + 1,
            )

            append_epoch_to_csv(
                csv_path=RESULTS_CSV_PATH,
                epoch=epoch + 1,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                best_val_acc=best_val_acc,
                epochs_without_improvement="disabled",
                checkpoint_saved=checkpoint_saved,
            )

            scheduler.step()

            # Early stopping disabled for this fixed 2-epoch run.
            # if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            #     print("\nEarly stopping triggered.")
            #     print(f"Best validation accuracy: {best_val_acc:.4f}")
            #     print(f"Best model saved at: {CHECKPOINT_PATH}")
            #     break

        mlflow.log_metric("final_best_val_accuracy", best_val_acc)
        mlflow.log_artifact(str(RESULTS_CSV_PATH), artifact_path="training_outputs")

        if CHECKPOINT_PATH.exists():
            mlflow.log_artifact(str(CHECKPOINT_PATH), artifact_path="checkpoints")

        print("\nTraining finished.")
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        print(f"Epoch metrics saved to: {RESULTS_CSV_PATH}")
        print("MLflow run logged to Databricks.")


if __name__ == "__main__":
    main()
