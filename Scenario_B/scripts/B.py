import csv
import io
import os
import time
from pathlib import Path
from urllib.parse import urlparse

import boto3
from botocore.config import Config
import mlflow
import mlflow.pytorch
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from tqdm import tqdm
from torchvision.models import resnet18, ResNet18_Weights

from preprocessing import get_tinyimagenet_transforms


# =========================
# Configuration
# =========================

# S3_DATA_URI must point to the dataset root in S3, for example:
#   s3://my-bucket/datasets/tiny-imagenet-200
#
# Expected S3 layout:
#   s3://my-bucket/datasets/tiny-imagenet-200/train/<class_id>/*.JPEG
#   s3://my-bucket/datasets/tiny-imagenet-200/val/<class_id>/*.JPEG
#
# This script does NOT download the dataset locally. It streams each image
# directly from S3 when the DataLoader asks for it.
S3_DATA_URI = os.getenv("S3_DATA_URI")

# Optional S3 output location, for example:
#   s3://my-bucket/experiments/scenario-a/run_01
#
# Local output files are still created temporarily inside OUTPUT_DIR so that:
#   1. the CSV can be written normally
#   2. PyTorch can write the checkpoint normally
#   3. MLflow can log artifacts normally
#
# The dataset itself is never copied locally.
S3_OUTPUT_URI = os.getenv("S3_OUTPUT_URI")

BATCH_SIZE = 128
EPOCHS = 3
LR = 3e-4
NUM_CLASSES = 200

NUM_WORKERS = 4
IMAGE_SIZE = 224

# EARLY_STOPPING_PATIENCE = 5  # Disabled: early stopping removed for fixed 3-epoch run
MIN_DELTA = 0.0001
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = OUTPUT_DIR / "resnet18_tinyimagenet_best.pth"
RESULTS_CSV_PATH = OUTPUT_DIR / "training_results.csv"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "databricks")
MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "/Users/1749412@uab.cat/resnet18-tinyimagenet",
)
MLFLOW_RUN_NAME = os.getenv("MLFLOW_RUN_NAME", "resnet18-tinyimagenet-s3-streaming")


# =========================
# S3 helpers
# =========================

def parse_s3_uri(uri):
    if not uri:
        raise ValueError("S3 URI is empty. Set S3_DATA_URI and/or S3_OUTPUT_URI.")

    parsed = urlparse(uri)
    if parsed.scheme != "s3":
        raise ValueError(f"Expected an S3 URI like s3://bucket/prefix, got: {uri}")

    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/").rstrip("/")

    if not bucket:
        raise ValueError(f"Missing bucket name in S3 URI: {uri}")

    return bucket, prefix


def make_s3_client():
    return boto3.client(
        "s3",
        config=Config(
            retries={"max_attempts": 10, "mode": "standard"},
            connect_timeout=10,
            read_timeout=120,
        ),
    )


def list_s3_image_keys(bucket, prefix):
    s3 = make_s3_client()
    paginator = s3.get_paginator("list_objects_v2")

    image_keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            lower_key = key.lower()
            if lower_key.endswith((".jpeg", ".jpg", ".png")):
                image_keys.append(key)

    return sorted(image_keys)


def upload_file_to_s3(local_path, s3_uri):
    if not s3_uri:
        return

    local_path = Path(local_path)
    if not local_path.exists():
        return

    bucket, prefix = parse_s3_uri(s3_uri)
    s3_key = f"{prefix}/{local_path.name}" if prefix else local_path.name

    s3 = make_s3_client()
    s3.upload_file(str(local_path), bucket, s3_key)
    print(f"Uploaded {local_path} to s3://{bucket}/{s3_key}")


def upload_outputs_to_s3():
    if not S3_OUTPUT_URI:
        return

    upload_file_to_s3(RESULTS_CSV_PATH, S3_OUTPUT_URI)
    upload_file_to_s3(CHECKPOINT_PATH, S3_OUTPUT_URI)


# =========================
# S3 streaming dataset
# =========================

class S3ImageFolder(Dataset):
    """
    Streaming replacement for torchvision.datasets.ImageFolder.

    It lists image keys once during initialization, but it does not download
    the image files. Each image is fetched from S3 inside __getitem__.

    Expected layout:
        root_prefix/split/class_name/image.JPEG

    Example:
        tiny-imagenet-200/train/n01443537/images_0.JPEG
        tiny-imagenet-200/val/n01443537/val_0.JPEG
    """

    def __init__(self, s3_root_uri, split, transform=None):
        self.s3_root_uri = s3_root_uri.rstrip("/")
        self.split = split.strip("/")
        self.transform = transform

        bucket, root_prefix = parse_s3_uri(self.s3_root_uri)
        self.bucket = bucket
        self.root_prefix = root_prefix

        self.split_prefix = f"{self.root_prefix}/{self.split}".strip("/")
        self.image_keys = list_s3_image_keys(self.bucket, self.split_prefix)

        if not self.image_keys:
            raise RuntimeError(
                f"No images found under s3://{self.bucket}/{self.split_prefix}. "
                "Check S3_DATA_URI and dataset layout."
            )

        class_names = set()
        samples = []

        for key in self.image_keys:
            relative = key[len(self.split_prefix):].lstrip("/")
            parts = relative.split("/")

            if len(parts) < 2:
                # Skip files not inside class folders.
                continue

            class_name = parts[0]
            class_names.add(class_name)

        self.classes = sorted(class_names)
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}

        for key in self.image_keys:
            relative = key[len(self.split_prefix):].lstrip("/")
            parts = relative.split("/")

            if len(parts) < 2:
                continue

            class_name = parts[0]
            if class_name in self.class_to_idx:
                samples.append((key, self.class_to_idx[class_name]))

        self.samples = samples
        self.targets = [target for _, target in self.samples]

        if not self.samples:
            raise RuntimeError(
                f"No class-folder image samples found under s3://{self.bucket}/{self.split_prefix}. "
                "Expected layout: split/class_name/image.JPEG"
            )

        self._s3_client = None

    @property
    def s3_client(self):
        # Create one S3 client per process/DataLoader worker.
        # This avoids trying to pickle a boto3 client.
        if self._s3_client is None:
            self._s3_client = make_s3_client()
        return self._s3_client

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        key, label = self.samples[index]

        cloud_read_start = time.perf_counter()
        response = self.s3_client.get_object(Bucket=self.bucket, Key=key)
        image_bytes = response["Body"].read()
        cloud_read_time = time.perf_counter() - cloud_read_start

        # Count the actual payload bytes downloaded from S3 for this image object.
        # This measures image bytes read from S3, not local RAM usage and not upload bytes.
        bytes_downloaded = len(image_bytes)

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label, cloud_read_time, bytes_downloaded


# =========================
# MLflow setup
# =========================

def configure_mlflow():
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
    if not S3_DATA_URI:
        raise EnvironmentError(
            "S3_DATA_URI is required for S3 streaming mode. "
            "Example: export S3_DATA_URI='s3://my-bucket/datasets/tiny-imagenet-200'"
        )

    train_tfms, val_tfms = get_tinyimagenet_transforms(image_size=IMAGE_SIZE)

    train_ds = S3ImageFolder(S3_DATA_URI, split="train", transform=train_tfms)
    val_ds = S3ImageFolder(S3_DATA_URI, split="val", transform=val_tfms)

    print(f"S3 dataset root: {S3_DATA_URI}")
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
            "avg_cloud_read_time_seconds_per_batch",
            "total_cloud_read_time_seconds",
            "s3_bytes_downloaded_per_epoch",
            "s3_bytes_downloaded_per_batch",
            "s3_megabytes_downloaded_per_epoch",
            "s3_megabytes_downloaded_per_batch",
            "validation_s3_bytes_downloaded_per_epoch",
            "validation_s3_bytes_downloaded_per_batch",
            "validation_s3_megabytes_downloaded_per_epoch",
            "validation_s3_megabytes_downloaded_per_batch",
            "total_s3_bytes_downloaded_cumulative",
            "total_s3_megabytes_downloaded_cumulative",
            "avg_compute_time_seconds_per_batch",
            "validation_time_seconds",
            "validation_throughput_samples_per_sec",
            "best_val_accuracy",
            "epochs_without_improvement",
            "checkpoint_saved",
            "learning_rate",
        ])


def append_epoch_to_csv(
    csv_path,
    epoch,
    train_metrics,
    val_metrics,
    best_val_acc,
    epochs_without_improvement,
    checkpoint_saved,
    learning_rate,
    total_s3_bytes_downloaded_cumulative,
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
            train_metrics["avg_cloud_read_time_per_batch"],
            train_metrics["total_cloud_read_time"],
            train_metrics["total_s3_bytes_downloaded"],
            train_metrics["s3_bytes_downloaded_per_batch"],
            train_metrics["total_s3_bytes_downloaded"] / (1024 * 1024),
            train_metrics["s3_bytes_downloaded_per_batch"] / (1024 * 1024),
            val_metrics["total_s3_bytes_downloaded"],
            val_metrics["s3_bytes_downloaded_per_batch"],
            val_metrics["total_s3_bytes_downloaded"] / (1024 * 1024),
            val_metrics["s3_bytes_downloaded_per_batch"] / (1024 * 1024),
            total_s3_bytes_downloaded_cumulative,
            total_s3_bytes_downloaded_cumulative / (1024 * 1024),
            train_metrics["avg_compute_time"],
            val_metrics["eval_time"],
            val_metrics["throughput"],
            best_val_acc,
            epochs_without_improvement,
            checkpoint_saved,
            learning_rate,
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
    total_cloud_read_time = 0.0
    total_s3_bytes_downloaded = 0
    total_compute_time = 0.0

    end_of_previous_batch = time.perf_counter()

    for images, labels, cloud_read_times, s3_bytes_downloaded in tqdm(loader, desc="Training"):
        data_wait_time = time.perf_counter() - end_of_previous_batch
        total_data_wait_time += data_wait_time

        # Sum the S3 read time reported by the Dataset for all samples in this batch.
        # With multiple DataLoader workers, these reads may happen in parallel, so this
        # is cumulative S3 read time, not necessarily wall-clock batch wait time.
        total_cloud_read_time += cloud_read_times.sum().item()

        # Sum the actual S3 object payload bytes downloaded for all images in this batch.
        total_s3_bytes_downloaded += int(s3_bytes_downloaded.sum().item())

        batch_compute_start = time.perf_counter()

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

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
    avg_cloud_read_time_per_batch = total_cloud_read_time / len(loader)
    s3_bytes_downloaded_per_batch = total_s3_bytes_downloaded / len(loader)
    avg_compute_time = total_compute_time / len(loader)

    return {
        "loss": epoch_loss,
        "accuracy": epoch_acc,
        "epoch_time": epoch_time,
        "throughput": throughput,
        "avg_data_wait_time": avg_data_wait_time,
        "avg_cloud_read_time_per_batch": avg_cloud_read_time_per_batch,
        "total_s3_bytes_downloaded": total_s3_bytes_downloaded,
        "s3_bytes_downloaded_per_batch": s3_bytes_downloaded_per_batch,
        "avg_compute_time": avg_compute_time,
        "total_data_wait_time": total_data_wait_time,
        "total_cloud_read_time": total_cloud_read_time,
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
    total_cloud_read_time = 0.0
    total_s3_bytes_downloaded = 0

    eval_start_time = time.perf_counter()

    for images, labels, cloud_read_times, s3_bytes_downloaded in tqdm(loader, desc="Validation"):
        total_cloud_read_time += cloud_read_times.sum().item()
        total_s3_bytes_downloaded += int(s3_bytes_downloaded.sum().item())

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
    avg_cloud_read_time_per_batch = total_cloud_read_time / len(loader)
    s3_bytes_downloaded_per_batch = total_s3_bytes_downloaded / len(loader)

    return {
        "loss": epoch_loss,
        "accuracy": epoch_acc,
        "eval_time": eval_time,
        "throughput": throughput,
        "avg_cloud_read_time_per_batch": avg_cloud_read_time_per_batch,
        "total_cloud_read_time": total_cloud_read_time,
        "total_s3_bytes_downloaded": total_s3_bytes_downloaded,
        "s3_bytes_downloaded_per_batch": s3_bytes_downloaded_per_batch,
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
            "dataset_storage": "s3_streaming",
            "s3_data_uri": S3_DATA_URI,
            "s3_output_uri": S3_OUTPUT_URI,
            "num_classes": NUM_CLASSES,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LR,
            "optimizer": "AdamW",
            "weight_decay": WEIGHT_DECAY,
            "loss": "CrossEntropyLoss",
            "label_smoothing": LABEL_SMOOTHING,
            "scheduler": "CosineAnnealingLR",
            "scheduler_t_max": EPOCHS,
            "scheduler_eta_min": 1e-6,
            "num_workers": NUM_WORKERS,
            "image_size": IMAGE_SIZE,
            "early_stopping_patience": "disabled",
            "min_delta": MIN_DELTA,
            "device": device.type,
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

        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

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

        total_s3_bytes_downloaded_cumulative = 0

        print(f"CSV results will be saved locally to: {RESULTS_CSV_PATH}")
        print(f"S3 outputs will be uploaded to: {S3_OUTPUT_URI}")
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

            epoch_s3_bytes_downloaded = (
                train_metrics["total_s3_bytes_downloaded"]
                + val_metrics["total_s3_bytes_downloaded"]
            )
            total_s3_bytes_downloaded_cumulative += epoch_s3_bytes_downloaded

            print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")
            print(f"Val loss:   {val_loss:.4f} | Val acc:   {val_acc:.4f}")
            print(f"Training time per epoch: {train_metrics['epoch_time']:.2f} seconds")
            print(f"Training throughput: {train_metrics['throughput']:.2f} samples/sec")
            print(f"Average data loading latency: {train_metrics['avg_data_wait_time']:.4f} seconds/batch")
            print(f"Average S3 cloud read time: {train_metrics['avg_cloud_read_time_per_batch']:.4f} seconds/batch")
            print(f"S3 bytes downloaded this training epoch: {train_metrics['total_s3_bytes_downloaded']}")
            print(f"S3 bytes downloaded per training batch: {train_metrics['s3_bytes_downloaded_per_batch']:.2f}")
            print(f"Average compute time: {train_metrics['avg_compute_time']:.4f} seconds/batch")
            print(f"Validation time: {val_metrics['eval_time']:.2f} seconds")
            print(f"Validation throughput: {val_metrics['throughput']:.2f} samples/sec")
            print(f"Validation average S3 cloud read time: {val_metrics['avg_cloud_read_time_per_batch']:.4f} seconds/batch")
            print(f"S3 bytes downloaded this validation epoch: {val_metrics['total_s3_bytes_downloaded']}")
            print(f"S3 bytes downloaded per validation batch: {val_metrics['s3_bytes_downloaded_per_batch']:.2f}")
            print(f"Total S3 bytes downloaded cumulative: {total_s3_bytes_downloaded_cumulative}")
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
                print("No validation improvement. Early stopping is disabled for this 3-epoch run.")

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "training_time_seconds": train_metrics["epoch_time"],
                    "training_throughput_samples_per_sec": train_metrics["throughput"],
                    "avg_data_loading_latency_seconds_per_batch": train_metrics["avg_data_wait_time"],
                    "cloud_read_time_per_batch": train_metrics["avg_cloud_read_time_per_batch"],
                    "total_cloud_read_time_seconds": train_metrics["total_cloud_read_time"],
                    "s3_bytes_downloaded_per_epoch": train_metrics["total_s3_bytes_downloaded"],
                    "s3_bytes_downloaded_per_batch": train_metrics["s3_bytes_downloaded_per_batch"],
                    "s3_megabytes_downloaded_per_epoch": train_metrics["total_s3_bytes_downloaded"] / (1024 * 1024),
                    "s3_megabytes_downloaded_per_batch": train_metrics["s3_bytes_downloaded_per_batch"] / (1024 * 1024),
                    "validation_s3_bytes_downloaded_per_epoch": val_metrics["total_s3_bytes_downloaded"],
                    "validation_s3_bytes_downloaded_per_batch": val_metrics["s3_bytes_downloaded_per_batch"],
                    "validation_s3_megabytes_downloaded_per_epoch": val_metrics["total_s3_bytes_downloaded"] / (1024 * 1024),
                    "validation_s3_megabytes_downloaded_per_batch": val_metrics["s3_bytes_downloaded_per_batch"] / (1024 * 1024),
                    "total_s3_bytes_downloaded_cumulative": total_s3_bytes_downloaded_cumulative,
                    "total_s3_megabytes_downloaded_cumulative": total_s3_bytes_downloaded_cumulative / (1024 * 1024),
                    "avg_compute_time_seconds_per_batch": train_metrics["avg_compute_time"],
                    "validation_time_seconds": val_metrics["eval_time"],
                    "validation_throughput_samples_per_sec": val_metrics["throughput"],
                    "validation_cloud_read_time_per_batch": val_metrics["avg_cloud_read_time_per_batch"],
                    "validation_total_cloud_read_time_seconds": val_metrics["total_cloud_read_time"],
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
                learning_rate=current_lr,
                total_s3_bytes_downloaded_cumulative=total_s3_bytes_downloaded_cumulative,
            )

            # Upload after every epoch so partial results survive interruption.
            upload_outputs_to_s3()

            scheduler.step()

            # Early stopping disabled for this fixed 3-epoch run.
            # if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            #     print("\nEarly stopping triggered.")
            #     print(f"Best validation accuracy: {best_val_acc:.4f}")
            #     print(f"Best model saved at: {CHECKPOINT_PATH}")
            #     break

        mlflow.log_metric("final_best_val_accuracy", best_val_acc)
        mlflow.log_metric("final_total_s3_bytes_downloaded", total_s3_bytes_downloaded_cumulative)
        mlflow.log_metric("final_total_s3_megabytes_downloaded", total_s3_bytes_downloaded_cumulative / (1024 * 1024))
        mlflow.log_artifact(str(RESULTS_CSV_PATH), artifact_path="training_outputs")

        if CHECKPOINT_PATH.exists():
            mlflow.log_artifact(str(CHECKPOINT_PATH), artifact_path="checkpoints")

        upload_outputs_to_s3()

        print("\nTraining finished.")
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        print(f"Epoch metrics saved locally to: {RESULTS_CSV_PATH}")

        if S3_OUTPUT_URI:
            print(f"Outputs uploaded to: {S3_OUTPUT_URI}")

        print("MLflow run logged to Databricks.")


if __name__ == "__main__":
    main()
