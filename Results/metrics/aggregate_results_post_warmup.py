#!/usr/bin/env python3
"""
Aggregate 30 runs for Scenarios A, B, and C into one run-level CSV.

Key GPU rule:
- Scenario A/B: GPU comparison metrics are calculated from the first to the
  last GPU sample with utilization >= the activity threshold.
- Scenario C: the cache-warmup interval is removed first. The comparison
  window is then calculated from the first to the last active GPU sample
  after warmup.

This avoids understating Scenario C GPU utilization by averaging the long
cache-warmup period into the training-phase result.

Expected folder layout:
    Results/
        aggregate_results_post_warmup.py
        run_1_A/
        ...
        run_30_A/
        run_1_B/
        ...
        run_30_B/
        run_1_C/
        ...
        run_30_C/

Each run folder should contain:
    training_results.csv
    gpu_usage_log.csv
    docker_resource_log.csv
    system_resource_log.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


DEFAULT_OUTPUT_NAME = "aggregated_run_summary_post_warmup.csv"
DEFAULT_GPU_ACTIVITY_THRESHOLD = 20.0


# ---------------------------------------------------------------------------
# General cleaning helpers
# ---------------------------------------------------------------------------

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names while preserving punctuation used by nvidia-smi."""
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )
    return df


def to_numeric(series: pd.Series) -> pd.Series:
    """Convert a Series to numbers; malformed values become NaN."""
    return pd.to_numeric(series, errors="coerce")


def clean_percent(series: pd.Series) -> pd.Series:
    """Convert values such as '97%' or '97 %' to 97.0."""
    cleaned = (
        series.astype(str)
        .str.replace("%", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def clean_number_with_units(series: pd.Series) -> pd.Series:
    """Convert values such as '24564 MiB' to 24564.0."""
    extracted = series.astype(str).str.extract(r"([-+]?[0-9]*\.?[0-9]+)")[0]
    return pd.to_numeric(extracted, errors="coerce")


def first_existing_column(
    df: pd.DataFrame,
    possible_names: Iterable[str],
) -> Optional[str]:
    """Return the first candidate column that exists."""
    for name in possible_names:
        if name in df.columns:
            return name
    return None


def numeric_values(
    df: pd.DataFrame,
    possible_columns: Iterable[str],
) -> pd.Series:
    """Return a cleaned numeric Series for the first matching column."""
    column = first_existing_column(df, possible_columns)
    if column is None:
        return pd.Series(dtype=float)
    return to_numeric(df[column]).dropna()


def safe_mean(df: pd.DataFrame, possible_columns: Iterable[str]) -> float:
    values = numeric_values(df, possible_columns)
    return float(values.mean()) if not values.empty else np.nan


def safe_sum(df: pd.DataFrame, possible_columns: Iterable[str]) -> float:
    values = numeric_values(df, possible_columns)
    return float(values.sum()) if not values.empty else np.nan


def safe_first(df: pd.DataFrame, possible_columns: Iterable[str]) -> float:
    values = numeric_values(df, possible_columns)
    return float(values.iloc[0]) if not values.empty else np.nan


def safe_last(df: pd.DataFrame, possible_columns: Iterable[str]) -> float:
    values = numeric_values(df, possible_columns)
    return float(values.iloc[-1]) if not values.empty else np.nan


def safe_max(df: pd.DataFrame, possible_columns: Iterable[str]) -> float:
    values = numeric_values(df, possible_columns)
    return float(values.max()) if not values.empty else np.nan


def read_csv_safely(file_path: Path) -> Optional[pd.DataFrame]:
    """Read a normal CSV without stopping the full aggregation on one bad file."""
    try:
        df = pd.read_csv(file_path, low_memory=False)
        return normalize_columns(df)
    except Exception as exc:
        print(f"WARNING: Could not read {file_path}: {exc}")
        return None


def read_system_csv_safely(file_path: Path) -> Optional[pd.DataFrame]:
    """
    Read system_resource_log.csv.

    Some versions of the logging script write an extra CPU-count field after
    cpu_idle_percent but omit cpu_count from the header. Standard pandas
    parsing then shifts timestamp and CPU columns. This reader detects that
    layout and inserts the missing header so values remain correctly aligned.
    """
    try:
        with file_path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            raw_header = next(reader)
            raw_rows = [
                row for row in reader
                if row and any(str(cell).strip() for cell in row)
            ]
    except Exception as exc:
        print(f"WARNING: Could not read {file_path}: {exc}")
        return None

    header = [str(value).strip().lower().replace(" ", "_") for value in raw_header]

    # Remove repeated header rows that can appear if logging was restarted.
    filtered_rows = []
    for row in raw_rows:
        if row and str(row[0]).strip().lower() == "timestamp":
            continue
        filtered_rows.append(row)
    raw_rows = filtered_rows

    if not raw_rows:
        return pd.DataFrame(columns=header)

    row_lengths = pd.Series([len(row) for row in raw_rows])
    most_common_length = int(row_lengths.mode().iloc[0])

    # Known layout issue:
    # timestamp,cpu_used,cpu_idle,cpu_count,mem_total,...
    # while header omits cpu_count.
    if (
        most_common_length == len(header) + 1
        and "cpu_idle_percent" in header
        and "cpu_count" not in header
    ):
        insert_at = header.index("cpu_idle_percent") + 1
        header.insert(insert_at, "cpu_count")

    cleaned_rows = []
    for row in raw_rows:
        row = list(row)

        if len(row) < len(header):
            row.extend([""] * (len(header) - len(row)))
        elif len(row) > len(header):
            # Preserve all expected fields and ignore unexplained trailing data.
            row = row[: len(header)]

        cleaned_rows.append(row)

    try:
        df = pd.DataFrame(cleaned_rows, columns=header)
        return normalize_columns(df)
    except Exception as exc:
        print(f"WARNING: Could not structure {file_path}: {exc}")
        return None


# ---------------------------------------------------------------------------
# GPU phase helpers
# ---------------------------------------------------------------------------

def calculate_gpu_metrics(
    gpu_data: pd.DataFrame,
    prefix: str,
    idle_threshold: float,
) -> dict:
    """Calculate mean, median, idle ratio, count, and duration for one window."""
    result: dict = {}

    if gpu_data.empty:
        result[f"mean_gpu_utilization_{prefix}_percent"] = np.nan
        result[f"median_gpu_utilization_{prefix}_percent"] = np.nan
        result[f"gpu_idle_ratio_{prefix}_percent"] = np.nan
        result[f"gpu_{prefix}_sample_count"] = 0
        result[f"gpu_{prefix}_duration_seconds"] = np.nan
        return result

    values = gpu_data["gpu_utilization_percent"].dropna()

    result[f"mean_gpu_utilization_{prefix}_percent"] = float(values.mean())
    result[f"median_gpu_utilization_{prefix}_percent"] = float(values.median())
    result[f"gpu_idle_ratio_{prefix}_percent"] = float(
        (values < idle_threshold).mean() * 100
    )
    result[f"gpu_{prefix}_sample_count"] = int(len(values))

    start_time = gpu_data["timestamp"].min()
    end_time = gpu_data["timestamp"].max()
    result[f"gpu_{prefix}_duration_seconds"] = float(
        (end_time - start_time).total_seconds()
    )

    return result


def select_gpu_training_phase(
    candidate_data: pd.DataFrame,
    activity_threshold: float,
) -> tuple[pd.DataFrame, str]:
    """
    Select a comparable GPU execution window.

    The window starts at the first sample >= activity_threshold and ends at
    the last sample >= activity_threshold. All samples between those points
    are retained, including low-utilization waiting time. This is important
    for Scenario B because S3 loading can leave the GPU idle between compute
    bursts.

    If no active sample is found, the full candidate window is returned and
    the status records the fallback.
    """
    if candidate_data.empty:
        return candidate_data.copy(), "empty_candidate_window"

    candidate_data = (
        candidate_data.sort_values("timestamp")
        .reset_index(drop=True)
    )

    active = candidate_data[
        candidate_data["gpu_utilization_percent"] >= activity_threshold
    ]

    if active.empty:
        return candidate_data.copy(), "fallback_no_sample_above_threshold"

    phase_start = active["timestamp"].iloc[0]
    phase_end = active["timestamp"].iloc[-1]

    phase = candidate_data[
        (candidate_data["timestamp"] >= phase_start)
        & (candidate_data["timestamp"] <= phase_end)
    ].copy()

    return phase, "first_to_last_active_sample"


# ---------------------------------------------------------------------------
# Main run aggregation
# ---------------------------------------------------------------------------

def aggregate_run(
    run_dir: Path,
    scenario: str,
    run_id: int,
    gpu_activity_threshold: float,
) -> Optional[dict]:
    training_file = run_dir / "training_results.csv"
    gpu_file = run_dir / "gpu_usage_log.csv"
    docker_file = run_dir / "docker_resource_log.csv"
    system_file = run_dir / "system_resource_log.csv"

    if not training_file.exists():
        print(f"WARNING: Skipping {run_dir.name}: no training_results.csv")
        return None

    train = read_csv_safely(training_file)
    if train is None or train.empty:
        print(
            f"WARNING: Skipping {run_dir.name}: "
            "training_results.csv is empty or unreadable"
        )
        return None

    row: dict = {
        "scenario": scenario,
        "run_id": run_id,
        "run_folder": run_dir.name,

        # Training performance
        "mean_epoch_training_time_seconds": safe_mean(
            train,
            [
                "training_time_seconds",
                "train_time_seconds",
                "epoch_training_time_seconds",
            ],
        ),
        "total_training_time_seconds": safe_sum(
            train,
            [
                "training_time_seconds",
                "train_time_seconds",
                "epoch_training_time_seconds",
            ],
        ),
        "mean_training_throughput_samples_sec": safe_mean(
            train,
            [
                "training_throughput_samples_per_sec",
                "train_throughput_samples_per_sec",
            ],
        ),
        "mean_data_loading_latency_sec_batch": safe_mean(
            train,
            [
                "avg_data_loading_latency_seconds_per_batch",
                "avg_data_loading_latency_sec_batch",
                "data_loading_latency_seconds_per_batch",
            ],
        ),
        "mean_compute_time_sec_batch": safe_mean(
            train,
            [
                "avg_compute_time_seconds_per_batch",
                "avg_compute_time_sec_batch",
                "compute_time_seconds_per_batch",
            ],
        ),

        # Validation/model metrics
        "total_validation_time_seconds": safe_sum(
            train,
            ["validation_time_seconds", "val_time_seconds"],
        ),
        "mean_validation_time_seconds": safe_mean(
            train,
            ["validation_time_seconds", "val_time_seconds"],
        ),
        "mean_validation_throughput_samples_sec": safe_mean(
            train,
            [
                "validation_throughput_samples_per_sec",
                "val_throughput_samples_per_sec",
            ],
        ),
        "final_val_accuracy": safe_last(
            train,
            ["val_accuracy", "validation_accuracy"],
        ),
        "final_val_loss": safe_last(
            train,
            ["val_loss", "validation_loss"],
        ),
        "best_val_accuracy": safe_max(
            train,
            ["best_val_accuracy", "best_validation_accuracy"],
        ),
    }

    # S3 metrics measured during epochs
    cloud_read_sum = safe_sum(
        train,
        [
            "total_cloud_read_time_seconds",
            "cloud_read_time_seconds",
            "total_s3_read_time_seconds",
        ],
    )
    row["total_cloud_read_time_seconds"] = (
        0.0 if pd.isna(cloud_read_sum) else cloud_read_sum
    )

    epoch_s3_mb = safe_sum(
        train,
        [
            "s3_megabytes_downloaded_per_epoch",
            "s3_mb_downloaded_per_epoch",
            "megabytes_downloaded_per_epoch",
            "s3_megabytes_downloaded",
        ],
    )
    row["total_s3_mb_downloaded_during_epochs"] = (
        0.0 if pd.isna(epoch_s3_mb) else epoch_s3_mb
    )

    row["mean_cloud_read_time_sec_batch"] = safe_mean(
        train,
        [
            "avg_cloud_read_time_seconds_per_batch",
            "avg_cloud_read_time_sec_batch",
            "avg_s3_read_time_seconds_per_batch",
        ],
    )

    # Scenario C cache-warmup metrics.
    # These are run-level values repeated on every epoch row, so take the first
    # non-null value rather than summing them.
    cache_warmup = safe_first(
        train,
        ["cache_warmup_time_seconds", "warmup_time_seconds"],
    )
    row["cache_warmup_time_seconds"] = (
        0.0 if pd.isna(cache_warmup) else cache_warmup
    )

    warmup_bytes = safe_first(
        train,
        [
            "cache_warmup_s3_bytes_downloaded",
            "warmup_s3_bytes",
            "cache_warmup_s3_bytes",
        ],
    )
    row["cache_warmup_s3_bytes_downloaded"] = (
        0.0 if pd.isna(warmup_bytes) else warmup_bytes
    )

    warmup_mb = safe_first(
        train,
        [
            "cache_warmup_s3_megabytes_downloaded",
            "warmup_s3_megabytes",
        ],
    )
    if pd.isna(warmup_mb) and row["cache_warmup_s3_bytes_downloaded"] > 0:
        warmup_mb = row["cache_warmup_s3_bytes_downloaded"] / (1024 ** 2)
    row["cache_warmup_s3_megabytes_downloaded"] = (
        0.0 if pd.isna(warmup_mb) else warmup_mb
    )

    warmup_files = safe_first(
        train,
        [
            "cache_warmup_files_downloaded",
            "cache_files_downloaded",
            "warmup_files_downloaded",
        ],
    )
    row["cache_warmup_files_downloaded"] = (
        0.0 if pd.isna(warmup_files) else warmup_files
    )

    row["cache_warmup_files_already_cached"] = safe_first(
        train,
        ["cache_warmup_files_already_cached"],
    )
    row["train_cache_hit_ratio"] = safe_mean(
        train,
        [
            "train_cache_hit_ratio",
            "cache_hit_ratio_train",
            "cache_hit_ratio",
        ],
    )
    row["val_cache_hit_ratio"] = safe_mean(
        train,
        [
            "val_cache_hit_ratio",
            "validation_cache_hit_ratio",
        ],
    )
    row["cache_hits"] = safe_sum(
        train,
        ["train_cache_hits", "cache_hits"],
    )
    row["cache_misses"] = safe_sum(
        train,
        ["train_cache_misses", "cache_misses"],
    )
    row["mean_cache_read_time_sec_batch"] = safe_mean(
        train,
        [
            "avg_cache_read_time_seconds_per_batch",
            "cache_read_time_seconds_per_batch",
        ],
    )
    row["validation_mean_cache_read_time_sec_batch"] = safe_mean(
        train,
        [
            "validation_avg_cache_read_time_seconds_per_batch",
            "val_avg_cache_read_time_seconds_per_batch",
        ],
    )

    row["total_s3_mb_downloaded_including_warmup"] = (
        row["total_s3_mb_downloaded_during_epochs"]
        + row["cache_warmup_s3_megabytes_downloaded"]
    )

    training_total = row["total_training_time_seconds"]
    validation_total = row["total_validation_time_seconds"]

    row["end_to_end_time_seconds"] = (
        (0.0 if pd.isna(training_total) else training_total)
        + (0.0 if pd.isna(validation_total) else validation_total)
        + row["cache_warmup_time_seconds"]
    )

    # -----------------------------------------------------------------------
    # GPU metrics
    # -----------------------------------------------------------------------
    if gpu_file.exists():
        gpu = read_csv_safely(gpu_file)

        if gpu is not None and not gpu.empty:
            timestamp_col = first_existing_column(
                gpu,
                ["timestamp", "time", "datetime"],
            )
            gpu_util_col = first_existing_column(
                gpu,
                [
                    "utilization.gpu_[%]",
                    "utilization.gpu",
                    "gpu_utilization",
                    "gpu_utilization_percent",
                    "utilization_gpu_percent",
                ],
            )
            gpu_mem_col = first_existing_column(
                gpu,
                [
                    "memory.used_[mib]",
                    "memory.used",
                    "gpu_memory_used_mib",
                    "memory_used_mib",
                ],
            )

            if timestamp_col is not None and gpu_util_col is not None:
                timestamps = pd.to_datetime(
                    gpu[timestamp_col],
                    errors="coerce",
                )
                utilization = clean_percent(gpu[gpu_util_col])

                gpu_data = pd.DataFrame(
                    {
                        "timestamp": timestamps,
                        "gpu_utilization_percent": utilization,
                    }
                )

                row["invalid_gpu_utilization_rows"] = int(
                    gpu_data[
                        ["timestamp", "gpu_utilization_percent"]
                    ].isna().any(axis=1).sum()
                )

                gpu_data = (
                    gpu_data.dropna(
                        subset=["timestamp", "gpu_utilization_percent"]
                    )
                    .sort_values("timestamp")
                    .reset_index(drop=True)
                )

                if not gpu_data.empty:
                    # Full-log metrics retained for diagnostics only.
                    full_metrics = calculate_gpu_metrics(
                        gpu_data,
                        prefix="full_log",
                        idle_threshold=gpu_activity_threshold,
                    )
                    row.update(full_metrics)

                    # Preserve the original output names for compatibility.
                    row["mean_gpu_utilization_percent"] = row[
                        "mean_gpu_utilization_full_log_percent"
                    ]
                    row["median_gpu_utilization_percent"] = row[
                        "median_gpu_utilization_full_log_percent"
                    ]
                    row["gpu_idle_ratio_percent"] = row[
                        "gpu_idle_ratio_full_log_percent"
                    ]

                    candidate_gpu_data = gpu_data.copy()

                    if (
                        scenario == "C"
                        and row["cache_warmup_time_seconds"] > 0
                    ):
                        gpu_log_start = gpu_data["timestamp"].min()
                        estimated_warmup_end = (
                            gpu_log_start
                            + pd.to_timedelta(
                                row["cache_warmup_time_seconds"],
                                unit="s",
                            )
                        )

                        row["estimated_cache_warmup_end"] = (
                            estimated_warmup_end.isoformat()
                        )

                        candidate_gpu_data = gpu_data[
                            gpu_data["timestamp"] >= estimated_warmup_end
                        ].copy()

                        post_warmup_metrics = calculate_gpu_metrics(
                            candidate_gpu_data,
                            prefix="post_warmup",
                            idle_threshold=gpu_activity_threshold,
                        )
                        row.update(post_warmup_metrics)

                        row["gpu_comparison_window_method"] = (
                            "after_estimated_warmup_end_then_"
                            "first_to_last_active_sample"
                        )
                    else:
                        row["gpu_comparison_window_method"] = (
                            "first_to_last_active_sample"
                        )

                    training_phase, phase_status = select_gpu_training_phase(
                        candidate_gpu_data,
                        activity_threshold=gpu_activity_threshold,
                    )

                    row["gpu_phase_detection_status"] = phase_status
                    row["gpu_activity_threshold_percent"] = (
                        gpu_activity_threshold
                    )

                    if not training_phase.empty:
                        row["gpu_training_phase_start"] = (
                            training_phase["timestamp"].min().isoformat()
                        )
                        row["gpu_training_phase_end"] = (
                            training_phase["timestamp"].max().isoformat()
                        )

                    phase_metrics = calculate_gpu_metrics(
                        training_phase,
                        prefix="training_phase",
                        idle_threshold=gpu_activity_threshold,
                    )
                    row.update(phase_metrics)

            if gpu_mem_col is not None:
                gpu_memory = clean_number_with_units(gpu[gpu_mem_col])
                row["mean_gpu_memory_used_mib"] = float(gpu_memory.mean())
                row["max_gpu_memory_used_mib"] = float(gpu_memory.max())
                row["invalid_gpu_memory_rows"] = int(
                    gpu_memory.isna().sum()
                )

    # -----------------------------------------------------------------------
    # Docker metrics
    # -----------------------------------------------------------------------
    if docker_file.exists():
        docker = read_csv_safely(docker_file)

        if docker is not None and not docker.empty:
            docker_cpu_col = first_existing_column(
                docker,
                ["cpu_percent", "cpu_%", "docker_cpu_percent"],
            )
            docker_mem_col = first_existing_column(
                docker,
                [
                    "memory_percent",
                    "mem_percent",
                    "docker_memory_percent",
                ],
            )

            if docker_cpu_col is not None:
                docker_cpu = clean_percent(docker[docker_cpu_col])
                row["mean_docker_cpu_percent"] = float(docker_cpu.mean())
                row["max_docker_cpu_percent"] = float(docker_cpu.max())
                row["invalid_docker_cpu_rows"] = int(
                    docker_cpu.isna().sum()
                )

            if docker_mem_col is not None:
                docker_mem = clean_percent(docker[docker_mem_col])
                row["mean_docker_memory_percent"] = float(docker_mem.mean())
                row["max_docker_memory_percent"] = float(docker_mem.max())
                row["invalid_docker_memory_rows"] = int(
                    docker_mem.isna().sum()
                )

    # -----------------------------------------------------------------------
    # Host-system metrics
    # -----------------------------------------------------------------------
    if system_file.exists():
        system = read_system_csv_safely(system_file)

        if system is not None and not system.empty:
            host_cpu_col = first_existing_column(
                system,
                [
                    "cpu_used_percent",
                    "host_cpu_used_percent",
                    "cpu_percent",
                ],
            )
            host_mem_col = first_existing_column(
                system,
                [
                    "mem_used_mb",
                    "memory_used_mb",
                    "host_memory_used_mb",
                ],
            )

            if host_cpu_col is not None:
                host_cpu = to_numeric(system[host_cpu_col])
                row["mean_host_cpu_used_percent"] = float(host_cpu.mean())
                row["max_host_cpu_used_percent"] = float(host_cpu.max())
                row["invalid_host_cpu_rows"] = int(host_cpu.isna().sum())

            if host_mem_col is not None:
                host_mem = to_numeric(system[host_mem_col])
                row["mean_host_memory_used_mb"] = float(host_mem.mean())
                row["max_host_memory_used_mb"] = float(host_mem.max())
                row["invalid_host_memory_rows"] = int(host_mem.isna().sum())

    return row


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate experiment runs and calculate Scenario C GPU "
            "utilization after cache warmup."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help=(
            "Folder containing run_1_A ... run_30_C. "
            "Default: the folder containing this script."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output CSV path. Default: "
            "aggregated_run_summary_post_warmup.csv inside results-dir."
        ),
    )
    parser.add_argument(
        "--gpu-threshold",
        type=float,
        default=DEFAULT_GPU_ACTIVITY_THRESHOLD,
        help=(
            "GPU utilization percentage used to identify active execution "
            f"(default: {DEFAULT_GPU_ACTIVITY_THRESHOLD})."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    results_dir = args.results_dir.expanduser().resolve()
    output_file = (
        args.output.expanduser().resolve()
        if args.output is not None
        else results_dir / DEFAULT_OUTPUT_NAME
    )

    if not results_dir.exists():
        print(f"ERROR: Results directory does not exist: {results_dir}")
        return 1

    folder_pattern = re.compile(r"^run_(\d+)_([abc])$", re.IGNORECASE)
    run_folders = []

    for item in results_dir.iterdir():
        if not item.is_dir():
            continue

        match = folder_pattern.fullmatch(item.name)
        if match:
            run_id = int(match.group(1))
            scenario = match.group(2).upper()
            run_folders.append((scenario, run_id, item))

    run_folders.sort(key=lambda value: (value[0], value[1]))

    if not run_folders:
        print("ERROR: No valid run folders were found.")
        print(f"Searched inside: {results_dir}")
        print("Expected names such as run_1_A, run_1_B, and run_1_C.")
        return 1

    rows = []

    for scenario, run_id, run_dir in run_folders:
        row = aggregate_run(
            run_dir=run_dir,
            scenario=scenario,
            run_id=run_id,
            gpu_activity_threshold=args.gpu_threshold,
        )
        if row is not None:
            rows.append(row)

    summary = pd.DataFrame(rows)

    if summary.empty:
        print("ERROR: Run folders were found, but no runs were aggregated.")
        return 1

    summary = summary.sort_values(["scenario", "run_id"])
    summary.to_csv(output_file, index=False)

    print()
    print(f"Created: {output_file}")
    print(f"Total aggregated runs: {len(summary)}")
    print()
    print("Runs per scenario:")
    print(summary["scenario"].value_counts().sort_index())

    expected_counts = {"A": 30, "B": 30, "C": 30}
    actual_counts = summary["scenario"].value_counts().to_dict()
    if any(actual_counts.get(key, 0) != value for key, value in expected_counts.items()):
        print()
        print("WARNING: Expected 30 runs for each scenario.")
        print(f"Actual counts: {actual_counts}")

    print()
    print("GPU columns recommended for the thesis comparison:")
    print("  mean_gpu_utilization_training_phase_percent")
    print("  median_gpu_utilization_training_phase_percent")
    print("  gpu_idle_ratio_training_phase_percent")

    if "gpu_phase_detection_status" in summary.columns:
        print()
        print("GPU phase detection status:")
        print(summary["gpu_phase_detection_status"].value_counts(dropna=False))

    invalid_columns = [
        column
        for column in summary.columns
        if column.startswith("invalid_")
    ]

    if invalid_columns:
        invalid_totals = (
            summary[invalid_columns]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .sum()
        )
        invalid_totals = invalid_totals[invalid_totals > 0]

        print()
        if invalid_totals.empty:
            print("No invalid numeric log rows were detected.")
        else:
            print("Invalid numeric log rows detected:")
            print(invalid_totals)

    print()
    print("Important:")
    print(
        "The legacy mean_gpu_utilization_percent column still represents "
        "the full GPU log."
    )
    print(
        "Use the *_training_phase_percent columns for the fair A/B/C "
        "comparison."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
