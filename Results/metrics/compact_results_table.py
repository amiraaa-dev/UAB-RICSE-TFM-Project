#!/usr/bin/env python3
"""
Create a compact thesis results table from aggregated_run_summary_post_warmup.csv.

Output statistics:
- Mean epoch time: mean across 30 runs with a two-sided 95% Student-t CI.
- Training throughput: mean across 30 runs with a two-sided 95% Student-t CI.
- Data-loading latency: mean across 30 runs with a two-sided 95% Student-t CI.
- Mean training-phase GPU utilization: mean across 30 runs with a 95% Student-t CI.
- Median training-phase GPU utilization: median of the 30 run-level medians
  with a percentile-bootstrap 95% CI.
- GPU idle ratio: mean across 30 runs with a 95% Student-t CI.
- End-to-end time: mean across 30 runs with a 95% Student-t CI, converted to minutes.
- Final validation accuracy: mean across 30 runs with a 95% Student-t CI,
  converted to percentage points.

The script creates:
1. compact_results_table_numeric.csv
   - Separate estimate, lower-CI, upper-CI, SD, and n columns.
2. compact_results_table_formatted.csv
   - One thesis-ready row per scenario, with values formatted as:
     estimate [95% CI lower, upper]

Example:
    python compact_results_table.py
or:
    python compact_results_table.py --input "aggregated_run_summary_post_warmup(1).csv"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats


SCENARIO_LABELS = {
    "A": "A – Private",
    "B": "B – Hybrid streaming",
    "C": "C – Optimized hybrid",
}

# Display name, source column, unit conversion, decimals, statistic type.
METRICS = [
    {
        "name": "Mean epoch time (s)",
        "column": "mean_epoch_training_time_seconds",
        "scale": 1.0,
        "decimals": 2,
        "statistic": "mean",
    },
    {
        "name": "Training throughput (samples/s)",
        "column": "mean_training_throughput_samples_sec",
        "scale": 1.0,
        "decimals": 2,
        "statistic": "mean",
    },
    {
        "name": "Data-loading latency (s/batch)",
        "column": "mean_data_loading_latency_sec_batch",
        "scale": 1.0,
        "decimals": 4,
        "statistic": "mean",
    },
    {
        "name": "Mean GPU utilization (%)",
        "column": "mean_gpu_utilization_training_phase_percent",
        "scale": 1.0,
        "decimals": 2,
        "statistic": "mean",
    },
    {
        "name": "Median GPU utilization (%)",
        "column": "median_gpu_utilization_training_phase_percent",
        "scale": 1.0,
        "decimals": 2,
        "statistic": "median",
    },
    {
        "name": "GPU idle ratio (%)",
        "column": "gpu_idle_ratio_training_phase_percent",
        "scale": 1.0,
        "decimals": 2,
        "statistic": "mean",
    },
    {
        "name": "End-to-end time (min)",
        "column": "end_to_end_time_seconds",
        "scale": 1.0 / 60.0,
        "decimals": 2,
        "statistic": "mean",
    },
    {
        "name": "Final validation accuracy (%)",
        "column": "final_val_accuracy",
        "scale": 100.0,
        "decimals": 2,
        "statistic": "mean",
    },
]


def mean_t_confidence_interval(
    values: np.ndarray,
    confidence: float = 0.95,
) -> dict[str, float]:
    """
    Calculate a mean and two-sided Student-t confidence interval.

    The experimental run is treated as the independent unit of analysis.
    """
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    n = clean.size

    if n == 0:
        return {
            "estimate": np.nan,
            "lower_ci": np.nan,
            "upper_ci": np.nan,
            "ci_half_width": np.nan,
            "sd": np.nan,
            "q1": np.nan,
            "q3": np.nan,
            "n": 0,
        }

    estimate = float(np.mean(clean))
    sd = float(np.std(clean, ddof=1)) if n > 1 else 0.0

    if n > 1:
        standard_error = sd / np.sqrt(n)
        critical_value = float(
            stats.t.ppf((1.0 + confidence) / 2.0, df=n - 1)
        )
        half_width = critical_value * standard_error
        lower_ci = estimate - half_width
        upper_ci = estimate + half_width
    else:
        half_width = np.nan
        lower_ci = np.nan
        upper_ci = np.nan

    return {
        "estimate": estimate,
        "lower_ci": float(lower_ci),
        "upper_ci": float(upper_ci),
        "ci_half_width": float(half_width),
        "sd": sd,
        "q1": float(np.percentile(clean, 25)),
        "q3": float(np.percentile(clean, 75)),
        "n": int(n),
    }


def median_bootstrap_confidence_interval(
    values: np.ndarray,
    confidence: float = 0.95,
    bootstrap_samples: int = 20_000,
    seed: int = 42,
) -> dict[str, float]:
    """
    Calculate the median and a percentile-bootstrap confidence interval.

    The input values are the 30 run-level median GPU-utilization values.
    """
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    n = clean.size

    if n == 0:
        return {
            "estimate": np.nan,
            "lower_ci": np.nan,
            "upper_ci": np.nan,
            "ci_half_width": np.nan,
            "sd": np.nan,
            "q1": np.nan,
            "q3": np.nan,
            "n": 0,
        }

    estimate = float(np.median(clean))
    q1 = float(np.percentile(clean, 25))
    q3 = float(np.percentile(clean, 75))
    sd = float(np.std(clean, ddof=1)) if n > 1 else 0.0

    if n > 1:
        rng = np.random.default_rng(seed)
        samples = rng.choice(
            clean,
            size=(bootstrap_samples, n),
            replace=True,
        )
        bootstrap_medians = np.median(samples, axis=1)

        alpha = 1.0 - confidence
        lower_ci = float(np.quantile(bootstrap_medians, alpha / 2.0))
        upper_ci = float(np.quantile(bootstrap_medians, 1.0 - alpha / 2.0))
        half_width = (upper_ci - lower_ci) / 2.0
    else:
        lower_ci = np.nan
        upper_ci = np.nan
        half_width = np.nan

    return {
        "estimate": estimate,
        "lower_ci": lower_ci,
        "upper_ci": upper_ci,
        "ci_half_width": float(half_width),
        "sd": sd,
        "q1": q1,
        "q3": q3,
        "n": int(n),
    }


def find_default_input(directory: Path) -> Path:
    """Find the most likely aggregated post-warmup CSV in a directory."""
    exact = directory / "aggregated_run_summary_post_warmup.csv"
    if exact.exists():
        return exact

    matches = sorted(
        directory.glob("aggregated_run_summary_post_warmup*.csv")
    )
    if len(matches) == 1:
        return matches[0]

    if not matches:
        raise FileNotFoundError(
            "No file matching aggregated_run_summary_post_warmup*.csv "
            f"was found in {directory}"
        )

    raise FileNotFoundError(
        "More than one matching input file was found. "
        "Use --input to specify the required file:\n"
        + "\n".join(f"  {path.name}" for path in matches)
    )


def validate_input(df: pd.DataFrame) -> None:
    required_columns = {
        "scenario",
        *[metric["column"] for metric in METRICS],
    }
    missing = sorted(required_columns.difference(df.columns))

    if missing:
        raise ValueError(
            "The input CSV is missing required columns:\n"
            + "\n".join(f"  {column}" for column in missing)
        )

    scenarios = set(df["scenario"].dropna().astype(str).str.upper())
    missing_scenarios = {"A", "B", "C"}.difference(scenarios)

    if missing_scenarios:
        raise ValueError(
            "The input CSV is missing these scenarios: "
            + ", ".join(sorted(missing_scenarios))
        )


def build_tables(
    df: pd.DataFrame,
    confidence: float,
    bootstrap_samples: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["scenario"] = df["scenario"].astype(str).str.upper().str.strip()

    numeric_rows: list[dict] = []
    formatted_rows: list[dict] = []

    for scenario in ["A", "B", "C"]:
        group = df[df["scenario"] == scenario]

        formatted_row = {
            "Scenario": SCENARIO_LABELS[scenario],
        }

        for metric in METRICS:
            raw_values = pd.to_numeric(
                group[metric["column"]],
                errors="coerce",
            ).to_numpy(dtype=float)

            converted_values = raw_values * metric["scale"]

            if metric["statistic"] == "median":
                result = median_bootstrap_confidence_interval(
                    converted_values,
                    confidence=confidence,
                    bootstrap_samples=bootstrap_samples,
                    seed=seed,
                )
                method = "median with percentile-bootstrap CI"
            else:
                result = mean_t_confidence_interval(
                    converted_values,
                    confidence=confidence,
                )
                method = "mean with Student-t CI"

            numeric_rows.append(
                {
                    "scenario": scenario,
                    "scenario_label": SCENARIO_LABELS[scenario],
                    "metric": metric["name"],
                    "source_column": metric["column"],
                    "statistic": metric["statistic"],
                    "ci_method": method,
                    "estimate": result["estimate"],
                    "lower_ci": result["lower_ci"],
                    "upper_ci": result["upper_ci"],
                    "ci_half_width": result["ci_half_width"],
                    "sd": result["sd"],
                    "q1": result["q1"],
                    "q3": result["q3"],
                    "n": result["n"],
                }
            )

            decimals = metric["decimals"]
            estimate = result["estimate"]
            lower = result["lower_ci"]
            upper = result["upper_ci"]

            if np.isfinite(estimate) and np.isfinite(lower) and np.isfinite(upper):
                formatted_row[metric["name"]] = (
                    f"{estimate:.{decimals}f} "
                    f"[{lower:.{decimals}f}, {upper:.{decimals}f}]"
                )
            elif np.isfinite(estimate):
                formatted_row[metric["name"]] = (
                    f"{estimate:.{decimals}f}"
                )
            else:
                formatted_row[metric["name"]] = "NA"

        formatted_rows.append(formatted_row)

    numeric_table = pd.DataFrame(numeric_rows)
    formatted_table = pd.DataFrame(formatted_rows)

    return numeric_table, formatted_table


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate scenario-level estimates and 95% confidence "
            "intervals for the compact thesis results table."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "Path to aggregated_run_summary_post_warmup.csv. "
            "If omitted, the script searches its own directory."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for output CSV files. "
            "Default: the input file's directory."
        ),
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level between 0 and 1 (default: 0.95).",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=20_000,
        help=(
            "Number of bootstrap resamples for the median CI "
            "(default: 20000)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the bootstrap median CI (default: 42).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    if not 0.0 < args.confidence < 1.0:
        raise ValueError("--confidence must be between 0 and 1.")

    if args.bootstrap_samples < 1_000:
        raise ValueError("--bootstrap-samples should be at least 1000.")

    script_directory = Path(__file__).resolve().parent

    input_path = (
        args.input.expanduser().resolve()
        if args.input is not None
        else find_default_input(script_directory)
    )

    output_directory = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else input_path.parent
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, low_memory=False)
    validate_input(df)

    counts = (
        df.assign(
            scenario=df["scenario"].astype(str).str.upper().str.strip()
        )
        .groupby("scenario")
        .size()
        .reindex(["A", "B", "C"], fill_value=0)
    )

    print(f"Input: {input_path}")
    print("Runs per scenario:")
    print(counts.to_string())

    if not all(counts == 30):
        print(
            "WARNING: The thesis design expects 30 runs per scenario, "
            "but the counts above differ."
        )

    numeric_table, formatted_table = build_tables(
        df=df,
        confidence=args.confidence,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )

    numeric_output = output_directory / "compact_results_table_numeric.csv"
    formatted_output = output_directory / "compact_results_table_formatted.csv"

    numeric_table.to_csv(numeric_output, index=False)
    formatted_table.to_csv(formatted_output, index=False)

    confidence_percent = args.confidence * 100

    print()
    print(f"Created: {numeric_output}")
    print(f"Created: {formatted_output}")
    print()
    print(
        f"Values below are estimate [{confidence_percent:.0f}% CI lower, upper]."
    )
    print(
        "All metrics use a mean with a Student-t CI except "
        "Median GPU utilization, which uses a median with a "
        "percentile-bootstrap CI."
    )
    print()
    print(formatted_table.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
