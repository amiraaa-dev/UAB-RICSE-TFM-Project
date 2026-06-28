#!/bin/bash

set -e

# ============================================================
# Scenario C:  Docker experiments with S3-optimization dataset
# and full monitoring.
#
# This version supports temporary AWS credentials by passing:
#   AWS_SESSION_TOKEN
#
# It:
#   - runs the configured number of Docker experiments
#   - uses S3_DATA_URI as the cloud dataset location + optimization techniques
#   - uses S3_OUTPUT_BASE_URI as the cloud output base directory
#   - creates separate local output folders for each run
#   - records GPU, Docker/container, and full-system monitoring logs
#   - uploads each run folder to S3 using boto3 inside a temporary Docker container
#  - deletes /tmp/tinyimagenet_s3_cache before each run
#
# Required environment variables:
#
#   export DATABRICKS_HOST="https://your-databricks-workspace-url"
#   export DATABRICKS_TOKEN="your-databricks-token"
#
#   export AWS_ACCESS_KEY_ID="your-aws-access-key"
#   export AWS_SECRET_ACCESS_KEY="your-aws-secret-key"
#   export AWS_DEFAULT_REGION="your-aws-region"
#
# If using temporary AWS credentials, also export:
#
#   export AWS_SESSION_TOKEN="your-aws-session-token"
#
#   export S3_DATA_URI="s3://your-bucket/path/to/tiny-imagenet-200"
#   export S3_OUTPUT_BASE_URI="s3://your-bucket/experiments/Scenario_C"
#
# Run:
#   chmod +x run_experiments_scenario_c_s3_optimization_with_full_monitoring_session_token.sh
#   ./run_experiments_scenario_c_s3_optimization_with_full_monitoring_session_token.sh
# ============================================================

EXPERIMENT_NAME="/Users/1749412@uab.cat/resnet18-tinyimagenet"
IMAGE_NAME="resnet18-tinyimagenet"

SCENARIO_NAME="Scenario_C"
LOCAL_OUTPUT_BASE="$(pwd)/outputs/${SCENARIO_NAME}"

# Host-side cache directory mounted into the training container at the
# same path used by the Scenario C Python script. The script deletes
# this directory before each run so every experiment starts with an
# empty cache and cache-warmup metrics are measured cleanly.
HOST_CACHE_DIR="/tmp/tinyimagenet_s3_cache"
CONTAINER_CACHE_DIR="/tmp/tinyimagenet_s3_cache"

# ------------------------------------------------------------
# Validate required environment variables
# ------------------------------------------------------------

required_vars=(
  "DATABRICKS_HOST"
  "DATABRICKS_TOKEN"
  "AWS_ACCESS_KEY_ID"
  "AWS_SECRET_ACCESS_KEY"
  "AWS_DEFAULT_REGION"
  "S3_DATA_URI"
  "S3_OUTPUT_BASE_URI"
)

for var_name in "${required_vars[@]}"; do
    if [ -z "${!var_name}" ]; then
        echo "ERROR: Required environment variable $var_name is not set."
        exit 1
    fi
done

# AWS_SESSION_TOKEN is optional for long-lived IAM access keys,
# but required for temporary STS credentials, which often start with ASIA.
if [[ "$AWS_ACCESS_KEY_ID" == ASIA* && -z "$AWS_SESSION_TOKEN" ]]; then
    echo "ERROR: AWS_ACCESS_KEY_ID starts with ASIA, which usually means temporary credentials."
    echo "You must also export AWS_SESSION_TOKEN."
    exit 1
fi

# ------------------------------------------------------------
# Validate required commands
# ------------------------------------------------------------

if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker is not installed or not on PATH."
    exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi is not installed or not on PATH."
    exit 1
fi

mkdir -p "$LOCAL_OUTPUT_BASE"


# ------------------------------------------------------------
# Helper: reset the Scenario C cache directory.
#
# The cache is host-mounted into the training container. Files created
# by the container may be owned by root, so a normal host-side `rm -rf`
# can fail with "Permission denied". This helper deletes the cache
# contents from inside a temporary Docker container running as root.
# ------------------------------------------------------------

reset_cache_dir() {
    echo "Resetting Scenario C cache directory: $HOST_CACHE_DIR"

    mkdir -p "$HOST_CACHE_DIR"

    docker run --rm \
      -v "$HOST_CACHE_DIR:/cache" \
      "$IMAGE_NAME" \
      /bin/sh -lc "find /cache -mindepth 1 -maxdepth 1 -exec rm -rf {} + && chown $(id -u):$(id -g) /cache"

    echo "Cache directory reset complete."
}

# ------------------------------------------------------------
# Helper: upload a local directory to S3 using boto3 in Docker
# This avoids needing AWS CLI on the host.
# ------------------------------------------------------------

upload_directory_to_s3() {
    local local_dir="$1"
    local s3_output_uri="$2"

    echo "Uploading local directory to S3 using boto3:"
    echo "  Local: $local_dir"
    echo "  S3:    $s3_output_uri"

    docker run --rm -i \
      -e AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
      -e AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
      -e AWS_SESSION_TOKEN="$AWS_SESSION_TOKEN" \
      -e AWS_DEFAULT_REGION="$AWS_DEFAULT_REGION" \
      -e LOCAL_OUTPUT_DIR="/upload" \
      -e S3_OUTPUT_URI="$s3_output_uri" \
      -v "$local_dir:/upload" \
      "$IMAGE_NAME" \
      python - <<'PY'
import os
from pathlib import Path
from urllib.parse import urlparse

import boto3

local_dir = Path(os.environ["LOCAL_OUTPUT_DIR"])
s3_uri = os.environ["S3_OUTPUT_URI"]

parsed = urlparse(s3_uri)
bucket = parsed.netloc
prefix = parsed.path.lstrip("/").rstrip("/")

if not bucket:
    raise ValueError(f"Invalid S3 URI: {s3_uri}")

s3 = boto3.client("s3")

files = [p for p in local_dir.rglob("*") if p.is_file()]
print(f"Found {len(files)} files to upload.")

for path in files:
    relative_key = path.relative_to(local_dir).as_posix()
    s3_key = f"{prefix}/{relative_key}" if prefix else relative_key
    print(f"Uploading {path} -> s3://{bucket}/{s3_key}")
    s3.upload_file(str(path), bucket, s3_key)

print("S3 upload complete.")
PY
}

# ------------------------------------------------------------
# Main loop: 2 experiments
# ------------------------------------------------------------

for i in $(seq -w 1 1)
do
    RUN_NAME="scenario-c-run-${i}"
    OUTPUT_DIR="${LOCAL_OUTPUT_BASE}/run_${i}"
    RUN_S3_OUTPUT_URI="${S3_OUTPUT_BASE_URI%/}/run_${i}"

    mkdir -p "$OUTPUT_DIR"

    GPU_LOG="$OUTPUT_DIR/gpu_usage_log.csv"
    DOCKER_LOG="$OUTPUT_DIR/docker_resource_log.csv"
    SYSTEM_LOG="$OUTPUT_DIR/system_resource_log.csv"
    STATUS_LOG="$OUTPUT_DIR/run_status.log"

    CONTAINER_NAME="resnet18_tinyimagenet_scenario_c_run_${i}"

    echo "==========================================" | tee "$STATUS_LOG"
    echo "Starting Scenario C experiment $i/2" | tee -a "$STATUS_LOG"
    echo "MLflow run name: $RUN_NAME" | tee -a "$STATUS_LOG"
    echo "Local output folder: $OUTPUT_DIR" | tee -a "$STATUS_LOG"
    echo "S3 dataset URI: $S3_DATA_URI" | tee -a "$STATUS_LOG"
    echo "S3 output URI: $RUN_S3_OUTPUT_URI" | tee -a "$STATUS_LOG"
    echo "Container name: $CONTAINER_NAME" | tee -a "$STATUS_LOG"
    echo "AWS access key prefix: ${AWS_ACCESS_KEY_ID:0:4}" | tee -a "$STATUS_LOG"

    if [ -n "$AWS_SESSION_TOKEN" ]; then
        echo "AWS session token: present" | tee -a "$STATUS_LOG"
    else
        echo "AWS session token: not set" | tee -a "$STATUS_LOG"
    fi

    echo "==========================================" | tee -a "$STATUS_LOG"

    # -----------------------------------------
    # Reset Scenario C local cache before this run.
    # The cache is mounted into the container at /tmp/tinyimagenet_s3_cache,
    # which is the path used by the Scenario C training script.
    # This guarantees that each run starts from an empty cache.
    # -----------------------------------------
    echo "Deleting local cache before run: $HOST_CACHE_DIR" | tee -a "$STATUS_LOG"
    reset_cache_dir 2>&1 | tee -a "$STATUS_LOG"

    # -----------------------------------------
    # GPU monitoring: whole GPU
    # nvidia-smi writes its own CSV header.
    # -----------------------------------------
    nvidia-smi \
      --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
      --format=csv \
      -l 1 > "$GPU_LOG" &

    GPU_MONITOR_PID=$!

    # -----------------------------------------
    # System monitoring: full host CPU and memory
    # -----------------------------------------
    echo "timestamp,cpu_used_percent,cpu_idle_percent,mem_total_mb,mem_used_mb,mem_free_mb,mem_available_mb,swap_total_mb,swap_used_mb,swap_free_mb,load_1min,load_5min,load_15min" > "$SYSTEM_LOG"

    while true; do
        timestamp=$(date "+%Y-%m-%d %H:%M:%S")

        cpu_line=$(top -bn1 | grep "Cpu(s)")
        cpu_idle=$(echo "$cpu_line" | awk '{print $8}')
        cpu_used=$(awk -v idle="$cpu_idle" 'BEGIN {printf "%.2f", 100 - idle}')

        mem_values=$(free -m | awk '
            /Mem:/ {mem_total=$2; mem_used=$3; mem_free=$4; mem_available=$7}
            /Swap:/ {swap_total=$2; swap_used=$3; swap_free=$4}
            END {print mem_total "," mem_used "," mem_free "," mem_available "," swap_total "," swap_used "," swap_free}
        ')

        load_values=$(awk '{print $1 "," $2 "," $3}' /proc/loadavg)

        echo "$timestamp,$cpu_used,$cpu_idle,$mem_values,$load_values" >> "$SYSTEM_LOG"

        sleep 1
    done &

    SYSTEM_MONITOR_PID=$!

    # -----------------------------------------
    # Start Docker training container.
    #
    # No full local dataset mount is used here.
    # The training script streams from S3_DATA_URI and writes only its
    # temporary Scenario C cache to /tmp/tinyimagenet_s3_cache.
    #
    # /app/outputs is mounted so local outputs and logs remain persistent.
    # The script also receives S3_OUTPUT_URI so it can upload training outputs.
    # -----------------------------------------
    docker run --rm --gpus all \
      --name "$CONTAINER_NAME" \
      --shm-size=8g \
      -e GIT_PYTHON_REFRESH=quiet \
      -e AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
      -e AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
      -e AWS_SESSION_TOKEN="$AWS_SESSION_TOKEN" \
      -e AWS_DEFAULT_REGION="$AWS_DEFAULT_REGION" \
      -e S3_DATA_URI="$S3_DATA_URI" \
      -e S3_OUTPUT_URI="$RUN_S3_OUTPUT_URI" \
      -e DATABRICKS_HOST="$DATABRICKS_HOST" \
      -e DATABRICKS_TOKEN="$DATABRICKS_TOKEN" \
      -e MLFLOW_EXPERIMENT_NAME="$EXPERIMENT_NAME" \
      -e MLFLOW_RUN_NAME="$RUN_NAME" \
      -v "$OUTPUT_DIR:/app/outputs" \
      -v "$HOST_CACHE_DIR:$CONTAINER_CACHE_DIR" \
      "$IMAGE_NAME" &

    DOCKER_RUN_PID=$!

    # -----------------------------------------
    # Docker/container CPU and memory monitoring
    # -----------------------------------------
    echo "Waiting for container to start..." | tee -a "$STATUS_LOG"

    until docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; do
        sleep 1

        # If the docker process already exited before appearing, stop waiting.
        if ! kill -0 "$DOCKER_RUN_PID" 2>/dev/null; then
            break
        fi
    done

    echo "timestamp,container,cpu_percent,memory_usage,memory_percent,block_io" > "$DOCKER_LOG"

    while docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; do
        timestamp=$(date "+%Y-%m-%d %H:%M:%S")

        docker stats --no-stream \
          --format "$timestamp,{{.Name}},{{.CPUPerc}},{{.MemUsage}},{{.MemPerc}},{{.BlockIO}}" \
          "$CONTAINER_NAME" >> "$DOCKER_LOG"

        sleep 1
    done &

    DOCKER_MONITOR_PID=$!

    # -----------------------------------------
    # Wait for the experiment to finish.
    # Temporarily disable set -e so we can capture failure,
    # stop monitors, upload logs, then decide whether to stop.
    # -----------------------------------------
    set +e
    wait "$DOCKER_RUN_PID"
    EXIT_CODE=$?
    set -e

    # -----------------------------------------
    # Stop monitors
    # -----------------------------------------
    kill "$GPU_MONITOR_PID" 2>/dev/null || true
    kill "$SYSTEM_MONITOR_PID" 2>/dev/null || true
    kill "$DOCKER_MONITOR_PID" 2>/dev/null || true

    wait "$GPU_MONITOR_PID" 2>/dev/null || true
    wait "$SYSTEM_MONITOR_PID" 2>/dev/null || true
    wait "$DOCKER_MONITOR_PID" 2>/dev/null || true

    echo "Finished Scenario C experiment $i/2 with exit code $EXIT_CODE" | tee -a "$STATUS_LOG"

    if [ "$EXIT_CODE" -eq 0 ]; then
        echo "Run completed successfully." | tee -a "$STATUS_LOG"
    else
        echo "Run failed with exit code $EXIT_CODE." | tee -a "$STATUS_LOG"
    fi

    # -----------------------------------------
    # Upload all local monitoring logs and outputs to S3.
    # This runs even if the experiment failed, so failure logs are preserved.
    # -----------------------------------------
    echo "Local files to upload:" | tee -a "$STATUS_LOG"
    find "$OUTPUT_DIR" -type f | sort | tee -a "$STATUS_LOG"

    upload_directory_to_s3 "$OUTPUT_DIR" "$RUN_S3_OUTPUT_URI"

    echo "Uploaded run folder to S3: $RUN_S3_OUTPUT_URI" | tee -a "$STATUS_LOG"
    echo

    if [ "$EXIT_CODE" -ne 0 ]; then
        echo "Experiment $i failed. Stopping the loop."
        exit "$EXIT_CODE"
    fi

done

echo "All Scenario C experiments finished."
