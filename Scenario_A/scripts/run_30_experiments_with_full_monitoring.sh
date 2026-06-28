#!/bin/bash

set -e

EXPERIMENT_NAME="/Users/1749412@uab.cat/resnet18-tinyimagenet"
IMAGE_NAME="resnet18-tinyimagenet"

mkdir -p outputs

for i in $(seq -w 26 30)
do
    RUN_NAME="scenario-a2-run-${i}"
    OUTPUT_DIR="$(pwd)/outputs/run_${i}"

    mkdir -p "$OUTPUT_DIR"

    GPU_LOG="$OUTPUT_DIR/gpu_usage_log.csv"
    DOCKER_LOG="$OUTPUT_DIR/docker_resource_log.csv"
    SYSTEM_LOG="$OUTPUT_DIR/system_resource_log.csv"

    CONTAINER_NAME="resnet18_tinyimagenet_run_${i}"

    echo "=========================================="
    echo "Starting experiment $i/30"
    echo "MLflow run name: $RUN_NAME"
    echo "Output folder: $OUTPUT_DIR"
    echo "Container name: $CONTAINER_NAME"
    echo "=========================================="

    # -----------------------------------------
    # GPU monitoring: whole GPU
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
    # Start Docker training container
    # -----------------------------------------
    docker run --rm --gpus all \
      --name "$CONTAINER_NAME" \
      --shm-size=8g \
      -e GIT_PYTHON_REFRESH=quiet \
      -e DATABRICKS_HOST="$DATABRICKS_HOST" \
      -e DATABRICKS_TOKEN="$DATABRICKS_TOKEN" \
      -e MLFLOW_EXPERIMENT_NAME="$EXPERIMENT_NAME" \
      -e MLFLOW_RUN_NAME="$RUN_NAME" \
      -v "$(pwd)/tiny-imagenet-200:/app/tiny-imagenet-200" \
      -v "$OUTPUT_DIR:/app/outputs" \
      "$IMAGE_NAME" &

    DOCKER_RUN_PID=$!

    # -----------------------------------------
    # Docker/container CPU and memory monitoring
    # -----------------------------------------
    echo "Waiting for container to start..."

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
    # Wait for the experiment to finish
    # -----------------------------------------
    wait "$DOCKER_RUN_PID"
    EXIT_CODE=$?

    # -----------------------------------------
    # Stop monitors
    # -----------------------------------------
    kill "$GPU_MONITOR_PID" 2>/dev/null || true
    kill "$SYSTEM_MONITOR_PID" 2>/dev/null || true
    kill "$DOCKER_MONITOR_PID" 2>/dev/null || true

    wait "$GPU_MONITOR_PID" 2>/dev/null || true
    wait "$SYSTEM_MONITOR_PID" 2>/dev/null || true
    wait "$DOCKER_MONITOR_PID" 2>/dev/null || true

    echo "Finished experiment $i/30 with exit code $EXIT_CODE"
    echo

    if [ "$EXIT_CODE" -ne 0 ]; then
        echo "Experiment $i failed. Stopping the loop."
        exit "$EXIT_CODE"
    fi

done

echo "All 30 experiments finished."
