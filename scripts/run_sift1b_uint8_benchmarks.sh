#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
RAW_DIR="${RAW_DIR:-$ROOT_DIR/../shared_datasets/sift1b}"
DATASET_NAME="${DATASET_NAME:-sift1b-128-euclidean}"
OUTPUT_H5="${OUTPUT_H5:-$ROOT_DIR/data/${DATASET_NAME}.hdf5}"
COUNT="${COUNT:-10}"
RUNS="${RUNS:-1}"
GPU="${GPU:-0}"
FORCE_DATASET="${FORCE_DATASET:-0}"
FORCE_RESULTS="${FORCE_RESULTS:-0}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/sift1b-uint8}"
TMPFS_ENV_FILE="${TMPFS_ENV_FILE:-$ROOT_DIR/.tmp/sift1b-tmpfs.env}"

mkdir -p "$LOG_DIR"

if [[ -f "$TMPFS_ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$TMPFS_ENV_FILE"
  echo "Using tmpfs index override from $TMPFS_ENV_FILE"
fi

BASE_FILE="$RAW_DIR/base.bin"
QUERY_FILE="$RAW_DIR/query.bin"
GT_FILE="$RAW_DIR/gt.bin"

if [[ ! -f "$BASE_FILE" || ! -f "$QUERY_FILE" || ! -f "$GT_FILE" ]]; then
  echo "Missing raw SIFT1B files under $RAW_DIR" >&2
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

if [[ "$FORCE_DATASET" == "1" || ! -f "$OUTPUT_H5" ]]; then
  "$PYTHON_BIN" -u "$ROOT_DIR/scripts/convert_sift1b_to_hdf5.py" \
    --base "$BASE_FILE" \
    --query "$QUERY_FILE" \
    --groundtruth "$GT_FILE" \
    --output "$OUTPUT_H5" \
    2>&1 | tee "$LOG_DIR/dataset-generation.log"
fi

if [[ "$FORCE_RESULTS" == "1" ]]; then
  rm -rf "$ROOT_DIR/results/$DATASET_NAME"
fi

for algorithm in gustann-mem gustann-original flashanns; do
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -u "$ROOT_DIR/run.py" \
    --local \
    --batch \
    --run-disabled \
    --dataset "$DATASET_NAME" \
    --count "$COUNT" \
    --runs "$RUNS" \
    --algorithm "$algorithm" \
    --force \
    2>&1 | tee "$LOG_DIR/${algorithm}.log"
done
