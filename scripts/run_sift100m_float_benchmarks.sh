#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="${RAW_DIR:-$ROOT_DIR/../shared_datasets/sift100m}"
DATASET_NAME="${DATASET_NAME:-sift100m-128-euclidean-float}"
OUTPUT_H5="${OUTPUT_H5:-$ROOT_DIR/data/${DATASET_NAME}.hdf5}"
COUNT="${COUNT:-10}"
RUNS="${RUNS:-1}"
GPU="${GPU:-1}"
INSTALL_PROC="${INSTALL_PROC:-1}"
BUILD_IMAGES="${BUILD_IMAGES:-1}"
RUN_DISABLED_DOCKER="${RUN_DISABLED_DOCKER:-0}"
FORCE_DATASET="${FORCE_DATASET:-0}"
FORCE_RESULTS="${FORCE_RESULTS:-0}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/sift100m-float}"

mkdir -p "$LOG_DIR"

BASE_FILE="$RAW_DIR/base.100m.u8bin"
QUERY_FILE="$RAW_DIR/query.public.10K.u8bin"
GT_FILE="$RAW_DIR/groundtruth.100m.ibin"

if [[ ! -f "$BASE_FILE" || ! -f "$QUERY_FILE" || ! -f "$GT_FILE" ]]; then
  echo "Missing raw SIFT100M files under $RAW_DIR" >&2
  exit 1
fi

if [[ "$FORCE_DATASET" == "1" || ! -f "$OUTPUT_H5" ]]; then
  python3 -u "$ROOT_DIR/scripts/convert_sift100m_to_hdf5.py" \
    --base "$BASE_FILE" \
    --query "$QUERY_FILE" \
    --groundtruth "$GT_FILE" \
    --output "$OUTPUT_H5" \
    --output-dtype float32 \
    2>&1 | tee "$LOG_DIR/dataset-generation.log"
fi

if [[ "$FORCE_RESULTS" == "1" ]]; then
  rm -rf "$ROOT_DIR/results/$DATASET_NAME"
fi

required_folders="$(
python3 - <<'PY'
import os
from pathlib import Path
import yaml

from ann_benchmarks.definitions import get_config_files, get_definitions

run_disabled = os.environ.get("RUN_DISABLED_DOCKER", "0") == "1"
defs = get_definitions(
    dimension=128,
    point_type="float",
    distance_metric="euclidean",
    count=10,
    base_dir="ann_benchmarks/algorithms",
)
required_tags = {
    d.docker_tag
    for d in defs
    if d.docker_tag is not None and (run_disabled or not d.disabled)
}

for config_file in sorted(get_config_files("ann_benchmarks/algorithms")):
    folder = Path(config_file).parent
    dockerfile = folder / "Dockerfile"
    if not dockerfile.exists():
        continue
    with open(config_file, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    tags = set()

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "docker_tag" and isinstance(value, str):
                    tags.add(value)
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(config)

    if tags.intersection(required_tags):
        print(folder.name)
PY
)"

if [[ "$BUILD_IMAGES" == "1" ]]; then
  docker build --rm -t ann-benchmarks -f "$ROOT_DIR/ann_benchmarks/algorithms/base/Dockerfile" "$ROOT_DIR" \
    2>&1 | tee "$LOG_DIR/docker-build-base.log"

  while IFS= read -r folder; do
    [[ -z "$folder" ]] && continue
    docker build --rm -t "ann-benchmarks-$folder" -f "$ROOT_DIR/ann_benchmarks/algorithms/$folder/Dockerfile" "$ROOT_DIR" \
      2>&1 | tee "$LOG_DIR/docker-build-$folder.log"
  done <<< "$required_folders"
fi

docker_args=()
if [[ "$FORCE_RESULTS" == "1" ]]; then
  docker_args+=(--force)
fi
if [[ "$RUN_DISABLED_DOCKER" == "1" ]]; then
  docker_args+=(--run-disabled)
fi

python3 -u "$ROOT_DIR/run.py" \
  --dataset "$DATASET_NAME" \
  --count "$COUNT" \
  --runs "$RUNS" \
  --batch \
  "${docker_args[@]}" \
  2>&1 | tee "$LOG_DIR/docker-benchmark.log"

CUDA_VISIBLE_DEVICES="$GPU" python3 -u "$ROOT_DIR/run.py" \
  --local \
  --batch \
  --run-disabled \
  --dataset "$DATASET_NAME" \
  --count "$COUNT" \
  --runs "$RUNS" \
  --algorithm gustann-mem \
  --force \
  2>&1 | tee "$LOG_DIR/gustann-benchmark.log"
