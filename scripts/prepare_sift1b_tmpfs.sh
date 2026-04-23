#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="${SRC_DIR:-/mnt/disk1/starrydream/ann_benchmarks_data/gustann_indices/94adb9cfbf65cbc8}"
TMPFS_DIR="${TMPFS_DIR:-/mnt/disk1/starrydream/tmpfs/sift1b-index}"
TMPFS_SIZE="${TMPFS_SIZE:-760G}"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/.tmp/sift1b-tmpfs.env}"
SKIP_MOUNT="${SKIP_MOUNT:-0}"
SUDO_BIN="${SUDO_BIN:-sudo}"
RSYNC_BIN="${RSYNC_BIN:-rsync}"

FILES=(
  ann_disk.index
  ann_pq_compressed.bin
  ann_pq_pivots.bin
  ann_disk.index_centroids.bin
  ann_disk.index_medoids.bin
  nav_data.bin
  nav_index
  nav_index.data
  nav_index.tags
  map.txt
)
BASE_BIN="$SRC_DIR/base.bin"

for name in "${FILES[@]}"; do
  if [[ ! -f "$SRC_DIR/$name" ]]; then
    echo "Missing source file: $SRC_DIR/$name" >&2
    exit 1
  fi
done

mkdir -p "$(dirname "$ENV_FILE")"

if [[ "$SKIP_MOUNT" == "1" ]]; then
  mkdir -p "$TMPFS_DIR"
else
  $SUDO_BIN mkdir -p "$TMPFS_DIR"
  if ! mountpoint -q "$TMPFS_DIR"; then
    $SUDO_BIN mount -t tmpfs -o "size=$TMPFS_SIZE" tmpfs "$TMPFS_DIR"
  fi
fi

rsync_args=(-ah)
if [[ "$SKIP_MOUNT" != "1" ]]; then
  rsync_args+=(--info=progress2)
fi

"$RSYNC_BIN" "${rsync_args[@]}" \
  "$SRC_DIR/ann_disk.index" \
  "$SRC_DIR/ann_pq_compressed.bin" \
  "$SRC_DIR/ann_pq_pivots.bin" \
  "$SRC_DIR/ann_disk.index_centroids.bin" \
  "$SRC_DIR/ann_disk.index_medoids.bin" \
  "$SRC_DIR/nav_data.bin" \
  "$SRC_DIR/nav_index" \
  "$SRC_DIR/nav_index.data" \
  "$SRC_DIR/nav_index.tags" \
  "$SRC_DIR/map.txt" \
  "$TMPFS_DIR/"

printf 'export ANNB_INDEX_DIR_OVERRIDE="%s"\n' "$TMPFS_DIR" > "$ENV_FILE"
if [[ -f "$BASE_BIN" ]]; then
  printf 'export ANNB_EXTERNAL_BASE_U8BIN="%s"\n' "$BASE_BIN" >> "$ENV_FILE"
fi

echo "Prepared SIFT1B tmpfs mirror at $TMPFS_DIR"
echo "Env file written to $ENV_FILE"
