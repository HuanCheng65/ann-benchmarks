#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMPFS_DIR="${TMPFS_DIR:-/mnt/disk1/starrydream/tmpfs/sift1b-index}"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/.tmp/sift1b-tmpfs.env}"
SKIP_MOUNT="${SKIP_MOUNT:-0}"
FORCE_RELEASE="${FORCE_RELEASE:-0}"
SUDO_BIN="${SUDO_BIN:-sudo}"

active_processes="$(pgrep -af 'run.py --local --batch|search_disk_mem_|search_disk_hybrid_bench|flashanns_search' || true)"
if [[ -n "$active_processes" && "$FORCE_RELEASE" != "1" ]]; then
  echo "Active benchmark or search processes detected:" >&2
  echo "$active_processes" >&2
  echo "Set FORCE_RELEASE=1 to unmount anyway." >&2
  exit 1
fi

sync

if [[ "$SKIP_MOUNT" == "1" ]]; then
  rm -f "$ENV_FILE"
  echo "Removed tmpfs env file $ENV_FILE"
  exit 0
fi

if mountpoint -q "$TMPFS_DIR"; then
  $SUDO_BIN umount "$TMPFS_DIR"
fi

rm -f "$ENV_FILE"
echo "Released SIFT1B tmpfs mirror from $TMPFS_DIR"

