#!/usr/bin/env python3
import argparse
from pathlib import Path

import h5py
import numpy as np


def read_bin_header(path: Path) -> tuple[int, int]:
    with path.open("rb") as f:
        header = np.fromfile(f, dtype=np.int32, count=2)
    if header.size != 2:
        raise ValueError(f"Failed to read 2-int header from {path}")
    return int(header[0]), int(header[1])


def memmap_matrix(path: Path, dtype: np.dtype) -> np.memmap:
    rows, cols = read_bin_header(path)
    return np.memmap(path, dtype=dtype, mode="r", offset=8, shape=(rows, cols))


def copy_dense_dataset(
    h5: h5py.File,
    dataset_name: str,
    src: np.memmap | np.ndarray,
    rows_per_chunk: int,
) -> None:
    dst = h5.create_dataset(
        dataset_name,
        shape=src.shape,
        dtype=np.uint8,
        chunks=(min(rows_per_chunk, src.shape[0]), src.shape[1]),
    )
    for start in range(0, src.shape[0], rows_per_chunk):
        stop = min(start + rows_per_chunk, src.shape[0])
        dst[start:stop] = np.asarray(src[start:stop], dtype=np.uint8)
        print(f"{dataset_name}: copied rows [{start}, {stop})")


def write_output_dataset(
    output: Path,
    train: np.ndarray,
    test: np.ndarray,
    neighbors: np.ndarray,
    distances: np.ndarray,
    train_copy_chunk: int = 250_000,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output, "w") as h5:
        h5.attrs["type"] = "dense"
        h5.attrs["distance"] = "euclidean"
        h5.attrs["dimension"] = int(train.shape[1])
        h5.attrs["point_type"] = "uint8"

        copy_dense_dataset(h5, "train", train, train_copy_chunk)
        h5.create_dataset("test", data=np.asarray(test, dtype=np.uint8), dtype=np.uint8)
        h5.create_dataset("neighbors", data=np.asarray(neighbors, dtype=np.int32), dtype=np.int32)
        h5.create_dataset("distances", data=np.asarray(distances, dtype=np.float32), dtype=np.float32)


def compute_gt_distances(
    base: np.memmap,
    queries: np.memmap,
    gt_ids: np.memmap,
    query_block_size: int,
) -> np.ndarray:
    nq, k = gt_ids.shape
    dim = queries.shape[1]
    distances = np.empty((nq, k), dtype=np.float32)

    for start in range(0, nq, query_block_size):
        stop = min(start + query_block_size, nq)
        q_block = np.asarray(queries[start:stop], dtype=np.float32)
        ids_block = np.asarray(gt_ids[start:stop], dtype=np.int64)
        flat_ids = ids_block.reshape(-1)
        base_block = np.asarray(base[flat_ids], dtype=np.float32).reshape(stop - start, k, dim)
        diff = base_block - q_block[:, None, :]
        distances[start:stop] = np.sqrt(np.sum(diff * diff, axis=2, dtype=np.float32))
        print(f"distances: computed rows [{start}, {stop})")

    return distances


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert SIFT1B uint8 bin files into ann-benchmarks HDF5 format."
    )
    parser.add_argument("--base", type=Path, required=True, help="Path to base.bin")
    parser.add_argument("--query", type=Path, required=True, help="Path to query.bin")
    parser.add_argument("--groundtruth", type=Path, required=True, help="Path to gt.bin")
    parser.add_argument("--output", type=Path, required=True, help="Output HDF5 file path")
    parser.add_argument(
        "--train-copy-chunk",
        type=int,
        default=250_000,
        help="How many base vectors to copy into HDF5 per chunk",
    )
    parser.add_argument(
        "--query-block-size",
        type=int,
        default=256,
        help="How many queries to use per block when computing GT distances",
    )
    parser.add_argument(
        "--neighbors",
        type=int,
        default=100,
        help="How many ground-truth neighbors to keep",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base = memmap_matrix(args.base, np.uint8)
    query = memmap_matrix(args.query, np.uint8)
    gt_full = memmap_matrix(args.groundtruth, np.int32)

    if base.shape[1] != query.shape[1]:
        raise ValueError(f"Base/query dimension mismatch: {base.shape[1]} vs {query.shape[1]}")
    if gt_full.shape[0] != query.shape[0]:
        raise ValueError(f"Ground-truth/query count mismatch: {gt_full.shape[0]} vs {query.shape[0]}")
    if args.neighbors > gt_full.shape[1]:
        raise ValueError(f"Requested {args.neighbors} neighbors, but GT only has {gt_full.shape[1]}")

    gt_ids = gt_full[:, : args.neighbors]
    distances = compute_gt_distances(base, query, gt_ids, args.query_block_size)

    write_output_dataset(
        args.output,
        base,
        query,
        gt_ids,
        distances,
        train_copy_chunk=args.train_copy_chunk,
    )

    print(f"wrote {args.output}")
    print(f"train={base.shape} test={query.shape} neighbors={gt_ids.shape}")


if __name__ == "__main__":
    main()
