import hashlib
import os
import struct
import subprocess
from pathlib import Path

import numpy as np

from ..base.module import BaseANN


DEFAULT_GUSTANN_HOME = Path("/home/xieminhui/starrydream/gustann-mem")


class Gustann(BaseANN):
    def __init__(self, metric, index_params, search_params=None):
        self._metric = metric
        if search_params is None and "index_params" in index_params and "search_params" in index_params:
            search_params = index_params["search_params"]
            index_params = index_params["index_params"]
        self._index_params = dict(index_params)
        self._search_params = dict(search_params)
        self._query_params = {}
        self._root = (Path.cwd() / "data" / "gustann_indices").resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._gustann_home = self._resolve_gustann_home(self._index_params.get("gustann_home"))
        self._workdir = None
        self._data_type = None
        self._batch_results = None
        self.name = "GustANN"

    def fit(self, X):
        if self._metric != "euclidean":
            raise ValueError(f"GustANN integration currently only supports euclidean, got {self._metric}")

        if X.dtype == np.uint8:
            data_type = "uint8"
        elif np.issubdtype(X.dtype, np.floating):
            data_type = "float"
        else:
            raise ValueError(f"Unsupported dtype for GustANN: {X.dtype}")

        self._data_type = data_type
        X = np.ascontiguousarray(X)
        dataset_hash = hashlib.sha256(X.view(np.uint8)).hexdigest()[:16]
        self._workdir = self._root / dataset_hash
        self._workdir.mkdir(parents=True, exist_ok=True)

        base_bin = self._workdir / "base.bin"
        index_prefix = self._workdir / "ann"
        nav_prefix = self._workdir / "nav_index"
        self._write_diskann_bin(base_bin, X)

        index_file = Path(str(index_prefix) + "_disk.index")
        pq_prefix = Path(str(index_prefix) + "_pq")
        if not index_file.exists():
            self._run_diskann_build(base_bin, index_prefix, data_type)
        if not nav_prefix.exists():
            self._run_nav_build(base_bin, nav_prefix, data_type, X.shape[0])

        self._index_file = index_file
        self._index_prefix = index_prefix
        self._pq_prefix = pq_prefix
        self._index_dir = self._workdir
        self._refresh_name()

    def set_query_arguments(self, ef_search):
        self._query_params["ef_search"] = int(ef_search)
        self._refresh_name()

    def batch_query(self, X, n):
        if self._workdir is None:
            raise RuntimeError("fit must be called before batch_query")

        X = np.ascontiguousarray(X)
        search_bin = "search_disk_mem_uint8" if self._data_type == "uint8" else "search_disk_mem_float"
        requested_minibatch = int(self._search_params.get("minibatch", 0))
        batch_size = len(X) if requested_minibatch <= 0 else min(requested_minibatch, len(X))
        query_file = self._workdir / "queries_0000.bin"
        output_ids = self._workdir / "result_ids_0000.bin"
        self._write_diskann_bin(query_file, X)
        if output_ids.exists():
            output_ids.unlink()

        cmd = [
            str(self._gustann_home / f"build/bin/{search_bin}"),
            "--index-dir", str(self._index_prefix),
            "--query", str(query_file),
            "--output_ids", str(output_ids),
            "--topk", str(n),
            "--ef_search", str(self._query_params.get("ef_search", self._search_params["ef_search"])),
            "--cache_pct_list", str(self._search_params.get("cache_pct", 0)),
            "--blocks_per_sm_list", str(self._search_params.get("blocks_per_sm", 1)),
            "--batch_size_list", str(batch_size),
        ]
        pipeline_width = int(self._search_params.get("pipeline_width", 0))
        if pipeline_width:
            cmd.extend(["--pipeline_width", str(pipeline_width)])

        self._run(cmd, cwd=self._gustann_home)
        self._batch_results = self._read_ids(output_ids, n)

    def get_batch_results(self):
        return self._batch_results

    def get_additional(self):
        return {
            "io_backend": "memory",
            "gustann_home": str(self._gustann_home),
        }

    def _run_diskann_build(self, base_bin, index_prefix, data_type):
        diskann_dir = self._gustann_home / "deps/DiskANN/build/apps"
        cmd = [
            str(diskann_dir / "build_disk_index"),
            "--data_type", data_type,
            "--dist_fn", "l2",
            "--index_path_prefix", str(index_prefix),
            "--data_path", str(base_bin),
            "--QD", str(self._index_params.get("pq_size", 32)),
            "-B", str(self._index_params.get("search_memory_gb", 4)),
            "-M", str(self._index_params.get("build_memory_gb", 16)),
            "-R", str(self._index_params.get("max_degree", 128)),
            "-L", str(self._index_params.get("l_build", 200)),
        ]
        if "build_threads" in self._index_params:
            cmd.extend(["--num_threads", str(self._index_params["build_threads"])])
        self._run(cmd, cwd=self._gustann_home)

    def _run_nav_build(self, base_bin, nav_prefix, data_type, num_points):
        sample_size = min(int(self._index_params.get("pivot_sample_size", 1000000)), int(num_points))
        data_size = 1 if data_type == "uint8" else 4
        sample_bin = self._workdir / "nav_data.bin"
        map_txt = self._workdir / "map.txt"
        tag_bin = Path(str(nav_prefix) + ".tags")
        cmd1 = [
            str(self._gustann_home / "build/bin/gen_small_file"),
            str(sample_size),
            str(data_size),
            str(base_bin),
            str(sample_bin),
            str(map_txt),
        ]
        cmd2 = [
            str(self._gustann_home / "deps/DiskANN/build/apps/build_memory_index"),
            "--data_type", data_type,
            "--dist_fn", "l2",
            "--index_path_prefix", str(nav_prefix),
            "--data_path", str(sample_bin),
            "-R", str(self._index_params.get("pivot_graph_degree", 32)),
            "-L", str(self._index_params.get("pivot_graph_l_build", 50)),
        ]
        self._run(cmd1, cwd=self._gustann_home)
        self._run(cmd2, cwd=self._gustann_home)
        self._write_tag_bin(map_txt, tag_bin)

    def _write_diskann_bin(self, path, X):
        if self._data_type == "uint8":
            data = np.ascontiguousarray(X, dtype=np.uint8)
        else:
            data = np.ascontiguousarray(X, dtype=np.float32)
        with open(path, "wb") as f:
            f.write(struct.pack("ii", data.shape[0], data.shape[1]))
            f.write(data.tobytes(order="C"))

    def _write_tag_bin(self, txt_path, bin_path):
        values = []
        with open(txt_path, "r", encoding="ascii") as f:
            for line in f:
                line = line.strip()
                if line:
                    values.append(int(line))
        arr = np.asarray(values, dtype=np.int32).reshape((-1, 1))
        with open(bin_path, "wb") as f:
            f.write(struct.pack("ii", arr.shape[0], arr.shape[1]))
            f.write(arr.tobytes(order="C"))

    def _read_ids(self, path, expected_topk):
        with open(path, "rb") as f:
            num_queries, topk = struct.unpack("ii", f.read(8))
            if topk != expected_topk:
                raise RuntimeError(f"Unexpected topk in result file: {topk} != {expected_topk}")
            ids = np.frombuffer(f.read(), dtype=np.int32)
        ids = ids.reshape((num_queries, topk))
        deduped = []
        for row in ids:
            seen = set()
            unique = []
            for raw_idx in row:
                idx = int(raw_idx)
                if idx < 0 or idx in seen:
                    continue
                seen.add(idx)
                unique.append(idx)
            deduped.append(unique)
        return deduped

    def _run(self, cmd, cwd):
        env = os.environ.copy()
        env.setdefault("CUDA_VISIBLE_DEVICES", str(self._search_params.get("cuda_visible_devices", 0)))
        subprocess.run(cmd, cwd=cwd, env=env, check=True)

    def _refresh_name(self):
        self.name = (
            "GustANN("
            f"B={self._search_params.get('minibatch', 0)},"
            f"cache={self._search_params.get('cache_pct', 0)}%,"
            f"blocks={self._search_params.get('blocks_per_sm', 1)},"
            f"W={self._search_params.get('pipeline_width', 0)},"
            f"ef={self._query_params.get('ef_search', self._search_params['ef_search'])}"
            ")"
        )

    def _resolve_gustann_home(self, configured_home):
        candidates = []
        if configured_home:
            candidates.append(Path(configured_home))
        candidates.append(DEFAULT_GUSTANN_HOME)

        repo_root_guess = Path(__file__).resolve().parents[4]
        candidates.append(repo_root_guess.parent / "gustann-mem")

        for candidate in candidates:
            resolved = candidate.expanduser().resolve()
            if (resolved / "build/bin/search_disk_mem_float").exists() and (resolved / "deps/DiskANN/build/apps/build_disk_index").exists():
                return resolved

        checked = ", ".join(str(c.expanduser().resolve()) for c in candidates)
        raise FileNotFoundError(
            "Unable to locate a built GustANN checkout. Checked: "
            f"{checked}"
        )
