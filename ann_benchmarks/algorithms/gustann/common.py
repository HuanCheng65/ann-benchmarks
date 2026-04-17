import hashlib
import math
import os
import re
import struct
import subprocess
from pathlib import Path

import numpy as np

from ..base.module import BaseANN


class GustannBase(BaseANN):
    default_home = None
    repo_guess_name = None

    def __init__(self, metric, index_params, search_params=None, *, algorithm_name):
        self._metric = metric
        self._algorithm_name = algorithm_name
        if search_params is None and "index_params" in index_params and "search_params" in index_params:
            search_params = index_params["search_params"]
            index_params = index_params["index_params"]
        self._index_params = dict(index_params)
        self._search_params = dict(search_params or {})
        self._query_params = {}
        self._root = (Path.cwd() / "data" / "gustann_indices").resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._gustann_home = self._resolve_gustann_home(self._index_params.get("gustann_home"))
        self._workdir = None
        self._data_type = None
        self._batch_results = None
        self._batch_latencies = None
        self._best_search_time_override = None
        self._index_file = None
        self._index_prefix = None
        self._pq_prefix = None
        self._nav_prefix = None
        self.name = algorithm_name

    def fit(self, X):
        if self._metric != "euclidean":
            raise ValueError(f"{self._algorithm_name} integration currently only supports euclidean, got {self._metric}")

        if X.dtype == np.uint8:
            data_type = "uint8"
        elif np.issubdtype(X.dtype, np.floating):
            data_type = "float"
        else:
            raise ValueError(f"Unsupported dtype for {self._algorithm_name}: {X.dtype}")

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
        self._nav_prefix = nav_prefix
        self._index_dir = self._workdir
        self._refresh_name()

    def set_query_arguments(self, ef_search):
        self._query_params["ef_search"] = int(ef_search)
        self._refresh_name()

    def get_batch_results(self):
        return self._batch_results

    def get_batch_latencies(self):
        return self._batch_latencies

    def get_additional(self):
        additional = {
            "io_backend": self._io_backend_name(),
            "gustann_home": str(self._gustann_home),
        }
        if self._best_search_time_override is not None:
            additional["best_search_time"] = self._best_search_time_override
        return additional

    def _io_backend_name(self):
        return "memory"

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

    def _write_bvecs_or_fvecs(self, path, X):
        if self._data_type == "uint8":
            data = np.ascontiguousarray(X, dtype=np.uint8)
            header = struct.pack("i", data.shape[1])
            with open(path, "wb") as f:
                for row in data:
                    f.write(header)
                    f.write(row.tobytes(order="C"))
            return

        data = np.ascontiguousarray(X, dtype=np.float32)
        header = struct.pack("i", data.shape[1])
        with open(path, "wb") as f:
            for row in data:
                f.write(header)
                f.write(row.tobytes(order="C"))

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
        completed = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return completed.stdout

    def _parse_reported_time_seconds(self, output):
        for pattern, transform in self._timing_patterns():
            match = re.search(pattern, output, re.MULTILINE)
            if match:
                return transform(float(match.group(1)))
        raise RuntimeError(f"Missing native timing report in {self._algorithm_name} output")

    def _set_batch_latency_from_output(self, output, query_count):
        total_seconds = self._parse_reported_time_seconds(output)
        batch_size = max(1, int(self._search_params.get("minibatch", query_count)))
        batch_count = max(1, int(math.ceil(float(query_count) / float(batch_size))))
        self._batch_latencies = [total_seconds / float(batch_count)] * query_count
        self._best_search_time_override = total_seconds / float(query_count)

    def _timing_patterns(self):
        raise NotImplementedError

    def _required_paths(self):
        raise NotImplementedError

    def _refresh_name(self):
        raise NotImplementedError

    def _resolve_gustann_home(self, configured_home):
        candidates = []
        if configured_home:
            candidates.append(Path(configured_home))
        if self.default_home is not None:
            candidates.append(self.default_home)

        repo_root_guess = Path(__file__).resolve().parents[3]
        candidates.append(repo_root_guess.parent / self.repo_guess_name)

        required_paths = self._required_paths()
        for candidate in candidates:
            resolved = candidate.expanduser().resolve()
            if all((resolved / relative).exists() for relative in required_paths):
                return resolved

        checked = ", ".join(str(c.expanduser().resolve()) for c in candidates)
        raise FileNotFoundError(
            f"Unable to locate a built {self._algorithm_name} checkout. Checked: {checked}"
        )
