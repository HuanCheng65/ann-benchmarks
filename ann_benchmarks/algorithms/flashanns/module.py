from pathlib import Path

from ..gustann.common import GustannBase


DEFAULT_FLASHANNS_HOME = Path("/home/xieminhui/starrydream/quiver")


class Flashanns(GustannBase):
    default_home = DEFAULT_FLASHANNS_HOME
    repo_guess_name = "quiver"

    def __init__(self, metric, index_params, search_params=None):
        normalized_index_params = dict(index_params)
        if "flashanns_home" in normalized_index_params and "gustann_home" not in normalized_index_params:
            normalized_index_params["gustann_home"] = normalized_index_params["flashanns_home"]
        super().__init__(metric, normalized_index_params, search_params, algorithm_name="FlashANN")

    def batch_query(self, X, n):
        if self._workdir is None:
            raise RuntimeError("fit must be called before batch_query")

        query_file = self._workdir / "queries_0000.bin"
        output_ids = self._workdir / "result_ids_0000.bin"
        output_distances = self._workdir / "result_distances_0000.bin"
        self._write_diskann_bin(query_file, X)
        if output_ids.exists():
            output_ids.unlink(missing_ok=True)
        if output_distances.exists():
            output_distances.unlink(missing_ok=True)

        cmd = [
            str(self._gustann_home / "build/bin/flashanns_search"),
            "--index-dir", str(self._index_dir),
            "--query", str(query_file),
            "--data-type", self._data_type,
            "--topk", str(n),
            "--repeat", str(self._search_params.get("repeat", 20)),
            "--ef-search", str(self._query_params.get("ef_search", self._search_params["ef_search"])),
            "--num-blocks", str(self._search_params.get("num_blocks", 756)),
            "--poll-threads", str(self._search_params.get("poll_threads", 6)),
            "--poll-contexts", str(self._search_params.get("poll_contexts", 7)),
            "--pipe-width", str(self._search_params.get("pipe_width", 4)),
            "--output-ids", str(output_ids),
            "--output-distances", str(output_distances),
        ]

        output = self._run(cmd, cwd=self._gustann_home)
        total_seconds = self._parse_reported_time_seconds(output)
        expected_rows = len(X)
        repeat = int(self._search_params.get("repeat", 20))
        effective_queries = max(1, expected_rows * repeat)
        per_query_seconds = total_seconds / float(effective_queries)
        self._batch_latencies = [per_query_seconds] * expected_rows
        self._best_search_time_override = per_query_seconds
        self._batch_results = self._normalize_repeated_rows(
            self._read_ids(output_ids, n), expected_rows, repeat, "result ids"
        )
        self._batch_distances = self._normalize_repeated_rows(
            self._read_distances(output_distances, n), expected_rows, repeat, "result distances"
        )
        if len(self._batch_results) != len(X):
            raise RuntimeError(
                f"Expected {len(X)} result rows from FlashANN batch search, got {len(self._batch_results)}"
            )

    def _normalize_repeated_rows(self, rows, expected_rows, repeat, label):
        row_count = len(rows)
        if row_count == expected_rows:
            return rows
        if repeat > 1 and row_count == expected_rows * repeat:
            return rows[:expected_rows]
        raise RuntimeError(
            f"Expected {expected_rows} rows in FlashANN {label}, got {row_count}"
        )

    def _refresh_name(self):
        self.name = (
            "FlashANN("
            f"blocks={self._search_params.get('num_blocks', 756)},"
            f"T={self._search_params.get('poll_threads', 6)},"
            f"C={self._search_params.get('poll_contexts', 7)},"
            f"W={self._search_params.get('pipe_width', 4)},"
            f"repeat={self._search_params.get('repeat', 20)},"
            f"ef={self._query_params.get('ef_search', self._search_params['ef_search'])}"
            ")"
        )

    def _timing_patterns(self):
        return [
            (r"^\s*Time:\s*([0-9.eE+-]+)\s*s\s*$", lambda value: value),
        ]

    def _required_paths(self):
        return [
            Path("build/bin/flashanns_search"),
            Path("build/bin/gen_small_file"),
            Path("deps/DiskANN/build/apps/build_disk_index"),
        ]
