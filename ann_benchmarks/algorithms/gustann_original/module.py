from pathlib import Path

from ..gustann.common import GustannBase


DEFAULT_GUSTANN_HOME = Path("/home/xieminhui/starrydream/GustANN")


class GustannOriginal(GustannBase):
    default_home = DEFAULT_GUSTANN_HOME
    repo_guess_name = "GustANN"

    def __init__(self, metric, index_params, search_params=None):
        super().__init__(metric, index_params, search_params, algorithm_name="GustANNOriginal")

    def batch_query(self, X, n):
        if self._workdir is None:
            raise RuntimeError("fit must be called before batch_query")

        requested_minibatch = int(self._search_params.get("minibatch", len(X)))
        batch_size = len(X) if requested_minibatch <= 0 else min(requested_minibatch, len(X))
        query_suffix = ".bvecs" if self._data_type == "uint8" else ".fvecs"
        query_file = self._workdir / f"queries_0000{query_suffix}"
        output_ids = self._workdir / "result_ids_0000.bin"
        self._write_bvecs_or_fvecs(query_file, X)
        if output_ids.exists():
            output_ids.unlink()

        cmd = [
            str(self._gustann_home / "build/bin/search_disk_hybrid_bench"),
            "--query", str(query_file),
            "--index", str(self._index_file),
            "--pq_data", str(self._pq_prefix),
            "--nav_graph", str(self._nav_prefix),
            "--data_type", self._data_type,
            "--io_backend", "memory",
            "--topk", str(n),
            "--ef_search", str(self._query_params.get("ef_search", self._search_params["ef_search"])),
            "-B", str(batch_size),
            "-T", str(self._search_params.get("thread", 1)),
            "-C", str(self._search_params.get("ctx_per_thread", 1)),
            "--output_ids", str(output_ids),
        ]

        output = self._run(cmd, cwd=self._gustann_home)
        self._set_batch_latency_from_output(output, len(X))
        self._batch_results = self._read_ids(output_ids, n)
        if len(self._batch_results) != len(X):
            raise RuntimeError(
                "Expected "
                f"{len(X)} result rows from GustANN original batch search, got {len(self._batch_results)}"
            )

    def _refresh_name(self):
        self.name = (
            "GustANNOriginal("
            f"B={self._search_params.get('minibatch', 0)},"
            f"T={self._search_params.get('thread', 1)},"
            f"C={self._search_params.get('ctx_per_thread', 1)},"
            f"ef={self._query_params.get('ef_search', self._search_params['ef_search'])}"
            ")"
        )

    def _timing_patterns(self):
        return [
            (r"\[REPORT\]\s+Time\s+([0-9.eE+-]+)", lambda value: value),
        ]

    def _required_paths(self):
        return [
            Path("build/bin/search_disk_hybrid_bench"),
            Path("deps/DiskANN/build/apps/build_disk_index"),
        ]
