from pathlib import Path

from .common import GustannBase


DEFAULT_GUSTANN_HOME = Path("/home/xieminhui/starrydream/gustann-mem")


class Gustann(GustannBase):
    default_home = DEFAULT_GUSTANN_HOME
    repo_guess_name = "gustann-mem"

    def __init__(self, metric, index_params, search_params=None):
        super().__init__(metric, index_params, search_params, algorithm_name="GustANN")

    def batch_query(self, X, n):
        if self._workdir is None:
            raise RuntimeError("fit must be called before batch_query")

        search_bin = "search_disk_mem_uint8" if self._data_type == "uint8" else "search_disk_mem_float"
        requested_minibatch = int(self._search_params.get("minibatch", 0))
        batch_size = len(X) if requested_minibatch <= 0 else min(requested_minibatch, len(X))
        query_file = self._workdir / "queries_0000.bin"
        output_ids = self._workdir / "result_ids_0000.bin"
        self._write_diskann_bin(query_file, X)
        if output_ids.exists():
            output_ids.unlink(missing_ok=True)

        cmd = [
            str(self._gustann_home / f"build/bin/{search_bin}"),
            "--index-dir", str(self._index_prefix),
            "--query", str(query_file),
            "--output_ids", str(output_ids),
            "--topk", str(n),
            "--ef_search_list", str(self._query_params.get("ef_search", self._search_params["ef_search"])),
            "--cache_pct_list", str(self._search_params.get("cache_pct", 0)),
            "--pq_dim", str(self._index_params.get("pq_size", 32)),
            "--blocks_per_sm_list", str(self._search_params.get("blocks_per_sm", 1)),
            "--batch_size_list", str(batch_size),
        ]
        pipeline_width = int(self._search_params.get("pipeline_width", 0))
        if pipeline_width:
            cmd.extend(["--pipeline_width", str(pipeline_width)])

        output = self._run(cmd, cwd=self._gustann_home)
        self._set_batch_latency_from_output(output, len(X))
        self._batch_results = self._read_ids(output_ids, n)
        if len(self._batch_results) != len(X):
            raise RuntimeError(
                f"Expected {len(X)} result rows from GustANN batch search, got {len(self._batch_results)}"
            )

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

    def _timing_patterns(self):
        return [
            (r"BATCH_LAT_MS=([0-9.eE+-]+)", lambda value: value / 1000.0),
            (r"\[REPORT\]\s+Time\s+([0-9.eE+-]+)", lambda value: value),
        ]

    def _required_paths(self):
        return [
            Path("build/bin/search_disk_mem_float"),
            Path("deps/DiskANN/build/apps/build_disk_index"),
        ]
