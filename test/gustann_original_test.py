from pathlib import Path

import numpy as np
import pytest

from ann_benchmarks.algorithms.gustann_original.module import GustannOriginal


def test_original_batch_query_uses_bench_binary_and_memory_backend(monkeypatch, tmp_path):
    monkeypatch.setattr(
        GustannOriginal,
        "_resolve_gustann_home",
        lambda self, configured_home: Path("/tmp/fake-gustann"),
    )

    algo = GustannOriginal(
        "euclidean",
        {"gustann_home": "/tmp/fake-gustann", "pq_size": 32},
        {
            "minibatch": 4,
            "thread": 1,
            "ctx_per_thread": 1,
            "ef_search": 40,
        },
    )

    workdir = tmp_path / "index"
    workdir.mkdir()
    algo._workdir = workdir
    algo._index_prefix = workdir / "ann"
    algo._data_type = "uint8"
    algo._nav_prefix = workdir / "nav_index"

    calls = []
    monkeypatch.setattr(algo, "_write_bvecs_or_fvecs", lambda path, X: None)
    monkeypatch.setattr(algo, "_read_ids", lambda path, expected_topk: [[0]] * 5)
    monkeypatch.setattr(algo, "_run", lambda cmd, cwd: calls.append((cmd, cwd)) or "[REPORT] Time 0.100\n")

    algo.batch_query(np.zeros((5, 4), dtype=np.uint8), 10)

    cmd, cwd = calls[0]
    assert cmd[0] == str(Path("/tmp/fake-gustann") / "build/bin/search_disk_hybrid_bench")
    assert cmd[cmd.index("--io_backend") + 1] == "memory"
    assert cmd[cmd.index("--output_ids") + 1] == str(workdir / "result_ids_0000.bin")
    assert cwd == Path("/tmp/fake-gustann")
    assert algo.get_batch_latencies() == [0.05] * 5


def test_original_batch_query_requires_native_timing_report(monkeypatch, tmp_path):
    monkeypatch.setattr(
        GustannOriginal,
        "_resolve_gustann_home",
        lambda self, configured_home: Path("/tmp/fake-gustann"),
    )

    algo = GustannOriginal(
        "euclidean",
        {"gustann_home": "/tmp/fake-gustann", "pq_size": 32},
        {
            "minibatch": 4,
            "thread": 1,
            "ctx_per_thread": 1,
            "ef_search": 40,
        },
    )

    workdir = tmp_path / "index"
    workdir.mkdir()
    algo._workdir = workdir
    algo._index_prefix = workdir / "ann"
    algo._data_type = "uint8"
    algo._nav_prefix = workdir / "nav_index"

    monkeypatch.setattr(algo, "_write_bvecs_or_fvecs", lambda path, X: None)
    monkeypatch.setattr(algo, "_read_ids", lambda path, expected_topk: [[0]] * 5)
    monkeypatch.setattr(algo, "_run", lambda cmd, cwd: "search completed\n")

    with pytest.raises(RuntimeError, match="timing report"):
        algo.batch_query(np.zeros((5, 4), dtype=np.uint8), 10)
