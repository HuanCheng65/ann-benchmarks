from pathlib import Path

import numpy as np
import pytest

from ann_benchmarks.algorithms.flashanns.module import Flashanns


def test_batch_query_passes_output_ids_to_flashanns(monkeypatch, tmp_path):
    monkeypatch.setattr(
        Flashanns,
        "_resolve_gustann_home",
        lambda self, configured_home: Path("/tmp/fake-quiver"),
    )

    algo = Flashanns(
        "euclidean",
        {"flashanns_home": "/tmp/fake-quiver", "pq_size": 32},
        {
            "num_blocks": 756,
            "poll_threads": 6,
            "poll_contexts": 7,
            "pipe_width": 4,
            "ef_search": 40,
        },
    )

    workdir = tmp_path / "index"
    workdir.mkdir()
    algo._workdir = workdir
    algo._index_dir = workdir
    algo._data_type = "uint8"
    algo.set_query_arguments(30)

    calls = []

    monkeypatch.setattr(algo, "_write_diskann_bin", lambda path, X: None)
    monkeypatch.setattr(algo, "_read_ids", lambda path, expected_topk: [[0]] * 5)
    monkeypatch.setattr(algo, "_run", lambda cmd, cwd: calls.append((cmd, cwd)) or "  Time: 0.250000 s\n")

    algo.batch_query(np.zeros((5, 4), dtype=np.uint8), 10)

    cmd, cwd = calls[0]
    assert cmd[0] == str(Path("/tmp/fake-quiver") / "build/bin/flashanns_search")
    assert cmd[cmd.index("--index-dir") + 1] == str(workdir)
    assert cmd[cmd.index("--repeat") + 1] == "1"
    assert cmd[cmd.index("--output-ids") + 1] == str(workdir / "result_ids_0000.bin")
    assert cwd == Path("/tmp/fake-quiver")


def test_batch_query_uses_memory_backend_contract(monkeypatch, tmp_path):
    monkeypatch.setattr(
        Flashanns,
        "_resolve_gustann_home",
        lambda self, configured_home: Path("/tmp/fake-quiver"),
    )

    algo = Flashanns(
        "euclidean",
        {"flashanns_home": "/tmp/fake-quiver", "pq_size": 32},
        {
            "num_blocks": 756,
            "poll_threads": 6,
            "poll_contexts": 7,
            "pipe_width": 4,
            "ef_search": 40,
        },
    )

    workdir = tmp_path / "index"
    workdir.mkdir()
    algo._workdir = workdir
    algo._index_dir = workdir
    algo._data_type = "float"

    calls = []

    monkeypatch.setattr(algo, "_write_diskann_bin", lambda path, X: None)
    monkeypatch.setattr(algo, "_read_ids", lambda path, expected_topk: [[0]] * 2)
    monkeypatch.setattr(algo, "_run", lambda cmd, cwd: calls.append((cmd, cwd)) or "  Time: 0.100000 s\n")

    algo.batch_query(np.zeros((2, 4), dtype=np.float32), 10)

    cmd, _ = calls[0]
    assert "--ssd-list-file" not in cmd
    assert cmd[cmd.index("--data-type") + 1] == "float"


def test_batch_query_uses_native_flashanns_timing(monkeypatch, tmp_path):
    monkeypatch.setattr(
        Flashanns,
        "_resolve_gustann_home",
        lambda self, configured_home: Path("/tmp/fake-quiver"),
    )

    algo = Flashanns(
        "euclidean",
        {"flashanns_home": "/tmp/fake-quiver", "pq_size": 32},
        {
            "num_blocks": 756,
            "poll_threads": 6,
            "poll_contexts": 7,
            "pipe_width": 4,
            "ef_search": 40,
        },
    )

    workdir = tmp_path / "index"
    workdir.mkdir()
    algo._workdir = workdir
    algo._index_dir = workdir
    algo._data_type = "uint8"

    monkeypatch.setattr(algo, "_write_diskann_bin", lambda path, X: None)
    monkeypatch.setattr(algo, "_read_ids", lambda path, expected_topk: [[0]] * 5)
    monkeypatch.setattr(algo, "_run", lambda cmd, cwd: "  Time: 0.250000 s\n")

    algo.batch_query(np.zeros((5, 4), dtype=np.uint8), 10)

    assert algo.get_batch_latencies() == [0.05] * 5
    assert algo.get_additional()["best_search_time"] == 0.05


def test_batch_query_reads_output_ids_file(monkeypatch, tmp_path):
    monkeypatch.setattr(
        Flashanns,
        "_resolve_gustann_home",
        lambda self, configured_home: Path("/tmp/fake-quiver"),
    )

    algo = Flashanns(
        "euclidean",
        {"flashanns_home": "/tmp/fake-quiver", "pq_size": 32},
        {
            "num_blocks": 756,
            "poll_threads": 6,
            "poll_contexts": 7,
            "pipe_width": 4,
            "ef_search": 40,
        },
    )

    workdir = tmp_path / "index"
    workdir.mkdir()
    algo._workdir = workdir
    algo._index_dir = workdir
    algo._data_type = "uint8"

    reads = []
    monkeypatch.setattr(algo, "_write_diskann_bin", lambda path, X: None)
    monkeypatch.setattr(
        algo,
        "_read_ids",
        lambda path, expected_topk: reads.append((path, expected_topk)) or [[0]] * 5,
    )
    monkeypatch.setattr(algo, "_run", lambda cmd, cwd: "  Time: 0.250000 s\n")

    algo.batch_query(np.zeros((5, 4), dtype=np.uint8), 10)

    assert reads == [(workdir / "result_ids_0000.bin", 10)]


def test_batch_query_rejects_partial_result_sets(monkeypatch, tmp_path):
    monkeypatch.setattr(
        Flashanns,
        "_resolve_gustann_home",
        lambda self, configured_home: Path("/tmp/fake-quiver"),
    )

    algo = Flashanns(
        "euclidean",
        {"flashanns_home": "/tmp/fake-quiver", "pq_size": 32},
        {
            "num_blocks": 756,
            "poll_threads": 6,
            "poll_contexts": 7,
            "pipe_width": 4,
            "ef_search": 40,
        },
    )

    workdir = tmp_path / "index"
    workdir.mkdir()
    algo._workdir = workdir
    algo._index_dir = workdir
    algo._data_type = "uint8"

    monkeypatch.setattr(algo, "_write_diskann_bin", lambda path, X: None)
    monkeypatch.setattr(algo, "_run", lambda cmd, cwd: "  Time: 0.250000 s\n")
    monkeypatch.setattr(algo, "_read_ids", lambda path, expected_topk: [[0]] * 3)

    with pytest.raises(RuntimeError, match="Expected 5 result rows"):
        algo.batch_query(np.zeros((5, 4), dtype=np.uint8), 10)

def test_flashanns_missing_checkout_raises_file_not_found(monkeypatch):
    monkeypatch.setattr(
        Flashanns,
        "_required_paths",
        lambda self: [Path("missing/flashanns/binary")],
    )
    with pytest.raises(FileNotFoundError, match="FlashANN"):
        Flashanns(
            "euclidean",
            {"flashanns_home": "/tmp/missing-flashanns-home", "pq_size": 32},
            {
                "num_blocks": 756,
                "poll_threads": 6,
                "poll_contexts": 7,
                "pipe_width": 4,
                "ef_search": 40,
            },
        )
