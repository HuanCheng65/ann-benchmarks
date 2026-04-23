from pathlib import Path
import struct

import numpy as np
import pytest

from ann_benchmarks.algorithms.gustann import common as gustann_common
from ann_benchmarks.algorithms.gustann.module import Gustann


def _touch_runtime_index_files(index_dir: Path) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "ann_disk.index",
        "ann_pq_compressed.bin",
        "ann_pq_pivots.bin",
        "ann_disk.index_centroids.bin",
        "ann_disk.index_medoids.bin",
        "nav_data.bin",
        "nav_index",
        "nav_index.data",
        "nav_index.tags",
        "map.txt",
    ]:
        (index_dir / name).write_bytes(b"test")


def test_batch_query_passes_index_prefix_to_search_binary(monkeypatch, tmp_path):
    monkeypatch.setattr(
        Gustann,
        "_resolve_gustann_home",
        lambda self, configured_home: Path("/tmp/fake-gustann"),
    )

    algo = Gustann(
        "euclidean",
        {"gustann_home": "/tmp/fake-gustann"},
        {
            "minibatch": 0,
            "cache_pct": 0,
            "blocks_per_sm": 1,
            "pipeline_width": 0,
            "ef_search": 40,
        },
    )

    workdir = tmp_path / "index"
    workdir.mkdir()
    algo._workdir = workdir
    algo._index_dir = workdir
    algo._index_prefix = workdir / "ann"
    algo._data_type = "uint8"
    algo.set_query_arguments(20)

    calls = []

    monkeypatch.setattr(algo, "_write_diskann_bin", lambda path, X: None)
    monkeypatch.setattr(algo, "_read_ids", lambda path, expected_topk: [[0]] * 2)
    monkeypatch.setattr(algo, "_read_distances", lambda path, expected_topk: [[0.0]] * 2)
    monkeypatch.setattr(algo, "_run", lambda cmd, cwd: calls.append((cmd, cwd)) or "[REPORT] BATCH_LAT_MS=1.000\n")

    algo.batch_query(np.zeros((2, 4), dtype=np.uint8), 10)

    cmd, cwd = calls[0]
    index_dir_pos = cmd.index("--index-dir") + 1
    assert cmd[index_dir_pos] == str(workdir / "ann")
    assert cmd[cmd.index("--output_distances") + 1] == str(workdir / "result_distances_0000.bin")
    assert cwd == Path("/tmp/fake-gustann")


def test_batch_query_invokes_search_binary_once_for_all_queries(monkeypatch, tmp_path):
    monkeypatch.setattr(
        Gustann,
        "_resolve_gustann_home",
        lambda self, configured_home: Path("/tmp/fake-gustann"),
    )

    algo = Gustann(
        "euclidean",
        {"gustann_home": "/tmp/fake-gustann"},
        {
            "minibatch": 2,
            "cache_pct": 0,
            "blocks_per_sm": 1,
            "pipeline_width": 0,
            "ef_search": 40,
        },
    )

    workdir = tmp_path / "index"
    workdir.mkdir()
    algo._workdir = workdir
    algo._index_dir = workdir
    algo._index_prefix = workdir / "ann"
    algo._data_type = "uint8"
    algo.set_query_arguments(20)

    writes = []
    runs = []
    reads = []

    monkeypatch.setattr(algo, "_write_diskann_bin", lambda path, X: writes.append((path, X.shape)))
    monkeypatch.setattr(
        algo,
        "_read_ids",
        lambda path, expected_topk: reads.append((path, expected_topk)) or [[0]] * 5,
    )
    monkeypatch.setattr(
        algo,
        "_read_distances",
        lambda path, expected_topk: reads.append((path, expected_topk)) or [[0.0]] * 5,
    )
    monkeypatch.setattr(algo, "_run", lambda cmd, cwd: runs.append((cmd, cwd)) or "[REPORT] BATCH_LAT_MS=1.000\n")

    algo.batch_query(np.zeros((5, 4), dtype=np.uint8), 10)

    assert len(writes) == 1
    assert writes[0][0] == workdir / "queries_0000.bin"
    assert writes[0][1] == (5, 4)

    assert len(runs) == 1
    cmd, cwd = runs[0]
    batch_size_pos = cmd.index("--batch_size_list") + 1
    assert cmd[batch_size_pos] == "2"
    assert cwd == Path("/tmp/fake-gustann")

    assert reads == [
        (workdir / "result_ids_0000.bin", 10),
        (workdir / "result_distances_0000.bin", 10),
    ]


def test_batch_query_uses_readme_style_search_flags(monkeypatch, tmp_path):
    monkeypatch.setattr(
        Gustann,
        "_resolve_gustann_home",
        lambda self, configured_home: Path("/tmp/fake-gustann"),
    )

    algo = Gustann(
        "euclidean",
        {"gustann_home": "/tmp/fake-gustann", "pq_size": 32},
        {
            "minibatch": 64,
            "cache_pct": 0,
            "blocks_per_sm": 1,
            "pipeline_width": 0,
            "ef_search": 40,
        },
    )

    workdir = tmp_path / "index"
    workdir.mkdir()
    algo._workdir = workdir
    algo._index_prefix = workdir / "ann"
    algo._data_type = "uint8"
    algo.set_query_arguments(30)

    calls = []

    monkeypatch.setattr(algo, "_write_diskann_bin", lambda path, X: None)
    monkeypatch.setattr(algo, "_read_ids", lambda path, expected_topk: [[0]] * 5)
    monkeypatch.setattr(algo, "_read_distances", lambda path, expected_topk: [[0.0]] * 5)
    monkeypatch.setattr(algo, "_run", lambda cmd, cwd: calls.append((cmd, cwd)) or "[REPORT] BATCH_LAT_MS=1.000\n")

    algo.batch_query(np.zeros((5, 4), dtype=np.uint8), 10)

    cmd, _ = calls[0]
    assert "--ef_search_list" in cmd
    assert cmd[cmd.index("--ef_search_list") + 1] == "30"
    assert "--pq_dim" in cmd
    assert cmd[cmd.index("--pq_dim") + 1] == "32"


def test_batch_query_rejects_partial_result_sets(monkeypatch, tmp_path):
    monkeypatch.setattr(
        Gustann,
        "_resolve_gustann_home",
        lambda self, configured_home: Path("/tmp/fake-gustann"),
    )

    algo = Gustann(
        "euclidean",
        {"gustann_home": "/tmp/fake-gustann", "pq_size": 32},
        {
            "minibatch": 32,
            "cache_pct": 0,
            "blocks_per_sm": 1,
            "pipeline_width": 0,
            "ef_search": 40,
        },
    )

    workdir = tmp_path / "index"
    workdir.mkdir()
    algo._workdir = workdir
    algo._index_prefix = workdir / "ann"
    algo._data_type = "uint8"

    monkeypatch.setattr(algo, "_write_diskann_bin", lambda path, X: None)
    monkeypatch.setattr(algo, "_run", lambda cmd, cwd: "[REPORT] BATCH_LAT_MS=1.000\n")
    monkeypatch.setattr(algo, "_read_ids", lambda path, expected_topk: [[0]] * 32)
    monkeypatch.setattr(algo, "_read_distances", lambda path, expected_topk: [[0.0]] * 32)

    try:
        algo.batch_query(np.zeros((100, 4), dtype=np.uint8), 10)
    except RuntimeError as exc:
        assert "Expected 100 result rows" in str(exc)
    else:
        raise AssertionError("Expected batch_query to reject partial result sets")


def test_batch_query_uses_user_visible_batch_latency(monkeypatch, tmp_path):
    monkeypatch.setattr(
        Gustann,
        "_resolve_gustann_home",
        lambda self, configured_home: Path("/tmp/fake-gustann"),
    )

    algo = Gustann(
        "euclidean",
        {"gustann_home": "/tmp/fake-gustann", "pq_size": 32},
        {
            "minibatch": 2,
            "cache_pct": 0,
            "blocks_per_sm": 1,
            "pipeline_width": 0,
            "ef_search": 40,
        },
    )

    workdir = tmp_path / "index"
    workdir.mkdir()
    algo._workdir = workdir
    algo._index_prefix = workdir / "ann"
    algo._data_type = "uint8"

    monkeypatch.setattr(algo, "_write_diskann_bin", lambda path, X: None)
    monkeypatch.setattr(algo, "_read_ids", lambda path, expected_topk: [[0]] * 5)
    monkeypatch.setattr(algo, "_read_distances", lambda path, expected_topk: [[1.0]] * 5)
    monkeypatch.setattr(algo, "_run", lambda cmd, cwd: "[REPORT] Time 0.250\n")

    algo.batch_query(np.zeros((5, 4), dtype=np.uint8), 10)

    expected_batch_latency = 0.25 / 3.0
    assert algo.get_batch_latencies() == [expected_batch_latency] * 5
    assert algo.get_additional()["best_search_time"] == 0.05


def test_batch_query_requires_native_timing_report(monkeypatch, tmp_path):
    monkeypatch.setattr(
        Gustann,
        "_resolve_gustann_home",
        lambda self, configured_home: Path("/tmp/fake-gustann"),
    )

    algo = Gustann(
        "euclidean",
        {"gustann_home": "/tmp/fake-gustann", "pq_size": 32},
        {
            "minibatch": 2,
            "cache_pct": 0,
            "blocks_per_sm": 1,
            "pipeline_width": 0,
            "ef_search": 40,
        },
    )

    workdir = tmp_path / "index"
    workdir.mkdir()
    algo._workdir = workdir
    algo._index_prefix = workdir / "ann"
    algo._data_type = "uint8"

    monkeypatch.setattr(algo, "_write_diskann_bin", lambda path, X: None)
    monkeypatch.setattr(algo, "_read_ids", lambda path, expected_topk: [[0]] * 5)
    monkeypatch.setattr(algo, "_read_distances", lambda path, expected_topk: [[0.0]] * 5)
    monkeypatch.setattr(algo, "_run", lambda cmd, cwd: "search completed\n")

    with pytest.raises(RuntimeError, match="timing report"):
        algo.batch_query(np.zeros((5, 4), dtype=np.uint8), 10)


def test_read_distances_normalizes_euclidean_l2_scale(monkeypatch, tmp_path):
    monkeypatch.setattr(
        Gustann,
        "_resolve_gustann_home",
        lambda self, configured_home: Path("/tmp/fake-gustann"),
    )

    algo = Gustann(
        "euclidean",
        {"gustann_home": "/tmp/fake-gustann", "pq_size": 32},
        {
            "minibatch": 64,
            "cache_pct": 0,
            "blocks_per_sm": 1,
            "pipeline_width": 0,
            "ef_search": 40,
        },
    )

    path = tmp_path / "result_distances_0000.bin"
    with path.open("wb") as f:
        f.write(struct.pack("ii", 2, 2))
        f.write(np.asarray([4.0, 9.0, 16.0, 25.0], dtype=np.float32).tobytes())

    distances = algo._read_distances(path, 2)

    assert distances == [[2.0, 3.0], [4.0, 5.0]]


def test_batch_query_exposes_native_distances(monkeypatch, tmp_path):
    monkeypatch.setattr(
        Gustann,
        "_resolve_gustann_home",
        lambda self, configured_home: Path("/tmp/fake-gustann"),
    )

    algo = Gustann(
        "euclidean",
        {"gustann_home": "/tmp/fake-gustann", "pq_size": 32},
        {
            "minibatch": 2,
            "cache_pct": 0,
            "blocks_per_sm": 1,
            "pipeline_width": 0,
            "ef_search": 40,
        },
    )

    workdir = tmp_path / "index"
    workdir.mkdir()
    algo._workdir = workdir
    algo._index_prefix = workdir / "ann"
    algo._data_type = "uint8"

    monkeypatch.setattr(algo, "_write_diskann_bin", lambda path, X: None)
    monkeypatch.setattr(algo, "_run", lambda cmd, cwd: "[REPORT] BATCH_LAT_MS=1.000\n")
    monkeypatch.setattr(algo, "_read_ids", lambda path, expected_topk: [[0, 1]] * 5)
    monkeypatch.setattr(algo, "_read_distances", lambda path, expected_topk: [[1.0, 2.0]] * 5)

    algo.batch_query(np.zeros((5, 4), dtype=np.uint8), 2)

    assert algo.get_batch_distances() == [[1.0, 2.0]] * 5


def test_fit_uses_index_dir_override_for_runtime_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(
        Gustann,
        "_resolve_gustann_home",
        lambda self, configured_home: Path("/tmp/fake-gustann"),
    )

    override_dir = tmp_path / "tmpfs-index"
    _touch_runtime_index_files(override_dir)

    algo = Gustann(
        "euclidean",
        {
            "gustann_home": "/tmp/fake-gustann",
            "index_dir_override": str(override_dir),
        },
        {
            "minibatch": 64,
            "cache_pct": 0,
            "blocks_per_sm": 1,
            "pipeline_width": 0,
            "ef_search": 40,
        },
    )
    algo._root = tmp_path / "work"

    monkeypatch.setattr(algo, "_write_diskann_bin", lambda path, X: None)
    monkeypatch.setattr(algo, "_run_diskann_build", lambda *args, **kwargs: None)
    monkeypatch.setattr(algo, "_run_nav_build", lambda *args, **kwargs: None)

    algo.fit(np.zeros((4, 4), dtype=np.uint8))

    assert algo._workdir != override_dir
    assert algo._index_dir == override_dir
    assert algo._index_prefix == override_dir / "ann"
    assert algo._index_file == override_dir / "ann_disk.index"
    assert algo._pq_prefix == override_dir / "ann_pq"
    assert algo._nav_prefix == override_dir / "nav_index"


def test_fit_uses_environment_index_dir_override_when_config_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(
        Gustann,
        "_resolve_gustann_home",
        lambda self, configured_home: Path("/tmp/fake-gustann"),
    )

    override_dir = tmp_path / "tmpfs-index"
    _touch_runtime_index_files(override_dir)
    monkeypatch.setenv("ANNB_INDEX_DIR_OVERRIDE", str(override_dir))

    algo = Gustann(
        "euclidean",
        {"gustann_home": "/tmp/fake-gustann"},
        {
            "minibatch": 64,
            "cache_pct": 0,
            "blocks_per_sm": 1,
            "pipeline_width": 0,
            "ef_search": 40,
        },
    )
    algo._root = tmp_path / "work"

    monkeypatch.setattr(algo, "_write_diskann_bin", lambda path, X: None)
    monkeypatch.setattr(algo, "_run_diskann_build", lambda *args, **kwargs: None)
    monkeypatch.setattr(algo, "_run_nav_build", lambda *args, **kwargs: None)

    algo.fit(np.zeros((4, 4), dtype=np.uint8))

    assert algo._index_dir == override_dir
    assert algo._index_prefix == override_dir / "ann"


def test_fit_skips_materializing_train_data_when_runtime_index_override_is_complete(monkeypatch, tmp_path):
    monkeypatch.setattr(
        Gustann,
        "_resolve_gustann_home",
        lambda self, configured_home: Path("/tmp/fake-gustann"),
    )

    override_dir = tmp_path / "tmpfs-index"
    _touch_runtime_index_files(override_dir)

    algo = Gustann(
        "euclidean",
        {
            "gustann_home": "/tmp/fake-gustann",
            "index_dir_override": str(override_dir),
        },
        {
            "minibatch": 64,
            "cache_pct": 0,
            "blocks_per_sm": 1,
            "pipeline_width": 0,
            "ef_search": 40,
        },
    )
    algo._root = tmp_path / "work"

    class FakeLazyArray:
        dtype = np.dtype(np.uint8)
        shape = (1_000_000_000, 128)

    monkeypatch.setattr(
        gustann_common.np,
        "ascontiguousarray",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected materialization")),
    )
    monkeypatch.setattr(
        algo,
        "_write_diskann_bin",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected base.bin write")),
    )
    monkeypatch.setattr(
        algo,
        "_run_diskann_build",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected index build")),
    )
    monkeypatch.setattr(
        algo,
        "_run_nav_build",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected nav build")),
    )

    algo.fit(FakeLazyArray())

    assert algo._index_dir == override_dir
    assert algo._workdir != override_dir
    assert not (algo._workdir / "base.bin").exists()
