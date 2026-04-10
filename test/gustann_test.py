from pathlib import Path

import numpy as np

from ann_benchmarks.algorithms.gustann.module import Gustann


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
    monkeypatch.setattr(algo, "_run", lambda cmd, cwd: calls.append((cmd, cwd)))

    algo.batch_query(np.zeros((2, 4), dtype=np.uint8), 10)

    cmd, cwd = calls[0]
    index_dir_pos = cmd.index("--index-dir") + 1
    assert cmd[index_dir_pos] == str(workdir / "ann")
    assert cwd == Path("/tmp/fake-gustann")
