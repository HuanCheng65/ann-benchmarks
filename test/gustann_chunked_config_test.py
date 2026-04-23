from pathlib import Path

import numpy as np
import yaml

from ann_benchmarks.algorithms.gustann.module import Gustann


def _load_config():
    config_path = Path(__file__).resolve().parents[1] / "ann_benchmarks" / "algorithms" / "gustann" / "config.yml"
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _assert_chunked_mode(entry):
    index_params = entry["run_groups"]["baseline"]["arg_groups"][0]["index_params"]
    assert index_params["host_register_mode"] == "single_gpu_chunked_mapped"
    assert index_params["host_register_chunk_gb"] == 32


def test_gustann_float_config_uses_chunked_single_gpu_host_registration():
    config = _load_config()
    _assert_chunked_mode(config["float"]["euclidean"][0])


def test_gustann_uint8_config_uses_chunked_single_gpu_host_registration():
    config = _load_config()
    _assert_chunked_mode(config["uint8"]["euclidean"][0])


def test_batch_query_passes_host_register_flags_to_native_binary(monkeypatch, tmp_path):
    monkeypatch.setattr(
        Gustann,
        "_resolve_gustann_home",
        lambda self, configured_home: Path("/tmp/fake-gustann"),
    )

    algo = Gustann(
        "euclidean",
        {
            "gustann_home": "/tmp/fake-gustann",
            "pq_size": 32,
            "host_register_mode": "single_gpu_chunked_mapped",
            "host_register_chunk_gb": 32,
        },
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
    assert cmd[cmd.index("--host_register_mode") + 1] == "single_gpu_chunked_mapped"
    assert cmd[cmd.index("--host_register_chunk_gb") + 1] == "32"
