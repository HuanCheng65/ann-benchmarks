from pathlib import Path

import yaml


def _load_config():
    config_path = Path(__file__).resolve().parents[1] / "ann_benchmarks" / "algorithms" / "gustann_original" / "config.yml"
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _assert_original_sweep(entry):
    run_group = entry["run_groups"]["baseline"]
    search_params = run_group["arg_groups"][0]["search_params"]

    assert [params["minibatch"] for params in search_params] == [64, 256, 1024, 10000]
    assert all(params["thread"] == 40 for params in search_params)
    assert all(params["ctx_per_thread"] == 1 for params in search_params)
    assert all(params["ef_search"] == 30 for params in search_params)
    assert run_group["query_args"] == [[30]]


def test_gustann_original_float_config_matches_memory_backend_throughput_sweep():
    config = _load_config()
    _assert_original_sweep(config["float"]["euclidean"][0])


def test_gustann_original_uint8_config_matches_memory_backend_throughput_sweep():
    config = _load_config()
    _assert_original_sweep(config["uint8"]["euclidean"][0])
