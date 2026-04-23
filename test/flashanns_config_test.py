from pathlib import Path

import yaml


def _load_config():
    config_path = Path(__file__).resolve().parents[1] / "ann_benchmarks" / "algorithms" / "flashanns" / "config.yml"
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _assert_flashanns_sweep(entry):
    run_group = entry["run_groups"]["baseline"]
    search_params = run_group["arg_groups"][0]["search_params"]

    assert [params["num_blocks"] for params in search_params] == [540, 648, 756, 864, 972]
    assert all(params["poll_threads"] == 6 for params in search_params)
    assert all(params["poll_contexts"] == 7 for params in search_params)
    assert all(params["pipe_width"] == 4 for params in search_params)
    assert all(params["ef_search"] == 30 for params in search_params)
    assert all(params["repeat"] == 20 for params in search_params)
    assert run_group["query_args"] == [[30]]


def test_flashanns_float_config_matches_l20_safe_sweep():
    config = _load_config()
    _assert_flashanns_sweep(config["float"]["euclidean"][0])


def test_flashanns_uint8_config_matches_l20_safe_sweep():
    config = _load_config()
    _assert_flashanns_sweep(config["uint8"]["euclidean"][0])
