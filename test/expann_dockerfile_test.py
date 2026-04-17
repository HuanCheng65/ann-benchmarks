from pathlib import Path


def test_expann_dockerfile_does_not_depend_on_github_api_cache_bust():
    dockerfile = Path("ann_benchmarks/algorithms/expann/Dockerfile").read_text(encoding="utf-8")

    assert "https://api.github.com/repos/jacketsj/expANN/git/refs/heads/ann-benchmarks-stable-v1" not in dockerfile
    assert "git clone -b ann-benchmarks-stable-v1 https://github.com/jacketsj/expANN.git" in dockerfile
