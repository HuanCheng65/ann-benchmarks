from pathlib import Path


def test_nmslib_dockerfile_avoids_github_clone_during_build():
    dockerfile = Path("ann_benchmarks/algorithms/nmslib/Dockerfile").read_text(encoding="utf-8")

    assert "git clone https://github.com/searchivarius/nmslib.git" not in dockerfile
    assert "pip3 install" in dockerfile
    assert "pybind11" in dockerfile
    assert "nmslib" in dockerfile
