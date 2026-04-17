from pathlib import Path


def test_diskann_dockerfile_uses_current_intel_oneapi_repo_setup():
    dockerfile = Path("ann_benchmarks/algorithms/diskann/Dockerfile").read_text(encoding="utf-8")

    assert "apt-key add" not in dockerfile
    assert "apt.repos.intel.com/mkl" not in dockerfile
    assert "signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg" in dockerfile
    assert "apt.repos.intel.com/oneapi" in dockerfile
    assert "intel-oneapi-mkl" in dockerfile
