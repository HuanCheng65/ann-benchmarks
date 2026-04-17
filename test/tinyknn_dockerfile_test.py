from pathlib import Path


def test_tinyknn_dockerfile_pins_compatible_build_toolchain():
    dockerfile = Path("ann_benchmarks/algorithms/tinyknn/Dockerfile").read_text(encoding="utf-8")

    assert "Cython<3" in dockerfile
    assert "numpy<2" in dockerfile
    assert "--no-build-isolation" in dockerfile
