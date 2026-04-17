from pathlib import Path


def test_milvus_dockerfile_pins_marshmallow_below_v4_for_pymilvus():
    dockerfile = Path("ann_benchmarks/algorithms/milvus/Dockerfile").read_text(encoding="utf-8")

    assert "pymilvus==2.4.1" in dockerfile
    assert "marshmallow<4" in dockerfile
