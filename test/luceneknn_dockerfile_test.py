from pathlib import Path


def test_luceneknn_dockerfile_uses_archived_pylucene_source_url():
    dockerfile = Path("ann_benchmarks/algorithms/luceneknn/Dockerfile").read_text(encoding="utf-8")

    assert "https://dlcdn.apache.org/lucene/pylucene/pylucene-9.7.0-src.tar.gz" not in dockerfile
    assert "https://archive.apache.org/dist/lucene/pylucene/pylucene-9.7.0-src.tar.gz" in dockerfile
