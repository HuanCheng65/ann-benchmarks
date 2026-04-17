from pathlib import Path


def test_pgvectorscale_uses_matching_cargo_pgrx_version():
    dockerfile = Path("ann_benchmarks/algorithms/pgvectorscale/Dockerfile").read_text()

    assert "cargo install --locked cargo-pgrx@0.16.1" in dockerfile
    assert "cargo install --locked cargo-pgrx@0.12.9" not in dockerfile
