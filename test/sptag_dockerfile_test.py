from pathlib import Path


def test_sptag_dockerfile_does_not_apply_stale_mutex_patch():
    dockerfile = Path("ann_benchmarks/algorithms/sptag/Dockerfile").read_text(encoding="utf-8")

    assert "bd9c25d1409325ac45ebeb7f1e8fc87d03ec478c.patch" not in dockerfile
    assert "git apply" not in dockerfile
