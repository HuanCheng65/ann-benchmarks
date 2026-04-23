import os
import subprocess
from pathlib import Path


def _write_sample_runtime_files(src_dir: Path) -> None:
    src_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "ann_disk.index",
        "ann_pq_compressed.bin",
        "ann_pq_pivots.bin",
        "ann_disk.index_centroids.bin",
        "ann_disk.index_medoids.bin",
        "nav_data.bin",
        "nav_index",
        "nav_index.data",
        "nav_index.tags",
        "map.txt",
        "extra.bin",
    ]:
        (src_dir / name).write_text(name, encoding="ascii")


def test_prepare_sift1b_tmpfs_script_copies_runtime_files_and_writes_env_file(tmp_path):
    root_dir = Path(__file__).resolve().parents[1]
    script = root_dir / "scripts" / "prepare_sift1b_tmpfs.sh"

    src_dir = tmp_path / "src"
    dst_dir = tmp_path / "dst"
    env_file = tmp_path / "sift1b-tmpfs.env"
    _write_sample_runtime_files(src_dir)

    env = os.environ.copy()
    env.update(
        {
            "SRC_DIR": str(src_dir),
            "TMPFS_DIR": str(dst_dir),
            "ENV_FILE": str(env_file),
            "SKIP_MOUNT": "1",
        }
    )

    subprocess.run(["bash", str(script)], check=True, env=env, cwd=root_dir)

    assert (dst_dir / "ann_disk.index").exists()
    assert (dst_dir / "nav_index").exists()
    assert not (dst_dir / "extra.bin").exists()
    assert env_file.read_text(encoding="utf-8").strip() == f'export ANNB_INDEX_DIR_OVERRIDE="{dst_dir}"'


def test_tmpfs_scripts_have_valid_bash_syntax():
    root_dir = Path(__file__).resolve().parents[1]
    for script_name in [
        "prepare_sift1b_tmpfs.sh",
        "release_sift1b_tmpfs.sh",
        "run_sift1b_uint8_benchmarks.sh",
    ]:
        subprocess.run(
            ["bash", "-n", str(root_dir / "scripts" / script_name)],
            check=True,
            cwd=root_dir,
        )
