# SIFT100M Float Dataset Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `float32` SIFT100M dataset path and automate end-to-end benchmark execution on it.

**Architecture:** Extend the existing raw-data converter with an output dtype switch, register a new external dataset alias, and add a shell wrapper that generates the dataset and runs the benchmark commands in the right order. Verification should cover dataset registration and dtype conversion behavior before launching any long-running job.

**Tech Stack:** Python, h5py, numpy, shell, ANN-Benchmarks dataset registry

---

### Task 1: Add failing tests for float dataset support

**Files:**
- Modify: `test/distance_test.py`
- Create: `test/sift100m_float_dataset_test.py`

**Step 1: Write failing tests**
- Verify `ann_benchmarks.datasets.DATASETS` contains `sift100m-128-euclidean-float`
- Verify the conversion helper can emit `float32` train/test datasets while preserving metadata and GT arrays

**Step 2: Run tests to verify failure**

Run: `pytest -q test/sift100m_float_dataset_test.py`

**Step 3: Implement minimal code**
- Add the new dataset alias
- Expose conversion helpers that tests can call directly

**Step 4: Run tests to verify they pass**

Run: `pytest -q test/sift100m_float_dataset_test.py`

### Task 2: Extend the raw converter

**Files:**
- Modify: `scripts/convert_sift100m_to_hdf5.py`

**Step 1: Add output dtype support**
- Allow `uint8` and `float32`
- Set `point_type` metadata accordingly
- Cast train/test blocks to the requested dtype while keeping GT ids/distances unchanged

**Step 2: Keep CLI backward-compatible**
- Preserve the current default `uint8`
- Add explicit `--output-dtype` support

**Step 3: Verify with tests**

Run: `pytest -q test/sift100m_float_dataset_test.py`

### Task 3: Register the new dataset and add orchestration

**Files:**
- Modify: `ann_benchmarks/datasets.py`
- Create: `scripts/run_sift100m_float_benchmarks.sh`

**Step 1: Register `sift100m-128-euclidean-float`**
- Use `external_hdf5_dataset` just like the existing prebuilt SIFT100M dataset

**Step 2: Add orchestration script**
- Generate `data/sift100m-128-euclidean-float.hdf5` if missing
- Run Docker-backed algorithms on `sift100m-128-euclidean-float`
- Run local `gustann-mem` separately on the same dataset
- Log to dedicated files suitable for `tmux`/background use

**Step 3: Smoke-test the script**
- Generate only, or dry-run the commands

### Task 4: Verify and launch

**Files:**
- No code changes

**Step 1: Run verification**

Run:
- `pytest -q test/sift100m_float_dataset_test.py test/distance_test.py`
- `python3 -m py_compile ann_benchmarks/datasets.py scripts/convert_sift100m_to_hdf5.py`

**Step 2: Generate the float dataset**

Run the converter against:
- `../shared_datasets/sift100m/base.100m.u8bin`
- `../shared_datasets/sift100m/query.public.10K.u8bin`
- `../shared_datasets/sift100m/groundtruth.100m.ibin`

**Step 3: Launch long-running benchmark jobs**
- Start the orchestration script in a detached session or background log runner
- Report the exact command and log file locations
