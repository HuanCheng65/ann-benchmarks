# Euclidean Distance Plot Fix Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Normalize Euclidean distance outputs for FlashANN, GustANN MEM, and GustANN original so result HDF5 files and `plot.py` use the same `L2` distance scale, then capture the benchmark outputs in a Markdown note.

**Architecture:** Apply the distance fix once in `GustannBase._read_distances`, because all three wrappers inherit from the same reader path. Keep the native binaries unchanged for this pass, convert squared `L2` distances to `L2` when the wrapper reads the result file, and write a running benchmark note under `docs/` that starts with FlashANN and accumulates the next two algorithm results.

**Tech Stack:** Python, pytest, HDF5 result files, ann-benchmarks plotting.

---

### Task 1: Lock The Distance Contract In Tests

**Files:**
- Modify: `test/gustann_test.py`

**Step 1: Write the failing test**

Add a test that writes a small binary distance file with squared Euclidean values and asserts `_read_distances()` returns square-rooted `L2` values for a Euclidean wrapper.

**Step 2: Run test to verify it fails**

Run: `./.venv/bin/pytest test/gustann_test.py::test_read_distances_normalizes_euclidean_l2_scale -q`
Expected: FAIL because `_read_distances()` currently returns the raw squared values.

**Step 3: Write minimal implementation**

Update `ann_benchmarks/algorithms/gustann/common.py` so Euclidean wrappers apply `sqrt` to the loaded distances before returning them.

**Step 4: Run test to verify it passes**

Run: `./.venv/bin/pytest test/gustann_test.py::test_read_distances_normalizes_euclidean_l2_scale -q`
Expected: PASS.

### Task 2: Capture Current Benchmark Results In Markdown

**Files:**
- Create: `docs/2026-04-20-sift1b-tmpfs-results.md`

**Step 1: Write the benchmark note**

Create a short document with:
- environment and dataset context
- FlashANN sweep results already completed
- sections reserved for GustANN MEM and GustANN original

**Step 2: Fill in current FlashANN table**

Record `blocks`, `repeat`, `QPS`, `best_search_time`, and `Recall@10` for the finished FlashANN runs.

### Task 3: Verify Plot Output With Corrected Recall Axis

**Files:**
- Verify: `logs/sift1b-uint8-tmpfs-20260420/flashanns-recall-qps.png`

**Step 1: Run focused tests**

Run: `./.venv/bin/pytest test/gustann_test.py test/flashanns_test.py test/gustann_original_test.py -q`
Expected: PASS.

**Step 2: Recompute the FlashANN-only plot**

Run `plot.py` against the FlashANN result set after the distance normalization lands.

**Step 3: Inspect the resulting plot**

Confirm the FlashANN points now appear around `Recall≈0.856` instead of collapsing near `0`.
