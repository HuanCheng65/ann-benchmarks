# GustANN Benchmark-Safe Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Integrate original `GustANN` with a memory-backend benchmark-safe path and align `gustann-mem` so both report comparable search time without counting result export overhead.

**Architecture:** Introduce benchmark-focused CLI entrypoints in the external GustANN repositories that run the same search code path but emit a single machine-readable timing line and dump result ids after search completes. In `ann-benchmarks`, extract shared wrapper helpers, add an original-GustANN wrapper/config, and update the existing `gustann-mem` wrapper to parse native timing reports into `get_batch_latencies()`.

**Tech Stack:** Python, pytest, C++, CUDA, CMake, ANN-Benchmarks algorithm wrappers

---

### Task 1: Add failing tests for benchmark-safe wrapper behavior

**Files:**
- Modify: `test/gustann_test.py`
- Create: `test/gustann_original_test.py`

**Step 1: Write the failing tests**
- Verify the existing `gustann-mem` wrapper stores parsed native batch latency and exposes it via `get_batch_latencies()`.
- Verify the original `GustANN` wrapper invokes a benchmark binary with `--io_backend memory` and `--output_ids`.
- Verify stdout parsing rejects missing timing reports.

**Step 2: Run tests to verify they fail**

Run: `pytest -q test/gustann_test.py test/gustann_original_test.py`

**Step 3: Write minimal implementation**
- Add latency parsing helpers and wrapper support.
- Add the original wrapper module/config.

**Step 4: Run tests to verify they pass**

Run: `pytest -q test/gustann_test.py test/gustann_original_test.py`

### Task 2: Add benchmark-safe original GustANN binary

**Files:**
- Modify: `../GustANN/bin/CMakeLists.txt`
- Create: `../GustANN/bin/search_disk_bench.cu`

**Step 1: Add a bench-only entrypoint**
- Accept `--query`, `--index`, `--pq_data`, `--nav_graph`, `--data_type`, `--topk`, `--ef_search`, `-B`, `-T`, `-C`, `--io_backend`, `--output_ids`
- Restrict use to `memory` backend for now

**Step 2: Keep benchmark output machine-readable**
- Run the existing search path
- Emit a single `[REPORT] Time <seconds>` line after search completes
- Write ids in `(int32 nq, int32 topk, int32 ids...)` format after timing is recorded

**Step 3: Keep artifact binary untouched**
- Do not alter `search_disk_hybrid` semantics beyond any required shared helper extraction

### Task 3: Align gustann-mem benchmark timing

**Files:**
- Modify: `../gustann-mem/bin/CMakeLists.txt`
- Modify: `../gustann-mem/bin/search_disk.cu`

**Step 1: Add a benchmark-safe native timing report**
- Ensure output-id mode emits a single `[REPORT] Time <seconds>` line that reflects search time only
- Keep result-id export after that timing report

**Step 2: Preserve existing analysis paths**
- Do not break current CSV/reporting behavior outside benchmark mode

### Task 4: Wire ann-benchmarks to both variants

**Files:**
- Modify: `ann_benchmarks/algorithms/gustann/module.py`
- Modify: `ann_benchmarks/algorithms/gustann/config.yml`
- Create: `ann_benchmarks/algorithms/gustann_original/__init__.py`
- Create: `ann_benchmarks/algorithms/gustann_original/module.py`
- Create: `ann_benchmarks/algorithms/gustann_original/config.yml`

**Step 1: Extract shared wrapper pieces**
- Reuse index building, nav graph prep, id file parsing, and common path resolution patterns

**Step 2: Update `gustann-mem` wrapper**
- Invoke the benchmark-safe binary
- Parse `[REPORT] Time` from stdout
- Implement `get_batch_latencies()` so ANN-Benchmarks uses native time instead of Python wall time

**Step 3: Add original `GustANN` wrapper**
- Use the same index preparation flow
- Invoke the original bench binary with `--io_backend memory`
- Parse timing and ids exactly like the memory variant

### Task 5: Verify the integration

**Files:**
- No code changes

**Step 1: Run Python verification**

Run:
- `pytest -q test/gustann_test.py test/gustann_original_test.py`
- `python3 -m py_compile ann_benchmarks/algorithms/gustann/module.py ann_benchmarks/algorithms/gustann_original/module.py`

**Step 2: Run source-level smoke checks if feasible**

Run:
- `cmake --build ../GustANN/build --target search_disk_hybrid_bench`
- `cmake --build ../gustann-mem/build --target search_disk_mem_bench_float search_disk_mem_bench_uint8`

Expected:
- New bench binaries compile, or any external build blockers are captured explicitly
