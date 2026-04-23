# GustANN Chunked Host Register Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a single-GPU chunked mapped host registration path to GustANN and wire ann-benchmarks to use it for large MEM benchmark runs without changing the existing multi-GPU path.

**Architecture:** GustANN keeps the current `portable_whole` registration path and gains a `single_gpu_chunked_mapped` path that registers file-backed host memory in fixed-size chunks and exposes a chunk table to device code. ann-benchmarks passes the new mode and chunk size explicitly for large MEM runs and constrains those runs to one visible GPU.

**Tech Stack:** C++, CUDA, Python, pytest

---

### Task 1: Add failing tests for chunked address translation

**Files:**
- Create: `/home/starrydream/gustann-mem/tests/page_data_chunk_test.cu`
- Modify: `/home/starrydream/gustann-mem/CMakeLists.txt`

**Step 1: Write the failing test**

Add CUDA-side tests that construct a synthetic chunk table and verify:

- a node fully inside chunk 0 resolves correctly
- a node in chunk 1 resolves correctly
- a node near a chunk boundary resolves correctly

**Step 2: Run test to verify it fails**

Run: `ctest --test-dir /home/starrydream/gustann-mem/build -R page_data_chunk_test --output-on-failure`

Expected: FAIL because `PageData` only supports a single `disk` pointer.

**Step 3: Write minimal implementation**

Update `PageData` in `/home/starrydream/gustann-mem/src/impl/page_wrapper.cuh` to support chunked addressing.

**Step 4: Run test to verify it passes**

Run: `ctest --test-dir /home/starrydream/gustann-mem/build -R page_data_chunk_test --output-on-failure`

Expected: PASS

### Task 2: Add failing tests for config and wiring

**Files:**
- Create: `/home/starrydream/ann-benchmarks/test/gustann_chunked_config_test.py`
- Modify: `/home/starrydream/ann-benchmarks/ann_benchmarks/algorithms/gustann/config.yml`
- Modify: `/home/starrydream/ann-benchmarks/ann_benchmarks/algorithms/gustann/__init__.py`

**Step 1: Write the failing test**

Cover:

- GustANN config includes the new host registration fields
- ann-benchmarks surfaces those fields to the constructor/runtime config

**Step 2: Run test to verify it fails**

Run: `pytest /home/starrydream/ann-benchmarks/test/gustann_chunked_config_test.py -v`

Expected: FAIL because the config fields do not exist yet.

**Step 3: Write minimal implementation**

Add the new config fields and constructor plumbing.

**Step 4: Run test to verify it passes**

Run: `pytest /home/starrydream/ann-benchmarks/test/gustann_chunked_config_test.py -v`

Expected: PASS

### Task 3: Implement chunk-aware MEM backend context

**Files:**
- Modify: `/home/starrydream/gustann-mem/include/ssd_search.hpp`
- Modify: `/home/starrydream/gustann-mem/src/ssd_search.cu`

**Step 1: Write the failing test**

Extend backend tests or add a focused initialization test that expects chunk metadata to be created for chunked mode.

**Step 2: Run test to verify it fails**

Run: `ctest --test-dir /home/starrydream/gustann-mem/build -R ssd_search_chunked_init --output-on-failure`

Expected: FAIL because backend context only tracks one mapping.

**Step 3: Write minimal implementation**

Add:

- `HostRegisterMode`
- `HostRegisterChunk`
- chunk geometry fields
- cleanup for both whole and chunked paths

**Step 4: Run test to verify it passes**

Run: `ctest --test-dir /home/starrydream/gustann-mem/build -R ssd_search_chunked_init --output-on-failure`

Expected: PASS

### Task 4: Implement chunked mapped registration path

**Files:**
- Modify: `/home/starrydream/gustann-mem/src/ssd_search.cu`

**Step 1: Write the failing test**

Add a small integration test that:

- creates a tiny file-backed region
- initializes the chunked MEM backend
- verifies all chunks receive device pointers

**Step 2: Run test to verify it fails**

Run: `ctest --test-dir /home/starrydream/gustann-mem/build -R ssd_search_chunked_register --output-on-failure`

Expected: FAIL because chunked registration is missing.

**Step 3: Write minimal implementation**

Implement:

- visible-GPU validation
- chunked file-backed `mmap`
- `cudaHostRegisterMapped`
- `cudaHostGetDevicePointer`
- chunk-level logging and cleanup

**Step 4: Run test to verify it passes**

Run: `ctest --test-dir /home/starrydream/gustann-mem/build -R ssd_search_chunked_register --output-on-failure`

Expected: PASS

### Task 5: Wire chunked pointers into device-side search

**Files:**
- Modify: `/home/starrydream/gustann-mem/src/impl/page_wrapper.cuh`
- Modify: `/home/starrydream/gustann-mem/src/ssd_search.cu`

**Step 1: Write the failing test**

Add a small end-to-end search fixture where device code must load nodes from chunk 1 and return a deterministic result.

**Step 2: Run test to verify it fails**

Run: `ctest --test-dir /home/starrydream/gustann-mem/build -R ssd_search_chunked_device_access --output-on-failure`

Expected: FAIL because the search path still assumes one `disk` pointer.

**Step 3: Write minimal implementation**

Pass chunk tables to the device and update node access helpers to resolve chunk-local pointers.

**Step 4: Run test to verify it passes**

Run: `ctest --test-dir /home/starrydream/gustann-mem/build -R ssd_search_chunked_device_access --output-on-failure`

Expected: PASS

### Task 6: Wire ann-benchmarks single-GPU path

**Files:**
- Modify: `/home/starrydream/ann-benchmarks/ann_benchmarks/algorithms/gustann/config.yml`
- Modify: `/home/starrydream/ann-benchmarks/ann_benchmarks/algorithms/gustann/__init__.py`
- Modify: `/home/starrydream/ann-benchmarks/scripts/run_sift1b_uint8_benchmarks.sh`

**Step 1: Write the failing test**

Add assertions that the SIFT1B MEM benchmark path requests:

- `host_register_mode=single_gpu_chunked_mapped`
- `host_register_chunk_gb=32`
- single visible GPU for the launched process

**Step 2: Run test to verify it fails**

Run: `pytest /home/starrydream/ann-benchmarks/test/gustann_chunked_config_test.py -v`

Expected: FAIL because the benchmark path still uses the old config.

**Step 3: Write minimal implementation**

Update the benchmark config and runner script wiring.

**Step 4: Run test to verify it passes**

Run: `pytest /home/starrydream/ann-benchmarks/test/gustann_chunked_config_test.py -v`

Expected: PASS

### Task 7: Verify and document

**Files:**
- Modify: `/home/starrydream/ann-benchmarks/docs/plans/2026-04-19-cuda-host-register-probe-summary.md`

**Step 1: Run focused tests**

Run:

- `pytest /home/starrydream/ann-benchmarks/test/gustann_chunked_config_test.py -v`
- `ctest --test-dir /home/starrydream/gustann-mem/build -R 'page_data_chunk_test|ssd_search_chunked' --output-on-failure`

**Step 2: Run a small manual smoke check**

Run a small GustANN MEM initialization with:

- one visible GPU
- `host_register_mode=single_gpu_chunked_mapped`
- small chunk size for a tiny fixture index

**Step 3: Record findings**

Update the probe summary or add a short note describing the production mitigation now implemented.
