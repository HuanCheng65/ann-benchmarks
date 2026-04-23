# SIFT1B Tmpfs Index Mirror Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a reusable tmpfs mirror flow for the SIFT1B query-time index files and let GustANN mem, GustANN original, and FlashANN read from that mirror during ann-benchmarks runs.

**Architecture:** Keep the hashed ann-benchmarks working directory as the canonical build/output directory, then add an optional runtime `index_dir_override` that repoints query-time search paths to a separate tmpfs mirror. Provide prepare/release shell helpers for mounting, syncing, and exporting the override path used by the benchmark commands.

**Tech Stack:** Python, pytest, shell, rsync, tmpfs, CMake, ann-benchmarks wrapper modules

---

### Task 1: Document the approved tmpfs mirror design

**Files:**
- Create: `docs/plans/2026-04-20-sift1b-tmpfs-index-mirror-design.md`
- Create: `docs/plans/2026-04-20-sift1b-tmpfs-index-mirror-plan.md`

**Step 1: Write the design doc**

Document:

- source and tmpfs mirror directories
- required query-time files
- wrapper override behavior
- prepare/release workflow
- failure handling and tests

**Step 2: Save the implementation plan**

List the wrapper, script, and test tasks in TDD order.

### Task 2: Add failing wrapper override tests

**Files:**
- Modify: `test/gustann_test.py`
- Modify: `test/flashanns_test.py`

**Step 1: Write the failing tests**

Add tests that prove:

- `fit()` keeps `_workdir` on the hashed disk directory but moves runtime paths to `index_dir_override`
- `ANNB_INDEX_DIR_OVERRIDE` acts as a fallback override
- FlashANN uses the override directory in `--index-dir`

**Step 2: Run tests to verify they fail**

Run:

```bash
cd /home/starrydream/ann-benchmarks
./.venv/bin/pytest test/gustann_test.py test/flashanns_test.py -q
```

Expected:

- failing assertions around missing override behavior

### Task 3: Add failing prepare-script tests

**Files:**
- Create: `test/sift1b_tmpfs_scripts_test.py`

**Step 1: Write the failing tests**

Cover:

- prepare script copies only the query-time file set in a local no-mount mode
- prepare script writes an env file exporting `ANNB_INDEX_DIR_OVERRIDE`
- release script syntax is valid

**Step 2: Run tests to verify they fail**

Run:

```bash
cd /home/starrydream/ann-benchmarks
./.venv/bin/pytest test/sift1b_tmpfs_scripts_test.py -q
```

Expected:

- missing script failures

### Task 4: Implement wrapper override support

**Files:**
- Modify: `ann_benchmarks/algorithms/gustann/common.py`

**Step 1: Add minimal implementation**

Implement:

- `index_dir_override` lookup from `index_params`
- environment fallback from `ANNB_INDEX_DIR_OVERRIDE`
- runtime index path resolution that keeps `_workdir` on disk and repoints `_index_dir`, `_index_prefix`, `_index_file`, `_pq_prefix`, and `_nav_prefix`
- validation that override directories contain the expected query-time files

**Step 2: Run wrapper tests**

Run:

```bash
cd /home/starrydream/ann-benchmarks
./.venv/bin/pytest test/gustann_test.py test/flashanns_test.py -q
```

Expected:

- passing override tests

### Task 5: Implement tmpfs prepare and release scripts

**Files:**
- Create: `scripts/prepare_sift1b_tmpfs.sh`
- Create: `scripts/release_sift1b_tmpfs.sh`
- Modify: `scripts/run_sift1b_uint8_benchmarks.sh`

**Step 1: Write minimal implementation**

Prepare script requirements:

- mount tmpfs unless already mounted
- sync the fixed query-time file set
- emit an env file exporting `ANNB_INDEX_DIR_OVERRIDE`
- support a local no-mount mode for tests

Release script requirements:

- check for active benchmark/search processes
- sync and unmount
- support no-mount local cleanup

Run script requirements:

- source the env file when present so standard benchmark commands use the tmpfs mirror automatically

**Step 2: Run script tests**

Run:

```bash
cd /home/starrydream/ann-benchmarks
./.venv/bin/pytest test/sift1b_tmpfs_scripts_test.py -q
bash -n scripts/prepare_sift1b_tmpfs.sh
bash -n scripts/release_sift1b_tmpfs.sh
bash -n scripts/run_sift1b_uint8_benchmarks.sh
```

Expected:

- tests pass
- shell syntax checks pass

### Task 6: Rebuild and verify runtime prerequisites

**Files:**
- None

**Step 1: Rebuild the native binaries**

Run:

```bash
cmake --build /home/starrydream/gustann-mem/build --target gen_small_file search_disk_mem_uint8 search_disk_mem_float -j"$(nproc)"
cmake --build /home/starrydream/GustANN/build --target gen_small_file search_disk_hybrid_bench -j"$(nproc)"
cmake --build /home/starrydream/quiver/build --target gen_small_file flashanns_search -j"$(nproc)"
```

Expected:

- all three builds succeed

**Step 2: Run the focused verification suite**

Run:

```bash
cd /home/starrydream/ann-benchmarks
./.venv/bin/pytest test/gustann_test.py test/flashanns_test.py test/sift1b_tmpfs_scripts_test.py -q
```

Expected:

- all tests pass

### Task 7: Record the operational commands

**Files:**
- Modify: `scripts/run_sift1b_uint8_benchmarks.sh`

**Step 1: Ensure the final operational flow is stable**

Document in comments or output:

- prepare command
- benchmark command
- release command

**Step 2: Final verification**

Run the shell syntax checks and focused pytest suite again after any final script edits.
