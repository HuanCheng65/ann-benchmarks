# Quiver FlashANNS Branch Refresh Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refresh local `huancheng/flashanns-result-ids` with relevant upstream FlashANNS fixes, preserve current result-ids and debugging work, and rerun focused FlashANNS validation.

**Architecture:** Keep the ann-benchmarks-facing result-ids branch as the working branch, preserve current local debugging work on that branch, then selectively integrate upstream FlashANNS commits and port any stall-fix logic that maps onto the FlashANNS lane scheduler. Validate in native FlashANNS first, then rerun the ann-benchmarks FlashANNS path.

**Tech Stack:** git, CMake, CUDA, FlashANNS, ann-benchmarks, shell, pytest-style native smoke tests

---

### Task 1: Record the approved design

**Files:**
- Create: `docs/plans/2026-04-20-quiver-flashanns-branch-refresh-design.md`
- Create: `docs/plans/2026-04-20-quiver-flashanns-branch-refresh-plan.md`

**Step 1: Save the design doc**

Write the branch refresh goal, selected upstream commits, validation flow, and success criteria.

**Step 2: Save the implementation plan**

Write the task-by-task implementation plan in TDD order.

### Task 2: Preserve current local FlashANNS debugging work on the target branch

**Files:**
- Modify: `/home/starrydream/quiver/bin/CMakeLists.txt`
- Modify: `/home/starrydream/quiver/src/flashanns/io_control.cuh`
- Modify: `/home/starrydream/quiver/src/flashanns/kernel.cuh`
- Modify: `/home/starrydream/quiver/src/flashanns/search.cu`
- Modify: `/home/starrydream/quiver/src/flashanns/shared/index/nav_kernel.cuh`
- Modify: `/home/starrydream/quiver/src/flashanns/shared/io/mem_loader.cpp`
- Modify: `/home/starrydream/quiver/src/gustann/shared/io/mem_loader.cpp`
- Modify: `/home/starrydream/quiver/src/shared/io/mem_loader.cpp`
- Create: `/home/starrydream/quiver/bin/flashanns_mem_loader_test.cpp`
- Create: `/home/starrydream/quiver/bin/flashanns_nav_entry_test.cu`
- Create: `/home/starrydream/quiver/bin/flashanns_node_validation_test.cu`

**Step 1: Switch to the working branch**

Run:

```bash
git -C /home/starrydream/quiver switch -c huancheng/flashanns-result-ids
```

Expected:

- local branch points at the result-ids branch tip

**Step 2: Commit the current local debugging baseline**

Run:

```bash
git -C /home/starrydream/quiver add \
  bin/CMakeLists.txt \
  src/flashanns/io_control.cuh \
  src/flashanns/kernel.cuh \
  src/flashanns/search.cu \
  src/flashanns/shared/index/nav_kernel.cuh \
  src/flashanns/shared/io/mem_loader.cpp \
  src/gustann/shared/io/mem_loader.cpp \
  src/shared/io/mem_loader.cpp \
  bin/flashanns_mem_loader_test.cpp \
  bin/flashanns_nav_entry_test.cu \
  bin/flashanns_node_validation_test.cu
git -C /home/starrydream/quiver commit -m "debug: add flashanns entry and mem loader diagnostics"
```

Expected:

- working tree is clean before upstream integration

### Task 3: Integrate upstream FlashANNS refresh commits

**Files:**
- Modify: `/home/starrydream/quiver/src/flashanns/io_control.cuh`
- Modify: `/home/starrydream/quiver/src/flashanns/kernel.cuh`
- Modify: `/home/starrydream/quiver/src/flashanns/search.cu`

**Step 1: Cherry-pick the direct FlashANNS refactor commit**

Run:

```bash
git -C /home/starrydream/quiver cherry-pick 1819760
```

Expected:

- direct FlashANNS files update
- conflicts resolved in favor of preserving current diagnostics unless the upstream logic supersedes them cleanly

**Step 2: Inspect upstream stall-fix commits as references**

Run:

```bash
git -C /home/starrydream/quiver show e74593a -- src/quiver/kernel.cuh src/quiver/search.cu src/quiver/io_control.cuh
git -C /home/starrydream/quiver show 6aa0c57 -- src/quiver/kernel.cuh src/quiver/search.cu
```

Expected:

- identify exact slot-retirement and pipe-loop changes worth porting

### Task 4: Write the failing stall and correctness tests

**Files:**
- Modify: `/home/starrydream/quiver/bin/flashanns_nav_entry_test.cu`
- Modify: `/home/starrydream/quiver/bin/flashanns_node_validation_test.cu`

**Step 1: Add or extend failing checks**

Cover:

- invalid entry ids do not escape `get_entry_kernel`
- invalid requests retire cleanly instead of leaving the pipe active forever

**Step 2: Run tests to verify the failure before the fix**

Run:

```bash
cmake --build /home/starrydream/quiver/build --target flashanns_node_validation_test flashanns_nav_entry_test -j4
/home/starrydream/quiver/build/bin/flashanns_node_validation_test
env CUDA_VISIBLE_DEVICES=GPU-d75f73fc-a8d6-1a88-1fd8-aa278e3dd742 /home/starrydream/quiver/build/bin/flashanns_nav_entry_test --index-dir /mnt/disk1/starrydream/tmpfs/sift1b-index --query /mnt/disk1/starrydream/ann_benchmarks_data/gustann_indices/33e88b8f2218b4fa/queries_0000.bin --data-type uint8 --topk 10 --ef-search 30 --repeat 1 --max-queries 10000
```

Expected:

- the pre-fix failure is visible in the targeted reproduction

### Task 5: Implement the minimal FlashANNS lane-retirement fix

**Files:**
- Modify: `/home/starrydream/quiver/src/flashanns/io_control.cuh`
- Modify: `/home/starrydream/quiver/src/flashanns/kernel.cuh`
- Modify: `/home/starrydream/quiver/src/flashanns/search.cu`

**Step 1: Add the minimal fix**

Implement:

- a completion path for invalid or exhausted lane requests
- lane state transitions that let the pipeline retire empty work instead of spinning forever
- any small supporting state needed in the FlashANNS kernel args or control structs

**Step 2: Rebuild and rerun the focused tests**

Run:

```bash
cmake --build /home/starrydream/quiver/build --target flashanns_node_validation_test flashanns_nav_entry_test flashanns_search -j4
/home/starrydream/quiver/build/bin/flashanns_node_validation_test
env CUDA_VISIBLE_DEVICES=GPU-d75f73fc-a8d6-1a88-1fd8-aa278e3dd742 /home/starrydream/quiver/build/bin/flashanns_nav_entry_test --index-dir /mnt/disk1/starrydream/tmpfs/sift1b-index --query /mnt/disk1/starrydream/ann_benchmarks_data/gustann_indices/33e88b8f2218b4fa/queries_0000.bin --data-type uint8 --topk 10 --ef-search 30 --repeat 1 --max-queries 10000
env CUDA_VISIBLE_DEVICES=GPU-d75f73fc-a8d6-1a88-1fd8-aa278e3dd742 /home/starrydream/quiver/build/bin/flashanns_search --index-dir /mnt/disk1/starrydream/tmpfs/sift1b-index --query /tmp/flashanns_queries_0100.bin --data-type uint8 --topk 10 --repeat 1 --ef-search 30 --num-blocks 16 --poll-threads 6 --poll-contexts 7 --pipe-width 4 --output-ids /tmp/flashanns_result_ids_0100.bin
```

Expected:

- tests pass
- the `100` query native run completes with finite sample distances

### Task 6: Scale the native validation

**Files:**
- None

**Step 1: Run increasing query prefixes**

Run:

```bash
env CUDA_VISIBLE_DEVICES=GPU-d75f73fc-a8d6-1a88-1fd8-aa278e3dd742 /home/starrydream/quiver/build/bin/flashanns_search ...
```

Use prefixes:

- `1000`
- `2000`
- `5000`
- `10000`

Expected:

- identify the first failing prefix if a hang remains
- otherwise confirm full completion

### Task 7: Rerun the ann-benchmarks FlashANNS path

**Files:**
- None

**Step 1: Run the FlashANNS benchmark wrapper**

Run:

```bash
cd /home/starrydream/ann-benchmarks
source .tmp/sift1b-tmpfs.env
CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python -u run.py --local --batch --run-disabled --dataset sift1b-128-euclidean --count 10 --runs 1 --algorithm flashanns --force
```

Expected:

- ann-benchmarks FlashANNS run completes with the refreshed binary

