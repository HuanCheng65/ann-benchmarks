# FlashANN Sweep And Repeat Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add FlashANN search-parameter sweep support for the 1B tmpfs workflow and make native `repeat` configurable from `search_params`.

**Architecture:** Keep the existing ann-benchmarks definition expansion flow and express the sweep directly in `config.yml` as a list of `search_params` dicts. Extend the FlashANN wrapper to forward `repeat` into the native command line and include it in the display name so result files and logs carry the exact runtime configuration.

**Tech Stack:** Python, pytest, YAML config expansion in ann-benchmarks.

---

### Task 1: Lock Expected Sweep In Tests

**Files:**
- Modify: `test/flashanns_config_test.py`

**Step 1: Write the failing test**

Assert that FlashANN config expands to the 1B sweep `540,648,756,864,972` and each entry carries `repeat: 20`.

**Step 2: Run test to verify it fails**

Run: `./.venv/bin/pytest test/flashanns_config_test.py -q`
Expected: FAIL because config still carries a single `num_blocks` value.

**Step 3: Write minimal implementation**

Update `ann_benchmarks/algorithms/flashanns/config.yml` to define the sweep entries explicitly.

**Step 4: Run test to verify it passes**

Run: `./.venv/bin/pytest test/flashanns_config_test.py -q`
Expected: PASS.

### Task 2: Forward Repeat Through FlashANN Wrapper

**Files:**
- Modify: `test/flashanns_test.py`
- Modify: `ann_benchmarks/algorithms/flashanns/module.py`

**Step 1: Write the failing test**

Add a wrapper test that sets `search_params["repeat"] = 20` and expects `--repeat 20` in the native command plus `repeat=20` in the algorithm name.

**Step 2: Run test to verify it fails**

Run: `./.venv/bin/pytest test/flashanns_test.py -q`
Expected: FAIL because wrapper still hardcodes `--repeat 1` and omits repeat from the name.

**Step 3: Write minimal implementation**

Use `self._search_params.get("repeat", 20)` for the native command and display name.

**Step 4: Run test to verify it passes**

Run: `./.venv/bin/pytest test/flashanns_test.py -q`
Expected: PASS.

### Task 3: Verify End-To-End Formal Run

**Files:**
- Verify: `ann_benchmarks/algorithms/flashanns/config.yml`
- Verify: `ann_benchmarks/algorithms/flashanns/module.py`

**Step 1: Run focused pytest**

Run: `./.venv/bin/pytest test/flashanns_test.py test/flashanns_config_test.py -q`
Expected: PASS.

**Step 2: Run formal FlashANN benchmark**

Run:

```bash
source .tmp/sift1b-tmpfs.env && \
CUDA_VISIBLE_DEVICES=GPU-d75f73fc-a8d6-1a88-1fd8-aa278e3dd742 \
./.venv/bin/python -u run.py --local --batch --run-disabled \
  --dataset sift1b-128-euclidean --count 10 --runs 1 \
  --algorithm flashanns --force
```

Expected: ann-benchmarks emits five FlashANN result files for the sweep, each tagged with the chosen `num_blocks` and `repeat=20`.
