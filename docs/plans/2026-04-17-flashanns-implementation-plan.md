# FlashANN Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add FlashANN as a local batch benchmark target in `ann-benchmarks` using Quiver's memory backend and native FlashANN timing.

**Architecture:** Extend Quiver's `flashanns_search` binary with a result export option, then add a Python wrapper in `ann-benchmarks` that reuses the existing DiskANN index build flow and parses native timing from FlashANN stdout. Keep runtime on `--local --batch`, keep backend selection on memory mode, and keep serialization outside the native timing region.

**Tech Stack:** Python, pytest, C++, CUDA, CMake, DiskANN typed-bin IO

---

### Task 1: Add Quiver result export helper

**Files:**
- Modify: `../quiver/bin/common/search_setup.hpp`

**Step 1: Write the failing test**

Document the expected helper behavior in the Python wrapper test that will consume an `output_ids` file with the layout `int32 nq, int32 topk, int32[nq * topk] ids`.

**Step 2: Run test to verify it fails**

Run: `pytest test/flashanns_test.py::test_batch_query_reads_output_ids_file -v`
Expected: FAIL because the wrapper and native binary contract do not exist yet.

**Step 3: Write minimal implementation**

Add a helper that writes result ids from `SearchSetup` to a binary file:

```cpp
inline void write_result_ids(const SearchSetup& setup,
                             int topk,
                             const std::string& output_ids_path) {
  if (output_ids_path.empty()) {
    return;
  }
  FILE* file = std::fopen(output_ids_path.c_str(), "wb");
  if (file == nullptr) {
    ERROR("Failed to open output ids file {}", output_ids_path);
    std::exit(1);
  }
  const int32_t query_count = static_cast<int32_t>(setup.queries.count);
  const int32_t topk32 = static_cast<int32_t>(topk);
  std::fwrite(&query_count, sizeof(int32_t), 1, file);
  std::fwrite(&topk32, sizeof(int32_t), 1, file);
  std::fwrite(setup.nns.get(), sizeof(int32_t),
              static_cast<size_t>(query_count) * static_cast<size_t>(topk32),
              file);
  std::fclose(file);
}
```

**Step 4: Run test to verify it passes**

Run: `pytest test/flashanns_test.py::test_batch_query_reads_output_ids_file -v`
Expected: still FAIL because the CLI and wrapper are pending.

**Step 5: Commit**

```bash
git add ../quiver/bin/common/search_setup.hpp
git commit -m "feat: add flashanns result export helper"
```

### Task 2: Add `--output-ids` to `flashanns_search`

**Files:**
- Modify: `../quiver/bin/flashanns_search.cu`
- Modify: `../quiver/bin/common/search_setup.hpp`

**Step 1: Write the failing test**

Add a wrapper test that expects `--output-ids` in the native command:

```python
def test_batch_query_passes_output_ids_to_flashanns(...):
    ...
    assert "--output-ids" in cmd
```

**Step 2: Run test to verify it fails**

Run: `pytest test/flashanns_test.py::test_batch_query_passes_output_ids_to_flashanns -v`
Expected: FAIL because the wrapper and CLI do not emit the new flag.

**Step 3: Write minimal implementation**

Register and use the CLI argument:

```cpp
std::string output_ids;
program.add_argument("--output-ids").store_into(output_ids);
...
flashanns::run_persistent_search(...);
bin_common::write_result_ids(setup, shared_args.topk, output_ids);
```

Keep `write_result_ids(...)` after `run_persistent_search(...)` returns.

**Step 4: Run test to verify it passes**

Run: `pytest test/flashanns_test.py::test_batch_query_passes_output_ids_to_flashanns -v`
Expected: PASS

**Step 5: Commit**

```bash
git add ../quiver/bin/flashanns_search.cu ../quiver/bin/common/search_setup.hpp
git commit -m "feat: export flashanns result ids"
```

### Task 3: Add the FlashANN wrapper module

**Files:**
- Create: `ann_benchmarks/algorithms/flashanns/module.py`
- Create: `ann_benchmarks/algorithms/flashanns/__init__.py`

**Step 1: Write the failing test**

Create tests that assert:

```python
def test_batch_query_passes_output_ids_to_flashanns(...):
    ...

def test_batch_query_uses_memory_backend_contract(...):
    assert "--ssd-list-file" not in cmd

def test_batch_query_uses_native_flashanns_timing(...):
    assert algo.get_additional()["best_search_time"] == 0.05
```

**Step 2: Run test to verify it fails**

Run: `pytest test/flashanns_test.py -v`
Expected: FAIL because the module does not exist.

**Step 3: Write minimal implementation**

Implement a `Flashanns` class that:

```python
class Flashanns(GustannBase):
    default_home = Path("/home/xieminhui/starrydream/quiver")
    repo_guess_name = "quiver"

    def batch_query(self, X, n):
        query_file = self._workdir / "queries_0000.bin"
        output_ids = self._workdir / "result_ids_0000.bin"
        self._write_diskann_bin(query_file, X)
        cmd = [
            str(self._gustann_home / "build/bin/flashanns_search"),
            "--index-dir", str(self._index_dir),
            "--query", str(query_file),
            "--data-type", self._data_type,
            "--topk", str(n),
            "--ef-search", str(self._query_params.get("ef_search", self._search_params["ef_search"])),
            "--num-blocks", str(self._search_params.get("num_blocks", 756)),
            "--poll-threads", str(self._search_params.get("poll_threads", 6)),
            "--poll-contexts", str(self._search_params.get("poll_contexts", 7)),
            "--pipe-width", str(self._search_params.get("pipe_width", 4)),
            "--output-ids", str(output_ids),
        ]
        output = self._run(cmd, cwd=self._gustann_home)
        self._set_batch_latency_from_output(output, len(X))
        self._batch_results = self._read_ids(output_ids, n)
```

Override `_timing_patterns()`, `_refresh_name()`, and `_required_paths()`.

**Step 4: Run test to verify it passes**

Run: `pytest test/flashanns_test.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add ann_benchmarks/algorithms/flashanns/module.py ann_benchmarks/algorithms/flashanns/__init__.py test/flashanns_test.py
git commit -m "feat: add flashanns ann-benchmarks wrapper"
```

### Task 4: Add FlashANN algorithm config

**Files:**
- Create: `ann_benchmarks/algorithms/flashanns/config.yml`

**Step 1: Write the failing test**

Add a config test:

```python
def test_flashanns_config_registers_float_and_uint8_euclidean():
    configs = load_configs("float")
    assert "flashanns" in configs
```

**Step 2: Run test to verify it fails**

Run: `pytest test/flashanns_config_test.py -v`
Expected: FAIL because the config file does not exist.

**Step 3: Write minimal implementation**

Create a config that registers:

```yaml
float:
  euclidean:
  - base_args:
      - '@metric'
    constructor: Flashanns
    disabled: true
    docker_tag: ann-benchmarks-flashanns
    module: ann_benchmarks.algorithms.flashanns
    name: flashanns
    run_groups:
      baseline:
        arg_groups:
          - index_params:
              flashanns_home: ../quiver
              pq_size: 32
              search_memory_gb: 4
              build_memory_gb: 16
              max_degree: 128
              l_build: 200
              pivot_sample_size: 1000000
              pivot_graph_degree: 32
              pivot_graph_l_build: 50
            search_params:
              - num_blocks: 756
                poll_threads: 6
                poll_contexts: 7
                pipe_width: 4
                ef_search: 40
        query_args: [[20, 30, 40, 50, 60, 80, 100]]
```

Mirror the same structure for `uint8/euclidean`.

**Step 4: Run test to verify it passes**

Run: `pytest test/flashanns_config_test.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add ann_benchmarks/algorithms/flashanns/config.yml test/flashanns_config_test.py
git commit -m "feat: register flashanns benchmark config"
```

### Task 5: Verify the local integration path

**Files:**
- Modify: `test/flashanns_test.py`
- Modify: `test/flashanns_config_test.py`

**Step 1: Write the failing test**

Add one final integration-style unit test that checks the resolved Quiver path validation:

```python
def test_flashanns_required_paths_match_quiver_build():
    ...
```

**Step 2: Run test to verify it fails**

Run: `pytest test/flashanns_test.py::test_flashanns_required_paths_match_quiver_build -v`
Expected: FAIL until `_required_paths()` is finalized.

**Step 3: Write minimal implementation**

Ensure `_required_paths()` references:

```python
return [
    Path("build/bin/flashanns_search"),
    Path("deps/DiskANN/build/apps/build_disk_index"),
]
```

**Step 4: Run test to verify it passes**

Run: `pytest test/flashanns_test.py test/flashanns_config_test.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add test/flashanns_test.py test/flashanns_config_test.py ann_benchmarks/algorithms/flashanns/module.py
git commit -m "test: cover flashanns local integration path"
```

### Task 6: Run focused verification commands

**Files:**
- Modify: `../quiver/bin/flashanns_search.cu`
- Modify: `../quiver/bin/common/search_setup.hpp`
- Modify: `ann_benchmarks/algorithms/flashanns/module.py`
- Modify: `ann_benchmarks/algorithms/flashanns/config.yml`
- Modify: `test/flashanns_test.py`
- Modify: `test/flashanns_config_test.py`

**Step 1: Run unit tests**

Run: `pytest test/flashanns_test.py test/flashanns_config_test.py -v`
Expected: PASS

**Step 2: Run syntax validation**

Run: `python -m compileall ann_benchmarks/algorithms/flashanns`
Expected: PASS

**Step 3: Run native build validation when Quiver build context is available**

Run: `cmake --build ../quiver/build --target flashanns_search -j$(nproc)`
Expected: PASS

**Step 4: Smoke-check CLI contract**

Run: `../quiver/build/bin/flashanns_search --help`
Expected: output contains `--output-ids`

**Step 5: Commit**

```bash
git add ../quiver/bin/flashanns_search.cu ../quiver/bin/common/search_setup.hpp ann_benchmarks/algorithms/flashanns test/flashanns_test.py test/flashanns_config_test.py docs/plans/2026-04-17-flashanns-design.md docs/plans/2026-04-17-flashanns-implementation-plan.md
git commit -m "feat: integrate flashanns local batch benchmark"
```
