# FlashANN Local Benchmark Integration Design

**Date:** 2026-04-17

**Goal:** Integrate Quiver's `flashanns_search` into `ann-benchmarks` for local batch benchmarking with the memory backend and native FlashANN timing.

## Scope

- Add a local `ann-benchmarks` algorithm wrapper for FlashANN.
- Keep the runtime path on `python run.py --local --batch`.
- Use Quiver's memory backend by leaving `--ssd-list-file` unset.
- Preserve benchmark accuracy by sourcing latency and QPS from FlashANN's native timing output.

## Current State

- `ann-benchmarks` already supports local wrappers that build DiskANN indexes and invoke external binaries.
- The closest reference is the GustANN integration in `ann_benchmarks/algorithms/gustann/`.
- Quiver's `flashanns_search` already accepts DiskANN index directories, typed-bin queries, `topk`, `ef-search`, and FlashANN-specific runtime parameters.
- `flashanns_search` currently keeps result ids in memory and prints only sample rows, so the Python wrapper lacks a stable way to collect the full top-k result set.

## Recommended Approach

Add an `--output-ids` option to `flashanns_search`, write the full query result ids to a binary file after the measured search completes, and add a Python wrapper in `ann-benchmarks` that mirrors the GustANN integration style.

This approach keeps the C++ search path intact, adds a minimal interoperability seam, and avoids introducing a Python binding layer or stdout parsing protocol.

## Architecture

### Quiver side

- Extend `flashanns_search` CLI with `--output-ids`.
- Add a helper in `bin/common/search_setup.hpp` that writes `setup.nns` as:
  - `int32 num_queries`
  - `int32 topk`
  - `int32[num_queries * topk] ids`
- Invoke the write helper after `flashanns::run_persistent_search(...)` returns.

### ann-benchmarks side

- Add `ann_benchmarks/algorithms/flashanns/module.py`.
- Reuse the existing DiskANN build and typed-bin helper logic from the GustANN integration.
- Resolve the Quiver checkout from:
  - explicit `flashanns_home`
  - repo-local sibling checkout `../quiver`
- In `fit()`:
  - build the DiskANN index if absent
  - reuse the generated index directory across runs
- In `batch_query()`:
  - write query typed-bin
  - call `build/bin/flashanns_search`
  - parse native timing from stdout
  - read `--output-ids` results

## Performance Accuracy

- The benchmark result should use FlashANN native time from stdout.
- The C++ timer should stop before result ids are serialized.
- Python-side file reading should stay outside the native timing region.
- The wrapper should continue to set `best_search_time` from native time divided by query count, matching the existing GustANN pattern.

## Backend Choice

- Runtime backend: memory backend.
- Control point: omit `--ssd-list-file` from the wrapper command.
- Build dependency: Quiver may still compile with SPDK enabled. That build choice remains compatible with a memory-backend runtime.

## Parameters

### Index parameters

- `flashanns_home`
- `pq_size`
- `search_memory_gb`
- `build_memory_gb`
- `max_degree`
- `l_build`
- `pivot_sample_size`
- `pivot_graph_degree`
- `pivot_graph_l_build`

### Search parameters

- `num_blocks`
- `poll_threads`
- `poll_contexts`
- `pipe_width`
- `ef_search`
- `cuda_visible_devices`

### Query parameters

- `ef_search`

## Supported Modes

- Initial support:
  - `float` + `euclidean`
  - `uint8` + `euclidean`
- Execution mode:
  - local
  - batch

## Testing

- Add unit tests for the Python wrapper:
  - command line contains `--output-ids`
  - command line omits `--ssd-list-file`
  - native timing is parsed into `best_search_time`
  - partial result files raise a runtime error
- Add a small Quiver-side CLI regression test only if an existing C++ test pattern exists nearby; otherwise keep Quiver coverage lightweight and rely on Python-side invocation tests.

## Risks

- Dirty working trees exist in `ann-benchmarks`, so changes must stay scoped to new FlashANN files and the specific Quiver CLI helper path.
- Quiver build artifacts are required locally for execution, so wrapper path resolution must fail with a clear message when the checkout is missing or unbuilt.
- FlashANN output format changes would break timing parsing, so the regex should target the stable `Time:` line already emitted by the native program.
