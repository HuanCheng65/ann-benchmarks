# GustANN Benchmark-Safe Integration Design

**Context:** `ann-benchmarks` currently integrates `gustann-mem` by invoking a search binary that writes result ids to disk. The original `../GustANN` artifact does not expose a stable batch result file, and naive result dumping inside the timed command path would distort benchmark numbers.

**Decision:** Add benchmark-specific CLI entrypoints for both original `GustANN` and `gustann-mem`, and have the Python wrapper consume each binary's self-reported search time via `get_batch_latencies()`. The wrapper will read result ids only after timing is complete.

**Why this design:**
- It preserves the algorithm search path used by the native binaries.
- It keeps result serialization outside the benchmarked latency path.
- It avoids parsing human-oriented output formats beyond a single `[REPORT] Time` line.
- It lets the original artifact CLI stay intact for paper-style manual runs.

**Scope:**
- Original `../GustANN`: add a memory-backend bench binary that accepts `--output_ids`, writes `int32` ids in the existing `(nq, topk, ids...)` format, and emits `[REPORT] Time`.
- Existing `gustann-mem`: add a dedicated bench binary or mode that reports a single batch timing line suitable for wrapper parsing, then writes ids after search.
- `ann-benchmarks`: split the current wrapper into shared utilities plus separate `gustann-mem` and `gustann-original` integrations, both returning benchmark-safe batch latencies.

**Testing:**
- Wrapper tests should verify command construction, timing parsing, and `get_batch_latencies()` behavior.
- Local verification should cover Python tests and syntax checks.
