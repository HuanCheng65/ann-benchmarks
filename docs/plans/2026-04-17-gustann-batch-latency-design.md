# GustANN Batch Latency Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Record GustANN batch-mode latency as user-visible batch wall-clock latency so plotted p50/p95/p99 values represent the latency of one submitted batch.

**Architecture:** Keep batch search behavior unchanged and only change the latency contract returned by the GustANN wrapper. In batch mode, one native search invocation corresponds to one user-visible batch, so the wrapper should assign the full batch wall-clock duration to every query slot in that batch.

**Tech Stack:** Python, pytest, ann-benchmarks batch runner.

---

## Design

Batch-mode `times` should represent batch latency. `ann_benchmarks.runner.batch_query()` already consumes one latency value per returned query row and then computes percentiles over that array. For GustANN, one `batch_query()` call launches one native batch search process and returns one full result matrix, so the correct user-facing latency for that call is the full reported batch duration.

The wrapper currently parses `BATCH_LAT_MS` or `[REPORT] Time ...` and divides by the number of queries before storing `_batch_latencies`. That division changes the metric from batch latency to amortized per-query time. The change is to keep the parsed duration intact and repeat that full value across all query rows in the batch.

## Scope

Files to touch:
- `ann_benchmarks/algorithms/gustann/common.py`
- `test/gustann_test.py`

Behavioral expectation:
- `algo.get_batch_latencies()` returns `[batch_seconds] * query_count`
- Existing qps and recall behavior stays unchanged
- Existing result row validation stays unchanged
