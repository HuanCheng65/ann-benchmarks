# SIFT1B Tmpfs Benchmark Results

Date: `2026-04-20`

Dataset: `sift1b-128-euclidean`

Environment:
- Query-time index mirror: [sift1b-index tmpfs](/mnt/disk1/starrydream/tmpfs/sift1b-index)
- Count: `10`
- Batch mode: `true`
- Query group: `ef_search=30`

## FlashANN

Status: completed

Plot:
- [FlashANN recall-qps repeat20](/home/starrydream/ann-benchmarks/logs/sift1b-uint8-tmpfs-20260420/flashanns-recall-qps-repeat20.png)

Results:

| Config | best_search_time (s/query) | QPS | Recall@10 |
| --- | ---: | ---: | ---: |
| `blocks=540,T=6,C=7,W=4,repeat=20,ef=30` | `0.0002940558` | `68014.30` | `0.85635` |
| `blocks=864,T=6,C=7,W=4,repeat=20,ef=30` | `0.0003059217` | `65376.21` | `0.85632` |
| `blocks=648,T=6,C=7,W=4,repeat=20,ef=30` | `0.0003126672` | `63965.78` | `0.85647` |
| `blocks=972,T=6,C=7,W=4,repeat=20,ef=30` | `0.0003247216` | `61591.22` | `0.85618` |
| `blocks=756,T=6,C=7,W=4,repeat=20,ef=30` | `0.0003325534` | `60140.72` | `0.85650` |

Artifacts:
- [flashanns-batch results dir](/home/starrydream/ann-benchmarks/results/sift1b-128-euclidean/10/flashanns-batch)
- [flashanns sweep log](/home/starrydream/ann-benchmarks/logs/sift1b-uint8-tmpfs-20260420/flashanns-formal-sweep-rerun.log)

Notes:
- Current `best_search_time` comes from ann-benchmarks batch result files.
- Current `QPS` for `repeat=20` rows uses the native steady-state runtime divided by `200000` queries.
- Euclidean distance outputs in the result HDF5 files now use `L2` scale, so `plot.py` recall and manual `Recall@10` are on the same metric basis.

## GustANN MEM

Status: completed

Artifacts:
- [gustann-mem log](/home/starrydream/ann-benchmarks/logs/sift1b-uint8-tmpfs-20260420/gustann-mem-formal.log)

Results:

| Config | best_search_time (s/query) | QPS | Recall@10 |
| --- | ---: | ---: | ---: |
| `B=10000,cache=0%,blocks=1,W=0,ef=30` | `0.0000023318` | `428853.25` | `0.84111` |
| `B=1024,cache=0%,blocks=1,W=0,ef=30` | `0.0000029598` | `337860.67` | `0.84111` |
| `B=256,cache=0%,blocks=1,W=0,ef=30` | `0.0000044247` | `226004.02` | `0.84111` |
| `B=64,cache=0%,blocks=1,W=0,ef=30` | `0.0000135937` | `73563.49` | `0.84111` |

## GustANN Original

Status: running next

Artifacts:
- [gustann-original log](/home/starrydream/ann-benchmarks/logs/sift1b-uint8-tmpfs-20260420/gustann-original-formal.log)

Results:

| Config | best_search_time (s/query) | QPS | Recall@10 |
| --- | ---: | ---: | ---: |
| pending | pending | pending | pending |
