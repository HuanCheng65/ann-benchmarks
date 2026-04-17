# SIFT100M Float Dataset Design

**Goal:** Generate a `float32` variant of the existing SIFT100M ANN-Benchmarks dataset and provide a repeatable way to benchmark Docker-backed algorithms plus local `gustann-mem` on the same float dataset.

**Approach:** Extend the existing raw-file conversion script so it can emit either `uint8` or `float32` HDF5 output from the same raw `u8bin/ibin` inputs. Register a new external dataset name, then add a small runner script that generates the float dataset if needed and launches the benchmark runs in the correct order.

**Scope:**
- Reuse the existing raw SIFT100M files in `../shared_datasets/sift100m/`
- Reuse ground-truth ids and distances when converting `uint8 -> float32`
- Register `sift100m-128-euclidean-float` as a prebuilt external HDF5 dataset
- Provide a background-friendly orchestration script for dataset generation and benchmark execution

**Non-goals:**
- Recomputing ground truth from scratch
- Adding `uint8` support to other ANN wrappers
- Building every missing Docker image automatically
