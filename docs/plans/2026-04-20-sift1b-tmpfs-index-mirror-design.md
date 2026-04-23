# SIFT1B Tmpfs Index Mirror Design

## Goal

Keep one persistent tmpfs mirror of the SIFT1B query-time index files and reuse it across repeated ann-benchmarks runs for GustANN mem, GustANN original, and FlashANN.

## Context

The current SIFT1B index lives under [data/gustann_indices/94adb9cfbf65cbc8](/home/starrydream/ann-benchmarks/data/gustann_indices/94adb9cfbf65cbc8). The ann-benchmarks wrapper writes `base.bin`, query files, and result files into that working directory. Query-time search binaries only need a subset of those files:

- `ann_disk.index`
- `ann_pq_compressed.bin`
- `ann_pq_pivots.bin`
- `ann_disk.index_centroids.bin`
- `ann_disk.index_medoids.bin`
- `nav_data.bin`
- `nav_index`
- `nav_index.data`
- `nav_index.tags`
- `map.txt`

That query-time file set is about 667 GiB. `/dev/shm` on the target machine is about 504 GiB, so the practical solution is a dedicated tmpfs mount with a larger size limit.

## Approach Options

### Option 1: Dedicated tmpfs mirror

Mount a dedicated tmpfs directory, copy only the query-time files into it, and point all three wrappers at that mirrored directory during search.

This keeps `base.bin` and short-lived query/result artifacts on disk, minimizes tmpfs usage, and supports repeated benchmark runs without repeated copies.

### Option 2: Expand `/dev/shm`

Reconfigure the system shared-memory mount and use `/dev/shm` directly.

This keeps paths short but broadens system impact and couples the benchmark to a shared system resource.

### Option 3: Rely on page cache

Keep the files on disk and warm the page cache before each run.

This reduces setup work but gives weaker repeatability across long benchmark sessions.

## Recommendation

Use Option 1. It is the most controlled setup for a long-lived benchmark session and keeps the runtime path explicit.

## Architecture

### Source and mirror directories

- Source index directory: `/mnt/disk1/starrydream/ann_benchmarks_data/gustann_indices/94adb9cfbf65cbc8`
- Tmpfs mirror directory: `/mnt/disk1/starrydream/tmpfs/sift1b-index`

The source directory remains the canonical build output. The tmpfs directory is a query-time mirror.

### Wrapper behavior

The wrappers continue to use the hashed ann-benchmarks working directory for:

- `base.bin`
- generated query files
- result files
- logs and transient benchmark outputs

The wrappers gain an optional `index_dir_override`. When present, query-time paths resolve from that directory:

- GustANN mem uses `<index_dir_override>/ann`
- GustANN original uses `<index_dir_override>/ann_disk.index`, `<index_dir_override>/ann_pq_*`, and `<index_dir_override>/nav_index*`
- FlashANN uses `<index_dir_override>` as its `--index-dir`

The default behavior stays unchanged when no override is set.

### Configuration surface

Support the override in two ways:

- explicit `index_dir_override` inside `index_params`
- fallback environment variable for local runs, so the standard `run.py` commands do not need new CLI arguments

This keeps the integration easy to use in repeated local benchmarking sessions.

## Mount and sync workflow

### Prepare step

A helper script prepares the tmpfs mirror:

1. create the tmpfs mount point
2. mount tmpfs with a size around 760 GiB
3. sync only the query-time files from the source directory
4. write an environment file exporting the override path for ann-benchmarks runs

### Release step

A matching release script:

1. checks for active benchmark/search processes
2. syncs outstanding writes
3. unmounts the tmpfs mirror

## Running the benchmarks

The benchmark commands stay on the normal `run.py --local --batch` path. The new behavior comes from the override environment prepared by the tmpfs script. That keeps the invocation stable for:

- `gustann-mem`
- `gustann-original`
- `flashanns`

## Failure handling

### Missing tmpfs mirror files

If `index_dir_override` is set and required runtime files are missing, the wrappers should raise a clear error naming the missing path. This prevents silent fallback to a different directory.

### Mount or sync failure

The prepare script should stop immediately on mount failure, copy failure, or missing source files. Partial mirror directories are acceptable as long as the script exits with a non-zero status and leaves clear output.

### Runtime fallback

Removing the override returns the wrappers to the canonical hashed working directory. That keeps disk-based benchmarking available as a recovery path.

## Testing strategy

### Wrapper tests

Add tests that prove:

- `index_dir_override` changes runtime search paths while preserving the normal work directory
- the environment-variable override path is honored
- each wrapper emits commands against the override directory

### Script tests

Add script tests that run the prepare helper in a no-mount local mode against temporary directories, then verify:

- only the expected files are copied
- the generated environment file exports the override path

### Verification

After implementation:

- run the new Python tests
- run syntax checks for the shell scripts
- rebuild the three native search binaries

