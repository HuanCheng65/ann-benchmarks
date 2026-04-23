# GustANN Chunked Host Register Design

**Date:** 2026-04-19

**Goal:** Make large MEM-backend GustANN indices usable on single-GPU benchmark runs by replacing one-shot host registration with a chunked mapped registration path, while preserving the current multi-GPU path.

## Context

The current MEM backend in [ssd_search.cu](/home/starrydream/gustann-mem/src/ssd_search.cu) mmaps one large file-backed region and registers the whole region with a single `cudaHostRegister(..., cudaHostRegisterPortable)` call. The CUDA host-register probe documented in [2026-04-19-cuda-host-register-probe-summary.md](/home/starrydream/ann-benchmarks/docs/plans/2026-04-19-cuda-host-register-probe-summary.md) showed that this machine can pin and GPU-map much more total host memory when the registration is split into multiple chunks. The failure mode is driven by registration shape and driver overhead, not raw host RAM capacity.

## Design Goals

1. Keep the existing `portable_whole` path available for existing multi-GPU and legacy runs.
2. Add a stable `single_gpu_chunked_mapped` path for large benchmark indices.
3. Keep device-side search logic and cache logic as unchanged as possible.
4. Make ann-benchmarks explicitly select the single-GPU path for large MEM benchmark runs.
5. Provide enough logging and validation to diagnose registration failures by chunk.

## Approach Options

### Option 1: Backend dual-path registration

- Add a registration mode to GustANN.
- Keep the current whole-region portable registration path.
- Add a chunked mapped registration path for single-GPU runs.
- Let ann-benchmarks explicitly select the new mode and constrain visibility to one GPU.

**Recommendation:** Use this option. It keeps the behavior boundary clear and isolates the new complexity inside the MEM backend.

### Option 2: ann-benchmarks-only GPU visibility workaround

- ann-benchmarks constrains `CUDA_VISIBLE_DEVICES`.
- GustANN keeps a single registration implementation.

This reduces integration work but leaves the large single-region registration failure unresolved.

### Option 3: Fully automatic runtime switching

- GustANN inspects visible GPUs, index size, and registration failures.
- GustANN selects a strategy automatically.

This would be more convenient later, but it adds more branching and failure ambiguity than needed for the first fix.

## Architecture

The MEM backend will gain a registration strategy abstraction with two concrete paths:

- `portable_whole`
  - current behavior
  - one large file-backed `mmap`
  - one `cudaHostRegisterPortable`
- `single_gpu_chunked_mapped`
  - split the file-backed data region into fixed-size chunks
  - `mmap` each chunk independently
  - `cudaHostRegisterMapped` each chunk
  - resolve each chunk with `cudaHostGetDevicePointer`
  - present the device with a chunk table and fixed chunk geometry

ann-benchmarks will explicitly choose the chunked path for large GustANN MEM benchmark runs and constrain CUDA visibility to a single device for those runs. Existing multi-GPU runs keep the current default path.

## Configuration

Add two GustANN configuration fields:

- `host_register_mode`
  - `portable_whole`
  - `single_gpu_chunked_mapped`
- `host_register_chunk_gb`
  - fixed chunk size in GiB
  - default initial value: `32`

For `single_gpu_chunked_mapped`, the process must see exactly one CUDA device.

ann-benchmarks will wire these values into the GustANN benchmark configuration and select the single-GPU path for large MEM benchmark scenarios.

## Host-Side Data Structures

Expand `BackendContext` in [ssd_search.hpp](/home/starrydream/gustann-mem/include/ssd_search.hpp) to track chunked host registration:

- `HostRegisterChunk`
  - `map_base`
  - `host_data_ptr`
  - `device_data_ptr`
  - `mapped_size`
  - `data_size`
  - `file_offset`
  - `registered`
- backend-level geometry
  - `chunk_size_bytes`
  - `chunk_shift`
  - `num_chunks`

The backend will still expose a single logical data region to the rest of GustANN, but internally it will be backed by chunk metadata instead of one pointer.

## Device-Side Addressing

Update `PageData` in [page_wrapper.cuh](/home/starrydream/gustann-mem/src/impl/page_wrapper.cuh) from a single `disk` pointer to chunked backing:

- `uint8_t **disk_chunks`
- `uint64_t chunk_size_bytes`
- `int chunk_shift`
- `int num_chunks`

Node lookup remains logically unchanged:

1. Compute the global byte offset from page id and node offset.
2. Derive `chunk_id = global_byte >> chunk_shift`.
3. Derive `in_chunk = global_byte & (chunk_size_bytes - 1)`.
4. Resolve `src = disk_chunks[chunk_id] + in_chunk`.

This keeps existing node load helpers mostly intact and localizes the change to address translation.

## Failure Handling and Logging

The chunked path will validate:

1. exactly one visible GPU
2. power-of-two chunk size
3. correct tail chunk sizing

Registration will proceed chunk-by-chunk. On failure, the backend will clean up all successfully registered and mmapped chunks before exiting.

Logs will include:

- selected registration mode
- chunk size and total chunk count
- target CUDA device
- per-chunk progress
- failing chunk id and file offset on error

## Testing Strategy

### Unit tests

- address translation across:
  - regular page lookup
  - chunk boundaries
  - tail chunk boundaries

### Integration tests

- small synthetic MEM index
- chunked backend initialization
- GPU-visible pointer access to expected node content

### ann-benchmarks wiring tests

- benchmark config passes:
  - `host_register_mode`
  - `host_register_chunk_gb`
  - single-GPU environment constraint for the selected benchmark path

## Rollout

1. Keep `portable_whole` as the default to avoid breaking existing runs.
2. Add explicit benchmark configuration for the large MEM benchmark path.
3. Validate on a small local index first.
4. Use the new path for large SIFT1B-class runs.
