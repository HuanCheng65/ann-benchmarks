# CUDA Host Register Probe Summary

**Date:** 2026-04-19

**Goal:** Understand how much host memory this machine can pin and expose to a GPU with `cudaHostRegister`, and explain why a single large `cudaHostRegister(ptr, 600+ GiB, ...)` can fail even though chunked registration reached `700+ GiB`.

## Environment

- Host memory:
  - `MemTotal`: about `1.0 TiB`
  - `MemAvailable` during the experiment: about `845 GiB`
- Swap:
  - `8 GiB`
- GPUs:
  - `3 x NVIDIA A100-SXM4-40GB`
  - Driver `580.126.09`
  - CUDA `13.0`
- CUDA visibility during the probe:
  - `CUDA_VISIBLE_DEVICES=2`
  - The probe used logical device `0`, which mapped to the physical idle GPU 2
- Reported `RLIMIT_MEMLOCK`:
  - soft: about `125.97 GiB`
  - hard: about `125.97 GiB`

## Question

The reference code in [ssd_search.cu](/home/xieminhui/starrydream/gustann-mem/src/ssd_search.cu#L243) does:

```cpp
CHECK_CUDA(cudaHostRegister(backend_ctx_.data_ptr, total_size, cudaHostRegisterPortable));
```

The practical question was:

- How much host memory can this machine pin and expose to the GPU in practice?
- Why can chunked registration exceed `700 GiB`, while a single `600+ GiB` registration in GustANN still fails with `cudaErrorMemoryAllocation`?

## Probe Design

I added a standalone CUDA probe at [cuda_host_register_probe.cu](/home/xieminhui/starrydream/ann-benchmarks/scripts/cuda_host_register_probe.cu).

The probe intentionally did more than just call `cudaHostRegister`:

1. `mmap` one anonymous host region per chunk
2. Touch one byte per page on the CPU so the pages are materialized
3. Call `cudaHostRegister(..., cudaHostRegisterPortable | cudaHostRegisterMapped)`
4. Call `cudaHostGetDevicePointer`
5. Launch a GPU kernel that reads one byte per page from the mapped host region
6. Keep all successful chunks alive simultaneously

This means the probe measured:

- host memory that was actually resident
- successfully pinned by CUDA
- mapped into the GPU-visible address space
- and directly read by the GPU

This is stronger than merely checking whether `cudaHostRegister` returned success.

## Important Difference From GustANN

The probe and GustANN do not exercise the driver in the same way.

### Probe behavior

- Many independent chunks
- Typical steps: `32 GiB` or `64 GiB`
- One CUDA-visible GPU
- Used `cudaHostRegisterMapped`

### GustANN behavior

In [ssd_search.cu](/home/xieminhui/starrydream/gustann-mem/src/ssd_search.cu#L187), GustANN:

1. does one large anonymous `mmap(total_size)`
2. reads the whole file into that single region
3. calls one large `cudaHostRegister(ptr, total_size, cudaHostRegisterPortable)`

So GustANN is asking the driver to register one very large contiguous virtual region in one shot, and with `Portable`.

That is materially different from the probe.

## Method

Two main probe runs were performed.

### Run 1

- Command shape:
  - `--steps 32GiB,8GiB,2GiB,512MiB,128MiB --max 512GiB`
- Result:
  - succeeded through `512 GiB`
  - GPU touch validation passed for every chunk

### Run 2

- Command shape:
  - `--steps 64GiB,16GiB,4GiB,1GiB,256MiB --max 832GiB`
- Observed progress before the machine became unstable:
  - succeeded through `768 GiB`
  - process RSS was observed around `832 GiB` while still running
- User-reported outcome:
  - the system crashed before a clean final result was recorded

## Observed Results

### Hard findings

- At least `512 GiB` of host memory could be pinned, mapped, and directly touched by the GPU.
- At least `768 GiB` was also reached successfully in the chunked test before the final high-water run destabilized the machine.
- The effective capacity exceeded the reported `RLIMIT_MEMLOCK` value by a large margin on this system.

### Practical reading

For this machine, the limiting factor did not appear to be:

- GPU framebuffer size
- BAR1 size in the simple sense
- or the reported user-space `memlock` limit alone

The practical limit looked much closer to:

- currently available system RAM
- plus driver overhead
- plus kernel/system stability once the machine is pushed near exhaustion

## Why `700+ GiB` Chunked Success Does Not Contradict `600+ GiB` Single-Block Failure

This is the main conclusion.

`cudaHostRegister` success depends on more than total bytes. It also depends on how those bytes are presented to the driver.

### 1. One huge contiguous registration is not equivalent to many smaller registrations

The probe used many separate `mmap` regions. GustANN uses one single very large region.

Driver-side bookkeeping for:

- page pinning
- GPU mapping metadata
- address translation structures
- internal allocation for the registration itself

can fail for one massive registration even if the same total number of bytes works when split across smaller regions.

### 2. `cudaErrorMemoryAllocation` here is not “GPU HBM is full”

When `cudaHostRegister` returns `cudaErrorMemoryAllocation`, it means CUDA could not complete the registration request. In this context it is best read as:

- driver/resource allocation failure during host registration

not:

- device memory allocation failure in the usual HBM sense

### 3. `cudaHostRegisterPortable` likely increases overhead

The probe used:

- `cudaHostRegisterPortable | cudaHostRegisterMapped`

but only with one visible GPU.

GustANN currently uses:

- `cudaHostRegisterPortable`

and may run in a process that can see multiple GPUs.

`Portable` means the pinned allocation is usable across CUDA contexts. In practice that can increase driver-side work and metadata overhead compared with a single-GPU, single-context case.

### 4. The probe explicitly used one visible GPU

The probe constrained visibility with `CUDA_VISIBLE_DEVICES=2`.

That matters because the registration/mapping problem becomes simpler than a process that can see multiple GPUs and uses portable registration semantics.

### 5. The probe used `Mapped`, but GustANN currently does not request it explicitly

If the real requirement is “GPU kernel directly dereferences host memory”, the robust path should be explicit:

- `cudaHostRegisterMapped`
- `cudaHostGetDevicePointer`
- validate that the kernel can access the mapping

The current GustANN code only uses `cudaHostRegisterPortable` at the registration site shown above.

## Secondary Observation About Safety

This experiment also showed that probing near the machine's RAM ceiling is operationally risky.

Observed failure mode:

- the process continued growing resident memory toward the available RAM ceiling
- system stability degraded
- the machine crashed before the last run completed cleanly

So future experiments should keep a conservative margin instead of probing to exhaustion on a shared system.

## Recommended Design Direction

If GustANN needs large host-resident data that the GPU can access directly, the safer direction is:

1. restrict the process to a single visible GPU
2. split the host backing store into chunks
3. `mmap` and `cudaHostRegisterMapped` each chunk independently
4. call `cudaHostGetDevicePointer` for each chunk
5. access data through chunk base pointers plus offsets
6. avoid one-shot registration of a `600+ GiB` contiguous region

## Concrete Implication For GustANN

The current design in [ssd_search.cu](/home/xieminhui/starrydream/gustann-mem/src/ssd_search.cu#L187) to [ssd_search.cu](/home/xieminhui/starrydream/gustann-mem/src/ssd_search.cu#L244) should be treated as a high-risk path for very large indices because it combines:

- one giant `mmap`
- one giant `cudaHostRegister`
- `Portable`
- and no explicit chunk-level registration strategy

That design can fail even when the machine is capable of holding and GPU-mapping a larger total amount of host memory in chunked form.

## Summary

- The experiment did show that this machine can pin and GPU-map far more than `600 GiB` in total.
- The strongest verified successful point was at least `768 GiB` in chunked form.
- The experiment did not prove that a single contiguous `600+ GiB` registration should succeed.
- The likely issue in GustANN is registration shape and driver overhead, not raw host capacity.
- The next implementation step should be chunked host registration rather than trying to make the single-region design succeed.
