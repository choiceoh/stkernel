# GB10 mapped NVMe staging

Implemented, default off (`nvme_mapped_staging=0`). A CUDA host-mapped pinned
allocation supplies both the CPU O_DIRECT I/O window and the GPU gather/scatter
window. Either tensor alias keeps the allocation alive. Ordinary cudaMalloc
memory is never treated as CPU-readable or NIC-registerable.

Paged block demotion now gathers directly into the mapped window; restoration
scatters directly from it. The staging/scratch copy disappears. Contiguous
extra state still uses its existing pinned transfer. FP32 KDA bytes, compression,
file layout, generation publication and per-transfer stream synchronization
are unchanged. The conservative boot memory budget is retained; the recorder
reports actual saved staging bytes. With the nominal 64/32 MiB windows this
removes up to 96 MiB per rank, reduced by block-size rounding.

Shutdown now refuses to free staging if a worker's close timeout expires.
An unfinished future and its buffers are retained for retry; a completed
failed transfer still permits shutdown. Unique allocation accounting counts
the mapped aliases once.

Validation on 2026-09-13:

- 58 ST-image CPU tests passed, including actual O_DIRECT I/O through a CPU
  shared-alias oracle, permuted blocks, extra-state tails, compressed restore,
  close timeout, idempotent release, default flags and boot paths.
- SM121a native compilation passed without initializing CUDA; see compile.json.
- `probes/engine_mapped_tier_check.py` is the admitted real-GPU gate for
  host/device visibility, both alias lifetimes and lossless NVMe round trips.
  GPU qualification and serving interference/performance remain pending.

This is a memory and cold-state transfer experiment. No normal decode tok/s
improvement is inferred from removed storage.
