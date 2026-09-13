# GB10 mapped NVMe staging

Implemented, default on (`nvme_mapped_staging=1`) as requested on 2026-09-13.
A CUDA host-mapped pinned allocation supplies both the CPU O_DIRECT I/O window
and the GPU gather/scatter window. Either tensor alias keeps the allocation alive. Ordinary cudaMalloc
memory is never treated as CPU-readable or NIC-registerable.

Paged block demotion now gathers directly into the mapped window; restoration
scatters directly from it. The staging/scratch copy disappears. Contiguous
extra state still uses its existing pinned transfer. FP32 KDA bytes, compression,
file layout, generation publication and per-transfer stream synchronization
are unchanged. The conservative boot memory budget is retained; the recorder
reports saved allocation bytes. With the nominal 64/32 MiB windows this
removes up to 96 MiB per rank, reduced by block-size rounding and 8190 bytes
of explicit alignment padding across the two allocations.

Shutdown now refuses to free staging if a worker's close timeout expires.
An unfinished future and its buffers are retained for retry; a completed
failed transfer still permits shutdown. Unique allocation accounting counts
the mapped aliases once, including their alignment padding.

Validation on 2026-09-13:

- 58 ST-image CPU tests passed, including actual O_DIRECT I/O through a CPU
  shared-alias oracle, permuted blocks, extra-state tails, compressed restore,
  close timeout, idempotent release, default flags and boot paths.
- SM121a native compilation passed without initializing CUDA; see compile.json.
- The first admitted GPU gate passed alias visibility/lifetime, but its next
  allocation exposed sub-page alignment from cudaHostAlloc. The allocator now
  reserves 4095 extra bytes and aligns both aliases with the same offset, while
  their shared owner frees the original allocation.
- `probes/engine_mapped_tier_check.py` is the admitted real-GPU gate for
  host/device visibility, both alias lifetimes and lossless NVMe round trips.
  Both tests passed on the corrected source through the single-GB10 ticket
  `st-mapped-tier0913r3` (52.94 seconds including native preparation). The gate
  covers repeated aligned allocations, alias lifetimes, permuted blocks,
  extra-state tails and compressed restoration. `gpu-consumer.json` preserves
  the exact source hashes. Serving interference/performance remain pending.

This is a memory and cold-state transfer experiment. No normal decode tok/s
improvement is inferred from removed storage.
