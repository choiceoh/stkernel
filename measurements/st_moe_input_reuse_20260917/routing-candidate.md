# Input reuse with route preparation

Mode 3 extends the warp-striped quantization cache with a compact route table.
During phase 0, CTA 0 computes each expert's first occurrence, count and each
route's row. Other CTAs quantize token inputs. The existing first grid barrier
publishes both. Phase 1 reads the prepared mapping and cached quantization;
matching scales reuse bytes and other scales retain the original quantizer.

The temporary route table adds 512/1,024 bytes inside the existing unreachable
workspace tail. Shared scratch is the not-yet-loaded FC2 weight stage, with an
explicit compile-time capacity check. The C2 cell's removed epilogue buffer
must not be used for this scratch. There are two CTA-local setup barriers in
CTA 0, no additional grid barriers, and no per-route registration barriers in
phase 1. Split/even/private route-scatter schedules are refused.

A shared async-proxy fence retires CTA 0's generic scratch accesses before TMA
reuses the FC2 stage. A CTA barrier alone is not a cross-proxy ordering guarantee
([PTX async proxy](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-data-movement-async-proxy)).

This removes route-registration CAS, spin waits and row-count atomics. It also
changes compact expert/row order, so matching quantized input bytes alone is
insufficient: the final FP32 output gate remains mandatory, with the same
unchanged tolerance and repeated baseline used for the other candidates.

`compile-routing.jsonl` compiles eight cells with CUDA hidden.
`native-resources-routing.json` inspects the final source including the shared
scratch bound: all cells remain at 96 registers, zero stack/local memory and
unchanged shared storage. Every mode 0/1/2 native binary is byte-identical to
its earlier counterpart in `native-resources-v1.json`. None of these counts is
a GPU timing result. GPU correctness and latency are pending.
