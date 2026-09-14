# C1 forward: reduce W4 partials inside the CUDA block

Base: `4dbc0713` (#938). Target: K7 C1, 24 consumer step/s.
This changes the forward GEMMs, enabled in the existing default-on decode
fastpath and DSA query owners. It does not change KDA state precision, weights,
MoE kernels or transport protocol. GPU execution and consumer timing are pending.

The ordinary kernel splits K across three CTAs, writes FP32 partials to device
memory, then uses arrival atomics and fences before folding them. The candidate
extends the existing local-reduction kernel to K=1536/2048/3072. The three warps
for each output tile exchange partials in shared memory and fold in the same
slice order. Input quantization, W4 decoding, MMA K order, FP32 accumulation,
row scales and final BF16 rounding remain in their existing arithmetic order.
K=2048 retains the unequal 5/5/6 groups; K=1536 uses 4/4/4, K=3072 uses 8/8/8.

The real hybrid rank header contains 34 KDA output projections, three dense
MLP down projections, and 22 DSA query projections at these shapes: **59 GEMMs
per target forward**. Each has 8x4096x3 FP32 partials. Removing one write and one
read of those partials eliminates **44.25 MiB of logical device traffic** and
5,664 slice-arrival atomics per forward. These are source work counts, not
measured DRAM bandwidth or latency savings. Output projections now add one
input-pack launch each; C1 query pairs keep one shared pack launch, reducing
their scratch from 50,688 to 12,672 bytes. The existing C4 paths remain selected.

Model-owned capture declarations select the output path before replay.
The native entry rejects a changed split count instead of silently changing
the arithmetic. Direct TX outputs still resolve their descriptor after PDL
and system-fence every final writer. The C1 paired-query control is available
as the final `local_c1=False` argument to the native probe API. Production calls
its true default; no process-global tuning switch was added.

Validation so far:

- `cpu-tests.log`: 30 checks, 21 passed and nine GPU-only skips. Checks include
  the real DenseLinear/slot-writer routing and boot rejection when a newly bound
  output owner has not executed its C1 path.
- `compile.json`: complete production-flag CUDA/Torch extension compile and
  load, with CUDA hidden. Five new native specializations; ordinary variants
  use 76 registers, direct-output variants 78, all with zero stack/local spill.
  Dynamic shared memory remains the existing 3-slice kernel's size.
- The initial compile succeeded but the report reader expected a different
  cuobjdump header. The repaired reader reused the same compiled binary.
  CPU staging initially included macOS AppleDouble sidecars; removing those
  from the task-owned staging directory resolved package text-reader errors.
  No numerical tolerance was changed.

The GPU entry `engine_kernel_check.py --lanes forward_reduce --ranks RANKS`
checks real rank-0 weights against the same native build: zero and signed
changed inputs, padded strides, poisoned outputs, private scratch, both replay
orders and alternating direct-output addresses with guards. Paired-query
comparisons explicitly retain the previous implementation as the control,
including C4 and a return to C1. Warm/evicted B/A/A/B timings follow exact checks.
This check also joins the existing `dsa_inputs` bundle, avoiding another model
boot. A passing component check does not establish 24 step/s or acceptance.

Final consumer validation remains K7, 32K/128K, C1 twice and C4 once. Report
pooled/window step/s, tokens/s, tokens/step and acceptance. Answer grading does
not decide the performance target, per the operator's instruction.
