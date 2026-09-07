# GLM-5.3 full-chunk MoE M64 experiment

The current dynamic workspace chooses M128 from its maximum capacity and reuses
that tile for smaller calls. This experiment retains the original workspace and
adds an independent M64 workspace only when `VLLM_GLM53_B12X_PREFILL_M64=1`.
The default is 0. No measured speed or cumulative 40% improvement is claimed yet.

The candidate admits actual 6,144–8,192-token eager wrapper calls with E288,
H4096, I512, top-8, NVFP4, BF16 output, SwiGLU-OAI alpha=1/beta=0/limit=10,
and SM121. Short calls, capture, other geometry, forced static backend and the
functional API retain their original workspace. Failure to query capture state
also uses the original workspace. MoE is called once on the complete chunk;
transport and per-expert quantization scales remain stock. The candidate now
ports the pinned gated kernel to M64 as described below. Existing M128-only prefill reuse kernels are not eligible for M64.

The separately allocated workspace is bounded at 8,192 tokens even if the wrapper
has a larger capacity. It adds approximately 185 MiB of device storage at that
capacity, shared across layers by the existing wrapper cache. Allocation uses
M64 geometry for packed rows and tasks and still aligns the scale plane to
128-row atoms. Launch/compiled-cache selection consumes the stored workspace's
`tile_m`; the functional workspace cache and its selector are unchanged.

## Evidence budget and its limits

`measurements/glm53_moe_m64_20260908/padding-budget.json` retains the exact expert
counts and source hash from PR #453's completed diagnostic. These were synthetic
inputs, not a production routing trace. For balanced 6,912/8,192-token inputs,
M64 reduces padded MMA rows by 14.95%/11.08% with BF16 gather and 7.43%/6.92% with
FP8-v3 gather. Concentrated routing saves no rows. Smaller tiles increase the
number of tasks and weight loads, so this is an available arithmetic budget,
not a predicted speedup. The current source profile uses FP8 v3 for chunks of
at least 4,096 tokens; both transport modes must be tested.

## Validation

The CPU contracts execute the wrapper's allocation/routing code with fakes to
verify boundary rows, capture, error fallback, other geometry, bounded storage
and independent stock/M64 allocation. The inherited offline lifecycle tests
exercise exact container identity checks and recovery after partial failures.
The distributed API binding is checked against the frozen source before CUDA.

`probes/glm53_moe_m64_check.py` runs real TP4 balanced/concentrated inputs at
4096, 6143, 6144, 6912 and 8192 tokens in both BF16 and FP8-v3 transport processes.
It records every rank's candidate and independent stock-control per-row errors,
changed-input/route reuse, retained-output lifetime, local MoE before transport,
and changed-input capture/replay of the original M128 path. Original per-row
L2 .02 / peak .04 floors and three-times same-row stock repeat error are retained.
A stock-control failure is inconclusive and blocks serving; it never approves
the candidate. Six alternating timing samples use the slowest rank. These
component times are not full-model TTFT.

GPU work uses `glm53_offline_checks.py` inside a normal `fleet.sh run --gpu` boot
hold. It pins all source copies, preserves the 128 GiB disk and 16 GiB probe
memory guards, and restores the exact incoming containers in `finally`.
Only a passing GPU gate can proceed to a matched same-build fresh-cache
2K/32K/128K B1/A/B2 TTFT bracket and public recovery.

Status: PR #455 remains draft and default-off. Check1 failed before M64 launch
and restored the original serving fleet; see below. The gated M64 port is pinned
as `a1622f17fc4be5d0d5325130f533270427e16886` for check3 on main `d489639`.
Check2 was rejected before submission because main advanced; it never ran CUDA. CPU checks
pass (6685 logic, 30 megakernel, 92 fleet; 6 M64 contracts, 3 API, 8 recovery;
10 serving gate, 7 comparator, 4 fresh-cache and 4 memory tests). The serving
collector requires check3's TP4 numerics, both sanitizers, matching source and
recovery. No candidate speedup or direct TTFT result is available yet.

## Check1 failure and gated M64 port

Check1 ended before the first M64 kernel launch. The image factory deliberately
chooses its generic kernel for M64, while the production tile-major weights are
supported only by the optimized gated subclass. All four BF16 fallback cases
passed; FP8, candidate numerics and candidate timings were not reached. Exact
incoming container recovery completed at 2026-09-08 01:19:28 KST. Raw logs,
source identity and recovery proof are in `check1/`. Do not rerun check1 unchanged.

The follow-up is an actual port of the pinned gated implementation, not merely
a workspace selector. `MoEGatedDynamicKernelM64Tiled` initializes stock attributes
and changes its three M-dependent constructor dimensions before the inherited
`__call__` derives layouts: compute (64,128,128), FC1 (64,64,128), epilogue (64,128).
The stock 4x2 MMA warp grid has a 64-row atom; M64 runs one M iteration. N128,
paired N64 branches, K128, physical N128 scale blocks, barriers and weight grouping
are retained. Its Q0 holds 8192 BF16 elements, larger than scoped H4096. This
reasoning does not prove numerical correctness or race freedom. Every GPU gate
must be rerun for this new kernel geometry, with sanitizer checks before serving.

The original image source remains unchanged. A cached SHA-256 contract admits
only gated.py `993783308233288ddfa77293e9dbabdc825ba5bfdcc4dcc41e842a895ec33445`.
Unknown source or non-tiled weights retain the original wrapper workspace.
The M64 port has a distinct compiled-cache suffix; default M128 keys stay unchanged.


The follow-up runner also executes local M64 memcheck and racecheck after both
TP4 transport processes pass, using the host CUDA tool mounted read-only into
the same pinned image (Compute Sanitizer 2025.3.1.0). Each tool checks balanced
and concentrated 6144/6912/8192 rows with changed routes and retained outputs.
The serving gate requires both zero-error/zero-hazard tool summaries, source
provenance and all six cases; a missing report or warning blocks deployment.
