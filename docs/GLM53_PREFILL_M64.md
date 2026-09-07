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

Status: PR #455 remains draft and default-off. Both check1 and check3 failed
before candidate execution and completed exact incoming fleet recovery. No M64
speedup, numerical pass, sanitizer pass or serving TTFT result exists. Check3's
TMA failure was reproduced without GPU access. The physical-block port below
now passes the real CPU compiler; a fresh GPU gate is still required. Details
and raw evidence follow.

## Check1 failure and gated M64 port

Check1 ended before the first M64 kernel launch. The image factory deliberately
chooses its generic kernel for M64, while the production tile-major weights are
supported only by the optimized gated subclass. All four BF16 fallback cases
passed; FP8, candidate numerics and candidate timings were not reached. Exact
incoming container recovery completed at 2026-09-08 01:19:28 KST. Raw logs,
source identity and recovery proof are in `check1/`. Do not rerun check1 unchanged.

The follow-up is an actual port of the pinned gated implementation, not merely
a workspace selector. `MoEGatedDynamicKernelM64Tiled` initializes stock attributes
and uses compute (64,128,128), FC1 (64,64,128), epilogue (64,128). Its adapted
`__call__` and kernel explicitly retain M128 physical A/SFA storage and select
the appropriate M64 half, as described in the latest follow-up below.
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


## Check3: scale-atom mismatch, reproduced without GPU

Check3 on main `d489639`, source `a1622f17fc4be5d0d5325130f533270427e16886`,
entered the normal boot hold at 01:33:35 KST. Its probe ran 01:34:28–01:35:17;
exact original fleet recovery completed at 01:38:05. Four BF16 fallback cases
passed. M64 tracing reached the SFA TMA descriptor and rejected the physical
M128 shared scale layout versus the logical M64 CTA V-map. Candidate execution,
FP8, sanitizers and direct TTFT were not reached. Raw rank logs, full lifecycle
and source identity are retained in `check3/`; this frozen job must not be rerun.

The actual dispatcher now has a CPU compile harness. In the identical image,
with runtime `runc`, no GPU devices and no network, M128 compiled in 3.96 seconds;
M64 reproduced the same MLIR error in 0.013 seconds. Constructor-only checks
were insufficient to catch this mismatch. The offline driver now runs that
compiler before the serving snapshot/stop; a regression test proves a compiler
failure never reaches any container transition.

The next change must handle 128-row A/SFA physical blocks explicitly, including
TMA block indexing and the correct M64 half. The image's generic kernel already
uses `sa_tile_shape_mk`, `sfa_tile_shape_mk`, `*_tiles_per_block`, per-task half
selection and first-half FC2 staging for sub-128 tiles. Those semantics need to
be reconciled with this gated kernel's Q1 reuse and aliased A5/SFA4 storage.
Changing only the TMA tile to 128 would select wrong rows. Do not claim a fix
from constructor dimensions, relax the source/numerical gate, or perform another
GPU restart until both dispatcher paths compile. CPU compile success will still
require fresh TP4 numeric/capture/sanitizer and direct TTFT evidence afterward.


## Physical-block port after check3

The candidate now retains physical (128,128) A/SFA TMA loads and A5/SFA4
shared-memory storage. The MMA and epilogue remain M64. The producer addresses
physical block `task_m_tile_idx // 2`; FC1 selects half `task_m_tile_idx % 2`,
including tasks whose expert base falls on an odd half. FC2 always reads the
first half written by the original Q1 byte-swizzle helpers. The weight grouping
adapter remains in place, and B/SFB, pipeline, quantization and scatter helpers
stay pinned to the original image. The new cache suffix is
`glm53_prefill_m64_v2`; stock cache keys are unchanged.

The two adapted method bodies were diffed against the exact pinned source;
changes are limited to weight grouping and A/SFA layout, descriptor, indexing
and copy-partition selection. Host-side regression checks cover the original
compute geometry, physical A5/SFA4 layouts and unchanged B/SFB attributes.

A CPU-only staging compile on 2026-09-08 around 02:03 KST used the actual
dispatcher in the pinned image, runtime runc, no network and no GPU access.
Both M128 (9.74 s) and the physical-block M64 port (6.94 s) compiled successfully.
These are compiler wall times, not kernel performance. A clean, frozen-source
compile is required again before queue admission, followed by all TP4, capture,
memcheck and racecheck gates. Loading 128 physical rows per 64-row task may
offset reduced padded arithmetic; only matched full-model TTFT can decide.


The clean frozen source `ad8cf1b879cc28c968d5c29de1d2cc8ce51d2d73`, based on
main `4b0f1d1`, compiled both M128 (4.00 s) and M64 (2.88 s) without GPUs at
02:13 KST. Evidence is retained in `preparation-layout4/`, alongside 6689 logic
checks, 30 megakernel regressions, 107 fleet regressions and 28 focused tests.
Local torch-dependent host checks were skipped; the actual image compiler uses
torch and ptxas. Four immutable check4 copies share that exact source. The
serving collector now points to check4 and will reject it unless the complete
GPU/recovery evidence passes. `--refresh-gate` can run the GPU gates and then
the direct fresh-cache TTFT bracket in the same normal fleet hold.


## Check4 numerical failure and Q0 scale-address correction

Check4 began at 02:41:58 KST and completed exact incoming recovery at 02:46:51.
The candidate compiled and launched on all four ranks. All six active M64
cases failed on every valid row, including local MoE before transport.
Independent stock/changed/local controls and M128 capture passed; four short
fallback cases also passed. FP8, sanitizers and serving TTFT were blocked.
Evidence is in `measurements/glm53_moe_m64_20260908/check4/`.

The Q0 producer still used logical M64 to address the global M128 scale atoms.
For physical row 64 it wrote the first scale at 32768 instead of 8; near the
end this exceeds the allocation. The Q0 override uses `_m64_q0_scale_row`
for physical M128 coordinates in all four scale-store paths. Routing and task
publication remain M64. An AST audit verifies all other executable statements
match the pinned source after qualifying original helpers. The cache suffix
is `glm53_prefill_m64_v3`. CPU tests cover canonical offsets across even/odd
M64 boundaries and the allocation end. A fresh GPU gate is required to prove
the numerical fix; check4/serving1 are failed, completed jobs.


The Q0-corrected source `f7d3b4b4efbe231ddc4b1282f2fe13a37e4155fb` is frozen
on all four nodes as check5. Actual CPU compilation passed M128 (3.94 s) and
M64 (2.95 s) at 02:57 KST. CPU address contracts and the full 6689 logic /
30 megakernel / 107 fleet gate pass. Two existing macOS fixture cases needed
their uncached total-memory query mocked along with worker Popen; that fixture
correction and its dependency audit are included. Evidence is in
`preparation-q0/`. The collector now pins check5 and still requires full GPU
correctness/recovery before direct TTFT.
