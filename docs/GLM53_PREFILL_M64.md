# GLM-5.3 full-chunk MoE M64 experiment

The current dynamic workspace chooses M128 from its maximum capacity and reuses
that tile for smaller calls. This experiment retains the original workspace and
adds an independent M64 workspace only when `VLLM_GLM53_B12X_PREFILL_M64=1`.
The default is 0. No measured speed or cumulative 40% improvement is claimed yet.

Latest status (2026-09-08 07:30 KST): `reusediag2` again stopped after 40/48
memcheck trials with exit 15; this time all recorded candidate/control rows
passed. Docker/cgroup evidence shows no OOM or limit event and a peak of
2,909,007,872 bytes against the 16 GiB limit. However, the kernel journal records
an NVIDIA driver `NV_ERR_NO_MEMORY` allocation failure immediately before both
this exit and the earlier reusediag1 exit. Container OOM is ruled out; the exact
driver allocation limit remains unresolved. Racecheck was not reached. Exact
incoming recovery completed at 07:30:28. See `reusediag2/` for raw resource,
driver-journal, numerical and recovery evidence.

The next diagnostic releases completed case input/output references and the
unused CUDA allocator cache only after every original/changed-input lifetime
check and the existing case-end synchronization. Wrapper/weights and all trial
evidence remain live. No within-case ordering, numerical threshold, launch filter
or sanitizer reporting option changes. Case-boundary allocator/CUDA memory and
host MemAvailable observations distinguish cached probe allocations from other
driver/host limits. CPU BF16 tests verify all finished case tensors are released
before cache clearing and reject missing release evidence.

Earlier (2026-09-08 06:26 KST): the bounded local reuse diagnostic completed
all 48 plain trials with zero candidate/control failures. Memcheck recorded 40
of 48 trials, with zero candidate and six independent stock-control failing
row-trials, before its container exited 15 without a Python traceback. No final
memory-resource/container state was retained, so the termination cause remains
unknown. This partial result neither attributes the earlier M64 failure nor
admits serving. Racecheck was not reached. Exact recovery completed at 06:26:54.
Raw failed-row BF16 payloads independently reconstruct the logged error norms.
See `measurements/glm53_moe_m64_20260908/reusediag1/`.

The follow-up retains the same runtime, local fixture, thresholds and 48-trial
plan. It runs only the unfinished memcheck/racecheck processes, capturing their
exact exit, Docker OOM state, cgroup peak/current memory and memory events before
removing the owned container. Unavailable resource readings remain unknown.
The completed plain collection is not repeated; the distinct outer completion
marker is `MOE_M64_REUSE_SANITIZER_COLLECTION_COMPLETE`, still never acceptance.

Earlier (2026-09-08 06:09 KST): `int8gate2` passed the API, out-of-bounds
CuTe write and shared-memory race detector controls, then all twenty TP4 cases.
Memcheck reported zero API/device errors in the executed portion, but the normal
M64 program failed its unchanged numerical criterion on eight rows during changed-
input reuse at 6912/concentrated routing. Maximum row relative L2/peak were
0.0132188825/0.13671875. The program stopped before completing all M64/INT8 cases;
racecheck and direct TTFT were not reached. This is a failed gate. Exact incoming
four-node recovery finished at 06:09:02. Evidence is in
`measurements/glm53_moe_m64_20260908/int8gate2/`.

The bounded `--reuse-diagnostic` follow-up holds the runtime and limits unchanged.
It runs the same local M64 fixture in plain, memcheck and racecheck processes,
with driver-first initialization and mandatory detector controls. Each process
compares 6144/6912/8192 balanced/concentrated cases, original and changed inputs,
and four alternating independent stock-control/candidate trials. The baseline
and repeat are fixed within each phase, as in the original sanitizer gate.
All failing row metrics and raw BF16 input/baseline/repeat/control/candidate rows
are retained; a 128-row payload limit aborts rather than truncates evidence.
Completion requires input/lifetime/source/hash/coverage proof and always sets
numerical and serving acceptance false. Its purpose is to distinguish candidate
excess from stock repeat variation before choosing a runtime fix.

Earlier (2026-09-08 05:21 KST): the full INT8 combination gate passed all
20 BF16/compressed TP4 numerical cases but failed memcheck with 34
`cuGetProcAddress_v2` invalid-value API reports. Six M64 and forty INT8 sanitizer
program checks completed; the sanitizer itself failed. Racecheck and direct
TTFT were not reached. Exact incoming fleet recovery completed at 05:21:40 KST.
See `measurements/glm53_moe_m64_20260908/int8gate1/`. A separate minimal CUDA
binding diagnostic will isolate the API reports without weakening checks.

Follow-up at 05:49 KST: two minimal diagnostics isolated the reports to CUDA
Python initialization after PyTorch CUDA initialization. Driver-first produces
zero bootstrap reports and still detects an intentionally invalid driver call.
The second diagnostic stopped on a singular/plural error-count parser defect;
its device controls were not reached. The new normal runner corrects parsing
and requires API, out-of-bounds CuTe write and shared-memory race detector
controls to succeed before running the TP4 cases or driver-first sanitizers.
Their deliberately bad kernels run only in separate diagnostic containers.
The serving collector requires this detector evidence and exact source hashes.

Earlier at 05:05 KST, the separate RS-only INT8 diagnostic
completed 72 TP4 trials and exact incoming fleet recovery. All 442,368 row-trials
per arm have zero INT8 candidate/control failures under the original thresholds;
the same actual partials through FP8 have 94/17 failures. All 1,152 packet and
decoded-output checks pass, as do all four ranks' CPU-reference codec and short
BF16 identity cases. See `measurements/glm53_moe_m64_20260908/int8diag1/`.
This is not full numerical/sanitizer or serving acceptance. Both flags remain 0.

The next full gate is explicitly `glm53_offline_checks.py --int8-gate` inside a
normal fleet GPU hold. It repeats all existing BF16 and compressed-transport
MoE checks, now labeling the latter `fp8-v3-rs-int8`; the original FP8 gate is
still available unchanged. The new combination additionally requires INT8 pack
and unpack checks for all four packet destinations, odd-row padding, changed
inputs and retained outputs under both sanitizers. Its completion marker is
`MOE_M64_INT8_ALL_GATES_PASS`, distinct from both diagnostics and the original
FP8 acceptance. Direct full-model TTFT and quality remain required afterwards.

The candidate admits actual 6,144–8,192-token eager wrapper calls with E288,
H4096, I512, top-8, NVFP4, BF16 output, SwiGLU-OAI alpha=1/beta=0/limit=10,
and SM121. Short calls, capture, other geometry, forced static backend and the
functional API retain their original workspace. Failure to query capture state
also uses the original workspace. MoE is called once on the complete chunk;
per-expert quantization scales remain stock. The optional RS-only INT8 experiment
changes only the reduce-scatter codec, retaining FP8 all-gather. The candidate now
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

Historical check1/check3 status (superseded by the later results below): PR #455 remains draft and default-off. Both check1 and check3 failed
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


## Check5: BF16 passes, FP8 control instability blocks TTFT

Check5 ran at 03:02:54–03:04:50 KST; exact incoming recovery completed at
03:07:27. All ten BF16 cases passed. Every local MoE/capture case also passed
under FP8-v3. The check4 Q0 corruption no longer reproduces. FP8 4096/skew
failed on one independent stock-control row; FP8 8192/balanced failed on ten
candidate rows and two stock-control rows. Both are inconclusive comparisons,
not accepted candidate results. Sanitizers and direct TTFT remain blocked.

Synthetic balanced BF16 component latency fell 12.68–20.40%, while concentrated
routing regressed 8.95–15.41%. FP8 balanced 6144/6912 fell 7.55/6.66%, but these
are component observations from one run, not full-model prefill evidence.
The full table, raw eight-rank-process logs, recovery and fixed next diagnostic
plan are in `measurements/glm53_moe_m64_20260908/check5/`. No numerical tolerance
was changed, and check5/serving2 must not be rerun unchanged.


## Fixed FP8 comparison diagnostic

`--fp8-diagnostic` now selects a distinct offline label and emits only
`MOE_M64_FP8_DIAGNOSTIC_COMPLETE`, with serving/numerical acceptance false.
It reuses the exact MoE and TP fixture with fixed cases (4096/skew,
8192/balanced, 6144/balanced), three input seeds (9211+rows with offsets
0/104729/209759), and eight alternating trials per seed. Every trial includes
a fresh stock baseline/repeat and independent control/candidate, both before
and after TP transport. All failing row IDs, errors, unchanged limits, repeat
noise and intersections are retained. Input identity and all-rank source and
trial coverage are checked. This diagnostic cannot satisfy the serving gate.

The M64 kernel and thresholds are unchanged from check5. A completed diagnostic
may still contain numerical failures; it reports them rather than approving
the candidate. Its statistics group rows by case/seed/phase and describe row-
trial counts without treating rows or reused trials as independent experiments.

The fixed diagnostic completed all 72 trials at 03:36:29 KST; exact incoming
four-node recovery completed at 03:39:07. At 8192/balanced, candidate/control
failed row-trial counts were 24/2, 15/4 and 9/0 across the three seeds. Stock
4096/concentrated totaled 14/14; 6144/balanced totaled 1/0. All local MoE phases
passed, all outputs were finite and every failure was peak-only. Candidate
excess is reproducible within this fixed diagnostic and cannot be dismissed
as stock noise. Numerical/serving acceptance remains false.

Thirteen of 48 candidate failures at 8192 (and the one at 6144) are one float32
step above the normalized limit. They remain failures, and 35 larger candidate
exceedances at 8192 still need explanation. The next bounded diagnostic will
capture the actual pre-transport partial from the same invocation, replay its
FP8 transport, compare native BF16 reduction and retain failed rows' packed
values/scales and raw comparison numerators. No gate or threshold has changed.
All raw logs, per-seed counts, CPU validation and exact recovery are archived
in `measurements/glm53_moe_m64_20260908/fp8diag1/`. This result does not provide
new full-model TTFT evidence or a cumulative 40% improvement.

`--fp8-trace` implements the bounded follow-up as a separate collection mode.
It keeps the original 72-trial plan and normalized comparisons, captures each
actual invocation's pre-reduce-scatter partial, checks its frozen FP8 replay
against both the original output and the unmodified helper, and compares native
BF16 reductions of those same partials. It retains the production pack kernel's
outgoing FP8 bytes/scales and the unpack kernel's FP32 sum before BF16 storage.
Raw comparison numerators/denominators accompany the original failure values.

For the union of failed rows, every rank logs a compressed NPZ payload containing
all four arms' partial bits, packet bytes/scales and owned output/sum rows. A
128-row per-trial budget fails collection explicitly instead of dropping rows.
All 72 payloads from all four process logs are hash-checked against collective
metadata. The dedicated `MOE_M64_FP8_TRACE_COMPLETE` marker never grants numerical
or serving acceptance, including when frozen replay is bitwise equal. No kernel,
normal gate, tolerance or default has changed.

The trace completed at 04:15:20 KST with exact recovery at 04:17:57. Every frozen
FP8 replay and helper comparison was bitwise equal; actual partial and native
BF16 comparisons all passed. Candidate/control FP8 failing row-trials were
83/2 at 8192, 7/1 at 6144 and 10/10 in the stock-only 4096 fallback. CPU packet
and source-order sum reconstruction matches the retained GPU trace exactly.
Small partial differences can cross E4M3 rounding boundaries and become much
larger output differences, including -0.21875 before quantization versus -4.25
afterward without a scale change. The normalized 1-ULP boundary explains some
but not all failures; the unchanged numerical gate still blocks serving.

A CPU-only symmetric INT8 replay over the 106 previously selected failed-row
unions produces zero candidate/control failures under the same thresholds and
reduces quantization error in that subset. It is selected-row evidence only,
with no GPU implementation or performance proof. The next experiment is an
explicit default-off, RS-only INT8 encoding at the same payload width, with
unchanged FP8 all-gather and short BF16 routing. GPU pack fidelity and full-row
comparisons must precede any full gate/TTFT. Full trace, CPU reconstruction,
INT8 recipe, tests and recovery are in `measurements/glm53_moe_m64_20260908/fp8trace1/`.

## Default-off INT8 reduce-scatter candidate

`VLLM_GLM53_PREFILL_SP_RS_INT8=0` now declares the explicit RS-only experiment.
When enabled with sequence-parallel FP8 v3, its pack kernel uses per-2048-block
power-of-two scales, round-to-nearest-even and symmetric signed INT8. The same
packet layout, alignment and single all-to-all are retained. The existing
unpacker reads signed bytes and accumulates in FP32 before BF16 storage. FP8
all-gather and the executed-chunk short BF16 gate are unchanged. Invalid/incompatible
settings fail during import rather than selecting a per-rank fallback.

The codec has a distinct serving marker. The FP8-v3 family proof uses an actual
packed-exchange marker shared by both encodings; that generic marker cannot
satisfy the separate INT8 proof. The normal gate and its tolerances are unchanged.

`--int8-diagnostic` compares all rows in the fixed 72-trial plan, preserving the
original FP8 failures alongside INT8 results on the same actual pre-transport
partials. Each rank first checks 32 codec cases against CPU bytes, including
padding/alignment, zero, random, signed ties and large finite values, and checks
real 2128/4095-token BF16 identity with the INT8 option toggled. Every trial/arm
checks all packet bytes against an independent tensor recipe; the first trial
of each seed uses the CPU recipe, subsequent trials use the tensor recipe on GPU.
An independent FP32 reduction of the decoded recipe must match the actual INT8
output bitwise. All-row quantization error is recorded against the same arm's
unquantized FP32 reduction. Diagnostic completion always denies numerical/serving
acceptance. Actual INT8 pack/unpack compilation is added before any service stop.
Full-row GPU evidence and speed/quality results are still pending.
