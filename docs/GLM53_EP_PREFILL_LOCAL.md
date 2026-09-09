# TP SF6 Q0 prefill and historical EP experiments

## Final implementation and default decision

The selected implementation keeps TP4, 288 I512 expert shards per GPU and the
existing packed SF6 weights and static TP decode kernels. It changes only the
SF6 dynamic Q0 producer for eligible **executed chunks of 4096..8192 tokens**,
reusing routing/scale metadata within the call. BF16 output, task publication,
communication and weight ownership remain unchanged; there is no duplicate
model weight copy. Request context length is not the executed chunk size.
See [TP SF6 Q0 prefill](GLM53_TP_SF6_Q0.md) for the precise gate and rollback.

The requested default change selects this bounded TP prefill path, startup
trim and skipped unused graph profiling; EP/local/compact-warmup/zero-weight
micro remain off. This removes the EP decode architecture responsible for the
large earlier regressions. It is a default decision with **unresolved
performance uncertainty**, not statistical proof of no decode loss or a
canonical benchmark win. The latest pair showed no large aggregate loss;
the previous negative TP result remains part of the evidence. The original
40% prefill improvement goal has not been demonstrated.

## Direct serving evidence

[Onepass27](../measurements/glm53_ep_local_20260908/onepass27-completed/README.md)
ran B → A on frozen source `ea413ac4c39ba3e6e4009c73587b0d536053b4bf`.
Only A enables TP Q0. Both arms passed quality 18/18 and Korean 0/8 with no
traffic issues; proof was B 2/2 and A 3/3. Payload and supervisor returned 0,
and the owned holder was released. This is not public-service adoption proof.

| Onepass27 metric | B | A | A relative to B |
|---|---:|---:|---:|
| Fixed-1024 decode, pooled output tok/s | 67.4398 | 69.2342 | +2.66% |
| 2K best warm input tok/s | 2513.05 | 2421.55 | −3.64% |
| 32K input tok/s | 3022.07 | 3065.47 | +1.44% |
| 128K input tok/s | 3113.66 | 3140.25 | +0.85% |

Fixed-output repetitions were B **68.54 / 65.17 / 68.73** and A
**79.66 / 68.43 / 61.87** tok/s. Pooled decode is
`sum(completion_tokens - 1) / sum(decode_s)`; all three repetitions are kept.
The [original judge](../measurements/glm53_ep_local_20260908/onepass27-completed/job/verdicts.jsonl)
reports `incomplete`, `decision=unresolved`, `floor_n=1`, `floor=null`:
+2.7% with no noise floor yet. Three requests are not three baseline boots.
This result does not establish statistical noninferiority.

The prefill rows are **descriptive, not accepted prefill wins**. B has
`cold_compile=true` and A does not, so the canonical `prefill_ttft` comparison
rejects the baseline as incompatible; `decode_tokens` permits this pairing.
2K warm chooses the faster of two warm requests: TTFT B 0.84678 s, A 0.87878 s.
First-request TTFT was B 2.36728 s and A 1.92673 s, retained separately.
32K TTFT was 10.76910 → 10.61665 s; 128K was 41.28867 → 40.93907 s,
one request per arm. The 2K calls remain below the Q0 gate.

In [onepass25](../measurements/glm53_ep_local_20260908/onepass25-completed/README.md),
A was **67.1018** versus **71.9144** pooled tok/s across quality-valid
B0/B1/B3, or **−6.69%**. B2 is excluded because its Korean gate failed;
compile-cold B0 remains valid for decode. This pooled descriptive comparison
is separate from the original judge's latest-baseline result: A versus B3
was −8.0%, within the ±10.7% floor (`n=3`), `inconclusive/unresolved`.
That is not proof of no regression. Runs25 and27 have different full sources
and are not pooled together to manufacture a positive verdict.

## Validation and deployment boundaries

The original [CPU24 receipt](../measurements/glm53_ep_local_20260908/decode24-cpu/README.md)
has 165 CPU tests and 30 compiled kernels. Exact kernel/contract bytes were
bound to GPU27 before reuse; no CPU27 compile is claimed. The final profile
changes leave all 19 CPU-mounted sources and 36 of 37 contract files unchanged;
the changed profile has two separate real-loader tests, including explicit
zero opt-out. Those tests do not relabel the original CPU receipt or test the
final default commit on GPUs.

The four-rank canary checks the **first eligible actual-weight layer per
rank**, using four fixtures, changed values at reused addresses and
route/Q0/output comparisons in eager and two graph-stream contexts. It is
not all-layer coverage or a sanitizer pass. Strict B/A snapshots independently
bind image/source/capacity/flags and real graph/trim evidence. The production
launch marker may occur during model profiling and is not an HTTP receipt.
CPU bindings13.0.3 and serving bindings13.3.1 remain distinct recorded runtimes.

Startup trim and skipped unused graph profiling are common to every arm.
Real graph capture, live weights/KV/workspaces and capacity are retained.
Prior onepass24 trim receipts showed only 6–8 MiB less GPU reserved memory and
about33 MiB less process RSS per rank, not multi-GiB reclamation. Common
`MM_LIMIT` image4/video0 already comes from approved main PR #508. PR #509
startup instrumentation is included in the frozen GPU27 source.

Native channel diagnostics preserve the existing requests, text precedence,
timings, Korean classifier and gates. Earlier failures remain failures:
onepass24's Korean2/8 result is not waived by later channel attribution.
Onepass25 B2 (Q0 off) contained `Halvorsen博士` in `reasoning` only, with
two mixed-CJK characters and no replacement, lone-jamo or control characters.
Its final-content channels had no flagged characters. The combined-text gate
correctly retained its original failure under the existing policy; the new
diagnostics explain the channel without changing that policy. This baseline
example supports ordinary language mixing rather than candidate-specific
character corruption. The channel of the older onepass24 examples is unknown.
Repository defaults and live public recovery/adoption are separate; the latter
requires its own release and runtime proof.

## Historical EP implementation and evidence — not an adoption plan

The remainder preserves the earlier EP work and its original verdicts.
Onepass6's descriptive fixed-output decode fell 76.899 → 62.489 tok/s;
onepass20 stopped after53.1 tok/s without a matched baseline. These rejected
EP results are not performance evidence for TP Q0. Future-tense plans and
unmeasured-status statements below belong to those historical revisions.

### Decode regression follow-up

Default adoption is conditional on removing the recorded decode regression.
The proposed profile promotion was withdrawn before commit, merge or deploy;
EP/local/compact-warmup defaults remain off.

Onepass4's real SPEC_K=5 graphs contain 6/12/18/24 tokens. The fixed EP
fallback expands each token into eight routed rows and executes 6/12/18/24
top-k=1 micro calls per layer. The stock-top-k experiment admits only
8/16/32-token shapes. The separate zero-weight micro experiment previously
handled full 8-token prefixes and their tails but missed a whole batch below
eight tokens, including the normal six-token verify step.

The follow-up connects that existing padded-tail path for 1..7 tokens as
well. With `VLLM_B12X_EP_ZERO_WEIGHT_MICRO=1`, the real graph shapes become
1/2/3/3 top-k=8 calls. Complete prefixes retain their original contract; every
short tail is staged as eight tokens with zero router weights on duplicate
padding rows. Only real output rows are copied back. The existing exact
E72/M8/H4096/I2048/top8 dispatcher, arithmetic, 64-row workspace and fixed
fallback remain unchanged. Staging allocation failure keeps the fallback and
does not emit the completed zero-weight-micro proof marker.

CPU tests execute the actual wrapper routing and staging methods with a row
storage model: all short sizes, all-remote rows, changed input reuse, the
SPEC_K=5 graph sequence, failed padding allocation and explicit opt-out.
They do not establish GPU numerics or a recovered decode rate. The next
canonical B1/A/B2 onepass must measure output tok/s and prefill TTFT together;
the earlier numerical failure and onepass4 performance verdict remain intact.

Experimental `VLLM_GLM53_EP_PREFILL_LOCAL=1` plus `ENABLE_EP=1` changes the
MoE execution geometry from 288 experts with I512 TP shards per GPU to 72
complete I2048 experts per GPU. Attention retains TP4. The exact EP4/DP1,
non-transformed b12x runner still has one late TP sum, so MHC token sharding
can continue through the existing all-gather/reduce-scatter path.

The new M128 producer consumes original [T,4096] activations and remapped
[T,8] routes. Remote sentinel IDs and zero weights are excluded before any
expert-indexed address. Each warp compacts at most eight local routes into
its existing shared cache. It reads/quantizes original token blocks and
scatters to the original output rows. The inherited FC1/Q1/FC2 code and task
descriptor protocol are pinned to the image's gated-source SHA-256. A narrow
entry-point override admits I2048 as sixteen N128 slices. The producer always
publishes four tasks retaining four slices each; it never enlarges inherited
Q1 storage or uses the stock variable-task policy at T4096. Workspace capacity
already includes the four slice groups.

For this exact four-task contract, aligned task publication writes the two
four-word descriptor arrays with two `st.global.v4.u32` stores instead of
eight scalar stores. The publisher, slot order and synchronization are
unchanged. Pointers without 16-byte alignment and other slice contracts
retain the inherited scalar publication path.

The selected-scale cache uses at most eight slots for the exact top8 contract.
Lane 0 decides scale equality while copying each selected expert scale,
using the already-loaded raw word. It publishes this decision with the route
count, removing the consumer scale-scan loop. After histogram publication, each CTA
prepares the 72 expert scales once in the histogram's now-idle 288 shared
bytes. The existing Q0 initialization barrier publishes those stores. This
removes token/route-level global scale loads and reciprocal work without
adding shared storage or a barrier. Disabled flags and ineligible short calls
return before querying CUDA capture state; the query is lazy and only runs
after the exact shape/activation gate.

After allocating a selected route's row, lane 0 copies its transformed scale
bits into the existing route slot, where the expert ID is no longer needed.
The existing warp barrier publishes those words; the next batch's existing
CTA barrier protects reuse. Equal-scale quantization keeps the first scale in
a scalar, while varied-scale quantization reads the shared slot. This removes
the dynamically indexed per-thread `route_gs[8]` array without extra shared
storage, atomics or barriers. Float32 equality, signed-zero selection and NaN
bits passed to the quantizer retain their previous behavior.

The admitted candidate also remaps global routes into the existing output
scratch in one Triton launch. It replaces the expert-map path's 14 Torch
operations; this count describes source operations, not a measured speedup.
Unsupported dtypes/layouts/devices retain the existing Torch remap. The
integer map/offset conversion order is preserved, and weights are copied as
bits so local NaN payloads and signed zero survive while remote weights
become exact positive zero. Other EP, decode and TP paths keep their remap.
The latest remap masks weight loads after determining which routes the
legacy mapping keeps; remote weights need not be read. Empty-map
specializations write only sentinel IDs and zero weights without input loads.
When an admitted prefill exactly fills both output scratch buffers, the
wrapper reuses the Tensor objects and avoids two redundant slice views.
Smaller calls still take exact row views, with no cached view or map-content
assumption.
Remap admission returns the validated pair count and map length to the same
call's launch preparation, avoiding repeated Tensor metadata reads. Shape,
dtype, device, contiguity and map bounds are still checked on every call;
there is no cache of Tensor metadata or contents across calls. The wrapper
also reuses the expert count it already validated against the weight shape.

This removes the existing EP prefill path's GPU nonzero/host count boundary,
expanded pair_x/pair_out, pair-list chunking and external index_add. It does
not remove the model's arithmetic, TP communication or the router remap.
Input/scale/row scratch must still cover worst-case concentrated routing;
no assumed balanced distribution reduces buffer capacity. Extra padding,
wide-I task splitting and complete serving memory remain measurement topics.

The candidate is restricted to eager SM121 E72/H4096/I2048/top8 M128,
SwiGLU-OAI (1,0,10), row-major NVFP4 and 4096..16384 executed rows.
Unsupported shapes retain existing EP paths. Eligible calls reject inherited
source drift or incompatible forced backends before submitting invalid IDs
to a stock kernel. A distinct cache suffix separates candidate artifacts.
`ENABLE_EP` and the candidate both remain off by default. EP also changes
the decode layout, which must pass its own existing-quality/latency checks.

The historical September 7 profile assigns 28–32% of occupied prefill time
to MoE. It is source context, not a current critical-path fraction or a
forecast. Global useful FLOPs remain unchanged by TP-to-EP repartitioning;
75% fewer experts per rank is not a 75% speedup. The 1.40x direct prefill
throughput objective remains open.

The scale-cache source `cf1365b8fab6091833400efd7784071316eb9f6a` passed actual
E72/I2048 CuTe compilation, all 24 Triton specializations and 55 pinned CPU
tests without skips or CUDA initialization. [CPU9 evidence](../measurements/glm53_ep_local_20260908/cpu9/README.md)
records 168 registers, 112 stack bytes and 1024 shared bytes. Against CPU8,
stack use fell from 1040 to 112 bytes while registers and shared storage
stayed unchanged. This is compiler resource evidence, not a measured prefill
speedup. Static PTX local loads/stores fell from 29/12 to 4/5; inherited
local-memory uses remain. All 24 remap PTX files and executable `.text`
sections match CPU8. Whole remap cubin hashes differ only in the payloads of
`.debug_line` and `.nv.merc.debug_line`, as recorded by the section comparison.
The selected-scale raw-bit tests cover signed zero, infinities, NaN
payloads, duplicate routes, all four active warps and changed scales at the
same addresses. Remap tests revalidate reused Tensor metadata and launch sizes.
The initial CPU9 attempt on srv4 stopped before compilation because available
host RAM was below the unchanged 12 GiB guard. The same frozen source then
passed through the normal CPU wrapper on head with the same 4 GiB/2 CPU limit.

After the PR #484 lifecycle integration, source
`a9d0d1e3e8161ee44eac02191021ea9512a3e2cc` passed all 61 pinned CPU tests
without skips and actual CuTe plus 24-variant Triton compilation in the normal
no-device head lane. [CPU10](../measurements/glm53_ep_local_20260908/cpu10/README.md)
binds the updated lifecycle contracts to the unchanged kernel. Its CuTe PTX
and cubin match CPU9 exactly; REG168/STACK112/SHARED1024 are unchanged.

Source `695274d9d4c62041dacaa4fcd0861ce9e357bd4b` simplifies route allocation's physical-row address from
`(base + row / M) * M + row % M` to `base * M + row`. Both expressions address
the same row within the expert's padded tile prefix. The CPU oracle executes
the actual allocation expression across tile boundaries and balanced,
concentrated, empty and tail histograms, checking overlap, padding and integer
bounds. Route ordering, row-allocation atomics and barriers are unchanged.
CPU9/10 PTX retained signed quotient/remainder correction instructions at
this site. [CPU11](../measurements/glm53_ep_local_20260908/cpu11/README.md)
passed all 63 pinned CPU contracts without skips plus actual CuTe and 24
Triton compilations. REG168/STACK112/SHARED1024 are unchanged. CuTe PTX and
cubin sizes fell from 958390/308384 to 955476/307264 bytes. GPU performance
remains unmeasured.
All seven unroll/tail allocator copies shrink from 14 PTX arithmetic
instructions to two, removing 84 static instructions across the artifact.
The following nine address/store instructions match at each site under
register renaming; row atomics and base loads are unchanged. The
[instruction trace](../measurements/glm53_ep_local_20260908/cpu11/row-address-inspection.md)
binds every line to the two compilation receipts. These are not SASS or
per-request execution counts.

Source `0ae4c08f49c22122f499771ca6cefab1113b3476` also removes signed
quotient/remainder reconstruction from both Q0 scale-store paths. The M128
scale layout is expressed as unsigned physical-row/SF bit fields, with the
same final Int32 byte offset. The actual-AST oracle proves additive row/column
separation and checks every admissible physical row, all 256 SF columns, and
complete M128 tile coverage without overlap. Its conservative full allocation
is 35,913,728 bytes, below the signed 32-bit limit. Quantization, shared loads,
global stores, route order and synchronization are unchanged.

[CPU12](../measurements/glm53_ep_local_20260908/cpu12/README.md) passed 65 pinned
CPU tests without failures, errors, skips or CUDA initialization, plus actual
E72/I2048 CuTe and all 24 Triton compiles. REG168/STACK112/SHARED1024 stay
unchanged. CuTe PTX/cubin sizes fall from 955476/307264 to 942077/300864 bytes;
all 24 remap PTX hashes match CPU11. This establishes compilation and CPU
address equivalence, not GPU performance. Initial head and worker checks
refused insufficient available RAM before compilation. The unchanged normal
4 GiB/two-CPU runner started after a later boot transition freed 92.02 GiB on
head; no serving memory was reclaimed. That revision selected the CPU12 receipt.
The receipt-bound PTX inspection shows 41 to nine address arithmetic
instructions at each of ten equal/varied compiler copies: 320 static
instructions removed. All 500 before/after instruction lines and shared-load/
scale-store endpoints were verified. These counts exclude input SF setup,
physical-row loads, payload stores and pointer widening; they are not SASS,
executed work or measured prefill improvement.

Source `18148116a7abb242741d5b112c8d735353d9fc71` moves selected-scale
equality into lane 0's existing route-allocation loop. It compares the
already-loaded transformed Float32 scales while the input copy is in flight,
and stores the original raw bits unchanged. The first route is not compared
with itself: one NaN retains the original equal-scale behavior, repeated NaNs
remain unequal, and signed-zero equality still selects the first scale bits.

Per-warp slot 31 now carries the count in bits 0–3 and equality in bit 4,
using the same single shared store/load and existing warp barrier. Consumers
decode the count before checking for an empty route list: state 16 is an empty
count with equality set, so it must not read a stale scale. Each positive
consumer reloads the first scale once and decodes the published flag, avoiding
the old serial scale scan. The inherited pinned kernel does not read these
route slots after this override returns; the same following CTA/grid barriers
protect their lifetime before sA is reused for compute. No shared slot,
allocation, store, atomic or barrier is added. Varied-scale quantization still
reads each selected scale at its original quantizer site.

[CPU13](../measurements/glm53_ep_local_20260908/cpu13/README.md) passes actual
E72/I2048 CuTe and all 24 Triton compilations, plus 68 pinned CPU tests without
failures, errors, skips or CUDA initialization. The actual-source oracle
executes the complete lane-0 filter/allocation/publication and isolated
consumer namespaces for all four active warps and 32 lanes. It covers counts
0–8, raw NaN and signed-zero bits, invalid/remote/zero-weight holes, stale
state transitions and unchanged surrounding slots. REG168/STACK112/SHARED1024
remain unchanged; CuTe PTX/cubin sizes change from 942077/300864 to
941539/298456 bytes. All 24 remap PTX hashes match CPU12. That revision selected CPU13; GPU numerics, sanitizer and performance remained pending.

The receipt-bound [scale-state PTX inspection](../measurements/glm53_ep_local_20260908/cpu13/scale-state-inspection.md)
confirms consumer selected-scale load sites fall from seven to one, while
the seven static Float32 comparisons move into the existing producer loop.
The count/state word still has one shared load and one store. For C valid
local routes, the consumer reads C scales before this change and one after
it when C is positive; empty routes read none. These accesses are warp
broadcasts. This is an instruction-level observation, not a transaction or
speedup claim; moving comparisons into lane 0 can change runtime scheduling.

The preceding source `7254422f044ab3c5d042f32ee7f33add7baf5e00` passed actual
E72/I2048 CuTe compilation and 48 focused
CPU tests without skips in the immutable-image no-device runner. CUDA remained
uninitialized. All 24 admitted Triton dtype/branch specializations also
compiled for explicit SM121 without a device. [CPU8 evidence](../measurements/glm53_ep_local_20260908/cpu8/README.md)
confirms zero `ld.global` instructions in all six empty-map variants and
route-predicated weight loads in the other 18 variants. CuTe resources remain
168 registers, 1040 stack bytes and 1024 shared bytes. Its static PTX
`st.global.v4.u32` count changed from 1 to 33 while scalar fallback code
remains; this is not an executed-store count or a latency measurement.
Neither the CPU8 load masking, Tensor reuse and vector publication nor the
CPU9 selected-scale storage and host metadata changes have a new-source GPU
numerical or performance result yet.

The historical [CPU7 evidence](../measurements/glm53_ep_local_20260908/cpu7/README.md)
for source `38aa70f239e1e5a5b9052ae7839438eccadf66dc` passed 37 tests and
had the same resource counts. Its CuTe PTX/cubin matched
[CPU6](../measurements/glm53_ep_local_20260908/cpu6/README.md), which passed
29 tests. The original candidate used 168 registers and 1520 stack bytes;
the 480-byte (31.6%) stack reduction remains a compiler resource result,
not a GPU latency result. The unchanged stock generic
E72/I2048 arm last compiled at 255 registers and 432 stack bytes in
[cpu4](../measurements/glm53_ep_local_20260908/cpu4/README.md).

The preceding source `36d4f006bdb0850011dccdbe2a5b8de64789e0b3` completed
eight plain GPU fixtures as `eplocal0908v2`; all passed the numerical gates.
[Attempt 2 evidence](../measurements/glm53_ep_local_20260908/attempt2/README.md)
records 2.105x–3.545x compact-to-local component speedups across the five
timed balanced/concentrated fixtures. Both arms received pre-remapped routes;
these single-GB10 timings exclude remap, shared expert, transport and full-model
prefill. Remote, duplicate and zero-weight fixtures were not timed. The first
memcheck cell failed before execution because the configured sanitizer path
did not exist (exit 127); no memcheck/racecheck verdict was obtained. The exact
four-node original service was restored, and normal fleet release is recorded
at 15:30:44 KST. This result cannot validate the newer fused remap or CTA
scale cache, or establish production TP4/TTFT improvement.

The intervening v3 queue entry was cancelled normally before its payload ran.
The corrected v4 source `71e804e7aa6b29d6ddf4577809a5fa5e05a999e6`
ran from 15:58:21 KST. [Attempt 4 evidence](../measurements/glm53_ep_local_20260908/attempt4/README.md)
records a passing 24-variant GPU remap byte oracle and all eight MoE numerical
fixtures, including changed scales at fixed addresses. The five timed fixtures
showed 2.053x–3.495x versus the existing EP compact wrapper with each arm's
remap included. These are single-GB10 component results and do not measure
incremental improvement against v2, which used a different timing scope.
Attempt4 predates the CPU8 load masking, full-scratch Tensor reuse and vector
task publication and CPU9 scale-cache changes, so it does not validate their
incremental benefit.

The mounted sanitizer preflight and remap memcheck passed; the latter reported
zero errors. MoE memcheck then exited 86 with 34 CUDA_ERROR_INVALID_VALUE
reports on cuGetProcAddress_v2. Every reported stack was in the original
compact arm's initial hardware-info/binding path, before the first kernel
compile. No device-memory fault heading was reported, but the sanitizer gate
failed; subsequent two memcheck and four racecheck cells did not run. The
instrumented probe's numerical PASS does not override that failure. Exact
original four-node recovery completed before normal fleet release at
16:08:37 KST. Driver/binding compatibility needs a separate bounded diagnosis;
the errors have not been suppressed or accepted as sanitizer success.
[No-device diagnostics](../measurements/glm53_ep_local_20260908/bindings_diagnostic/README.md)
confirmed cuda-bindings 13.3.1 with driver API 13000. Sanitizer did not begin
API instrumentation in those no-device processes, so they neither reproduced
nor cleared the 34 errors. A minimal reserved-GPU reproduction must open a
Torch context before the first binding device-count call, matching the original
ordering without importing or running MoE.
The reserved binding-reproducer v2 attempt was refused before its GPU
payload because all four incoming service containers were already stopped.
The normal supervisor restored the four public containers and health 200,
then released the reservation at 16:49:14 KST. No binding diagnosis follows
from that refused attempt. The CPU8-pinned retry `epbindinggpu0908v3` received
GO at 17:35:38 KST after a normal restore-responsibility handoff. Its incoming
snapshot had the head stopped and all three workers running; the same strict
guard rejected it before the GPU payload. The supervisor passed restoration
responsibility to the next queued boot. Its exit 1 is not an exact-original
restore or binding result. A later read-only snapshot found all four public
containers running with health 200; this is separate from the failed attempt.
V4 used the same diagnostic/CPU8 source with the latest normal scheduler after
observing all four public containers healthy and idle. It received GO at
17:55:30, but a request was active at its mandatory idle check, so again no GPU
payload ran. The supervisor's separate public restoration failed an
approved-main CPU regression gate and released at 17:56:37. This does not
establish exact restoration or GPU evidence; later observations belong to
the next holder's boot. Repeated submissions are paused until the idle
boundary and normal restoration path are ready. This describes the v4 stop;
the subsequent v5 submission is recorded below.
[Submission evidence](../measurements/glm53_ep_local_20260908/binding_gpu_submission/README.md)
keeps the pre-GPU failures and retry separate.

The v4 restore CPU failure was reproduced as a test-fixture environment leak:
the supervisor's FLEET_PREPARE_MANIFEST caused synthetic pending-job edits to
prepare fake executables. The isolated fixture fix is in draft
[PR #486](https://github.com/choiceoh/stkernel/pull/486). The same injected
environment changed from two failures plus two errors among seven tests to
eight passes without skips. This changes no runtime admission or restoration
rule. [Reproduction evidence](../measurements/glm53_ep_local_20260908/restore_env_diagnostic/README.md)
does not establish that approved main contains the fix or that live restoration
has passed.

Approved main subsequently incorporated the independent PR #487 fixture
repair (`d95a2cd`), and the normal recovery holder released at 18:48. At 18:59,
`epbindinggpu0908v5` passed normal preflight and entered the queue using
frozen `8dc665b0` with its matching CPU11 63-test receipt and PR484 lifecycle.
The [v5 snapshot](../measurements/glm53_ep_local_20260908/binding_gpu_submission/v5queued/README.md)
records submission, not GPU execution or successful recovery by this job.
This API-only diagnostic deliberately keeps its already-compiled source;
it neither executes nor validates the later CPU12/13 Q0 changes. GO-time
incoming-state, idle, identity and recovery checks remain enforced.

V5 subsequently received GO at 19:51:06 and [completed](../measurements/glm53_ep_local_20260908/binding_gpu_submission/v5completed/README.md)
with the same 34 `cuGetProcAddress_v2` API errors, sanitizer exit 86 and outer
exit 1. Torch context creation completed first; all reports fall inside the
first binding `cuDeviceGetCount()` call. Count and driver-version queries
returned success (1 device, API 13000). No MoE or CuTe was imported or run.
Thus the binding initialization reproduces the blocker independently of the
kernel; the log alone does not establish which lookup versions caused it.
All four original public container identities, configuration and source
hashes were restored with their running state before the normal handoff at
19:54:31. The separate later health snapshot belongs to the next holder's boot.
No identical retry is queued; compatibility needs a changed, bounded experiment
before the full GPU suite resumes.

Static analysis of the installed, package-RECORD-matched `cydriver` binary
subsequently mapped every reported site to an ABI request above the observed
driver version: 9 at 13010, 12 at 13020 and 13 at 13030. These match all 34
newer-version requests in NVIDIA's 13.3.1 source. The official 13.0.3 loader
has none above 13000. A bounded no-device inventory confirms the relevant
Torch/Cutlass/FlashInfer dependency constraints allow a paired bindings and
cuda-python 13.0.3 capsule, with the base image preserved. This identifies a
concrete compatibility experiment, not a successful runtime replacement.

`glm53_ep_bindings_capsule.py` stages the two exact official 13.0.3 wheels in
a separate site directory and validates every file against an externally
pinned manifest. The fixed no-device capsule runner checks target-runtime
dependencies and actual import origins while retaining base pathfinder.
`run_glm53_ep_bindings_pair_offline.py` then compares two fresh processes under
one normal fleet pause: baseline installed 13.3.1 and candidate capsule 13.0.3.
Both preserve Torch-context-before-binding ordering and unsuppressed memcheck.
The baseline's 34 errors remain a failed cell and nonzero outer exit; only an
exactly matched reproduction plus a clean candidate can yield the separate
`COMPATIBILITY_OBSERVED` diagnostic. Capsule identity is checked before and
after, and original serving identity/state recovery stays mandatory. This
runner does not run MoE/CuTe or establish the current kernel's GPU acceptance.

The [capsule CPU2 check](../measurements/glm53_ep_local_20260908/bindings_capsule_cpu/README.md)
passed at 20:50:20 KST with both actual 13.0.3 binary imports verified and
base pathfinder unchanged. The image contains 268 raw distribution records;
Python's actual metadata lookup selects 264 and shadows four. CPU1 correctly
refused ambiguous raw duplicates before imports; the corrected checker binds
each runtime-selected record to the preserved original path and hash. No new
or selected-package dependency conflict was introduced; two existing unrelated
conflicts remain recorded. No Torch/context/device or CUDA API was used.
The [v6 pair](../measurements/glm53_ep_local_20260908/binding_gpu_submission/v6queued/README.md)
then entered the normal queue at 20:52:34 using frozen `63f56a54` and the exact
CPU2 capsule. The subsequent [completed v6 pair](../measurements/glm53_ep_local_20260908/binding_gpu_submission/v6completed/README.md)
received GO at 21:02:53 and independently verified **34 errors/exit86 for
installed13.3.1 versus 0 errors/exit0 for capsule13.0.3**. Both returned one
device and driver API13000. The exact baseline failure is retained, so the
separate `COMPATIBILITY_OBSERVED` verdict still has outer exit1. All four
original stopped container records match the restored records exactly before
handoff at 21:03:09. The later live container changes are separately recorded
in the next holder's interval. V6 is complete and no longer queued; this is
minimal binding compatibility, not full CuTe/MoE or prefill validation.

The subsequent CPU14 candidate removes the Q0 store helper's short-request
L2-retention predicate from both store paths: this dispatcher only admits
4096–16384 rows, so the existing plain `st.global.u64` behavior applies across
its entire domain. Histogram and lane-0 route allocation now reject non-local
expert IDs before loading their unused weights. Valid weights retain the
same Float32 comparison, NaN/zero behavior, route order and atomics. Focused
actual-source oracles pass 8 route-cache and 10 publication tests, including
poisoned invalid-route weight storage and exact store addresses/payloads.
That revision required a matching CPU14 receipt before GPU
execution; queued binding v6 retains its independent frozen CPU13 kernel.
The [CPU14 preparation](../measurements/glm53_ep_local_20260908/cpu14/README.md)
first refused head's 6.10 GiB available memory against the unchanged 12 GiB
guard. After the prior holder ended, memory rose to 88.89 GiB and one normal
CPU14 run passed at 21:04:38 KST: actual CuTe, 24 Triton variants and 71 pinned
tests with zero failures/errors/skips/CUDA initialization. Source `881456a1`
binds all 13 mounted and 18 contract files. REG168/STACK112/SHARED1024 stay
unchanged; PTX941539→939211B and cubin298456→288664B. All remap PTX hashes match
CPU13. The [receipt-bound PTX inspection](../measurements/glm53_ep_local_20260908/cpu14/q0-load-store-inspection.md)
checks eight static weight loads now guarded by a valid-ID branch and ten Q0
adaptive stores replaced by ten plain stores. It does not claim fewer executed
transactions or measured speed. This uses the original image's bindings: a full capsule-bound CuTe
compile and GPU suite still precede any performance conclusion.

The [CPU15 integration](../measurements/glm53_ep_local_20260908/cpu15/README.md)
connects the proven capsule to the actual CuTe compiler and every MoE/remap
GPU cell. Fixed read-only mounts and exact Python environment are shared;
actual binding binaries, paired metadata and base pathfinder identity are
checked before accelerator work and after it. The current runner requires
CPU15's matching runtime/source receipt, explicit PASS/complete and successful
final recheck; a failed result with full artifacts is rejected. Local focused
contracts and independent review pass. Actual capsule-bound compiler and GPU
results are recorded separately when executed, not inferred from v6.
CPU15 source `30608530` and its bundle stayed fixed. The bounded readiness
controller submitted once at 21:53:07 after the unchanged 12 GiB memory guard
passed, then exited. Actual CuTe and all 24 remap compiles completed with the
same PTX as CPU14; CuTe cubin also matches. The 129-test suite had one error
and no skips: the old sanitizer ordering test omitted the new required CLI
arguments. Its [original failure](../measurements/glm53_ep_local_20260908/cpu15/failed-compile/README.md)
and all 58 job files are preserved. Runtime identity rechecked successfully,
but the incomplete FAIL receipt cannot admit GPU work. The corrected fixture
must now reach the intended sanitizer preflight and prove no service or GPU
process action follows its failure.

The subsequent CPU16 candidate caches the row-only M128 SFA offset in the
unused physical-row slots 8–15 of each warp's existing 32-word shared region.
Lane 0 computes each base after row allocation. Both equal- and varied-scale
paths load that base and combine only the SF-column fields. Physical rows
retain slots 0–7, and raw scales/count state remain in their separate region.
The existing warp publication and next-batch CTA barrier cover these writes;
no buffer, launch or barrier is added. This trades repeated integer address
work for one producer store and an extra broadcast shared load per consumer
iteration. Compiler and GPU results must determine the actual benefit.
The nonnegative row-count prefix also uses unsigned M128 ceil division,
removing unnecessary signed-division correction without changing task order.
CPU16 is the new source-bound receipt selected by the offline runner; CPU15
is historical partial evidence and is never overwritten or retried.
The [actual CPU16 job](../measurements/glm53_ep_local_20260908/cpu16/README.md)
then passed at 22:04:46 KST: CuTe, all 24 remap variants and all 134 pinned
CPU tests without skips. Source `111fff02` matches all 13 mounted and 27
contract files; capsule runtime identity passed before and after execution,
and CUDA stayed uninitialized. REG168/STACK112/SHARED1024 are unchanged,
while PTX grew 939211→964424B and cubin 288664→300032B. The additional shared
load and code-size growth remain performance tradeoffs to measure. This is
complete CPU admission evidence, not GPU numerics or a throughput verdict.

Compute Sanitizer 2025.3.1.0's executable SHA-256 and its actual head/image
no-device launch are recorded in [cpu7](../measurements/glm53_ep_local_20260908/cpu7/README.md).
The v4 CPU proof, mounted sources, raw logs, recovery and release records are
preserved in attempt4. Full-model and production TP4 improvement remain open.

The isolated GPU runner uses the actual legacy compact wrapper as its control
with the profile's 8192-token pair-slice capacity. Eight fixtures cover balanced,
concentrated, empty-local, duplicate, zero-weight and odd-tail routes, plus
16384 rows. It changes input/routes/scales at fixed addresses, poisons output, checks
nondefault streams and includes memcheck/racecheck cells. Three stock repeats
must first agree within fixed per-row relative-L2 0.02 / normalized-peak 0.04
bounds; unstable stock cannot inflate the candidate tolerance. Candidate
bounds remain the larger of those floors and three times the bounded stock
noise. Failed runs retain the phase and partial measurements in JSON.
Both timed arms include their actual route-remap method before MoE, so new
timings are not directly interchangeable with earlier MoE-only timings.
The synthetic 16384-row case gets matching remap scratch while preserving
the legacy 8192-token pair-slice limit; it is not serving-capacity proof.
A separate 24-specialization byte oracle tests map, empty-map and offset
remapping, including bounds, duplicate IDs, changed storage, tail rows,
NaN payloads, infinities and signed zero. It also runs under each sanitizer.

Before any service inventory or pause, the runner verifies that every mounted
MoE source and probe/test contract matches the passing no-device compilation
receipt and runs a bounded no-device check of the pinned sanitizer executable.
Missing or stale proof fails closed. The GPU container checks the
installed source hashes again. A tested normal-fleet
lifecycle stops and restores exact incoming containers around these checks.
The EP and binding runners also accept four existing containers that a fleet
donor has stopped. They preserve the original container identities, source and
configuration, and restore each original running flag. A stopped incoming set
never takes the start or health-wait path; idempotent stop operations inspect
already-stopped containers without a Docker mutation. The fleet supervisor
retains the final public restore or queue handoff decision. Mixed running/stopped
sets, partial inventories, changed images and missing source evidence remain
refused. CPU tests cover stopped success, probe failure, cancellation, identity
changes, and both actual runner entry paths with fake system commands. These
control-flow tests do not establish GPU numerics or performance. The changed
runner/test hashes require fresh source-bound CPU evidence before an EP GPU run;
the earlier CPU8 receipt is not relabeled as validation of this change.
After integrating PR #484, a local 35-test lifecycle/binding/local/sanitizer
run had 31 passes and four existing host-Torch numerics skips, with no errors
or failures. At that revision the 13 mounted MoE sources matched CPU9; six
runner/test contract files had changed. Its output and hashes remain in
serving_metadata. CPU10 subsequently supplied fresh pinned evidence for that
lifecycle. CPU11 covers the earlier row-address kernel; CPU12 covers Q0
scale addressing. CPU13 covers the packed scale-state producer and
all 18 probe contract files without skips.
GPU correctness and sanitizer checks must
compare full-token output with the existing E72 compact path using identical
weights, balanced/concentrated/empty-local routes, odd tails and changed
inputs. TP4 wire numerics and current-capacity fresh 2K/32K/128K TTFT, output
quality, decode and memory checks follow before any default recommendation.

The existing `bench/prefill_serving.py` bracket still needs an EP-specific
contract before serving validation: it admits single-knob MoE/MLA candidates
and forces max-length 262144 / KV blocks 415. The next bracket must snapshot
and retain current capacity across B1/A/B2 and restoration. The launcher
recomputes KV blocks from KV_TOKENS plus its hybrid-block reservation, so
setting a captured KV_BLOCKS environment value alone does not preserve the
effective capacity. The frozen input must reproduce the observed block count,
which must then be checked against every arm's actual launch arguments.

`ENABLE_EP` is a launcher input converted to `--enable-expert-parallel`; it
is not itself passed into the container environment. The earlier prototype
recorded a separate `parallelism` field from the encoded container launch command,
bound to its observed container ID. The pure parser reports EP, TP size,
node count/rank and command hashes without executing the payload or storing
raw arguments. Unsupported commands produce unknown topology with an issue,
not a claimed EP-disabled launch. This is configured-launch metadata, not
proof that an EP kernel executed. The EP-local serving marker is also in the
generic head-log proof table. The SP arming marker alone is insufficient;
the model also emits `MHC token shards selected` after its shape, metadata and
all-layer reduction gates pass. Collect that selection marker from fresh logs
on all four ranks; selection does not establish layer completion or numerics.
That prototype's environment and command lookups used the observed container ID to avoid
mixing settings if a container is replaced under the same name. The eight
parser and three collector/proof tests passed, as did 6795 core checks and 38
megakernel regressions. Real configured-launch inputs from all four current
public containers also parsed successfully; this is compatibility evidence,
not EP execution proof. Canonical `bench/onepass.py` was subsequently restored
for the fleet onepass-only workflow and does not collect this field. The two
prototype collector tests were retired; the pure parser and actual EP launch
proof tests remain. Current serving evidence requires separate private
four-node inspections bound to container IDs, as captured in onepass3.
The historical archive and its source hashes remain unchanged. See the
[metadata evidence](../measurements/glm53_ep_local_20260908/serving_metadata/README.md).

`bench/glm53_ep_serving_contract.py` now provides pure configuration checks
for the future bracket. It accepts known public launch inputs for the incoming
capacity snapshot and private inputs for B1/A/B2, verifies exact rank/TP/EP
flags and the explicit EP-local/compact settings, and rejects other per-node
environment or normalized-command differences. It derives replay controls
from observed max length, batch/sequence limits, effective GMU, graph and
prefix-cache settings, setting CG_UTIL_DELTA=0 to avoid applying the launcher's
GMU deduction twice. Both resolved KV_TOKENS and KV_HYBRID_BLOCKS must be
supplied and reproduce the observed block count. They cannot be uniquely
recovered from Docker Cmd. The original four public-rank capacity records
are mandatory, so three consistently reduced new arms cannot pass. Exact
COMPILE_CFG is retained and replayed; the existing launcher custom-ops
transformation must preserve its bytes, or the helper rejects the config.
The latest helper also requires original Cmd/Env and container ID/start time,
accepting only typed per-node public-to-private host/port substitutions. It
compares the full original argv and environment against every arm except
the explicit EP flag and two EP settings, and preserves supplied image/model/
source provenance. An arm-envelope checksum detects accidental changes to
knobs or replay controls after configuration; it does not authenticate input
snapshots. Twenty focused tests passed without skips, including three-arm
configuration drift and outer-envelope mutation. The earlier eight parser
tests are unchanged. This helper starts no processes and is not yet connected
to the full serving runner; live source/image/log identity, all-rank execution,
actual capacity restoration and direct fresh-request metrics remain required.

A future EP arm must separately record requested ENABLE_EP and compare it
with configured topology, require all four ranks' E72/I2048 EP-local launch
and actual MHC token-shard selection, and attest source/image/model/log
identity. Raw command hashes differ when the EP flag changes: only that
exact flag may be excluded in the normalized argv comparison, while the
prelude hash and every other argument remain checked. The candidate's
launcher-added VLLM_B12X_EP_COMPACT setting also needs an explicit contract.
The current generic comparator does not yet implement these EP rules; its
baseline classification still reads environment knobs without rejecting
EP-only launch changes through the new metadata.
Current fleet admission uses canonical onepass deployment and its standard
2K/32K/128K requests. The earlier custom fresh-request wrapper and chain hooks
are not part of this run. All-rank configuration/execution checks are retained
as separate evidence; the unresolved GPU numerical gate still blocks acceptance.
Short requests and decode use other EP paths and still need direct checks.

The next canonical onepass enables `VLLM_B12X_EP_WARM_COMPACT=1` only on A.
The default remains 0. Preparation enumerates reachable padded/sliced compact
rows and uses the dispatcher's real backend, workspace and compiler keys to
deduplicate them. At the current 8192-token capacity this covers 128 possible
launch sizes with 14 calls: four dynamic specializations and ten static tails,
including 128 and 192. Matching layers reuse completed preparation; any launch,
synchronization or missing-key failure aborts required preparation. This moves
compact compilation before readiness, without claiming unrelated JIT is gone.

All arms retain the shared unused-graph-estimate skip and real graph capture.
Its proof now requires both log events. `ONEPASS_BASELINE_KNOBS` declares the
exact shared control for this session; production-default baseline semantics
and strict active-knob proof remain unchanged. Same build, session, workload,
endpoint and boot identity are required, with separate launch inspections for
image/capacity because canonical rows omit that runtime metadata. A standard
B1/A/B2 run supplies direct tok/s and TTFT; two baseline boots still do not
satisfy the existing three-sample prefill noise gate. Previous rows and verdicts
remain historical evidence, including the unresolved decode regression.

A preceding two-stripe prototype was withdrawn before GPU submission after
finding preserved failures on branch `codex/glm53-prefill-moe-overlap`
(`021131d`). It is saved only on local branch
`codex/glm53-prefill-moe-pipeline` at `15b7bd0`. Do not queue it.

### CPU16 full GPU admission

[Full GPU v5](../measurements/glm53_ep_local_20260908/gpu5-queued/README.md)
entered the normal queue at 22:12:31 KST using frozen source `d53fd44f` and
CPU16's matching 134-test/capsule receipt. At the preserved snapshot it was
first behind arconsumer0908v17, with no GPU payload started. A shared-clone
preparation failure was resolved by a fresh explicit-SHA shallow source;
the original partial source and failure are preserved, with no duplicate job.

The [CPU16 PTX inspection](../measurements/glm53_ep_local_20260908/cpu16/sfa-row-base-inspection.md)
confirms row-only scale arithmetic moved into the producer. Each emitted
consumer now uses a cache-address add, shared load and SF-field combine.
The compiler also fully unrolled the 72-expert prefix and increased varied
Q0 static copies from three to seven. These explain code-shape changes,
not executed store counts or a measured improvement.

### Full GPU v5 numerical rejection

The [closed offline capture](../measurements/glm53_ep_local_20260908/gpu5-result-pending-restore/README.md)
records GO at22:17:24 and wrapper completion at22:20:32 KST. Remap24 and
balanced4096/6912/8192 passed. Concentrated6912 failed in changed-candidate
with one bad row: relative L2 max0.0118141882, normalized peak max0.0406976752.
The stock control passed and runtime identity rechecked successfully. The
first candidate and its nondefault-stream replay had passed; the failure
followed in-place input/routes/scales changes. Tolerances remain unchanged.
Four remaining MoE fixtures and all eight sanitizer cells were not run.
The current candidate is rejected by this gate; no speed/default acceptance
follows from the successful balanced cells. This result does not isolate
CPU16's address cache from earlier kernel changes or accumulation order.

The offline before/stopped/restored records are exactly equal for all four
incoming stopped containers. The earlier snapshot was taken while the normal
fleet supervisor was still restoring public serving. The later
[terminal archive](../measurements/glm53_ep_local_20260908/gpu5-completed/README.md)
separately confirms health/proof completion, restore-finished rc0 and normal
release at22:26:43 KST. The outer job retains exit1 for the numerical failure.
All204 archived inputs have original/stored hashes, including the closed job,
frozen source and CPU16 evidence, runtime receipts and fleet lifecycle records.
No current-container equality after release is inferred from those snapshots.
The next step is to identify the failing row/element and repeatability before
changing kernel arithmetic or running another full GPU suite.

### Bounded numerical diagnostics

The candidate comparison now retains its first failure and records up to eight
bad rows with their actual per-row L2/peak errors, reference noise, limits and
reference norms. It stores the original top8 routes and expert scales, plus
BF16 words from B1/B2/B3/C1 at the eight largest absolute output differences.
Capture runs only after a failed comparison; even a diagnostic error leaves
the original numerical failure intact. It does not execute another candidate
or alter the fixture, thresholds, successful comparisons or kernel arithmetic.

The normal offline runner accepts `--diagnose-case concentrated6912` to execute
that existing cell once through the same source, capsule, resource and lifecycle
checks. Completion identifies the diagnostic mode and explicitly withholds full
GPU acceptance. The default full suite is unchanged. These changed probe/helper
contracts require a fresh CPU17 receipt; CPU16 still describes the measured
kernel source but cannot admit the new diagnostic source.
The [local preparation evidence](../measurements/glm53_ep_local_20260908/diagnostics-prepared/README.md)
records eight diagnostic and17 wrapper tests passing without skips. Full local
integration has135 passes,11 skips for missing host Torch/packaging and no errors
or failures. The server gate still requires all146 tests without skips, actual
CuTe/24-remap compilation and matching runtime/source receipts.
CPU17 admission was attempted once against published diagnostic source
`81b8ef91`; host MemAvailable7791744KiB was below the unchanged12GiB guard.
No remote CPU17 source, bundle transfer or job was created. The numerical
diagnostic has not been submitted to GPU.
The23:05 KST follow-up found6867044KiB available and still no CPU17 source/job,
so the preparer was not rerun. Source-bound CPU17 collection and the subsequent
single-case GPU submission are now prepared in `diagnostics-prepared`, with
syntax checks and independent read-only review. Both remain unexecuted. The
GPU preparer requires the successful CPU17 receipt commit to be the immediate
single child of the actual CPU17 source revision; no unrelated commit may be
inserted between that source and its receipt for this prepared depth2 path.

### CPU17 completion and binary provenance

CPU17 ran through the normal CPU lane on September9 at00:12:23–00:12:39 KST,
using frozen source `123fbb01211eaf8cec46fd0965dd9320d02a375b`. All146 pinned
tests passed without skips, errors or failures, with29 contracts,13 mounted
sources and24 remap compilations. CUDA stayed uninitialized and the13.0.3
capsule identity passed its before/after checks. Available host memory was
99919020KiB; the12GiB guard was retained. The approved scheduler uses its own
independent checkout; normal shared fleet coordination and production
restoration policy are unchanged.

The first collector rejected exact binary equality and preserved the original
files and comparison under `cpu17.collecting`. CuTe PTX, cubin and resource
output, and all24 remap PTX files, match CPU16 exactly. The24 remap cubins
have different hashes. A strict ELF comparison found exactly four differing bytes per cubin,
confined to the source mtime in two non-allocated DWARF debug-line sections.
The decoded times match the two frozen checkouts. All executable/allocated
sections, headers and every other byte are identical. Six altered-byte checks
were rejected. The original mismatch is preserved; full cubin equality is
not claimed. The [CPU17 archive](../measurements/glm53_ep_local_20260908/cpu17/README.md)
records this separate adjudication plus original/stored hashes and successful
remote identity checks before and after copying. This is compiler evidence
only. Concentrated6912 numerics and the full GPU/sanitizer gates remain open.


The [single-case diagnostic](../measurements/glm53_ep_local_20260908/diag-gpu1-queued/README.md)
was submitted with frozen receipt source `69f5b833`. Approved scheduler source
`73b2ca05` passed its mandatory validation, then enqueued `eplocaldiag0908v1`
at00:25:55 KST. Its captured queue position is historical and does not prove
GPU execution. The concentrated case includes paired component timing only
after all original numerical comparisons pass.

### Concentrated diagnostic result

The [terminal diagnostic capture](../measurements/glm53_ep_local_20260908/diag-gpu1-completed/RESULTS.md)
records an earlier-than-estimated GO at00:29:01 KST, cell completion at00:30:27
and restored incoming containers plus accepted normal handoff/release at00:32:55.
Source `69f5b833` used the matching CPU17 receipt and13.0.3 runtime. All six stock
control comparisons and four candidate comparisons passed. There was no failed
row to capture in this invocation. The original GPU5 numerical rejection remains.

For the synthetic6912-row concentrated case, eight alternating-order paired
rounds (three calls per arm per round) measured wall medians53.9281ms compact
versus14.6132ms local:3.6904x, or72.90% lower component latency. Device medians
were53.9244ms and14.6102ms. Each round favored local (3.52–3.83x wall ratio).
This is EP remap plus MoE wrapper timing; it excludes shared expert, TP transport,
attention and full-model TTFT. It does not establish the campaign's1.40x direct
prefill target or an incremental CPU16-versus-CPU14 improvement.

The changed-input peak error was0.0327869 (default stream) and0.0338983
(nondefault stream), compared with GPU5's earlier failing0.0406977. The kernel,
runtime and fixture-generation algorithms match, but input tensor hashes were
not recorded across runs. The passing invocation has slightly higher maximum
L2 error than the failing invocation, so this is not a general numerical-error
improvement. It is consistent with execution variability and does not isolate
its cause. A bounded repeatability diagnostic with fixed inputs and candidate
self-comparisons is the next check; a passing repeat must not erase the first
failure or authorize an unchanged full-suite rerun.

#### Short EP decode follow-up: measured regression remains

The repaired short-only micro path completed canonical onepass B1/A on source
`e2a54cff881465c2bb7dbbd3f5ec39ca240c7f74`. Fixed 1024-token decode (three requests
per arm) measured **76.899 tok/s B1 versus 62.489 tok/s A, -18.74%**. The actual
SPEC_K5 graph shapes 6/12/18/24 use 1/2/3/3 top-k8 calls, but this call reduction
did not remove the regression. The user stopped the remaining B2 while loading;
no new GPU run was submitted. Defaults remain off and no promotion was merged.

32K/128K prefill measured +12.66%/+13.57% against B1, descriptively. Both arms
scored 18/18 facts; A had 0/8 dirty responses, while B1 had 1/8 due to two CJK
characters. The canonical verdict is incomplete because the baseline failed its
quality gate. The previous GPU numerical failure remains unresolved. See
[exact records and cancellation evidence](../measurements/glm53_ep_local_20260908/onepass6-completed/README.md).
