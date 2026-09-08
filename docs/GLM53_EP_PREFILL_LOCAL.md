# Full-token expert-local prefill

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
Scale equality is folded into the existing scale-load loop, avoiding repeated
checks for each quantization block. After histogram publication, each CTA
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

The latest source `cf1365b8fab6091833400efd7784071316eb9f6a` passed actual
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
boundary and normal restoration path are ready.
[Submission evidence](../measurements/glm53_ep_local_20260908/binding_gpu_submission/README.md)
keeps the pre-GPU failures and retry separate.

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
or failures. The 13 mounted MoE sources match CPU9; six runner/test contract
files changed. This integration check does not replace fresh pinned evidence
for the new probe contract. Its output and hashes are in serving_metadata.
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
is not itself passed into the container environment. Onepass now records a
separate `parallelism` field from the known encoded container launch command,
bound to its observed container ID. The pure parser reports EP, TP size,
node count/rank and command hashes without executing the payload or storing
raw arguments. Unsupported commands produce unknown topology with an issue,
not a claimed EP-disabled launch. This is configured-launch metadata, not
proof that an EP kernel executed. The EP-local serving marker is also in the
generic head-log proof table; the old SP marker still proves only arming.
Environment and command lookups both use the observed container ID to avoid
mixing settings if a container is replaced under the same name. The eight
parser and three collector/proof tests passed, as did 6795 core checks and 38
megakernel regressions. Real configured-launch inputs from all four current
public containers also parsed successfully; this is compatibility evidence,
not EP execution proof. See the
[metadata evidence](../measurements/glm53_ep_local_20260908/serving_metadata/README.md).

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
Existing normal chain hooks and fresh 2K/32K/128K collection can be reused
after those gaps and the complete source-bound GPU/sanitizer gate are closed.
Short requests and decode use other EP paths and still need direct checks.

A preceding two-stripe prototype was withdrawn before GPU submission after
finding preserved failures on branch `codex/glm53-prefill-moe-overlap`
(`021131d`). It is saved only on local branch
`codex/glm53-prefill-moe-pipeline` at `15b7bd0`. Do not queue it.
