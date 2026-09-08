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

The register-memory scale cache has eight slots for the exact top8 contract.
Scale equality is folded into the existing scale-load loop, avoiding repeated
checks for each quantization block. After histogram publication, each CTA
prepares the 72 expert scales once in the histogram's now-idle 288 shared
bytes. The existing Q0 initialization barrier publishes those stores. This
removes token/route-level global scale loads and reciprocal work without
adding shared storage or a barrier. Disabled flags and ineligible short calls
return before querying CUDA capture state; the query is lazy and only runs
after the exact shape/activation gate.

The admitted candidate also remaps global routes into the existing output
scratch in one Triton launch. It replaces the expert-map path's 14 Torch
operations; this count describes source operations, not a measured speedup.
Unsupported dtypes/layouts/devices retain the existing Torch remap. The
integer map/offset conversion order is preserved, and weights are copied as
bits so local NaN payloads and signed zero survive while remote weights
become exact positive zero. Other EP, decode and TP paths keep their remap.

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

The final candidate passed actual E72/I2048 CuTe compilation and 37 focused
CPU tests without skips in the immutable-image no-device runner. CUDA remained
uninitialized. All 24 admitted Triton dtype/branch specializations also
compiled for explicit SM121 without a device. [Final compilation evidence](../measurements/glm53_ep_local_20260908/cpu7/README.md)
for source `38aa70f239e1e5a5b9052ae7839438eccadf66dc` records 168 registers,
1040 stack bytes and 1024 shared bytes, compared with 168/1520 registers/stack for the
original candidate. This 480-byte (31.6%) stack reduction is a compiler
resource result, not a GPU latency result. Its CuTe PTX and cubin hashes are
unchanged from the historical [cpu6 compilation](../measurements/glm53_ep_local_20260908/cpu6/README.md),
which passed 29 tests before the sanitizer preflight contracts were added.
The unchanged stock generic
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
GPU proof for the newest source is still pending; the corrected v4 submission
is a subsequent step, not a completed validation. The head's no-device
sanitizer preflight now passes with Compute Sanitizer 2025.3.1.0, whose
executable SHA-256 is pinned and recorded in
[cpu7](../measurements/glm53_ep_local_20260908/cpu7/README.md). Running its
version check does not provide device memcheck or racecheck evidence.

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
GPU correctness and sanitizer checks must
compare full-token output with the existing E72 compact path using identical
weights, balanced/concentrated/empty-local routes, odd tails and changed
inputs. TP4 wire numerics and current-capacity fresh 2K/32K/128K TTFT, output
quality, decode and memory checks follow before any default recommendation.

A preceding two-stripe prototype was withdrawn before GPU submission after
finding preserved failures on branch `codex/glm53-prefill-moe-overlap`
(`021131d`). It is saved only on local branch
`codex/glm53-prefill-moe-pipeline` at `15b7bd0`. Do not queue it.
