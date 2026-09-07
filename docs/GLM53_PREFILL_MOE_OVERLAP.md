# MoE prefill communication overlap

The four-rank September 7 attribution measured approximately 12–13% of
prefill occupied time in collectives, with almost no compute overlap.
MoE itself occupied 28–32%. This candidate pipelines those two components
without replacing the expert kernel. It has no measured serving gain yet.
The prior trace used NVFP4 static scale 0, whereas current defaults use 16;
its percentages identify a target, not a matched current performance claim.

`VLLM_GLM53_PREFILL_MOE_OVERLAP=1` admits only the existing exact pure eager
TP4/EP1/DP1 b12x MoE sequence-parallel contract, with EPLB disabled and
6,144–8,192 actual executed tokens. Default is 0. Short chunks, dense MLPs,
decode/capture, unsupported transport, DP metadata and microbatch contexts
retain their existing path. Attention metadata is never sliced.

Each rank divides its local MHC shard into two contiguous stripes. All ranks
issue AG0, AG1, RS0, RS1 in that order on one auxiliary communicator stream.
The original compute stream waits for AG0, executes MoE0, then waits for
AG1 and executes MoE1. RS0 can overlap MoE1. Concatenating the two received
local stripes restores the original residual row order, including at most
three final padding rows. MoE sees only real rows. The two MoE invocations
remain sequential, including the runner's own shared-expert stream join.
Invocation-owned outputs are retained and recorded on consumer streams.

Both stripes use the parent chunk's BF16/FP8 v3 decision, preserving the
existing per-2048-value codec and reduction arithmetic. Letting the smaller
stripes choose transport independently would confound the experiment.
No new quantizer, weight layout, expert selection rule or tolerance is added.
Different MoE tile occupancy and BF16 atomic ordering still require direct
numerical and quality validation. At about 4K, splitting can double padded
M128 expert work; the initial gate therefore begins at 6K, pending data.

Older Torch/vLLM runners resolve a layer by incrementing a forward-context
counter. Each stripe gets a shallow child context with explicit layer-name
resolution, and the parent counter advances once after both succeed.
The layer mapping and incoming counter are verified before any collective.
No runner attributes or global model configuration are mutated.

CPU tests exhaust all 4,096–8,192 row partitions, test the actual two-stream
schedule and ownership calls, verify context/counter restoration on failure,
and cover default/short/capture/transport fallback. Existing collective and
container lifecycle regression tests also run. They do not prove GPU safety,
overlap, text quality, or TTFT.

The offline four-rank probe uses actual stock b12x E288/top8/H4096/I512 MoE
with rank-specific NVFP4 weights and a shared-expert auxiliary stream. It
checks balanced and skewed routes at 4096, 4143, 6912 and 8192 rows, every
valid output row, changed input/routes, repeated use, retained outputs and
allocator churn. BF16 and FP8 v3 execute in separate processes. Timings use
balanced B/A ordering and the slowest rank per sample. Every imported overlay
must match the same frozen source on all ranks. This is a numerical and
kernel/transport gate; full-model fresh 2K/32K/128K TTFT and Korean/retrieval
quality remain mandatory before recommending a default.

Run only through the normal fleet boot queue using
`probes/glm53_offline_checks.py --probe-source <frozen checkout>
--probe-revision <40-character commit> --out <new evidence directory>` with
`OFFLINE_SOURCE_REV` matching the runner commit. The identical frozen checkout
must be present on all four nodes. The lifecycle helper stops only its
verified incoming idle containers and restores their exact identities/configs
and healthy endpoint in `finally`; the probe removes only its own unique
four-rank containers. It requires 128 GiB free disk per node and 16 GiB
available memory before launching isolated, memory-limited probe containers.


## Direct serving comparison

The prepared `bench/prefill_serving.py` runner admits only the completed,
recovered four-rank gate above. It requires both BF16 and FP8-v3 reports,
all eight size/routing cases per transport, eager and changed-input
numerical success, the final four-rank completion marker, and exact hashes
for all 56 generated overlays, the profile, launcher and probe sources.
Adding measurement code does not authorize changing GPU-validated code.
Kernel timing is not an admission speed threshold: direct serving TTFT
remains the performance decision.

Once that gate passes, a new frozen source runs normal fleet B1/A/B2, each
with an excluded priming ladder followed by measured 2K/32K/128K requests.
Each request gets a new cache salt; the comparator rejects prefix-cache
hits, outside traffic, changed inputs/tokens, mixed sources/configs, failed
quality or absent candidate launch proof on any rank. The collector reads
the exact inspected container's redirected file log, rejects stale/empty
files and preserves incomplete evidence on failure. It freezes B1's actual
memory/scheduler controls and finally restores the public default arm.

This reuses the reviewed measurement helpers from PR #439 without importing
its MLA or failed MoE kernel candidates. The 61 relevant GPU/launch/probe
files remain byte-identical to frozen GPU revision `44d76c0`; current-main
rebase `944f65c` changes only fleet CPU handoff code. GPU and TTFT are still
pending; preparing this runner is not a measurement result.


## API correction and connected retry

The first four-rank attempt stopped before numerics because the probe used
`pipeline_parallel_size` where the pinned distributed API requires
`pipeline_model_parallel_size`. See `failed-init-api/` for every rank's log
and exact original-container/health recovery at 22:50:21 KST. No speed or
numerical verdict was obtained. The candidate's overlays, profile and
launcher are unchanged.

Corrected GPU source `ce71af6` adds source-signature binding before any CUDA
import and before the offline helper stops serving. The lifecycle context
cleans up partial initialization failures as well as the probe body. The
new serving runner's `--refresh-gate` connects this corrected full four-rank
GPU gate, original recovery, and then B1/A/B2 within one normal fleet hold.
A failed GPU gate cannot reach serving deployment. The old prepared
serving1 request is retired without submission; use the new retry receipt.
