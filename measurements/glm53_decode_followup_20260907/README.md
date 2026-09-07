# C=1 decoding follow-up: MHC projection work

The new lossless BF16 MHC path reduces T=6 cold kernel latency by **13.23%**.
Combined with the previous M8 fastpath, four matched serving boots show
**+1.21% engine steps/s** and **+1.55% pooled output tokens/s**. Output-rate
pairs disagree and baseline output variation is 2.71%; the throughput gain
remains inconclusive. Both candidate boots pass retrieval/Korean guards,
while the baseline boots have corruption in 3/20 outputs. The code is retained;
all four experimental flags remain default-off. No EP restructuring or new
weight quantization was introduced.

## Current decode attribution

The baseline source was `b13dc8b`, overlay `83bc6639a0a2`, and immutable image
`sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211`.
The fleet reserved the existing idle server for `profile-step.py decode:600`
at 07:59 KST. The request completed 600 tokens over 286 steps. The saved
rank-0 trace contains 284 complete analyzed steps:

| Category | Median kernel time/step | Calls/step |
|---|---:|---:|
| MoE expert | 22.87 ms | 42 |
| MK GEMM | 9.23 ms | 237 |
| One-shot AllReduce | 4.54 ms | 102 |
| MK MHC | 2.30 ms | 90 |

The profiled span is 47.81 ms, with 2.30 ms idle and 2.49 ms overlap.
These are attribution measurements with profiler overhead, not unprofiled
serving throughput. Full composition is [profile-composition.txt](profile-composition.txt).
The original trace is on srv2 at
`/home/choiceoh/vllm-prof/dp0_pp0_tp0_dcp0_ep0_rank0.1788735581053433310.pt.trace.json.gz`.

The profiler's stop/export request returned HTTP 500 after the rank-0
worker exited. The underlying cause is not established. This task restored
the unchanged baseline with the fleet-held `M8PROFRESTORE`, healthy at
08:05:58. Another fleet probe ran next; that serving worker exited at
08:07:03, before our offline probe received the fleet at 08:07:12. This
timing does not establish causation. A later `prefillgate` serving campaign
owns a different deployment (`7cb8c033a0b2`); our queued duplicate restore
was cancelled to avoid replacing its deployment. No repeat profiler capture
is planned. See [capture log](profile-current.log),
[first restoration](profile-restore.log), and
[cancelled restoration](offline-restore.log).

## SIMT attempts

Both prototypes are private probe-source transformations, not deployed code.
They keep FP32 arithmetic, reuse projection weights across tokens, and
replace token-arrival atomics with a separate tail launch. Version 1 groups
all tokens; version 2 limits groups to two and parallelizes partial-sum
loads over eight warps. Both 64- and 128-coordinate chunks were measured
against the unmodified kernel within the same extension.

| T=6, best of the two new geometries | Baseline | Candidate | Latency change |
|---|---:|---:|---:|
| V1 warm | 15.471 us | 17.249 us | +11.50% |
| V1 cold | 23.552 us | 28.672 us | +21.74% |
| V2 warm | 15.627 us | 17.161 us | +9.82% |
| V2 cold | 23.584 us | 26.272 us | +11.40% |

All stock and baseline oracles, exact residual-rounding checks, and graph
replays passed. Each geometry/regime has 18 samples with all six execution
orders balanced. Both attempts lose at the served T=6 shape, so neither
will consume a serving A/B boot. [Parsed results](simt-results.json),
[V1 log](mhc-reuse-v1.log), [V2 log](mhc-reuse-v2.log).

## Other projection attempts

Compensated TF32 uses three products (high/high plus the two high/low
corrections). It preserves the residual's FP32 calculation and BF16 stores,
but changes projection rounding. The split and resident fused variants
passed the stock oracle, independent FP64 formula and repeated graphs.
Neither improved cold latency. The exact Sinkhorn fixed-point prototype
stops only when all 16 matrix values have unchanged bit patterns; it also
passed exact output gates but did not improve the measured cold path.

| T=6 | Baseline | Candidate | Change |
|---|---:|---:|---:|
| TF32 split, warm | 15.628 us | 15.518 us | -0.71% |
| TF32 split, cold | 23.568 us | 26.848 us | +13.91% |
| TF32 resident fused, warm | 15.526 us | 15.224 us | -1.95% |
| TF32 resident fused, cold | 23.552 us | 26.672 us | +13.25% |
| Exact fixed-point (period 4), cold | 23.536 us | 23.552 us | +0.07% |

Those earlier fixtures use random FP32 weights, unit scales and zero bases;
they are synthetic, not trained MHC weight distributions. They support no
serving promotion or broad rejection of all possible tensor-core/fixed-point
variants. Historical prototype source transforms must be run at their
recorded revisions, before the integrated MHC load refactor.

The tensor-core fragment mapping follows NVIDIA's
[PTX matrix-fragment specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-1688).
CPU enumeration checked complete fragment coverage, bijective shared
permutations and distinct banks for all 32 threads of each B-fragment load.

## Implemented winner: lossless BF16 storage

The checkpoint's `hc_attn_fn` and `hc_ffn_fn` tensors are BF16, while the
model allocates FP32 Parameters for them. The new opt-in
`VLLM_GLM53_MK_MHC_BF16=1` reads an exact BF16 copy and expands each value
to FP32 before the original arithmetic. It halves the 24 x 16,384 projection
weight footprint from 1,572,864 to 786,432 bytes. It does not change MMA,
reduction order, Sinkhorn iteration count or either residual rounding point.

Both original and BF16 kernels are in the same compiled extension. Each
resolves its own resident grid. The final binary uses 162 registers/thread,
28,736 bytes shared memory and no local memory for both variants; this is
a reduction in weight traffic, not an occupancy change.
[Resource report](integrated.resources).

The driver checks exact FP32 integer-bit restoration and finite values.
Unrepresentable weights, inference tensors without mutation versions,
unsupported layouts, cold cache misses during graph capture and capacity
exhaustion all keep the FP32 path. Copies are keyed by device, pointer and
mutation version. Strong references retain source storage and previous
copies so later eager mutations cannot free pointers baked into old graphs.
As with other packed model weights, weight updates require graph recapture.
The cache is bounded at 256 entries (192 MiB of BF16 copies).

The boot gate requires all four outputs to be bit-identical at T=2/6/8;
a separate receipt proves the path was captured at a served token count.
Both regular MHC and the identity-post standalone pre path use this driver.
The flag remains 0 in the profile: the kernel gate passed, while the complete
serving promotion gate remains unresolved as detailed below.

### GPU evidence

Eight actual checkpoint weight sets (attention and FFN from layers 0, 1,
23 and 44), including their trained scales and bases, were read using
bounded safetensors byte ranges. Payload SHA-256 values are in the logs;
model files were not changed. Tests cover T=1/2/6/8/12/32, ordinary and
zero residuals, the independent FP64 oracle, all four exact output tensors,
and repeated CUDA graph replay. The final-driver test additionally covers
mutation/version refresh, retained old packs, NaN/unrepresentable/inference
fallback, capture cache misses and capacity limits. All passed.

| Final integrated driver | FP32 baseline | BF16 storage | Latency change |
|---|---:|---:|---:|
| T=6 warm | 15.255 us | 13.641 us | -10.58% |
| T=6 cold | 23.344 us | 20.256 us | -13.23% |
| T=8 warm | 17.096 us | 15.431 us | -9.74% |
| T=8 cold | 23.296 us | 20.224 us | -13.19% |

Each cell has 18 interleaved, balanced samples. Warm graphs contain 32
calls; cold graphs flush L2 before one timed call while retaining the small
inputs. Events exclude flushing and one-time packing. Some samples contain
outliers, all retained in [parsed results](kernel-results.json).
The original BF16/BF16-pair prototype and final integrated driver agree on
the gain; simpler scalar BF16 loads were selected because paired loads
provided no cold advantage.

- Final source SHA-256: `490570583b2af29382fb80f03be4a488edfc8dbb7e6a4d6cc04dcca64f59e0c2`.
- Source MD5 in serving fingerprint: `554fa17b`.
- Implementation: `0e4e438`; current-main merge for deployment: `fa185275`.
- [Prototype log](mhc-packed.log), [integrated driver log](mhc-integrated.log).
- [Local gate](logic.log): 6,665 checks, 30 megakernel regressions and 32 fleet regressions.

## Serving follow-up

Running on overlay `990fe078c790`, source `b535405d` (measurement harness
`b17860aa`), the same immutable image
as the GPU gate. Order is B1/A1/A2/B2, four independent boots. Each arm runs
the identical Korean quality/prefill workload and five fixed 2,048-token
requests. Only complete decode windows count as the primary engine-step
metric. Output tok/s and TPOT are reported separately, with prompt hashes,
request seeds, completion lengths, traffic exclusivity, quality, Korean
corruption and capture receipts checked before comparison.

A combines the prior M8 fastpath flags with the new BF16 MHC flag. B leaves
all four off. Both have `VLLM_GLM53_FP8_CACHE=0` because srv1 had no writable
disk space. This affects startup caching, not the in-memory representation;
it is honestly recorded as a non-default knob. These B arms are matched
runtime baselines, not reusable profile-default baselines. The fleet judge retains an incomplete verdict because these are not empty-knob
profile-default baselines. The accompanying analyzer verifies the exact shared
cache override and computes the matched boots' descriptive spread without
rewriting their attested knobs or borrowing a floor from another build.

The first public-endpoint run, `MHCBF16B1`, was invalidated by an external
client's eleventh generation request arriving during the ten-request workload.
It is excluded, not repaired by dropping affected windows. The harness restored
the public baseline, then the comparison restarted on 127.0.0.1:18000.
[Endpoint isolation](endpoint-isolation.txt) verifies loopback health=200 and
LAN health=000. Generation and metric sampling use the same recorded endpoint;
ordinary launches still bind 0.0.0.0:8000. An exit handler restores port 8000.

### Final matched results

Four independent boots, 248 interior windows, 20 fixed requests and 40,960
fixed completion tokens. All traffic, source, image, flag, prompt-hash,
request-seed and completion-length checks passed. The analyzer treats each
boot as one replicate and gives both boots equal weight.

| Measurement | Baseline B | Candidate A | Change |
|---|---:|---:|---:|
| Pooled engine steps/s, averaged across boots | 21.6782 | 21.9403 | +1.21% |
| Pooled output tokens/s, averaged across boots | 70.0769 | 71.1626 | +1.55% |
| Pooled TPOT, averaged across boots | 14.2727 ms | 14.0527 ms | -1.54% |
| Median request tokens/s, averaged across boots (secondary) | 69.5980 | 72.2721 | +3.84% |
| Retrieval quality | 48/48 | 48/48 | all answered correctly |
| Outputs with Korean corruption | 3/20 | 0/20 | baseline guard failed |

| Boot | Engine steps/s | Pooled output tokens/s | Korean corruption | Proof |
|---|---:|---:|---:|---:|
| MHCBF16LB1 | 21.6324 | 71.0281 | 1/10 | controls |
| MHCBF16LA1 | 22.0482 | 70.8003 | 0/10 | 4/4 |
| MHCBF16LA2 | 21.8324 | 71.5248 | 0/10 | 4/4 |
| MHCBF16LB2 | 21.7240 | 69.1256 | 2/10 | controls |

Paired engine changes are +1.92% and +0.50%. Paired pooled output changes
are -0.32% and +3.47%. Baseline spread between boots is 0.42% for the pooled
engine metric and 2.71% for pooled output rate; candidate engine spread is
0.98%. Thus the engine signal is more encouraging than the earlier M8-only
round, but two boots per arm do not establish a narrow confidence interval
and the output-rate result still does not clear observed baseline variation.
The earlier build's numbers are context only, not a matched additional arm.

The second baseline had corruption in 2/10 outputs; the first had 1/10.
These failures remain in the summary and block promotion. This is not a
claim that M8/BF16 fixes baseline corruption: zero of five output hashes
match even between the two baseline boots at identical request seeds.
The new MHC kernel's exact differential result is a separate, narrower claim.

The [summary](summary.json) and [raw records](records.raw.jsonl) preserve every
metric and failure. [Analysis gate](analyze.py) checks the runtime evidence
before computing descriptive timing; it explicitly records quality blockers.
[All four rank receipts](runtime-LA1-srv2.json) have matching image/source
hashes, the candidate environment and an actual `mhc-bf16 CAPTURED T=6`
marker (corresponding srv1/srv3/srv4 files are beside it).

The [isolated serving log](isolated-serving.log), [isolated runner](isolated-runner.sh),
[earlier public runner](runner.sh), and [invalid public run](serving.log)
retain the full procedure. The public API was restored at 09:55:31 KST. LAN health is 200 on port 8000,
the fleet hold is released, and all four rank snapshots verify that the four
experimental flags are 0. Disk FP8 caching also remains disabled in this
particular boot; the profile still names `/cache/glm53-fp8` for ordinary future
launches. [Restoration log](public-restore.log), [final state](final-state.txt)
and `runtime-public-srv*.json` retain the checks. Windows within a boot
are correlated; more windows do not create more independent boots.

### Capacity recovery

Before deploying, 878 generated FP8 cache files older than 08:00 KST
(8,610,071,526 bytes) were copied from srv1 to
`srv2:/home/choiceoh/glm53-logs/MHCBF16/cache-archive-srv1/` and every payload
was SHA-256 verified before removing the original. Current later cache
entries and model weights were retained. Available space became 7.7 GiB.
The prior M8 archive remains untouched. The first archive attempt lacked
read permission and removed nothing; the retry used noninteractive sudo.
[Archive procedure](cache-archive.py) and [initial failure](serving-preflight-failed.log).
