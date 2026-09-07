# GLM-5.3-Flash prefill: 40% throughput campaign

Baseline source: `8d36ee8`. Original PR: #368 (merged).
The earlier direct-output candidate is `fa1371e`.

The target is at least **1.40x input tokens/second** on matched requests,
equivalent to reducing prefill wall time by at least **28.5714%**. It is not
a 40% reduction in wall time. Cold and warmed measurements are separate.

## Current evidence after rebase (2026-09-07)

**Current default (operator request, PR #425):** gated FP8 v3 is enabled
with `VLLM_GLM53_PREFILL_SP_FP8=3` and a 4096-real-token chunk threshold.
Short chunks use BF16; set the mode to 0 for BF16 at every length.
This promotion supersedes the historical off-by-default decisions below;
it does not establish a long-context speedup or the original 40% target.
The gated serving source was rebased onto `05c4a65`.

Rebased onto `41af400` (including the chat contract, startup caches and SF
packing). The two prefill helper changes are unchanged by these upstream commits.
Follow-up branch:
`codex/glm53-prefill-packed-transport`. **The original 41.2056% forecast below
is superseded by measurements, not a current prediction or achieved gain.**
The implementation gate already opened; subsequent corrections and candidates
can be tested without reusing the disproved assumptions to cross it again.

MEASUREMENTS.md 39차 §4f/§4g/§4j records:

- BF16 SP: +12.6% / +13.3% at 32K / 128K against BASE39-stock; adopted.
- KDA direct output + QK norm: +1.3% / +2.4%; adopted.
- NVFP4 dense with MIN_M=1024: +5.1% / +3.5% against BASE39-sp; adopted.
- MLA group2: its kernel took 1.85x stock time; group4 also regressed.
- MoE reuse now compiles but measured -1.9% / -2.0%; N128 measured
  -71% / -74%. The old prediction of a MoE gain is rejected.
- SP FP8 v2: -2.1% / +1.7% against REF41; neutral, still off.

These arms have different references/configurations. Do not multiply their
percentages into a claimed gain against `8d36ee8`. The 40% target is open.

The new candidate `VLLM_GLM53_PREFILL_SP_FP8=3` preserves v2's rank-local
block scales, FP8 values and ordered FP32 sum, but changes the wire layout
to `[destination][FP8 values | FP32 scales]`. The pack kernel writes this
layout directly and the unpack kernel reads it directly: no new copy or
interleave launch. v2 actually uses **two** `all_to_all_single` calls per
reduction; v3 uses **one**, with two compute kernels. Each v3 packet is padded
to a 128-byte stride (at most 120 additional bytes for this model's row width);
the final pack CTA initializes the gap without another launch. All-gather
uses the same aligned packet layout in v3; v1/v2 remain unchanged.
This removes v2's extra metadata collective. BF16 already uses one native
reduce-scatter, so a stable win against BF16 remains unproven. The later
operator promotion above enables v3 with the short-chunk BF16 gate.

NVFP4 scale construction also takes N/COUNT as non-specialized runtime
arguments. Prompts sharing a power-of-two final reduction capacity can reuse
the same kernels; the arithmetic and scale recipe are unchanged.

Validation commands (the GPU commands require a fleet turn):

```bash
python3 -m unittest discover -s tests -p 'test_glm53_prefill_collectives.py'
bash probes/run_glm53_prefill_transport_check.sh --compile-only
bash probes/run_glm53_prefill_transport_check.sh
# From srv2 with this checkout at the same path on all four hosts:
bash probes/run_glm53_prefill_transport_check.sh --tp4
```

The small GPU probe simulates four ranks on one device and checks every
packet byte, padding, output bits versus v2 and an ordered reference, and
one-call dispatch. It cannot prove actual NCCL behavior or serving quality.
The real TP4 probe now covers v2/v3 against the ordered decoded reference.

The first real TP4 run passed all 60 numerical cases but **rejected the
unaligned v3 transport on latency**. At T=8185, the median paired
reduce-scatter times were BF16 3871.696 us, v2 2480.416 us and v3
47169.312 us (five mirrored-order rounds, maximum across ranks per sample).
At T=128, whose packet stride was already aligned, v3 took 388.880 us
versus BF16 420.144 us. This motivated the 128-byte alignment correction.
Raw evidence: `srv2:/tmp/glm53-prefill-tp4.o9CdNc/rank-0.log`.

The aligned follow-up ran on 2026-09-07 00:59:38–01:00:23 KST under fleet
`prefill40align`. All 28 simulated-rank cases and all 60 real TP4 numerical
checks passed; every TP4 output matched the ordered transport reference
exactly. The same pinned image and helper SHA recorded in the ledger were
used on all four ranks. Five mirrored-order rounds gave these medians:

| Rows | Operation | BF16 us | v2 us | Aligned v3 us |
| --- | --- | ---: | ---: | ---: |
| 128 | all-gather | 505.104 | 446.960 | 427.808 |
| 128 | reduce-scatter | 427.664 | 418.528 | 338.480 |
| 129 | all-gather | 428.320 | 472.048 | 436.816 |
| 129 | reduce-scatter | 436.480 | 364.608 | 326.000 |
| 8185 | all-gather | 3263.872 | 5609.456 | 2258.656 |
| 8185 | reduce-scatter | 3965.456 | 2535.296 | 2079.712 |

At T=8185, v3 reduces collective latency by 30.8% for all-gather and 47.6%
for reduce-scatter versus BF16 (ratios of the displayed medians). T=129
all-gather remains slightly slower than BF16. These are isolated collective
results, not a model prefill throughput or quality verdict; the 40% target
and the default-off gate remain open. The unchanged NVFP4 checks were not
repeated. Logs: `srv2:/tmp/glm53-prefill-transport.N910dY/run.log` and
`srv2:/tmp/glm53-prefill-tp4.vg9y8E/rank-0.log`.

### Direct serving comparison

The serving build is source `5d44ef99284b47199f9c6064a52c9a67ed7a5ddf`,
overlay `67048ebe4790`, with the same pinned image as the TP4 probe. CPU
prerequisite `1f2ac74627e74966b7be` passed before GPU admission. Both modes use
SPEC_K=5 and the same 2K/32K/128K Korean onepass workload on independent boots.

The initial BF16 arm `SPV3S0907B1` (06:37 KST) measured 2955.886 tok/s at
32K (11.010 s TTFT), and 2968.183 tok/s at 128K (43.312 s). It passed
retrieval 9/9 and Korean corruption 0/5; decode median was 21.44 step/s.
The 2K cold/warm figures were 831.007 / 2423.427 tok/s and are separate
from the long-context comparison.

The legacy chain stalled after yielding because `busy_procs` counted its own
waiting `chain.sh` as legacy work. Only this yielded chain and its wait
children were stopped; the completed baseline was retained. The remaining
candidate/BF16 pair was experiment `754d6f2f4f1940d49b79`, using the standard
asynchronous pair path and the same CPU prerequisite. It completed at
07:00:31 KST; the fleet was released on BF16 at 07:00:32.

| Context | First BF16 tok/s | v3 tok/s | Pair BF16 tok/s | v3 vs pair BF16 |
| --- | ---: | ---: | ---: | ---: |
| 2K warm | 2423.427 | 2288.031 | 2519.934 | -9.20% |
| 32K | 2955.886 | 3021.035 | 2971.994 | +1.65% |
| 128K | 2968.183 | 3063.626 | 2963.025 | +3.40% |

The same pair's TTFTs were 10.951 → 10.773 s at 32K and 43.388 →
41.963 s at 128K. Against the first BF16 arm, v3 gained 2.20% / 3.22%
on the two long contexts, corroborating only a small improvement. All three
arms passed retrieval 9/9 and Korean corruption 0/5. v3's serving proof was
1/1. The 2K cold numbers are not a steady-state comparison: the first
source boot and the candidate's `cold_compile` flag differ.

Decode medians were 21.445 / 21.436 / 21.924 step/s in run order. The
generic pair judge evaluates decode and returned `incomplete` (-2.2%, one
compatible baseline and no noise floor). The initial manual baseline lacks
the pair's `runtime` field and is not silently injected into its automatic
baseline pool. This does not erase the measured prefill figures, and it
does not establish a stable overall performance win.

**Decision at 07:00 (before the later operator promotion): keep FP8 off.** The direct serving gain is
small on long contexts, warm 2K regresses, and the original 40% target is
not achieved. No extra baseline was run just to improve the verdict label.
Raw records: [three onepass records](GLM53_PREFILL_V3_SERVING_20260907.json).

### Short-chunk BF16 candidate

The follow-up v3 path uses native BF16 all-gather/reduce-scatter below
`VLLM_GLM53_PREFILL_SP_FP8_MIN_TOKENS=4096`. The gate uses the actual
scheduled chunk's real global rows before TP padding, including auxiliary
and final gathers, rather than the entire request or a rank's local rows.
It runs before FP8 scratch allocation or codec launches. At/above the
boundary v3 retains its aligned packed transport. `0` restores ungated v3
for experiments; modes 1/2 and the default FP8-off mode are unchanged.

4096 is an initial candidate between the observed 2128-token short request
and the 6912-token chunks used by long requests, not a measured optimum.
Native BF16 already uses one collective; v3's two-to-one reduction applies
only against v2. Its codec/temporary-buffer costs are the leading explanation
for the short-request regression, not a measured GPU time breakdown.

All three previous boots reported prefix-cache hit rate 0%. The onepass
"warm" value is the minimum of two post-first-run requests, not a cache-hit
measurement or a repeated-sample median. Future records now retain every
TTFT sample.

The follow-up ran 2K/4K/8K/32K/128K onepass requests on source `60a7de0`
(main `05c4a65`), overlay `7cb8c033a0b2`, and the same pinned image on all
four ranks. CPU prerequisite `62328ea3d97047cb9571` passed nine contracts.
The deployment gate passed 6656 available logic checks, 30 megakernel and
32 fleet regressions (torch-dependent CPU portions remain skipped on the
host); all 56 overlays were verified on all nodes. The asynchronous pair
submission rejected the stale pre-boot cache stamp after deployment, so the
documented `fleet.sh pair` path performed the boots and matched comparison.
No stamp was overwritten to bypass that check.

Candidate `SPGATE0907A` ran at 08:19:31 KST; BF16 `SPGATE0907ABASE` at
08:27:37. Both passed retrieval 15/15 and Korean corruption 0/11. The
candidate's short BF16 gather/scatter markers fired at the first real 2121
token request, and repeated requests used 2128 tokens. Its packed FP8 marker
also passed. Node attestations confirm mode 3 versus 0 with threshold 4096,
SPEC_K=5, identical image and manifest, and independent container boots.

**Short-request result:** BF16's post-first 2K TTFT samples were 846.921 /
843.019 ms; gated v3's were 844.359 / 881.140 ms. The harness's best sample
is 843.019 → 844.359 ms (2524.260 → 2520.256 tok/s, -0.16%). The two-sample
medians are 844.970 → 862.749 ms (+2.10% latency). The earlier 9.2% loss is
not reproduced, but two samples do not establish a stable percentage. This
2K phase had no observed prefix hits or concurrent requests.

**Long-context timing is not a clean comparison.** During baseline 8K and
later phases, a non-loopback client submitted requests. At 08:28:24 the log
shows two running requests and a mixed-batch metadata fallback; at 08:28:44
it shows a queued request, and at 08:29:24/34 a deferred request. Baseline
32K took 16.614 s versus candidate 10.793 s, but the apparent 53.9% gain is
not accepted. 8K/128K timing deltas are excluded from a speedup verdict too.
Warm 4K/8K requests also reuse prefixes, so they do not establish a pure
codec crossover. No additional traffic was sent outside the fleet hold.

Decision at 08:30 (before the later operator promotion): retain the short-chunk gate in the opt-in candidate, keep FP8 off
by default, and leave the optimal threshold and long-context benefit
unresolved. The original 40% target is not established. The pair released
the fleet on BF16 at 08:30:14. Raw records, node attestations and selected
traffic evidence: [gated serving report](GLM53_PREFILL_GATED_SERVING_20260907.json).
Remote logs: `srv2:/home/choiceoh/glm53-logs/fleet/experiments/754d6f2f4f1940d49b79/run.log`
and `boot-SPV3S0907B1.log`, `boot-EXP-754d6f2f4f1940d49b79.log`,
`boot-EXP-754d6f2f4f1940d49b79BASE.log` under `srv2:/home/choiceoh/glm53-logs/`.

The later historical trace
`dp0_pp0_tp0_dcp0_ep0_rank0.1788700171065354541.pt.trace.json.gz` also exposes
another candidate to investigate: its MHC pre kernels use local row counts
1728 (360 calls), 1152 (90), and 1150 (90). With 90 such calls per target
prefill forward, that is six forwards: four 6912-row chunks, one 4608-row
chunk and a final shard-padded 4600-row chunk. The sampler's nine calls also
include decode and are not a prefill denominator. Under APC's 2304-token
alignment, the configured 8192-token budget is not fully used. An aligned
batch budget is an observation, not a new performance candidate: the ledger's
APC3 arm already tried `MAX_BATCHED=9216` without recovering the cost.
Changing the budget again requires a demonstrated scheduler difference,
including speculative lookahead and memory; these counts predict no gain.

The operator requested continued implementation and **no tests until the
combined expected gain exceeds 40%**. After that threshold, validation is one
combined campaign. During the implementation phase, source inspection and
analysis of existing traces were allowed; tests, compiles, JIT warmup,
model invocations and benchmarks were deferred. First-phase validation recorded
under `fa1371e` predates this instruction and does not validate this bundle.

## Historical evidence for the forecast

The 2026-09-06 07:05 KST rank-0 trace is
`srv2:/home/choiceoh/vllm-prof/dp0_pp0_tp0_dcp0_ep0_rank0.1788645910055338719.pt.trace.json.gz`.
It was copied for read-only analysis to
`/tmp/glm53-prefill-40-evidence/baseline-32k.pt.trace.json.gz`.
The existing capture, not a fresh measurement, contains 13,793.551 ms of GPU
kernel durations. These profiler durations are forecasting weights, not the
matched baseline TTFT or a current-serving performance claim.

| Component | Historical GPU ms | Scope/correction |
| --- | ---: | --- |
| Dynamic MoE | 3,058.102 | 210 calls; 42 layers x 5 target forwards |
| BF16 NCCL all-reduce | 2,246.160 | 455 calls; includes calls outside eligible scopes |
| MK MLA | 1,733.598 | 55 calls; exclude mqa logits and top-k from MLA reuse gains |
| MHC post / pre / prenorm | 1,162.450 / 691.824 / 533.087 | Sum 2,387.361 ms; post includes auxiliary reconstruction |
| Eligible target FP8 dense GEMMs | 1,118.286 | Shape list below; exclude drafter FC and unclassified BF16/nvjet |
| Per-token FP8 quantization | 244.398 | 690 calls; **not** the entire roughly 1.1-second quant/norm/copy bucket |
| Q/K L2 norm | 85.730 | 340 calls; copy removal is a different cost |

Eligible dense `(N,K):ms` is `(6528,4096):520.317`,
`(4096,2048):174.210`, `(1024,4096):109.828`,
`(4096,4096):107.574`, `(4096,512):81.757`,
`(2048,4096):57.318`, `(6144,4096):45.743`,
`(4096,3072):21.539`.

The ledger's 10-step label cannot be used as the target-forward count:
170 KDA calls / 34 layers and 55 MLA calls / 11 layers both give **5 target
forwards**. Other model calls must be counted separately. Also, the older
NVFP4 line “2.30 -> 1.00 seconds, 7.7%” means a projected absolute saving
of 1.30/16.9, not a 7.7% eligible dense-GEMM share. Old MoE FLOP estimates
using the full I=2048 instead of per-rank I=512 overcount TP4 rank work by four.

## Implemented candidates and accounting rules

Every knob here arms only with the exact string **"1"** and stays off
otherwise, except the FP8 transport modes **2**/**3** and
`VLLM_GLM53_MK_MLA_PREFILL_GROUP`, whose default **2**
keeps the pair candidate's original arm and is inactive while the pair
knob is 0. An armed knob is not evidence of invocation -- since 39차 every
lane logs once whether it engaged (`[prefill-sp] MHC token shards selected`
/ `NOT selected: <gate>`, `[kda-prefill] direct output engaged`,
`[b12x prefill reuse] ENGAGED | NOT taken`, `[megakernel] mla prefill pair
engaged`, `[fp8-dense] nvfp4 prefill route engaged`).
`VLLM_GLM53_KDA_PREFILL_DIRECT_OUT`, `VLLM_GLM53_KDA_PREFILL_QK_NORM` and
`VLLM_GLM53_PREFILL_SP` (originally BF16 transport) are
**ON by default since 2026-09-06** (39차: KDA lanes +1.0 / +1.3 / +2.4 %
prefill at 2K / 32K / 128K; SP alone +13 / +12.6 / +13.3 % on the unified
onepass, decode and acceptance unchanged). Gated FP8 v3 became the transport
default on 2026-09-07 by operator request, with BF16 below 4096 real chunk
tokens. The measured state of the other
lanes is in MEASUREMENTS.md 39차 §4d-§4j: the MoE reuse / N128 compile failure
was fixed but neither improved long prefill, the MLA pair kernel runs 1.85x the
stock kernel's time (-10 % prefill even at group 2), the NVFP4 dense route's
GEMM saving (-520 ms) is eaten by its per-call quantization kernels and
launch glue (+850 ms), and historical FP8 v1 transport for SP was slower than
BF16 on this fabric. Note that
`VLLM_GLM53_PREFILL_NVFP4_BPROJ=1` is a no-op while
`VLLM_GLM53_FP8_DENSE_BPROJ=1` (this profile's default): the fp8 pattern
already owns every b_proj linear, and the candidate only adopts layers
still unquantized.

| Knob | Change | Forecast constraint |
| --- | --- | --- |
| `VLLM_GLM53_PREFILL_SP` | Row-shard MHC/residual on TP4; full-token attention/MLP; defer their terminal sum to reduce-scatter | At most 75% of eligible MHC work, before auxiliary/final gathers and smaller-shape efficiency |
| `VLLM_GLM53_PREFILL_SP_FP8` | FP8 transport; common block scale for reduction; packed data+scales in one all-gather | Count maxima, encode/decode, metadata, allocations and extra gathers; half the network bytes is not half the collective time |
| `VLLM_GLM53_MHC_PREFILL_POST_PRENORM` | Preserve BF16 post boundary, fuse post with prenorm input read | Apply only to the remaining MHC shard after SP; no double counting |
| `VLLM_GLM53_MK_MLA_PREFILL_PAIR` | Bounded multiset union; shared FP8 KV reads for adjacent queries | Only MK MLA time; overlap, extra math, hash and occupancy costs remain uncertain |
| `VLLM_GLM53_MK_MLA_PREFILL_GROUP=4` | Four-query reuse with two math subgroups and one FP8 ring | Requires pair knob; default group2 is inactive when pair=0; 512-thread register/spill cost is unverified |
| `VLLM_GLM53_B12X_PREFILL_REUSE` | Q0 staging/scan and task-local FC2 operand reuse | 128 -> 4 A/SFA load groups is not a 96.9% MoE speed claim |
| `VLLM_GLM53_B12X_PREFILL_FC1_N128` | Paired FC1 N128; includes Q0/FC2 reuse | Requested TMA bytes/task 5.875 -> 4.5 MiB; no extra weight or shared allocation; pipeline/cache/register effects remain |
| `VLLM_GLM53_FP8_DENSE_PREFILL_NVFP4` | Existing opt-in A4/W4 dense GEMMs | Prior large-shape kernel result roughly 2.3x; apply only to eligible GEMMs and charge quantization overhead |
| `VLLM_GLM53_NVFP4_SCALE_FUSED` | Two-launch activation maximum, global scale and alpha; no full abs temporary | Part of dense setup cost, not an independent full-bucket saving |
| `VLLM_GLM53_PREFILL_NVFP4_BPROJ` | Pure-prefill q_b/indexer.wq_b/eligible kv_b; BF16 decode and absorbed weights | Sparse MK MLA does not call kv_b GEMM; installed packs alone earn no forecast credit |
| `VLLM_GLM53_KDA_PREFILL_DIRECT_OUT` | Chunk kernel writes its final destination directly | Removes a copy, not chunk computation |
| `VLLM_GLM53_KDA_PREFILL_QK_NORM` | Stock reduction body reads strided Q/K directly | At T8192 removes 128 MiB copy traffic/layer if both inputs are noncontiguous; strided efficiency unmeasured; arming gate is bit-equality probe `probes/qk_norm_strided_check.py` |

The sequence-parallel gate accepts only eager, actual-token, pure-prefill TP4,
PP1/DP1, no EP/static-SP/context parallelism, and the reviewed terminal
reduction contracts. Interception is before the registered all-reduce custom
op in `GroupCoordinator`; returning an input alias inside that custom op
would violate its schema. Exactly one terminal reduction must be deferred
per attention/MLP call. Unexpected reductions fail loudly.

Historical target row counts are 8185 for four forwards and 4276 for the last
one, from MHC launch grids. Requiring T divisible by four would miss the main
workload. The candidate therefore pads only the local residual shards and
trims all attention/MLP/aux/final consumer gathers to actual T. Attention and
MoE metadata never see dummy rows. FP8 reduce-scatter creates its padding
during encoding; a full BF16 copy at every layer would cost about another
1.5 percentage points of baseline time. BF16 control mode pays that copy.

FP8 transport and NVFP4 dense paths change precision. They require output,
state and serving-quality evidence in addition to speed. The source recipe
for BF16 MHC rounding is retained; changed reduction schedules still require
numerical checks. Native FP8 NCCL support is documented by NVIDIA since
[NCCL 2.24](https://developer.nvidia.com/blog/networking-reliability-and-observability-at-scale-with-nccl-2-24/)
and exists in the pinned PyNCCL datatype mapping. This is source support,
not proof that this fleet's selected reduce-scatter algorithm executes it.

## Historical forecast gate (superseded) and combined validation

**Implementation forecast gate opened after source review, 2026-09-06. The
measurements above subsequently disproved this scenario.**
The explicit nominal scenario below gives **41.2056% throughput improvement**.
It is an unvalidated engineering forecast used to start testing, not achieved
performance or a statistical confidence estimate. In particular, MLA0.53 is
an optimistic conditional estimate and the MoE gain can be zero or negative.
The baseline total includes unaffected kernels and they retain their full cost.

| Component | Assumed candidate/baseline time | Saved historical ms | Basis and uncertainty |
| --- | ---: | ---: | --- |
| MHC | 0.27 | 1742.774 | 1/4 rows plus smaller-shape/launch allowance; actual T padding adds at most3 rows |
| Communication | 0.77 | 516.617 | Half network payload; about4.75V additional device traffic at225GB/s vs142Gb/s fabric, metadata/aux allowance |
| MLA group4 | 0.53 | 814.791 | **75% common selection assumed**: KV traffic0.4375, plus hash/math allowance; requires sufficient register/occupancy efficiency |
| Eligible dense | 0.50 | 559.143 | Earlier large-K GEMM roughly2.3x, discounted for activation scale/quant setup; only1118.286ms eligible |
| MoE N128 | 0.90 | 305.810 | 23.4% fewer requested TMA bytes, discounted for cache visibility/two-stage costs; no claimed measured gain |
| KDA copies | absolute86ms | 86.000 | Direct output and strided normalization; below removed-byte roof, layout performance still unknown |

Total saved **4025.135ms / 13793.551ms = 29.1813% time**, hence
`1 / (1 - 0.291813) - 1 = 41.2056%` throughput. Neither BPROJ nor MHC
post/prenorm fusion earns extra credit in this scenario; the primary combined
arm leaves those two optional experiments off. Existing warmup and decode
optimizations also earn no credit.

The historical MoE launch records168 registers/thread,1 CTA/SM and101376B
shared memory. The N128 source estimate adds roughly56 live registers against
a248-register math-warp allowance, which makes the scenario plausible but is
not spill proof. MLA group4 instead has a tighter512-thread register budget;
that must be checked during compilation. At75% common selection, MLA0.53
allows about2.923ms for additional costs per old31.6ms attention. If common
selection is only50% and its ratio becomes0.72, the total forecast falls to
about36.6%; if MoE also gives no gain it falls to about32.6%. These cases
must be reported as misses, not hidden by a favorable isolated kernel result.

Once the gate opens, run the combined static/source-parity and numerical
checks, then matched baseline/candidate requests in a free fleet window.
Keep actual imported source hashes, startup/runtime lane fingerprints,
input lengths, cache state, image identity, and ownership receipts. Include
BF16 collective equivalence; FP8 scales/headroom/reduction behavior; KDA
outputs/state and padding; MLA duplicates, tails and weak-overlap fallback;
MHC BF16 rounding; MoE intermediate/output numerics and spill/occupancy.
Use the existing `bench/onepass.py` ladder for 2K/32K/128K throughput,
retrieval, Korean output, decode and speculative acceptance. Do not claim a
successful runtime result from stock fallback, a compile, or an isolated
kernel benchmark. Keep cold/warm and quality outcomes separate.

Other sessions' running service/bench ownership is not a usable GPU window.
Do not restart or evict them. No default is promoted without matched runtime
proof. If the first combined validation misses 1.40x or regresses quality,
the target remains open and failed assumptions are removed from the forecast.
