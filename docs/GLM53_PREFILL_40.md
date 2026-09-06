# GLM-5.3-Flash prefill: 40% throughput campaign

Baseline source: `8d36ee8`. Worktree: `work/glm53-flash-prefill-20260906`.
The earlier direct-output candidate is `fa1371e`.

The target is at least **1.40x input tokens/second** on matched requests,
equivalent to reducing prefill wall time by at least **28.5714%**. It is not
a 40% reduction in wall time. Cold and warmed measurements are separate.

The operator requested continued implementation and **no tests until the
combined expected gain exceeds 40%**. After that threshold, validation is one
combined campaign. During the implementation phase, source inspection and
analysis of existing traces are allowed; no new test, compile, JIT warmup,
model invocation or benchmark is being run. First-phase validation recorded
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

All knobs remain default **0**. An armed knob is not evidence of invocation.

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
| `VLLM_GLM53_KDA_PREFILL_QK_NORM` | Stock reduction body reads strided Q/K directly | At T8192 removes 128 MiB copy traffic/layer if both inputs are noncontiguous; strided efficiency unmeasured |

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

## Forecast gate and combined validation

**Implementation forecast gate opened after source review, 2026-09-06.**
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
