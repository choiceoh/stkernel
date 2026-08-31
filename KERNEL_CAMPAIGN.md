# Kernel tuning campaign — GB10 (sm_121a, 48 SMs)

Status: **CAMPAIGN CLOSED (2026-08-11). P2a (b12x MoE) DONE — no win.
Mystery-wmma DONE — router-gate fused kernel ADOPTED (+3.5%). P2b DONE —
mhc small-M tile (256,6,4) ADOPTED (decode −0.22ms/step, bracket +0.57%);
tf32 prenorm n_splits REJECTED (stock n_splits=1 optimal, splits strictly
lose).** Clock lever vetoed (cooling). Remaining kernel work requires
source surgery on image-owned kernels (quant armada) — no overlay-ownable
targets left.
Goal: attack the prefill MFU gap (~25-30% of blended fp8/W4A16 ceiling) at the
kernel level. Decode is OFF the table — its dominant kernels measure at the
273 GB/s LPDDR floor (ledger), where no kernel can win; only byte-diet applies.

## GB10 architecture dossier (2026-08-10 — local evidence + Chips&Cheese + backend.ai)

**What it is**: a consumer Blackwell die of the RTX 5070 class (GB205 lineage,
48 SMs) integrated on a TSMC N3 2.5D package with a Mediatek S-dielet (CPU +
LPDDR5X-8533 256-bit controllers, 273 GB/s unified). The GPU reaches memory
over the package fabric — nvidia-smi's "PCIe Gen1 x1" is a stub.

**ISA (cc 12.1 / sm_121a)** — Ampere-lineage `mma.sync` EXTENDED with
FP8/FP6/FP4. Missing vs datacenter sm_100: `tcgen05`, TMEM (256KB/SM),
WGMMA (sm_90), thread-block clusters / DSMEM multicast ("each SM executes
independently"). Present: TMA (deep_gemm sm120 impls use CUtensorMap loads).
L1/smem 128 KB/SM (vs 256 datacenter), L2 24 MB (~1 TB/s write), FP64 1:64.
Local device: 48 SMs @ cap 2000 MHz (default APPLICATION clock 2418, C&C
measured max 2.55 GHz), 121.6 GB visible unified.

**This explains the ledger's dead ends**: FlashMLA/FA4/TRTLLM (need
WGMMA/tcgen05), swiglu-clamp SM100 gating, and the `cutlass_80_wmma`
sightings — same-generation programming model, NOT an accidental fallback;
the optimization question is shape/wave tuning, not instruction upgrades.

**Campaign implications**:
1. **Wave quantization for 48 SMs** is the concrete deep_gemm angle: sm120
   impls were plausibly tuned on RTX 5090 (170 SMs); identical tile configs
   leave large last-wave idle at 48 SMs. Sweep tile sizes to align tile
   counts to multiples of 48.
2. **smem stage count** under the 128 KB budget (current traces show 4
   stages) — sweep 2..6.
3. ~~clock lever~~ — **VETOED by operator (2026-08-10): cooling limits mean
   raising past 2000 only trades stability. Do NOT re-propose.** Cap stays
   2000 (gpu-clock-cap.service).

## P1 findings

Prefill kernel budget (87MB rank0 trace, morning capture; shares rescaled to
the 200G fabric era where AllReduce dropped 23.2% → ~9%):

| target | share | source form | tunability |
|---|---|---|---|
| b12x `W4A16FusedMoeKernel` | **~25%** | CUTLASS-generated, `b12x` pkg | unknown — check if configs are Python-side |
| `mhc_post_tilelang_kernel` + `mhc_pre_big_fuse_with_norm` | ~12% | **tilelang Python DSL** (`vllm/model_executor/kernels/mhc/tilelang.py`, 726 lines) | direct — tile/stage params in source |
| `sparse_mla_prefill_mg_dual_kernel<...,16,128,64,64,1>` | ~11% | CUDA template | hard |
| `sm120_fp8_fp4_gemm_1d1d_impl<...,128,128,...,64,64,64,128,4,...>` family | ~10% | deep_gemm .so heuristics; template args visible in traces | medium — instantiation choice buried in C++ |
| `sm120_tf32_hc_prenorm_gemm_impl<24,16384,128,32,64,1,4,256,128>` | ~4% | .so kernel + Python call-site knobs (`block_m=64, block_k=64, n_splits` in mhc/tilelang.py:170) | easy to sweep call-site knobs |

Key facts:
- `deep_gemm.get_num_sms/set_num_sms` exist; only caller is the (inactive)
  ubatch wrapper → default = device-detected 48. "Wrong SM count" hypothesis: weak.
- GB10 = 48 SMs, sm_121 (12,1). deep_gemm ships sm90/sm100/sm120 variants of
  hc_prenorm; we run the sm120 one on sm_121a.
- All hc weights are fp32 by design (hc_head_fn etc.) — the tf32 GEMM is a
  *precision choice*; a bf16 variant is a numerics experiment, not free.
- Prior: the fork author developed ON GB10 — tilelang kernels are likely
  already hand-tuned. The generic libraries (deep_gemm heuristics, CUTLASS
  b12x configs) tuned for datacenter parts are the better-odds targets.

## P2 design (engine-down window, ~30-60 min; cloud fallback covers lightweight roles)

1. Stop dsv4-tp4 (fallbackModel=wormhole/deepseek-v4-flash-api takes over).
2. Fresh container on srv2 (NEVER docker-exec CUDA in the serving container —
   see the 08-09 collective-stall incident).
3. Microbench harness per target, real shapes from the trace:
   - mhc tilelang: sweep block/threads/stages around current values; bit-exact
     or ≤1e-6 rel gate vs current kernel.
   - tf32_hc_prenorm: sweep call-site `block_m/block_k/n_splits`; then a
     bf16-weights variant behind a numerics gate (rel err + downstream mix).
   - b12x MoE: first READ the package to find whether tile configs are
     Python-visible; if compiled-only, drop the target.
4. Winners → overlay bake → engine restart → bracket A/B (prefill ×3 + quality
   9/9 + bench-dec 3) per the ledger discipline.

Expected value, honest: +5-10% prefill best case, 0% plausible. Decode: none.

## P2a result — b12x W4A16 MoE tile sweep (2026-08-10, engine-down window)

Harness: fresh container (serving-container CUDA ban respected), real TP4
shapes (E=64/rank, H=4096, I_tp=512, topk=6, M=4096/2048 route-block 64),
random-bit MXFP4 weights (identical object across candidates — timing-valid),
force_tile_config native hook. Contract learnings (recorded for reuse):

- `force_tile_config = (fc1_tile_k, fc1_tile_n, fc2_tile_k, fc2_tile_n)`;
  threads = n*k/64, must match across FC1/FC2; only 128/256 CTA threads.
- compile `max_m_blocks` = WORST-CASE ROUTE BLOCKS (M*topk/64 + E), not
  cta_m_blocks — run_w4a16_moe re-derives and rejects smaller.
- `_W4A16_REGS_SM121` gates unknown shapes → setdefault(255) unblocks
  (conservative occupancy).
- Most tile shapes die on the 99KB smem fits-check at 4 stages: the shipped
  candidate tables ARE essentially the feasible set. K._STAGES patchable.

| config | M=4096 | M=2048 |
|---|---|---|
| heuristic (production) | 8.34 ms | 4.38 ms |
| best forced (s4 t128 pair) | 8.25 ms (+1.1%) | 4.33 ms (+1.1%) |
| stages=3 family | −0.1..−1.0% | −1.6..−2.5% |
| stages=2 family | −6..−10% | (worse) |

**Verdict: the author's heuristic sits within ~1% of the best config in the
entire feasible space at both M regimes — the 25%-share target is CLOSED.**
The +1.1% best-forced is microbench-noise-grade (= +0.27% prefill end-to-end,
below adoption bar). Stage reduction strictly loses. The numerics-mismatch
flags on some cells are an artifact of random-bit weights + cross-tile
accumulation order, not a correctness signal (real gate = engine 9/9).

### ③ Mystery bf16-out wmma — RESOLVED into three populations (2026-08-10)

Grid-level decomposition of the ~2.8 ms/step bf16 wmma family (steady-state,
topk-confirm trace):

| population | /step | per-call | ms/step | identity |
|---|---|---|---|---|
| grid=[8,253,1] | 1.0 | 1,144 µs | 1.16 | **TARGET lm_head verification logits** — 253=⌈129,280/512⌉; bf16 shard 265MB/rank streamed EVERY step (265MB/273GB/s=970µs ✓). The draft lm_head got fp8 (maybe_build_fp8_lm_head) but the TARGET's never did |
| grid=[8,2,8] | 46.5 | 25 µs | 1.16 | ~~unresolved~~ **RESOLVED 08-10: MoE router gate** (43 layers + 3 mtp, bf16 [256,4096] Tier-4 F.linear; +splitKreduce+fp32-cast = 1.71 ms/step chain) → fused triton gate ADOPTED, **C=1 +3.5%** (ledger) |
| grid=[4,1,1] | 6.1 | 77 µs | 0.47 | **RESOLVED 08-10: indexer.weights_proj** on the 6 F-layers/step (21 C4A / freq4), aux-stream contention-stretched; real bytes 3MB/step → no action |

**New candidate: target-lm_head fp8 → decode +2.7%** (halve the 1.16ms).
CAUTION unlike every adopted byte-diet so far: verification logits decide
accept/reject AND the sampled token — OUTSIDE the rejection-sampling
losslessness that protected the markov/draft-head fp8 moves. Adoption needs a
greedy-divergence gate (bracket boots, N fixed prompts, exact/near-exact
match) + the standard 9/9. Precedent for the mechanism: dspark_v2's
_quantize_fp8_deepgemm/_fp8_gemm pair, wired at the logits_processor for the
verify path.

### ① Acceptance axis — knob triage (2026-08-10)

- CONFIDENCE_SCHEDULER=threshold: structurally ~0 for C=1 (the fixed draft
  block still runs; only EMITTED tokens get truncated → same step time, same
  accepted length). Batched-verify benefit only. Deprioritized.
- GREEDY_DRAFT: already measured/rejected in the ledger (probabilistic won).
- **MARKOV_SCALE (default 1.0): the live lever** — scales the markov bias on
  draft logits; changes draft q ONLY → rejection sampling keeps the target
  distribution intact = quality-lossless BY CONSTRUCTION; acceptance is the
  sole metric. Sweep 1.3/0.7/1.6 running.
- EAGLE3.1 alternative drafter checkpoint exists on disk — larger project,
  parked.

Remaining P2b: mhc tilelang pair (~12% prefill) + tf32_hc_prenorm call-site
knobs (~4%) — harness pattern above transfers; expected value modest (±0.4%
prefill scale).

## Guardrails

- One knob per relaunch; bracket (base→cand→base) for anything kernel-level.
- Quality gate: check-quality 9/9 minimum; numerics gates offline first.
- Ledger every cell including rejections.

## Decode campaign II — kernel count & dense byte diet (2026-08-31, REOPENED)

The 08-11 closure ("no overlay-ownable targets left") is obsolete. The GLM-5.3
DFlash2 bring-up produced a full decode census (srv2 rank-0 trace, 264 steps,
2,160 kernels/step) and six owned axes; five landed as PRs. Every number below
is trace- or /metrics-derived unless a PR is linked.

Step model (clean 5-point bench): `step = 42.0 ms fixed + 0.706 ms × distinct
experts`. The variable part measures at 91% of bandwidth — closed. The fixed
part splits into dense-weight reads (~17 ms — the byte diet below) and
kernel-count overhead (~32 ms at the 15.4 µs/kernel *upper bound*; the real
constant is unmeasured until the pending boot).

### Axis-by-axis disposition

| axis | /step | disposition |
|---|---|---|
| osar AR 5-kernel chain | 522 | #89 (5→3), #90 (3→1, reset-free monotonic counter), #93 (idle-block spin/fence skip, pollers 256→16) → ~104 |
| dense bf16 projections | 15.44 GB reads ≈ 17 ms | #92/#94 fp8 W8A8 blocks (dsv4's proven scheme, ue8m0/[128,128]), #95 extends to the drafter — −8.7 −1.2 ms expected |
| gate double launch | 84→42 | #96: the MoE runner re-applies the gate it holds (overlapped with the shared-expert aux stream); our explicit call was pure waste |
| elementwise 3종 | 441.6 | drafter conv already inductor-fused; eager glue 98/step ≈ 0.85 ms GPU (closed); the rest is the KDA cluster below |
| **KDA/MLA per-layer index/mask** | 179 (measured subset) | **CLOSED 2026-08-31 with evidence**: the MLA indexer glue lives in `vllm/models/glm5next/nvidia/attention.py` under `@torch.compile` — the fill/cat/pad arithmetic is ALREADY inductor-fused (that is what the `triton_poi/red_fused` kernels in MLA blocks are), and the remaining eager per-layer ops are numerics-load-bearing by documentation (the fp32 head-gate: bf16 flips pool rankings, HLE drops). No overlay-level reduction exists; a takeover would add a 624-line contract for zero gain. The residual int cluster is drafter-side (image base class) — same closure |
| MoE output write-back | ~100 | two copies/layer (flashinfer `memcpy32_post` out of the static workspace + our `output.copy_`) — contract-bound by the static-buffer+alias design; removing needs a flashinfer `out=` redesign |
| mhc 2종 | 179 | structural (attention sits between pre and post_pre) + tiles exhausted R1–R3 — closed |
| cutlass_80_wmma | 354.7 | bandwidth floor; the fp8 byte diet is the lever, not the kernel (split-K at M=1–8 is reasonable) |
| shared-expert overlap | — | already `MULTI_STREAM_OVERLAPPED` in-image (aux stream, threshold 256 tokens — verified) — closed |
| osar wait phase | P99 89.0 of 101.5 µs/collective (88%) | **CLOSED 2026-08-31, measured**: #100's backoff cut the spin 44% (89.0→50.0 µs) and end-to-end bought **+0.1%** — the wait is network-latency-bound (the release is the flag's arrival, not the spin's cost) and already overlapped (shared-expert aux stream + #93's idle-block skip). The AR axis has no kernel-level headroom left; only structural levers remain (fewer collectives via enable_sp, deeper async) |
| gate+topk true fusion | −42 | deferred with reasoning: a single-CTA full-width design makes one CTA read the 2.4 MB gate weight (~2× slower GEMM) to save one launch |

### Acceptance — the multiplier kernel work cannot touch

tok/s = tokens/step ÷ step time, and the whole 2× gap to dsv4 is tokens/step:
GLM 1.538 (pos-1 63.1%) vs dsv4 2.55 (78.5%) at 47.6 vs 44.5 ms. The "raw
21%" alarm was a unit artifact — a 63%-accurate drafter at k=7 necessarily
lands at 22% accepted/drafted. Pre-registered 2-boot ladder lives in
profiles/glm53.env (SPEC and TARGET head knobs are SEPARATE and both =1;
dsv4 hit 78.5% with both heads fp8, so precision is the hypothesis to retire).

### Selector GEMV (this change)

The drafter's candidate scoring is the only fp32 cuBLAS GEMV in the trace
(`gemmSN_NN`, 10.9/step, ~330 µs real) — dtype promotion somewhere in
`_score_edges`. `VLLM_DFLASH2_SELECTOR_BF16` forces the operands to bf16
(half the read). It scores candidate RANKING: near-ties may flip, acceptance
is the gate, default off.

### Verification debt (the boot that scores everything)

#89–#96 are trace/code-derived. One boot measures them all AND calibrates
µs/kernel (−408 osar kernels = −6.3 ms at 15.4 µs, −0.8 ms at 2 µs), which
decides the KDA takeover. Boot fingerprints (PR #91): `[osar] kernels=N`,
`[fp8-dense] N linears …`.
