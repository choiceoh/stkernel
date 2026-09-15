# C=2 routed experts: tile-major w13 goes from 512 to 256 chunks (2026-09-15)

Operator order (2026-09-15 11:3x): "MoE 전문가 27.6 ms (46%)... c=2 이경로 더 깎고".

**Adopted.** The served bind now lays each layer's w13 out in 256-wide chunks (`moe_dispatch._W13_TILE_CHUNK = 256`).
It was 512. Against the same-build 512 control, on real rank-3 weights and each layer's real router:

- **Exact.** Every decode, short-prefill and prefill cell sits at the kernel's own add-order floor. No BF16 model-boundary
  value differs in any decode cell.
- **C=2 decode.** Two requests (U≈80 experts per layer), three-layer chain, evicted: **−1.61%** (brackets −1.12, −1.50,
  −2.76, −1.04). Warm: −0.68%. Sixteen independent rows: −0.88% evicted, −0.39% warm.
- **C=1 decode.** Chain evicted −0.95%, warm −1.14%.
- **Prefill.** Eager, evicted, one layer: m=2304 −1.88%, m=9216 −2.44%.
- **128 chunks.** Rejected: they misread the reform tile.

A view that names `TILED_W13_K_IN` (512) is the control. No knob was added.

## Where a C=2 step's expert time goes

A stamped kernel ran on layer 3 (ticket `c2moe-cells2-0915`, served 512 chunk). Occupancy was calibrated so that one
8-row request reads the fleet's measured 41.9 distinct experts per layer (`../st_draft_rank_overlap_20260915`).
C=2 is two independent such requests.

| Tile | U | Span µs | Frontend (phase 0 + barrier / routing, quantization + barrier) | FC1 + quant per item | FC2 per item | Items per CTA | Idle tail |
|---|---:|---:|---|---:|---:|---|---:|
| C=2 served | 80 | 1273.7 | 19.7 (3.5 / 16.2) | 121.7 | 51.5 | 6..7 | 11.0 |
| C=1 served | 36 | 610.9 | 13.6 (2.8 / 10.8) | 119.0 | 54.4 | 3 | 5.4 |

- **Items cost the same at both widths.** C=2's extra expert time is its extra items: two requests share few experts,
  so U roughly doubles (320 items over 48 CTAs against 144). The only row-dependent cost is routing and input
  quantization, +5 µs per layer.
- **Streaming is the time.** A CTA streams an item at about 5.0 GB/s for FC1 (610 KB) and 5.5 GB/s for FC2 (287 KB).
  48 CTAs together put about 215–220 GB/s of expert bytes through (the `rate` records).
- **Time follows cold bytes.** The probe-only timing cells price two of an item's boxes (ticket `c2moe-chunk0915`, C=2,
  U=79):

  | Skipped box | Share of item bytes | Single evicted | Chain evicted | Chain warm |
  |---|---:|---:|---:|---:|
  | FC1 SF6 scales (`xs`) | 5.5% | −7.54% | −6.01% | −6.19% |
  | FC1 A + SFA (`xa`) | 4% | −1.88% | −2.49% | −1.77% |

  The scale blocks are cold. The activation input was written by the frontend moments earlier.

## The chunk lever

The served bind lays w13 out tile-major (spec cell `t`, 39차). With a 512 chunk the layout is
`[E][K/512 k tiles][rows][256 B]`, so the `t` tile's (64 rows × K512) FC1 box is one contiguous 16 KB run.

Two served kernels read that storage with narrower K tiles:

- the M16 decode reform (C=1 since 09-09, C=2 since #974) stages FC1 as 128 rows × **K256**;
- the gated prefill kernels stage FC1 as 64 rows × **K128**.

Each row then gives part of its 256 B chunk. Their TMA boxes are 128 B (64 B) runs with the rest of the chunk between
rows, the segment class 35차 §6 measured below a linear read.

With a 256 chunk:

- the reform box is one contiguous 16 KB run;
- the prefill box is half a chunk per row;
- the `t` tile's K512 box reads two adjacent chunks.

`_WeightViews.w13_chunk` names the chunk. The relayout, the in-place marker (`plain` = 512, `plain256`), the static and
dynamic compile fakes and their cache keys follow it. The 512 chunk keeps its original keys and names.
`expert_layout.w13_chunk_bytes` reads the chunk back from the marker for the reference lane and the dense dataflow
oracle. The kernels and their arithmetic are unchanged: only the 4-D weight tensor they compile against changes.

### Numerics: the add-order floor

The first ticket stopped on a 1.19e-07 difference, and the cause is not the chunk. The kernel adds route partials to its
FP32 accumulator atomically in an order it does not specify, so the same handle replayed twice differs in the last ulp
of a few sums.

Every cell therefore also replays a second copy of the control:

- **Reported.** The count of differing FP32 elements and their largest distance in ulps, and the count of differing BF16
  model-boundary values (`BF16(FP32 scatter)`, what the finalizer consumes).
- **Gate.** 64 ulps of the larger magnitude or of the tensor RMS. An add order stays within a few ulps; wrong bytes move a
  value by its whole scale.
- **Stimulus.** Each cell is three input/route draws in two replay orders, with poisoned accumulators and a zero route.

| Fixture | Scope | Elements | 512 vs 512 (FP32 differing, max) | 256 vs 512 | BF16 differing (both) |
|---|---|---:|---|---|---:|
| C=2 two requests | single | 393,216 | 20, 1 ulp | 15, 1 ulp | 0 |
| C=2 two requests | chain | 1,179,648 | 25, 1 ulp | 17, 1 ulp | 0 |
| C=2 independent | single | 393,216 | 20, 1 ulp | 18, 1 ulp | 0 |
| C=2 independent | chain | 1,179,648 | 34, 1 ulp | 31, 1 ulp | 0 |
| C=1 one request | single | 196,608 | 9, 1 ulp | 6, 1 ulp | 0 |
| C=1 one request | chain | 589,824 | 6, 1 ulp | 3, 0.5 ulp | 0 |
| short prefill `t` tile, 32 rows, independent | chain | 2,359,296 | 69, 1 ulp | 75, 1 ulp | 0 |
| short prefill `t` tile, 32 rows, one request | chain | 2,359,296 | 177, 2 ulps | 174, 2 ulps | 0 |
| short prefill `t` tile, 12 rows, independent | chain | 884,736 | 25, 1 ulp | 28, 1 ulp | 0 |
| short prefill `t` tile, 12 rows, one request | chain | 884,736 | 41, 1 ulp | 45, 1 ulp | 0 |

- **Prefill m=2304** (Q0 words, U=261, FP32 sum): 18.9 M outputs per comparison, 0 BF16 values differ in either pair.
- **Prefill m=9216** (long SF6 words, BF16 sums): the control differs from itself in 4.13 M of 75.5 M outputs, and 256
  from 512 in 4.30 M. Both have max 0.0234 (3.6% relative), an order-of-addition spread the long prefill already has.

### Timing (ticket `c2moe-cells3-0915`)

Each row is 256 against 512, four B/A/A/B brackets. Warm is 64 replays per sample; evicted is 32 replays with a 128 MiB
flush before each, outside the events. µs are per replay; a chain replays three layers (L3, L4, L5).

| Fixture | U (single / chain) | Scope | Cache | 512 µs | 256 µs | Mean | Min | Brackets |
|---|---|---|---|---:|---:|---:|---:|---|
| C=2 two requests | 79 / 79,79,83 | chain | **evicted** | 3827.7 | 3766.1 | **−1.61%** | −1.05% | −1.12 −1.50 −2.76 −1.04 |
| same | | chain | warm | 3754.7 | 3729.2 | −0.68% | −1.09% | +0.11 −1.05 −0.85 −0.93 |
| same | | single | evicted | 1292.3 | 1275.0 | −1.34% | −1.14% | −1.53 −1.18 −1.21 −1.46 |
| same | | single | warm | 1240.4 | 1231.8 | −0.69% | −0.96% | +0.28 −0.81 −1.33 −0.93 |
| C=2 independent | 93 / 93,85,98 | chain | **evicted** | 4369.5 | 4331.0 | **−0.88%** | −1.00% | −1.04 −0.99 −0.33 −1.16 |
| same | | chain | warm | 4332.8 | 4315.7 | −0.39% | −0.85% | −1.12 −0.34 −0.92 +0.81 |
| same | | single | evicted | 1499.9 | 1483.4 | −1.10% | −1.11% | −0.83 −1.25 −0.97 −1.36 |
| same | | single | warm | 1444.3 | 1432.2 | −0.84% | −0.96% | −0.87 −0.43 −1.00 −1.04 |
| C=1 one request | 49 / 49,41,43 | chain | **evicted** | 2152.1 | 2131.6 | **−0.95%** | −1.04% | −1.02 −1.01 −0.84 −0.94 |
| same | | chain | warm | 2102.3 | 2078.3 | −1.14% | −0.96% | −1.08 −0.53 −2.02 −0.92 |
| same | | single | evicted | 851.0 | 850.6 | −0.05% | −1.08% | −1.15 +3.64 −1.36 −1.36 |
| same | | single | warm | 783.9 | 783.7 | −0.02% | −0.76% | +0.30 −0.04 +0.35 −0.69 |

- **Rates.** Expert bytes through the chain, evicted: 216.8 → 220.4 GB/s (C=2 two requests), 217.5 → 219.5 GB/s
  (independent), 212.8 → 214.9 GB/s (C=1).
- **Stamped per-item view** (`c2moe-cells2-0915`, C=2, U=80, stamped handles): 512 / 256 give FC1 + quant 121.2 / 120.5 µs,
  FC2 51.8 / 52.2 µs, span 1271.9 / 1263.8 µs.
- **Prefill** (eager, the host in the timing, a flush before each call, four brackets):

  | m | 512 µs | 256 µs | Mean | Min | Brackets |
  |---:|---:|---:|---:|---:|---|
  | 2304 | 10367.8 | 10172.5 | −1.88% | −1.69% | −3.01 −2.72 −0.80 −0.97 |
  | 9216 | 16215.8 | 15820.9 | −2.44% | −2.13% | −2.12 −2.39 −3.19 −2.04 |

  The first ticket's two-bracket run agreed at m=2304 (−1.87%).

- **Decision.** Judged on the chain with L2 evicted, 256 beats 512 beyond the ~±0.5% floor at C=2 (both fixtures) and at
  C=1. The worst warm or single cell is the C=1 single layer at −0.02% / −0.05%, which is flat, not a regression. It is
  adopted.

Size of the win: the served chain evicted is 1,276 µs per layer at U≈80. −1.61% is about 20 µs per MoE layer, about
0.9 ms per C=2 step over 42 MoE layers at that occupancy. The 09-15 profile window measured 27.6 ms of expert time (U≈47
per call), where the same share is about 0.45 ms.

## Rejected or neutral, same tickets

- **128 chunks misread the reform tile.** C=2 two requests, single layer (`c2moe-cells2-0915`):

  | Arm vs 512 | FP32 differing (of 393,216) | Max | BF16 differing |
  |---|---:|---:|---:|
  | 512 again | 17 | 1 ulp | 0 |
  | 256 | 15 | 1 ulp | 0 |
  | 128 | **393,216** | **133.0 (3.3e7 ulps)** | **393,180** |

  - **Why.** The reform's K256 box over 128 chunks spans two chunks, and the TMA read wrong bytes: no fault, wrong values.
  - **Not a general limit.** The `t` tile's K512 box over two 256 chunks is exact (table above), and so is the prefill
    K128 box over one 128 chunk (m=2304: 0 differing; m=9216 at the control's spread).
  - **Consequence.** 128 is removed from `TILED_W13_CHUNKS`, and the reason is recorded there. Prefill alone would have
    liked it (m=2304 −3.73%, m=9216 −1.56%), but decode cannot read it.
- **FC2 prefetch ring depth** (#962's third slot, against two slots; exact).
  - **Timing.** C=2 two requests: single evicted +0.09% (brackets +0.05 +0.26 +0.29 −0.25), chain evicted −0.34%
    (−0.20 −0.41 −0.83 +0.06), chain warm −1.00% (min −0.29%). The single warm mean of −11.44% is one bracket with a slow
    control sample (brackets −0.30 −5.32 −30.68 −0.32; min −0.28%).
  - **Stamps.** The third slot only moves bandwidth between the two phases: FC1 118.5 → 121.7 µs, FC2 55.7 → 51.5 µs,
    span 1268.5 → 1273.7 µs. The bus sets the rate, not the ring.
  - **Verdict.** Neutral. Left as served.
- **The M16 reform tile for every static row count** (probe config `reform_every_static`; not served).
  - **Timing.** Against the served `t` tile over 512 at short-prefill widths, chain evicted: 32 rows independent −1.71%
    over 512 and −2.59% over 256; 32 rows one request −0.33% / −1.22%; 12 rows independent −1.82% / −2.72%; 12 rows one
    request −2.16% / −2.92%. All cells are exact.
  - **Not adopted.** 45차's capacity run rejected wider reform tiles for experts reused by many rows (M21/M28 with eight
    reused experts: +9.2–20.1%), because such an expert takes two M16 tiles. This fixture did not reach that case. The
    numbers are kept for that question.

## Method

- **Probe.** `probes/engine_moe_c2_cells.py`, lane `engine_kernel_check --lanes moe_c2_cells[:sections][:chunks=...]`.
  - Rank file: `/home/choiceoh/models/st-glm53-9391-up-gate-full/rank3of4.safetensors` (production's rank, folded
    scales), MoE layers L3–L5. Sample sha256s are in each `weights` record.
  - The served lane is `lanes.served(moe_static='t,r,sf6,batch,q0')`. Its `moe_prepare` lays the arena out at the served
    chunk (512 in the tickets, which ran before this change); other chunks are copies of the same row-major bytes.
  - Calls go through `b12x_fused_moe` with the served finalizer contract, where the FP32 accumulator is copied out.
    Collectives, shared experts and the output cast/add are outside the timing.
- **Occupancy.** Each layer's real router over grouped synthetic rows: a base per request plus 1.2499 × independent noise.
  The spread was bisected so an 8-row request reads 41.92 distinct experts, averaged over the three layers and four seeds.
  Two requests then read 76.0 and sixteen independent rows 96.6.
- **Exactness before timing.** Every cell replays both arms plus a second control copy, as described above.
- **Timing.** Four B/A/A/B brackets per cell (`--samples 4`). Captured graphs; events inside; the 128 MiB flush outside.
  Prefill is timed eagerly with the host included.

| Ticket | Frozen source | GO – release (KST) | Sections |
|---|---|---|---|
| `c2moe-chunk0915` | `823af35e` | 12:07:39 – 12:09:55 | chunk (stopped at the first 1-ulp difference: zero tolerance), stamps, prefill, price |
| `c2moe-cells2-0915` | `39516cdf` (main `3acae017`) | 12:47:54 – 12:50:29 | chunk (stopped at 128), depth, stamps, prefill |
| `c2moe-cells3-0915` | `a13b361a` (main `34751523`) | 13:57:02 – 14:00:07 | chunk (512, 256), shapes |

All three ran on srv4's GB10 through the single-GPU lane, with image
`sha256:b45454b5fdc138bfaafa6470cf9cb49bbcd74783ceffb343b0a7b1d0cf87fcd5` (Torch 2.13.0+cu132, CUDA 13.2), no model boot
and a 12 GiB budget. Peak allocation was 7.65 / 8.17 / 5.55 GiB. The lane admits one probe at a time; what else srv4
ran is not recorded here (the second ticket ran beside restored production).

## CPU gates on the adopted tree

- **Native compile.** `native-compile.json`, on the same merged tree: CuTe lowering, PTXAS and TVM-FFI with
  `CUDA_VISIBLE_DEVICES=` empty, in the image above. 25/25 handles compile, for chunks 512 and 256:
  - static C=1 (8 rows), C=2 (16), 12 and 32 rows, stamped 16;
  - the two-slot FC2 arm, stamped C=1, `xa`, `xs`, and the reform tile at 12 and 32 rows;
  - dynamic prefill Q0 words (m=2304), long SF6 words (m=16384) and its FFN packets.
- **Tests.** `cpu-tests.log`: 118 tests in 18 modules pass (4 GPU skips), on the tree merged with main `2fddde9e`
  (which includes #1002's vectorized MoE scatter). They cover the relayout read-back at the served
  and every named chunk, the idempotent marker and the refusal to re-lay tiled bytes at another chunk, the MoE config,
  scatter, sync, SF6 and batch-reform contracts, the boot paths and the dense dataflow.
- **Pre-existing failures.** `tests.test_engine_kernel_shape` fails 18 tests on pristine origin/main `5a134d48` (the
  Qwen3.8 tables), so it is not in the list.

## Not measured

- No four-rank consumer run, onepass, acceptance or step/s (D17).
- The first production boot after this change compiles the chunk-256 handles once per node (new cache keys).
- The `t` tile over 256 at short-prefill widths is exact, but its timing against 512 was not taken. That path serves 9–15
  and 17–80 token prefills.

## Reproduce

```sh
# CPU, in the image above
CUDA_VISIBLE_DEVICES= CUTE_DSL_ARCH=sm_121a PYTHONPATH=/repo python3 probes/engine_moe_chunk_compile.py --output /out/native-compile.json
# GPU, from a frozen checkout on srv2 through the single-GPU lane
ST_IMAGE=sha256:b45454b5fdc138bfaafa6470cf9cb49bbcd74783ceffb343b0a7b1d0cf87fcd5 ST_PROBE_TREE=<tree> ST_PROBE_GIB=12 \
bash bench/fleet.sh run --gpu --detach <session> 25 '<note>' -- \
  bash probes/run_engine_probe.sh probes/engine_kernel_check.py --lanes moe_c2_cells:chunk:shapes:chunks=512,256 \
    --ranks /home/choiceoh/models/st-glm53-9391-up-gate-full/rank3of4.safetensors --samples 4 --output /cache/<session>.jsonl
python3 summarize.py c2moe-cells3-0915.jsonl   # stdlib only; also c2moe-cells2-0915.jsonl, c2moe-chunk0915.jsonl
```
