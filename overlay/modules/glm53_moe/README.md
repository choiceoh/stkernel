# glm53_moe

GLM-5.3 MoE — b12x 공유 워크스페이스, EP 마이크로커널 레인, 직접 출력.

2026-09-05 (34차, 운영자 "디폴트화된 모듈들을 4~5개씩 하나로 묶어라") 에 아래 모듈들을 이 디렉터리 하나로 합쳤다. **매니페스트 행·베이스 계약·소스 파일·노브·기본값은 그대로**이고 디렉터리와 `manifest.tsv`·`requires`·README 만 합쳐졌다(합성 결과 `build/glm53/` 의 파일은 바이트 동일(메가커널 .cu 주석의 경로 한 줄 제외), 행 순서만 바뀜). 옛 이름은 원장·런북·커밋에 그대로 남아 있고, 아래 절이 옛 모듈 하나씩이다.

| 옛 모듈 | 파일 | 무엇 |
|---|---|---|
| `b12x_shared_workspace` | `flashinfer_b12x_moe.py` | `B12xMoEWrapper` 형상당 하나 + EP 경로 |
| `b12x_zero_weight_micro` | `moe_dispatch.py`, `moe_micro_kernel.py` | EP 마이크로커널 레인 (`B12X_EP_ZERO_WEIGHT_MICRO`, 실험) |
| `glm53_b12x_out` | `b12x_moe.py` | b12x MoE 직접 출력 (`B12X_DIRECT_OUT`) |

---

## b12x_shared_workspace (was `overlay/modules/glm53_moe/`)

## b12x_shared_workspace

One `B12xMoEWrapper` per geometry instead of one per layer.

Each wrapper carries graph-stable scratch sized by the MoE geometry. Measured on
this deployment (288 experts, top_k 8, hidden 4096, intermediate 2048,
max_num_tokens 2048):

```
wrapper 1개: allocated +541.1 MiB
MoE 층 43개: 22.72 GiB per rank
공유 시 절감: 22.19 GiB
```

GLM-5.3-Flash has 43 MoE layers of identical geometry, so the duplicate buffers
were a quarter of the entire GMU budget. The arithmetic matched what the engine
reported: 45.8 GiB of weights + 22.7 GiB of wrappers against an 87.4 GiB budget
at GMU 0.73 leaves ~16 GiB for KV, and the engine logged 16.52 GiB.

Sharing is safe because MoE layers run one after another on the same stream: the
wrapper writes into its buffers on every call and nothing outlives the call.
Values are held weakly, so the wrapper dies with the last layer referencing it.

Same idea as vllm-project/vllm#48698, written against the file this image ships.
(#53081 shares the *workspaces* rather than the wrapper, but needs FlashInfer's
`shared_static_workspace` from flashinfer-ai/flashinfer#4603, which this build --
0.6.18.dev20260819 -- does not have.)

### Expert parallelism

The fused kernel still raises at `num_local != num_experts` (flashinfer #3383:
spec sizes weights by `weight_E`, then illegal-address on local tensors). This
module does not lift that. With the default `VLLM_B12X_EP_NO_DUMMY=1`, EP keeps
only the local weight rows (`E = num_local`) and remaps remote routes to a
sentinel that is removed before the kernel call. The default #146 decode
fallback replaces remote slots with zero-weight repeats and submits at most
eight `top_k=1` rows per call. Stable 8/16/32-token C=1/2/4 shapes therefore
take 8/16/32 micro calls per MoE layer and never enter the static backend that
hangs on this EP geometry. Its `pair_out` remains zero-initialised and masked
before `index_add_`.

Exact `VLLM_B12X_EP_STOCK_TOPK_MICRO=1` enables a narrower experiment for the
pinned GLM E=288/global, E=72/local, K=4096, N=2048, top-k=8 geometry and only
the stable 8/16/32-token shapes. Remote sentinels become same-token local IDs
at router weight zero, then the original token-major tensors run in balanced
chunks of at most five tokens / 40 routed rows. That removes pair flatten,
sort, gathers, pair output, mask, and `index_add_`, reducing the call counts to
2/4/7. The knob is strict, latched, default-off, and mutually exclusive with
`VLLM_B12X_EP_ZERO_WEIGHT_MICRO`; live GPU numerics and graph replay are still
pending. Every shape outside the exact gate retains the #146 fallback.

Prefill (`tokens * top_k > 640`) drops remote slots instead of paying GEMM for
them. All direct paths return before the large graph wrapper and its capacity
probe are constructed.

An experimental third direct lane is mounted by `b12x_zero_weight_micro` but
stays off by default. Exact `VLLM_B12X_EP_ZERO_WEIGHT_MICRO=1` requires
`VLLM_B12X_EP_NO_DUMMY=1` and `VLLM_B12X_EP_DISABLE_MICRO=0`. It preserves the
same physical E=72 weights and leaves each remote route as sentinel id 72 at
weight zero for the micro kernel to discard before row materialization. Only
stable 8/16/32-token GLM decode shapes are admitted, as 1/2/4 disjoint
eight-token top-k=8 calls; every mismatch uses this module's existing fixed
#146 fallback. A separate pinned top-k=8/max_rows=64 workspace preserves the
#147 CUDA-graph lifetime contract. This has no GPU correctness or E2E win yet.

The default fallback and both experiments hold separate shared workspaces:
top-k=1/r8, stock top-k=8/r40, and overlay top-k=8/r64. Compact prefill keeps
using FlashInfer's replaceable functional cache, but growing that cache cannot
invalidate addresses captured by decode CUDA graphs.
`VLLM_B12X_EP_NO_DUMMY=0` is the rollback path: it restores the historical
`E = num_local + 1` zero-weight dummy and the wrapper-backed call. The remap
always writes into preallocated scratch (`out=` / in-place), and vLLM's EP
all-reduce (DP=1) combines the partial hidden states.

`ENABLE_EP=1` on the glm53 launcher. Off by default — the TP-sharded path is
the measured one. EPLB is refused (`_supports_parallel_config`).
`VLLM_B12X_EP_COMPACT=0` keeps every batch fixed-shape; with the default
no-dummy path, remote slots become slice-local zero-weight repeats.
`VLLM_B12X_EP_DISABLE_MICRO=1` is a plain-static diagnostic and is rejected
with the default no-dummy path; also set `VLLM_B12X_EP_NO_DUMMY=0` to use it.
Default setup verifies `_MICRO_MAX_TOKENS >= 8`. The stock experiment also
requires `_MICRO_COMPACT_CUTOVER_PAIRS_MULTI_TOPK >= 40` and automatic backend
selection. A missing or smaller private boundary fails closed instead of
silently selecting static.
PIECEWISE graphs force compaction off so prefill capture keeps a fixed shape.

---

## b12x_zero_weight_micro (was `overlay/modules/glm53_moe/`)

## b12x_zero_weight_micro

Experimental, default-off micro-kernel lane for GLM-5.3-Flash expert-parallel
decode on the pinned `glm53:v13-b12x` FlashInfer sources.

`VLLM_B12X_EP_ZERO_WEIGHT_MICRO=1` is accepted only together with the default
local-only mode (`VLLM_B12X_EP_NO_DUMMY=1`) and micro enabled
(`VLLM_B12X_EP_DISABLE_MICRO=0`). Unset or `0` preserves the #147 direct
top-k=1 path byte-for-byte at the wrapper boundary. Any other spelling is a
setup error.

### Exact lane

The wrapper admits only the stable DFlash token counts 8, 16, and 32 and splits
them into 1, 2, or 4 disjoint eight-token calls. Every call is the exact GLM EP
geometry: E=72 local weight rows, top-k=8, hidden=4096, intermediate=2048,
clamped SiLU (`swigluoai_uninterleave`, limit 10.0), NVFP4. Other token counts
keep the #147 fixed top-k=1 fallback. A separate strongly-held E=72/top-k=8/
max_rows=64 workspace keeps CUDA graph addresses independent of the functional
workspace cache.

The local-only remap represents a remote route as sentinel expert id 72 (equal
to E) at exact router weight zero. The Triton pre-pass compacts at most 64
routed ids into `weight_expert_ids[0:unique]`; `unique <= routed_rows = 64 <=
state_E = 72`, so the sentinel is a map value and never an out-of-bounds map
index. In the micro kernel, the CTA leader resolves compact id to that weight id
and assigns row=-1 only when both `weight == 0` and `expert_id == 72`. It does
not append row_counts/token_map/token_weights or quantize that pair. The compute
scheduler sees no row for the sentinel, therefore no alpha, W13, or W2 plane is
read. A real expert carrying an exact-zero router weight is intentionally not
skipped.

The sentinel id is part of the compiled micro cache key. The stock and
experimental variants therefore cannot alias either the in-process cache or
the disk kernel name. Single-token, shared-input, forced-backend, direct-micro,
static, non-GLM, and routed_rows>state_E cases fail closed.

No GPU correctness or throughput win is claimed. The knob stays off until the
operator runs numerical and C=1/2/4 graph replay brackets on the fleet.

### GLM TP dispatcher tuning controls

The same dispatch overlay now exposes default-off controls for the current
TP-sharded GLM path. They are admitted only for the exact deployed geometry:
NVFP4, `E=288`, `H=4096`, `N=2048`, top-k=8, clamped
`swigluoai_uninterleave` with limit 10. EP (`E=72/73`), other models, W4A16,
and shape drift keep the shipped values even if one of these variables is set.

- `VLLM_GLM53_B12X_FORCE_BACKEND=auto|micro|static|dynamic`. `micro` is
  decode-only (`M<=8`); longer calls stay automatic instead of failing a
  prefill launch. `static` across the full `MAX_BATCHED` range can allocate a
  much larger static workspace, so it is a diagnostic rather than a serving
  default.
- `VLLM_GLM53_B12X_STATIC_CUTOVER_PAIRS=N`. The value is in routed pairs
  (`tokens * top-k`); zero forces dynamic for every exact GLM call. The older
  launcher-only `MOE_CUTOVER` remains available but is process-wide rather
  than exact-shape gated.
- `VLLM_GLM53_B12X_{MICRO,STATIC,DYNAMIC}_MAC_LADDER` replaces that backend's
  ladder with comma-separated `max_rows:mac` cells, for example
  `"64:48,128:48,640:40"`. Row bounds must increase strictly and MAC values
  must be positive. The runtime still clamps MAC to the hardware limit; rows
  beyond the final override cell use the hardware default.

All values are parsed once during module import. Empty/unset values preserve
the existing selection; malformed values fail before CUDA launch. The profile
keeps every value empty because this change adds experiment control only -- it
does not claim a benchmark winner or alter the production default.

---

## glm53_b12x_out (was `overlay/modules/glm53_moe/`)

## glm53_b12x_out

Takes over flashinfer's `b12x_moe.py` to add one thing: `B12xMoEWrapper.run(...,
out=...)`, which makes the caller's buffer the MoE kernel's scatter target.
Without it, every MoE layer materializes its result into the wrapper's static
workspace and the caller (our `flashinfer_b12x_moe.py`) copies it out — two
writes of the same bytes per layer, ~42 copy kernels/step on this lane. With
`VLLM_B12X_DIRECT_OUT=1` (default) the apply passes `out=output` and returns.

Capture-safe by the same evidence the copy provided: the replaced `copy_`
wrote that exact address under capture and replays, so a direct kernel write
to it does too. The base-preimage contract pins this fork to the image's
flashinfer file (43a12178…) — an image bump that touches b12x_moe.py aborts
deploy loudly instead of silently dropping the patch.

Only glm53 mounts this module (dsv4's MoE path does not go through
b12x_shared_workspace).

## Static v2 / v3 / v4 (35차, from b12x_zero_weight_micro): the decode-streaming static kernel

**34차 §8 (2026-09-06, 운영자 "전부 지워")**: `moe_static_kernel_v2.py`(레인 `1`/`m..`/`d`)와 `moe_static_kernel_v3.py`(레인 `w`/`e`/`k`)는 삭제됐다 — v4 `u`(38차 부록 +2%)가 두 세대 앞선다. 둘이 공유하던 스탬프 슬롯·PTX 헬퍼는 `moe_static_common.py` 로 옮겨 v4 가 거기서 import 한다. 파서는 옛 토큰을 조용히 재매핑하지 않고 거부한다(`u`/`v` 만 유효). 아래 v2/v3 서술은 기록이다.

`VLLM_GLM53_B12X_STATIC_V2` (profile default `u` = v4 since 38차; `w` = v3 was the 35차 default, `""` = stock) routes the exact
GLM-5.3 TP geometry's static (decode) MoE launches to `MoEStaticKernelV2`, a
rework of flashinfer's `MoEStaticKernel` with the same workspace, weight
views, routing frontend and arithmetic (FC1 accumulation order, fp4 quant of
the intermediate, FC2 accumulation, bf16 atomic scatter) but a different
streaming structure:

- gate and up come in on one TMA pipeline stage (A + B_gate + B_up + scales on
  one mbarrier) and accumulate in the same k loop -- the stock kernel runs
  two passes with a drained pipeline and a CTA-wide barrier between them;
- FC2 has its own stage buffers, so the DMA warp issues the item's 32 down
  tiles the moment the FC1 loads are out and the next item's FC1 loads right
  after -- no barrier at the item boundary (the quantized intermediate lives
  in its own `sA2/sSFA2` instead of FC1's stage 0);
- the A TMA box is `a_rows` rows (default = tile_m) instead of 128, and
  tile_m = 32 is admitted, which keeps three accumulators plus two B fragment
  sets inside the MMA warps' 232 registers;
- pipeline states advance monotonically across items (no `reset_count`), so
  any stage count works; `producer_tail` drains at exit.

Spec: `"1"` = `m32,f2,g4,a32` (tile_m 32, FC1 2 stages, FC2 4 stages, 32-row
A box, static item schedule); or a comma list of `m<tile_m>`, `f<fc1 stages>`,
`g<fc2 stages>`, `a<A rows>`, `d` (dynamic item schedule: the DMA warp claims
items from a global counter the kernel zeroes in its phase 0 and hands the
coordinates to the MMA warps through a smem ring published by each item's
first FC1 stage barrier; a sentinel stage ends the loop) and `s` (per-CTA
`%globaltimer` stamps into an int64 `[grid, STAMP_SLOTS]` tensor -- probe
only; the serving spec must not carry `s`), and `w` (`moe_static_kernel_v3.py`:
FC1 streamed as two 64-wide halves over 256-wide K stages so every w13 TMA
box row is a full 128 B line -- the v2 stamps put FC1 at 216 GB/s against
FC2's 242 with the same pipeline, and a vectorized-load streamer measured
DRAM at 167 GB/s for 64 B row segments, 202 for 128 B, 222 for 256 B; FC2
and the intermediate layout are v2's, tile_m 32, static schedule, FC2 3
stages by default), and `e` (v3 only, even waves: the routing phase counts the
items -- one m-tile per 32 rows of an expert, times the 4 intermediate
slices -- into `next_item[0]`, and after the second grid barrier every CTA
picks the CTA count in {48, 44, 40, 36, 32} whose last wave leaves the fewest
empty item slots (ties to the larger); CTAs beyond it exit. U=40 experts =
160 items run as 40 CTAs x 4 full waves instead of 48 x 3.33, whose last
third-of-a-wave streamed at 16 CTAs' worth of bandwidth; multiples of 4 keep
an expert's 4 slice-CTAs co-scheduled so DRAM still sees whole w2 rows;
measured a wash -- 40 CTAs stream 220 GB/s against 48 CTAs' 229, the
aggregate rate grows sub-linearly with the stream count, ledger 38차), `k` (v3
last-wave split: when the last wave has p items and 2p CTAs, each item runs on
two CTAs -- the striding owner streams FC1 half 0, the helper `bidz + p` half
1, each zeroes the other half of the intermediate and runs the full FC2, the
bf16 atomic scatter adds the partials; the item count comes from the same
`next_item[0]` counter; U=40: -1~2%, numerics within the gate), `u`
(`moe_static_kernel_v4.py`: v3 with 512-wide FC1 K stages and gate/up in
separate stages, so every w13 TMA box row is 256 B -- the timing-only cells
`xs`/`xa` showed FC1 at 200 GB/s = the streamer's 128 B-segment figure while
FC2 sits at 237; 32 KB stages, FC2 2 stages by default, smem 101,376 B =
the opt-in maximum), `t` (`moe_static_kernel_v5.py`, 39차: v4 over
**tile-major expert weights** -- the dispatcher's `_get_weight_views(tiled=
True)` re-lays each expert's w13 out as [K/512 k-tiles][rows][256 B] and w2
as [I_tp/128][rows][64 B], so a (64 rows x 512 K) FC1 box is one contiguous
16 KB run and a (128 rows x 128 K) down box one 8 KB run instead of 64 / 128
row segments 2 KB / 256 B apart; the kernel body is v4's, only `__call__`
groups the 4-D tensors' two K modes into one hierarchical K for the TMA
map, so results are bit-identical to v4 up to the scatter order. The 35차 §6
streamer rated the kernel's 256 B-segment access at 222 GB/s against 239
linear, and v4's own stamps sit at 200-206 (38차 §3, §5) while FC2 -- whose
four slice CTAs happen to cover whole w2 rows -- streams at 237. Phase 1 wires
the tiled layout to the v5 static kernel only: the wrapper keys its cached
weight views on the lane's tiled flag, the entry refuses the dynamic
(prefill) backend with tiled views, and the static launch checks views
against lane -- a tiled view can never reach a kernel compiled for the
row-major layout. Serving (phase 2, same round): the vLLM wrapper re-lays
the layer's packed bytes out in place once at weight post-processing (TP
only; proof line `[b12x static v5] expert weights re-laid out tile-major in
place`), the dispatcher views them without a copy, the micro lanes are off
under tiled weights (they read row-major), and the gated dynamic (prefill)
kernel is an overlay of the image's `_moe_dynamic/gated.py`
(`moe_dynamic_gated.py`, pinned preimage) that groups the 4-D weights into a
hierarchical K -- its 64 B tiles divide the 256 B / 64 B chunks -- and is
compiled and cached per layout (`dynamic_..._tiled`); both kernels compile
on the CPU check), `h` (v4/v5, 39차: the FC1 SFB TMA box covers the 64
rows a gate/up stage reads instead of the 128-row scale block -- v4 loaded
the whole 4 KB block per 64-row stage and read half, 32 KB of an item's
~830 KB from DRAM; the smem block stays 128 rows and the box lands in the
half the MMA warps read), `z` (39차 v6: `t`'s tile-major storage
pre-swizzled into the smem layouts' own byte order -- B1 `S<3,4,3>` over
(64 rows x 512 K): `lin = r*256 + k%256 + (k//256)*16384`, `phys = lin ^
(((lin>>7)&7)<<4)`; B2 `S<2,4,3>` over (128 x 128): `phys = lin ^
(((lin>>7)&3)<<4)`; measured with `probes/b12x_static_layout_print.py
--dump` and cross-checked byte for byte -- so the DMA warp lands each B
stage with ONE 1-D `cp.async.bulk` (16 KB / 8 KB) on the stage's mbarrier
instead of a 2-D TMA box of 64 / 128 row segments: the path is L2 request
rate bound (38차 §8) and a bulk copy is the fewest requests a stage can be.
Probe only until the gated prefill kernel reads that order: the entry
refuses the dynamic backend on swizzled storage), and the probe-only timing
cells `xs` (skip the FC1 SFB boxes) and `xa` (skip the A + SFA boxes) that
the serving parse rejects. The
cache key and on-disk kernel name carry the config; the source files are in
`_kernel_source_files()`, so an edit invalidates the module cache like any
other kernel file.

Measured by `probes/b12x_static_probe.py` (stock vs v2, DRAM-cold, graph
replay, numerics gate, stamps) and `probes/b12x_dram_pattern_bench.py` (the
kernel's TMA access pattern vs a linear read, plain CUDA). A kernel edit is
compiled to its .o on the CPU first -- `MK_PROBE_NO_GPU=1 bash
probes/run_mk_probe.sh probes/b12x_static_compile_check.py --specs "u|t"`
(~2 s a spec, no CUDA context: the host may be serving). Ledger: 35차, 38차,
39차 (`t`).
