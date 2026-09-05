# b12x_zero_weight_micro

Experimental, default-off micro-kernel lane for GLM-5.3-Flash expert-parallel
decode on the pinned `glm53:v13-b12x` FlashInfer sources.

`VLLM_B12X_EP_ZERO_WEIGHT_MICRO=1` is accepted only together with the default
local-only mode (`VLLM_B12X_EP_NO_DUMMY=1`) and micro enabled
(`VLLM_B12X_EP_DISABLE_MICRO=0`). Unset or `0` preserves the #147 direct
top-k=1 path byte-for-byte at the wrapper boundary. Any other spelling is a
setup error.

## Exact lane

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

## GLM TP dispatcher tuning controls

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

## Static v2: the decode-streaming static kernel (`moe_static_kernel_v2.py`)

`VLLM_GLM53_B12X_STATIC_V2` (profile default `""` = stock) routes the exact
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
A box); or a comma list of `m<tile_m>`, `f<fc1 stages>`, `g<fc2 stages>`,
`a<A rows>`, and `s` (per-CTA `%globaltimer` stamps into an int64
`[grid, STAMP_SLOTS]` tensor -- probe only; the serving spec must not carry
`s`). The cache key and on-disk kernel name carry the config; the source file
is in `_kernel_source_files()`, so an edit invalidates the module cache like
any other kernel file.

Measured by `probes/b12x_static_probe.py` (stock vs v2, DRAM-cold, graph
replay, numerics gate, stamps) and `probes/b12x_dram_pattern_bench.py` (the
kernel's TMA access pattern vs a linear read, plain CUDA). Ledger: 33차.
