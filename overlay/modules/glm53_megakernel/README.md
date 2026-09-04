# glm53_megakernel

The GLM-5.3-Flash decode megakernel for GB10 (sm_121a, 48 SM, TP=4): ONE
persistent 48-block CUDA launch per inter-collective segment of the decode
step, compiled AOT by nvcc (`-arch=sm_121a`) instead of JIT.

## MK-GEMM is the W4 lane (the fp8 W8 arm was removed 2026-09-02)

Weights are stored as e2m1 nibbles x one pow2 scale per 16 elements
(0.56x the fp8 bytes), and the kernel expands every nibble to an **exact
e4m3 byte** -- e2m1's 1-bit mantissas always fit, and the 2^s product is
an exponent-field add -- then runs the e4m3 `mma.sync` pipeline on fp8
activations (the W4A4 20-25% loss is the thing the literature warns about;
QServe's compromise axis). The expansion is a byte-permute table lookup:
the raw nibbles are the `prmt` selector over the LUT's two halves with the
group exponent folded in per 16 elements.

**It is faster than the stock quant+deepgemm pair on every decode shape**
(srv2, PDL on, second launch of a pair of different weights = the decode
step's steady state): 76/55/34.5/24 us vs 131/93/51/31 for
n=6416/4096/2048/1024 (MEASUREMENTS.md 5차). The fp8 MK arm that used to
sit beside it reached 118/76/44/27 -- ahead of stock, but a second arm to
select, self-test, bracket and maintain for 10-18% where W4 gives 22-42%.
The operator's call was to remove it: **the lane is W4 or stock**. One
kernel, one smem budget, one pack builder, one knob (`VLLM_GLM53_MK_GEMM`).

Two gates at boot: bit-exact expansion (1e-5, the kernel against a torch
fp32 matmul of the kernel-quantized activations and the dequantized pack,
on weights built ON the e2m1 grid -- anything above is a layout bug) and
by-design error (<= 0.15 vs the stock pair on the decode shapes). The
ceiling of the encoding is documented: s is clamped to [-5, 6], so |w| >
384 saturates (never NaN); real projection weights are orders of magnitude
below.

This lane CHANGES SERVED NUMERICS on every eligible decode linear, the KDA
in_proj included (there is no fp8 pack to keep it on): one numerics axis
per boot, bracket with quality 9/9 + Korean 0/16 + pos-1 acceptance within
2 pct or it reverts. The in-kernel KDA projections (MK-KDA) stream the same
W4 packs, so the KDA shadow diff against stock is gated at the e2m1
by-design class (0.15), not the 2e-2 noise class the fixture uses (the
fixture snaps its weights to the grid so both arms see the same values).

## K-chunked lane (2026-09-04)

The kernel's K contract is one launch of `MK_GEMM_KMAX` = 4096 (KBLK_MAX =
32 k-blocks: the A-quant tiles and the per-row scale stage in smem). The
DFlash2 drafter's `fc` is `[4096, 5 x 4096]` -- the single largest GEMM of
the decode tail (168 MB bf16, 792 us at 212 GB/s in the armed 09-03 trace)
and the one dense projection no quantized arm could take. Rather than widen
KBLK_MAX (and the smem budget of the 173 launches/step that already serve at
the W4 stream floor), `build_mk_weight_w4_kchunks` packs one W4 pack per
4096-column chunk and `gemm_w4a8` runs one launch per chunk with the
partials summed in fp32 (`_gemm_kchunks`): 301 us against 682 bf16 / 489 fp8
(`probes/drafter_fc_check.py`, srv2). A chunk that fails the per-launch
contract keeps the whole linear on the fp8 pair. `glm53_fp8_dense` attaches
the chunked pack automatically for any admitted linear wider than the lane.

## What it absorbs (and what it cannot)

The 2026-09-01 step decomposition fixes the budget: C=1 step = 66.0 ms, of
which **41 ms (62%) is expert weight reads at 91% bandwidth -- b12x tiles,
not touchable from here**. The remaining 25 ms is dense reads + kernel
launches + glue + AR, and the kernel census prices launches at 5.4 us each
(an upper bound that has been weakening). The megakernel attacks the launch
and glue component only; the bytes it reads are the same bytes.

| segment | replaces (per step, census counts) | stock | MK |
|---|---|---|---|
| `MK_SEG_MHC` | mhc_fused + pre_big_fuse_with_norm | 179 | 45 |
| `MK_SEG_GEMM` | per_token_group_quant + deepgemm, M<=32 | ~360 | ~180 |
| `MK_SEG_KDA` | whole linear-attention block (~15 kernels x 34 layers) | ~510 | 34 |

Ceiling if all three hold: ~900 launches x 5.4 us = **4.9 ms (7.4% of the
step)**. The honest expectation is BELOW that (5.4 us is an upper bound,
and the MK kernels' own phases add barrier cost), which is why adoption is
bracket-only: **none of these numbers is a measurement** until the EXP
bracket in RUNBOOK_KERNEL_CAMPAIGN2.md runs. Prefill is untouched in v1
(deepgemm and the big-fuse path stay; M > 32 always falls back). The one
prefill-side claim that CAN be made from construction: armed decode shapes
stop JIT-ing TileLang/deepgemm kernels at cold boot, shrinking the 11-41%
cold tax by the share this module covers.

## sm_121a contract

- Ampere lineage + FP4 extension: **no WGMMA / tcgen05 / TMEM / clusters /
  DSMEM**. The GEMM is `mma.sync.m16n8k32.f32.e4m3.e4m3.f32`; the W stream
  (the only bandwidth-heavy operand) streams its raw (tile, k-block)
  records -- 8 KB of nibbles + 1 KB of group exponents, contiguous -- through
  a **3-stage cp.async pipeline** (`VLLM_GLM53_MK_NBUF`, 3..5; deeper
  measured worse) and expands each landed record into one of two dense,
  swizzled e4m3 tiles in smem. A synchronous load->sync->mma chain leaves
  ~20% of the stream idle, so 2-in-flight is the difference between
  matching deepgemm and beating it. TMA stays a drop-in for later.
- **Programmatic dependent launch** (`griddepcontrol`, works on this part
  with CUDA 13): every MK kernel triggers its dependents at entry and
  waits before its first read of the previous kernel's output, so the next
  MK launch starts on the SMs this one frees and pulls its first W tiles
  during this one's tail. Two launches back to back measure 17-19% less
  per launch than one alone. Behind `VLLM_GLM53_MK_PDL=1` (default off
  until the serving bracket; the probe sets it).
- Fixed 48-block grid everywhere; the never-reset monotonic ticket barrier
  is what keeps CUDA-graph replay with baked pointers exact (the osar
  `done_ctr` trick). A larger grid deadlocked on this part (#150).
- Dynamic smem: 69,632 B (2 expanded tiles + 3 raw stages + A tiles +
  scales), 1 block/SM; the KDA kernel inlines the same phase on the same
  budget. It includes 1 KB of slack: the phase re-aligns the dynamic base
  at runtime, because the static `s_last`/`s_unit` push it to +16 and every
  128 B tile row then straddles a bank-line boundary (that alone hid 15% of
  the W stream; MEASUREMENTS.md 4차).
- The expanded e4m3 tile rows are dense 128 B with a 16 B-chunk XOR swizzle
  (`mk_swz`, keyed by row & 7) on both the expansion stores and the mma
  fragment loads. A padded 144 B pitch cost a pure stream 16% (194 vs 230
  GB/s, clean regime).

## Integration (all inside files this repo owns)

| hook | file | behavior when disarmed |
|---|---|---|
| MK-MHC | `glm53_mhc_tilelang/tilelang.py` small-M branch | falls through to ONEPASS/stock pair, byte-identical |
| MK-GEMM | `glm53_fp8_dense` `Fp8DenseMethod.apply` + build | stock quant+deepgemm pair |
| MK-KDA | `glm53_mk_kda_wiring`'s `kda.py` overlay (image preimage `ec090aab...`) | stock forward body verbatim |
| MK-MLA | `glm53_mk_mla_wiring` (FlashInfer SM90 sparse backend) | the wrapper's own plan+run |
| MK-MHC (dsv4) | `dsv4_mhc_tilelang` small-M branch, the same hook | stock swept pair, byte-identical |

Both MHC wirings call ONE entry point in this module -- `mhc_hook(...)`,
which arms and then tries -- through a resolver that caches the import once
(the hook is on the decode hot path, one call per layer per step). The two
image files are separate forks that no compose rule ties together, so the
code inside those blocks is kept byte-identical and `test_megakernel_core_is_shared`
compares them; the comments differ per lane. dsv4 additionally reaches the
wrapper only while `VLLM_USE_B12X_MHC` is off -- see that module's README.

The kda.py copy came from the image (`/tmp/deployed/kda.py`, extracted
2026-08-31). If a future image bumps that file, deploy-overlays' preimage
gate catches it before a boot lies about what it is running.

## This module is the core, and it is model-agnostic (2026-09-03)

It binds TWO rows -- `glm53_megakernel.py` and `.cu` -- both new files
(`absent` preimage) at a target RELATIVE to the profile's `TARGET_PREFIX`
(`vllm/model_executor/layers/`). Nothing in it names a model directory, no
row can drift against an image that never shipped these files, and the
python module imports only stdlib + torch at import time (every vllm import
is inside a function). `glm5next_kda.py` -- the one row that WAS
GLM's -- moved to `glm53_mk_kda_wiring`, and the `requires` line went with
it, to the hook that actually needs the fp8-dense arm.

That is what lets **`profiles/dsv4.env` mount this same module**
(2026-09-03, every knob 0). DeepSeek-V4-Flash reaches exactly ONE segment:

| segment | dsv4 | why |
|---|---|---|
| `MK_SEG_MHC` | **applies** | same MHC geometry (hc_mult 4, sinkhorn 20, hc_eps 1e-6, hidden 4096), identical wrapper signature -- `dsv4_mhc_tilelang` carries the same branch GLM's `tilelang.py` does |
| `MK_SEG_KDA` | no | the model has no linear-attention layer at all |
| `MK_SEG_MLA` | no | this kernel is NoPE / kv_lora 512 / topk 2048; V4-Flash has rope 64, topk 512, a compressor and a sliding window |
| `MK_SEG_GEMM` | no | the lane is W4 (e2m1 packs built from bf16); V4-Flash's dense weights are block-fp8 with no bf16 source. The fp8 W8 arm that WOULD have fit was removed 2026-09-02 (`cfeae2b`) |

The unreachable segments still compile into the extension there; they are
never launched. Nothing is measured on that model yet, its stock MHC pair is
already swept (unlike GLM's), and it is production. The ladder below applies
unchanged, on that image, before any arm.

**The serving window is the wrapper's, not the kernel's** -- true on BOTH
models and not previously written down: the MHC hook sits inside the
wrapper's `use_small_fma` branch (`T <= 16`) while the kernel's own gate is
`T <= 32`. So the hook is offered `C <= 2` on GLM (8 tokens/seq) and `C <= 2`
on dsv4 (6 tokens/seq); `16 < T <= 32` is the stock post+big_fuse branch,
which MK never sees. Every T=32 number recorded for MK-MHC is a KERNEL
measurement (`probes/megakernel_glm53_bench.py` calls `_mhc_call` directly),
not a served shape. `--stock dispatch` in that probe measures the arm a boot
would really take and prints a `hit` column that says so.

The module name still says `glm53`. Renaming it (dir, sources, container
path, `VLLM_GLM53_MK_*`) is 86 references across 21 files including the
ledger, which must not be rewritten; it waits for a measured win on the
second model.

## Arming

```
VLLM_GLM53_MEGAKERNEL=1     # master, default 0
VLLM_GLM53_MK_MHC=1         # per segment, each default 0
VLLM_GLM53_MK_GEMM=1
VLLM_GLM53_MK_KDA=1         # shadow first (state-index section below)
VLLM_GLM53_MK_KDA_SHADOW=1  # dual-run KDA eagerly, stock stays real
VLLM_GLM53_MK_PDL=1         # programmatic dependent launches, default 0
```

Arm happens lazily on the first eligible call: device must be exactly cc
12.1 / 48 SMs, then a per-segment boot self-test diffs the MK kernel
against the stock path it replaces (`torch.cuda.synchronize` before the
verdict -- the w4a8 lesson: a python fallback cannot contain an async CUDA
launch failure, so nothing arms unverified). A failed self-test disarms
that segment and logs `[megakernel] ... -> DISARM`. Armed hooks do NOT
try/except their launch.

First arm compiles the extension (34.4 s of nvcc) and logs the source md5
+ kernel count fingerprint. The build directory sits on the container's
PERSISTENT cache mount -- `/cache/mk_build`, next to `TRITON_CACHE_DIR`
and `VLLM_CACHE_ROOT`, overridable with `VLLM_GLM53_MK_BUILD_ROOT` and
falling back to `/root/.cache` then the old container-local path -- under
a name that hashes the source, the nvcc flags and the torch/CUDA pair. So
a restart reloads the .so in 0.3 s while a deploy, a probe flag sweep or
an image bump still recompiles (measured 59.4 s cold, 0.3 s warm).

## The state-index contract (settled 2026-09-02 from the stock source)

`spec_state_indices_tensor` is `[n_spec, max_query_len]`: **one state slot
per query position**. The stock kernels (`fused_recurrent.py`,
`IS_SPEC_DECODING`; `causal_conv1d.py`, the spec update kernel) read
`num_accepted_tokens` as the PREVIOUS step's acceptance count:

- recurrence: resume from slot `[r, acc - 1]`, and after every token j
  store the running state into slot `[r, j]` (slot <= 0 is NULL_BLOCK_ID:
  the program returns without outputs);
- short conv: the history for this step's first token is the state
  window's entries `[acc - 1, acc, acc + 1]` (the three ending at the last
  accepted draft), and the window written back keeps `old[acc], old[acc+1]`
  ahead of the new tokens (conv slot = `[r, 0]`, as the serving overlay
  passes `spec_state_indices_tensor[:, 0]`).

This kernel used to resume from `[r, 0]`, write the recurrent state only
at `j == acc - 1`, and read the conv history from the buffer's newest end
-- right only when every draft was accepted, which is why the fixture's
acc=8 passed and acc=1/3 differed by ~1.3 in the output. The fixture now
addresses distinct slots per position (`SLOT .. SLOT + 7`), so the boot
self-test and the bench probe check the per-position contract itself.
Eager **shadow mode** (`VLLM_GLM53_MK_KDA_SHADOW=1`) stays the live gate:
both arms run, outputs AND the states the next step reads are diffed every
64 calls and on drift (gated at the e2m1 by-design class, see above), graph
replay stays stock. Run shadow on a bench boot, read the log, then decide
the arm.

## MK-MHC structure (2026-09-02)

No grid barrier. p1 runs one block per (chunk of 256 hidden dims, token
group): the chunk's `fn` slice -- 24 outputs x 4 streams for the thread's
dim -- sits in 96 registers and the group's tokens stream through it (the
old (token, chunk)-pair mapping re-read `fn` per token: 12 MB through L2 at
T=8 in one round, the L2 rate, not DRAM). The next token's inputs and mix
coefficients are prefetched, the 25 per-token partials reduce through a
transposed smem tile with 8-lane groups, and each chunk's completion is
counted per token. Blocks that finish their tokens take **tail tickets**:
wait on a token's 16 arrivals, rearm its counter, and run that token's p2
(one warp: rms, mixes, sinkhorn) and fused p3+p4 (registers, no ol_stash /
sq round trips). The last block out rearms the ticket counter, so graph
replay needs no reset. Things that measured worse on the way (MEASUREMENTS
9차): `fn` in smem (fill latency-bound, 1 block/SM), the tail on each
token's last arriver (tails serialize on one block), a 24-value shuffle
broadcast in the tail (spills under the p1 register pressure).

Bench (srv2, PDL on, **sinkhorn_repeat=4** -- the basis before 2026-09-03, reproduce with `--sinkhorn 4`): T=8 27.4 us, T=32 42.0 us vs stock 32.8 / 71.6
(MEASUREMENTS.md 9차 has the eight experiments that got here).

## MK-KDA phase budget (2026-09-02, srv2, acc=3, L2 drained before launch)

in_proj 76 | gates 6 | conv 4 | delta 34 | norm 0.5 | o_proj 35 |
barriers ~17 = **176 us** per layer-step (402 before the phase
stamps went in; stock's five kernels 640+). Four grid barriers: gates and
conv share a phase (both read only in_proj's output). The gates are a
cp.async + `mma.sync m16n8k16 bf16` GEMM over 32-row weight tiles; the
conv is unrolled to the 8-token spec window (the host refuses a wider
`max_query_len`); the delta rule runs two blocks per head (rows split, S
register-resident, per-token state stores staged through smem) while the
16 head-less blocks warm L2 with the o_proj pack (`prefetch.global.L2`);
p4 emits the o_proj's fp8 A tiles itself so p5 starts without a prologue;
split-K never makes a slice shorter than 8 k-blocks (o_proj: r=2, not the
cost model's 3). `-DMK_PHASE_TS=1` + `read_kda_ts` give the per-phase,
per-block stamps; the fixture's `mk_run(drain=True)` keeps its own 10 MB
state clones from polluting the first phase; `VLLM_GLM53_MK_KSR_IN/OUT`
force a split for probing (MEASUREMENTS.md 8차, 10차).

## Review fixes already folded in (2026-09-01)

* conv-state slot stride was missing its `*(CONV_W-1)` factor — slot >= 1
  read/wrote the wrong region (OOB on the last slots). Caught by re-reading
  the kernel, NOT by the original self-test: the fixture had used slot 0.
  The fixture now addresses a nonzero slot on purpose, and the slot-stride
  constants are pinned in test_logic.
* `maybe_arm` refuses to compile/self-test under graph capture (vLLM warms
  up eager first, but the guard makes that a contract instead of a hope).
* `kv_cache` may be unset on early forwards; the takeover gate now checks
  instead of raising.

## Memory note

MK-GEMM keeps its W4 pack per linear beside the deepgemm pair (which must
stay as the M>32 path): 0.56x the fp8 bytes, ~+2.3 GB/rank at full
coverage. At GMU 0.73 that may not fit; the first MK-GEMM boot should
watch the KV-cache line and be ready to drop GMU a notch (memfree-preflight
computes it). If it does not fit, arming MK-GEMM only for the KDA in/out
projections is the fallback scope (a one-line change in the build attach).

## The sinkhorn basis (2026-09-03)

`SINKHORN_SERVED = 20` in the driver is what both models pass as
`sinkhorn_repeat` (`hc_sinkhorn_iters`), and it is now the ONE number: the
boot self-test gates on it and `probes/megakernel_glm53_bench.py` defaults to
it. It used to be 4 in both, which ran 3 loop iterations where serving runs
19 -- the first row/col pass places its eps differently from the loop's, so a
divergence that only opens up later would have armed clean and served wrong.
The gate is stricter now: if MHC DISARMs on the next boot, that is a finding
rather than a regression (the lane falls back to stock either way) -- but read
the logged rel_errs first. `_TOL_MHC` was calibrated at 4 and has not been
re-derived at 20, so an accumulation artefact and a real MK/TileLang
divergence trip the same boolean; `--sinkhorn 4` on the probe separates them.

Every MHC number recorded before that date -- the bench line below, and the
4/9/10차 rows in MEASUREMENTS -- was taken at 4 and is not comparable to a
fresh run without that flag.

## Verification ladder (in order, no skips)

1. `python3 tests/test_logic.py` -- pure logic + .cu/.py geometry parity +
   manifest invariants (this repo, no GPU).
2. `bash probes/run_megakernel_bench.sh [--profile glm53|dsv4]` in a fresh
   container (srv4, never the serving one; the wrapper binds the profile's
   composed overlay at that image's real paths, and passes its package
   root as `MK_PKG_PATH`): numerics vs stock (rel gates 1e-3 MHC / 0.15 GEMM
   by-design + 1e-5 exact-grid, 2e-2 KDA outputs and states on grid-snapped
   weights) + CUDA-event timing per segment + a replay-stability
   check (re-launch drift <= 1e-6 over the shared workspace -- the
   monotonic-barrier contract). `!` marks any cell over gate; a `!` cell
   disqualifies that shape.
3. `VLLM_GLM53_MK_KDA_SHADOW=1` bench boot: shadow log clean for a full
   bench-tp4 pass.
4. EXP bracket (RUNBOOK_KERNEL_CAMPAIGN2.md): base -> cand -> base on
   C=1 step/s, quality 9/9, Korean 0/16, pos-1 acceptance within 2 pct.

Rollback is one env line per segment; unmounted hooks are inert imports.
