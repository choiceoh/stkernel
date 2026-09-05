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

## The shared expert's pair: the barrier-free local-quant kernel (2026-09-04, `VLLM_GLM53_MK_LOCALQ`)

The decode step's 30-45 us class is the shared expert's gate_up `[1024 x
4096]` and down `[4096 x 512]` -- two launches in each of the 42 MoE layers
(the 28차 trace reading counted 86), 3.5 ms, 2.4 ms of it not covered by
another stream -- 1-2 MB each, i.e. 5-12 us of DRAM at the lane's rate.
Both are one-unit-per-block launches (32 units on the 48-block grid), and
serving forks them onto the aux stream beside the routed MoE call of the
same layer.

`mk_gemm_lq_kernel` (`mk_gemm_phase_t<true>`, a compile-time instantiation
that carries none of the barrier path) is the lane's kernel for them:
every block quantizes the A k-blocks of ITS unit straight into smem from x
in L2 as it stages them (x loaded two k-blocks ahead in a register ring,
the reduce + convert ahead of the mma, only the smem stores behind the
barrier), instead of the grid-wide prologue (one k-block per block into
`g_mk_aq`, a grid barrier, `sxs` from `g_mk_axs`). The three A quantizers
of the lane -- that prologue, this path, KDA p4 -- share one set of helpers
(`mk_warp_amax`, `mk_pow2_scale`, `mk_pow2_rcp`, `mk_pack4`), so the bytes
are the same by construction; the boot self-test runs the exact e2m1
fixture through BOTH kernels, checks the plan really took the local one,
and requires the two outputs bitwise equal. It is launched on as many
blocks as it has units (`mk_lq_launch_grid`), has no grid barrier, and a
block with nothing to do leaves before the PDL wait. A host/kernel drift on
the unit rule traps (it must never run the barrier path on a launch sized
to the units).

What the 29차 probes say (srv2): **standalone it is slower than the global
kernel** -- the prologue it skips costs ~5 us on the stamps and the in-loop
quant cost more than that back (the first form +4 us at m=8, +12 at m=32;
the 32-block v2 form +8 us on the pair; the ring/row-bound form of this
text is unmeasured). **Under the routed MoE kernel**
(`probes/mk_gemm_concurrent_probe.py`: one graph, the pair on a forked
stream beside a U=40 b12x call, 5 back-to-back replays per bracket) the
global kernel's pair is exposed whole in either issue order (47.4 us a
layer); the local kernel on 32 blocks is exposed 36.2 (MoE issued first) /
31.8 (pair first, serving's order). x42 layers that is a *projection* of
about -0.66 ms/step, not a step number -- the probe's main stream holds
only the MoE kernel, serving runs the router + topk there first. Whether
the difference is the missing barrier or the smaller grid is what the
probe's control row (the global kernel on 32 blocks, its own ticket
counter) separates -- pending.

Knob: `VLLM_GLM53_MK_LOCALQ` = 0 (default, declared in `profiles/glm53.env`
so the launcher forwards it) / 1 = the launches the fp8-dense hook marks
background (`_mk_bg`: `mlp.shared_experts.*` -- by module name, which
assumes the runner keeps forking the shared expert onto its aux stream) /
2 = every one-unit-per-block launch (the bench's sweep). The lq launch
grid and the fewer-blocks control are the bench's only, through
`set_probe` -- no env surface. `mk_gemm_kernel` stays at 80 registers (one
kernel with both paths allocated for the union, 128), but its prologue's
quantizer changed with the shared helpers (SASS 3,360 -> 3,240 lines), so
the n=6416/4096 control rows are re-measured, not assumed. The
host's plan (`mk_gemm_plan_for`: `mk_units` <= grid on the phase's own
`mk_split_ok` gate, the lq kernel's resident grid) decides the kernel and
the bench's `gemm_plan` prints that same plan. Probes: `--gemm-sweep`
(local x split, bitwise `same`, replay), `--stamps` (phase stamps of a
`VLLM_GLM53_MK_PHASE_TS=1` build), the concurrent probe above.
## The v2 lane: the same GEMM as a non-persistent grid (2026-09-05, 30차)

The persistent kernel is right for a kernel that owns the GPU and wrong
for one that shares it. The 09-04 armed trace, read per launch position,
put 5.7 of the lane's 14.1 ms/step on ONE shape: the shared expert's down
[4096 x 512], 18 us alone and 135 us in the step. Its 48 x 69.6 KB blocks
cannot share an SM with the routed MoE kernel (90 KB, 48 blocks; the SM
has 102,400 B), so on the side stream they land one at a time as MoE
blocks retire and the publish barrier holds every landed block for the
last one -- which lands when MoE ends. deep_gemm (92 KB, also 1/SM) ran
the same GEMM inside the MoE tail because its blocks are independent
(09-01 stock trace: 36 us, all of it under the MoE kernel). On the main
stream the once-per-launch prologue and the static first-unit assignment
cost 15-35 us a launch (dense gate_up [6144 x 4096]: 119 us for 62 us of
bytes -- 48 tiles on 48 blocks, nothing to balance them).

`mk_gemm2_kernel` (`VLLM_GLM53_MK_GEMM2=1`, default off; the persistent
kernel stays as the kill-switched lane) is the same W4A8 GEMM as grid =
(n/128) x ksr independent blocks: no grid barrier, no shared A tiles
(each block quantizes the A k-blocks of its own slice from x in L2 --
same amax, same pow2 scale, same conversion, so the mma sees the bytes
the persistent kernel sees), the e2m1 expansion done in registers
straight into the mma B fragments with the same table lookup (`mk_w4x4`),
which frees the 32 KB of expanded tiles: **37 KB of smem, two blocks per
SM** (`__launch_bounds__(256, 2)`). One `__syncthreads` per k-block. ksr
> 1 slices assign an fp32 partial and the last slice to arrive folds the
tile in fixed order (deterministic, no zero pass). The slice rule
(`mk_choose_ksr2`, from the 30차 sweeps) takes ONE exact wave of the
device's resident slots (blocks/SM x SMs = 96: 48 tiles x 2, 32 x 3,
16 x 6), the longest slices that fit one wave when the tile count does
not divide it (8 x 8), and the finest slices (>= 4 k-blocks) above half
the slots, where a short second wave is the worst case (51 tiles x 2 =
102 units measured +30 us); `VLLM_GLM53_MK_KSR2` forces it for sweeps.
With that rule v2 matches or beats the persistent lane on every m <= 16
shape back to back (in_proj 79.4 vs 80.4 us, [4096x4096] 49.7 vs 56.3,
[4096x3072] 39.9 vs 46.1, shared expert 21.5/14.8 vs 22.5/15.4); m = 32
is level (38.9 vs 37.9 on [2048x4096]; the per-block x quant scales
with m). Under the routed MoE kernel the shared-expert pair exposes 18.6-
28.6 us a layer on v2 against 47.3-47.7 on the persistent lane (the
overlap probe, 30차 §6): -0.8 to -1.2 ms a step from that pair alone.

Round 3 (2026-09-05): the kernel is instantiated per m class (rows
quantized per warp 1 / 2 / 4 for m <= 8 / 16 / 32 -- the m-tiles and the
x lane mapping at compile time; the m = 32 quant is one 8-lane shuffle
chain over four rows instead of four 32-lane chains), and the mma's k
axis is a per-lane permutation of the k-block: lane q of a quad owns
natural elements [32q, 32q + 32) (W's raw chunk q, e2m1 groups 2q and
2q+1) and feeds the word (ks + q) & 3 of them at step ks, so a lane
builds two LUT pairs per W row per k-block instead of eight (216 of the
~670 instructions per warp per k-block on the SASS) and the quad's four
raw loads land on four different banks. ptxas: 96 / 96 / 117 registers,
no spills, no stack.

Tail units (`VLLM_GLM53_MK_KTAIL=<k-blocks>`, default off): every slice
gives its last k-blocks to a second unit placed at the end of the grid, so
the main units fill one wave and the tail units are dispatched into the
slots the fastest mains free -- the 4-9 us DRAM-arbitration tail of a
96-unit launch (30차 §6) is then work for blocks that would otherwise
idle. Each slice becomes two partials folded in fixed order; the rule
takes it only when every slice keeps at least `tail` k-blocks. Swept by
`--ktail2-sweep` (30차 §11): -5.6% on [1024x4096] and -2% on [6144x4096]
at tail=1, 5-17% worse on m=32, [4096x2048] and in_proj -- the second
partial and the short unit's ring fill cost more than the 4-9 us tail they
absorb. Stays off.

MK_SEG_SMLP2 (`VLLM_GLM53_MK_SMLP2`, default off): the shared-expert /
dense MLP as two PDL-chained v2 launches -- gate_up with the pair-
activation epilogue (the block storing the second final tile of a (gate,
up) pair computes the clamped SwiGLU over those 128 columns and emits
the e4m3 group + row scale), down on the a_ready path (it stages those
groups instead of quantizing x). No grid barrier and no 48-block
residency, unlike the persistent MK_SEG_SMLP launch, so it shares SMs
with the routed MoE kernel the way the standalone v2 lane does; it also
retires the act_and_mul launch and the down GEMM's own quant. The hook
in Glm5NextMLP.forward prefers it over MK_SMLP when both are armed.

Contract, inherited from the persistent lane: v2's partials and per-tile
arrival counters are one device-wide set, so a v2 launch must never
overlap another MK GEMM launch on a different stream (two launches
folding the same tile index would sum each other's slices). Serving
keeps it by stream order -- the side-stream pair is joined before the
next main-stream GEMM.

Gates and numbers: the boot self-test's exact-grid gate runs on whichever
lane the boot serves and the fingerprint names it (`lane v2, in_proj plan
on/ksr/units/bps=...`). The bench times both lanes side by side
(`--gemm2 both`: v2's output differs from the persistent lane's only by
summation order -- the round-3 lane k-permutation inside each mma step
and, on split shapes, the slice order -- and the exact gate is the gate;
`--ksr2-sweep 1,2,4,6,8`), the exact gate runs on both lanes at five
slice counts, and `probes/mk_gemm_moe_overlap_probe.py` measures the
shared-expert pair's exposure under the MoE kernel for both lanes.
MEASUREMENTS.md 30차 carries the trace analysis and the numbers.

## What it absorbs (and what it cannot)

The 2026-09-01 step decomposition fixes the budget: C=1 step = 66.0 ms, of
which **41 ms (62%) is expert weight reads at 91% bandwidth -- b12x tiles,
not touchable from here**. The remaining 25 ms is dense reads + kernel
launches + glue + AR, and the kernel census prices launches at 5.4 us each
(an upper bound that has been weakening). The megakernel attacks the launch
and glue component only; the bytes it reads are the same bytes.

| segment | replaces (per step, census counts) | stock | MK (planned) | **measured** (09-04 armed trace) |
|---|---|---|---|---|
| `MK_SEG_MHC` | mhc_fused + pre_big_fuse_with_norm | 179 | 45 | **89** + 7 stock leftovers · 2.54 -> 2.07 ms |
| `MK_SEG_GEMM` | per_token_group_quant + deepgemm, M<=32 | ~360 | ~180 | **187** (mk_gemm 185 + 2 lm_heads) · dense GEMM time flat, quant share -1.5 ms |
| `MK_SEG_KDA` | whole linear-attention block (~15 kernels x 34 layers) | ~510 | 34 | not armed (`MK_SEG_KDA=0`) |
| `MK_SEG_MLA` | sparse MLA decode (NoPE, fp8 KV) | 22 | 11 | default since 28차 -- decode +1.0%, prefill +15~18% |

Ceiling if all three hold: ~900 launches x 5.4 us = **4.9 ms (7.4% of the
step)**. The honest expectation is BELOW that (5.4 us is an upper bound,
and the MK kernels' own phases add barrier cost), which is why adoption is
bracket-only.

**What the armed trace says (2026-09-04, STEP_KERNEL_MAP.md supplement 5).**
The set removed **304 launches/step** (1,886 -> 1,582) and moved the repo's
share of the step's kernels from 7.6% to 25.4%. Step time moved in exactly
two places: MHC (-0.5 ms, two launches folded into one per site) and the
quantization the GEMM now does inside itself (-1.5 ms). The dense GEMM
region itself did not move -- 197 deep_gemm launches at 14.06 ms became 185
mk_gemm launches at 14.13 ms. **Launch count is not the currency; folded
work is.** The end-to-end verdict stays the ledger's (28차 §8), not this
README's. Prefill is untouched in v1
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
  per launch than one alone. Behind `VLLM_GLM53_MK_PDL=1` -- on in
  `profiles/glm53.env` and in `ab-glm53.sh`'s cand arm since 2026-09-04;
  until then serving had never carried it (the driver reads the env, only
  the bench probe set it), so every armed boot ran the lane without PDL.
  The captured-chain form is checked by `probes/mk_pdl_graph_check.py`.
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
| MK-MHC | `glm53_kernels/tilelang.py` (was `glm53_mhc_tilelang`) small-M branch | falls through to ONEPASS/stock pair, byte-identical |
| MK-GEMM | `glm53_model/glm53_fp8_dense.py` `Fp8DenseMethod.apply` + build | stock quant+deepgemm pair |
| MK-KDA | `glm53_model/glm5next_kda.py` (was `glm53_mk_kda_wiring`; image preimage `ec090aab...`) | stock forward body verbatim |
| MK-MLA | `glm53_model/flashinfer_mla_sparse_sm90.py` (was `glm53_mk_mla_wiring`; FlashInfer SM90 sparse backend) | the wrapper's own plan+run |
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
VLLM_GLM53_MEGAKERNEL=1     # master, default 1 since 28차 (MHC+GEMM+MLA on, KDA off)
VLLM_GLM53_MK_MHC=1         # per segment, each default 0
VLLM_GLM53_MK_GEMM=1
VLLM_GLM53_MK_KDA=1         # shadow first (state-index section below)
VLLM_GLM53_MK_KDA_SHADOW=1  # dual-run KDA eagerly, stock stays real
VLLM_GLM53_MK_PDL=1         # programmatic dependent launches, default 0
VLLM_GLM53_MK_LOCALQ=1      # gemm: the local-quant kernel for the shared
                            # expert's pair (bg), default 0; 2 = all small
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

## Correctness review (2026-09-05)

The MHC deferred-publication branch now synchronizes before reusing its
shared reduction tile. LOCALQ's two-buffer ring uses the split-relative
k-block index, including odd split starts. KDA shifts the complete retained
conv window for short verify requests, rather than repeating its third
retained value. These changes require fresh GPU numerical and racecheck
validation; CPU control-flow regressions are not GPU proof.

Boot error comparisons reject nonfinite outputs, references and overflowed
norms. The GEMM exact gate explicitly selects v1 global/local and v2, checks
v2's M=8/16/32 classes and the LOCALQ KSR=3 odd-start case, then restores all
seven probe overrides even after failure. KDA adds nq=1/4/6/8 cases, including
accepted=8 with nq=1, and compares every written recurrent position.

An MLA real-KV shadow failure now raises a persistent worker error: changing
a Python arm flag cannot remove a kernel from an existing CUDA graph. Restart
with `VLLM_GLM53_MK_MLA=0` after such a failure. KDA shadow mode takes precedence
when both shadow and takeover are enabled; capture continues to use stock.

`tests/test_logic.py` includes the behavioral regressions. To run the actual
boot gates before the offline benchmark in an idle fleet window:

```bash
bash probes/run_megakernel_bench.sh --segments exact,mhc,kda --boot-gates --iters 5
bash probes/run_mk_probe.sh probes/mk_smlp2_concurrent_probe.py --rounds 10 --replays 5
```

The second probe checks SMLP2 versus v2 plus the native CUDA activation in
seven graphs (alone and both MoE issue orders), validates every captured
output and replay before timing, and balances forward/reverse timing order.
Missing native activation is a failed comparison, never a slower Torch
fallback. Its per-layer exposure and x42 projection are not a serving gain.

## Verification ladder (in order, no skips)

1. `python3 tests/test_logic.py` -- pure logic + .cu/.py geometry parity +
   manifest invariants (this repo, no GPU).
2. `bash probes/run_megakernel_bench.sh [--profile glm53|dsv4]` in a fresh
   container (srv4, never the serving one; the wrapper binds the profile's
   composed overlay at that image's real paths, and passes its package
   root as `MK_PKG_PATH`): numerics vs stock (rel gates 1e-3 MHC / 0.15 GEMM
   by-design + 1e-3 exact-grid with no element over one bf16 ULP,
   2e-2 KDA outputs and states on grid-snapped
   weights) + CUDA-event timing per segment + a replay-stability
   check (re-launch drift <= 1e-6 over the shared workspace -- the
   monotonic-barrier contract). `!` marks any cell over gate; a `!` cell
   disqualifies that shape.
3. `VLLM_GLM53_MK_KDA_SHADOW=1` bench boot: shadow log clean for a full
   bench-tp4 pass.
4. EXP bracket (RUNBOOK_KERNEL_CAMPAIGN2.md): base -> cand -> base on
   C=1 step/s, quality 9/9, Korean 0/16, pos-1 acceptance within 2 pct.

Rollback is one env line per segment; unmounted hooks are inert imports.

## 33차 — W4 pack accuracy levers (same bytes, same kernel)

The operator's brief: raise the GEMM lane's accuracy without touching its
speed. Four levers, three of them pack-time; the served bytes per weight
and the inner loop are unchanged.

1. **Exact activation scale** (unconditional, kernel). The prologue's
   per-row, per-128-k scale is `amax * fp32(1/448)` instead of the pow2
   `2^frexp_exp(amax/448)`, which wasted up to one of e4m3's three mantissa
   bits on every activation row. The torch twin `_mk_quant_x_ref` does the
   same three fp32 ops in the same FORM (torch divides by a scalar as a
   multiply by its fp32 reciprocal; an IEEE divide in the kernel failed the
   exact gate at 4e-3, over-ulp 1526). Every activation quantizer feeding
   a W4 pack -- the GEMM prologues (global, local-quant, v2), the SMLP pair
   emitters, KDA p4 -- uses it.
2. **Per-row shift** (`VLLM_GLM53_MK_PACK_ROWSHIFT=1`, default). The shift
   that keeps the group exponents inside the byte-add expansion's 11
   octaves is per output row (median of the row's covering exponents),
   undone on the output column at the bf16 store (`MKPack.rgs`, fp32
   [n_pad]) instead of on the activation scales. The 1.5% of clamped
   groups on the production [6416, 4096] in_proj were rows octaves away
   from the tensor median; they no longer clamp.
3. **GPTQ error-feedback rounding** (`VLLM_GLM53_MK_PACK_GPTQ=1`, default,
   effective only with a calibration dump). A calibration boot
   (`VLLM_GLM53_MK_CALIB=1`; drive it with `bench/mk-calib-run.py`) makes
   every W4 launch's Python entry accumulate `H = sum x x^T` per pack until
   `VLLM_GLM53_MK_CALIB_TOKENS` rows, then each rank dumps
   `<VLLM_GLM53_MK_CALIB_DIR>/rank<r>/<linear>.pt`. The next boot's packer
   finds the dump by the linear's name and rounds each column with the
   error of the previous columns fed forward through the inverse Hessian
   (OBQ/GPTQ, blocks of 128, group scales re-derived on the updated
   weights). The pack is cached under `VLLM_GLM53_MK_PACK_CACHE` (weight
   md5 + shape + levers + `MK_PACK_VERSION`), so the solve is paid once per
   rank.
4. **Low-rank error correction** (`VLLM_GLM53_MK_PACK_LORC=r`, default 0;
   8..32 in eights). `E = W - deq(Q)`; with `S` = rms of each input channel
   from the Hessian, the SVD of `E S` gives `A = U_r S_r`, `B = V_r^T S^-1`
   and the served product is `x W_q^T + (x B^T) A^T` (the correction on the
   unquantized x; `mk_pack_twin` is the reference). The v2 kernel's `LR`
   instantiation adds it -- the plain instantiation carries none of the
   code: `LR_CTAS` extra blocks at the front of the grid reduce `t = x B^T`
   into a per-stream scratch and raise a flag (they are the lowest block
   indices, dispatched first; on the stamps they publish by 10-15 us); a
   tile's final store waits for the flag, stages t and the tile's 128 rows
   of A in the freed smem rings, computes the correction tile once, adds it
   per store, and the launch's last final store rearms the scratch (graph
   replay reuses it). v1 / KDA / SMLP lanes serve such a pack WITHOUT the
   correction (said once on stderr).

Gates: the exact fixture (weights on the grid: the correction is zero, the
per-row shift is byte-exact, the twin is the reference) and the by-design
fixture, both in `_selftest_gemm`; `probes/megakernel_glm53_bench.py
--segments exact,gemm`; and `probes/mk_pack_accuracy.py`, which ranks the
arms against the bf16 truth (the bench's rel_err is the gap to the stock
fp8 pair, two quantized arms disagreeing, and cannot rank them).

## The vocab head on the v2 lane (30차 §13, 2026-09-06)

`VLLM_GLM53_MK_HEAD_DRAFT` / `VLLM_GLM53_MK_HEAD_TARGET` (both default 0). The
served head is fp8 (`VLLM_TARGET_LM_HEAD_FP8`, `VLLM_SPEC_FP8_LM_HEAD`:
deep_gemm W8A8 -- the `sm120_fp8_fp4` kernel name is deep_gemm's unified sm120
kernel with every FP4 flag false), 158 MB/rank at ~190 GB/s = 836 us, twice a
step (target verify m=8, draft candidates m=7): already at the DRAM floor for
fp8 bytes. The W4 pack halves the bytes (the v1 lane measured 418 us on the
same shape, 33차; v2 is 303 tiles = 3.2 waves, expected ~420 us).

Wiring: `fp8_lm_head.Fp8HeadLogitsProcessor._apply_head` asks `head_logits()`
first (None = its fp8/bf16 path, unchanged); `head_pack()` builds the pack
once per head object on the first call (pack cache; the target and the
drafter alias one head, so one pack, two independent endpoint gates); the
first eligible call per endpoint -- the eager warm-up, before capture -- is
checked against the pack's twin on the first and the last eight tiles (the
padded tail), a miss disarms that endpoint for the boot, loudly; then the
serving proof line, and the processor logs the W4-vs-fp8 argmax agreement of
that call (`fp8 lm_head: W4 head lane ... argmax agree N/N rows`). Needs
MK_GEMM=1 (packs, exact gate) and MK_GEMM2=1 (the persistent kernel was never
sized for 303 tiles; with GEMM2 off the head disarms itself).

Kernel side: `MK2_TILES_MAX` 64 -> 320; the ksr rule takes ONE slice when
the tiles alone fill two or more waves (a split adds 2.5 MB of fp32 partials
per slice at n = 38,784 plus the fold -- the head sweep decides); the partial
bound is a split's contract only (ksr 1, tail 0 stores bf16 straight from
the accumulators, so m = 32 on the head serves whole).

Gates, in order: bench `--segments gemm --gemm-shapes 8:38720:4096,7:38720:4096
--gemm2 both --ksr2-sweep 1,2` (exact PASS; the stock column IS the served fp8
head), then a DRAFT-head bracket (acceptance-normalised step/s, pos-1 within
2 pct, quality 9/9, Korean 0/16) -- a coarser draft head only moves
acceptance. TARGET is the served logits, the decision `VLLM_TARGET_LM_HEAD_FP8`
documents one notch coarser: operator decision only.
