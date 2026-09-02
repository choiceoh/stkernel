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
| MK-KDA | this module's `kda.py` overlay (image preimage `ec090aab...` in manifest) | stock forward body verbatim |

The kda.py copy came from the image (`/tmp/deployed/kda.py`, extracted
2026-08-31). If a future image bumps that file, deploy-overlays' preimage
gate catches it before a boot lies about what it is running.

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

First arm compiles the extension in-container (`/root/.mk_build`, ~a
minute, same pattern as tp_oneshot_ar) and logs the source md5 + kernel
count fingerprint.

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

## Verification ladder (in order, no skips)

1. `python3 tests/test_logic.py` -- pure logic + .cu/.py geometry parity +
   manifest invariants (this repo, no GPU).
2. `bash probes/run_megakernel_bench.sh` in a fresh container (srv4, never
   the serving one; the wrapper binds the composed overlay at its real
   image paths): numerics vs stock (rel gates 1e-3 MHC / 0.15 GEMM
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
