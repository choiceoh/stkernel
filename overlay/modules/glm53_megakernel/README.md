# glm53_megakernel

The GLM-5.3-Flash decode megakernel for GB10 (sm_121a, 48 SM, TP=4): ONE
persistent 48-block CUDA launch per inter-collective segment of the decode
step, compiled AOT by nvcc (`-arch=sm_121a`) instead of JIT.

## The W4 arm: the ledger's last unpulled lever, done exactly

**2026-09-02: it is faster than the stock W8 pair on every decode shape**
(srv2, two different weights back to back, PDL on): 112/82/52/35 us vs
151/115/65/40 for n=6416/4096/2048/1024. What got it there was making the
W4 loop stop being compute-bound -- register-only expansion on the LUT's
closed form (byte-lane SIMD), a swizzled A tile, the A staging load
hoisted out of the sync window and the second m-tile compiled away for
m <= 16. The arm still changes served numerics, so arming it is the
quality bracket's call, not this module's.

PR #192's own closing line names dense W8A8 -> W4 as the only remaining
sizable lever (est. -3.7 ms/step; the W8A8 dense read floor is ~2 GB/step
= ~8.7 ms). `VLLM_GLM53_MK_W4=1` arms it WITHOUT new tensor-core PTX:

weights are stored as e2m1 nibbles x one pow2 scale per 16 elements
(0.56x the fp8 bytes -- the W4 arm is *cheaper* in memory than the W8
MK arm), and the kernel expands every nibble to an **exact e4m3 byte** --
e2m1's 1-bit mantissas always fit, and the 2^s product is an exponent-field
add -- then runs the same e4m3 mma pipeline. Activations stay fp8 (the
W4A4's 20-25% loss is the thing the literature warns about; QServe's
compromise axis). Two gates at boot: bit-exact expansion (1e-5, against
weights built ON the e2m1 grid -- anything above is a layout bug) and
by-design error (<= 0.15 vs the stock pair). The ceiling of the encoding
is documented: s is clamped to [-5, 6], so |w| > 384 saturates (never
NaN); real projection weights are orders of magnitude below.

This arm CHANGES SERVED NUMERICS: one numerics axis per boot, bracket with
quality 9/9 + Korean 0/16 + pos-1 acceptance within 2 pct or it reverts.
The KDA in_proj stays fp8 (`all` includes it) -- its output feeds the
recurrence, where error accumulates in the state instead of washing out.

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
  (the only bandwidth-heavy operand) stages through a **3-buffer cp.async
  pipeline** -- a synchronous load->sync->mma chain leaves ~20% of the
  stream idle, ~1.3 ms/step at the ~2 GB/step W8A8 dense footprint, so
  2-in-flight is the difference between matching deepgemm and losing to it.
  Depths 4 and 5 measured no gain (bytes in flight are not the limiter
  once the wait is exact); the W4 arm streams its raw (tile, k-block)
  records through its own 3-stage pipeline and expands them in registers.
  TMA stays a drop-in for later.
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
- Dynamic smem: W8 kernel 58,368 B (dense 128 B rows, pre-swizzled pack, 1 KB alignment slack) (the 3 W pipeline buffers), W4 kernel
  72,832 B (2 expanded tiles + 3 raw stages) -- separate instantiations
  with separate budgets, because one shared budget cost the W8 loop 4-7%.
  Both resolve to 1 block/SM. Deeper W4 staging (4, 5) measured worse.

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
VLLM_GLM53_MK_KDA=1         # see the open item below before arming this
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

## Open item before MK-KDA serves: the state-index contract

`fused_recurrent_kda` receives `spec_state_indices_tensor` shaped
`[n_spec, max_query_len]`; this kernel treats `[r, 0]` as THE state slot of
request r for both conv and recurrent state (read at entry, written at the
accepted boundary). If the stock kernel actually addresses per-position
slots, eager **shadow mode** (`VLLM_GLM53_MK_KDA_SHADOW=1`) is the gate
that settles it: both arms run, outputs AND the states the next step reads
are diffed every 64 calls and on drift, graph replay stays stock. Run shadow
on a bench boot, read the log, then decide the arm.

Shadow is self-sufficient: the KDA packs build lazily per layer from the
bf16 source weights (cached, eager-only), so a shadow boot does not need
MK-GEMM armed. It does require `VLLM_GLM53_FP8_DENSE=1` so the stock arm of
every comparison runs the same W8A8 axis — against bf16 stock the 2e-2 gate
could not tell a broken kernel from quantization noise.

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

MK-GEMM duplicates the fp8 weight bytes per linear in its own layout (the
deepgemm pair must stay as the M>32 path), i.e. ~the W8A8 footprint again,
~+4 GB/rank at full coverage. At GMU 0.73 that may not fit; the first
MK-GEMM boot should watch the KV-cache line and be ready to drop GMU a
notch (memfree-preflight computes it). If it does not fit, arming MK-GEMM
only for the KDA in/out projections is the fallback scope (a one-line
change in the build attach).

## Verification ladder (in order, no skips)

1. `python3 tests/test_logic.py` -- pure logic + .cu/.py geometry parity +
   manifest invariants (this repo, no GPU).
2. `bash probes/run_megakernel_bench.sh` in a fresh container (srv4, never
   the serving one; the wrapper binds the composed overlay at its real
   image paths): numerics vs stock (rel gates 1e-3 MHC / 2e-2 GEMM, KDA
   outputs and states) + CUDA-event timing per segment + a replay-stability
   check (re-launch drift <= 1e-6 over the shared workspace -- the
   monotonic-barrier contract). `!` marks any cell over gate; a `!` cell
   disqualifies that shape.
3. `VLLM_GLM53_MK_KDA_SHADOW=1` bench boot: shadow log clean for a full
   bench-tp4 pass.
4. EXP bracket (RUNBOOK_KERNEL_CAMPAIGN2.md): base -> cand -> base on
   C=1 step/s, quality 9/9, Korean 0/16, pos-1 acceptance within 2 pct.

Rollback is one env line per segment; unmounted hooks are inert imports.
