# dsv4_mhc_tilelang

Hyper-connection tilelang kernels, swept for this fleet's decode shapes. Tied to the hc_mult architecture, not to the checkpoint.

## MK_SEG_MHC hook (2026-09-03, disarmed)

The small-M branch now also offers the fused pair to `glm53_megakernel`'s
`MK_SEG_MHC` -- one persistent 48-block nvcc launch instead of
`mhc_fused` + `mhc_pre_big_fuse_with_norm`. The same ~45-line branch GLM-5.3
carries, at the same place (after the tile heuristic, before the `gemm_out`
allocations), because the two images' wrappers have identical signatures and
the two models have identical MHC geometry:

| | DeepSeek-V4-Flash | GLM-5.3-Flash |
|---|---|---|
| `hc_mult` | 4 | 4 |
| `hc_sinkhorn_iters` | 20 | 20 |
| `hc_eps` | 1e-06 | 1e-06 |
| `hidden_size` | 4096 | 4096 |

The MK gate is that geometry plus `T <= 32`, and nothing else -- no model
class, no image path. **It is off**: the kernel arms only under
`VLLM_GLM53_MEGAKERNEL=1 VLLM_GLM53_MK_MHC=1` (both default 0 in
`profiles/dsv4.env`) and only after a boot self-test diffs it against the
stock pair in this file. An unmounted core, an unset knob, a failed
self-test or an ineligible shape all fall through to the swept stock path,
byte-identical to today.

Two things this hook does NOT claim:

- **The window is small.** At `SPEC_TOKENS=5` a step carries `C x 6` tokens,
  so only `C <= 5` reaches the gate; production `MAX_NUM_SEQS=32` (M=192)
  never does. This is a low-concurrency lever, which is also where the
  fleet's adoption bracket (C=1/2/4) lives.
- **Nothing is measured on this model yet.** GLM's numbers (T=8 27.4 us,
  T=32 42.0 us vs stock 32.8 / 71.6) came from an UNSWEPT stock pair; this
  lane's pair is already swept (R1/R2/R3, per-call 15.6 -> 13.1 us), so the
  delta here has to be measured before anything is claimed. Ladder:
  `probes/run_megakernel_bench.sh` in a fresh container on this image, then
  the ordinary bracket. This is production -- the MK pair is a reduce-order
  change (rel 1e-3 class), not bit-exact.
