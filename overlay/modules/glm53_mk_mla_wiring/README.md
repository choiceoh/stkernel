# glm53_mk_mla_wiring

Routes the sparse MLA decode of GLM-5.3 (NoPE, fp8 KV) to `MK_SEG_MLA`, the
megakernel's own sm_121a kernel, behind `VLLM_GLM53_MK_MLA=1` (master
`VLLM_GLM53_MEGAKERNEL=1`). The kernel lives in `glm53_megakernel`; this
module is the image-bound hook: an overlay of
`vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py` (the FA2
wrapper backend vLLM selects on sm_12x) with two hunks:

* `forward_mqa`: call `mla_decode` on the same `topk_slots` /
  `valid_counts` / fp8 latent bytes the wrapper would have used, and return.
  Decode (T <= 64) uses the split + log-sum-exp path; **prefill (T > 64) runs
  at splits == 1**, where the kernel normalises in place and needs no fp32
  partial scratch. The builder's per-step `plan()` still runs -- it is on the
  builder, not this call -- so the wrapper stays a working fallback.
* a one-shot shadow on the first eager call: kernel vs wrapper on the same
  bytes, `rel > 2e-2` or a non-finite output DISARMs the segment for the boot
  (`[megakernel] mla shadow vs wrapper rel=... -> ARMED for decode` is the
  arming line to look for).

Numerics: bf16 q, bf16 latent (e4m3 -> bf16 is lossless), bf16 P, fp32
accumulation -- the same class as the FA2 path (rel 3.4e-3 between the two,
both ~3e-3 from an fp32 reference). A served-numerics change: quality 9/9,
Korean 0/16, pos-1 acceptance +/-2pct, C=1 step/s bracket.

Isolated (srv4, W=2048, L2-cold), per layer:

| T | MK_SEG_MLA | FlashInfer run |
|---|---|---|
| 8 (C=1) | 67.7 us | 124.3 us |
| 16 | 96.7 us | 199.8 us |
| 32 | 212.8 us | 547.2 us |
| 8192 (prefill) | 31.6 ms @ 204 GB/s | ~76 ms/layer-chunk |

Decode: 11 layers x (124.3 - 67.7) = 0.62 ms/step at C=1. Prefill: attention
is 25% of a 32K prefill (4.19 s of 16.9 s) at 113 GB/s; this kernel does the
same gather at 204 GB/s, 91% of the part's 225 GB/s scattered ceiling. Pass
the knobs as caller env:

```bash
VLLM_GLM53_MEGAKERNEL=1 VLLM_GLM53_MK_MLA=1 bash launchers/start-glm53-nvfp4-tp4.sh
```
