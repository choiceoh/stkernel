# glm53_kda_onepass

Two opt-in Triton paths for the **stock** (non-megakernel) KDA layer of
GLM-5.3-Flash, part of launch-count bundle 2 (RUNBOOK EXP-20). Both default
to the stock chain; both are profile-declared keys, so arm them as caller
env, never through `EXTRA_ENV`:

```bash
VLLM_GLM53_KDA_DUAL_GEMM=1 VLLM_GLM53_KDA_ONEPASS=1 bash launchers/start-glm53-nvfp4-tp4.sh
```

Checkpoint shape this was written for (`glm53-redhat-nvfp4`, TP=4):
`linear_num_heads 64 -> 16 local`, `head_dim 128`, `short_conv_kernel_size 4`,
verify block `SPEC_K 7 -> 8` tokens, so the merged `in_proj_qkvbfg_a` row is
`q|k|v (3 x 2048) | beta (16) | f_a (128) | g_a (128) = 6416` columns and the
conv state holds `3 + 7 = 10` slots per line. Every guard reads the real
tensors; a mismatch is announced once (`[kda-onepass] ... -> stock`) and the
stock chain runs.

## `VLLM_GLM53_KDA_DUAL_GEMM` -- f_b + g_b in one launch

`f_a` and `g_a` are adjacent 128-column slices of the same merged projection
row, and `f_b_proj` / `g_b_proj` are both `[2048, 128]` bf16. The stock
layer issues two cutlass wmma GEMMs per layer (grid `(8,16,1)`, 4.7-6 us
each in the 2026-09-04 trace, 68 launches/step). One Triton program per
128-column output tile loads both input slices once and runs two `tl.dot`s.
Numerics: fp32 accumulation over K=128 in both, bf16 output; the probe
reports whether cuBLAS' accumulation order is matched bit for bit -- treat
it as **not bit-exact** unless the probe says `bit-exact` for the shape.
M <= 32 only (decode); prefill keeps the stock GEMMs.

## `VLLM_GLM53_KDA_ONEPASS` -- conv + recurrence + gated norm in one launch

For a **pure spec-verify decode step** (every request is a draft-verify
block: `num_prefills == 0`, `num_decodes == 0`, no non-spec tokens) the
stock chain per KDA layer is seven launches: `_causal_conv1d_update_kernel`,
three `elementwise` copies (`q/k/v.contiguous()` of the conv output views)
plus the beta copy, `fused_recurrent_gated_delta_rule_fwd_kernel`,
`layer_norm_gated_fwd_kernel` -- 238 launches/step over the 34 layers. The
one-pass kernel reads q/k/v/beta straight from the merged projection rows,
runs the conv taps in registers, the recurrence, and the gated RMSNorm, in
the stock recurrent grid `(V/BV, N*H)` of single-warp programs.

What is transcribed, op for op, from the image's Triton kernels:

- conv: `acc += x_j * w_j` over the 4 taps (3 history slots at
  `num_accepted_tokens - 1`, then the token), `acc / (1 + exp(-acc))`,
  **rounded to bf16** where the stock kernel stores into the bf16 qkv
  buffer; the rolled conv state (`IS_SPEC_DECODING` roll: slots `[0, 2)`
  from `old[acc + j]`, then the block's tokens);
- recurrence: `IS_KDA` + `COMPUTE_GATE` (bounded gate
  `lb / (1 + exp(-exp(A) * (g + bias)))`) + `SIGMOID_BETA` +
  `USE_QK_L2NORM_IN_KERNEL`, state resumed from slot `[n, acc-1]`, stored to
  slot `[n, t]` after each token;
- gated RMSNorm: `x * rsqrt(mean(x^2) + eps) * w * sigmoid(g)`, sigmoid
  activation, `eps = 1e-5`.

So the recurrent state, the conv state and the pre-norm output are
bit-identical to the stock chain by construction (the probe asserts the two
states bit for bit). The norm's 128-wide sum of squares is reduced by a
single-warp `[8, 128]` tile instead of the stock `[32, 128]` 4-warp tile;
the probe found the served output bit-identical too on the fleet shapes
(0 bf16 mismatches over the six cases), but the tree is not the stock
kernel's, so the class is declared **not bit-exact** (bf16 reduce-order)
and the bundle boot carries the standard numerics gates: quality 9/9,
Korean 0/16, acceptance profile within 2 pct.

**Layout trap (2026-09-04)**: the token loop must keep the stock recurrent
kernel's exact shape -- every operand loaded inside the iteration. A
prefetched or hoisted `[K]` operand (loop-carried) changed Triton's layout
choice for the loop's vectors and with it the l2norm reduction tree: the
recurrent state came out 1 ulp off the stock chain on ~3% of elements. The
3 us/layer that prefetching bought is not worth losing the bit-exact state.

Two cross-program dependencies that separate launches used to provide are
last-arriver counters (one `atom.acq_rel.gpu` per program, no spinning, no
co-residency assumption): a head's conv state is rewritten by whichever of
its V-block programs arrives last after every program has read the old
history (each program needs all 128 q/k channels of the head, so the write
cannot be split by channel), and the same head's norm is applied in place by
the last program to finish its recurrence. The counters are a module buffer
the kernel zeroes behind itself.

## Offline gate

`bash probes/run_micro_fusion_check.sh` (fresh container, composed
overlay mounted) checks, on the fleet shapes: dual GEMM vs the two
`F.linear` calls, the one-pass kernel vs the stock three-kernel chain
(acceptance 1/3/8, both conv-state layouts, uniform and varlen blocks:
states bit-exact, output rel err and bf16 mismatch count), and times both
chains as CUDA-graph replays of 34 layers with distinct weights and states.
Numbers live in MEASUREMENTS (31차); this README does not repeat them.

## Log anchors

`[kda-onepass] dual gate GEMM serving: ...` and
`[kda-onepass] one-pass KDA serving: ...` once per boot when the paths run;
`[kda-onepass] ... -> stock` when a knob asked and the tensors were not
admitted; `[kda-onepass] knob set ... but ... is not mounted` when the
module is missing. In a trace the seven per-layer launches become one
`_kda_onepass_spec_kernel` and the two wmma GEMMs one
`_dual_gate_gemm_kernel`.

## Not in this module

MK_SEG_KDA (`VLLM_GLM53_MK_KDA`) replaces the whole block including both
GEMMs; when it serves, these knobs are inert (the layer takes the megakernel
branch first). The prefill chunk path and mixed prefill/decode steps keep
the stock kernels.
