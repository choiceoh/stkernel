# glm53_kda_onepass

Two opt-in Triton paths for the **stock** (non-megakernel) KDA layer of
GLM-5.3-Flash, part of launch-count bundle 2 (RUNBOOK EXP-20). Both default
to the stock chain; both are profile-declared keys, so arm them as caller
env, never through `EXTRA_ENV`:

```bash
VLLM_GLM53_KDA_DUAL_GEMM=1 VLLM_GLM53_KDA_ONEPASS=1 bash launchers/start-glm53-nvfp4-tp4.sh
```

> **When the one-pass serves.** It runs on a pure spec-verify step
> (`use_spec`, no prefill, no non-spec decode row) -- exactly the shape the
> uniform decode CUDA graph is captured at: the image's
> `mamba_hybrid.prepare_attn` fills `num_decode_draft_tokens_cpu` per row on
> real steps (drafts > 0 and exactly one non-draft token scheduled), and
> `GDNAttentionMetadataBuilder.build_for_cudagraph_capture` derives it from
> `query_start_loc`. A mixed step (a request on its first decode step is a
> `-1` row) takes the stock chain for that step. Served decode is a FULL
> graph replay, so the Python branch runs at capture: the proof is the
> capture-time `[kda-onepass] one-pass KDA serving` line and the trace, not
> a per-step log (MEASUREMENTS 32차 §12). An earlier note here that the knob
> "does nothing on this runner" rested on the chain-9 hypothesis that the V2
> runner never builds spec masks; 32차 §11's 04:50 correction and KDAPROOF3
> (`kda lane CAPTURED ... n_spec=4`) retracted it.

Checkpoint shape this was written for (`glm53-redhat-nvfp4`, TP=4):
`linear_num_heads 64 -> 16 local`, `head_dim 128`, `short_conv_kernel_size 4`,
verify block `SPEC_K 7 -> 8` tokens, so the merged `in_proj_qkvbfg_a` row is
`q|k|v (3 x 2048) | beta (16) | f_a (128) | g_a (128) = 6416` columns and the
conv state holds `3 + 7 = 10` slots per line. The model overlay
(`glm53_mk_kda_wiring/glm5next_kda.py`) only imports this module when a knob
is armed (the exact string `1`; the profile's `0` costs no import), calls `resolve()` once and then `gate_gemms()` / `spec_onepass()`;
every knob, guard, self-test, log line and counter lives here.

## `resolve()` -- knobs, counters, boot self-test

On the first eager forward (the profile run, never under CUDA-graph capture)
`resolve()` reads the two knobs, allocates the kernel's arrival counters and,
when the one-pass is asked for, runs `selftest()`: the stock three-kernel
chain and the one-pass on three fleet-shape fixtures (acceptance 1/3/8, a
varlen block, bf16 and fp32 conv states), compared bit for bit on conv
state, recurrent state and output. A mismatch (or an error) DISARMs the
one-pass for the boot -- `[kda-onepass] self-test FAIL (...) -> one-pass
DISARMED` -- because a Triton or image bump can change the layout assignment
and with it a reduction tree (see the trap below), and that must not serve
silently. `selftest`, `make_fixture` and `run_stock_chain` are the same code
the probe runs, so the boot gate and the offline gate cannot drift apart.

## `VLLM_GLM53_KDA_DUAL_GEMM` -- f_b + g_b in one launch

`f_a` and `g_a` are adjacent 128-column slices of the same merged projection
row, and `f_b_proj` / `g_b_proj` are both `[2048, 128]` bf16. The stock
layer issues two cutlass wmma GEMMs per layer (grid `(8,16,1)`, 4.7-6 us
each in the 2026-09-04 trace, 68 launches/step). One Triton program per
output tile loads both input slices once and runs two `tl.dot`s over 16-row
tiles, which the probe found bit-identical to cuBLAS' 16x16 wmma
accumulation for M <= 16; at M = 32 cuBLAS picks a different kernel and a
handful of elements differ by 1 ulp of bf16 (treat the class as **not
bit-exact** there). Decode M only (M <= 32); prefill keeps the stock GEMMs
by design and is not announced.

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
  **rounded to bf16** where the stock chain stores it; the rolled conv
  state (`IS_SPEC_DECODING` roll: slots `[0, 2)` from `old[acc + j]`, then
  the block's tokens). The taps follow the conv state's dtype (bf16, or
  fp32 under `MAMBA_CACHE_DTYPE=float32`, the dtype MK-KDA needs) exactly
  as the stock kernel does;
- recurrence: `IS_KDA` + `COMPUTE_GATE` (bounded gate
  `lb / (1 + exp(-exp(A) * (g + bias)))`) + `SIGMOID_BETA` +
  `USE_QK_L2NORM_IN_KERNEL`, state resumed from slot `[n, acc-1]`, stored to
  slot `[n, t]` after each token;
- gated RMSNorm: `x * rsqrt(mean(x^2) + eps) * w * sigmoid(g)`, sigmoid
  activation, `eps = 1e-5`, one 128-wide row at a time;
- the stock kernels' skip semantics stay separate: the conv part runs iff
  the request's conv line (spec slot 0) is not the null block, the
  recurrence iff its resume slot is valid, and a skipped conv leaves the
  recurrence reading the raw projection rows like the stock chain.

So the recurrent state, the conv state and the pre-norm output are
bit-identical to the stock chain by construction, and the probe found the
served output bit-identical too on all its cases (both cache dtypes and
layouts, acceptance 1/3/5/8, varlen, a two-step chain). The norm's reduction
tree is still not the stock kernel's `[32, 128]` 4-warp tile, so that class
is declared **not bit-exact** (bf16 reduce-order) and the bundle boot carries
the standard numerics gates: quality 9/9, Korean 0/16, acceptance profile
within 2 pct.

**Layout trap (2026-09-04/05)**: the token loop must keep the stock recurrent
kernel's exact shape -- every operand loaded inside the iteration -- and the
norm tail must stay 1-D. A prefetched or hoisted `[K]` operand, or a
2-D (rows x 128) norm tile, changed Triton's layout choice for the whole kernel
and with it the l2norm reduction tree: the recurrent state came out 1 ulp
off the stock chain on ~3% of elements. The 3-5 us/layer those bought are
not worth losing the bit-exact state; the self-test is the tripwire.

Two cross-program dependencies that separate launches used to provide are
last-arriver counters (one `atom.acq_rel.gpu` per program, no spinning, no
co-residency assumption): a head's conv state is rewritten by whichever of
its V-block programs arrives last after every program has read the old
history (each program needs all 128 q/k channels of the head, so the write
cannot be split by channel), and the same head's norm is applied in place by
the last program to finish its recurrence. The counters are monotonic: the
last arriver is the program that brings the count to a multiple of NV (a
power of two), so nothing resets them and an int32 wrap is harmless; a batch
beyond the buffer (256 requests x 16 heads) is declined, not raised. Because
the test is `count % NV`, a counter must only ever see launches of one NV:
each block width owns its own buffer, `resolve()` allocates the serving
width's (BV=8, NV=16) before any capture, and a width missing under capture
declines to the stock chain instead of allocating.

## Offline gate

`bash probes/run_micro_fusion_check.sh` (fresh container, composed
overlay mounted) checks, on the fleet shapes: dual GEMM vs the two
`F.linear` calls with an N-tile sweep, the module self-test, the one-pass
vs the stock chain (both cache dtypes and layouts, acceptance 1/3/5/8,
varlen, a second launch on the same counters, a two-step chain, the
applicability guard's admit/decline cases), the kpool update, and times
each pair as CUDA-graph replays of a decode step's worth of layers with
distinct weights and states -- reporting device time (second of two
back-to-back replays) and launch-inclusive time separately. Numbers live
in MEASUREMENTS (31차); this README does not repeat them.

## Log anchors

`[kda-onepass] self-test PASS (...) -> one-pass ARMED` and `resolve:
dual=... onepass=...` once per boot; `[kda-onepass] dual gate GEMM serving:
...` and `[kda-onepass] one-pass KDA serving: ...` once when the paths run.
Warnings: `... decode shape not admitted -> stock two GEMMs`, `... tensors
not admitted -> stock chain`, `self-test FAIL ... DISARMED`, and `a knob is
set but ... is not mounted` when the module is missing -- any of these means
that axis is not serving. Prefill (M > 32) keeps the stock GEMMs by design
and logs nothing. In a trace the seven per-layer launches become one
`_kda_onepass_spec_kernel` and the two wmma GEMMs one
`_dual_gate_gemm_kernel`.

## Not in this module

MK_SEG_KDA (`VLLM_GLM53_MK_KDA`) replaces the whole block including both
GEMMs; when it serves, the layer takes the megakernel branch before this
module is consulted. The prefill chunk path and mixed prefill/decode steps
keep the stock kernels.
