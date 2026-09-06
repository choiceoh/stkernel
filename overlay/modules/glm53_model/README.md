# glm53_model

GLM-5.3 모델 배선 — 모델·어텐션·KDA·MLA 파일 접수, 밀집 GEMM fp8/W4 패스, KDA 원패스.

2026-09-05 (34차, 운영자 "디폴트화된 모듈들을 4~5개씩 하나로 묶어라") 에 아래 모듈들을 이 디렉터리 하나로 합쳤다. **매니페스트 행·베이스 계약·소스 파일·노브·기본값은 그대로**이고 디렉터리와 `manifest.tsv`·`requires`·README 만 합쳐졌다(합성 결과 `build/glm53/` 의 파일은 바이트 동일(메가커널 .cu 주석의 경로 한 줄 제외), 행 순서만 바뀜). 옛 이름은 원장·런북·커밋에 그대로 남아 있고, 아래 절이 옛 모듈 하나씩이다.

| 옛 모듈 | 파일 | 무엇 |
|---|---|---|
| `glm53_model_wiring` | `glm53_prefill_fastpath.py`, `glm5next_model.py` | 모델 파일(`model.py`) 접수 + 프리필 메타데이터 웜업 (union 프리필·fused K+gate·SM121 캐시 전용 경로는 34차 §8 일몰) |
| `glm53_indexer_gate_splitk` | `glm53_indexer_gate.py`, `glm5next_attention.py` | 어텐션 파일 접수 + 인덱서 head-gate split-K (`INDEXER_GATE_SPLITK`, EXP-9) |
| `glm53_mk_kda_wiring` | `glm5next_kda.py` | KDA 블록 파일 접수 — MK-KDA 인수 훅은 34차 §8 일몰; 스톡 체인 + `KDA_ONEPASS` 배선만 남음 |
| `glm53_mk_mla_wiring` | `flashinfer_mla_sparse_sm90.py` | sparse MLA 백엔드 접수 — MK-MLA 라우팅 (`MK_MLA`) |
| `glm53_kda_onepass` | `glm53_kda_onepass.py` | KDA 원패스·듀얼 게이트 GEMM (`KDA_ONEPASS`·`KDA_DUAL_GEMM`, EXP-20) |
| `glm53_fp8_dense` | `glm53_fp8_dense.py` | 밀집 투영의 블록 fp8/W4 레인 (`FP8_DENSE`·`DFLASH2_FP8_DENSE`·`FP8_DENSE_BPROJ`) |
| `glm53_prefill_nvfp4` (신규, 368차) | `glm53_nvfp4_scale.py`, `glm53_nvfp4_bproj.py` | 프리필 NVFP4 스케일 퓨전 + B-projection 라우팅 (`NVFP4_SCALE_FUSED`·`PREFILL_NVFP4_BPROJ`, 기본 0) |

---

## glm53_model_wiring (was `overlay/modules/glm53_model/`)

## glm53_model_wiring

Constructs the router through `DenebGateLinear` instead of `GateLinear`, so
GLM takes the fused small-M gate. The kernel itself is `moe_gate_sm121`;
this is only the construction site, which is why it is a separate module.

(The cache-only sparse-indexer path behind `VLLM_GLM53_SM121_MLA_PREFILL=1`
was sunset in 34차 §8.) It once installed that path when the dense-MLA arm
admitted a fresh prefill. Dense
MLA never consumes the indexer's sparse scores, so the path projects and
normalizes only K, computes the kpool gate, and lets the existing kpool op
write its index-K and persistent tail caches. It avoids the otherwise dead
query projection, FWHT/FP8 quantization, fp32 head-weight projection/scaling,
and the unused rows of the fused K/head-weight projection before the custom op.
K and gate consume the same hidden rows, so their bf16 source weights are also
combined once after checkpoint loading. The cache-only path produces both with
one 256-row GEMM instead of two independent 128-row GEMMs. The extra weight is
a non-persistent buffer; the original parameters stay intact for every fallback.

The model path and kpool op share the same request-level no-consumer predicate.
CUDA graph capture, mixed decode/prefill, cached-context or long MQA prefill,
profiling metadata, module-name drift, and any production shape/dtype drift
fall back to the original `Indexer.forward`. When the dense-MLA env arm is not
the exact value `1`, the installer does not modify `Indexer.forward` at all.
If the fused weight cannot be built, that layer also uses the original path.
Rollback remains that single env value.

Base contract from `glm53:v13-b12x`.

---

## glm53_indexer_gate_splitk (was `overlay/modules/glm53_model/`)

## glm53_indexer_gate_splitk

The sparse indexer's fp32 head-gate projection as a deterministic split-K
Triton path, behind `VLLM_GLM53_INDEXER_GATE_SPLITK` (profile default `1` since 2026-09-06, MKG3 bracket; `0` = stock
`torch.mm`). This README is the single home of the measured numbers; the
ledger and RUNBOOK cite it.

### What the stock code does

`Indexer.forward` (`vllm/models/glm5next/nvidia/attention.py`) computes

```python
weights = torch.mm(hidden_states.float(), self._wp_fp32)   # [M, 4096] x [4096, index_n_heads]
```

once per full-attention layer, in fp32 on purpose: bf16 head-gates (~1e-2
error) flip near-tie pool rankings, and the ranking is what the sparse
attention selects. The fleet checkpoint (`glm53-redhat-nvfp4`) has
`index_n_heads = 32`, so the weight is `[4096, 32]` (the first version of this
module admitted N <= 16 and never ran on the fleet; the offline numbers were
taken on a synthetic N=16 -- corrected 2026-09-03). cuBLAS answers the M<=16
shape with a two-block `gemmSN` kernel: 88 us DRAM-cold on an idle GB10 and
88 us in the 2026-09-01 serving trace, eleven times per decode step, for
512 KB of weights.

### What this module does

Two small kernels (`glm53_indexer_gate.py`): program `s` of the first loads
one `[128, N]` weight slice once and multiplies it against every row of x
(bf16 loaded and cast to fp32 in registers -- the same exact conversion
`.float()` does), writing a `[16, N]` partial; the second sums the 32 partials
in a fixed order. No atomics and no memset, so the result is **bitwise
reproducible run to run and identical on every TP rank** (the indexer is
replicated per rank and the ranks' top-k pool selections must agree). Used
only when the knob is `1` **and** `splitk_applicable` admits the shape:
2-D x with unit inner stride, M <= 16 (C=1 verify batches are M=8, C=2 M=16),
`x.shape[1] == w.shape[0]`, fp32 contiguous w with N <= 32, K a multiple of
128. Everything else -- prefill, C>=3, unexpected layouts -- keeps `torch.mm`.
(The fused-indexer forward in `glm53_prefill_fastpath.py`,
`VLLM_GLM53_FUSED_K_GATE=1`, that routed through the same helper was sunset
in 34차 §8; the file keeps the metadata warm-up only.)

Both paths accumulate in fp32; only the summation order differs, so this is
**not bit-exact** with stock -- it is a served-numerics change and carries the
full quality bracket.

### Offline numbers (GB10, `probes/indexer_gate_check.py --config <checkpoint>/config.json`, 2026-09-03)

CUDA-graph replay with 46 distinct weights cycled (DRAM-cold), N=32, K=4096:

| M | stock `torch.mm` | split-K | route |
|---|---|---|---|
| 1 | 20.8 us | 9.5 us | split-K |
| 2 | 51.0 us | 10.1 us | split-K |
| 4 | 86.4 us | 11.1 us | split-K |
| 8 (C=1) | 89.5 us | 12.7 us | split-K |
| 12 | 93.9 us | 14.7 us | split-K |
| 16 (C=2) | 87.7 us | 16.8 us | split-K |
| 24 (C=3) | 17.3 us | -- | `torch.mm` kept |
| 32 (C=4) | 17.7 us | -- | `torch.mm` kept |

Numerics over 200 trials / 1,571 rows (bf16 activations x1.5, fp32 weights
x0.02): max |diff| 2.4e-6 absolute, 7.2e-7 of the row's max gate, 0 top-1
flips, 0 top-4 set changes; 50 repeated launches bit-identical; strided x,
K mismatch, M=0 and M=17 all route to `torch.mm` (checked by the probe, which
exits non-zero on any failure).

Ceiling: 11 layers x (89.5 - 12.7) us = 0.84 ms/step at C=1, **~1.2%** of the
71 ms step (C=2: 0.78 ms). Below the ledger's boot bar on its own; it rides
the EXP-7 boot (bit-exact, so this stays the only numerics axis on that
boot) -- not EXP-8, whose gate is that acceptance must not move.

### Arming and verdict

`VLLM_GLM53_INDEXER_GATE_SPLITK` is a profile-declared key: pass it as caller
env, never through `EXTRA_ENV`.

```bash
VLLM_GLM53_INDEXER_GATE_SPLITK=1 bash launchers/start-glm53-nvfp4-tp4.sh
```

Boot log: the helper announces each verdict ONCE (not once per shape -- prefill
M differs on every request) --
`[indexer-gate] ...=1: first split-K routing, e.g. x(8, 4096) torch.bfloat16 @ w(4096, 32) (M<=16 admitted)`.
The companion `first stock torch.mm routing` line is normal (prefill and C>=3
take it). Only the split-K line missing means the kernel never ran -- that is
how the N=16 version would have shown up.
If `glm53_model_wiring` is mounted without this module, its fastpath logs
`[indexer-gate] ... not mounted -> stock torch.mm`. In a trace the eleven
`gemmSN` launches become eleven `_gate_splitk_partial_kernel` +
`_gate_splitk_reduce_kernel` pairs.

Gate: quality 9/9, Korean 0/16, pos-1 acceptance +/-2pct, C=1 step/s bracket
base -> cand -> base.

### Shape of the overlay

`attention.py` is overlaid whole (2 changed lines, preimage `a0870c31...` of
`glm53:v13-b12x`, identical in `glm53:sm121-fi618`), the repo's standard for
image-bound files: an image bump that touches `attention.py` fails the
launcher's preimage check and stops the boot even with the knob at `0`, which
is the intended signal (the alternative, a runtime copy of `Indexer.forward`,
would make a second source of truth for that function next to the fastpath's).
No other module may own `attention.py`. `requires` is empty: the wiring
fastpath optionally imports this module, not the reverse.

---

## glm53_mk_kda_wiring (was `overlay/modules/glm53_model/`; the MK-KDA takeover was sunset in 34차 §8 -- history)

## glm53_mk_kda_wiring

The image-bound hook for `MK_SEG_KDA`: GLM-5.3's linear-attention block
(`vllm/models/glm5next/nvidia/kda.py`) with the megakernel takeover and the
shadow epilogue spliced in. Behind `VLLM_GLM53_MK_KDA=1` /
`VLLM_GLM53_MK_KDA_SHADOW=1` (master `VLLM_GLM53_MEGAKERNEL=1`); disarmed, the
file runs the stock forward body verbatim.

It used to be the third row of `glm53_megakernel`'s manifest. It is its own
module now because it is the ONE thing in that module that is GLM's: the
container path `vllm/models/glm5next/` does not exist in another model's
image, and its pinned preimage would fail the deploy gate there. The kernel
core (`glm53_megakernel.py` + `.cu`) binds only
`vllm/model_executor/layers/`, relative to the profile's `TARGET_PREFIX`, and
carries no model file at all -- which is what lets a second profile mount it.

The split is bytes-neutral for GLM: `glm5next_kda.py` moved unmodified, and
`compose-overlays.sh glm53` renders the same `build/glm53/` as before (the
manifest's rows are the same three, redistributed across two modules).

### Requires

`glm53_megakernel` (the kernel it calls) and `glm53_fp8_dense`: the KDA packs
are built from a stock fp8-dense arm (`isinstance(in_m, Fp8DenseMethod)`), so
without that module the takeover has no weights to stream. That requirement
used to sit on the core module; it belongs here, on the hook that actually
needs it.

### Gate

Unchanged, and it is the strict one: MK-KDA's state-index contract is checked
by the boot self-test AND by eager shadow mode
(`VLLM_GLM53_MK_KDA_SHADOW=1`, outputs and next-step states diffed every 64
calls at the e2m1 by-design class). Run shadow on a bench boot, read the log,
then decide the arm. See `overlay/modules/glm53_megakernel/README.md`.

---

## glm53_mk_mla_wiring (was `overlay/modules/glm53_model/`)

## glm53_mk_mla_wiring

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
* a one-shot shadow on the first eager call **whose rows read real KV**
  (`valid_counts.max() >= 16`, `|ref| >= 1e-3`, T <= 4096 -- a request's
  prefill chunk): kernel vs wrapper on the same bytes, `rel > 2e-2` or a
  non-finite output DISARMs every later eager call and logs
  `mla SHADOW FAIL ... this boot is invalid`. The decode graphs captured at
  boot still bake the kernel, so a failed boot is a failed boot, not a
  fallback. Until 28차 the shadow ran on the profile dummy, whose KV is
  empty: both sides returned zeros and `rel=0.00e+00` armed the kernel on
  all four ranks for nothing. `[megakernel] mla shadow vs wrapper rel=...
  (T=.., rows<=.., real KV) -> ARMED` is the line to look for; the boot
  self-test (`selftest mla`) covers the direct path (40 and 100 rows) too.
* the split path's fp32 scratch is one fixed allocation (`MLA_WS_ROWS` =
  192 rows, 6.3 MB) and `mla_splits()` never asks for more. It used to grow
  on demand, freeing the old tensors under every decode graph captured
  before the first larger call -- and v5 routes prefill rows through the
  same entry point, so a 37-token prompt (48 splits, 1,776 rows) did that on
  the first request of a boot: two serving deaths on 2026-09-04 (a gather
  index out of bounds at the 4th request; an illegal memory access on all
  ranks right after a 23K prefill). Both boots with the segment off survived
  the same bracket. With the fix the MLA-on arm passed the bracket twice
  (C=1 step/s 16.4 vs 16.235 off; 23K prefill 2,700 vs 2,300 tok/s, TTFT
  10.0 -> 8.5 s; quality 9/9, Korean 0/16, pos-1 67.0% vs 64.5%) and the
  profile ships `VLLM_GLM53_MK_MLA=1` within the megakernel set (28차).

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

---

## glm53_kda_onepass (was `overlay/modules/glm53_model/`)

## glm53_kda_onepass

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

### `resolve()` -- knobs, counters, boot self-test

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

### `VLLM_GLM53_KDA_DUAL_GEMM` -- f_b + g_b in one launch

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

### `VLLM_GLM53_KDA_ONEPASS` -- conv + recurrence + gated norm in one launch

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

### Offline gate

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

### Log anchors

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

### Not in this module

MK_SEG_KDA (`VLLM_GLM53_MK_KDA`) replaces the whole block including both
GEMMs; when it serves, the layer takes the megakernel branch before this
module is consulted. The prefill chunk path and mixed prefill/decode steps
keep the stock kernels.

---

## glm53_fp8_dense (was `overlay/modules/glm53_model/`)

## glm53_fp8_dense

Block-fp8 (W8A8, ue8m0, 128x128) copies of the bf16 dense projections the
RedHat nvfp4 checkpoint leaves unquantized: attention projections (including
the KDA merged `in_proj_qkvbfg_a`, zero-padded to the block grid in the copy
only), the shared expert, and the first-3 dense MLPs — 15.44 GB checkpoint-wide,
~3.86 GB/rank read every forward, ~17 ms of the decode step at the measured
bandwidth. DSV4 serves its dense path in exactly this scheme on this fleet
(9/9 retrieval, pos-1 acceptance 78.5%), so this is convergence to the proven
lane config, not a precision experiment.

Quantized once after `load_weights` (the call site lives in `glm53_model_wiring`)
and before compile/capture, by swapping each Linear's `quant_method`; the bf16
originals stay for fallback. Armed by `VLLM_GLM53_FP8_DENSE=1` -- off when the
env is unset, and **the glm53 profile ships it 1**, so serving always has it. A
module's default and what the fleet runs are different statements; profiles/README.md
carries the second one.
The kernel pair is the one `fp8_lm_head` already runs under capture here.

### b-projection arm — `VLLM_GLM53_FP8_DENSE_BPROJ` (profile default on since 2026-09-06, MKG3 bracket; module default off)

STEP_KERNEL_MAP #108 §2: after W8A8, 145 `cutlass_80_wmma` bf16 GEMMs/step
remain — the rear halves of the low-rank projections. This arm extends the
pattern set to `self_attn.q_b_proj` `[4096,1536]`, `self_attn.kv_b_proj`
`[4096,512]` and `self_attn.indexer.wq_b` `[4096,1536]` (replicated — full
read on every rank). ~160 MB/step fewer bytes at C=1 ≈ 0.9% step ceiling at
the 273 GB/s floor; honest expectation is under that (the indexer GEMMs are
aux-stream contention-stretched, not bandwidth-bound, per the 08-10
decomposition).

Deliberately out: `f_b_proj`/`g_b_proj` (per-rank `[2048,128]` — under the
`min(shape) >= 512` guard, ~17 MB/step win at most), and `wk_weights_proj`
(the loader upcasts it to bf16 to keep the wk+weights_proj fusion; quantizing
it would break that contract). Requires `VLLM_GLM53_FP8_DENSE=1` to have any
effect; the existing per-layer guards, stale-copy check and fallbacks apply
unchanged. Rollback = the env alone.

Boot-log fingerprint: `[fp8-dense] N linears quantized (X GB), M kept bf16`.
Gates for adoption: 9/9 retrieval, 0/16 Korean corruption, pos-1 acceptance
within 2 pct of the same-boot control, C=1/2/4 bracket. Rollback = env only.

### DFlash2 drafter arm — `VLLM_DFLASH2_FP8_DENSE` (rewired 2026-09-04)

`glm53_dflash_loader_fp8` runs this pass on the drafter under its own knob.
Until 2026-09-04 the base pattern set was all it had, and that set lists
`q/k/v_proj` and the target's fused names — never the drafter's MERGED
`qkv_proj`, its aux-hidden `fc` (`[4096, 5 x 4096]`, ReplicatedLinear, read
whole on every rank) or the two conv `kernel_projection`s per layer. So the
knob covered `o_proj`/`gate_up`/`down` and left 43% of the drafter's bytes
bf16, the fc alone 23%. `_DRAFTER_INCLUDE` closes that under the drafter knob
only; the target's set is untouched.

Why it matters: the armed 09-03 trace ends every decode step in a 7.5-8.5 ms
tail (target head, drafter, draft head) the GPU is 95% busy through, and the
drafter's bf16 GEMMs are 3.2-3.6 ms of it (STEP_KERNEL_MAP supplement 4).
Offline at the fleet shapes, TP=4 sharding, DRAM-cold (`probes/
drafter_fc_check.py`, srv2): bf16 3.23 ms/step → fp8 pair 2.34 → MK W4 1.25.

Two things are different about the drafter:

- **Its forward is torch.compiled** (`@support_torch_compile`). The Python
  MK-or-fp8 choice `Fp8DenseMethod.apply` makes for the (eager) target is not
  traceable — the lane's eligibility test guards on the token count and the
  extension call is a pybind function — so drafter methods are marked
  `_opaque` and route through ONE custom op, `glm53_fp8_dense::gemm_mk_or_fp8`
  (`_mk_or_fp8_dense_gemm`), which arms, tries the lane and falls back to the
  verified fp8 pair at run time, exactly as eager does.
- **The fc is wider than the lane's K** (20480 vs `MK_GEMM_KMAX` 4096). It
  carries one pack per K-chunk (`build_mk_weight_w4_kchunks`) and `gemm_w4a8`
  runs five launches summed in fp32 — 301 µs against 682 bf16 / 489 fp8.

**Default on since 2026-09-04 (28차).** The bracket (same image, MK-MLA off
on both arms, W4 drafter actually served -- see below): C=1 step/s 15.95 →
16.235 (+1.8%, six reps each), pos-1 acceptance 64.5% vs 61.6%, quality 9/9,
Korean 0/16, prefill unchanged (2,300 tok/s at 23K both arms). The fp8-only
`w8` arm is gone: the operator's rule is that a proven improvement becomes
the default and the other side is removed, not kept as a second setting.
Offline gate: `probes/drafter_dense_path_check.py` (build → lane serves
bitwise → fullgraph and dynamic compile → CUDA-graph capture). Rollback =
`VLLM_DFLASH2_FP8_DENSE=0`.

**Armed is not served -- the compile cache.** vLLM keys its torch.compile
cache, and under `VLLM_USE_AOT_COMPILE=1` the whole AOT artifact, on the env
vars registered in `vllm.envs`, the vllm config and the forward's source,
then loads it with guard checks off. A `quant_method` swapped in after load
is no part of that key, so every boot with the knob on served the drafter
from the artifact of the first boot that ever compiled it (09-03: bf16
`F.linear` on all 30 layer projections) while the fingerprint reported 31
linears armed; only the eager fc reached the lane, and the first bracket
measured exactly nothing. Two pieces close that: the knob registers itself
into `vllm.envs.environment_variables` (`_register_compile_factor`, so each
value is its own artifact), and `install_drafter_serving_check` counts
opaque-op calls over the drafter's first forwards and writes the verdict
into the boot log -- `[fp8-dense] drafter lane serving: 30 of 31 opaque GEMM
calls per forward` (the fc runs outside the compiled forward) or
`drafter lane NOT SERVING`. A CUDA-graph replay runs no Python and is not
judged; a forward under stream capture is definitive.

The drafter's bf16 sources are released at load under the target's knob
(`VLLM_GLM53_FP8_DENSE_FREE_BF16=1`, 28차): the checkpoint walk is finished
when the pass runs, `_build_fused_kv_buffers` has already `torch.cat`'ed the
k/v halves of `qkv_proj.weight` into its own buffer, the compiled forward
reads the fp8 copies and W4 packs through the opaque op, and no drafter
linear carries a bias. `[fp8-dense] drafter: VLLM_GLM53_FP8_DENSE_FREE_BF16=1:
released 0.73 GB of bf16 sources` is the line; the bytes come back before KV
sizing.

### Prefill on the nvfp4 pair — `VLLM_GLM53_FP8_DENSE_PREFILL_NVFP4` (lever 7, 28차)

The nvfp4 scheme (`VLLM_GLM53_FP8_DENSE=nvfp4`) replaces the method and so
turns the MK lane off for every layer it takes (#263). This knob keeps the
fp8 method and its W4 pack and ATTACHES an nvfp4 pair to it: rows above the
MK lane's M (32) route to `mm_fp4` in `Fp8DenseMethod.apply`, decode keeps
the W4 lane, the fp8 pair stays the fallback. The alpha convention is settled
once and a sample of layers re-runs the value check, exactly as the scheme
does; a failed check keeps that layer's prefill on fp8. Cost ~+1 GB/rank of
copies. Target only (the drafter's forward never sees prefill rows). Gate:
prefill ladder (dense GEMM is 14% of a 32K prefill; expect ~-8%) plus the
served-numerics gates, since A4 activations are a quality change.

---

## glm53_prefill_nvfp4 (368차 신규, 기본 0)

Two opt-in prefill candidates on the fp8-dense lane. Both arm only with the
exact string "1" (`true`/`yes`/`on` do not).

`VLLM_GLM53_NVFP4_SCALE_FUSED=1` fuses the activation amax/scale pass with
GEMM alpha setup (2688.0 == 448.0 x `_NVFP4_MAX` identity, same op order) —
bit-exact vs the unfused helper by construction;
`probes/glm53_nvfp4_scale_check.py` is the GPU assertion. The gate is
tensor-property-only, so it changes every `_nvfp4_dense_gemm` caller,
decode included; acceptability rests on that bit-exactness.

`VLLM_GLM53_PREFILL_NVFP4_BPROJ=1` routes pure-prefill rows of still-
unquantized b_proj linears through an NVFP4 pair; decode and absorbed-MLA
weights are untouched (`get_and_maybe_dequant_weights` still reads the BF16
source, and the BF16 free pass skips this method). **No-op while
`VLLM_GLM53_FP8_DENSE_BPROJ=1`** (this profile's default): the fp8 pattern
already owns every b_proj linear. Neither knob has GPU validation yet —
keep both at 0 until the combined campaign.

### Startup artifacts: FP8 copies and rank checkpoints (2026-09-06, default off)

Two independent path-valued switches reuse work from a prior boot:

```bash
VLLM_GLM53_FP8_CACHE=/cache/glm53-fp8 \
VLLM_GLM53_RANK_CACHE=/cache/glm53-ranks \
bash launchers/start-glm53-nvfp4-tp4.sh
```

The launcher carries these profile-declared variables to every worker. `/cache`
is already the node-local persistent cache mount. Set either variable to an
empty string to disable it. `LOAD_FORMAT` remains the normal source loader
(`instanttensor` by default); **do not set it to `sharded_state` for these
artifacts**. Both switches remain empty in the profile until matched GPU boots
prove correctness, memory safety and a net latency improvement.

**FP8 cache.** `glm53_startup_cache.py` saves the exact padded E4M3 weight and
DeepGEMM scale backing allocations, including dtype, shape, stride, storage
offset and padding. Its key contains the current source weight's SHA-256,
shape/dtype/stride, quantization recipe, Torch/CUDA/device identity and imported
vLLM/DeepGEMM implementation digests. This applies to both target and drafter
FP8 folds. Content/layout checksums reject corrupt artifacts and rebuild;
failed writes keep freshly computed results. Publication is atomic and reads
use `weights_only=True`. The existing direct-kernel source check still runs on
every layer and evicts an artifact if it rejects the copy. MK/GPTQ packs,
precision, fallback behavior and BF16 release timing are unchanged.

`[fp8-cache]` reports hits, misses, errors and key/read/quantize/write host wall
time. `[fp8-dense]` additionally splits source-check, MK attachment and
`empty_cache` time. These are host phase times including existing syncs, not
isolated CUDA kernel timings. Per-linear allocator cleanup remains mandatory.

**Rank checkpoint cache.** `glm53_rank_cache.py` intercepts only the GLM model
that owns the complete checkpoint walk. The first source load snapshots its
rank-local parameters and persistent buffers **before** GLM FP8/MK hooks and
vLLM quantization finalization. On a hit it restores those tensors without
advancing the source iterator, then executes the same outer post-load hooks
once. This avoids the missing plain-attribute FP8/MK copies and freed BF16
weights in a dump of a running model. It uses the same rank-local loading idea
as [vLLM's ShardedStateLoader](https://docs.vllm.ai/en/latest/api/vllm/model_executor/model_loader/sharded_state_loader/),
with a separate pre-finalization artifact format and lifecycle.

A readiness MIN vote on the established world device group requires **every
rank** to have a matching, complete cache. Otherwise all ranks use the source
loader: InstantTensor itself uses that group, so mixed cache/source paths could
strand its collectives. No additional Gloo object exchange is introduced.

Rank identity includes the local checkpoint index/config and every source
file's resolved path, device/inode, size, nanosecond mtime and ctime, plus model
config, TP/rank, environment and runtime code. This is an immutable-source,
node-local cache; copying checkpoint files or changing configuration/code causes
a miss. It does not rehash the full original checkpoint on warm boots. Keep
source files read-only and consistent across nodes, as with the source loader.
Source identity is rechecked after loading/restoration to reject concurrent
changes. Unsupported EP/EPLB, PP/DP/context parallelism, LoRA, meta tensors,
noncontiguous state or secondary weight sources use the original loader.
Later weight reloads bypass this startup cache.

Each rank writes one raw payload plus a complete schema/chunk-checksum manifest
in a temporary directory, then atomically publishes the directory. Serialization
and restoration operate in at most 64 MiB chunks. Cold writes sync and discard
file pages per chunk; warm reads discard consumed mmap/file pages on Linux to
avoid retaining a rank-sized host page cache on UMA. Allow disk space for the
whole rank state plus 512 MiB; source/runtimes with different identities keep
separate directories. There is no automatic cache eviction.

Missing/incompatible metadata falls back before restoring any tensors. A
payload checksum error **aborts the boot**, since some earlier chunks may have
been copied; it never serves a partial checkpoint. The log names rejected
metadata directories. Remove the affected artifact directory during an idle
window to rebuild it; existing published directories are never replaced under
readers. A single missing rank makes the next boot use the source loader on all
ranks, while only missing caches need writing.

`[rank-cache]` is the authoritative hit/save timing. Default boot stamps start
the source loader timer even when its iterator is unused; if
`DENEB_BOOT_STAMPS=0`, ignore the stock `Loading weights took` line on a hit
(the stock loader starts that timer only from its skipped iterator).

Validation: CPU byte/layout/invalidation/fallback tests and a real four-process
Gloo readiness test live in `tests/test_glm53_startup_artifacts.py`. They do not
prove NCCL, actual DeepGEMM GPU output, full GLM post-load compatibility, UMA
peak memory or serving acceptance. Before default-on, compare cache-disabled,
cold-write and all-rank-hit boots with identical model/profile/prompt sets;
check source-vs-hit outputs, drafter acceptance, all-rank hit logs, phase times,
RSS/MemAvailable/CUDA peaks and total health-ready time. The earlier 54.9 s fold
and 52.1 s read/apply budgets predate #366 deployment and are **not** measured
savings from these caches. The unrelated 302 s distributed-init pause remains
unresolved.
