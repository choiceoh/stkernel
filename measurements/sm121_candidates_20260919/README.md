# The seed image's kernels for the sm_121a intake, judged on one GB10 — 2026-09-19

> 그대로 두는 기록 — 이 날 srv4 단일 GPU 레인에서 잰 숫자와 그 원시 보고. 고치지 않는다.

engine/SM121_INTAKE.md: each kernel measurements/sm121_inventory_20260919 found in the image, held to a reference and
timed beside production (`st-glm53` serving; eager calls, so small calls carry launch and host time and every number
wobbles with production's steps). Probes: `probes/engine_sm121_candidates.py` (gdn, fp8_l2, gdn_diag),
`engine_sm121_attention.py`, `engine_sm121_fp4.py`, `engine_sm121_sparse_mla.py`, `engine_sm121_sanitizer.py`, run through
`engine_sm121_batch.py` (several lanes a ticket, each its own process).

| ticket | lanes | seconds |
|---|---|---|
| `sm121-batchA-0919b` | gdn, fp8_l2, sanitizer | 16 + 24 + 1 |
| `sm121-batchB-0919b` | fp4_gemm, fp4_moe, attention, sparse_mla | 700 + 97 + 1,240 + 57 |
| `sm121-batchC-0919c` | gdn (int64 cu_seqlens), fp8_l2 (arms interleaved), sanitizer (preflight) | 33 + 11 + 5 |
| `sm121-gdndiag-0919d` | gdn_diag | — |
| `sm121-gdnnorm-0919f` | gdn with q/k normalised first | — |

## U13 — GDN prefill (Qwen3.8's 4 / 12 heads x 128, carried state)

FlashInfer's `chunk_gated_delta_rule` (SM120 CuTe-DSL) returned **NaN in every element** under every convention tried
(`sm121-gdndiag-0919d`: GVA 4/12 and widened 12/12, gate as alpha and as log, carried / zero / swapped state). The cause is
flashinfer#5255 (open, 2026-09-17): the native prefill path ignores `use_qk_l2norm_in_kernel`, so unnormalised q/k make
the recurrence diverge. With q/k normalised first (`sm121-gdnnorm-0919f`): no NaN, output within 0.36–0.74% of the fp32
recurrence (the served kernel's own 0.36–0.74%), 0.36–0.53% of the served kernel.

| tokens | served chunk kernel | FlashInfer, kernel only | speed (median) |
|---:|---:|---:|---:|
| 128 | 121.8 / 106.4 | 322.0 / 86.0 | 0.38x |
| 1,024 | 387.9 / 365.1 | 231.5 / 218.5 | 1.68x |
| 8,192 | 4,245.0 / 4,001.2 | 993.7 / 871.9 | 4.27x |

(µs, median / best of 9.) The call as this probe builds it adds a host copy and allocations (2.5 ms at 128 tokens); the
engine lane builds them on the device (engine/kernels/gdn_prefill_sm120).

## U11 — the FP8 prefill GEMM past the L2

`sm121-batchA-0919b`: no cliff like vLLM's CUTLASS one — at vLLM's own shape (16,384 x 2,560, 42 MB) deep_gemm held
123 TFLOPS and cuBLASLt 121 at M = 16,384 (vLLM's default raster: 52). GLM's 101 MB gate_up moved non-monotonically
(cuBLASLt 129 → 43.5 → 106 → 84 TFLOPS). `sm121-batchC-0919c`, arms alternating in 9 rounds: every shape fell to 46–68
TFLOPS at M ≥ 8,192 **including the 17 MB control inside the L2** (105 → 53) — contention with production, not the L2.
No vLLM-style cliff is visible from this lane; a quiet GPU would say more.

## U7 / U8 — paged GQA with window, sinks, soft cap, and FP8 / NVFP4 KV (8q/2kv d128 and gpt-oss's 16q/2kv d64)

FlashInfer's CUDA-core and tensor-core (fa2) decode and the auto prefill compile for sm_121a and are exact within
0.2–0.5% of an fp32 reference for causal, `window_left` 127 and `logits_soft_cap` 50, at KV 1,000 and 33,000 and a batch
mixing the two. **`sinks=` is accepted and ignored** by both paths (error 0.04–1.04, each equal to the reference without
sinks) — the AttentionSink JIT variant and `xqa_batch_decode_with_kv_cache` compute sinks (0.2–0.6%). xqa has no soft cap.
cute-dsl refuses sm_121a (tcgen05). A 128-window at KV 33,000 decodes in 15.5 µs on the tensor-core path against 205 µs
causal.

KV: FP8 (e4m3, per tensor) — the kernels exact to 0.3–0.9% of their own dequantized cache, the cache 3.7–7.2% from the
BF16 one. NVFP4 (FlashInfer's `fp4_quantize`) — kernels exact to 0.3–1.0%, the cache **13–20%** from BF16. Split-KV off
(`disable_split_kv`) changed nothing numerically here and cost 1.4–11x.

## U5 — dense block-scaled FP4 GEMM

W4A4 NVFP4 through `mm_fp4` backends b12x, cutlass, cudnn: within 0.20–0.35% of the exact product of what the kernel
read, at (4096, 4096), (12288, 4096), (4096, 12288), (2560, 1536) and M 1..8,192. cute-dsl W4A4 refuses capability 121;
W4A16 (BF16 × FP4) through cute-dsl is exact but slow past M 128 (22 ms at M 8,192, 4096²); cuDNN's W4A16 needs cuDNN
9.23.1 (the image has 9.20). At M 8,192 the W4A4 backends run 1.3–4.0 ms against FP8 deep_gemm's 2.6–7.6 ms; the eager
small-M numbers are host-bound (≈190 µs floors).

## U3 — MXFP4 MoE

The CuTe-DSL MXFP8 × MXFP4 MoE (the only W4A8 path in the image) is SM100-only: "No supported CUDA architectures found
for major versions [10]". b12x's `mxfp4` mode runs (W4A4: it quantizes activations to MXFP4) and is 16–24% from both the
BF16 reference and one with the activations snapped to MXFP4 — not the lane the intake asked for.

## U6 — sparse MLA at DeepSeek-V3.2's rank (32 heads, 576-wide keys, top-k 2,048)

`flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla(backend="sparse")` built for `compute_121a` in 49 s. Decode 30.3 µs
(batch 1), 51.0 µs (batch 4); prefill of 256 tokens 1,156 µs. Error 5.2–8.3% against the oracle on the cache it read
(the kernel computes q and the attention weights in FP8; upstream tests allow 5%). **No livelock**: 20,000 back-to-back
batch-4 calls in 2.24 s, the longest sync gap 13 ms, the output unchanged after.

## U15 — compute-sanitizer

`compute-sanitizer --tool memcheck` (2026.1.1) cannot instrument inside the lane's container: a one-line CUDA program
ends "Target application terminated before first instrumented API call", while the same program without the tool prints
"cuda ok". The code audit (engine/SM121_INTAKE.md) stands; the tool needs a container that allows it.

Raw reports: every `*.json` beside this file (each lane's own, and each batch's summary).
