# CPU17: complete CPU gate passed

Normal fleet session `epdecodecpu0909v17` completed on 2026-09-09 at
10:53:50–10:54:10 KST. Payload, evidence copy, and outer return codes were 0.
The srv2 fleet ran its no-device CPU payload on srv1 using frozen source
`0331579f4b16b8b811f2ca7e5099f8f461507c67` at
`/home/choiceoh/stkernel-ep-onepass-0909-17`.

**94 tests passed with zero failures, errors, or skips.** The original result
records `verdict=PASS`, `phase=complete`, `cuda_initialized=false`, and
`binding_runtime_rechecked=true`. All 28 kernels compiled: three micro CuTe
variants, one dynamic-prefill CuTe variant, and 24 Triton preparation variants.
The successful actual lowering includes the direct path's CuTe coordinate and
one-stage host validator; earlier CPU15/16 failures remain separately archived.

The micro variants are M32/top8/direct/FP32, M64/top1/buffered/FP32, and
M64/top8/BF16. The direct key includes both `glm53_ep_micro_scatter_fp32_v1`
and `glm53_ep_micro_direct_scatter_v1`; its two constructor flags are true.
The buffered FP32 control has only the first tag; the BF16 control has neither.
Both M64 control PTX/cubin hashes exactly match CPU14.

In the receipt-bound direct PTX, all four FC2 output blocks (lines3215–3761,
3762–4311,4312–4855,4856–5402) have no shared stores, BF16 shared loads,
proxy fences, or pre-scatter barriers. Each retains its final `bar.sync 1, 128`.
Each block has 16 FP32 vector RED sites and 16 ID/16 weight metadata loads.
The resulting64 RED sites are statically unrolled register pairs, not64 runtime
reductions per output or an increase in measured traffic. Exact instruction
ranges/text and hashes are in `verification.json`; `verify.py` reproduces them.
The buffered controls retain their original shared scatter paths/barriers.

Direct M32 cubin resources are `REG161 STACK0 SHARED1024 LOCAL0`, compared with
CPU14 buffered M32 `REG128 STACK0 SHARED1024 LOCAL0`; the actual new PTX has
no local loads/stores. Register growth is recorded without a speed claim.
`resource-analysis.json` derives a64KiB dynamic shared layout from the retained
A/B/up double buffers, scale buffers, metadata alignment, and Q1 sC. This is
separate from cubin `SHARED1024`. The block has160 threads; both old/new PTX
retain the232/32 register-budget instructions. Two such shared allocations
exceed the100KiB/SM limit documented for CC12.x, so the source-derived maximum
remains one resident CTA per SM. This is not runtime occupancy measurement;
the CPU compile's48-block grid is not a live launch observation. Register and
instruction-scheduling changes can still affect performance. Architecture
limits: [NVIDIA CUDA Programming Guide, tables30–32](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html#features-and-technical-specifications).

`evidence.tar.gz` preserves **61 original files**:28 PTX,28 cubin,4 resource
logs, and the result. `result.json` is byte-identical to its original. `head/`
preserves the submitted command, driver, exit receipt, and complete log.
The runner retained its12GiB host admission gate,4GiB memory/2CPU limits,
runc runtime, and no-device configuration.

Before and after collection, all original worker, copied head, and archived
file hashes matched. Both frozen checkouts were clean at the exact revision,
with full history and no alternates; actual worker source hashes matched the
submission and compiled receipts. The immutable image remained
`sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211`.
The worker's116-file bindings capsule passed strict validation against manifest
`b29ac01b3a45a5c140ba205187c86411412c5b74e7ac31417747e028588debab`.

`capture.json` retains these observations and original hashes. `verify.py`
reproduces the exact artifact-set, source-receipt, completion, and static PTX
checks; its output is `verification.json`. Collection only read/copied existing
results after GPU17 submission was confirmed. It did not submit tests, touch
GPU ownership, or modify frozen sources. This archive establishes CPU evidence
only; GPU numerics and serving acceptance remain separate. `SHA256SUMS` covers
every archive file except itself.
