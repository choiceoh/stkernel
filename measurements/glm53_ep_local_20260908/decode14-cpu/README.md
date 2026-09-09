# CPU14: complete CPU gate passed

Normal fleet session `epdecodecpu0909v14` completed on 2026-09-09 at
10:16:31–10:16:57 KST. Payload, evidence copy, and outer return codes were 0.
The srv2 fleet ran its no-device CPU payload on srv1 using frozen source
`c3696a76c6674ad38e01967fa587b8d31a6d121e` at
`/home/choiceoh/stkernel-ep-onepass-0909-14`.

**85 tests passed with zero failures, errors, or skips.** The original result
records `verdict=PASS`, `phase=complete`, `cuda_initialized=false`, and
`binding_runtime_rechecked=true`. All 28 kernels compiled: three micro CuTe
variants, one dynamic-prefill CuTe variant, and 24 Triton preparation variants.

The compiled micro variants are M32/top8/FP32, M64/top1/FP32, and unchanged
M64/top8/BF16. Each FP32 variant's receipt-bound PTX contains four static
`red.relaxed.gpu.global.add.v2.f32` sites and no BF16 reduction sites. The BF16
variant contains four `red.relaxed.gpu.global.add.noftz.bf16x2` sites and no
FP32 vector reduction sites. These are static instruction counts, not executed
operations, traffic, numerical correctness, or performance measurements.

Both EP FP32 variants contain a new `bar.sync 1, 128` between their final
shared output store/proxy fence and the first scalar scatter load, in **all four
statically unrolled copies**. The existing post-scatter barrier remains in each
copy. M32's first copy has store end3464, pre-barrier3466, loads3508/3519,
RED3528, post-barrier3534; M64/top1 has4986,4988,5030/5041,5055,5061 respectively.
The unselected BF16 variant retains its previous post-only barrier placement.
Exact instruction text, line numbers and each PTX hash are recorded under
`verification.json/publication_sites`; `verify.py` reproduces these checks.
This demonstrates emitted synchronization, not successful GPU race/numerical
validation. The M32 current-tile bounds are covered by the focused CPU source
oracle included in the85-test gate.

`evidence.tar.gz` preserves **61 original files**: 28 PTX, 28 cubin,
4 resource logs, and the result. `result.json` is byte-identical to its original.
`head/` preserves the submitted command, driver, exit receipt, and complete log.
The runner retained its 12 GiB host admission gate, 4 GiB memory / 2 CPU limits,
runc runtime, and no-device configuration.

Before and after collection, all original worker, copied head, and archived
file hashes matched. Both frozen checkouts were clean at the exact revision,
with full history and no alternates; actual worker source hashes matched the
submission and compiled receipts. The immutable image remained
`sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211`.
The worker's 116-file bindings capsule passed strict validation against
manifest `b29ac01b3a45a5c140ba205187c86411412c5b74e7ac31417747e028588debab`.

`capture.json` retains those observations and original hashes. `verify.py`
reproduces the exact artifact-set, source-receipt, completion, and static PTX
checks from local evidence; its output is `verification.json`. Collection
only read/copied existing results after GPU14 submission was confirmed. It did
not submit tests, touch GPU ownership, or modify frozen sources. This archive
establishes CPU evidence only; GPU numerical and serving acceptance remain
separate. `SHA256SUMS` covers every archive file except itself.
