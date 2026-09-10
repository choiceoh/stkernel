# CPU11: compilation and CPU contracts passed

Normal fleet session `epdecodecpu0909v11` completed on 2026-09-09 at
09:25:46–09:26:01 KST. Payload, evidence copy, and outer return codes were 0.
Frozen source: `c7dec80a0f73d4a2b683ce2c4813978938694095` at
`/home/choiceoh/stkernel-ep-onepass-0909-11`.

**69 tests passed, with zero failures, errors, or skips.** The original result
records `verdict=PASS`, `phase=complete`, `cuda_initialized=false`, and
`binding_runtime_rechecked=true`. Three CuTe kernels (EP micro M32, stock
micro M64, EP prefill FP32 v2) and all 24 Triton preparation variants compiled.
This is CPU compilation/contract evidence, not GPU numerical or speed proof.
The original CPU10 failure remains separately preserved in `../decode10-cpu-failed/`.

The srv2 fleet ran the CPU payload on srv1 through SSH, using the existing
no-device runner with 4 GiB memory, 2 CPUs, and the unchanged 12 GiB host gate.
Admission recorded 17,420,712 KiB available. The immutable image was
`sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211`;
the 13.0.3 bindings capsule manifest was
`b29ac01b3a45a5c140ba205187c86411412c5b74e7ac31417747e028588debab`.

`evidence.tar.gz` preserves all **58 original files**: 27 PTX, 27 cubin,
3 resource logs, and the result. The separate `result.json` is byte-identical
to its archived original. `head/` preserves the submitted command, driver,
exit receipt, and complete fleet log.

Collection verified identical file hashes on the worker, its head-side copy,
and the local tar archive, before and after copying. Both frozen source
checkouts were clean at the expected revision, with full history and no
alternates. Worker source hashes matched the submission and compiled receipt.
The worker's immutable image ID and the capsule's strict 116-file validation
also passed during both collection snapshots. These are post-run observations;
the runtime's own final recheck remains recorded in the original result.

`capture.json` retains the original per-file hashes and observations.
`verify.py` reproduces the exact artifact-set, raw-byte, source-receipt, and
completion checks using only local archived data; its output is
`verification.json`. `collect.py` only read/copied existing evidence.
No tests, compilations, GPU calls, source edits, or fleet mutations were
performed for this archive. `SHA256SUMS` covers every file except itself.
