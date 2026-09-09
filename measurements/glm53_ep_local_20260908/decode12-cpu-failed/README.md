# CPU12: compilation completed; portable AST test failed

**Overall CPU FAIL is preserved.** Session `epdecodecpu0909v12` ran through
the normal srv2 fleet, with a no-device CPU payload on srv1, on 2026-09-09
09:51:55–09:52:17 KST. Payload/outer return code was 1. Frozen source was
`cb3f5c754a7c4a65a89affd0c436f4ea570014b0` at
`/home/choiceoh/stkernel-ep-onepass-0909-12` on both hosts.

All three micro CuTe variants and the dynamic FP32-prefill CuTe variant emitted
PTX/cubin/resource logs; all 24 Triton preparation variants also compiled.
The micro keys separately identify M32/top8/FP32, M64/top1/FP32, and the
unchanged M64/top8/BF16 path. Their register counts are 128, 220, and 220,
respectively, with zero stack and 1024 shared bytes. Dynamic prefill remains
168 registers, 112 stack bytes, and 1024 shared bytes.

CPU contracts ran **82 tests: 1 failure, 0 errors, 0 skips**. The sole failure
was `test_entire_kernel_math_routing_and_barriers_are_unchanged` in
`test_glm53_ep_micro_scatter_fp32.py`. It compared a Python-3.12 AST dump
against a digest generated with Python 3.14's different empty-field formatting.
The original c7 kernel AST reproduces both hashes locally merely by changing
`ast.dump(show_empty=...)`: false gives the expected `8525a206...`; true gives
the observed `f3448331...`. `ast-version-diagnosis.json` records this bounded
formatting diagnosis. It does not waive the original failed CPU gate.

`result.json` is byte-identical to the original and remains `verdict=FAIL`,
`phase=cpu-contracts`. Final `cuda_initialized` and
`binding_runtime_rechecked` fields were not reached/recorded. Compilation is
not GPU numerical, graph, throughput, or default-adoption acceptance.

`evidence.tar.gz` contains **61 original worker files**: 28 PTX, 28 cubin,
4 resource logs, and the result. Exact original and receipt hashes, the full
expected artifact set, and all three micro-pass mappings were checked. The
failed driver's success-only head evidence copy did not occur; collection
read the worker originals directly. `head/` preserves submission, driver,
exit, and complete fleet log bytes.

Before and after transfer, both frozen checkouts were clean at the expected
revision, with full history and no alternates. Worker source hashes matched
the submitted/compiled receipts. The pinned image ID remained
`sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211`;
strict validation of all 116 capsule files passed against manifest
`b29ac01b3a45a5c140ba205187c86411412c5b74e7ac31417747e028588debab`.
These collection observations do not replace the missing runtime postcheck.

`capture.json` preserves original hashes and source/image/capsule observations.
`verify.py` reproduces local archive reconciliation; `verification.json` is
its integrity result, explicitly retaining `original_verdict=FAIL`.
Collection did not submit workloads or modify frozen sources, services, or
fleet state. `SHA256SUMS` covers every archive file except itself.
