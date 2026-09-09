# CPU10: compilation completed; CPU contracts failed

**Overall FAIL is preserved.** Normal fleet session `epdecodecpu0909v10`
ran on 2026-09-09 at 09:20:30–09:20:49 KST, with payload/outer return code 1.
Frozen source was `98df4587e448c7587b334aeac31c0fb24f656c12` at
`/home/choiceoh/stkernel-ep-onepass-0909-10`. The srv2 fleet executed its CPU
payload over SSH on srv1; no worker fleet state was created.

The pinned no-device runner used image
`sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211`
and the 13.0.3 bindings capsule manifest
`b29ac01b3a45a5c140ba205187c86411412c5b74e7ac31417747e028588debab`.
Worker admission recorded 17,998,872 KiB available against the unchanged
12 GiB gate; the container retained its 4 GiB / 2 CPU / runc / no-device limits.

All three CuTe kernels and 24 Triton preparation variants emitted PTX and
cubin artifacts before the test phase failed:

| CuTe kernel | Registers | Stack bytes | Shared bytes | Cubin bytes |
| --- | ---: | ---: | ---: | ---: |
| EP micro M32 | 128 | 0 | 1024 | 87864 |
| Stock micro M64 | 220 | 0 | 1024 | 116600 |
| EP prefill FP32 v2 | 168 | 112 | 1024 | 304536 |

CPU contracts ran **69 tests: 3 errors, 0 assertion failures, 0 skips**.
All three errors came from subcases of
`test_apply_defers_the_capture_query_until_candidate_admission` in
`tests/test_glm53_ep_prefill_local.py`: the existing `SimpleNamespace` fake
owner lacked `_sf6_weight_views`, which the SF6 wrapper now reads. The raw
tracebacks remain in `head/fleet.log`. This is a CPU fixture failure, not a
CuTe lowering failure, and it still prevents overall CPU acceptance.

`result.json` is an exact copy of the original failed receipt; it remains
`verdict=FAIL`, `phase=cpu-contracts`. Its final runtime recheck and
`cuda_initialized` fields were not reached/recorded. The initial pinned
runtime identity is preserved, but this archive does not manufacture a
successful final postcheck. No GPU numerics or throughput were tested.

`evidence.tar.gz` contains all **58 original worker files**: 27 PTX, 27 cubin,
3 resource logs, and the result. Every original file hash matched before
and after transfer and every archived compiler artifact matched its receipt.
Head submission, driver, exit, and fleet log bytes are preserved separately.
Both source checkouts remained clean at the exact revision, with full history
and no alternates. `capture.json` retains those checks and original file hashes;
`verification.json` records the bounded artifact/source reconciliation.

The read-only `collect.py` copied existing evidence without submitting work,
changing source, or rerunning tests. `SHA256SUMS` covers every archive file
except itself. No failed receipt was rewritten or relabeled PASS.
