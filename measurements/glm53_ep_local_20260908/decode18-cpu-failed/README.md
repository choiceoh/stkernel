# CPU18: shared-path temporary type join rejected

Normal fleet CPU session `epdecodecpu0909v18` ran on srv1 through the srv2
scheduler on 2026-09-09 11:15:19–11:15:25 KST. Frozen source was
`ce7f222b0ae86124ccd177cfcf10975e411c6979` at
`/home/choiceoh/stkernel-ep-onepass-0909-18`. Payload and outer return codes
were 1; the success-only evidence copy was not performed.

The original result is **FAIL**, phase `micro-cute-compile`, with
`micro_passes=[]`. The first M32 candidate did not finish lowering. CuTe
reported `TYPE_UNSTABLE_JOIN` at `moe_micro_kernel.py:2310`: `k_next` has
`None` type on the shared FC1 path and `int` on the legacy path entering the
common FC2 output loop. The full emitted source trace is preserved in both
`result.json` and `head/fleet.log`. No PTX, cubin, or resource files were
produced. The planned 112-test suite and remaining kernels were not reached.
Final CUDA-initialization and runtime-recheck fields are absent.

The subsequent source correction initializes `k_next = 0` inside only the
shared FC1 branch, preserving the existing FC2 arithmetic and false path.
This archive does not establish successful actual lowering of that correction.

`evidence.tar.gz` preserves the worker's sole original result (8,883 bytes),
duplicated byte-for-byte in local `result.json`. `head/` retains original
submission, driver, exit receipt, and the complete fleet log. Before/after
collection verifies clean/full/no-alternates source checkouts, source hashes
matching the submission and receipt, immutable image
`sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211`,
and the strict 116-file bindings capsule at manifest
`b29ac01b3a45a5c140ba205187c86411412c5b74e7ac31417747e028588debab`.
Existing 12 GiB admission, 4 GiB / 2 CPU limits, and no-device runtime remain.

`capture.json` retains observations and hashes. `verify.py` reproduces the
local integrity checks in `verification.json`; its PASS concerns archive
integrity only, while the original CPU result remains FAIL. Collection did
not compile, run tests, use GPUs, alter frozen sources, or submit a job.
`SHA256SUMS` covers every archive file except itself.
