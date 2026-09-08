# CPU14 candidate: local contracts pass, compiler not started

Kernel/source commit `e0a2c2987cc37389a85ab5c86c3b9a15dfccccd4` adds two
prefill optimizations: reject non-local expert IDs before histogram/producer
weight loads, and use the existing plain u64 Q0 store across the admitted
T4096–16384 domain. Calculations, valid special-value semantics, route order,
row allocation, shared storage, atomics and synchronization are preserved.
The exact inherited adaptive helper source SHA993783… was independently
reviewed: its T>2048 arm is the same plain store operation.

Focused actual-source CPU suites pass **8 route-cache + 10 publication tests**,
zero skips. They include three new tests for poisoned invalid-route weight
reads, valid NaNs/infinities/subnormals/signed zeros and duplicate routes,
and exact Q0 store addresses/payloads across the dispatcher admission domain.
The source/log hashes and original commands are in `local-tests.json`.
Independent read-only review found no blocker. Overlay composition and
`git diff --check` also passed.

The normal head-only CPU14 preparer refused admission with 6,391,256 KiB
available (6.10 GiB), below the unchanged 12 GiB host guard. **No CPU14
source clone, compiler container or fleet job was created.** The intended
job is `/tmp/glm53-ep-local-compile0908-14-head`; it is not a running PID to poll.
`preparation.json` is a refusal receipt, not compiler evidence. No serving
memory was reclaimed and no worker fallback or GPU action occurred.

The expected pinned suite count is 71 (previous 68 plus three new cases),
but it has not run. CuTe import/lowering, generated conditional loads and
plain stores, register/stack changes, GPU numerics/sanitizer and speed remain
unverified. The full offline runner now requires `cpu14/local/result.json`
and refuses without matching evidence. The separate queued binding v6 uses
frozen63f56a54 and its still-matching CPU13 receipt; this kernel change does
not mutate that freeze or invalidate that diagnostic's source binding.

`prepare.py` is the exact guarded preparer used. Retry only after checking
memory and existing state; its committed revision must equal current clean
HEAD. Preserve every old frozen source and receipt. Compiler artifacts will
be archived here only after an actual normal CPU run completes.
