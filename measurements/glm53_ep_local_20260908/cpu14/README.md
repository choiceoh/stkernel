# CPU14 no-device compilation

**Actual CuTe and all 24 Triton remap variants compiled; 71 pinned CPU tests
passed with zero failures, errors, skips or CUDA initialization.** Frozen
compiler source `881456a1f43fcf61de1bb5822b883dd2fd32e693` includes kernel/source
commit `e0a2c2987cc37389a85ab5c86c3b9a15dfccccd4`, which adds two
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

The first head-only CPU14 preparation refused admission with 6,391,256 KiB
available (6.10 GiB), below the unchanged 12 GiB host guard. `preparation.json`
preserves that refusal before any source clone or job existed. After the prior
GPU holder finished, available memory was 93,212,272 KiB (88.89 GiB). One normal
`eplocalcpu0908v14head` job then ran under the immutable image, runc, no network
or CUDA devices, 4 GiB/two CPUs and the same host guard. No serving memory was
reclaimed. Clean scheduler `5ba0747a739fea095b4590d0c07c9c5e438eb7fe` equaled
approved main. Completion was 21:04:38.680752 KST, exit0; driver3881187 is done.

The compiler retains REG168 / STACK112 / SHARED1024. PTX shrinks
941539→939211 bytes; cubin 298456→288664 bytes (3.28% smaller). All 24 remap
PTX hashes match CPU13. These are compiler properties, not latency savings.
The [Q0 instruction inspection](q0-load-store-inspection.md) binds both original
PTX/cubin hashes and all cited instruction windows. Eight static weight-load
sites remain (one histogram, seven producer copies), but all eight now follow
a valid-ID branch. Ten adaptive Q0 store sites become ten plain stores.
`verify_q0_load_store_inspection.py` reproduces the saved report without CUDA;
the root's independent reproduction log is preserved here.
This compile used the unchanged base-image bindings; the isolated 13.0.3
capsule still needs its own full CuTe/compiler/runtime source binding before
MoE GPU acceptance. The separate minimal v6 diagnostic passing does not supply
that kernel proof.

All 58 original compiler-job files were hashed remotely and after transfer.
PTX/cubin files are gzip-compressed; `source-manifest.json` records original
sizes/hashes. `root-verification.json` confirms the current 13 mounted and
18 contract sources match the 71-test receipt and both compiled artifacts.
The full offline runner selects `cpu14/local/result.json`. The completed
binding v6 used its independent frozen63f56a54 and matching CPU13 receipt.
GPU numerics, sanitizers and matched performance of CPU14 remain pending.

`prepare.py` is the exact guarded preparer used. CPU14 is complete; do not
retry or poll its old job. A changed capsule environment needs a new numbered
source/job and its own compiler/import evidence. Preserve all old frozen
sources and receipts, and do not repeat unchanged tests or compilation.
