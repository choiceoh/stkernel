# V5 post-context CUDA binding diagnostic

**The minimal GPU diagnostic reproduced all 34 CUDA API errors without MoE or
CuTe.** The API calls themselves returned success: device count 1 and driver
API 13000. Compute Sanitizer still exited 86; the outer runner exited 1.
This is a failed sanitizer diagnostic, not a numerical or performance pass.

The normal fleet gave `epbindinggpu0908v5` GO at 19:51:06 KST on 2026-09-08.
The frozen source was `8dc665b018d1d86286c7d72f4d01a368581be106`, bound to its
CPU11 63-test receipt. The immutable image and sanitizer match the submission
and preflight records. Later CPU12/13 kernel changes are outside this API-only
diagnostic's scope. No errors were suppressed or waived.

In [memcheck.log](capture/memcheck.log.gz), lines 2–3 show Torch context creation
and synchronization complete. All 34 `cuGetProcAddress_v2` errors occur at
lines 5–718 after `EP_BINDING_DEVICE_COUNT_BEGIN` and before
`EP_BINDING_DEVICE_COUNT_END` on line 719. Their Python stack refers to
`glm53_ep_binding_check.py:27`, the first `cuDeviceGetCount()` call. Line 721
reports 34 errors. [binding.json](capture/binding.json) records successful
count/version returns, cuda-bindings 13.3.1 and CUDA initialization false→true.
`OBSERVED` means the API observations completed; it does not override exit 86.

These observations localize the sanitizer reports to initialization inside
the first binding call, and demonstrate that the MoE kernel is unnecessary
to reproduce them. The log does not name the requested lookup symbols,
versions or flags, so it alone cannot prove a version-mismatch cause.

The subsequent [installed-binary mapping](installed-loader/README.md) resolves
all 34 reported call sites to requests above driver API 13000: nine at 13010,
twelve at 13020 and thirteen at 13030, all flags 0 and result pointers NULL.
The binary's size/hash matches its package RECORD and the same immutable
image. An independent rerun matched every symbol/version pair to NVIDIA's
13.3.1 loader template. This is direct static evidence of the request-version
mismatch, while a corrected runtime remains untested.

The [compatibility investigation](compatibility/research.md) identifies the
official 13.0.3 bindings loader as a candidate with no requests above 13000.
A normal fleet CPU-only [dependency inventory](dependency-inventory/metadata.json)
read all 268 installed distributions without importing Torch/CUDA or exposing
device nodes. Torch, Cutlass and FlashInfer's relevant version constraints
allow 13.0.3; cuda-python's metapackage must move with cuda-bindings. Existing
unrelated dependency conflicts remain distinct from this proposed change.
This preparation does not establish import, compile or GPU compatibility.

All four original public containers were running at admission. Their
container IDs, image, configuration, host configuration, mounts, overlay
hashes and manifest were identical in the original and restored snapshots.
All four were running again. `restored.json` was written at
19:54:31.259640 after the pinned lifecycle's exact identity/running-state and
public health checks. The payload finished at 19:54:31.381560; normal handoff
to `attr3-0908` was accepted at 19:54:31.902388 and the driver finished at
19:54:32.044687. Restoration preceded the handoff.

The separate later public snapshot has no HTTP 200 observation while the next
holder is booting. It does not describe this diagnostic's restore result.
[verification.json](verification.json) independently checks exact identity,
state, error boundaries, log hash and restore-before-handoff ordering.

All 21 original job files were hashed remotely, transferred and re-read for
matching hashes. [source-manifest.json](source-manifest.json) records their
original paths, lengths, timestamps and digests. The fleet lifecycle excerpt
has separate full-source and excerpt hashes. Container environment and
commands are represented by hashes in inventory snapshots.

There is no queued retry and no new MoE, racecheck, transport or TTFT result.
The next experiment must change an evidence-supported compatibility condition
and retain the same unsuppressed sanitizer gate.

The original fleet and memcheck logs are gzip-compressed without changing their uncompressed bytes. Source-manifest sizes/hashes and cited line numbers refer to the decompressed originals.
