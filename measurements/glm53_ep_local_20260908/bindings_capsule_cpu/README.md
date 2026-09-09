# Pinned CUDA bindings capsule: CPU evidence

**Attempt 2 passed the no-device metadata and import check through normal
`fleet run --cpu` on 2026-09-08, 20:50:19–20 KST.** This establishes that the
official cuda-bindings/cuda-python 13.0.3 capsule imports in the pinned Linux
image with the recorded dependency conditions. It does not establish CUDA API,
sanitizer, compilation, kernel correctness, throughput or TTFT compatibility.

The successful session was `epbindingscapsule0908v2`, frozen at source
`63f56a54dbb91e994456b519a53b679963a93583`, with scheduler revision
`cf405a7e03bd8ac95933d76f590fdfc4b0685e99`. The immutable image was
`sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211`.
The [submission](attempt2/cpu2-submission.json),
[normal-fleet exit](attempt2/cpu2-exit.json), [wrapper receipt](attempt2/receipt.json)
and [inner result](attempt2/result.json) retain the exact command, source hashes
and exit-zero evidence.

The container used `runc`, no network, 4 GiB memory and two CPUs. The existing
12 GiB available-host-memory guard remained in force. Source and wheels were
mounted read-only; Python bytecode writes, user-site packages and inherited
`PYTHONPATH` were disabled. The result records no exposed accelerator device
nodes, no Torch import and no CUDA API or context-query calls by the probe.
Serving was not paused for this CPU check.

## What passed

The complete original metadata snapshot contains **268 records**. Actual
`importlib.metadata.distribution(canonical_name)` resolution selects **264
effective records**, with **four shadowed records** retained in full. Each
chosen path must match exactly one saved record, including its version,
requirements, metadata text and SHA256. The helper does not choose the newest,
first or last version from the snapshot. The compressed
[resolution record](attempt2/distribution-resolution.json.gz) preserves the
selected and shadowed records, original indices and Python search path.

The dependency check substituted only cuda-bindings and cuda-python 13.0.3.
It found no introduced conflicts and no selected-package conflicts. Two
pre-existing conflicts remain explicitly reported: FlashInfer requests
`nvidia-cutlass-dsl==4.7.0` while 4.6.2 is installed, and vLLM requests
`flashinfer-python==0.6.17` while `0.6.18.dev20260819` is installed. Thus the
metadata verdict does not mean the base image is free of dependency conflicts.

The actual imports of `cuda.bindings`, `cuda.bindings.driver` and
`cuda.bindings._bindings.cydriver` resolved to the capsule. Their file paths,
sizes and hashes matched the pinned wheel manifest, and both selected
distribution metadata versions were 13.0.3. `cuda.pathfinder` remained the
original base 1.7.0 module and metadata, with unchanged paths and hashes. Full
capsule validation passed before and after imports, including rejection of
extra files such as `__pycache__`.

The capsule manifest SHA256, fixed by the earlier local staging receipt and
matched by both CPU attempts, is
`b29ac01b3a45a5c140ba205187c86411412c5b74e7ac31417747e028588debab`.
[Wheel provenance](local/wheel-downloads.json), the
[local staging receipt](local/local-stage.json) and the
[CPU manifest](attempt2/capsule-manifest.json) identify the two exact official
wheels and all 116 extracted files (113,268,092 bytes). Local extraction alone
did not import Linux binaries; the successful CPU attempt supplies that
separate import evidence.

## Retained failure and local contracts

[Attempt 1](attempt1/result.json), source
`b4d4abecf3ca6430bd9c5c7bf27f4ebb5b9477c7`, exited 1 before importing the
capsule. Its full snapshot included multiple installed distributions named
cryptography, PyJWT and six, and the strict dependency checker rejected the
duplicate names. This was a metadata-inventory failure, not a failed binary
import. The original FAIL receipt, log and snapshot are retained. Attempt 2
added explicit runtime metadata resolution; it did not relax the dependency
checker's duplicate rejection or alter the pinned wheels.

The local contracts have separate scopes and evidence:

| Suite | Result | Evidence and scope |
| --- | --- | --- |
| Capsule | 12 PASS, zero skips | [Receipt](local/capsule-tests.json), [log](local/capsule-tests.log): wheel/manifest and dependency contracts. |
| Binding pair | 15 PASS, zero skips | [Receipt](local/pair-tests.json): captured tool-output evidence for pair contracts; no original redirected log was saved. This is not a live GPU pair. |
| Resolution and CPU wrapper | 13 PASS, zero skips | [Receipt](local/resolution-tests.json), [log](local/resolution-tests.log): lookup binding, shadow-record preservation, failure paths and wrapper contracts at the successful source revision. |

The older [10-test wrapper log](local/wrapper-tests.log) is historical evidence
for the earlier draft. It overlaps the later 13-test suite and is not added to
it as independent coverage. These local contracts and the CPU import check
are not combined into a GPU or performance pass.

## Baseline identity and archive boundaries

[Baseline identity](baseline-identity/receipt.json) records the original
13.3.1 driver binary's stable container/image identity, size and hash.
Its [verification](baseline-identity/verification.json) confirms the binary
matches the archived [package RECORD](baseline-identity/RECORD). The two
original [distribution metadata files and collection report](baseline-identity/metadata/report.json)
are also retained. These are baseline provenance for a later comparison;
they do not constitute a completed baseline/candidate GPU run.

Large metadata JSON files are stored as gzip archives without changing their
uncompressed bytes. The original lengths and SHA256 values in each attempt's
`source-manifest.json` refer to the decompressed files, not the `.gz` bytes.
To verify one, decompress it without rewriting the JSON and compare the
resulting byte count and SHA256 to the entry for its original `.json` path.
The result receipt also binds the original metadata and resolution hashes.
The copied capsule manifest is named `capsule-manifest.json` here; its source
manifest entry retains the original `capsule/capsule-manifest.json` path.

The [attempt-2 source manifest](attempt2/source-manifest.json) records verified
remote/local transfers. Frozen runtime source trees and the full extracted
capsules remain outside this evidence archive: attempt 2 used
`/home/choiceoh/stkernel-ep-binding-gpu0908-6b` and
`/tmp/glm53-bindings-capsule-cpu0908-2/capsule`; attempt 1 retains its separate
`...-6` source and `/tmp/glm53-bindings-capsule-cpu0908-1` output. Downloaded
wheels, the local full capsule and copied baseline binaries are likewise not
committed here. Their original locations and identities are recorded in the
linked receipts. This archive contains their provenance, manifests, metadata,
logs and results, without treating source or binary copies as runtime proof.
