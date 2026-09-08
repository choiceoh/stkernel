# Baseline-only SF6 retest

The user requested a clean baseline after external traffic invalidated original
A/B baseline B. The original candidate source is
`85e25370779c2b8a6c9aaefa275b1dc8f27d60d0`; its raw decode was
20.281701533352084 step/s (49.30552785995582 ms/step). Original v1 remains INVALID.

## Preboot failure preserved

Session `sf6-base-0909v2`, ticket `17889043822842777`, preserved its queue order
behind `mb9221-0909` and was admitted at 06:56:37 KST. It failed at 06:56:40
with supervisor/payload return code 2 before any model boot or inference.
The deployment source-base guard required main PR #503, which added the composed
KV-zero file already byte-identical in this branch. The guard was not bypassed.
The terminal reservation, raw run log, observer state, inputs and actual exit
are preserved under `v2-preboot-failure/`.

## Corrected retry

Current main `c5c759bb5623c625a48aa45b392e0c3f89550225` is merged. All 63 serving
overlays and the profile, benchmark scripts and frozen runtime Reader remain
byte-identical to source85; see `overlay-equivalence.json`. The new commit also
adds a passive observer and evidence only. Its exact serving revision is pinned
at launch and retained in the v3 source/launch evidence. Preparation again
requires `origin/main` ancestry, a clean checkout, the standard CPU checks and
immutable deployment image. Queue priority is unchanged.

The corrected session is `sf6-base-0909v3`, with exactly one canonical
`bench/ab-lever.sh` baseline arm `sf6-base-0909v3B`. Common static settings are
`t,r`, with no SF6 lever. The argv/spec under `v3/` are the exact planned inputs;
a reservation receipt is required before claiming it is queued or running.

The serving API, onepass client and health checks use `127.0.0.1:18000`, isolated
from existing port 8000 traffic. This is not authentication against arbitrary
local processes, so the original exclusive request-counter checks still apply.
The endpoint differs intentionally from old A. The strict full A/B runtime
checker is unchanged: this retry yields an independently validated baseline
record, and any cross-run raw step comparison must disclose endpoint isolation
and the previous common MHC selftest failure.

Model/kernel workload is unchanged: KV target 1100000, hybrid 187, max length
1048576, required actual KV 665, SPEC_K=5, AR consumer PDL=1, MK PDL=1,
compact AR=0 and inline RDMA=0. Workload is standard 2K/32K/128K plus exclusive
fixed 3x2048 decode, with the four-node 10 GiB host-memory guard.

The passive observer starts rank observations only under the exact admitted
session/ticket/PID/start token, checks the actual serving host and port, and
retains same-boot before/after raw reports, logs and sealed receipts. Collection
success is separate from runtime validation; a recurring MHC failure stays FAIL.
It issues no inference, GPU probe, or serving lifecycle action. No candidate
rerun, default promotion or PR merge is included. Central idle controller owns
serving recovery after the reservation.

Remote checkout: `/home/choiceoh/stkernel-sf6-direct-0909`.
Remote v3 inputs: `/home/choiceoh/glm53-logs/sf6-base-0909v3-inputs`.
Remote v3 results: `/home/choiceoh/glm53-logs/SF6-BASE-sf6-base-0909v3`.
