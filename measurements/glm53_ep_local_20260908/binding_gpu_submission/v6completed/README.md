# Binding v6 completed compatibility pair

Normal fleet v6 received GO at **21:02:53 KST** on 2026-09-08 and completed
its pair at 21:03:08.575426. Frozen source `63f56a54dbb91e994456b519a53b679963a93583`
ran two fresh same-image processes with the exact CPU2 capsule manifest.

| Arm | Binding version | API lookup errors | Sanitizer exit | Cell verdict |
|---|---|---:|---:|---|
| Installed baseline | 13.3.1 | 34 | 86 | FAIL |
| Isolated paired capsule | 13.0.3 | 0 | 0 | CLEAN_DIAGNOSTIC |

Both first created a Torch context, then verified actual imported binary and
metadata identities before the first binding device-count call. Both returned
one device and driver API13000. The baseline's exact 34-error window matches
v5; the candidate has the literal unsuppressed `ERROR SUMMARY: 0 errors`.
All log and JSON hashes match the completion receipt. Capsule contents were
validated before, between and after the cells.

The pair verdict is **COMPATIBILITY_OBSERVED**. Baseline failure remains a
failure; diagnostics/process/outer exit is 1. No MoE, CuTe, full-model,
transport or prefill performance was measured. The successful 13.0.3 minimal
API result allows preparing a full source-bound capsule compiler and GPU
suite; it does not itself pass that suite or authorize a default change.

Incoming state was four uniformly stopped pinned containers. All complete
before/stopped/restored records match exactly, including original IDs,
configuration/host-configuration hashes, mounts, overlays, manifest and
running=false. They were kept stopped, as required by the incoming contract.
Restoration preceded completion and normal handoff; the driver ended at
21:03:09.251673 KST. Later live changes belong to the subsequent holder's
interval and are not v6's restored snapshot or current health proof.

The normal queue is now released and driver3778025 is done; never repoll or
resubmit v6. `snapshot.json` hashes all 31 original job files, independently
verified after local transfer. Large metadata JSON and raw logs are gzip with
mtime=0; decompress to check original sizes/hashes. `submit.py` preserves the
exact preparer and `driver.py` the exact remote payload. The original queued
snapshot remains separately archived under v6queued.
