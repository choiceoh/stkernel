# CPU15: full MoE capsule integration

The compiler and every MoE/remap GPU cell now require the proven 13.0.3
capsule. Outer CLI inputs are mandatory, and output/source/capsule overlap is
rejected before creating outputs. The capsule is mounted read-only at a fixed
path with Python -B, exact PYTHONPATH and disabled user-site/bytecode writes.
Actual binary, paired METADATA and base pathfinder identities must match the
preserved CPU2 and v6 candidate evidence. Full tree validation runs before
and after work. The original image and kernel arithmetic are unchanged.

Compiler admission now requires an explicit PASS/complete verdict, a completed
runtime recheck, no error fields, current source hashes, and the exact capsule
runtime receipt. Full artifact fields cannot override a failed final recheck.
Every GPU cell compares the same receipt before accelerator work and after it.
The original numerical controls, sanitizer checks and exact lifecycle recovery
remain in place. A primary operation failure survives a secondary integrity
failure and both are recorded.

Local focused suites pass 7 runtime, 10 inner and 13 wrapper cases, plus 15
existing pair cases. The adapted probe module passes seven cases and retains
four explicit host-Torch skips. These do not replace the pinned compiler
suite. Independent read-only review passed after repairing failed-recheck
admission and package/CLI import compatibility. `local-contracts.json` binds
source/log hashes; no actual CPU15 compile or GPU result is claimed here yet.

CPU14's old-image compiler output is preserved. The full offline runner now
requires `cpu15/local/result.json`, which is only created by a completed
capsule-bound compiler run. The guarded preparer uses a new numbered immutable
source/job and the normal --cpu lane with the original 12 GiB host guard and
4 GiB/two-CPU/no-device limits. GPU queue admission follows actual compiler
PASS and archived source/runtime evidence.

Source is committed at `30608530f0c908bc5f81db6bdb69fd70ccee9a24`. The new
receipt binds 27 contract files and 13 test modules; the expected pinned
suite count is 129, but that suite has not run in this archived snapshot.
A fresh host import of `probes.glm53_ep_local_evidence` also passes without
adding the probes directory to sys.path.

The initial head-only preparer refused 8.68 GiB available against its unchanged
12 GiB guard, creating neither a source clone nor a compiler job. A bounded
readiness controller is now armed at `/tmp/glm53-cpu15-ready-0908`, PID4104331.
It checks only CPU readiness every 20 seconds until 22:43:26 KST, and submits
one normal `fleet --cpu` job if the existing guards pass. It never reserves a
GPU, changes queue order, stops services, reclaims memory or falls back to a
worker. A fatal preparation error stops the controller; it does not blindly
retry a submitted/failed job. Its prepared immutable bundle SHA256 is
`3edc1053293946cbf78e154b266155f6f93a0a553bf3a90e0fe05e961da17242`.

At 21:45:26 the state was WAITING_MEMORY, available 8.60 GiB. This is not CPU
compile or GPU queue admission. Read the controller's live state before any
follow-up to avoid duplicates. When submitted, the normal job will be
`/tmp/glm53-ep-local-compile0908-15-head` and frozen source
`/home/choiceoh/stkernel-ep-local-0908-15`, both at the committed source above.
`readiness/` preserves the actual controller, guarded preparer, launch,
initial refusal and a timestamped state snapshot; mutable live files were
read into byte snapshots before local hash verification.

## Completed state

The readiness controller submitted CPU15 at 21:53:07 KST and exited. The job
finished at 21:53:21 with one contract-test error after actual CuTe and 24
remap compiles. See [preserved failure](failed-compile/README.md). CPU15 is
complete and must not be retried or treated as GPU admission proof.
