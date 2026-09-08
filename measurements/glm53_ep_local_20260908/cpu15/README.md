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
