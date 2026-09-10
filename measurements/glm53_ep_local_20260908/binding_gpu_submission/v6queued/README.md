# Binding v6 queue submission

Normal fleet accepted `epbindinggpu0908v6` at 2026-09-08 20:52:34 KST.
Frozen source: `/home/choiceoh/stkernel-ep-binding-gpu0908-6b`, revision
`63f56a54dbb91e994456b519a53b679963a93583`. Driver PID 3778025;
job `/tmp/glm53-ep-binding-gpu0908-6`. The clean scheduler and advertised
origin/main were `cf405a7e03bd8ac95933d76f590fdfc4b0685e99` with approved
restore-fixture ancestry. Source, capsule and receipts were rechecked before
normal queue admission; no inherited FLEET overrides were retained.

At 20:54:03 this was queue position 1 of 1 behind `arconsumer0908v14`,
which started at 20:32 with a 45-minute estimate. The resulting 21:17 estimate
is conditional, not a guaranteed start. This archive records submission only.
No v6 GPU payload or restoration result existed at the snapshot.

The pair runs fresh installed 13.3.1 and isolated paired bindings/cuda-python
13.0.3 processes in the same immutable image under one pause, with an
8-minute reservation estimate. Each process retains context-before-binding
ordering and unsuppressed memcheck, and is bounded to 180 seconds.
All original container identity/state restoration checks remain enforced.
The CPU2 capsule manifest is externally pinned to
`b29ac01b3a45a5c140ba205187c86411412c5b74e7ac31417747e028588debab` at
`/tmp/glm53-bindings-capsule-cpu0908-2/capsule`.

CPU13's 68-test/13-mounted/18-contract/24-remap receipt still matches the
unchanged kernel sources at this freeze. Separate CPU2 PASS binds the new
capsule imports and metadata: neither substitutes for GPU acceptance.
Baseline failure remains FAIL and outer exit 1 even if the separate
`COMPATIBILITY_OBSERVED` verdict records a clean candidate with an exact
34-error baseline reproduction. Full MoE/CuTe and prefill timing are absent.

`snapshot.json` contains original file sizes/hashes, queue and holder state.
All 15 original job files were transferred and verified against those hashes.
Large JSON and raw fleet log use gzip with mtime=0; decompress before checking
the original hashes. `submit.py` is the exact local preparer used, while
`driver.py` is its remote payload. Do not rerun a completed/submitted job.
