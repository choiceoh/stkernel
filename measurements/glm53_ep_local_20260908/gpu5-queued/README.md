# Full GPU v5 queued

Normal preflight passed and session `eplocal0908v5` entered the GPU queue at
22:12:31 KST, first behind `arconsumer0908v17`. The captured state has no
GPU payload/capture directory or completion yet. The holder's original
20-minute estimate places its nominal boundary around 22:25 KST; it is not
a guaranteed start time.

Frozen source: `/home/choiceoh/stkernel-ep-local-gpu-0908-5b` at
`d53fd44f2b8a69869cddc9df10fa4cbb21bc56d9`. Job:
`/tmp/glm53-ep-local-gpu-0908-5`, driver PID79637. The complete CPU16 receipt
is bound to all 13 mounted and 27 contract sources, 134 passing tests and
24 remap compilations with the exact capsule runtime. Normal fleet owns
admission, queue order and GO-time resource/lifecycle checks.

The initial preparation failed before creating a GPU job because another
shared clone exceeded Git's alternate-object-store nesting limit. Its
partial source `...gpu-0908-5` is preserved. The new `5b` source fetches the
explicit immutable revision into an independent shallow repository; it has
no alternates and retains CPU16-parent ancestry. No frozen source was edited,
no job duplicated and no GPU queue was bypassed.

The full suite covers remap bytes, MoE numerics, changed-address contents,
nondefault streams, memcheck and racecheck, with exact incoming restoration.
It is eager component validation, not CUDA graph or full-model TTFT proof.
The queue receipt itself is not a GPU result. Inspect exit.json and capture
completion before any follow-up; do not resubmit or edit the frozen tree.

The original fleet log is stored as deterministic `fleet.log.gz`; decompress
it before checking its original-byte hash in `snapshot-manifest.json`. This
preserves its whitespace exactly. `SHA256SUMS` covers stored archive bytes.
