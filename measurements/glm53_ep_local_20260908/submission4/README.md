Normal fleet GPU admission: eplocal0908v4, 2026-09-08 15:54:23 KST.
Preflight passed. At this archived snapshot the session is first in the queue
behind holder arconsumer0908v6. The scheduler estimates a 17:03 KST start;
this is not a promised time. No GPU payload or cell has started.

Frozen execution source 71e804e7aa6b29d6ddf4577809a5fa5e05a999e6 is clean
at /home/choiceoh/stkernel-ep-local-gpu-0908-4 on the head. Every mounted
MoE source and probe/test contract matches cpu7: 37 tests without skips,
actual E72/I2048 CuTe and 24 remap specializations compiled without a device.
Driver PID 1445785; job /tmp/glm53-ep-local-gpu-0908-4. Read exit.json and
capture/completion.json before any retry. Do not change the frozen source.
Subsequent commits only record this admission.

The runner binds the compile receipt and launches the pinned sanitizer
version preflight inside the immutable image before service inventory/pause.
The corrected sanitizer directory is mounted read-only into sanitizer cells;
exit zero also requires the matching zero-error/hazard summary. The head
no-device launch smoke and binary identity are preserved in cpu7.

Seventeen isolated GPU cells cover the 24-specialization byte remap oracle,
eight MoE fixtures, and remap/balanced4096/remote4096/zeros4097 under each of
memcheck and racecheck. Both timed MoE arms include their remap; inputs,
routes and scales change in place, with output poison and nondefault streams.
Exact incoming-container recovery remains part of the normal fleet lifecycle.
The old v2 component measurements exclude remap and do not validate this
source. v3 was cancelled before execution; its receipts are in submission3.

Both experimental defaults remain off. GPU proof is pending, and component
results cannot establish production TP4 improvement or full-model TTFT.
sha256.json covers original bytes, including decompressed logs and status;
state.json records collection time, clean source and no-payload checks.
