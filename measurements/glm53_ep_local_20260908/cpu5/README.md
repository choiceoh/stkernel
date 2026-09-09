Final no-device CuTe compilation on srv4, normal fleet CPU lane, 2026-09-08.

Source a848630b0ba14b2212b068f19c5408feb7d58abe; immutable serving image
sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211.
The candidate ran in a runc container with no network/devices, 4 GiB memory,
2 CPUs and a 12 GiB host-MemAvailable guard. Actual CuTe compilation and all
23 pinned CPU tests passed without skips; CUDA remained uninitialized.

| Candidate revision | Registers | Stack bytes |
| --- | ---: | ---: |
| original cpu3 | 168 | 1520 |
| intermediate cpu4 | 168 | 1552 |
| final cpu5 | 168 | 1040 |

The final edit folds transformed-scale equality into the existing load loop
and keeps eight cache slots for the exact top8 contract. It changes neither
quantization arithmetic nor inherited FC1/Q1/FC2/task storage. The compiled
stack shrank by 480 bytes (31.6%) relative to the original candidate. This
does not establish occupancy, GPU correctness, latency or direct TTFT gains.

The stock generic E72/I2048 kernel was not compiled again: all recorded
compiler sources except the candidate-only moe_dynamic_ep_local.py are
byte-identical to ../cpu4/, where stock passed compilation and all 23 CPU
tests with 255 registers / 432 stack bytes. The complete local core run at
cpu4 is retained there; this final scale-loop change is covered by actual
CuTe compilation and the same 23 focused tests.

local/result.json binds all probe/test contracts and mounted MoE source
hashes. The GPU runner requires this receipt before service inventory/pause
and rechecks installed kernel hashes in its container. worker-sha256.json
covers original worker results, resources and PTX; decompressed PTX hashes
were verified after transfer. Cubin hashes are recorded; binaries remain on
srv4. Head/worker job: /tmp/glm53-ep-local-compile0908-5. Frozen source:
/home/choiceoh/stkernel-ep-local-0908-5. The CPU job completed with exit 0.
