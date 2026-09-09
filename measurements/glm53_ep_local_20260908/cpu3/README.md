Actual no-device CuTe compilation on srv4, normal fleet CPU lane, 2026-09-08.

Source 083d937f676ce4efa200e253995f3505241ad283; immutable serving image
sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211.
Each arm ran in a separate runc container with no network/devices, 4 GiB
memory, 2 CPUs and a 12 GiB host-MemAvailable guard. Both report CUDA
uninitialized, generated PTX and actual compiled-cubin resource usage.

| Arm | Actual kernel | Registers | Stack bytes |
| --- | --- | ---: | ---: |
| stock | MoEDynamicKernel (generic E72/I2048) | 255 | 432 |
| local | MoEGatedEPLocalKernel (four-slice tasks) | 168 | 1520 |

These are compiler resource diagnostics, not occupancy or speedup results.
The larger candidate stack is a GPU measurement concern. Numerics, barriers,
sanitizers, transport, model quality and direct TTFT remain unmeasured.

The prior source 7746135 failed at the inherited gated entry point's four-slice
width check; no device kernel ran. 083d937 admits exactly E72/I2048 and always
publishes four-slice tasks, including at T4096. It does not enlarge Q1 storage.

worker-sha256.json covers original worker result JSON, resources and PTX.
PTX is stored gzip-compressed; decompressed byte counts and SHA-256 were verified
after transfer. Cubin hashes and paths are recorded in each result.json; binaries
remain in the remote CPU job directory. submission.json/exit.json/fleet.log come
from the head coordinator at /tmp/glm53-ep-local-compile0908-3.
