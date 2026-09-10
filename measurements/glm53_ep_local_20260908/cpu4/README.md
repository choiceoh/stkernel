Actual no-device CuTe compilation on srv4, normal fleet CPU lane, 2026-09-08.

Source 7a16fc51587a2be4765a268b0ff52977c460906c; immutable serving image
sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211.
Both arms ran in separate runc containers with no network/devices, 4 GiB
memory, 2 CPUs and a 12 GiB host-MemAvailable guard. Both compiled, ran all
23 pinned CPU tests with no skips and left CUDA uninitialized.

| Arm | Kernel | Registers | Stack bytes |
| --- | --- | ---: | ---: |
| stock | MoEDynamicKernel (generic E72/I2048) | 255 | 432 |
| local | MoEGatedEPLocalKernel | 168 | 1552 |

This intermediate candidate reduced scale-cache capacity to the admitted
top8 and moved scale equality out of the quantization-block loop. Its stack
increased by 32 bytes from cpu3, so a subsequent refinement folded equality
into the scale-load loop. The final candidate is recorded in ../cpu5/.
These are compiler resource diagnostics, not GPU speed or numerics results.

The receipt binds the probe, lifecycle, CPU tests and all mounted MoE files
to their tested hashes. The 23 tests include real CPU Torch per-row error
controls, actual apply-method lazy capture dispatch and failure before any
service pause on stale compile evidence. The final local core logic run at
this revision passed 6783 checks, 38 megakernel regressions and 120 fleet
regressions; core-logic.json and the compressed original log record that
separate host check, whose tensor-dependent skips are not counted as passes.

worker-sha256.json covers original worker results, resources and PTX. PTX is
gzip-compressed; original-byte hashes were verified after transfer. Cubin
hashes are in each result.json; binaries remain on srv4. The head coordinator
job is /tmp/glm53-ep-local-compile0908-4 and the frozen source is
/home/choiceoh/stkernel-ep-local-0908-4. Both arms completed with exit 0.
