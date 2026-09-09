Actual CPU9 no-device compilation passed for source `cf1365b8fab6091833400efd7784071316eb9f6a` on 2026-09-08.
The 55 pinned CPU tests passed with zero skips, failures or errors; all 24 Triton
specializations and the E72/I2048 CuTe candidate compiled with CUDA uninitialized.

The first normal fleet CPU submission to srv4 exited 2 before launching the
container because available host memory was below the existing 12 GiB guard.
Its submission, driver, exit and failure log are retained in `srv4-refused/`.
The same frozen source was then run through the normal fleet CPU wrapper on
srv2/head with 92.45 GiB available before execution, using the same immutable
image, runc, 4 GiB memory, two CPUs, no network and no GPU devices. No serving
memory was reclaimed. The head job passed at 2026-09-08T17:10:38.244+09:00.

| CuTe compiler evidence | CPU8 | CPU9 |
|---|---:|---:|
| Registers | 168 | 168 |
| Stack bytes | 1040 | 112 |
| Shared bytes | 1024 | 1024 |
| PTX local depot bytes | 96 | 64 |
| Static local load instructions | 29 | 4 |
| Static local store instructions | 12 | 5 |
| PTX bytes | 1,220,058 | 958,390 |
| Cubin bytes | 445,632 | 308,384 |

These are static compiler observations, not executed instruction counts or
latency measurements. `st.global.v4.u32` occurrences remain 33. The resource
receipt reports LOCAL0 for both compilations; that field does not erase the
explicit PTX local loads/stores or the separately reported stack requirement.

All 24 decompressed remap PTX files and Triton compiler hashes exactly match
CPU8. The six empty-map variants still contain no `ld.global` instructions;
the 18 mapped/offset variants retain CPU8's local-route weight-load predicates.
All 24 complete cubin hashes differ. A read-only comparison of the 48 original
cubins, each verified against its receipt, confines every difference to
`.debug_line` and `.nv.merc.debug_line`. Executable `.text`, `.nv.info`, all other
sections, all section headers and all bytes outside those two payloads match.
The reason for the debug information difference is not established. Raw cubins
are retained under `local/remap/` and `comparison/cpu8-remap-cubin/`; section
hashes are in `remap-cubin-section-comparison.json`.

The 37 original job/identity/result/resource/PTX files match their captured
SHA-256 values. All 13 mounted-source and 18 contract-source hashes match the
frozen commit; the remote source was clean when fetched. `source-binding.json`
records each comparison. `core-logic.log.gz` preserves the separate local log
of 6793 checks, 38 megakernel regressions and zero fleet regressions, including
its host Torch skips. That log does not record a source revision, and zero
fleet regressions is not a fleet-suite pass.

`compiler-inspection.json` contains the resources, per-variant PTX checks,
source binding, timestamps and proof limits. `SHA256SUMS` covers every other
file in this directory. This archive supplies no new-source GPU numerics,
MoE sanitizer verdict, component timing, full-model TTFT or default acceptance.
Historical [CPU8](../cpu8/README.md) and [attempt4](../attempt4/README.md)
evidence remain separate.

Image: `sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211`.
Head job: `/tmp/glm53-ep-local-compile0908-9-head`; source:
`/home/choiceoh/stkernel-ep-local-0908-9`.
