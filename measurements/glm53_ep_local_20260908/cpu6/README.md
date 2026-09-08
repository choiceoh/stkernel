Actual no-device compilation of the CTA scale cache and fused route remap.

Source 2e65f1be221c5cdbda008794250b3fc70efc710c, 2026-09-08.
Immutable image sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211.
The normal fleet CPU job ran on srv4 in a fresh runc container with no
network/devices, 4 GiB memory, 2 CPUs and a 12 GiB host-MemAvailable guard.
It completed with exit 0 and CUDA uninitialized.

- Actual E72/I2048 CuTe compilation passed: REG168 STACK1040 SHARED1024,
  unchanged from cpu5. The CTA reuses the dead histogram's 288 bytes for
  expert scales and adds no barrier or shared-memory allocation.
- All 24 supported Triton remap dtype/branch specializations compiled for
  explicit SM121: 12 mapped, 6 empty-map and 6 offset variants. PTX and cubin
  hashes are in local/result.json. CPU compilation is not device execution.
- All 29 pinned CPU tests passed without skips, failures or errors. The new
  tests run the actual legacy Torch remap against a scalar conversion/range
  oracle and verify NaN payload/signed-zero preservation. These CPU tests do
  not execute the fused Triton kernel.
- The separate local core log passed 6793 checks and 38 megakernel
  regressions. Host tensor-dependent skips are not reported as passes.

Every mounted MoE source and every probe/test contract is hashed in the
receipt. The GPU runner rejects missing/stale source or incomplete remap
compile evidence before service inventory/pause. Both subsequent timing
arms include their route-remap method. New GPU checks also change scales at
the same addresses and run a 24-variant byte oracle in eager and sanitizer
cells. This source has no GPU numerical or performance verdict yet.

worker-sha256.json covers original result/resource/PTX files; compressed PTX
hashes were checked after decompression. All 24 remap PTX files additionally
match their individual hashes in local/result.json. Cubin binaries remain on
srv4. Head/worker job: /tmp/glm53-ep-local-compile0908-6, frozen source:
/home/choiceoh/stkernel-ep-local-0908-6. Core log hashes are in core-logic.json.
