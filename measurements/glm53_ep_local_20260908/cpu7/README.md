Actual no-device compilation and sanitizer-launch preflight for the latest
expert-local prefill validation source.

Source `38aa70f239e1e5a5b9052ae7839438eccadf66dc`, 2026-09-08.
Immutable image `sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211`.
The normal fleet CPU job ran on srv4 in a fresh runc container with no
network/devices, 4 GiB memory, 2 CPUs and a 12 GiB host-MemAvailable guard.
It completed with exit 0 and CUDA uninitialized.

- Actual E72/I2048 CuTe compilation passed: REG168 STACK1040 SHARED1024.
  The PTX and cubin hashes are identical to cpu6. This rerun binds the
  expanded sanitizer/probe contracts to the unchanged device kernel.
- All 24 supported Triton remap dtype/branch specializations compiled for
  explicit SM121: 12 mapped, 6 empty-map and 6 offset variants. PTX and cubin
  hashes are in local/result.json; all archived remap PTX hashes match after
  decompression. CPU compilation does not execute the device kernels.
- All 37 pinned CPU tests passed without skips, failures or errors. They
  include the real Torch remap oracle and sanitizer executable identity,
  bounded-container cleanup, pre-pause rejection and tool-summary contracts.
  The historical cpu6 receipt remains unchanged with its 29-test result.

The separate head check in head-sanitizer-preflight.json actually launched
the pinned sanitizer's version command inside the same immutable image.
It passed with version `2025.3.1.0` and no exposed CUDA devices, using runc,
no network, 256 MiB memory, one CPU, 64 PIDs and a read-only sanitizer mount.
The executable is `/usr/local/cuda-13.0/compute-sanitizer/compute-sanitizer`,
SHA-256 `7a7fcdefb67042731daf021478176f4919e1843d0b10cb697af28a7d8a3d108b`.
The archived preflight receipt SHA-256 is
`3c6e04bed57c51c66b402f70102efcc9c7609a36514ac2665f71c3e9771cc06d`,
also recorded in head-sanitizer-preflight.sha256. This proves the corrected
executable can start without a GPU; it is not memcheck or racecheck proof.

Every mounted MoE source and every probe/test contract is hashed in the
compile receipt. The GPU runner rejects incomplete or stale compilation
evidence and failed sanitizer preflight before service inventory or pause.
New GPU timing includes each arm's route-remap method and remains pending.
The preceding attempt2 source passed eight plain numerical fixtures but
failed before launching its first sanitizer; those measurements exclude
remap and cannot validate the new fused remap or CTA scale cache. No default
promotion, full-model TTFT or production TP4/EP4 performance claim follows.

worker-sha256.json covers original result/resource/PTX files; compressed PTX
bytes were checked after decompression. Cubin binaries remain on srv4.
Head/worker job: `/tmp/glm53-ep-local-compile0908-7`; frozen source:
`/home/choiceoh/stkernel-ep-local-0908-7`. Submission, normal CPU fleet output
and exit records are retained beside the receipts.
