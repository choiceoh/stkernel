Actual no-device compilation and PTX inspection of the additional EP optimizations.

Source `7254422f044ab3c5d042f32ee7f33add7baf5e00`, 2026-09-08.
Immutable image `sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211`.
The normal fleet CPU job ran on srv4 through the recognized no-device runc
wrapper, with 4 GiB memory, two CPUs, no network/devices and the existing
host-memory guard. It completed with exit 0 and CUDA uninitialized.

- All 48 pinned CPU tests passed without skips, failures or errors. This
  includes four CUDA-binding diagnostic contracts, the scratch-view reuse
  checks and task-publication layout checks. These are CPU contracts; the
  diagnostic GPU payload did not run as part of this compilation.
- Actual E72/I2048 CuTe compilation passed with REG168 STACK1040 SHARED1024,
  unchanged from CPU7. The candidate PTX/cubin bytes changed. Static PTX
  `st.global.v4.u32` instruction occurrences increased from 1 to 33, consistent
  with the vector task-publication branch. Scalar fallback instructions remain.
  Static occurrences do not measure executed stores, transactions or speed.
- All 24 Triton remap specializations compiled. Their archived PTX hashes
  match the receipt after decompression. All six empty-map variants now have
  zero `ld.global` instructions. All 18 mapped/offset variants predicate
  weight loads on both tail validity and the same local-route predicate used
  by the expert-ID select; CPU7 used tail-only predicates for these loads.
  This confirms code generation, not a measured latency improvement.
- `core-logic.log.gz` retains the local core log: 6793 checks, 38 megakernel
  regressions and zero fleet regressions. Its tensor-dependent host skips
  remain explicit. Zero fleet regressions is not a fleet-suite pass. The
  original log does not record a source revision; the CPU compile source is
  independently pinned above.

`compiler-inspection.json` records every specialization's original PTX hash,
load predicates/counts, CPU7 comparison and resource evidence. `worker-sha256.json`
was verified for all original result/resource/PTX files. `core-logic.json`
records raw/compressed core-log hashes. `SHA256SUMS` covers all archived and
generated files in this directory.

An earlier direct-Docker helper submission was refused before any GPU payload;
this passing archive belongs to the recognized normal CPU lane. It does not
supply GPU evidence for the binding-error investigation. The additional
optimizations have no new-source GPU benchmark result, and no MoE sanitizer,
full-model TTFT or production-default acceptance is claimed.

Head/worker job: `/tmp/glm53-ep-local-compile0908-8`; frozen source:
`/home/choiceoh/stkernel-ep-local-0908-8`. The historical CPU7 evidence and
attempt4 GPU result remain separate and unchanged.
