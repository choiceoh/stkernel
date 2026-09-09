# Decode repair CPU9: PASS

Frozen serving source: `e4425b984e3455614744f0f3072916b1b296bd0f`.
The immutable image and CUDA bindings capsule are recorded in `submission.json`.
The runc container exposed no NVIDIA devices, used 4 GiB and two CPUs, and
completed without initializing CUDA. This is compilation/CPU evidence only.

- Both CuTe micro variants emitted separately retained PTX/cubin. The exact
  zero-weight E72/M8 candidate selected M32/N128: REG128, STACK0, SHARED1024.
  Its M64/N128 control used REG220, STACK0, SHARED1024.
- All 24 fused preparation Triton specializations compiled (mapped, empty
  map, offset; admitted ID/map/weight dtypes). Shared memory was zero.
- All 43 CPU contracts passed with zero failures, errors or skips.
- The outer runner verified the full mounted-source and contract-source
  sets, capsule identity, every artifact hash, and the complete file set.
  The collected tarball's 2 micro pairs and 24 prepare pairs were rehashed
  locally against `result.json`.

The preceding CPU7 artifact collision and CPU8 stale SF6 test fixture are
preserved separately. The repaired serving kernel bytes did not change
between those CPU attempts. Direct serving validation uses canonical
session `eplocalonepass0909v9` on this exact frozen source.
