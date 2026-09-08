# Isolated CUDA bindings 13.0.3 compatibility experiment

The exact NVIDIA 13.3.1 loader source requests **34** default-stream entry points above the observed driver ABI 13000: nine at 13010, twelve at 13020, and thirteen at 13030. The 13.0.3 loader has **zero** such requests; its maximum ABI is 13000. This predicts the observed 34 errors, pending mapping to the installed binary and a matched experiment.

| Loader source | Default lookups | Lookups above 13000 | Maximum ABI |
|---|---:|---:|---:|
| 13.3.1 | 504 | 34 | 13030 |
| 13.0.3 | 470 | 0 | 13000 |

NVIDIA documents `CUDA_ERROR_INVALID_VALUE` when `cuGetProcAddress` is asked for a CUDA version newer than the driver. The Python binding's first `cuDeviceGetCount` initializes its full entry-point table before invoking the actual count function. [Driver contract](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-driver-api/group__CUDA__DRIVER__ENTRY__POINT.html), [13.3.1 loader](https://github.com/NVIDIA/cuda-python/blob/v13.3.1/cuda_bindings/cuda/bindings/_bindings/cydriver.pyx.in#L528-L569).

The proposed candidate is the complete official CPython3.12/aarch64 `cuda-bindings==13.0.3` wheel in a separate package target, on the same immutable base image. Keep Torch, CuTe, driver, sanitizer options, overlay, and workload unchanged. Compare fresh baseline and candidate processes using the exact minimal context/count/version diagnostic; errors must remain enabled and fatal. A clean candidate result clears only this loader issue, not MoE or full-model correctness.

The wheel's mandatory dependency is `cuda-pathfinder~=1.1` (>=1.1,<2). Avoid `[all]` extras because they can bring additional Toolkit packages. Inventory installed reverse requirements first: `cuda-python13.3.1` requires `cuda-bindings~=13.3.1`, so a targeted override can deliberately conflict with base metadata and cannot be described as a clean production environment. Exact installed Cutlass/FlashInfer requirements were not found in the existing archive. [Candidate package](https://pypi.org/project/cuda-bindings/13.0.3/).

Both binding versions officially support Linux drivers starting at580.65.06; this is not evidence that the entire R580 runtime is unsupported. The proposed test addresses optional entry-point probing under the sanitizer. [13.3.1 requirements](https://nvidia.github.io/cuda-python/cuda-bindings/13.3.1/install.html#runtime-requirements), [13.0.3 requirements](https://nvidia.github.io/cuda-python/cuda-bindings/13.0.3/install.html#runtime-requirements).

A direct official `cuDeviceGetCount` call is useful only as an independent diagnostic control. Replacing the count helper alone does not repair CuTe's later bindings initialization. Do not clamp new function ABIs to13000: that can select incompatible signatures.

All exact source URLs and SHA256 hashes, the wheel URL/hash, 34 symbols with versions, and the staged test plan are recorded in the JSON files beside this note. Only local research downloads were performed; no package, GPU, queue, or production changes were made.

Reproduce the static count with `python3 extract_loader_counts.py`; it checks the 470/504 totals and 34-request version breakdown. The call-site list in `report.json` retains exact line numbers and ABI versions.

The full upstream loader templates are not copied into this archive. Download the two versioned files from `report.json` into a separate directory, preserving their recorded filenames, then run `python3 extract_loader_counts.py --source-dir /path/to/sources`. The script verifies their original sizes/hashes before counting; `root-reproduced-counts.json` records the independent rerun.
