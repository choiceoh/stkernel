# CPU15 completed with a contract-test failure

Normal CPU session eplocalcpu0908v15head ran from 2026-09-08T21:53:07.865313+09:00 to 2026-09-08T21:53:21.961431+09:00
using frozen source `30608530f0c908bc5f81db6bdb69fd70ccee9a24`. The readiness controller submitted
once after the existing 12 GiB guard passed and exited. It is finished.

Actual E72/I2048 CuTe and 24 remap variants compiled inside the no-device
13.0.3 capsule environment. CuTe PTX/cubin and every remap PTX are identical
to CPU14. Binding binaries, paired metadata and base pathfinder identity
passed before compilation and after the test failure.

The complete pinned suite ran 129 tests: 128 passed, one error, zero skips.
The error is `RunnerOrderTests.test_failed_preflight_never_reaches_service_inventory_or_pause`:
its old mocked CLI omits the newly required capsule arguments, so argparse
raises SystemExit2 before its intended sanitizer preflight. Production
guards remain intact. This failure was not hidden or admitted to the GPU.

All 58 original job files were re-read remotely and verified after transfer.
All 27 contract hashes match the frozen commit. The failure occurs before
final mounted-source/cache/CUDA-state fields are emitted, so this is a
partial compiler result, not a complete source-bound GPU admission receipt.
The original FAIL and logs remain untouched. No GPU numerics or performance
is established. The corrected test and next kernel candidate require a new
source/job; CPU15 must not be changed or resubmitted.
