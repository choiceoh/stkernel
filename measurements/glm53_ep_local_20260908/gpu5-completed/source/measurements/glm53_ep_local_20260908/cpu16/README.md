# CPU16 SFA row-base cache candidate

The candidate moves the M128 row-only scale-offset calculation to the lane-0
route allocator and publishes it in unused per-warp shared slots 8–15. Both
Q0 quantizer paths retain exact operands and combine the cached row base with
SF-only fields. Existing synchronization and workspace size are unchanged.
The prefix computes nonnegative ceil(rows/128) with an unsigned shift.

The old sanitizer ordering fixture now supplies capsule admission inputs and
proves that it reaches the intended sanitizer failure before any service or
GPU action. Its eight focused tests and the 13 capsule-wrapper tests pass
locally with no skips. Actual pinned compilation and GPU outcomes are recorded
only after execution. CPU15's partial failure is separately preserved.

Local source-level validation passed 11 route/cache and 12 publication/address
tests, plus eight sanitizer and 13 wrapper tests: 44 total, no skips. The
address oracle covers every admitted physical row and SF index; producer and
consumer tests poison unused slots, shrink routes at unchanged addresses,
cover both scale branches and preserve the existing publication barriers.
Unsigned prefix checks exhaust row counts 0–131072. Independent review passed.
The expected pinned suite is 134 tests; this is not a claim that it ran.

## Actual compilation PASS

Source `111fff02f66f3e07a8332da1117afb75b161f0d6` completed one normal no-device CPU16 job from
2026-09-08T22:04:28.712751+09:00 to 2026-09-08T22:04:46.506080+09:00. Actual E72/I2048 CuTe, all 24 remap variants
and all 134 pinned tests passed, with zero failures/errors/skips. CUDA remained
uninitialized. The exact capsule13.0.3 binary/metadata/pathfinder identity was
verified before and after the full operation. The complete receipt validates
against all 13 mounted and 27 contract sources, both frozen and current.

REG168/STACK112/SHARED1024 are unchanged. PTX increased 939211→964424 bytes and
cubin 288664→300032 bytes relative to CPU14 (CPU15 produced identical kernel
artifacts before its unrelated test failure). All 24 remap PTX hashes match
CPU14. Code-size growth and a new shared load are costs to measure; no speedup
is inferred from the address-expression change.

All 58 original files were re-read remotely and verified after transfer. The
job/source are frozen at /tmp/glm53-ep-local-compile0908-16-head and
/home/choiceoh/stkernel-ep-local-0908-16. The job is complete; do not rerun it.
The wrapper retained runc/no devices/no network/4GiB/two CPUs and the 12GiB
host-memory guard. Host MemAvailable at submission was 27.19GiB.
Full GPU correctness/sanitizers and full-model TTFT are separate pending gates.
