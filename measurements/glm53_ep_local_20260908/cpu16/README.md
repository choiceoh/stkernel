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
