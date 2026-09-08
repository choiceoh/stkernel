# Minimal baseline acquisition

Code revision: `5a8ab6c`. Linux full gate passes 124 CPU tests with complete coverage and no skips. The initial macOS fleet/campaign gate passed 121 tests; the final 57 experiment tests passed again after adding vetted-baseline selection, shell reuse and saved-request compatibility tests. No new GPU experiment was run.

The real runner exercised against fake fleet/onepass commands changes a new pair from three baseline boots plus one candidate to one baseline plus one candidate. Two different candidates use three measurement boots total, sharing one baseline. An explicit confirmation adds only the two missing baseline samples; later compatible candidates require no new baseline. The shell pair reuses its one baseline and does not count duplicate records from one boot as independent evidence.

Minimal results are completed exploratory comparisons, with `promotion_ready: false`; the statistical judge is unchanged. Quality failures, source/runtime/workload mismatch and missing boot receipts still prevent reuse. One-baseline startup campaigns report no drift estimate. Their two-candidate plan uses six boots instead of seven, retaining PRIME and two measurements per candidate. Explicit confirmation restores the closing control. Boot-count reductions are fixture/plan evidence, not live wall-clock speedup measurements.
