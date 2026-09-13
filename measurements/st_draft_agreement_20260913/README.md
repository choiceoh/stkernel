# Agree the draft before target verification

Main gathered vocabulary candidates but still ran the final draft walk locally.
Different proposals can therefore produce different acceptance and context on
the next step, even when every rank has the same target picks. The CPU reproducer
commits 1/4/1/1 tokens before agreement and 1/1/1/1 afterward. The first numerical
origin of the incomplete `st-decode-native-consumer0913v4` run remains unproven.

This change extracts the draft agreement from the prefill candidate `3c7bcc0a`:
rank 0 supplies the greedy IDs, and sampled walks supply both their selected IDs
and the actual compact support/FP32 probability bits. The ordinary scalar and
batched proposal paths agree before target input or rejection verification.
Native int64 MAX carries the 6/24 greedy IDs inside bounded CUDA graphs; the
larger sampled packet uses the existing ordinary collective path.

PR849's fused candidate selection and reusable merge buffer remain integrated.
KDA state arithmetic, calibration files and precision are unchanged. PR848's
commit/dispatch consensus remains the guard for disagreements elsewhere.

## Evidence

- `cpu.json`: 80 focused tests, 75 passed and 5 explicitly skipped, in the ST
  image with CUDA hidden. Includes the divergent-commit reproducer, sampled
  rejection distribution, actual two-process Gloo, strided/native int64
  broadcast, drafter, graph ownership and consensus integration.
- The isolated-rank probe fixtures also implement the new identity broadcast;
  `cpu-probe-interfaces.json` records nine passing follow-up tests covering
  their local proposals, sampled probabilities and canonical probe interfaces.
  The first run lacked the bench tree in its CPU snapshot; restoring it fixed
  the fixture errors, without a production-code change.
- `reused-tp4/`: all four rank logs and the immutable source manifest from the
  admitted `st-prefill-phase2-L7r2` gate, completed in about 77 seconds before
  its consumer boot. That gate covers actual NIC/NCCL/native collectives,
  changed inputs, a retained proposal child in CUDA WHILE, C1/C4 shapes, and
  rank-1 early stops at each loop limit 1/2/4.
- `tp4-reuse.json`: agreement module, communication wrapper, native transport,
  bounded-graph implementation and draft walk hashes match those GPU-tested
  bytes. The candidate-selection implementation differs because of PR849;
  its independent exact GPU and timing gate is ticket
  `st-decode-agreement0913v3`, source `54367b03`, queue ticket
  `17892844502514310`. This is not a claim that the complete integrated model
  has already been tested.

The original v2 GPU waiter was automatically paused by its frozen probe-source
check after the probe changed. It was cancelled using `fleet.sh cancel`; v3 is
a fresh admitted queue entry. Neither action started or restarted a model.

The eventual candidate-only consumer still requires C1 twice and C4 once on
one boot, 32K/128K, decode step/s and acceptance. Answer grades are retained but
are not this task's performance decision. No new baseline boot is requested.
Neither this agreement repair nor PR849 establishes 22 step/s yet.
