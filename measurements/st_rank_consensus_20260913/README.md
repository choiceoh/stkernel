# Rank divergence at the decode publication boundary

Incident: `st-decode-native-consumer0913v4`, candidate
`4ff6c3729fa58f9551e8928f41f92fe57106efd3`, failed at 2026-09-13 15:40:52 KST
during `prepare-c1` of run `20260913T063818-ea98e032869c`.
This was the previously queued long onepass, before any measured result.

The four retained latency streams first differ at GPU iteration 60
(zero-based trace index 59), context 34036:

| Rank | Committed | Accepted drafts |
| --- | ---: | ---: |
| 0 | 6 | 5 |
| 1 | 7 | 6 |
| 2 | 6 | 5 |
| 3 | 6 | 5 |

At tagged host collective 975, rank 0's death note says `step` / `the step's
broadcast`; ranks 1-3 say `gather:rows` / `step 461`. All timed out after
120000 ms. The durable notes and per-rank latency files were under
`/home/choiceoh/glm53-logs/st-bracket-dumps/st-decode-native-consumer0913v4-native-gather/`
on each rank's host. Container logs collected after removal contained only
`No such container`.

## Change and limits

- Check each decoded host outcome before applying it or calling the streaming
  callback, including committed token IDs, counts, acceptance, end flags,
  row identities and context. Both bounded and ordinary asynchronous readback
  use the check. Uncommitted padding is excluded.
- Use a fixed control-group exchange of the SHA256 digest on the normal path.
  On disagreement, collect every rank's complete outcome into the durable death
  note. No unchecked result is accepted as a recovery or performance win.
- Agree on the planned rows, pending work and host progress before branching.
  If any rank needs synchronous sampling, all ranks drain and use it.
- Meet the tagged control all-reduce before broadcasting; checking a stamp
  after an unmatched broadcast cannot detect the incident's mixed operations.
- Collect each failed run's container logs and exit/OOM state before stopping
  its arm. Live probes collect evidence without stopping production.

These are host control/publication changes. They add control-group round trips
per publication and scheduling boundary. No GPU arithmetic, graph body, state
precision or output is substituted. A divergent engine still fails closed;
the kernel that first produced different results has not been isolated.
GPU performance, real TP4 transport and model quality were not measured here.

## CPU regression evidence

`tests/test_engine_decode_consensus.py` runs four independent rank states with
the existing CPU serving graph oracle and a barrier-backed host collective.
It injects count, token, end, context and acceptance disagreements at C=1/C=4,
checks publication is prevented on every rank, and covers normal async
readback, unanimous progress, mixed readiness and draining pending work.
`tests/test_engine_tripwire.py` reproduces the #975 broadcast/gather mismatch.
`tests/test_st_bracket_completion.py` checks collection-before-stop ordering
and preservation of the original failure when collecting logs also fails.

Results on the local CPU environment (PyTorch 2.14.0):

- Decode consensus, tripwire, burst, runner and pipeline: 65 tests passed;
  output in `cpu-tests.log`.
- Bracket completion and fleet bracket: 50 tests passed.
- Expanded consensus/serve/latency run: 183 tests, 18 skipped, one error in
  `MetricsTests.test_the_box_s_own_memory_is_on_the_scrape`. The same single
  test fails identically with `origin/main`'s serving module on macOS because
  the Linux host-memory metric is unavailable. It is not a new failure.
- Python compilation, shell syntax, diff whitespace and all existing fleet
  audit pins pass. No GPU test, boot or fleet ticket was started.
