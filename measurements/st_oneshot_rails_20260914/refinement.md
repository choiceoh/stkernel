# Reuse proxy descriptors and skip idle completion queues

Follow-up to PR #944 head `7a0b9b06`, requested to improve useful work per cost.
The rail assignment, wire messages, registered pages, GPU kernels and flag
modes stay the same. No new switch is needed.

- Both flag modes now prepare the four ring slots' three peer descriptors once.
  Each post updates its sequence and length; it no longer rebuilds the two WRs
  and SGEs or recomputes remote addresses and rail keys. Payload and flag remain
  ordered on the same QP, and only the flag requests a completion.
- The proxy keeps a local all-peer ACK watermark. When every posted sequence is
  acknowledged, it skips both CQ polls. Pending sends continue polling all rails
  even without new publications. A negative CQ result terminates the proxy and
  makes the existing health check fail immediately.
- The ordinary mode now holds twelve descriptor pairs for the proxy lifetime,
  increasing its fixed CPU stack storage. It keeps the original registered flag
  source; this change does not enable the optional inline transport mode.
- The standalone compile probe selects this checkout before importing engine
  modules, preventing an installed engine package from shadowing its sources.

Validation: `test_engine_oneshot_proxy.py` compiles the actual production proxy
body and transport header against mock verbs with UBSan. All four combinations
of one/two rails and registered/inline flags cover all ranks, 512 sequences per
normal run, 128 ring wraps, delayed rail completion, idle periods, poll errors,
WC errors and post errors. It checks stable descriptor addresses, payload/flag
order, per-rail keys, source/remote offsets, all-peer retirement and prompt
health failure. The mock makes any idle CQ call fail. The focused CPU suite
runs 13 tests with no skips. These are protocol checks, not RDMA or GPU evidence.

The actual one/two-rail Torch extension compile/load record is retained in
`compile-refine.json`; `cpu-refine.log` records the focused tests. No GPU,
reservation, model boot, service restart or image build was used. Existing queued
fleet work was not modified. Consumer latency/step rate and GPU numerics remain
unmeasured for this revision; the original fleet gate still applies.

## Further refinement: per-rail work and batched measurement events

- Track outstanding flag completions per rail. After one rail drains, only the
  other rail is polled until new sends arrive. The production-proxy mock rejects
  polls on a drained rail as well as globally idle polls, while exercising
  delayed completion and reuse for both rail counts and flag modes.
- Allocate and initialize every timing event before the existing warmup sync
  and rank barrier. Reuse the same pairs across cells (72 event objects become
  24). Enqueue all twelve graph replays, then wait only on the last
  end event. Same-stream order makes the earlier events readable. Sample count,
  first-sample exclusion, chain size, cells and median/p90 calculation stay the
  same; timed-batch host synchronizations fall from 36 to 3 across three cells.
- `oneshot_latency_method=batched-events-v1` identifies the new boot measurement
  method. Compare transport revisions using the same method; an improvement
  versus the former host-paced gauge alone is not transport-speed evidence.
- The new CPU lifecycle test runs the actual sampler with fake events/graphs:
  it verifies reducer dispatch, event creation before timing, all replays before
  one wait per cell, first-sample exclusion, statistics and graph reset.

The final focused CPU suite is 14 tests, zero failures or skips. Final native
one/two-rail compile/load evidence is `compile-batched.json`; focused output is
`cpu-batched.log`. The earlier refinement's evidence remains source-bound to
its own revision. First refinement CI `34815670903` passed 1,802 engine tests
(360 skipped), 115 onepass tests and 77 oracle tests. The further refinement
requires its own final CI and still has no GPU/RDMA or consumer timing result.

## Coalesce completion publication and release completed captures

- Count fixed peer placement once at proxy startup. Each successfully posted
  sequence updates outstanding counts once per rail instead of looking up the
  rail again after every peer post.
- Retire completions locally through the bounded CQ polling pass, then publish
  the final all-peer ACK once. In the CPU fixture's four-sequence completion
  batch this is one GPU-visible header store instead of four. Single-sequence
  completion still publishes at the end of that polling pass.
  The change moves intermediate ACK publication to the end of the same pass;
  its actual latency tradeoff still needs the fleet gate.
- Reset each completed measurement graph before allocating the next cell's
  graph/input, retaining at most one live capture instead of three. The existing
  finally block still resets an unfinished capture on failure. Timing event
  handles are reused across cells; 24 remain live during sampling, trading a
  larger event working set for fewer creations and host synchronizations.

The production proxy CPU oracle now covers burst widths 1/2/4 and one-CQE versus
16-CQE poll budgets, in every rail/flag/rank combination: 49,152 normal sequences
plus post/poll/WC failure cases. Every ACK is checked against delivered peer
completions. Unfragmented bursts must make exactly one ACK store per burst;
fragmented completion and single-request progress also pass. The sampler test
requires the prior graph to be reset before the next is created.

Current source-bound evidence is `compile-coalesced.json` and
`cpu-coalesced.log`. Earlier files retain their historical source identities.
This adds no GPU/queue/boot run and makes no step/s claim.
