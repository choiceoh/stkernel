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
