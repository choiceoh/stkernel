# Deferred KDA state materialization

This experiment tests whether saving recurrent-state writes pays for a second
kernel that reconstructs the accepted state. The serving lane is unchanged.
No engine throughput improvement has been established.

## Why this is a bounded first experiment

The current ring stores a full FP32 `[heads, key_dim, value_dim]` state after
every verified token. At the GLM rank shape (16 heads, 128 by 128 state), that
is 1 MiB per position per layer. Seven verified positions write 7 MiB per KDA
layer, or 238 MiB over 34 layers, excluding the initial read and other tensors.

The candidate records each update's normalized key, decay, and value residual
as FP32 vectors. For seven positions this is 168 KiB per layer. Verification
does not modify the ring. After acceptance, a commit kernel reloads the initial
state, replays only the accepted updates, and writes the final accepted state
plus any crossed prefix-boundary state. Ordinary steps write one dense state;
an accepted block crossing a 768-token boundary can require a second one.

The FP32 decay multiplication is explicitly rounded before the state update,
matching the verifier's order. Output and reconstructed state must agree bit
for bit with the ordinary ring before timing is allowed. This is a numerical
requirement to test, not an assumption based on the recurrence formula.

This prototype retains the ordinary arena layout. It is a write-traffic
experiment, not a claim that serving's resident-memory budget has shrunk.

## Existing cost evidence and limit

`MEASUREMENTS.md`, section 89, records 816.2 microseconds for the 34-layer,
one-sequence, seven-token ring work. That was a shared-GPU kernel measurement,
not a matched full-model result. It argues against spending a large state-ABI
rewrite before pricing verify plus commit. The current live service's stage
counters do not break down kernels inside target forward.

The other two architecture directions remain separate follow-ups:

| Direction | Concrete current boundary | Next required evidence |
| --- | --- | --- |
| Decode communication and consumer fusion | Row-parallel output, TP sum, mHC, norm, next projection | Captured full-model kernel profile; existing mHC/PDL fusion is already present |
| Prefill communication and GEMM input format | FP8 packet unpack to BF16, then 128-column FP8 activation quantization | Exact equivalence of packet scales, BF16 rounding and GEMM scales, followed by matched onepass |

## Verification and reproduction

- `compile.json`: 12 SM121 compilations, Torch 2.13.0+cu130 and Triton 3.7.1,
  with no CUDA device initialization. Covers ordinary/deferred verification
  and commit for widths 1, 6, 7 and 8. Width remains a runtime argument in
  verification, so some compiled variants intentionally share a cubin hash.
- CPU admission/probe tests: 26 passed in the ST image without GPU access.
  The local macOS interpreter lacks torch; two existing graph-profile tests
  could not import there. Their same-image rerun passed.
- Source-provenance pin check passes after refreshing `SOURCES.json`. The
  earlier full CI also reported two missing-checkpoint-config errors in
  `glm53.net` and `glm53.drafter` self-checks. The same-image no-GPU run
  reproduced both before merging main, which now supplies CPU fixtures.
- GPU suite: acceptance lengths 0 through full width; changing device slot,
  context and count during graph replay; cyclic ring positions; prefix
  boundaries; untouched slots and padding; tail dimensions and stale NaNs.
- Timing: one graph contains ordinary verification; the other contains both
  deferred verification and accepted-state commit. Arms alternate A/B and B/A
  for 20 samples in warm and evicted-cache regimes. The timing ring has the
  production width of seven, with identical slot strides on both arms. The
  initial state is restored outside every timer, since a full-width step can
  overwrite it. Widths 1/6/7, acceptance 1/3/full, and boundary/non-boundary
  contexts are reported separately.

```sh
# No GPU: compilation only.
ST_PROBE_NO_GPU=1 bash probes/run_engine_probe.sh \
  probes/engine_kda_deferred_compile.py --output /cache/kda-compile

# GPU: submit from the isolated checkout on srv2.
REPO=$PWD ST_IMAGE=st-engine:main-ff728f43 \
  ST_CACHE=/home/choiceoh/.cache/st-kda-deferred \
  bash bench/fleet.sh run --gpu --detach stkda-deferred0912v7 5 \
  "KDA deferred state correctness and paired verify commit timing" -- \
  bash probes/run_engine_probe.sh probes/engine_kda_deferred_check.py --samples 20
```

The v5 reservation (ticket `17892214701695707`, source `724042b7`) was
admitted but never started a GPU container: `BEAT=$(fleet_lease_beat ...)`
waited forever because the background loop retained the command substitution's
output pipe. `launcher-blocked-v5.log` records the canceled run. Only this
reservation's process group and exact-owner lease were stopped/released.

The helper now detaches the background loop's standard streams. A real-shell
regression test demonstrates that the original helper times out, while the
fixed helper returns its PID promptly and continues renewing. This preserves
both the lease and its heartbeat. The fixed same-image CPU suite ran 110 tests
without errors (3 GPU-dependent skips). The fresh reservation is
`stkda-deferred0912v7`, ticket `17892227321982710`, revision 1, source `19113939`;
it is queued. The v6 submission failed before admission because main had
advanced; v7 includes those documentation changes.

The GPU output is `/home/choiceoh/.cache/st-kda-deferred/kda-deferred.json`.
The kernel runner takes and releases the fleet lease.

## Serving integration gate

A kernel win alone is insufficient. Before making this a serving lane, commit
must be connected after both synchronous and asynchronous sampling, before
prefix staging, observation, parking, and the next target step. Captured
factor buffers must remain owned until commit completes, including slot
reuse, cancellation, and batch transitions. The commit count is the number of
positions actually retained after EOS and output-limit clipping, including the
anchor/correction position; it is not the raw number of accepted drafts. The
asynchronous path needs the pre-advance context and the real physical slot,
since `advance` changes the next context and can replace a finished slot with
zero. Rejected intermediate states must
not become visible to a consumer expecting the old complete ring.

Only after those contracts pass would the candidate be compared with the
same-source ordinary ring on TP4, with two unchanged `bench/onepass.py` runs
per boot, matching engine shape and runtime, explicit prefix-cache state,
output hashes, quality, acceptance, decode tok/s, and 2K/32K/128K TTFT.
