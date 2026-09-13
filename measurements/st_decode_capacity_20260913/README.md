# Decode capacity candidates after the five-lane component result

The previous five-lane reservation is complete. It produced no material M7
winner, so those results do not justify a fresh model boot. The full log,
numerical results and rejected timings remain in `../st_decode_bundle_20260913/`.

The next single reservation tests six independent candidates:

| Lane | Change | Numerical and timing scope |
|---|---|---|
| `moe_batch` | Apply the existing M16/N128/K256/FC2-N256 tile at M14/21/28 | Real TP4 L3 packs, 8–224 unique experts, including more than 16 rows per expert |
| `moe_stage_fc1_shared` | FC1/FC2 stages 3/1; alias disjoint epilogues | M7, shuffled expert IDs, changed routing and output poisoning |
| `moe_stage_fc2` | FC1/FC2 stages 1/3 | Same M7 contract |
| `router_batch` | BF16 operand tensor-core projection with FP32 logits | All 42 actual gates and correction biases at M14/21/28; exact selected expert IDs |
| `mhc_batch` | Lossless BF16 coefficient storage at M14/21/28 | Exact complete post/pre results over 89 packs; no transport or PDL change |
| `moe_raw_scale` | Original scale loads instead of SF6 expansion at M7 | Independent raw-scale owner with the same original weights and routing; memory cost remains explicit |

MoE tests keep FP32 scatter and the same activation rounding boundaries.
Every shape and routing case must pass before timing. Timings retain separate
warm and 64-MiB-evicted B/A/A/B samples. Input/output ownership outlives all
graph replays. Router timing includes projection and expert selection together.
These are component tests, not C=4 consumer reruns or engine speed claims.
All six candidates remain private probe choices; serving dispatch is unchanged.

CPU compilation caught the unaliased 3/1 stage plan at 102400 shared bytes,
above GB10's 101376-byte per-CTA limit (`cpu-initial/`). The aliased plan uses
98304 bytes, the same amount as the served 2/2 plan. FC1's final named barrier
ends its epilogue reads before FC2 writes; FC2's scatter barrier ends its reads
before the next FC1 item. The DMA warp does not use either epilogue. Separate
storage retains the original byte offsets and remains the default.

`cpu-qualified/` records actual native CuTe compilation without a CUDA device
on source `f961997b`. The baseline, aliased 3/1, 1/3 and all three wider shapes
compile. The deliberately retained unaliased control fails its memory guard,
so the raw report says `PARTIAL`; that control is not a GPU lane. Compilation
does not establish GPU numerics or throughput.

The Linux serving/step-observer/probe suite passed 200 tests in 49.354 s, with
one GPU-only test skipped. The new counter checks cover multi-token readbacks,
unfinished requests, cancellation and row reuse. The existing socket hangup
fixture now lets the HTTP watcher detect the close before a CPU fake completes
its 40-token request; the real socket detection and cancellation remain tested.

The next observer also records `st:generation_tokens_committed_total`, counting
actual generated IDs when each live row's host readback exposes them, plus
`st:decode_row_steps_total`. `bench/step_peek.py` reports the first as
`committed_tok_s` and the second as `mean_decode_rows`. The legacy generation
counter retains its completed-request semantics. Missing live counters remain
unavailable, not zero; the new count includes unfinished/cancelled requests'
already-produced IDs and never recounts a retired row. This is host-observed
generation throughput, not client network delivery timing.

The eventual improved consumer remains C=1 twice and C=4 once on one boot,
with 32K/128K coverage, cold-prefix identification, decode step/s and acceptance.
The operator excluded answer grades from this decision and requested no fresh
baseline boot. The current consumer evidence remains below 22 step/s.
