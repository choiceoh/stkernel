# Decode capacity candidates after the five-lane component result

Completed 2026-09-13 11:31 KST on frozen source `2b3a74b9`. No new serving candidate
was promoted. `gpu-summary.json` and `gpu-records.jsonl.gz` retain the results and
original remote-log hash. The six child processes took 86.525 seconds in total;
this was a component hold, with no consumer-model baseline boot.

- Wider M16 reform: mostly 0.4–3.5% lower latency for spread routes, but M21/M28
  with eight reused experts regressed 9.2–20.1%. Retain ordinary wider geometry.
- FC1 3/FC2 1 with aliased epilogues: no consistent win. FC1 1/FC2 3: mixed,
  including a 10.36% cold regression. Retain 2/2 and separate epilogues.
- Raw scales: generally 0.7–7.3% slower and a second scale owner. Retain SF6.
- Wider packed mHC: the exact gate failed at M14 (post output: 56/56 differ,
  maximum absolute 0.000341952); timing did not run. Retain its served geometry.
- Router: the already-served TC path passed all real gates; M28 chain time was
  about 1.596 ms versus 3.068 ms for the FP32 reference. This is not a new gain.

The later scatter branch removes the losing private hooks and probe entry points.
Reproduce this completed bundle from `2b3a74b9`, not the later branch.


The previous five-lane reservation is complete. It produced no material M7
winner, so those results do not justify a fresh model boot. The full log,
numerical results and rejected timings remain in `../st_decode_bundle_20260913/`.

The next single reservation tests six independent lanes:

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
The MoE and mHC candidates remain private probe choices. While preparing this
bundle, main PR #810 adopted the wide tensor-core router. Its lane now qualifies
the served path against the FP32 reference, rather than claiming another gain.
Main PR #809's private W4 scratch entry remains independent of the older optional
packing probes and retains its original defaults.

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
The additional raw-scale M7 handle also compiles at 98304 shared bytes;
`cpu-raw-scale/` retains that separate CPU result. The wider mHC probe uses
the already-built native coefficient-storage entry point with an explicit
M14/21/28 adapter; its new shape numerics remain a GPU gate.

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
