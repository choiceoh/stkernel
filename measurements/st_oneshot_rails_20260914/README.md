# One-shot collectives: the second RoCE rail and a quiet proxy header (2026-09-14)

Operator request: reduce the collective wait. This record states where the decode wait goes, what this change
does about it, and what remains to be measured on the fleet. **No speed gain is claimed yet.**

## Where a C=1 decode step waits

Source: the profiled diagnostic replay of run `20260913T230809-af6162bd0e38` (`hybrid-k7-readback`, source
`131d7a24`, K=7, C=1, 2K), `diagnostic-c1-2000/rank-{0..3}/decode-{2,3,4}.trace.json.gz` on srv2 under
`/home/choiceoh/expert-capture/onepass-runs/`. Reproduce with `python3 decompose_trace.py DIR 2 3 4`.

A PDL transport kernel's CUPTI span overlaps its producer. Summing raw durations gives about 11-12 ms per step,
and that figure is wrong. Measured instead as the time each transport kernel runs past the latest compute
kernel before it, then aligned across ranks by collective completion (offsets stable to ±1 µs over three steps):

| Collective (per step) | Mean rank wait | Straggler skew | Floor after the last publication |
|---|---:|---:|---:|
| `k_publish_packets`, direct W4 output (43) | 38.1 µs | 8.1 µs | **29.9 µs** |
| `k_oneshot_moe_packets` (36) | 44.1 µs | 12.0 µs | **32.0 µs** |
| `k_oneshot_consumer` (13) | 97.4 µs | 54.9 µs | 42.5 µs |
| int64 MAX / gather, ≤512 keys (1 + 1) | 22.7 / 45.1 µs | 6.6 / 27.6 µs | 16.0 / 17.5 µs |

- About 92 collectives wait roughly 5 ms per C=1 step (median rank), about 10% of a 50-53 ms step. A single
  NCCL all-gather of full logits adds 0.6 ms, but only on sampled rows; greedy onepass does not see it.
- Consumer skew is dominated by two step-boundary collectives in this synchronous replay, where one rank arrives
  about 0.5 ms early. The bounded burst loop has no such boundary.
- The floor has a fixed part of about 16 µs, measured with messages of a few kilobytes. On top of that is a
  size-dependent part of about 14 µs at 64 KiB. That matches three 64 KiB peer writes serialized at about
  14 GB/s.

## The change

1. **Second rail.** Every Spark's RoCE port is exposed as two PCIe x4 functions, both `PORT_ACTIVE` with
   MTU 4096: `rocep1s0f0` at 10.10.10.x and `roceP2p1s0f0` at 10.10.11.x. NCCL already uses both. One-shot used
   only the first, so a collective's three peer writes queued behind one x4 link.
   - With `OSAR_RAILS=2`, the pairs {0,1} and {2,3} use the second function (`osar_pair_rail`). Every rank then
     sends and receives two peers on rail 0 and one on rail 1.
   - Each rail has its own device context, protection domain, completion queue, source GID and MR over the same
     `Ctrl` pages. The proxy polls both CQs.
   - Both endpoints evaluate the same symmetric rule, check it at connect, and vote on the placement before
     the self-tests. The existing exactness self-tests (sums, cancellation order, captured replay, int64 MAX and
     gather against NCCL) then run over both rails.
   - A missing or inactive second function fails preparation on every rank (D3).
2. **Proxy poll count out of the registered header.** The proxy stored `Ctrl::proxy_beat` on every busy-poll
   pass, millions of CPU writes a second next to `tx_seq`, `ack_seq`, `done_ctr` and `nbytes`. The GPU reads and
   fences those fields on every collective, and no kernel reads `proxy_beat`. The count is now thread-local, and
   the stall word is read every 256 polls; a stall lasts seconds. The header layout is unchanged.
3. **Boot latency gauge.** After its self-tests, `OneShot` replays captured chains of 16 collectives 12 times
   per cell: 8-row sum (C=1), 32-row sum (C=4) and 8-key int64 MAX. Every rank reads only its own CUDA events.
   The median and p90 µs per collective become boot gauges (`oneshot_*_us`) and the `st:lane_info` label
   `oneshot_latency_us`, next to `oneshot_rails`. The cost is a few hundred collectives at boot.

`STK_oneshot_rails` defaults to 2 (D11, 2026-09-14: improvements on by default). Rollback is
`STK_oneshot_rails=1`, which expires 2026-09-30; production refuses the override.

Arithmetic for the second rail, not a measurement:
- 64 KiB: the busiest rail carries two writes instead of three, so the size part falls from about 14 to
  about 9 µs.
- 256 KiB (C=4): about 56 to 37 µs; the shared 200G wire caps both rails together.
- Over about 90 collectives that is roughly 0.4 ms per C=1 step and 1.7 ms per C=4 step.
- The effect of the proxy header change is unknown until the boot gauge compares builds.

## Validation so far (no GPU)

- `compile.json`: the actual extension compiles and loads with one and with two rails in
  `st-engine:bracket-192031513fd1` (Torch 2.13.0+cu130, CUDA 13.0) under `--runtime=runc --network=none`,
  with CUDA hidden. The two builds have distinct module names.
- New `tests/test_engine_oneshot_rails.py`:
  - The Python and header rules agree for all 24 ordered pairs, compiled with g++.
  - Every rank has exactly one second-rail peer.
  - Every queue-pair resource follows its peer's rail.
  - The proxy writes only `flag_src` and `ack_seq` into `Ctrl`.
- 61 focused tests pass (4 flashinfer skips).
- All 73 test modules that import comm or one-shot: 1,015 tests, 7 failures, 63 skips. The same 7 failures
  (`test_ar_consumer_campaign` ×5, a fleet-lease boot string, a turn-retention boot string) reproduce on a
  pristine `origin/main` archive.

## Pending

- The TP4 boot: rail placement vote, self-tests on both rails and the latency gauge.
- D17 fleet onepass against the parent commit.
- Transport changes can stall rather than fail. The existing stall word and 30 s trap remain the safety net.
