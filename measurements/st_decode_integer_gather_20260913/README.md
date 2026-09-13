# Native integer candidate gather in bounded decode

Source `9fb206d6` adds an exact one-shot all-gather for aligned CUDA int64 packets of 1–512 keys per rank. The drafter exchanges 96 keys at C=1 and 384 at C=4. It retains the existing local selection, global rank order, CUDA top-k tie policy and FP32/BF16 arithmetic. One CTA uses the existing 48-ticket publication, ACK guard, peer flags and ring slots; no NCCL event nodes are introduced by this exchange. The bounded-graph validator retains its restrictions and now reports the rejected node type/path.

The previous candidate consumer `st-decode-batch-consumer0913v5`, source `71eb3634`, passed the memory checkpoint after returning 6.53 GiB of inactive warmup allocator cache. It then failed bounded graph validation before any requests. This is a boot failure, with no decode or acceptance measurement. Its canonical log is `/home/choiceoh/glm53-logs/fleet/run-logs/ad80159ca5c539d231d7497f3b61beef5543dd2f79bdbef578666de5c5d80064.log` on srv2. The remaining drafter candidate NCCL gather was found in the deterministic iteration and is replaced here; full-model qualification remains necessary to establish that no other forbidden nodes remain.

## Qualification

- Full SM121a bounded graph, decode queue, one-shot and proxy-oracle extensions compiled with CUDA hidden; no CUDA context. The one-shot shared library SHA is in `cpu/compile.json`.
- 25 CPU tests passed, 5 GPU-only tests skipped. Real two-process Gloo layout checks and integer publication/rounding contracts are included.
- Canonical single-GPU ticket `st-integer-gather0913` on srv4 passed all 9 tests, no skips. Wait: 157.2 s; GPU payload including compilation: 163.3 s. The test suite itself took 158.5 s. These are qualification durations, not decode latency.
- Distinct peer packets cover all local ranks, signed extremes, odd tails, lengths through 512, changed replay and interleaved BF16/MAX/gather ring wrap.
- The actual vocabulary candidate function is executed in bounded 1/2/4-iteration graphs for 6/24 proposal rows and compared exactly to the previous rank-ordered gather contract, including tied scores and every iteration's output.
- Existing bounded commit, cancellation and mapped result-queue tests also pass. The event-node rejection test verifies that a prohibited graph still fails with a node location.

The single-GPU transport tests use owned mapped memory and a CPU proxy standing in for the NIC. Real TP4 all-gather versus NCCL is checked during serving transport initialization. They do not establish full-model speed, acceptance or answer quality.

The prior merged projection bundle was requalified on source `71eb3634`: 66 numerical cells and 3 direct producer/MHC GPU tests passed. Its raw log and summary are retained here. Some repeated wide-input timing cells regressed; no overall engine gain is inferred from component timings.

## Consumer

The candidate combines this change, merged PR #825's projection/wide-input improvements and main's PR #829 memory reclamation. Candidate-only C=1 twice and C=4 once on one boot, at 32K/128K, remains pending after the boot failure below. KV is explicitly 5 GiB. Completion/reasoning ceilings of 3072/2048 bound the speed sample and do not establish natural completion length. Raw answer grades are retained, while the requested decision uses decode step/s and speculative acceptance.

## Host watchdog follow-up

`st-decode-native-consumer0913` on `a0af5cda` passed transport initialization (including native integer gather versus real TP4 NCCL) and warmup memory qualification, but rank 2 failed in `OneShot.produce` during target graph capture with `one-shot proxy stopped progressing`. No consumer requests ran. The previous health predicate rejected any two consecutive equal nonzero heartbeat reads, even within the same CPU scheduling interval. This is not a valid elapsed-time watchdog.

Source `770b3ca4` replaces that predicate with a two-second monotonic no-progress deadline, immediate native thread-exit detection, explicit state reset and checked thread creation. Host heartbeat reads/writes use atomic operations. No GPU collective or producer code changed (`health/device-source.json`). Full Torch/CUDA extension compilation and 7 CPU tests passed; the watchdog test covers 10,000 rapid equal-beat polls, exact expiry, restored progress, thread exit, restart, zero beat and counter wrap.

Replacement canonical ticket `st-decode-native-consumer0913v2` retains KV 5 GiB and the same bounded candidate-only consumer workload. The failed boot and the queued replacement are not speed or acceptance results.

## Review and consumer follow-up

PR #830 review identified that the observer-based watchdog can renew grace for an already-stale heartbeat after an idle request gap. The proxy now publishes a monotonic timestamp on its first loop and every 256 polls, amortizing clock access in the busy loop. Startup grace starts at thread creation. Health queries only read the publication time and thread status, so neither the first query nor a newly observed old beat can extend the deadline. The old implementation fails the retained 10-second idle-gap regression; the replacement passes. Full native compilation and all 7 focused CPU tests pass (`health-publication/`). Device collective and producer code remain unchanged.

The `770b3ca4` consumer started at 15:05:18 KST and reached the final native-execution gate after decode capture, warmup, vision/grammar qualification and memory readiness. All four ranks exited at 15:12:37 because the gate required exactly two prefill markers even though five named paths executed. Rank logs and the zero-request result are retained in `consumer-v2/`. This is a separate boot-qualification failure, not a decode speed or acceptance result.
