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

The candidate combines this change, merged PR #825's projection/wide-input improvements and main's PR #829 memory reclamation. Candidate-only C=1 twice and C=4 once on one boot, at 32K/128K, is pending. KV is explicitly 5 GiB. Completion/reasoning ceilings of 3072/2048 bound the speed sample and do not establish natural completion length. Raw answer grades are retained, while the requested decision uses decode step/s and speculative acceptance.
