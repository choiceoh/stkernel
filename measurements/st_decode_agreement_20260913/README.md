# Native decode agreement, 2026-09-13

The PR830/PR840 candidate reached the door, then failed the first 32K **preparation** request. No `measure-c1` or C4 phase completed. Source `4ff6c3729fa58f9551e8928f41f92fe57106efd3`, session `st-decode-native-consumer0913v4`, onepass `20260913T063818-ea98e032869c`.

The 453 bounded device iterations span 21.9912 seconds on rank 0: 20.5992 step/s, 1575/2718 accepted drafts (57.9470%), 2028 committed tokens. These are incomplete preparation diagnostics, not a qualified performance result. Iteration 59 (zero based) first differs: rank 1 commits 7 tokens while ranks 0/2/3 commit 6. Counts later realign, then rank 0 gains one token at iteration 418. At the reasoning cap, workers enter `gather:rows` while rank 0 enters the next `step` control broadcast, all at tripwire call 975; Gloo times out after 120 seconds. Root cause of the initial token disagreement remains unproven.

`rank-*.tar.gz` preserve each node's own D12 death record, step ring, memory and onepass latency files after the canonical bracket stopped its own boot. `consumer-v4.json` records their SHA256 and derived metrics. `tests/test_engine_decode_agreement_cuda.py` (run by the canonical `probes/engine_bounded_loop_check.py`) isolates changing odd/even MAX packets under CUDA WHILE and the production-sized candidate merge, without a model boot. Its CPU proxy is not NIC/model proof.

PR830's post-merge P2 review also identified a missing coverage requirement: enabled tiled prefill must prove both `fp8_tiled_projection` and `fp8_packet_projection`. The focused test removes each marker separately.

## Candidate traffic reduction

While the short GPU reproduction waits, two additional changes preserve the candidate keys and the existing dense CUDA top-k tie order:

1. `select_logits` encodes keys in the first reduction's registers, removing the vocabulary-sized int64 pack allocation and its write/read.
2. A drafter-owned `CandidateBuffer`, allocated before capture, clears only the previous packet's columns before restoring the next packet. The dense matrix presented to the unchanged top-k is identical. A CTA barrier orders overlapping old/new candidate writes. Inactive rows retain separate previous-packet metadata across batch shrink/growth.

At 6/24 draft positions, this removes about 7.09/28.36 MiB of logical read/write traffic per selection (before the small previous-column clears). No speedup is claimed before the queued GPU gate. The canonical bounded gate includes exact changing-candidate and graph tests, plus captured original/fused/fused-and-reused timings, with warm and 64 MiB cache-flush samples. These component timings are not full-model/NIC evidence.

CPU validation: 2 actual Triton interpreter tests pass (42.923 s; the GPU graph test is skipped), 71 focused regression tests pass (11 GPU skips), and 16 SM121 kernel variants compile without initializing CUDA. Files and hashes are in `packet-optimization/`.

The separate PR848 provides host-side rank consensus and failed-bracket log preservation for this incident. It explicitly does not identify the initial numerical mismatch. This branch keeps the kernel reproduction and candidate traffic work separate from those guards.
