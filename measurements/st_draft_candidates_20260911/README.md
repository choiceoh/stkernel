# DFlash2 candidate exchange — 2026-09-11

The drafter now exchanges only 16 packed score/token candidates per position and
rank. The final branch version also fuses candidate encoding and restoration
into two Triton kernels. **The logical TP4 payload is 99.835% smaller. This is
not an end-to-end speedup percentage.**

The first implementation passed real GB10 TP4 correctness and real DFlash2
proposal checks. Its isolated selection/communication timing improved, while
complete drafter graph timings were mixed. The subsequent fused implementation
passed RTX 5050 CUDA and real-weight checks, but its GB10/TP4 qualification is
still outstanding: another ST launcher acquired the shared fleet lock before
that run. Keep these two revisions' evidence separate when reviewing promotion.

## Implementation and compatibility

- `9de1a199`: one int64 candidate gather replaces the full BF16 vocabulary gather
  in `Drafter.propose_tensor`. Each packet element contains an ordered FP32 score
  and a global token id. BF16/FP16 conversion is exact.
- `28fcdfbc`: fuse the encoding and sparse restoration for two-dimensional CUDA
  logits. CPU and other shapes retain the Torch implementation. No new serving
  environment switch is added.
- The local selection retains first vocabulary ids at the cutoff. Candidates
  are restored at their original global positions before the existing CUDA
  `topk` sorts them. Directly sorting a small candidate array can change tied
  candidate order and the subsequent DFlash2 greedy walk.
- Masking, signed zeros, NaNs, infinities, empty shards and graph replay are
  covered. NaN ordering is retained; NaN payload bits are not a contract. Input
  logits and context rings are not mutated; selection consumes no RNG.
- The full FP32 selection workspace remains. This optimization reduces network
  traffic and packing work; it does not remove all dense vocabulary work.

CUDA tie equivalence is scoped to the tested pinned runtime, Torch
`2.13.0+cu130`, git `cf30153c4c131c8164ee7798e5022d810682e2cb`, CUDA 13.0.
Recheck exact selected IDs when changing that runtime. CPU tie indices have a
different unspecified policy; CPU tests require exact IDs only for untied data.
[PyTorch topk tie contract](https://docs.pytorch.org/docs/2.14/generated/torch.topk.html)

| Five draft positions, TP4 | Local contribution per rank | Gathered result per rank |
|---|---:|---:|
| Original BF16 vocabulary, 154,880 tokens | 387,200 B | 1,548,800 B |
| 16 int64 candidates per position/rank | 640 B | 2,560 B |

These are logical tensor sizes; no wire-traffic counter was measured. Both
versions use one collective for the head selection.

## Measured results

**Initial Torch-packet implementation on four GB10s (`9de1a199`).**
`tp4/` has all 27 conditions on each of four ranks: row counts 1/5/20, three
decodable limits, random/tied/nonfinite logits. Eager and captured/replayed
values and IDs exactly match the original full gather. The 108 rank/condition
checks passed and all processes exited 0. Peak allocator reservation was below
183 MiB per rank, under the 512 MiB cap.

Five alternating AB/BA rounds, 30 CUDA-event samples per variant/round:

| Positions | Original selection + gather | Candidate selection + gather | Reduction across rank medians |
|---|---:|---:|---:|
| 1 | 923.71–973.50 µs | 814.10–831.63 µs | 10.36–15.84% |
| 5, production shape | 1,093.97–1,113.06 µs | 849.92–857.58 µs | 21.76–23.43% |
| 20 | 922.11–939.42 µs | 813.49–816.19 µs | 11.57–13.36% |

These timings include local selection, PyTorch/NCCL scheduling and peer waiting,
not head GEMM. They are an exploratory run, not a clean performance qualification.
Two nodes initially reported 96% utilization without listed GPU processes, and
a separate `st-gb10-kda-check-9391` container appeared on srv4 during the run.
All raw samples and before/after snapshots are retained.

`real-tp4/` then used the complete DFlash2 weights and all four real target
embedding/head shards. Contexts **0, 1, 17, 2047, 2048 and 2057** passed exact
legacy eager/new eager/legacy graph/new graph proposal comparisons on all four
ranks, including rank agreement and unchanged context rings. Each rank's peak
allocator reservation was **3.594 GiB** under a 5 GiB GPU / 8 GiB container cap.
This uses synthetic accepted-context activations, not prompts through the
45-layer target; it does not measure target acceptance or service throughput.

Complete drafter graphs used three alternating rounds of 30 samples. Relative
changes ranged from **1.61% slower to 1.54% faster** across the six contexts and
four ranks. Other kernel containers changed on srv4 in this period, and our
CPU-only regression container was present initially on srv1. These mixed results
do not establish an overall drafter or service speedup.

**Final fused implementation on RTX 5050 (`28fcdfbc`).**
`rtx-fused-tests.log`: **10/10 tests passed, zero skips**, including the actual
154,880-token vocabulary, strided FP32/BF16/FP16 inputs, ties and graph replay.
`real-rtx-fused.json`: all six real-weight contexts passed final-token and ring
equivalence. This single-GPU probe uses one target vocabulary shard and identity
collectives. It cannot stand in for TP4 correctness or performance.

`pack-rtx.json` isolates local selection work with identity communication,
five alternating forward/reverse rounds and 50 CUDA-event samples each:

| Positions | Dense reference | Torch packet | Fused packet |
|---|---:|---:|---:|
| 1 | 42.35 µs | 160.24 µs | 116.38 µs |
| 5 | 50.69 µs | 145.28 µs | 108.40 µs |
| 20 | 77.47 µs | 311.57 µs | 184.00 µs |

At five positions fusion removes **36.88 µs / 25.4%** of the packet path's local
cost. The fused path still costs about 57.71 µs more than dense local selection
without communication. Do not combine RTX deltas with the earlier GB10 network
timings to manufacture a final TP4 speedup estimate.

The final real-weight RTX graph comparison ranged from 0.09% faster to 0.61%
slower than the dense reference, consistent with a small added local cost.
`real-rtx.json`, `rtx-probe.py` and the earlier test logs preserve the initial
unfused RTX check; the summarizer verifies their earlier source hashes separately.

## Integration, lifecycle and reproduction

Main was integrated through PR #558 (`e80882d8`, merge `060a1b1e`). The final
native image CPU regression suite discovered **231 tests: 174 passed, 57 skipped**
because CUDA is unavailable in the CPU-only container. See `cpu-native-final.log`.
The first post-merge source copy omitted the launcher and retained kernel files
deleted by main; its two test failures are preserved in
`cpu-native-main-incomplete-copy.log`. A fresh complete source directory fixed
the copy, and both the corrected and final suites passed. No test was relaxed.

Both completed fleet runs acquired `/home/choiceoh/st-fleet.lock`, started four
private bounded containers and removed only those containers. They released
their own lock. Other services were never stopped by the dispatcher. The fused
fleet attempt failed at lock acquisition, before GPU work or container creation;
`fused-fleet-dispatch.log` records that refusal. The observed owner was an ST
launcher on srv4, with `st-glm53` running on the rank-0 host at the last check.

`fleet_check.py` runs on srv1. It expects a source archive at
`/tmp/st-draft-topk-fleet-f4d7.tar.gz` containing `engine/`, the three focused
test modules, the candidate probes, `launchers/lib/common-tp4.sh`, and the
pre-change drafter exported as `baseline-drafter.py`. `--real-weights` runs the
complete drafter check using each node's existing model files; `--unit-tests`
also runs the ten focused CUDA regressions on rank 1 after the probe. The
combined option is prepared but has not run. The archive mounts models read-only.

In an available fleet window, the remaining checks are the fused candidate
probe, fused real-weight TP4 proposals and a stable repeated timing comparison.
Full-target acceptance and service latency/throughput need a separate service
run if an end-to-end claim is desired.

Evidence can be checked without PyTorch or GPUs:

```bash
python3 measurements/st_draft_candidates_20260911/summarize.py
```

This validates the exact measured git revisions, archived RTX probe, every
timing sample/median, rank and case counts, pass flags, logical byte counts,
memory ceilings and successful cleanup. `summary.json` keeps the unfused TP4
and fused RTX results in distinct fields. Model weights, the temporary local
Torch environment and copied weights are outside the repository.

## PR integration validation

Before opening the PR, main through #561 (`b0e50398`) was merged without
conflicts. `merge-validation/provenance.json` identifies the exact integration
revision and source hashes. Its fresh native CPU suite discovered **246 tests:
182 passed, 64 skipped** because CUDA is unavailable in that container. The
RTX CUDA vocabulary/drafter/state/sampling regressions passed **20/20 with no
skips**, including main's shared sampling-graph change. See the two logs in
`merge-validation/`.

All four measurement summarizers were rerun against their recorded revisions.
The fused GB10/TP4 qualification remains pending under another task's fleet
lock; this PR integration check does not convert the earlier unfused fleet
measurements into results for the final fused kernels.
