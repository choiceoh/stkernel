# GB10 compressed-pool slot finalization — 2026-09-11

The GLM sparse indexer now sorts 512 selected pool IDs and writes the final
2,051 latent slots/counts in one Triton launch. It no longer materializes an
expanded-token tensor, a sorted-token tensor or PyTorch's unused sort indices.

Measured production code: `f3c270e3b9e15b81f87f72b590fe2d353f147be9`.
Baseline: main `2df6386f` (PR #545), with its exact `net.py` exported from git.
`final.json` records that file's SHA-256; `source-sha256.json` pins 158 current
engine/test/probe Python files, all verified byte-for-byte on srv1.
The measurement commit contains the production change; the following evidence
commit only adds this directory.

## Architecture and hardware contract

```mermaid
flowchart LR
    A[Selected pool IDs: 512] --> B[Mask incomplete pools and sort IDs]
    B --> C[Read one KV block address per pool]
    C --> D[Expand token offsets in registers]
    D --> E[Write tail, slots, padding and counts]
```

- `Lanes.pool_slots` replaces the `expand_pools` + `indexer_slots` serving
  interface. The independent torch oracle still expands, sorts and maps token
  positions. Old helper functions remain available to explicit historical
  probes, with no serving fallback or runtime backend switch.
- KV block size must be divisible by pool size. GLM uses four-token pools;
  reading a block address per pool avoids four repeated table lookups.
- One four-warp CTA owns each query row. The actual loaded SM121 kernel uses
  **55 registers/thread, 2,048 shared bytes, zero spills and zero global scratch**
  (`resources.json`, reproduced by `resources.py`). This GB10 reports 48 SMs,
  1,536 threads/SM and 102,400 shared bytes/SM.
- Duplicate IDs keep multiplicity: equal-ID runs expand into repeated *token*
  positions in descending order. Sorting pools and merely reversing each
  four-token group would be incorrect for duplicate IDs.
- Invalid/incomplete pools are masked before page-table reads. Tail tokens
  precede history; padding follows the valid prefix. Every output cell is
  written, including empty selection/all-padding cases. No output initialization
  kernel, device-to-host read or temporary torch allocation is needed.
- Shapes, strides, block mapping and output layout remain compatible with the
  existing cache. Graph replay reads updated pool IDs, sequence lengths and
  recycled block rows. LocalTP dispatch remains explicitly owned.

## Paired measurements

Primary data: `final.json`. Same process, image, inputs and GPU; five rounds
alternate A/B then B/A. Helper eager cases have 100 samples/round after 40 warmups.
Graph cases capture **100 consecutive calls** and time 50 replays/round after 10
warmups; table values divide the CUDA event span by 100. This reduces host launch
cost in the hardware comparison. It is a component graph, not model decode ITL.
CUPTI is initialized only after all timing, and copies are counted separately
from kernels. Outputs/counts are exact against both the old served path and the
independent expanded-token oracle.

| Query rows | Old graph µs/call | New graph µs/call | Reduction | Old eager µs | New eager µs |
|---:|---:|---:|---:|---:|---:|
| 1 | 30.063 | 5.289 | 82.4% | 195.974 | 52.359 |
| 6 | 31.636 | 5.660 | 82.1% | 199.603 | 53.191 |
| 24 | 32.840 | 5.721 | 82.6% | 201.570 | 52.710 |
| 256 | 139.890 | 13.983 | 90.0% | 257.453 | 56.877 |
| 512 | 310.831 | 23.284 | 92.5% | 402.601 | 63.882 |

For six or more rows, the helper changes from **5 kernels + 2 copies to
1 kernel + 0 copies** (one row: 4 kernels + 3 copies to 1 + 0).
Additional eager CUDA allocation drops from 215,040 to 0 bytes at six rows and
16,818,688 to 0 bytes at 512 rows. Caller-owned inputs/outputs are excluded from
these additional-allocation figures. `helper.json` records an earlier independent
helper-only run with the same implementation and consistent results; absolute
measurements from older PRs are not compared with this run.

The real-weight L3 indexer includes projections, quantization, pool writes,
logits, selection and slot finalization. Its paired eager test uses five rounds,
50 samples/round and ten warmups:

| L3 indexer phase | Old median | New median | Reduction | Old / new extra CUDA bytes |
|---|---:|---:|---:|---:|
| Decode, context 2,048, 6 tokens | 2.006555 ms | 1.869769 ms | 6.8% | 389,632 / 252,416 |
| Prefill, 256 tokens | 2.126346 ms | 1.954584 ms | 8.1% | 14,875,648 / 6,948,352 |

These are **L3 component results with real weights and seeded synthetic
activations**, not full-model throughput, quality, four-node communication or
end-to-end generation results. The larger kernel percentages must not be applied
to the entire engine.

## Correctness and environment

- `engine-tests.log`: **134 GPU tests passed, no skips** in the standalone native
  runtime. Includes cache release/reuse, request remapping, draft rollback,
  lane failure propagation and execution ownership.
- `kernel-tests.log`: four dedicated GPU tests. Widths 1/2/31/129/512/513 pools,
  pool sizes 1/2/4/8, duplicate runs, invalid/future IDs, empty shapes, large
  integer addresses, strided input/output guards and changed-input graph replay.
- `local-tests.log`: 96 CPU tests passed; 38 CUDA tests skipped on the local CPU
  interpreter environment (PyTorch 2.13.0+cpu).
- Real L3 oracle: contexts 63/256/2,048 × prefill/verify/rollback = nine exact
  slot/count, paged KV byte and state-ring comparisons. The rank-1 indexer-only
  file contains seven real tensors totaling 15,207,424 bytes; no expert weights
  or rank layout compatibility guard are bypassed.
- Four logical LocalTP ranks produce exactly the direct lane's slots/counts.
  This verifies thread dispatch, not four-server execution.
- srv1 NVIDIA GB10 SM121, driver 580.159.03; native `st-engine:9391` image
  `sha256:0d781f0a8f77d4735d0d09d57b081c9489b3a46131dc72773fd40fcbf267b446`;
  torch 2.13.0+cu130, CUDA 13.0, Triton 3.7.1. No vLLM installed or imported.
  Repository sources are mounted read-only at `/repo` with `PYTHONPATH=/repo`.
- Tests use 6 GiB and benchmarks use 10 GiB container limits, two CPUs and two
  OpenMP threads. Only this task's temporary containers run these checks.
  Existing services are left running; `environment.log` records the environment.
- Latest upstream's boot-path test still mocked removed `bind_tp`. Removing that
  stale mock restores the actual boot-mode assertions; no production boot API
  is changed by that test repair.

## Reproduce

Use the native image above and a read-only mount of this commit at `/repo`, a
writable `/cache` and `/evidence`, and `PYTHONPATH=/repo`:

```bash
git show 2df6386f:engine/profiles/glm53/net.py > baseline-net.py
python3 -m unittest discover -s tests -p 'test_engine_*.py' -v
python3 probes/engine_pool_slots.py \
  --baseline-net /repo/baseline-net.py \
  --checkpoint /checkpoint \
  --rank-file /indexer-only.safetensors \
  --output /evidence/final.json
python3 measurements/st_engine_pool_slots_20260911/resources.py
```

`/checkpoint/config.json` SHA-256:
`29c9f4171196910e99b9c069d6b76c56e3cdcd0f436dc1bacbc9513c9a7529ac`.
The existing indexer-only file is
`/home/choiceoh/st-engine-f4d7-indexer/indexer-only.safetensors` on srv1; its
original tensor hashes are recorded in `../st_engine_indexer_20260911/weights-sha256.json`.
