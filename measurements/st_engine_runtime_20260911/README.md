# ST engine runtime validation — 2026-09-11

PR #532 baseline: `a7fa81c8a494aa18af5de221ab6ab264ba32fd7b`.
The changes connect GLM to paged arena caches and the common runner, repair draft rollback,
bind served recurrent KDA, align presharded weights, and harden admission and NVMe ownership.

## Results

| Check | Result | Evidence |
|---|---|---|
| Engine unit/regression tests, CUDA enabled | 51 passed, no skips | `srv1/engine-tests.log` |
| Same suite without installed PyTorch | 39 passed, 12 skipped | `cpu-tests.log` |
| Eight base self-checks | passed | `srv1/base-selfchecks.log` |
| Original indexer under the new rollback regression | 18 of 20 cases fail | `srv1/rollback-baseline-failure.log` |
| Corrected indexer | all 20 phase/acceptance combinations pass; keys and scales byte-identical | engine regression suite |
| Served KDA vs reference, 1/6/64 tokens, initial state absent/present | all six cases pass | `srv1/served-kda.log` |
| Actual O_DIRECT NVMe round trip, independent producer stream and concurrent Future submissions | byte-identical, 19,660,800 bytes written and read | `srv1/nvme-io.log` |
| Four-node NCCL on RoCE | all ranks pass; TP linear max absolute error 1.1444091796875e-5; ten reductions and ordered gather | `srv*/comm-rank*.log` |
| Real GLM layers 0 and 3, TP4 LocalTP, reference and served lanes | pass | `srv3/real-reference.log`, `srv3/real-served.log` |
| Same real model slice, one rank per server, reference lanes | all four pass | `srv*/fleet-reference-rank*.log` |
| Same real model slice, one rank per server, served lanes | all four pass | `srv*/fleet-served-rank*.log` |

The model checks use 64 input tokens and 32-token prefill chunks. They compare contiguous and
paged caches exactly, check replicated hidden/logits and generated tokens across all four ranks,
exercise chunking, six-token verification and rejection/continuation, and finish two requests
with generation limits 3 and 1. Every request's KV blocks and state slots are returned.
The paged sequence begins at physical block 1; smaller CUDA regressions also cover fragmented
blocks `[1, 2, 0]`, two DSA layers, and an arena with earlier allocations.

Served KDA maximum relative errors across the six cases:

- Recurrent output: 0.000621891.
- Every token's recurrent state: 1.870414e-7.
- Chunk output: 0.006493507.
- Chunk final state: 0.005482833.

The original-indexer comparison loads `net.py` from the baseline commit and substitutes only its
`_indexer` into the current test fixture. A compatibility shim adds the new layer argument to
`token_slots` and `pool_slots`; the old ring arithmetic is preserved. This is a targeted regression
comparison, not a claim that the entire baseline engine was run with the new interface.

## Weight alignment repair

The original two-layer rank file placed `L0.hc.ffn_fn` at a byte offset congruent to 12 mod 16;
`L3.hc.attn_fn` and `L3.hc.ffn_fn` were at 8 and 4 mod 16. The served DeepGEMM mHC path failed
with `CUDA_ERROR_INVALID_VALUE` (`srv3/real-served-unaligned.log`).
The corrected writer inserts explicit U8 padding tensors so all actual weights begin at offsets
aligned to 256 bytes. The standard safetensors reader and the coalesced arena loader both recover
identical values. Tests cover small scalars before matrices, short writes, and truncated reads.
The isolated aligned rank slice was regenerated from the same checkpoint in 9 seconds
(`srv3/preshard.log`); production rank files were not replaced.

Stored logs have trailing whitespace removed; numerical output and diagnostics are preserved.

## Environment and reproduction

Four NVIDIA GB10 / SM121 servers, one CUDA device per server. All four resolved
`glm53:v13-b12x-it` to the same image:

`sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211`

The image reports PyTorch `2.13.0+cu130`. Runtime source hashes on all four nodes are in
`srv*/engine-source-sha256.json`; `source-sha256.json` records the tested local sources.
The checkpoint is `/home/choiceoh/models/glm53-redhat-nvfp4` and test copies live only in
`/home/choiceoh/st-engine-f4d7-20260911`. The slice holds layers 0 (KDA+dense) and 3 (DSA+NVFP4 MoE),
with about 1.78 GiB per rank. Checkpoint `config.json` is mounted separately for the fleet tests.

The saved `run-comm.sh`, `run-reference.sh`, and `run-served.sh` are the actual per-node commands,
including isolated container names, time/memory limits, RoCE interfaces, and per-node GID detection.
Run them concurrently in this rank order: **srv2=0, srv1=1, srv3=2, srv4=3**.
`dispatch-comm.py` is the used dispatcher from srv1; substitute the appropriate per-node script
and timeout for the model checks. The tested remote scripts were named `run-comm.sh`,
`run-model.sh`, and `run-fleet-served.sh`, respectively.
The rendezvous address must be the rank-0 server. No existing serving container was stopped or changed.

For the smaller checks, from the checkout with the listed dependencies:

```bash
python3 -m unittest discover -s tests -p 'test_engine_*.py' -v
bash probes/run_mk_probe.sh probes/engine_kda_check.py
python3 probes/engine_cuda_io_check.py
```

The NVMe probe needs an O_DIRECT-capable writable home directory; its saved run used a host
filesystem bind mount, 16 MiB of device KV, and 2 MiB of pinned staging.

## What these results establish

These are runtime, cache, numerical lane, and transport checks. The served model checks explicitly
use the **reference expert lane**; they do not qualify the unbound b12x expert path. The runtime
is eager greedy generation without DFlash2 and rejects nonzero draft slots. The net's verification
and rollback are tested independently from drafting.

The model check gates attention error against the first chunk's bf16 numerical floor. Dense/MoE
errors are reported, but are not independently gated; routing amplifies small numerical differences.
The suite is not a full 45-layer onepass quality test, an acceptance-rate study, a graph-serving
qualification, or an ITL/throughput benchmark. Wall times include loading/JIT and concurrent activity;
no model speedup is claimed from them.
