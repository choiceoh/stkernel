# Direct tree key-bank and convolution-ring reads

This follow-up to #958 removes staging from the explicit eager tree
verification experiment. It does not enable a production serving path.
The CPU comparison baseline is #958's merge commit
`82f9aca4e034dbbe7ebf67d2f500c062f8b08734`; both arms share this checkout's
unchanged target operators, weights and input/cache tensors.

## Work removed

- Gather canonical indexer keys directly into the final prefix/private bank.
  Previously the prefix was gathered into an intermediate bank and then
  copied again when private pools were appended. One byte-copy kernel now
  reads the paged prefix and private tail into their final positions.
  Keys and scales are copied as integer bits, including NaN payloads.
- For a tree completing at least one private pool, a 32K context removes
  1,081,344 bytes of temporary prefix storage and 2,162,688 bytes of concat
  reads/writes **per DSA layer**. At 128K these are 4,325,376 bytes (4.125 MiB)
  and 8,650,752 bytes (8.25 MiB). These are shape-derived counts, not measured
  bandwidth or engine-wide resident-memory savings. Without a completed
  private pool, the old code did not concatenate and had no duplicate bank.
- KDA convolution reads the last three inputs from the canonical ring
  directly. At 6,144 channels this removes a 36,864-byte final history
  buffer plus its gather/masking preparation. The CPU reference retains
  materialization. An absent prefix is masked before any load/math.
- Indexer start/end bounds and branch-column offsets are prepared once
  per verification transaction and reused across DSA layers.

Pool order, sibling masks, exact top-k column counts/tie ordering,
convolution accumulation order and BF16 boundaries are unchanged. KDA
recurrent state remains FP32. The scratch admission bound remains
conservative and unchanged. There is no full-context KV clone and no new
weight representation.

## Evidence and limits

`interpreter.txt`: 16 tests passed with CPU Torch 2.14.0 and Triton 3.7.1.
The actual copy kernel is interpreted against an independent address/bit
oracle at empty, page-boundary, 32K-context and 128K-context sizes. It covers
strided key/scale/table views, noncontiguous physical pages, 32 private
rows and arbitrary integer bit patterns. The actual convolution sum is
compared exactly at empty/short/wrapped/long contexts with poisoned unused
history, BF16/FP32 weights and padded channels. Whole tiny GLM and W4A8
target tests check branch isolation, ordinary continuation and commit state.

`compile-triton371.json`: all 13 SM121a variants compile with Triton 3.7.1
and its bundled CUDA 13.1.80 assembler, without initializing CUDA. This is
the deployed frontend version; CPU Torch 2.14.0 is not deployment Torch
2.13.0+cu132. The key-bank copy uses 78 registers, zero shared bytes and no
spills. Production-sized ring convolution uses 37/34 registers for BF16/FP32
weights versus 36/33 for the same kernel's materialized-history mode: a
small register cost replaces separate history preparation. Every variant
has zero stack bytes and zero assembled local-memory instructions. Source
and cubin hashes are recorded; cubins/PTX remain outside the repository.

Both CPU comparisons check **bit-exact hidden outputs, target features,
canonical state, paged cache, greedy tokens and committed paths** in all
six cases (8/15 nodes, contexts 0/129/1024). Each uses B/A/A/B twice, with
three warmups per arm and cache restoration excluded from timing.

- `cpu-linux-comparison.json`: 80 samples per arm/case inside local Docker.
  Median changes range from a 15.63% reduction to a 16.10% increase. The
  inconsistent timings are retained rather than reported as a speed win.
- `cpu-macos-comparison.json`: the same comparison outside Docker, 160
  samples per arm/case. Median changes range from a 0.83% reduction to a
  0.50% increase, effectively unchanged for this tiny CPU workload.

These results establish CPU/oracle equivalence and code generation only.
`engine-check.txt` records the Linux CPU suite with Docker `--init`:
206 files, 1,893 tests, zero failed/cannot-run and 377 explicitly skipped.
The new CUDA copy/ring paths still need actual device execution; the opt-in
tests cover bit copies, changed-input/page-map graph replay and ring-vs-
materialized GPU convolution but were not run. No GPU context, fleet queue,
boot, deployment or serving benchmark was used. Real-weight GPU numerics,
quality, acceptance, tok/s and full production tree graph integration remain
unmeasured.

## Reproduce without a GPU

```sh
CUDA_VISIBLE_DEVICES= TRITON_INTERPRET=1 PYTHONPATH=. python -m unittest tests.test_engine_tree_bank tests.test_engine_tree_decode tests.test_engine_w4a8_dataflow -v
CUDA_VISIBLE_DEVICES= PYTHONPATH=. python probes/engine_tree_bank_compile.py --output /tmp/tree-bank-compile
CUDA_VISIBLE_DEVICES= PYTHONPATH=. python probes/engine_tree_bank_bench.py --output /tmp/tree-bank-cpu.json --iterations 40
CUDA_VISIBLE_DEVICES= PYTHONPATH=. python tools/check.py --list
```

Use Docker `--init` for the full CPU suite so its process-reaping tests
have an init process. Pin Triton 3.7.1 to reproduce the recorded compiler.
