# MoE token input reuse, 2026-09-17

This continues the fixed-K7 campaign in `st_fixed_k_next_20260917`.
The earlier pair-local BF16 input reuse preserved bytes after repair, but
whole-MoE changes remained between -0.32% and +0.13%. The present experiment
reuses the **quantized FP4 payload and FP8 block scales** across selected
experts with the same input global scale. Other experts retain the original
quantizer. K, selected experts, weight format and arithmetic are unchanged.

## Implementation and limits

The first existing resident-grid barrier publishes a per-token cache made
during phase 0. Its 18,432 bytes at C1 and 36,864 bytes at C2 occupy the
unreachable suffix of the existing packed-input workspace. The host validates
the exact GLM TP4 geometry and capacity against the maximum number of compact
experts, even when all routed slots select different experts. No new allocation,
kernel launch or grid barrier is added.

Four separately keyed candidates are measured: contiguous quantization cache
(`input_reuse=1`), warp-striped cache (`2`), compact route preparation plus
striped cache (`3`), and register fanout (`4`). Mode 3 uses another 512/1024
bytes for route metadata. Mode 4 retains only that metadata, quantizes each
block once in phase 1, and writes it directly to every selected expert with
an equal input scale. Different scales use the original quantizer. Its compact
expert prefix uses warp ballots instead of scanning each first-occurrence flag.

Serving now selects mode 3 only for the exact GLM TP4 eight/sixteen-row SF6
reform geometry. The user explicitly accepted small or inconclusive gains as
part of the combined fixed-K bundle. Explicit mode zero remains the same-build
control. Mode 4 remains a private candidate pending its GPU comparison.

## Evidence

- `compile-v1.jsonl`: six cells (C1/C2 times baseline/two candidates) compile
  with CUDA devices hidden and without initializing CUDA. Its completion scope
  string is inherited from the GPU probe; this file proves compilation only.
- `native-resources-v1.json`: all six compiled cells use 96 registers,
  zero stack and zero local memory. Dynamic shared memory is unchanged:
  91,136 bytes for C1, 100,352 for C2. Native resource counts are not speed proof.
- `native-resources-control-0330.json`: the original control from `0330d576`
  produces byte-identical native binaries and disassembly to the new source
  with reuse disabled, at both C1 and C2. Both were compiled on the same host
  with the same image, dependency mount and flags.
- `cpu-config.txt`: 49 focused tests, seven GPU-dependent skips, no failures.
- `cpu-integrated.txt`: all 24 integrated configuration/package tests pass in
  the GPU-hidden Linux image on source `0c1cca60`. The earlier macOS attempt
  lacked `os.O_DIRECT` and Torch for two package tests; its errors are retained
  separately in `cpu-integrated-macos.txt` and are not counted as passes.
- `gpu-v1.jsonl`, source `53e0ee63`, ticket `moe-input-reuse-v1-0917`:
  28 raw-byte cases and 36 output comparisons pass on GB10. Whole-MoE timing
  does not establish a gain: evicted three-layer changes are +0.06%/+0.12%
  for the contiguous cache and +0.65%/+1.00% for striped C1/C2.
- `gpu-routing.jsonl`, source `0cb764d6`, ticket `moe-input-reuse-v4-0917`:
  42 raw-byte cases and 48 output comparisons pass. Mode 3 evicted three-layer
  changes are -0.15% C1 and -0.01% C2, with only 3/5 and 2/5 faster brackets.
  This is inconclusive. Stamped frontend medians fall from 13.10 to 12.01 us
  at C1 and 18.32 to 17.14 us at C2, but that is not whole-MoE speed proof.
- `summary-v1.json` and `summary-routing.json` preserve all bracket ranges,
  numerical comparisons, and phase medians. No gate was relaxed.
- The V2 ticket was cancelled after its edited source disagreed with the
  pinned runner; V3 failed admission before GPU use. V4 used a fresh frozen
  source and runner, with the required generic-to-async shared-memory fence.
- `native-resources-fanout.json`: mode 4 and the disabled control compile at
  C1/C2 with CUDA devices hidden, 96 registers and zero stack/local memory.
  GPU correctness and timing remain pending for this new mode.

The CPU compiler host is `ost-97x`, with GPU-hidden runc, CUDA 13.2 and Torch
2.13.0. The image alone has published FlashInfer, which lacks a required helper;
compilation explicitly mounts its existing fleet FlashInfer 0.6.18.dev20260819
source under `/vendored`. This is compilation evidence for sm_121a, not an RTX
5050 execution result. The GPU ticket uses the fleet's own ST runtime.

## GPU gate

`probes/engine_kernel_check.py --lanes moe_input_reuse` first poisons input
scratch, then compares registered route payloads and scales byte for byte.
Cases include unit/nonunit common scales, mixed scales, distinct scales, zero
scales, maximum distinct expert occupancy and all tokens sharing eight experts.

After that, the actual layer 3 and layer 3/4/5 sequence run through C1/C2
request-like, independent and shared fixtures. Graph replay checks changed
inputs, zero route weights, poisoned accumulators and both replay orders against
the same-build control and its repeat. Only passing cells are timed. Each warm
and cache-evicted comparison has five B/A/A/B brackets. A separate stamped
handle diagnoses the frontend and compute phases; its timings are not serving
latencies. Full-engine acceptance and throughput remain a separate gate.

```sh
python3 measurements/st_moe_input_reuse_20260917/summarize.py gpu-v1.jsonl
```

Mode 4 ticket: `moe-input-reuse-fanout-0917`, frozen source `b1233ea1`,
compares modes 0 through 4 on actual rank0 weights. It predates the serving
adoption, and every arm explicitly supplies its selector.

## Default integration

`cpu-defaults.txt`: 31 related tests pass under the GPU-hidden Linux/ST image
after adoption. They cover the exact shape and layout limits, explicit rollback,
cache keys, mHC/MLA consumer proof, router binding and native registration.
`native-resources-default.json` compiles the actual serving selection without an
explicit mode: C1/C2 binaries and SASS exactly match the GPU-qualified mode 3
in `native-resources-routing-fenced.json`. Both use 96 registers and no stack or
local memory. The optional mode 4 does not alter these mode 3 binaries.
