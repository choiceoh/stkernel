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

Two separately keyed candidates are measured: contiguous work assignment
(`input_reuse=1`) and warp-striped assignment across the resident CTAs
(`input_reuse=2`). The selector defaults to zero. The experiment is restricted
to eight/sixteen-row vector-input reform cells. It has no serving adoption yet.

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
- GPU ticket `moe-input-reuse-v1-0917`, admitted source `53e0ee63`:
  pending. It uses the canonical exclusive fleet queue and actual rank0 weights.
- GPU ticket `moe-input-reuse-v2-0917`, admitted source `133a7343`, additionally
  compares compact route preparation (mode 3). See `routing-candidate.md`.

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
