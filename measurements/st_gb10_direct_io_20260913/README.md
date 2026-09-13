# ST GB10 direct I/O serving evidence — 2026-09-13

PR #826 (merged as `a8d1e0d60adc378b784866d5a5040a3b7d204f67`) connects three serving paths:

1. Eligible single-pack W4 attention/dense final projections write BF16 directly into a replay-time reserved TP4 send slot. Every final writer fences system memory. Publication waits for the producer, preserves the existing 48-ticket collective accounting, then gives canonical rank pointers to MHC. Multi-pack sums, observers, ModelOpt dense, MoE, auxiliary features and final FFN reductions retain their existing paths.
2. Bounded decode publishes immutable iteration results through owned mapped memory with CPU/GPU system atomics. The server streams each available iteration without duplicates. Cancellation is voted through the existing rank MAX before another iteration; requests and memory slots retire only after the entire graph finishes.
3. Tiled FP8 KDA prefill consumes received packets directly as FP8 GEMM activations. It retains the existing global FP8 threshold, two-slot ownership and BF16 rounding in registers. Calibration observers retain their BF16 input tensor. KDA recurrent state stays FP32.

This follow-up removes the old five result-log `index_copy_` launches and their buffers from the shared-queue path. Only timing/stage readbacks remain. The CPU oracle retains ordinary result logs. It also exercises the final-projection callback through the real four-rank CPU model oracle.

## Evidence and limits

The reports contain exact source SHA-256 values. Durations are test-suite wall time, including any compilation/warmup; they are **not serving latency or throughput**.

| Gate | Frozen source | Result | Scope |
| --- | --- | --- | --- |
| `st-packet-prefill0913r1` | `32dfce3c` | 3 GPU tests passed | FP8 data/scales/GEMM equality, two-slot reuse; synthetic peers |
| `st-shared-queue0913r1` | `4bf48bcd` | 6 GPU tests passed | System atomics, early CPU reads, cancellation/reset, actual bounded decode adapter with toy target |
| `st-direct-producer0913r1` | `5b0b9528` | 3 GPU tests passed | Actual W4 GEMM into mapped ring slots, graph replay/ring wrap, mixed BF16/MAX ticket accounting, delayed ACK guard, MHC rounding |
| `cpu.log` | producer implementation before main integration | 220 tests, OK; 1 skipped | Lifecycle, streaming, burst, pipeline, execution plans and prefill |
| `integration.log` | `5b0b9528` plus projection-oracle test | 29 tests passed | Main's shape descriptor changes, four-rank producer routing and prefill |
| `retirement.log` | `d67708cd` (same tree as follow-up `f5253a0e`) | 17 tests passed | Removed unused result copies and complete projection routing |
| `compile.log` | source hashes in report | Passed without CUDA context | SM121a dense, one-shot and CPU-proxy-oracle native compilation |

All three GPU gates used the canonical single-GPU fleet lane on srv4 with `st-engine:glm53`. The tag was observed before/after this campaign as `sha256:8190d08e822e1f9d18dda5a127a5d9a9e8c53ef4b5136d1154f9ea4b7727f1ed`. Native ring tests use an owned mapped allocation and a CPU proxy oracle; they do not connect to a NIC or modify the serving transport. The 12 GPU tests are distinct component checks, not real-weight model acceptance or TP4 throughput proof.

PR #826 engine-check CI passed: https://github.com/choiceoh/stkernel/actions/runs/34738448027 . The queued repeat `st-shared-queue0913r2` was withdrawn: the follow-up only removes obsolete writes/buffers, and its CPU gate passed. It is not counted as a GPU result.

Real NIC/model C=1/C=4 tok/s, TTFT, 32K/128K prefill and acceptance/quality remain unmeasured for these changes. No speedup or quality improvement is claimed. The existing default-on parent controls (`direct_mhc`, `decode_iterations=4`, `prefill_project_tiles`) select the new paths; existing ordinary paths remain available through those controls. No precision experiment is re-enabled.

## Reproduction

Use a clean, frozen checkout on the fleet head. GPU access always goes through `bench/fleet.sh`; use a unique session and owned probe tree. Each command below is a separate single-GPU admission:

```sh
ST_IMAGE=st-engine:glm53 ST_PROBE_TREE=st-direct-io-check \
  bash bench/fleet.sh run --gpu --detach SESSION 8 "ST direct I/O component gate" -- \
  bash probes/run_engine_probe.sh probes/engine_direct_mhc_check.py
```

Substitute `probes/engine_bounded_loop_check.py` or `probes/engine_prefill_tiles_check.py` for the other gates. The direct-MHC and bounded-loop probes also accept `--compile-only` inside the ST image without GPU access.
