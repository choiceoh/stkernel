# Qwen3.8: GLM precision techniques carried into the engine

GLM's IEEE FP32 router, W8A16 vocabulary head and serving-input GPTQ collection now serve Qwen3.8's target. On
NVIDIA GB10, the Qwen cell suite passed **83 tests, no skips**, including four new numerical/graph/served-routing tests and the shared-expert fork tests.
These are cell and synthetic-input results. Full-model TP4 output quality, acceptance, throughput and a production
deployment are **not established** by this record.

Base: `103de8e9` (main with PLE gate fusion #1280 and serve continuation #1284). The final GPU source snapshot is
[`gpu-source.sha256`](gpu-source.sha256), checked against the isolated controller checkout. The launcher rollback
switch and documentation were checked separately. No source change was made to an admitted/running GPU checkout.

## What was carried

| GLM technique | Qwen implementation | Evidence/limits |
| --- | --- | --- |
| IEEE FP32 router projections, resident FP32 gates | Shared `kernels/router_fp32`, target's 48 routers and MTP's router; softmax/top-k preserves FP32 logits and coefficients, using the same kernel in prefill and decode | BF16 input and checkpoint weights remain identical. A constructed boundary lost by BF16 logits is preserved. GPU FP64 reference and graph replay pass at 512 experts × 2560 hidden. Equal logits choose the same lowest expert ids in both paths. Shared expert keeps its separate BF16 projection and torch FP32 sigmoid. |
| W8A16 target head (#1264) | `FP8Linear(decode_rows="w8a16")`, also used by the already-ported draft head | Same FP8 weight/scales, BF16 activations and FP32 accumulation for 1–16 rows. Wider batches keep the existing FP8 reader. |
| Self-calibration → W4/FP8 GPTQ (#650/#673), automatic filing | Shared `Calibration`/`PackStore`, attached to 192 target dense projections and the head | Every current target site fits: 3,010,492,416 admitted bytes (2.804 GiB/rank). Collect after warmup/capture, only prefill wider than the largest captured decode shape (at least 32). MTP/ghost rows do not collect. Head observes the closing mixer's full prompt before last-row selection. Shared-down Hessian uses its actual padded K=256, not source K=160. |
| Calibration identity checks (#1154 and automatic filing correction) | Strict identity required for Qwen; manual, shutdown and automatic saves use the same stamp | Export metadata/config, source file size/mtime, available preshard manifest content, PLE file and HC precision/version identify the domain. This is not a fresh full-file content audit. Foreign and unstamped blobs are ignored; matching old GLM cache policy is unchanged. |

The FP32 router reserve is **256,901,120 bytes/rank** including MTP. Missing Hessians have the shared 8 GiB cap;
the current complete target needs 2.804 GiB. Statistics do not change the current boot's weights. The next boot
builds GPTQ packs from matching saved statistics. Auto-file checks every 256 steps after all attached sites have
131,072 rows; manual control/shutdown requires 4,096 rows. `ST_SELF_CALIBRATE=0` / `--no-self-calibrate` disables
collection while still allowing matching existing GPTQ packs.

## Numerical results

Synthetic BF16 source weights at the **served rank head shape 62,080 × 2,560**, quantized once to FP8. Both readers
are compared to an FP64 multiplication using exactly those dequantized FP8 weights and the same BF16 inputs. Thus
the comparison excludes weight quantization error. It includes final BF16 output rounding; no zero-error claim.

| Rows | W8A8 RMSE | W8A16 RMSE | Reduction |
| ---: | ---: | ---: | ---: |
| 1 | 0.0266556703 | 0.0016835135 | 93.68% |
| 4 | 0.0267591868 | 0.0016813779 | 93.72% |
| 16 | 0.0269322693 | 0.0016807811 | 93.76% |

Changed-input graph replay matches the eager W8A16 reader byte-for-byte. Router outputs are FP32 and meet an FP64
reference at `rtol=2e-5, atol=2e-5`; the near-boundary winner survives replay. Native prefill calibration matches
FP64 Gram matrices at hidden K=2560 and padded K=256 (`rtol=1e-5, atol=1e-4`), with exact channel maxima and row
counts. Disarmed warmup contributes zero rows.

The CPU integration test collects 4,096 correlated BF16 rows, auto-files with a stamp, reopens a fresh pack store
and builds an FP8 GPTQ pack. On **512 separate held-out synthetic rows**, its mean squared output error must be
less than 65% of RTN's; it passes. That test is not evidence of improvement on checkpoint activations. W4 and FP8
both read the same shared store's Hessians; no new quantizer algorithm is introduced here.

## Verification and reproduction

- [`gpu.log`](gpu.log): fleet `q38precision-port-0919d`, ticket `17898155262760798`, payload exit 0. NVIDIA GB10,
  PyTorch `2.13.0+cu132`, CUDA `13.2`, image `st-engine:glm53`; 83 tests in 64.490 s. Single-GPU queue, 6 GiB budget,
  no fleet lease or production restart.
- [`cpu.log`](cpu.log): 86 tests, 83 pass and 3 GPU-only skips. Precision ports, pack/calibration, router, head,
  boot, image forward, shared overlap, kernel package/provenance and source contracts.
- [`adapter.log`](adapter.log): 84 tests, 71 pass and 13 GPU-only skips. Draft-ahead/chain, warmup, grammar binding
  and shared calibration collector regressions.
- [`qwen-suite.log`](qwen-suite.log): 52 Qwen engine/probe test files, 526 tests, no failures or unavailable files;
  424 pass and 102 GPU-only skips in the CPU-only run.
- [`launcher.log`](launcher.log): 89 boot-path and deployment-watch regressions pass after adding the collection
  rollback switch. No deployment was performed by this test.
- [`final-cpu.log`](final-cpu.log): 47 final routing/precision/boot/source/probe checks pass without skips.
- [`integration.log`](integration.log): 39 checks pass after integrating main's PLE gate.
- [`ci-regressions.log`](ci-regressions.log): 28 head/leave checks, 26 pass and 2 GPU-only skips. The full CI found an
  old GLM-only W8A16 declaration assertion and a launcher argument-order assertion; both now reflect the port.
- CPU runtime: isolated Docker `stk-test` checkout, PyTorch `2.14.0+cpu`, Triton `3.8.0`, `OMP_NUM_THREADS=1`.
  `cpu.log`, `final-cpu.log` and `integration.log` use `TRITON_INTERPRET=1`; `adapter.log` does not.

```sh
ST_PROBE_GIB=6 OMP_NUM_THREADS=1 bash bench/fleet.sh run --gpu --detach qwen38-precision-port 5 \
  'Qwen IEEE router, W8A16 head, target calibration numerics and graph replay' -- \
  bash probes/run_engine_probe.sh probes/engine_kernel_check.py --lanes qwen38_cells \
  --output /cache/qwen38-precision-port.json
```

The earlier 80-test run is retained in `gpu-initial.log` with its source hashes. Final review tightened the
FP32 coefficient comparison and used the same selector in eager/captured paths. Two added served-routing fixture
attempts are retained in `gpu-cell-setup.log` and `gpu-child-import.log`: one inherited GLM's process-global cell,
and one child imported the container's installed older engine. The final test binds Qwen in a fresh process with
its working directory and PYTHONPATH explicitly rooted at the tested checkout. Neither fixture error was waived.

## Techniques not copied blindly

- **Norm-folded channel smoothing:** GLM's plain norm-weight division is not Qwen's unit-offset `1 + weight` norm.
  A new fold must account for BF16 cancellation and every fan-out reader; GLM's prior missed FP32 gate reader shows
  why matching just the quantized projection is insufficient. Collected channel maxima are retained for a future
  model-correct transformation, but no smoothing fold is enabled here.
- **DFlash FC bias correction/rotation:** Qwen uses checkpoint MTP with different inputs/projections. There is no
  matching DFlash FC boundary to correct. Existing Qwen BF16 MTP dense/expert paths remain declared separately.
- **Recurrent FP32 state, deterministic expert accumulation:** Qwen already uses these shared facilities; the prior
  precision fix also aligned GDN beta rounding. No duplicate kernel or unrelated rounding change was introduced.
- GLM Hadamard/NVFP4 recoding experiments without valid quality wins are not promotion evidence for Qwen.
