# Onepass 4 queued receipt — NOT YET GPU

At the captured snapshot **2026-09-09 05:39:02 KST**, session `eplocalonepass0909v4` remained **queued, position 1**, behind `sf6-direct-0909v1`. Ticket `17888996582431852` was accepted at 05:34:18 KST. A previously reported 06:40 start was a scheduler estimate, not a reservation guarantee or a measured completion time. This archive contains no onepass4 GPU performance result.

The frozen source is `/home/choiceoh/stkernel-ep-onepass-0909-4` at `96cb599d8816ee2585988fc7b750a21e2eb66a0b`. Read-only identity checks before and after collection both reported that revision and an empty worktree status. `source/hashes.json` binds 142 actual generated-module, overlay-source and harness files; `source/local-verification.json` independently matches tracked files to the local frozen Git revision and generated files to local composed bytes. The copied composition manifest preserves target mappings; its upstream reference column is not substituted for current generated-file hashes.

The exact original local submission JSON, output and preparer are in `submission/`; the remote copies are preserved separately. The B1/A/B2 command keeps the pinned image, original KV capacity, actual decode graphs and canonical 2K/32K/128K workload. A enables `ENABLE_EP=1`, `VLLM_GLM53_EP_PREFILL_LOCAL=1` and required `VLLM_B12X_EP_WARM_COMPACT=1`; the scoped graph-profile control is common to all arms. These are submitted settings, not yet runtime proof.

## CPU validation evidence

| Evidence | Observed result | Scope/limit |
|---|---|---|
| New compact warmup source tests | 8 passed, 0 errors/skips, 0.415 s | Original tool output was not saved. `warmup-focused-reported.json` is explicitly a reported result, not a fabricated stdout log. |
| Scoped baseline/proof tests | 11 passed | Original local log retained. |
| Fixture repair tests | 12 passed | Original local log retained. |
| Local core | `core OK (6837 checks; 50 megakernel regressions; 0 fleet regressions)` | Includes no-Torch/host-dependent skips; not the full suite. |
| Overlay sync | 6 passed | Original local log retained. |
| Prepare contracts | 36 passed | Original local log retained. |
| Final local full suite | **FAILED**, one error | Despite its filename, `glm53-onepass4-logic-passed.log` is a failure. The retirement test's mocked Popen intercepts macOS `sysctl`, producing `ValueError: not enough values to unpack`. Earlier failed full-suite logs are also preserved. |
| Linux frozen-source logic command | Return code 0; `all OK (6839 checks; 50 megakernel regressions; 348 fleet regressions)` | The unchanged machine report says **`passed: false`, `coverage_complete: false`** because Torch/tensor tests were skipped. This is command success with incomplete coverage, not full CPU acceptance. |

The Linux command was `/usr/bin/python3 tests/test_logic.py` and took 74.146 s according to `validation/linux/cpu-report.json`; its complete log and skip list are archived unchanged. `fleet show eponepass4logic0909` was no longer resolvable at capture; the job report and log are the available completion evidence. CPU results do not establish GPU numerics, warmup completion, TTFT or throughput.

`originals.json` records original paths and raw-byte hashes for copied receipts/logs. `SHA256SUMS` covers every archive file except itself. Collection only read remote source/status/logs; it did not submit, edit, pause, reorder or run a GPU job, and did not alter older archives.
