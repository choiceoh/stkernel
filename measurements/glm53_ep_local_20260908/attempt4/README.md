# Expert-local prefill GPU attempt 4

General validation passed all nine cells: the 24-specialization exact remap byte oracle plus eight MoE numerical fixtures. The remap-only memcheck passed with zero errors. The following MoE memcheck failed with exit 86 and 34 CUDA API errors. Overall exit is 1; 11 of 17 cells ran.

Frozen source: `71e804e7aa6b29d6ddf4577809a5fa5e05a999e6`. The checkout was clean at archive time. All 13 mounted-source hashes and 13 contract-source hashes match CPU7; the archived CPU7 receipt also matches the existing CPU7 archive (37 tests, 24 compiled remap specializations).

| Fixture | Compact wall median (ms) | Local wall median (ms) | Component speedup | Compact device median (ms) | Local device median (ms) |
|---|---:|---:|---:|---:|---:|
| balanced4096 | 13.878 | 6.759 | 2.053x | 13.873 | 6.756 |
| balanced6912 | 19.783 | 7.426 | 2.664x | 19.771 | 7.416 |
| balanced8192 | 24.447 | 7.743 | 3.157x | 24.438 | 7.732 |
| concentrated6912 | 54.246 | 15.519 | 3.495x | 54.237 | 15.511 |
| remote4096 | not timed | not timed | numerics PASS | not timed | not timed |
| duplicate4096 | not timed | not timed | numerics PASS | not timed | not timed |
| zeros4097 | not timed | not timed | numerics PASS | not timed | not timed |
| balanced16384 | 40.423 | 12.158 | 3.325x | 40.414 | 12.150 |

Timing includes each arm's actual route-remap method plus MoE wrapper on one GB10. Eight alternating paired samples per arm use three calls per sample. Both arms use the same source, runtime and inputs. The baseline is the existing E72 compact top-k1 wrapper with the 8192-token pair-slice limit; this is not a production TP4 baseline. Shared expert, transport, attention/norm and full-model prefill are excluded. Attempt2 excluded remap, so its timings are not an interchangeable baseline.

Each plain MoE case passed six bounded stock-control comparisons and four candidate comparisons, including changed input/routes/scales at the same addresses and nondefault-stream replay. Remote, duplicate and zero-weight fixtures were not timed. Remap checks covered all 24 dtype/branch combinations with changed storage and exact byte comparisons.

`memcheck-remap`: PASS, `ERROR SUMMARY: 0 errors`. `memcheck-balanced4096`: FAIL, exit 86 and `ERROR SUMMARY: 34 errors`. Every reported API error names `cuGetProcAddress_v2` with `CUDA_ERROR_INVALID_VALUE`; all 34 stacks enter the original compact-control initialization. No device-fault heading is reported, but the sanitizer gate remains FAIL. The instrumented MoE probe JSON reports numerical PASS, but the sanitizer result is FAIL. The error cause remains unresolved; this archive does not classify those errors as harmless. The remaining six cells—memcheck remote/zero cases and all four racecheck cases—did not run.

All four original container IDs, image/config/host-config/mount/manifest/overlay identities matched across stop, restart and recovery. The original public endpoint was port 8000. Archived `wait_restore` writes `restored.json` only after the same original service returns health 200; no separate HTTP transcript or public refresh was recorded. Restoration had completed by runner completion at 16:08:36.985 KST. Independent fleet log/events record normal release at 16:08:37 KST; the outer process exited at 16:08:37.579 KST. The later holder/queue snapshot excludes this session.

All original job/capture files are retained under `job/`, with deterministic gzip logs. `source/` contains frozen composed files, manifest, CPU7 receipt and receipt-bound contracts. `fleet-state/` is the separate scheduler snapshot. `archive_manifest.json` records original/raw and stored checksums; `summary.json` holds recomputed medians and stage-specific verdicts. `SHA256SUMS` covers every archived/generated file. `collect.py` records the collection procedure; the final failure classification and stop/restart identity checks were additionally reviewed against raw artifacts.

Full-model TTFT, production TP4/EP4 performance, output quality, decode, serving capacity and completed MoE sanitizer validation remain separate gates. `performance_acceptance` is false and defaults remain off.
