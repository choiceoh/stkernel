# E72 expert-local prefill GPU attempt 2

Eight plain fixtures passed their existing numerical gates. The first sanitizer cell failed before launching compute-sanitizer (missing executable, exit 127), so the overall job exited 1. Exact original four-node recovery and normal fleet release were verified separately.

Frozen source: 36d4f006bdb0850011dccdbe2a5b8de64789e0b3. CPU5 source receipt and all 12 GPU-mounted source hashes agree across all eight fixtures. The frozen checkout was clean at archive time. Each fixture checked six stock-control comparisons and four candidate comparisons, including changed inputs/routes and nondefault-stream replay.

| Fixture | Compact wall median (ms) | Local wall median (ms) | Component speedup | Local device median (ms) |
|---|---:|---:|---:|---:|
| balanced4096 | 14.289 | 6.788 | 2.105x | 6.780 |
| balanced6912 | 19.537 | 7.377 | 2.648x | 7.369 |
| balanced8192 | 24.428 | 7.890 | 3.096x | 7.880 |
| concentrated6912 | 54.417 | 15.352 | 3.545x | 15.343 |
| remote4096 | not timed | not timed | numerics only | not timed |
| duplicate4096 | not timed | not timed | numerics only | not timed |
| zeros4097 | not timed | not timed | numerics only | not timed |
| balanced16384 | 39.895 | 12.151 | 3.283x | 12.148 |

Timing uses eight alternating paired samples per arm, with three calls per sample. These synthetic single-GB10 measurements compare the existing E72 compact top-k1 wrapper (max_num_tokens=8192) with the E72 local top8 wrapper. Both receive pre-remapped routes. They exclude EP remap, shared expert, transport and full-model prefill; they do not validate the subsequent fused-remap change or demonstrate production TP4/TTFT improvement.

The first memcheck cell ended at 2026-09-08T15:28:16.355881+09:00. The runner completed recovery by 2026-09-08T15:30:44.068567+09:00; the outer process exited at 2026-09-08T15:30:45.084757+09:00. All four original container IDs, image/config/host-config/mount/manifest/overlay identities match before and after. The original public port was already 8000. Archived lifecycle code writes restored.json only after its original /health returns 200; no separate public refresh or HTTP transcript was recorded. Fleet log and events independently record normal release at 15:30:44 KST.

Raw job/capture artifacts are under job/; every original job file was retained. Logs use deterministic gzip. source/ retains the frozen manifest, composed files, CPU5 receipt and receipt-bound contract sources. fleet-state/ contains the separate scheduler log/event/ledger and later holder/queue snapshot. archive_manifest.json verifies remote raw SHA256 against the stored files, including decompressed log bytes. summary.json contains recalculated medians and scoped conclusions. SHA256SUMS covers the archived and generated files.

Memcheck/racecheck, actual TP4/EP4 serving, full-model TTFT and answer quality remain unverified. Defaults remain off; performance_acceptance is false.
