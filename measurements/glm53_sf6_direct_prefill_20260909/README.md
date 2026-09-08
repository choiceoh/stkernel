# SF6 direct prefill implementation evidence

The direct SF6 code compiles for SM121 in the immutable serving image and
passes CPU byte, storage-lifetime, dispatch and installed-wrapper checks.
There is no new GPU numerical, model-boot, device-memory or speed result.
The queued PR #498 v5 source at 932b3fc4 is unchanged and cannot establish
acceptance of this implementation.

| Receipt | Source | Observed result |
| --- | --- | --- |
| `cpu/sf6-direct-cpu-0909/report.json` | 8b83a45a | CPU contracts pass: 71,136 core assertions, 50 megakernel cases and 68 targeted tests. M6/M8 baseline+SF6 compile. M16 static baseline+SF6 compile, but the raw dynamic wrapper fails on a misspelled DSL constexpr helper. Overall receipt remains failed. |
| `cpu/sf6-direct-cpu-0909b/report.json` | 97cf4442 | After the one-line constexpr correction: M16 static baseline/SF6 and raw dynamic compile; legacy u/v/t/q compile; 1024/2048/4096 expansion helpers compile; M80 static SF6 and M128 dynamic direct SF6 compile. All four selected stages pass. |
| `cpu/sf6-direct-cpu-0909c/report.json` | 266cbc82 | Installed-wrapper CPU probe stops at an import-time device-property query. Its CUDA initialization guard refuses the call before any device work. Failed receipt retained. |
| `cpu/sf6-direct-cpu-0909d/report.json` | 5fffd0f3 | With the import-only SM count stubbed and the CUDA guard retained, the actual decorated wrapper accepts raw scales=None and the same packed owner for M6/M16 static and M513/M4096 dynamic, three calls each. Four invalid argument combinations are rejected. CUDA remains uninitialized. |

The second source changes only `cute.const_expr` to `cutlass.const_expr` in
the dispatch JIT wrapper. The later two sources add/correct only the CPU
probe, its runner, a workspace-tail CPU test and documentation. Kernel code
compiled at 97cf4442 is retained unchanged. The final local dispatcher suite
also checks SF6 M128 workspace selection for small dynamic request tails.

All remote runs entered through explicit `fleet.sh run --cpu` with a clean,
composed, separate checkout. Containers used runc, hidden CUDA devices, no
network, two CPU cores, bounded memory/swap and separate caches. Original
reports and log bytes, including both failures, are preserved; cache/object
files are omitted. The runner's `coverage_complete=false` is intentional for
selected SF6 stages: unrelated transport combinations were not rerun.

Storage arithmetic for the 42 eligible layers observed in the earlier boot:
raw scales 4.4296875 GiB/rank; packed scales 3.3568725586 GiB/rank. Packed-only
releases 4.4296875 GiB versus the prior raw+packed SF6 candidate, or saves
1.0728149414 GiB versus raw-only. These are tensor-size calculations.
Actual model lifecycle, graph replay, GPU ordering/numerics, memory and
canonical prefill/decode speed remain unmeasured. Trace-dump mode is also
outside the installed-wrapper probe's scope. No defaults changed.
