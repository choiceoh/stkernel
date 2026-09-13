# KDA commit tiling and cursor experiment — 2026-09-13

The default-off deferred KDA path now broadcasts key/decay and update vectors over a contiguous state tile. Ring and prefix cursors advance without per-token modulo. FP32 multiply rounding and FMA order are unchanged. The flat implementation remains a direct component reference; its executable PTX instructions match the previously validated implementation after removing source locations.

The probe's `--commit-only` mode uses identical FP32 factors, initial arena, metadata and alternating launch order for flat/tiled materialization. It retains all raw samples, tests arena padding and large int64 contexts, and exercises graph replay. It does not boot the model or alter precision/acceptance policy.

## Validation

- 11 CPU execution/binding checks passed in the serving image (Torch 2.13.0+cu130).
- 25 SM121 variants compiled without initializing CUDA; both production and tail dimensions also compiled separately.
- Production C1 static PTX global load instructions: 36 → 29. This is compilation evidence, not a measured latency gain.
- GPU exactness and paired timings pending. `deferred_kda` remains disabled in production.

## Earlier consumer failure

`st-terminal-prefill-consumer0913v2` waited 718.4 seconds and stopped 56.2 seconds after admission, before loading the model or sending any request. Rank 0 failed `mmap` during anonymous cache reclamation; peers refused their arenas. The raw logs are retained here. No decode/acceptance values were produced.

The subsequent read-only srv2 snapshot showed strict `vm.overcommit_memory=2`, CommitLimit 75.85 GiB, Committed_AS 2.76 GiB, and MemAvailable 93.53 GiB. The failed run did not record its exact requested mapping size or commit counters; strict commit admission is the identified host constraint, and the next run must prepare physical memory before boot instead of repeating the same pressure allocation. No OS settings or other owner's containers were changed.
