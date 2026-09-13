# ST validation admission repair — 2026-09-13

The prior consumer failed before its first request while a large anonymous cache-reclaim mapping was refused. On the subsequent srv2 read, strict overcommit permitted about 73 GiB of additional anonymous commitments while physical MemAvailable was about 94 GiB. The original log did not retain the requested mapping size or commit counters.

`ST_RECLAIM_FILE_CACHE=1` explicitly invokes the existing `memfree-preflight.sh` after the launcher verifies/acquires its lease and checks the fleet is idle. It returns file-cache pages, leaves OS policy and anonymous workloads intact, and refuses before container launch if preparation fails. Its vLLM GMU result is discarded: ST continues to enforce its existing arena, workspace, host and OS byte budgets. The option defaults off and will be enabled only in the owned retry's production-shape env.

An anonymous mmap ENOMEM now preserves the requested GiB, overcommit policy, MemFree/MemAvailable and CommitLimit/Committed_AS rather than losing the allocation context. It does not retry or loosen any budget.

The KDA probe's `--commit-only` switch is admitted only for that pinned canonical probe. Literal absolute output paths can preserve its report under `/cache`; shell syntax and arbitrary GPU commands remain refused.

## CPU validation

- 111 arena, launcher and fleet ownership tests passed across the original run and environment repair. The original host run passed 78; 33 lacked Torch or the copied canonical probe. Only those 33 were repeated in the serving CPU image after supplying the missing files, and all passed.
- 25 canonical-admission tests passed, including the named commit mode and unsafe output strings.
- Tests execute real launcher control flow against a fake fleet: a foreign workload never triggers cache preparation; preparation failure launches no container and releases only our lease.
- No GPU work or fleet boot was performed for these host/control changes.
