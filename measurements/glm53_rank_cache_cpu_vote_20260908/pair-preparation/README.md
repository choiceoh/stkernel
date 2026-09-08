# Private memory-pair CPU preparation

`pinned-cpu-1.json` records exact inputs and the pinned CPU-only Docker command.
All 39 tests passed without skips; CUDA remained uninitialized. The stdout
`CPU_RESULT` marker interleaved with unittest stderr on one line; its JSON was
extracted from the retained combined raw gzip without rerunning the suite.

`offline-lifecycle.json` and `.log` record eight additional stdlib lifecycle
tests on the local host. These cover original identity, partial stop, failed
restore and loss of ownership; no remote operations were executed by them.

Tests establish configuration/evidence/lifecycle contracts only. There is no
live model request, memory savings measurement, TTFT, quality or GPU result in
this folder. See `docs/GLM53_CPU_VOTE_MEMORY_PAIR.md` for the fixed warm bracket.
