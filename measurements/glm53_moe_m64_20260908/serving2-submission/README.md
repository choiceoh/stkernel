# Corrected Q0 scale packing: GPU gate and TTFT job

Normal fleet GO was 2026-09-08 03:01:43 KST for `moem64serve20908`.
Supervisor PID 1077384; job `/tmp/glm53-moe-m64-serving2-0908`.
Runner source `2eb268b5401c58502027f909f76197fd7342fe53` is frozen at
`/home/choiceoh/stkernel-moe-m64-serving2-0908`.
GPU source `f7d3b4b4efbe231ddc4b1282f2fe13a37e4155fb` is frozen on all ranks
at `/home/choiceoh/stkernel-moe-m64-check5-0908`. All runtime/probe bytes match.

This is a changed candidate after the completed check4 failure, not a retry of
the old kernel. Check5 evidence is `/tmp/glm53-moe-m64-check5-0908/evidence`.
The runner's `--refresh-gate` first performs TP4 numerics and sanitizers plus
recovery; only a complete pass admits the same-hold direct fresh TTFT bracket.
No numerical fix or speedup is claimed by this submission receipt. Read the
live supervisor and completion records before deciding status.
