# M64 physical-block gate and direct TTFT submission

Submitted 2026-09-08 02:15:03 KST to the normal boot queue as
`moem64serve10908`, supervisor PID 846530. At 02:15 the job was first waiting
behind `earlymm0908e`; the fleet estimate was about 02:51, not a guarantee.

Runner: `1b6cf60dfaecf718d72d24c8a56d2af35071be9b`, immutable
`/home/choiceoh/stkernel-moe-m64-serving1-0908`. GPU source:
`ad8cf1b879cc28c968d5c29de1d2cc8ce51d2d73`, immutable four-node
`/home/choiceoh/stkernel-moe-m64-check4-0908`. All 56 generated overlays,
manifest, profile, launcher and GPU probes have exact byte parity. The latter
source already passed actual CPU compilation; all four distributed API checks
passed before queue admission.

Live job: `/tmp/glm53-moe-m64-serving1-0908`. The `--refresh-gate` runner first
executes BF16/FP8-v3 TP4, capture, memcheck and racecheck, with original fleet
recovery, under the same normal hold. Check4 gate evidence will appear at
`/tmp/glm53-moe-m64-check4-0908/evidence`. Only a complete gate admits
fresh-cache 2K/32K/128K B1/A/B2 TTFT and public recovery. Serving output will
appear under the live job's `evidence/`.

No GPU pass, TTFT result or speedup is represented by this submission receipt.
Read supervisor/completion and gate evidence before deciding status. Do not
edit frozen sources or submit a duplicate while this job is queued/running.
