# CPU-vote memory bracket admission

Normal fleet session `glm53cpuvotemem0908v1` received GO at 2026-09-08 13:07:54 KST.
The frozen source is `6e693f53a118bce9fc0a8b457943c3880a9fb59b` at
`/home/choiceoh/stkernel-cpu-vote-memory-0908-1` on all four nodes. Each clean
checkout matched all 166 inputs in the pinned CPU report. The detached driver
PID was 33752; the head output directory is `/tmp/glm53-cpu-vote-memory-0908-1`.

Before submission, both policies passed actual Docker create-only verification
on all four nodes: eight stopped containers were created and removed, no GPU
process started, and every original identity/running state remained intact.
`create-only.json` contains the exact source/config hashes and cleanup receipts.

`driver.py` and `submitted.json` record the normal queue invocation, 65-minute
estimate and explicit supervisor clone cleanup. No queue priority was changed.
The runner uses PRIME=0 then the warm BASE0=0/CPU=1/BASE1=0 sequence, with no
model requests. A/B results, restoration and release remain pending at admission.

During admission origin/main advanced to `db89b2a` through PR #460 and the
user-side merge of PR #469. The CPU vote remains default off. The fleet's normal
public normalization selects that approved main; it does not deploy the frozen
helper branch. The two explicitly rebound cache modules remain the frozen
same-source pair; all other installed sources and configuration are captured
from the normalized original and must match every arm. The changed main common
cache helper additionally excludes the default-zero unused-graph-profile flag
from keys; the frozen pair does not. That can alter initial artifact identity,
not the between-arm comparison. PRIME is excluded and every warm hit is required.

These admission files are not a measured memory saving, TTFT result, quality
result or completed restoration receipt. Do not rerun this session while active.
