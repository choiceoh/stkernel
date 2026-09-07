# Local M64 reuse diagnostic, 2026-09-08

The diagnostic did not complete. Plain execution completed all 48 trials with
zero candidate/control threshold failures. Memcheck recorded 40 of 48 trials
with zero candidate and six stock-control failing row-trials before its container
exited 15. Racecheck was not reached. None of these results admits serving or
establishes full-model speed, quality or numerical acceptance.

The fixed plan uses the exact local fixture from the failed normal sanitizer:
6144/6912/8192 tokens, balanced/concentrated routing, original/changed inputs,
and four alternating independent control/candidate trials. Baseline/repeat stay
fixed within each phase. The original row L2 .02 / peak .04 / 3x same-row repeat
criterion is unchanged. Input and retained-output lifetime checks pass in every
recorded trial. Row-trials from reused inputs are not independent experiments.

Plain covers 339,968 row-trials per arm. The partial memcheck covers 274,432.
Its six stock-control failures occur at 6144/balanced/changed (two) and
6912/concentrated/original (four). Every failing row retains the actual BF16
input, baseline, repeat, control and candidate. `analyze.py` verifies payload
hashes/shapes and independently reconstructs the raw error/noise/reference norms.
It also verifies the exact complete/prefix plan, source hashes, failure counts,
positive detector controls and incoming recovery. It preserves all failures.

Memcheck prints `process didn't terminate successfully`, with no Python traceback
and an error summary of zero. The Docker events retain exit 15 and no OOM event;
the queried kernel journal had no matching OOM/Xid report. This does not establish
a termination cause: the automatically removed container's final OOM/resource
state was not captured. It must not be called an OOM, disk failure, clean sanitizer
pass or M64 numerical failure on this evidence alone.

Normal fleet session `moem64reuse10908` received GO at 06:19:12 KST. Source
`b4c67dc6eabeae67d14e407dc5ff86fa68851101` was frozen on all four nodes. Probe
execution ran 06:20:24–06:24:05; the original containers/configuration/source and
healthy public endpoint were restored before outer exit 1 at **06:26:54 KST**.
Runtime bytes are unchanged from int8gate2. Frozen diagnostic sources, raw logs,
container events, CPU checks, source hashes and recovery are retained here.

The follow-up runs only unfinished sanitizer processes with exact exit and
cgroup/Docker state capture before owned-container cleanup. It preserves all
numerical limits and does not rerun the completed plain collection.
