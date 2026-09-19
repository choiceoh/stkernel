# Shared experiments for coding agents

> 살아 있는 참조 — **플릿 큐의 계약. 큐가 바뀌면 여기도 바뀐다.** 여기가 틀리면 그건 버그다.

Optimize the time from an agent's question to usable evidence. Submit once,
continue independent implementation, and read the shared result. `fleet.sh`
owns GPU admission and fast source preflight. ST candidate experiments default to
short screening; full onepass is selected for adoption. Canonical ST path checks
are available when device execution changes; they are not mandatory admission
stages for every change. The central idle controller owns production recovery after
at least five idle minutes.
Submissions never deploy or interrupt another holder. Waiting GPU jobs are
ranked at the next free fleet boundary by downstream benefit, duration and age.

## Validation by change risk (2026-09-13 operator policy)

| Change or decision | Required work |
|---|---|
| Logging, queue policy, host exception handling without device execution changes | Code review and relevant CPU tests; use `run --cpu`, without a GPU ticket |
| Kernel, CUDA graph execution, or communication changes | Relevant CPU checks and a short canonical check of the changed path; communication changes include distributed TP4 |
| Initial ST performance experiment | Default `st-pair`: one candidate boot, short C=1/C=4 screening, durable observations |
| Final performance adoption | `ST_BRACKET_VALIDATION=full`: full onepass including 32K/128K, quality/acceptance, and a comparable baseline |

Select the path from what the diff does, not merely its filename. Do not attach a
full engine/numerics campaign to a host-only repair. Reuse existing numerical
evidence only when its relevant source, weights, configuration, runtime and
hardware match; changes to the tested path require new evidence for that path.
The existing CPU receipt cache remains in use. A queued ticket keeps its frozen
approval; advancing main alone is not a reason to repeat its CPU or GPU checks.

Which CPU tests are "relevant", and which canonical check is "of the changed
path", is what `bench/feedback.py` answers — per file, the nearest check first,
the cheapest lane first:

```bash
python3 bench/feedback.py engine/kernels/mhc_contract.py
python3 bench/feedback.py --base origin/main      # every file this branch changed
python3 bench/feedback.py --index                 # the whole graph as JSON
```

| Rung | Read from | Cost |
| --- | --- | --- |
| `cpu` | `tests/test_*.py`, run through `tools/check.py` | seconds to minutes, no GPU |
| `single` | `fleet_onepass.ST_PROBES` | one Spark beside production, no fleet drain |
| `verdict` | `fleet.sh st-pair` | four Sparks; the only rung that is a speed verdict |

Nothing in it is declared by hand. A rung's lane is whatever the queue admits
today, read from `fleet_onepass.py` itself, and its distance is the import path
from the check to the file (0 = the file is the check, 1 = the check imports it
or names its repo path). It reads source with `ast` and imports nothing, so it
answers without torch. `tests/test_feedback_router.py` holds it to the queue:
every `single` command it prints must pass `fleet_onepass.validate` for one GPU.
A probe that reaches the file but is not in `ST_PROBES` is listed as
`unadmitted`: the queue rejects it, so it cannot run there until it is
byte-pinned. A rung narrows a hypothesis; a file that does not ship into serving
(`bench/`, `tests/`, `probes/`, `tools/`) has no speed verdict at all.

Runtime failures (OOM, nonfinite state, token-order or communication errors) stay
failures. A speed target miss is a recorded experimental result. Screening
quality under the short generation cap is an observation, not a passed quality
gate. It cannot satisfy final adoption or seed the next production baseline.

For a direct measurement, use `fleet.sh onepass SESSION NAME`: it waits for idle
serving and runs onepass once, without a boot. The checkout must describe that
running source. For changed code, use the ST bracket lanes (`st-pair`, `st-chain`,
`st-hold`): one committed sha per arm, booted by the release's own launcher.

`run --gpu` accepts only the canonical runners: the live onepass, the ST bracket
arms and the byte-pinned ST checks. Arbitrary wrappers, standalone GPU tests and
sanitizer campaigns are rejected before CPU preparation or queueing. The internal
`--probe` lane is retained only for the canonical live onepass. Pending command
edits and final execution recheck the policy; already-running older controllers
retain their accepted payloads.
Bare `request`/`wait` and unvalidated `adopt` cannot create new GPU holds; the
registered supervisor owns admission for every new GPU command.

The queue has three GPU lanes. A boot or a live onepass takes the
fleet: four Sparks, one holder. An ST check that needs **one** GPU
(`probes/run_engine_check.sh`, or `run_engine_probe.sh` without `--distributed`)
takes the single-GPU lane instead: **one Spark beside production**, the first of the
pool `FLEET_SINGLE_GPU_HOSTS` (srv4 srv3 srv1 srv2) with no live holder and room -- so up
to four checks run at once, one a Spark (operator, 2026-09-19) -- with a holder each (the
first host's is `holder-single`, the others' `holder-single@<host>`) and its own evidence.
An explicit `FLEET_SINGLE_GPU_HOST` names a pool of one, and empty turns the lane off. The
controller (srv2) is one of the four and runs its share itself, not over ssh; a fleet boot
waits for every single check on a Spark. Beside production the GPU is never
free, so the evidence is *room*: that box's MemAvailable less the check's budget
(`ST_PROBE_GIB`, 8 GiB by default) must clear the 16 GiB floor a `--test` boot
keeps, and only one probe container runs there at a time; a box that cannot
answer has no room. On a fleet box a fleet **boot** and a single check never
share it (the boot's admission needs that memory; the check is what earlyoom
would find first), while beside serving they run at once. The supervisor passes
`ST_PROBE_HOST` to the runner, which rsyncs `engine/` and `probes/` to that host,
waits for room, runs the container there on the image production runs there, and
takes no fleet lease. A check that needs every rank file cannot run on one node
(each Spark holds only its own rank): keep those on the fleet with
`run --gpu --fleet`. A box of its own (ost-97x, the operator's Windows PC on the
tailnet, once it has sshd in WSL2 and an x86_64 image) works the same way through
an ssh alias in the controller's `~/.ssh/config`, which owns address, user and
port. `status` shows the lane beside the fleet, a line a Spark, and `kick [--force] single [HOST]`
clears a holder (the pool's first host's without HOST).

The third lane is the **check lane** (operator, 2026-09-19: two one-GPU lanes at once):
`run --gpu --check` sends a one-GPU check to the RTX 5050 on ost-97x
(`FLEET_CHECK_GPU_HOST`; empty turns it off), with its own holder (`holder-check`), so it
never waits for the single lane or the fleet and they never wait for it. Its card is sm_120,
not a GB10: a verdict there is a compile, correctness or shape verdict, never a number for
`MEASUREMENTS.md` (CHARTER D5) -- a GB10 number stays in the single lane. The box's floor,
budget, check image and vendored flashinfer are facts in `bench/fleet_single.py` `HOSTS`
(`bench/OST_97X_LANE.md`), a probe that asks more than a kernel check's budget is refused
there, and `kick [--force] check` clears its holder.

What a single-GPU check measured comes back as a report, not only as an exit
code. The supervisor gives the ticket `ST_PROBE_REPORT` (a file under the
container's `/cache`, which is that host's `~/.cache/st`) and
`ST_PROBE_SESSION`; a check that calls `probes/probe_report.py`
`write_report(metrics, proof, samples, device)` leaves its numbers there, the
lane copies it to `results/<session>/` on release, and the queue's log says
what it says (`results of s: 3 file(s), report passed (...)`). A report written
for another ticket is unreadable, not evidence, and `passed` is recomputed from
the proof markers on the controller. It is what the check saw on that device,
never a speed verdict. The writer lives under `probes/` because the lane's host
is sent `engine/`, `probes/` and `tests/` and nothing else: a check the lane
admits may not import `bench/` at module level.

`python3 bench/feedback.py --lane-audit` lists every runner command a probe's
docstring tells a reader to run and whether the queue takes it;
`tests/test_engine_lane_promises.py` holds both rules in CI, so a probe that
names this lane without being in `ST_PROBES` fails there instead of on srv4.

## The fleet lease

Every boot holds the fleet lease -- one record, `engine/base/fleet_lease.py`, one
file on the head node -- and the queue is its authority for tickets. `run --gpu`
takes the lease as `queue/<session>` at GO and hands `ST_LEASE_OWNER` to the
payload; `launchers/start-st-glm53.sh` and `probes/run_engine_probe.sh` only
verify it. Production holds a `production` lease of its own (the supervisor and
deploy-watch boot with `ST_LEASE_KIND=production`); the queue asks that holder to
hand over only through the quiet gate -- `st:quiet` and no request for
`FLEET_QUIET_S` (120 s, deploy-watch's own rule) -- and never asks a `session`
boot (a ticket behind it waits). A handover is a transfer: the engine parks its
conversations and rewrites the lease to the ticket in one step, and at the
ticket's end the lease goes to the next waiting boot ticket, back to production
only when none waits. A bare `bash launchers/start-st-glm53.sh` or
`bash probes/run_engine_probe.sh` is refused: take a ticket, or say
`ST_LEASE_KIND=session` for a session's own boot by hand. A probe ticket
(`run --gpu --probe`, `st-probe`) is the one exception: it runs beside a
`production` lease when the door is idle -- `st:quiet` and nothing in flight,
the quiet gate's own reading -- and takes no lease; behind a `session` or a
ticket's boot it waits like everything else, and when nothing serves at all it
waits for a door (a probe measures one). Production comes back by its own
supervisor only after the queue has been quiet for a grace, and deploy-watch
takes the fleet for a deploy only after the same grace: a ticket that just ended
is likely to have the next one on its way, and a production boot in that window
is paid for twice (2026-09-13: 17 of 46 production boots were followed by a
ticket within 15 minutes). The grace is the queue's own pace, not a constant --
`bench/fleet_pace.py` rewrites `restore-grace.json` after every release with the
75th percentile of the last six hours' gaps from one ticket's end to the next
boot request, clamped to 5..20 minutes -- and a session in a campaign holds it
up with `fleet.sh window SESSION MINUTES` (`off` to close; `status` shows the
pace line). `ST_RESTORE_GRACE_S` / `--queue-grace` still set a constant. The
quiet gate itself opens at once for a production that has served nobody since
it booted, or whose last request is already older than the gate
(`vllm:request_success_total`, `st:idle_seconds`): tickets waited a median 13
minutes at that gate for a production booted for no one. `fleet.sh run ...
--replaces OLD` gives a fresh ticket OLD's place in line (a cancel followed by a
new name lost it 19 times in a day), and a one-GPU check that says `--fleet` is
told what the four Sparks cost it. The supervisor adopts a
fleet that is booting -- deploy-watch's, or its own -- and calls a launch done
only when a chat answers, not when the door listens. A boot whose ranks disagree
on what their NVMe tiers hold reconciles them and boots (`engine/base/serve.py`,
`_reconcile_parked`, run before the capture and again in the server): every rank
keeps the parked conversations and prefix boundaries every rank lists with the same
record, and drops the rest. The skew a split fleet leaves behind -- the ranks that
finished a turn parked it, the others had nothing to park -- is not a reason for
production to stay down, and what is dropped could not have been resumed without
every rank's part.

## The ST bracket

`fleet.sh st-pair SESSION SHA [--base SHA] [EST] [NOTE]` screens one committed
ST candidate, `fleet.sh st-chain SESSION
[EST] [NOTE] -- A=SHA B=SHA A B` runs arms in order (a repeated name is another
boot of the same commit: `A B A B` alternates, the way 45차 §93 asked), and
`fleet.sh st-hold SESSION SHA [EST]` boots a commit and keeps it for a session's
window (`fleet.sh cancel SESSION` ends it). All three are boot tickets: the queue
takes the fleet lease at GO and the arm's own launcher verifies it.

An arm is a sha origin has (a working tree is not citable). `bench/st_bracket.sh`
cuts it into `~/st-releases/<sha12>` with `launchers/st_release.py` -- the same
cut deploy-watch makes, so a winner is promoted by pointing production at that
directory -- pushes it to the four nodes, and boots it in production shape
(`~/.config/st-glm53.env`: KV, rows, `ST_PRODUCTION=1`) on port 8001 with a tier
and dump directory of its own. STK_ knobs are not arms.

The default `ST_BRACKET_VALIDATION=screen` leg is one boot, one short run of
`bench/st_screen.py`, then stop. It uses the onepass streaming client, a
deterministic hard question with a 2K document, and C=1/C=4. Each concurrency
first prepares for 64 tokens, then measures up to 512 tokens (minimum 128,
reasoning budget 256). Actual prompt tokens, TTFT, output tok/s, completion
time, acceptance, raw answers and per-request/chunk/host-stage latency are
retained in the usual ledger and `onepass-runs/` artifacts. Preparation has its
own artifacts. Additional compilation or cache reuse makes timing unverified;
it does not trigger another run or hide the observation. No profiler replay is
scheduled. Screening does not provide per-kernel GPU durations.

In screen mode, `st-pair` runs only the candidate; `--base` does not schedule a
baseline boot or a comparison. A deployed baseline is not required. `st-chain`
screens every requested arm, including repeated names. Its `--reuse` applies
only in full mode. These records carry `evidence_scope=screen` and
`adoption_eligible=false`; the judge excludes them from warm samples, cold
columns and noise floors, even if a short answer happens to pass its checks.

For adoption, set `ST_BRACKET_VALIDATION=full` when submitting the ticket:

```bash
ST_BRACKET_VALIDATION=full bash bench/fleet.sh st-pair SESSION SHA --base BASE_SHA
```

Full mode retains boot, onepass (the cold column), `POST /v1/prefix/reset`,
onepass (the warm column), stop; C=4 is measured on the first run only.
`bench/st_judge.py` judges warm against warm with the base's run-to-run spread
as the floor and prints the cold column beside it. Full `st-pair` boots the
base only when it has no warm sample yet. `FLEET_REHEARSE=1`
boots nothing and fabricates records, so the flow can be checked without GPUs.

The base is measured as little as possible (the operator's rule, 2026-09-13): a
candidate that is adopted brings its own measurement along as the next baseline,
and one that is not leaves the baseline as it was. A sample's identity is the
engine tree (`arm_tree`, `launchers/st_release.py tree`), not only the commit: the
squash main makes of a measured branch, or a fleet-side merge that left `engine/`
alone, is the same engine and the same sample. When the base has one boot, the
judge borrows the floor -- the median run-to-run spread of every commit with two
boots in the records -- and says so ("pooled floor"). A D17 probe on the live
door runs once (a run on a door that has been serving is a warm sample; a boot is one sample
however many runs it carries), deploy-watch keeps the deployed engine at one
sample (`--probe-samples`), and full `st-chain --reuse` boots no arm that already has
a sample. deploy-watch applies the same identity to deploys: a main that moved
without touching `engine/` is recorded as deployed, cut and followed by the
controller, and not booted -- the engine that serves is already that commit's.

`fleet.sh st-probe SESSION [SHA] [EST] [NOTE]` is the verb that boots nothing:
one **full** onepass run on the LIVE production door as a probe ticket, regardless of
`ST_BRACKET_VALIDATION`, so it runs beside production when the door is idle and
takes no lease. It resets nothing: production's prefix cache and prefix tier are
production's (until 2026-09-13 it POSTed `/v1/prefix/reset` first and emptied both
after every deploy). Its requests carry unique cache salts, so they cannot hit what
production cached, and say `retain: false`, so the engine caches their boundaries
for the turn only -- never on the prefix tier, first to give up a snapshot, dropped
with the row (`st:prefix_transient_dropped_total`). Its run is `cold=live`
(`cold=reset` on older records), which `st_judge` keeps out of the cold column (a
boot's). deploy-watch queues one after every deploy
(`d17-<sha12>`; `--no-probe` to stop it), so the deployed commit always has a
warm sample and `st-pair` never has to boot the base; it also moves the queue's
own checkout, `~/fleet-controller` (`--controller`, `FLEET_CONTROLLER_REPO`;
`--no-follow`), to the deployed commit, so the queue answers by production's
rules and 45차 §91's 639-commit drift cannot recur. The sample is kept, not just
queued once: on every later cycle with nothing to deploy, deploy-watch asks the
controller's judge whether the deployed commit has a warm sample and, when it
has none and no `d17-<sha12>` ticket is queued or holding, queues another --
at most `--probe-attempts` (3) per production boot, `--probe-gap` (1800 s) apart,
tallied in `deploy-state.json`. The probe itself refuses to run when the door's
`ST_RELEASE` is not the sha it was queued for, so a ticket queued before a
deploy cannot label the next engine's numbers with the old commit.

Plans now batch their independent CPU stages, publish reusable evidence before
creating another checkout on a cache hit, and keep core/fleet/startup results
separate. A consumer's declared dependencies still decide when it can execute.

## Agent workflow

Run the commands on the fleet head, from a **committed, clean checkout**. Each
experiment needing execution gets a detached private checkout of that commit, so the agent
can immediately continue editing its original checkout. Build outputs from CPU
checks are isolated too. Put submission manifests and reports outside the repo.

1. State the hypothesis. The fast CPU source admission is automatic; do not add
   the full logic suite as a routine prerequisite. Select a focused CPU test only
   when the change needs it.
2. If needed, submit that CPU check; an identical request joins it or reads its
   saved result. Do independent work while it runs.
3. Submit one GPU candidate with the CPU experiment ID in `depends_on`.
   Omit `depends_on` when no additional CPU test is necessary. The worker waits
   outside the GPU queue until requested prerequisites succeed. A failed,
   incomplete, interrupted or blocked prerequisite prevents GPU admission.
4. Read `inbox` for incremental results, or `result ID` for the complete evidence.
   Inspect `state` and `result.evidence`; an exit code or a marker count alone is
   not a performance verdict.
5. Submit the next hypothesis only when its dependencies are known. Use
   `--repeat 'reason'` for a deliberate independent sample, never a new name to
   bypass a saved failure.

Example CPU submission (the revision is frozen at submission):

```bash
export REPO=/home/choiceoh/stkernel
request_dir=$(mktemp -d /tmp/fleet-request.XXXXXX)
jq -n --arg rev "$(git -C "$REPO" rev-parse HEAD)" '{
  kind: "cpu", revision: $rev,
  hypothesis: "The deployment identity change preserves onepass source selection",
  command: ["python3", "bench/cpu_checks.py", "--suite", "fleet"],
  timeout_s: 30
}' > "$request_dir/cpu.json"
bash "$REPO/bench/fleet.sh" submit fusion "$request_dir/cpu.json"
# {"id":"...","disposition":"submitted|joined|reused","state":"..."}
```

```bash
bash "$REPO/bench/fleet.sh" inbox fusion --after 0
# Keep the returned cursor and use it as --after on the next call.
bash "$REPO/bench/fleet.sh" result EXPERIMENT_ID
bash "$REPO/bench/fleet.sh" await EXPERIMENT_ID --timeout 60
bash "$REPO/bench/fleet.sh" stats
```

`await` waits for at most 60 seconds. Submission does not leave a blocking tool
call in the agent. Workers, logs, subscriptions, reports and SQLite state live
under `$FLEET_DIR/experiments`; no scheduler service needs installation. `inbox`
provides durable events for agent supervisors to consume; it does not itself
send a message to a Codex/Claude session or automatically resume that session.
`result` returns evidence and artifact paths; add `--details` only when the full
pinned environment and input hashes are needed.

Results and terminal inbox events include `explanation`: the failed checks and
test tracebacks, blocking dependency IDs, relevant logs and suggested argv/cwd
for inspecting or reproducing the failure. These suggestions never run commands,
retry a failed experiment or send messages to an agent. CPU reproduction hides
CUDA devices and retains CPU-only evidence scope. Fix the input/revision before
submitting a corrected experiment; a successful process exit is still insufficient
to promote incomplete CPU or onepass evidence.

## Find your work and inspect a reservation

```bash
# Only your active structured requests; includes requests shared with peers.
bash bench/fleet.sh jobs --session fusion --active
bash bench/fleet.sh jobs --session fusion --limit 20

# Fast local snapshot: no Docker, SSH, serving health, or baseline queries.
bash bench/fleet.sh show
bash bench/fleet.sh show fusion
bash bench/fleet.sh show fusion --json
bash bench/fleet.sh logs fusion --tail 80
```

`jobs` retains its existing JSON list format. Session filtering happens before
the result limit, so unrelated recent jobs cannot hide your older request.
Withdrawn subscriptions are omitted only for that subscriber; `--active` excludes
all terminal states. The default limit is 100; `--limit` accepts 1..1000.

`show` with no name lists the current holder and queue. With a reservation name,
it reports the exact argv/cwd/revision, position and current blocker while queued,
phase and elapsed time while executing, and retained exit status after completion.
It also tells you whether editing is still possible and prints the matching edit,
log and structured-result commands. It does not run, retry, promote, cancel or
acquire anything. Use the existing `status` command when serving-health checks are
needed.

New `run --gpu` supervisors retain combined waiting/payload output in a private
per-ticket log while continuing live output. Sessions release the fleet after
measurement; they do not restore production. A cancelled or dead supervisor is
not reported as successful. A saved `log_error` warns that output may be incomplete. A blocked or disconnected viewer may miss live chunks; it can retrieve
the retained output with `logs`.

`logs` reads at most the last 256 KiB and accepts 1..2000 lines (default 80).
The command never follows a pipe or waits for future output. Older live controllers
can expose their existing stdout log if it is a regular file. Missing old terminal
output or completion status is reported as unavailable instead of guessed. A
reused session name shows the latest reservation; earlier per-ticket log files
remain separate.

## Edit a waiting reservation

For a reservation created by the current `fleet.sh run --gpu` (an ST bracket arm
or a canonical check), inspect and revise it before GO:

```bash
bash bench/fleet.sh edit fusion
bash bench/fleet.sh edit fusion --expect-revision 1 --est 20 --note "updated cells" \
  --cwd /home/choiceoh/stkernel -- bash bench/st_bracket.sh pair 0123456789abcdef
# Metadata only; the existing command is retained:
bash bench/fleet.sh edit fusion --note "CPU checks passed; smaller workload"
```

`--` replaces the entire argv, without shell interpolation. Use `env KEY=value
command ...` to set command-specific environment variables. Otherwise the
original supervisor environment is retained; `--cwd` changes the payload's
working directory. The executable and the actual replacement command must pass
preflight using the waiter's pinned controller before the edit commits. A failed
check, concurrent edit or admission during preflight keeps the previous command.
`--expect-revision` prevents an agent from overwriting a revision it has not read.

Edits retain the session, ticket, original enqueue time, PID, GPU kind and
queue neighbors. Normal boundary scheduling still applies; changing the duration
can change its priority. Admission and edits use the same fleet lock: after GO,
or during recovery, edits are refused. The private pending record retains prior
revisions; the lifecycle acceptance event records the revision actually executed.
No new reservation, baseline measurement or production restore is created by an
edit.

Already-running **older** controllers and bare `request`/`wait` reservations do
not have an editable command record and are explicitly refused. An installed
update cannot replace another process's pinned controller. Structured `submit`
experiments allow queue note/estimate edits, but their command/cwd remain bound to
the submitted evidence identity. Change their manifest through
`submit --supersedes` instead; that replacement follows normal submission order.

## Submit a batch

```json
[
  {"name": "math", "manifest": {"kind": "cpu", "revision": "FULL_COMMITTED_SHA", "hypothesis": "Check chunk boundaries", "command": ["python3", "bench/cpu_checks.py", "--contract", "math"]}},
  {"name": "layout", "manifest": {"kind": "cpu", "revision": "FULL_COMMITTED_SHA", "hypothesis": "Check shard coverage", "command": ["python3", "bench/cpu_checks.py", "--contract", "layout"]}}
]
```

```bash
bash bench/fleet.sh batch fusion /tmp/cpu-batch.json
```

A batch contains 1..32 named requests. Optional `requires` lists earlier request
names; `manifest.depends_on` can still name existing experiment IDs. The runner
attests all requests before atomically registering the DAG. Invalid requests,
cycles, changed inputs or mismatched prerequisite revisions register no jobs.
Identical source/environment reads share a memo only within that call, with a
fresh second pass before registration. There is no persistent fingerprint TTL.
Different commands, inputs, runtime context and resource budgets keep distinct
evidence identities. `--repeat REASON` requests independent executions.

## Plan from a decision

`plan` is CPU-only: it connects the CPU checks, optional CPU compilation and CPU
preparation stages into one submitted DAG. The GPU stage left with the pair lane
(2026-09-18) — GPU work is queued directly through `fleet.sh run --gpu` and the
ST bracket lanes, so a plan no longer carries a GPU manifest. Manifests stay
outside the checkout; every stage uses the same committed revision. The overlay
math contracts and the startup suite retired with the overlay stack; the fleet
suite is the planner's default CPU gate, and `cpu_tests` names additional files.

```json
{
  "kind": "cpu",
  "hypothesis": "The route-cache change preserves indexer coverage",
  "revision": "COMMITTED_SHA_OF_THE_CANDIDATE",
  "cpu_suites": ["fleet"],
  "cpu_tests": ["tests/test_prefill_route_cache.py"],
  "resources": {
    "nodes": ["local", "choiceoh@10.10.10.1", "choiceoh@10.10.10.3", "choiceoh@10.10.10.4"],
    "disk_path": "/home/choiceoh", "disk_mb": 4096, "node_memory_mb": 4096
  }
}
```

## CPU preparation and resource admission

Individual `python3 tests/test_foo.py` or exact unittest discovery commands are
adapted to `cpu_checks.py --test tests/test_foo.py` when the file imports unittest.
The runner records executed test counts, errors, failures and skips from the
unittest result object. Zero tests, skips, expected failures and failed checks
cannot unlock a dependent GPU job. Arbitrary commands keep process-exit evidence
and do not acquire the named-test content cache.

Managed CPU submissions share a resource pool in the experiment database.
`resources.cpu_slots` defaults to 1 and `resources.cpu_memory_mb` to 4096.
The pool defaults to two slots, at most 16 GiB or half host RAM, and a host RAM
reserve. Operators can set `$FLEET_DIR/cpu-policy.json`, for example:

```json
{"slots": 2, "memory_mb": 8192, "reserve_mb": 2048}
```

Jobs wait without a GPU hold in persistent FIFO ticket order until their
reservation fits both the pool and available RAM. New small jobs cannot pass an
older multi-slot request. This can temporarily leave capacity idle while the head
waiter drains the pool; it prevents starvation without preempting active work.
Dead, retired and newly over-capacity waiters are removed. Only an eligible head
waiter probes available host RAM; waiters recheck admission every 50 ms. All
workers must use the current runner to enforce this ordering.

The CPU timeout includes that wait. An approximately 100 ms process-group RSS
poll enforces the declared RAM budget, using a group/session-filtered process
query instead of collecting every host process. Process-exit waits return early
when a short command finishes; an over-budget job and its remaining
children are terminated before the reservation is released. This is sampled
accounting, not an OS hard memory sandbox: a burst can exceed the budget between
polls, and intentionally detached processes are outside the group. Direct legacy
`fleet.sh run --cpu` jobs do not use this managed pool. Use one shared experiment
root per host. GPU readiness thresholds are optional and checked outside the
queue and again at GO; they are observations, not reserved node memory or disk.
Waiting for an identical in-flight CPU owner has its own timeout of
`timeout_s`; if it cannot reuse that owner's result, a new resource/run budget
starts. Superseded resource waiters exit before launching their CPU command.

Add `prepare` stages to a plan when a kernel needs a compile check before a GPU
slot. This reviewed entrypoint only invokes `nvcc --compile` with an explicit
architecture, without loading a CUDA context:

```json
"prepare": [{
  "command": ["python3", "bench/cpu_compile.py", "--source", "path/to/kernel.cu",
              "--arch", "sm_90", "--output", "build/kernel.o"],
  "outputs": ["build/kernel.o"],
  "resources": {"cpu_memory_mb": 4096, "cpu_slots": 1},
  "timeout_s": 900
}]
```

Preparation stages now run independently of checks by default, within the CPU
pool. Set `"requires": ["checks"]` for a real check dependency, or name earlier
preparation stages such as `"requires": ["prepare-1"]` when a build consumes
another build's outputs. Only explicit dependencies transfer verified artifacts
into that stage's private checkout. A failed check still blocks the GPU stage
even if an independent compile has already completed. Named CPU checks with
generated-artifact prerequisites do not reuse the tracked-source content cache.

Other CPU preparation argv commands can declare outputs too. Only ignored files
under `build/` can be exported; source paths and symlinks are rejected. Each
successful preparation result records output hashes. The consumer copies them
into its private checkout after checking runtime context and hashes, then
rechecks them at execution. Changed/missing/conflicting artifacts block the job.
These outputs are available to the consumer; they do not replace deployment,
inject objects into the serving container, or prove GPU numerical correctness
automatically.

## Receive results without frequent polling

```bash
bash bench/fleet.sh inbox prefill --after CURSOR --wait 60
# After the agent/supervisor has persisted the returned events:
bash bench/fleet.sh ack prefill RETURNED_CURSOR
# Make an existing private JSON/JSONL result discoverable in that same inbox:
bash bench/fleet.sh collect prefill /absolute/path/to/private-results.jsonl
```

Long polling returns when subscribed events arrive. Repeated reads and acks
retain the first timestamps. A cursor must have been delivered to that session
before it can be acknowledged. The collector archives up to 64 MiB with a
content hash and publishes an `external-archive` event/result. Imported records
stay incomplete until separately adjudicated; they cannot unlock dependencies
or enter the trusted baseline ledger merely by being imported. No native agent
session wakeup or external messaging is installed by these commands.

## CPU checks and evidence quality

`bench/cpu_checks.py` runs existing behavioral suites and keeps one log per
command plus a JSON report. The experiment result embeds that report, including
failed checks. Checks run under the CPU lane with a bounded time budget and do
not reserve GPUs. Named suites hide devices and limit CPU math libraries to one
thread. Skipped checks (for example missing PyTorch) make the report incomplete
and return exit code 3, so a partial CPU run cannot unlock dependent GPU work.

| Suite | Evidence supplied |
| --- | --- |
| `logic` | Reference math, layouts, gates and extracted dispatch logic; includes the megakernel and fleet behavioral regressions in `tests/test_logic.py` |
| `core` | The same core math/source checks and megakernel regressions, without executing the fleet behavioral suite |
| `fleet` | Concurrent submissions, duplicate consumers, private source snapshots, CPU prerequisites, failures, timeouts, evidence compatibility and shell failure propagation |
| `startup` | Launcher/worker startup, file attestation, memory preflight and reclamation using local fakes |
| `sensitivity` | Three passing helper controls and four in-memory faults detected by the existing math/layout/dispatch assertions |

For incremental feedback, `cpu_checks.py --contract math`, `--contract layout`
and `--contract dispatch` run the actual audited tests from `tests/test_logic.py`:

| Contract | Audited helper |
| --- | --- |
| `math` | Indexer prefill chunk size, byte budget and request boundaries |
| `layout` | Sequence-parallel shard ownership ranges |
| `dispatch` | Indexer top-k reuse gate |

A plan can explicitly set `"cpu_contracts": ["math", "layout", "dispatch"]`.
Each contract becomes a separate CPU prerequisite with its own cache, alongside
the sensitivity check. Other requested suites/tests remain additional gates.
Dependency audits pin the test source, helper access graph and loader path
inventory. New imports/call targets/name/attribute access or unknown test edits
fall back to whole-tree cache identity and conservative automatic suite selection. No caller-provided path exclusion
can narrow this scope. These three small contracts provide incremental feedback;
the full deployment `logic` gate remains in place.

Sensitivity changes only in-memory ASTs: logits byte width, request end boundary,
shard stride and reuse-mask polarity. All original controls must pass and every
mutant must trigger the existing checker's failure. A surviving fault, skipped
test, changed/unsupported mutation target or unexpected error fails the report.
The catalog measures sensitivity to these four faults, not arbitrary mutation
coverage or GPU correctness; production source files are never modified.

Select by the changed contract. A scheduling change normally needs `fleet`;
a kernel math/layout change needs `logic`; a launcher change needs `startup`.
The repository's deployment gate still runs `logic`. Avoid running both `fleet`
and `logic` for the same revision unless investigating a new failure: `logic`
already includes `fleet`. Source checks and mocked dispatch tests cannot prove
device numerics, graph replay, race freedom, serving quality or throughput.

The default `python3 tests/test_logic.py` deployment gate still executes both
core and fleet components. `--component core` selects only the core component.
Reviewed fleet cases can run in separate processes, capped by the CPU slots
reserved for the job (and at most eight). Every case ID must appear exactly once;
failed/missing shards, skipped tests and zero-test reports cannot pass. Unknown
fleet test edits default to serial execution until their isolation audit is
reviewed. The standalone unittest runner allows explicit `--jobs N` for a caller
who has reviewed its test isolation. Concurrency/queue tests use state signals
instead of fixed sleeps; timeout enforcement still has real elapsed-time tests.

Complete fleet runs record per-case durations in
`$FLEET_EXPERIMENT_ROOT/cpu-test-timings.json` (standalone default: `build/`).
Profiles separate host, architecture, Python major/minor and worker count; each
case is keyed by its test file hash and ID. Subsequent runs assign the longest
cases first to the least-loaded shard. Unknown cases use the median known time;
without usable history the runner retains round-robin assignment. History is
size-bounded, locked and atomically replaced; unusable history is ignored. It
only changes scheduling: every child validates the complete assignment and the
parent still verifies exact executed-ID coverage. Test edits invalidate their
timing hints and the existing isolation audit still controls automatic sharding.

Custom CPU commands are also supported as argv arrays. Use explicit interpreter
and dependency identifiers in `context`, and hash external fixtures/lockfiles
with `inputs`. The existing GPU-use classifier still applies. If it identifies
device calls in a test file's source even though the test uses mocks, use the
named CPU suite entrypoint; do not weaken the classifier or add an override.

## The pair lane retired

The `kind: pair` / `kind: baseline` GPU experiments booted the vLLM overlay
stack's wrapper, which the decommission removed (2026-09-18). Queued pair jobs
are blocked at the worker with `PAIR_RETIRED`; the ST bracket (`st-pair`,
`st-chain`, `st-hold`) is the way one commit is measured against another.

## Onepass-only GPU submissions

The pair manifest form retired with the overlay stack (see the section above).
GPU work enters this queue through `fleet.sh st-pair` / `st-chain` / `st-hold`
(the ST bracket) or `fleet.sh onepass` (a live measurement on the idle door),
both admitted through bench/fleet_onepass.py's canonical-entry contract.
Completed historical reports remain readable.

Use an optional relevant CPU request in `depends_on`, then submit the candidate
arm directly. Onepass supplies serving quality, corruption and performance
evidence. Additional GPU microbenchmarks, prechecks, sanitizers and
post-measurement sweeps are excluded from this path.

## Queue policy and CPU content reuse

`bash bench/fleet.sh priority` explains the current ranking. At a free fleet
boundary, a ready job's score is `(1 + pending transitive dependents) /
estimate_min + wait_seconds / 1800`. At 30 minutes waiting, oldest-first takes
precedence, so a stream of tiny jobs cannot indefinitely starve a long one.
Explicit `front` retains its order. Legacy jobs participate with zero known
dependents. Ranking never interrupts a live holder;
CPU jobs and prerequisite waits never enter this GPU queue. Estimates use the
declared duration until at least three successful matching execution samples
exist, then the p90 of the latest 20 (rounded up to minutes). Matching includes
kind, host, runtime, command/configuration, resource budget, external inputs and
the physical workload union; source revision is excluded only from this timing
estimate, never from GPU evidence. Cache hits and shared consumers supply no
execution-duration samples. `fleet.sh estimate ID` explains the estimate used
by queue ranking. The DB read failing falls back to
zero known dependents, and a scheduler failure retains the existing file order.

Named `cpu_checks.py --suite ...` and `--test tests/test_*.py` submissions also
consult a content cache.
Submit the CPU request for the new revision normally; the new result can cite a
previous successful complete CPU report via `cache_source`, `tested_revision`
and `cache_identity`. It still carries the new revision for dependent requests.
An identical-tree merge can reuse the full-tree cache. The reviewed startup
tests use a narrower tests/launchers/bench/profiles scope, allowing unrelated
documentation/kernel edits to reuse their startup evidence. Test source hashes
pin this dependency audit: a changed test automatically falls back to the whole
tracked tree. Audited helper contracts use their pinned source/test/runner
dependencies and overlay path inventory. The reviewed fleet suite uses tests,
bench, launcher/profile sources, its actual helper dependencies and overlay path
inventory, so unrelated kernel-body changes can reuse its result. New/changed
fleet tests or helper dependencies fall back to the whole tracked tree. Core and
aggregate logic conservatively include the whole tree, including docs that tests
may inspect. There are no caller-supplied exclusions.

Keys also include interpreter and shell tool binaries, installed package
metadata and file size/mtime inventory, controlled environment, runtime context,
external input hashes, command, timeout, resource budgets and declared outputs. Editable Python installations and
unavailable runtime fingerprints disable content reuse. The package inventory
detects normal local package edits; use an immutable environment identifier in
`context` when package files may be replaced while preserving metadata. This is
a local experiment cache, not a cryptographic attestation of the whole machine.
Custom CPU commands, skipped/failed reports and deliberate `--repeat` requests
do not use the content cache. GPU results remain bound to their original build
and runtime.

The same identity also coalesces in-flight named CPU checks across different
commit SHAs. A filesystem claim is inherited by the fleet child, so a supervisor
crash cannot launch a second identical check while its child is running. The
follower waits without CPU-pool or GPU reservations, then revalidates its own
snapshot/environment and the completed report/artifact hashes. A failed shared
owner blocks its waiting consumers instead of repeating the same failing check.
Explicit repeats run independently, serialized behind the current claim.

A fresh completed cache hit is published during submission without creating a
private checkout or starting a worker. It still checks the new source/runtime
identity, all prerequisites and the saved report/artifact integrity. A concurrent
worker's lock prevents the fast path from completing work that is still running.
Cache misses and in-flight followers retain the isolated worker path. Results
carry `cache_source`, `tested_revision` and `cache_path: "before-checkout"` for
this path; they do not claim the new revision was re-executed.

## Replace obsolete requests

```bash
bash bench/fleet.sh submit fusion /tmp/new-request.json --supersedes OLD_ID
bash bench/fleet.sh retire fusion OLD_ID --replacement NEW_ID --reason 'New revision replaces this request'
# Also accepted when plan actually submits the GPU stage:
bash bench/fleet.sh plan fusion /tmp/new-plan.json --submit --supersedes OLD_GPU_ID
```

Only a subscribed session can replace its own request with another viable
subscribed request of the same kind. This withdraws that session's demand.
The old job becomes `retired` only before execution and when no active subscriber
or unfinished dependent needs it, including internal CPU/baseline/shared-boot
consumers. Running work finishes normally. Retirement removes only that job's
queue entry; managed wait/admission observes it without requeueing or launching.
It never signals processes or edits the live holder. Resubmitting a retired
request creates new work. `--prepare-only` does not supersede a GPU job; pass
`--supersedes` when submitting its returned `gpu.json` later.

## Identity, failure and recovery

The identity includes the committed source, runner version, host/interpreter,
controlled environment, command/knobs, workload settings, immutable runtime
identifiers, hashes of external inputs, and prerequisite IDs. Human labels,
hypotheses and estimates do not cause duplicate execution. SSH transport socket
paths do not affect identity. Unlisted inherited experimental environment
variables are not carried into the worker; declare inputs explicitly.

Requests from multiple agents attach to one job atomically. Completed failures
and incomplete results are shared too, with their status intact. Additional
samples require an explicit repeat reason, which is recorded. Prerequisites
must use the same revision and matching pinned external inputs.

Workers retain an OS lock through their child process. A second subscriber or
a crashed supervisor cannot launch a second copy while the first child is still
running. An abandoned in-flight job becomes `interrupted` when queried and is
not automatically retried; inspect its log and the fleet state before requesting
a repeat. Snapshot checkouts and logs are retained for inspection. Remove a
finished snapshot only with `git worktree remove` after preserving needed
artifacts; never remove a queued/running job's checkout.

`stats` separates CPU/pair/baseline counts and historical probe records, shared/reused requests, and p50/p95 time
to start and to a successful result. It also reports completion-to-delivery and
completion-to-acknowledgment p50/p95 across subscribed consumers. The regular
fields cover successful results; `terminal_*` fields also include failures,
blocked jobs, interrupted jobs and incomplete results. These measure
API receipt and explicit acknowledgment, not when an agent understands or acts
on a result. Queue time includes prerequisite waiting.
An incomplete/failed result is never counted as a successful fast result. The
initial implementation measures these timings; it makes no speedup claim until
matched live runs establish one.

`stats.phases` also reports dependency/shared-evidence/resource waits, GPU queue,
preflight, preparation, CPU execution, boot, measurement and restore durations.
`gpu_run` includes its nested boot/measurement/restore phases, and restore may
include another boot; do not sum overlapping phases. Legacy runners without the
managed experiment environment do not emit these markers. These observations
explain turnaround and guide estimates; fixture boot counts or CPU-cache timings
alone do not establish a live GPU wait-time or serving speedup.

## Finish once and hand off stopped serving

`fleet.sh run --gpu` now supervises boot payloads. Nested `pair`, `chain`, and
managed serving groups defer their bare restore to that supervisor. Measurement
baselines remain measurements. At completion, the supervisor selects the next
job using queue priority and requires a live boot supervisor receipt, including
PID start time. It records restore responsibility, pins the successor, releases,
and waits up to 30 seconds for admission. A cancelled receiver makes the donor
reclaim the hold through normal admission and restore. A receiver that fails
before boot still restores. Probes and unsupervised waiters cannot inherit this
responsibility. No live holder is preempted.

Protocol 2 commits the holder before transferring restore debt. If admission is
cancelled between those writes, the receiver reconciles ownership from its own
hold before recovery; the donor never accepts a transfer with no receiver hold.
Older pinned protocols finish at a restored boundary instead of receiving a new
protocol handoff.

The last vLLM production holder retired with the overlay stack; the queue no
longer hands off to a restore boot. `restore-debt.json` bookkeeping remains
inert for older queued protocols and is never written today.

Custom boot scripts must accept stopped serving and attest the serving image
with `docker image inspect` before stopping anything. Arbitrary shell code is
not exhaustively linted.

`lifecycle.jsonl` records ready/source hashes, acceptance, payload completion,
handoff, reclaim and restore duration. `fleet.sh version` exposes the active
protocol and source hashes. Fleet waiters now poll at one second instead of
15 seconds. Nested legacy yields defer to the supervised finish boundary.
Control scripts are pinned by content under `fleet/runners/` before admission;
updating the shared checkout affects new submissions, not an in-flight
supervisor's queue/restore helpers. Payload checkouts retain their own source
validation contract.

## Share startup controls across a campaign

Commit/deploy one build containing the candidate toggles, then use:

```bash
REPO="$PWD" bash bench/fleet.sh startup startup-agent bench/startup-campaign.example.json 45
```

The default example runs `PRIME, BASE1, FASTIOR1, SHAKEYR1, SHAKEYR2, FASTIOR2`:
six boots with one shared baseline. Each candidate still has two boots. Three
candidates use eight boots. Set `"baseline_policy": "confirm"` to append `BASE2`
and check drift: seven boots for two candidates, or nine for three. These are
boot counts, not measured wall-clock savings. Both shared cache directories stay
fixed; every arm explicitly sets the same knob keys. Malformed, duplicate,
cache-off or changed-cache campaigns fail validation.

Every arm retains four-node cache receipts, canonical onepass response/quality
evidence and a distinct matching boot. Pack IO/key campaigns retain their GPU
checks on PRIME. The campaign pins source/profile/deployed manifest/workload,
rejects changed identities. The minimal result is `exploration-unconfirmed`
with `baseline_drift_fraction: null`; no drift bound is inferred from one
baseline. Confirmation compares its two controls for drift (10% default, at
most 25%). `campaign-result.json` contains health timings and paired candidate
summaries. Both modes are exploration evidence with `promotion_ready: false`;
excessive observed drift makes confirmation incomplete. Final promotion needs the relevant direct
consumer metric and independent matched validation. Older unrelated builds'
baselines are never reused.

CPU validation of these changes uses the regular fleet suite, startup campaign
and receipt tests. Run `tests/test_boot_supervisor_linux.py` explicitly on Linux
for real shell admission, cancellation and handoff with fake system commands;
it never accesses GPUs, SSH or production containers.

## Detach, retry and inspect an exact reservation

```bash
REPO="$PWD" bash bench/fleet.sh run --gpu --detach agent 20 "candidate" -- bash probes/candidate.sh
REPO="$PWD" bash bench/fleet.sh run --cpu --detach cpu-agent -- python3 bench/cpu_checks.py --suite fleet
bash bench/fleet.sh history agent --json
bash bench/fleet.sh show agent --ticket TICKET
bash bench/fleet.sh logs agent --ticket TICKET
bash bench/fleet.sh classify --explain bash probes/candidate.sh
bash bench/fleet.sh retry agent EXPERIMENT_ID --reason "temporary dependency restored"
```

Detached launch returns a private log path, process identity and a durable queued
ticket or CPU-start receipt. A repeated identical launch joins the same process;
its timeout returns `accepted=false`, never a fabricated ticket. A completed launch
retains its result; use a fresh session for independent raw work. Queued commands
can still be replaced with `edit`. GPU reservation history keeps previous commands,
outcomes and logs by ticket, including when a session name is reused. Discovery
lists the recent 1,000 tickets; older known tickets remain directly accessible.
CPU detach exposes its startup log and completion receipt, without creating a GPU
reservation. `classify --explain` reports the matching argv or source line; the
classification policy remains unchanged and CPU refusal prints those reasons.

`retry` applies to experiments created by `submit`/`batch`, whose source, environment,
inputs and dependencies have a saved manifest. It accepts failed, blocked or
interrupted attempts and retains the original result. It freshly attests the saved
source, joins compatible work and keeps successful CPU evidence and baseline samples.
A failed shared baseline is acquired again only for an explicit retry. This is not
an independent `--repeat` measurement. Failed prerequisites must be repaired explicitly.
Raw shell requests lack a complete declared environment/dependency contract; retrying
those uses a new `run` with the intended command and inputs shown by `show`.

## Prepare ordinary runs before they take a turn

Every new `run` checks executable/script existence, shell/Python syntax and binds
literal source-file arguments and the execution checkout revision. A checkout
that moves while the ticket waits no longer stops it: the queue prepares the
same command again at the revision that is there now -- the same checks, the
same CPU preparation -- and the ticket keeps its place and its age, with the
two commits named in its history (`show`). A tree that fails those checks still
pauses the ticket with the reason; `FLEET_AUTO_REPIN=0` restores the stop of
45차 §95 (`queued checkout revision changed`). It recognizes
literal campaign ancestry and clean-tree guards, including scripts outside the
checkout. Known scripts that derive `REPO` from their parent and `cd "$REPO"` use
that source checkout. Arbitrary shell expressions, nested imports and dynamic `cd`
are not inferred; declare additional inputs or a bounded CPU check explicitly:

```json
{
  "required_paths": ["models/config.json", "build/cpu-proof.json"],
  "absent_paths": ["build/new-candidate-evidence"],
  "git": {"ancestor": "origin/main", "clean": true},
  "images": ["registry.example/model@sha256:REPLACE_WITH_DIGEST"],
  "cpu_command": ["python3", "bench/cpu_checks.py", "--test", "tests/test_candidate.py"],
  "timeout_seconds": 120
}
```

```bash
# Check immediately without reserving GPUs, or attach the same checks to a run.
bash bench/fleet.sh prepare agent --spec prepare.json -- bash probes/candidate.sh
bash bench/fleet.sh run --gpu --detach --prepare prepare.json agent 20 "candidate" -- bash probes/candidate.sh
```

CPU preparation runs once per preparation, with GPUs hidden and GPU-classified
commands refused. It does not run again at every queue poll. File/revision checks
repeat before GO; image checks run outside the fleet lock every 30 seconds while
waiting. Managed boot reservations fetch main and approve the exact deployment
candidate before queueing. The signed preparation stores each target's profile,
image/model overrides, candidate SHA and accepted main SHA; declared ancestry
checks also retain their resolved SHA. Later main commits alone cannot invalidate
that reservation. Deployment authenticates the running reservation and verifies
its unchanged clean candidate against the accepted receipt, without another fetch
or CPU suite. A changed candidate needs an explicit edit or new preparation;
there is no automatic rebase during the GPU hold. Generic CPU preparations and
older pinned controllers retain their original ref-check contract.
CPU preparation inputs are bound too. Local checks
and admission share the lock. A failed older check cannot discard a newer edit.
An explicit `edit agent -- bash probes/candidate.sh` rebinds changed source even
when the argv is identical. New reservations pause on preparation failure before
acquiring a hold. Their ticket, original age and owner survive for editing and
explicit resumption; they do not acquire restore responsibility. Existing pinned
controllers retain their original contract. Dynamic environment checks performed
by a payload after GO can still reveal a later change; preparation is not an
atomic snapshot of remote services.


## Reuse preparation and pause for revisions

```bash
MANIFEST=$(bash bench/fleet.sh prepare agent --spec prepare.json -- bash probes/candidate.sh)
bash bench/fleet.sh run --gpu --detach --prepared "$MANIFEST" agent 20 "candidate" -- bash probes/candidate.sh
bash bench/fleet.sh pause agent --reason "input needs revision"
# After revising the candidate, bind and validate its new inputs.
bash bench/fleet.sh edit agent -- bash probes/candidate.sh
bash bench/fleet.sh resume agent
```

`--prepared` accepts the same session, command, cwd, specification, source,
explicit input files, runtime, image and effective environment. A mismatch names
what changed and refuses reuse. Successful audited `cpu_checks.py` suites and
contracts reuse their passing evidence. Arbitrary CPU commands with unknown
transitive dependencies run again on fresh preparation or an ordinary edit;
explicit `--prepared` refuses to reuse them. Receipts are authenticated in the
private fleet preparation store. A successful CPU receipt first used by a boot reservation gains its
fixed deployment approval before queueing, without repeating the CPU command.
Changing SSH connection metadata does not force
revalidation; that metadata is removed from the payload environment too. Literal `env NAME=value`, `env -u` and `env -i`
prefixes select payload settings; the supervisor supplies its owned fleet and
recovery context after applying them.

An ordinary command edit attempts compatible preparation reuse and prepares
again only when needed. An explicit `--prepared` mismatch refuses the edit and
retains the original reservation. An identical command edit deliberately accepts
new source/input state after checking it. Editing a paused reservation keeps it
paused; `resume` validates it outside the queue lock and uses a revision comparison
before making it runnable. `--expect-revision N` protects pause, resume and edit
against concurrent changes.

Paused reservations preserve their ticket and original arrival time in the saved
record, outside the runnable queue. Existing pinned controllers therefore skip
them too. Other jobs can acquire GPUs while the owner fixes its command or inputs.
`show` exposes the pause reason and resume action; cancellation still stops the
owning waiter. Resuming restores the same ticket and age to queue priority.
A check of an older revision cannot pause or remove a newer edit.

Known audited CPU suites omit nonexecuted documentation and measurement outputs
from their source key. Declared inputs, executable files, symlinks, code and
runtime changes remain bound; unreviewed test changes restore the full source
scope. Eight audited campaign wrappers use the same source-aware main ancestry
check, allowing only irrelevant upstream prose/output changes. GPU revision,
build and baseline identities remain exact. Experiment results report CPU
`explanation.cache_reuse` as `cached`, `identity_match`, `changed` or `unknown`,
including changed file/component names without environment values.

## Production recovery

The five-minute idle controller and its `fleet_restore.sh` boot retired with
the vLLM overlay stack (2026-09-18). Production recovery is the ST supervisor's
own boot-start + crash-recovery loop (launchers/st-glm53.service,
launchers/st-glm53-supervisor.sh); `fleet.sh restore-needed` always answers no.
