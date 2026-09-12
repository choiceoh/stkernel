# Shared experiments for coding agents

> 살아 있는 참조 — **플릿 큐의 계약. 큐가 바뀌면 여기도 바뀐다.** 여기가 틀리면 그건 버그다.

Optimize the time from an agent's question to usable evidence. Submit once,
continue independent implementation, and read the shared result. `fleet.sh`
owns GPU admission and fast source preflight. GPU experiments run only the standard
onepass workload; separate GPU probes, sanitizer runs and custom measurements are
not admission stages. The central idle controller owns production recovery after
at least five idle minutes.
Submissions never deploy or interrupt another holder. Waiting GPU jobs are
ranked at the next free fleet boundary by downstream benefit, duration and age.

For a direct measurement, use `fleet.sh onepass SESSION NAME`: it waits for idle
serving and runs onepass once, without a boot. The checkout must describe that
running source. For changed code or knobs, use `fleet.sh pair SESSION NAME KNOBS`
or `fleet.sh chain SESSION EST NOTE -- NAME=KNOBS ...`. Each requested arm runs
onepass once; the first arm publishes its committed source only when necessary.
Both helpers reuse matching baselines and default to one baseline sample.

`run --gpu` accepts only these current canonical runners and recorded pair or
baseline jobs. Arbitrary wrappers, standalone GPU tests, sanitizer campaigns,
`chain --after`, `--legs`, and experiment prefill warmup requests are rejected
before CPU preparation or queueing. The internal `--probe` lane is retained only
for the canonical live onepass. Old `startup` request campaigns must be expressed
as pair/chain knob arms. Passive memory/proof collection can observe the same
onepass workload. Pending command edits and final execution recheck the policy;
already-running older controllers retain their accepted payloads.
Bare `request`/`wait` and unvalidated `adopt` cannot create new GPU holds; the
registered supervisor owns admission for every new GPU command.

The queue has two GPU lanes. A boot, a pair, a chain and a live onepass take the
fleet: four Sparks, one holder. An ST check that needs **one** GPU
(`probes/run_engine_check.sh`, or `run_engine_probe.sh` without `--distributed`)
takes the single-GPU lane instead: the 5050 on ost-97x (`FLEET_SINGLE_GPU_HOST`;
set it empty to turn the lane off; the controller's `~/.ssh/config` names the
alias's address, user and port -- the box is a Windows machine on the tailnet,
so that means sshd inside WSL2), with its own holder (`holder-single`) and its
own evidence (that host's GPU process list; unreachable is not free). The lanes
never block each other -- a check behind a queued boot runs now, and a boot
behind a queued check runs now. The supervisor passes `ST_PROBE_HOST` to the
runner, which rsyncs `engine/` and `probes/` to that host, runs the container
there and takes no fleet lease; a verdict from there is that card's (sm_120), not
the fleet's. `run --gpu --fleet` keeps a one-GPU check on the Sparks, `status`
shows the lane beside the fleet, and `kick [--force] single` clears its holder.

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
  command: ["python3", "bench/cpu_checks.py", "--test", "tests/test_onepass_deploy.py"],
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

For a reservation created by the current `fleet.sh run --gpu` (including pair
and chain wrappers), inspect and revise it before GO:

```bash
bash bench/fleet.sh edit fusion
bash bench/fleet.sh edit fusion --expect-revision 1 --est 20 --note "updated cells" \
  --cwd /home/choiceoh/stkernel -- bash bench/pair.sh revised "VLLM_TEST=1"
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

`plan` connects the CPU checks, optional CPU compilation, and the eventual GPU
pair. Manifests stay outside the checkout; every stage uses the same committed
revision, external inputs and runtime identifiers.

```json
{
  "hypothesis": "Reduce 2K prefill TTFT with all onepass quality gates intact",
  "knobs": {"VLLM_GLM53_KDA_ONEPASS": "1"},
  "context": {
    "image": "sha256:IMMUTABLE_64_CHARACTER_LOCAL_IMAGE_ID",
    "model": "IMMUTABLE_MODEL_REVISION",
    "hardware": "PINNED_GPU_AND_DRIVER_IDENTITIES"
  },
  "objective": {"metric": "prefill_ttft", "ctx": 2000},
  "workload": {"ctx": [2000, 32000, 128000]},
  "cpu_suites": ["logic"],
  "resources": {
    "nodes": ["local", "choiceoh@10.10.10.1", "choiceoh@10.10.10.3", "choiceoh@10.10.10.4"],
    "disk_path": "/home/choiceoh", "disk_mb": 4096, "node_memory_mb": 4096
  },
  "estimate_min": 15
}
```

```bash
bash bench/fleet.sh plan prefill /tmp/prefill-plan.json --base origin/main
# Preview only; pending-stage dependencies deliberately cannot be submitted.
bash bench/fleet.sh plan prefill /tmp/prefill-plan.json --prepare-only
# Runs CPU checks/preparation now, without requiring a deployment or a GPU hold.
# After deploying that revision through the fleet, submit the returned gpu.json:
bash bench/fleet.sh submit prefill /absolute/returned/plan/directory/gpu.json
# Or, for an already deployed revision, submit all stages in one call:
bash bench/fleet.sh plan prefill /tmp/prefill-plan.json --submit
```

`--base` uses the committed diff to suggest checks. Edits confined to the three
audited helpers below select their separate contracts plus sensitivity; any
changed non-helper AST node, unknown path or modified test audit falls back to
the conservative suite selection. Changes restricted to bench
and fleet tests select `fleet`; other changes select `logic`, with `startup`
added for launchers/profiles. This is a conservative convenience, not a
dependency coverage proof. Override with `cpu_suites` and/or `cpu_tests` when
the changed contract needs additional checks. A failed submission preserves
the plan path and all previously submitted IDs; it does not lose running work.

In a plan, `cpu_suites: ["logic", "startup"]` expands to independent required
`checks-core`, `checks-fleet` and `checks-startup` jobs. With one suite/test group,
its stage remains `checks`; contracts retain `checks-math`, `checks-layout` and
`checks-dispatch`. The GPU stage requires every check and preparation stage.
Fleet stages reserve up to two CPU slots by default (respecting a one-slot pool
policy); set `cpu_jobs` to 1..8 to choose explicitly. The CPU stages are submitted
as one batch. Their IDs and resolved manifests are saved and their workers start
before GPU deployment attestation. While deployment checks are still pending,
agents can read `plans/<plan-id>/plan.json` and use the recorded CPU IDs with
`result`, `wait`, or their session inbox. Explicit preparation dependencies and
all GPU prerequisites still apply. If deployment attestation fails, CPU work
continues and its IDs remain available in the saved plan and error response.

Dependency completion is checked every 50 ms using batched, indexed ID/state
reads, so short checks and preparation chains avoid a one-second sleep at each
edge. Full pinned payloads are not decoded on each poll. Recovery of unfinished
workers remains limited to one pass per second per waiting worker; completed
workers are skipped. Incomplete evidence is still refreshed before a dependent
job is blocked, and retirement ends the dependency wait before execution.

For multiple goals on the **same knobs/image/configuration**, replace
`objective`/`workload` with `evaluations`, an array of up to six objects of that
shape. Identical workloads produce one record used by multiple objectives;
different workloads run sequentially on the same boot. Before each workload
the source/artifact snapshot and serving boot are checked. A failed workload
stops the rest; the normal fleet policy decides one final production restore.
Different serving configurations still need separate submissions. Independent
baseline samples still require separate boots.

When a pair finishes definitively or is retired, it releases its internal
baseline subscription. An unstarted reservation with no remaining subscribers or
dependents is retired and removed from the queue. Shared demand, explicit
operator subscriptions, incomplete evidence, started jobs and matching live
holders are preserved. This does not stop an active boot or discard baseline
results. Baseline workers also check for old orphan reservations before preflight.

Separate agents' ready pair requests also share one serving boot when their
committed revision, deployed snapshot, configuration, environment, runtime,
inputs, prepared artifact hashes, resource requirements and port match exactly.
Each request must first pass its own prerequisites, preflight and baseline gate.
The first ready request collects peers for at most 0.5 seconds outside the GPU
hold; the group stays open while queued and seals atomically at GO. A group has
at most eight requests and six distinct workloads. Late/incompatible requests
and explicit repeats run separately. The union of workloads runs once, with each
request judged against its own original objectives. `execution_job` and the
runner-owned measurement binding point to the actual producer record; records
are never relabeled as new independent samples. Admission rechecks every member,
and result publication rechecks each consumer's source and artifacts. Execution
or restore failure reaches all consumers. This does not interrupt a live boot.

| Objective | Direct measurement and interpretation |
| --- | --- |
| `decode_steps` | onepass decode-window median step/s; existing default |
| `decode_tokens` | pooled `(completion_tokens - 1) / decode_s` from fixed-length client requests; requires `fixed_decode_tokens`, `fixed_decode_reps`, and `require_exclusive: true` |
| `prefill_ttft` plus `ctx` | first content-chunk latency at that context; lower is better; compile-cold records do not fill the planned steady-compile baseline and cannot mix with warm-compile evidence |
| `quality` | all declared onepass retrieval, corruption, decode-presence and lane-proof gates pass; no performance verdict or noise-floor claim |

All workload defaults and record metadata come from `measurement_contract.py`,
which both the actual onepass producer and shared baseline reader use. Older
harness/workload records remain incompatible. A prefill plan may need one extra
baseline boot if the first sample is marked compile-cold. Compiler-cache flags
are coarse existing build markers; this does not benchmark startup cache gains.

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
inject objects into vLLM, or prove GPU numerical correctness automatically.

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

## GPU pair manifest

```json
{
  "kind": "pair",
  "revision": "FULL_40_CHARACTER_COMMIT_SHA",
  "hypothesis": "The candidate improves decode while preserving all onepass gates",
  "knobs": {"VLLM_GLM53_KDA_ONEPASS": "1"},
  "context": {
    "image": "sha256:IMMUTABLE_64_CHARACTER_LOCAL_IMAGE_ID",
    "model": "IMMUTABLE_MODEL_REVISION",
    "hardware": "srv1-srv4 GPU identities and driver version"
  },
  "inputs": ["/absolute/path/to/model-config-or-weight-manifest.json"],
  "depends_on": ["CPU_EXPERIMENT_ID"],
  "estimate_min": 15
}
```

Deploy the requested revision through the existing fleet flow first. Pair
submission verifies the deployed manifest's `source_commit`, its overlay stamp,
overlay files and immutable local image ID. It rechecks them at GO and after the
run. Changes during the queue fail before a boot. The launcher pins `IMAGE` to
the declared ID. The manifest's model/hardware identifiers are caller-declared;
include external fixtures, model metadata and immutable weight manifests in
`inputs`. The runner hashes those files; it does not rehash hundreds of GB of
model weights or independently attest every hardware identifier.

Pairs default to `"baseline_policy": "minimal"`. Once prerequisites and
preflight pass, the worker reserves a shared defaults job only if the context
has no usable baseline for a requested workload. One baseline supports the
initial comparison; subsequent compatible candidates reuse it without a new
defaults boot. A thin noise floor does not automatically trigger more samples.
Compatible candidates join that reservation and wait outside the GPU queue.
The defaults job measures only workloads still missing samples on each separate boot,
then releases all waiting candidates. A new candidate therefore needs two
measurement boots (one baseline, one candidate), instead of four. Later
candidates need only their own boot while the context remains compatible.

A minimal result uses `evidence: "gpu-pair-screen"`, `comparison_complete: true`
and `promotion_ready: false`. `state: succeeded` means the comparison completed;
the unchanged statistical judge may still report `incomplete` or `inconclusive`.
Observed deltas are available to choose the next experiment without claiming a
confirmed speedup. Result polling never acquires another baseline.

Set `"baseline_policy": "confirm"` explicitly for confirmation. It requires
three independent baselines for a performance workload and one for quality.
Existing compatible samples count, so confirmation after a one-baseline screen
adds only the two missing defaults boots. Confirmation runs its candidate again.
Minimal and confirmation reservations remain distinct, but share compatible
records. Already submitted jobs without a policy keep their previous contract.
The shell `fleet.sh pair` path also defaults to one baseline; `PAIR_FLOOR_N=3`
explicitly asks it to replenish the independent floor one sample per call.

The reservation key includes revision, deployed sources, image, model/hardware
context, workload/environment, external inputs and ledger location. Failed
defaults block their consumers. Repeat the candidate with an explicit reason
to retry a failed shared reservation. Container ID plus StartedAt supplies
`boot_id`: repeated onepass runs on the same boot count once. Historical rows
without boot identity cannot fill the shared reservation.

Open baseline reservations now combine different objectives and workload
subsets for the same attested serving configuration. Quality demands need one
usable boot; confirmation performance demands need three, with the warm-compile rule
retained for TTFT. A reservation accumulates at most six evaluation requirements
and eight candidate dependencies. It collects for 0.5 seconds outside the GPU
hold and seals when execution starts. Running reservations accept only demands
already covered by their measurement plan, never additional workloads. Explicit
repeats remain separate. Each candidate checks its own baseline requirements
again before running; sharing does not turn repeated measurements into new boots.

Each managed pair measures its declared onepass workloads on one attested serving
boot and releases the fleet after the group. The `pair.sh` wrapper remains
available for direct callers. Other agents can enqueue their own pairs between
submissions; long multi-candidate experiments should be separate submissions.
Candidate/baseline evidence stays together. There is no GPU preemption and no
separate GPU probe stage.

Pair results require fresh onepass records bearing the experiment ID, the
requested knobs, a matching revision/build/workload/runtime, complete quality
and corruption gates, known serving proof for every enabled non-default knob,
and a usable baseline noise floor on the same build. A `0/0` proof with unknown
enabled knobs, missing quality data, nonfinite timing, or an incompatible
baseline cannot become a successful result. A measured slowdown can be a valid
result; an effect within the noise floor remains inconclusive.

`result ID` rejudges an incomplete pair if a later matching baseline becomes
available, without rerunning the candidate. Completion of a shared baseline
also publishes those updated results without needing an agent to poll them.
Existing `chain.sh` also publishes
each candidate's verdict before the next arm and stops dependent arms after
execution or proof/quality failure. Failed pairs/chains release the fleet;
production recovery belongs to the central five-minute idle controller.

## Onepass-only GPU submissions

GPU submissions use `kind: "pair"` with literal `VLLM_*` knobs. A custom `command`
is refused, so the worker always executes the standard onepass pair runner.
`kind: "probe"` and nonempty `probe_contract` are no longer accepted. Historical
queued probe requests are marked blocked before creating a worker checkout or
acquiring GPUs; retrying one requires a new pair manifest. Completed historical
reports remain readable.

Use an optional relevant CPU request in `depends_on`, then submit the candidate
pair directly. Onepass supplies serving quality, corruption, proof and performance
evidence. Compatible saved baselines remain reusable; unifying the workload does
not request another baseline, another boot, or another measurement. Additional
GPU microbenchmarks, prechecks, sanitizers and post-measurement sweeps are excluded
from this path.

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

The final holder uses `bench/fleet_restore.sh`: clean approved main, public
port 8000, profile defaults, warmup, no measurement leg. Candidate environment
overrides are removed. An already healthy public defaults arm of that approved
build and approved immutable image avoids a duplicate boot. The public bind is
read from the launcher's decoded static command, without executing shell text.
`FLEET_PRODUCTION_REPO` selects the production
checkout; it defaults to `/home/choiceoh/stkernel`. Restore failures return
nonzero and retain `restore-debt.json`; a subsequent supervised boot can recover
it before probes are admitted. SIGKILL/host loss cannot run a process's cleanup:
the debt remains visible for recovery; this is not a host-level watchdog.
An operator can put the path of a dedicated approved-main checkout in
`$FLEET_DIR/production-repo`; this separates restoration from a common checkout that
contains unmerged experiment work. An explicit `FLEET_PRODUCTION_REPO` wins.
The path may contain spaces and need not end with a newline. A clean checkout
ahead of main is detached at approved main, preserving its candidate branch;
dirty work is refused.

Custom boot scripts must accept stopped serving. Before stopping anything, use:

```bash
python3 "${FLEET_RUNNER_REPO:-$REPO}/bench/fleet_entry.py" idle "$out/before-metrics.txt"
```

This allows an absent/stopped container while requiring health and both zero
request counters for a live one, including an isolated experiment port. Validate
the immutable image with `docker image inspect`; an existing stopped container
can also attest its image. Guard a standalone restore fallback with
`[[ ${FLEET_RESTORE_MANAGED:-0} != 1 ]]`. The tracked CTA/reuse wrappers demonstrate
this contract; preflight rejects their old unconditional `touched` cleanup
pattern before queuing. Arbitrary shell code is not exhaustively linted.

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
literal source-file arguments and the execution checkout revision. It recognizes
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

## Quick experiment admission and stable release recovery

New boot requests run a short CPU admission check before joining the GPU queue:
shell syntax, composition of the selected profile, and syntax/manifest contracts
for its overlay files. The check has a 30-second execution limit, imports no
model kernels, runs no full logic/Fleet suite and starts no chat-check container.
Its receipt binds clean source, the selected profile, checker and required tools;
queue rechecks and deployment consume that receipt without repeating the checks.
Admission is a syntax/deployment check, not numerical or release evidence. The
experiment's GPU correctness and onepass validation remain responsible for actual
kernel outputs, communication and performance. Explicitly declared experiment CPU
prerequisites still run; they are not silently dropped.

Literal campaign checkout and image/model overrides are bound to preparation;
dynamic deployment scripts declare `deployment_targets` with `repo`, `profile`
and optional `image`/`model`. Direct deployment still requires the complete CPU
release gate: full logic, runtime guard audit, GLM overlay synchronization and
CPU-only chat release checks. To request it separately:

```bash
python3 bench/fleet_validation.py validate --repo "$PWD" --profile glm53 --level release
```

Recovery reuses an existing release-validated approved main checkout, even after
main advances. A private per-production-repository pointer keeps that choice
stable. Older evidence is verified by its original approved validator; quick
admission evidence can never authorize recovery. Source changes, rewritten main,
changed release dependencies and altered receipts still reject reuse. Only the
first setup without a valid recovery must acquire a complete release receipt.
Refresh the recovery version explicitly, outside a GPU hold:

```bash
python3 bench/fleet_validation.py prepare-recovery --repo "$PRODUCTION_REPO" --refresh-recovery
```

The pointer changes only after the new release check succeeds. A failed refresh
leaves the prior validated recovery intact. The fleet host can select its CPU
Python environment with a private `FLEET_VALIDATION_STORE/python` file containing
an absolute interpreter path. Full release evidence binds its installed packages,
Python startup inputs, image and tokenizer/config files. Admission uses isolated
stdlib Python and does not scan installed ML packages or tokenizer data.

A receipt miss during a GPU hold refuses without starting CPU validation.
Sessions prepare only their candidate; they no longer acquire a recovery receipt
or own a restoration obligation. Successful, failed and cancelled sessions clean
up their temporary resources and release immediately. `restore-needed` always
returns no. Pair/chain baseline measurements remain, but cleanup RESTORE/RECOVER
arms and automatic restarts of paused original containers are forbidden.

## Automatic recovery after five idle minutes

`fleet-idle-recovery.timer` checks every 15 seconds. Only its controller may run
`fleet_restore.sh`, under a process-bound fleet lease, after at least 300 seconds
of proven idle time. Enqueue, acquisition, release, cancellation and detected
serving traffic reset the monotonic clock. A host reboot or unknown Docker/GPU
state restarts observation. The controller rechecks requests, GPU processes and
runnable reservations immediately before claiming the hold. Dead/paused tickets
are excluded; probes waiting for absent serving can resume after recovery.

Resident GPU processes belonging to other services do not block GLM recovery.
The node check requires a readable driver process inventory, not exclusive GPU
ownership. GLM requests, unmanaged experiments, holders and runnable tickets
still block the idle window. Before boot, the launcher removes only the old GLM
containers, releases file caches and sizes memory from the smallest node's
`MemFree`, retaining the boot allowance and safety margin. It leaves unrelated
processes running and aborts on a failed memory probe or an explicit GMU above
the measured ceiling. The memory helper comes from the approved launcher's own
checkout, so recovery cannot silently use an older central copy.

Already healthy approved defaults need no reboot. Recovery consumes the stable
release receipt and never starts a full CPU suite. Missing evidence defers recovery;
prime or refresh it explicitly using the command above. Failures retry only after
another quiet window. `fleet.sh status` shows the controller state and reason.

Install the user service on srv2 (the repository stays at an approved clean commit):

```bash
install -m 0644 launchers/fleet-idle-recovery.{service,timer} ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now fleet-idle-recovery.timer
```

During migration, already running older controllers must also defer their final
restore. Their restore entrypoint can be replaced with a recorded no-boot bridge
after checking the current holder/queue and preserving the original script.
Payloads and measured baseline arms are not interrupted or rewritten. New runner
snapshots use the central policy directly.
