# Shared experiments for coding agents

Optimize the time from an agent's question to usable evidence. Submit once,
continue independent implementation, and read the shared result. `fleet.sh`
still owns GPU admission, preflight, short-probe yielding and production restore.
Submissions never deploy or interrupt another holder. Waiting GPU jobs are
ranked at the next free fleet boundary by downstream benefit, duration and age.

Plans now batch their independent CPU stages, publish reusable evidence before
creating another checkout on a cache hit, and keep core/fleet/startup results
separate. A consumer's declared dependencies still decide when it can execute.

## Agent workflow

Run the commands on the fleet head, from a **committed, clean checkout**. Each
experiment needing execution gets a detached private checkout of that commit, so the agent
can immediately continue editing its original checkout. Build outputs from CPU
checks are isolated too. Put submission manifests and reports outside the repo.

1. State the hypothesis and choose the relevant CPU suite.
2. Submit the CPU check; another agent's identical request joins it or reads its
   saved result. Do independent work while it runs.
3. Submit one GPU candidate with the CPU experiment ID in `depends_on`.
   The worker waits outside the GPU queue until prerequisites succeed. A failed,
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
  hypothesis: "The changed kernel preserves CPU math, layout and dispatch contracts",
  command: ["python3", "bench/cpu_checks.py", "--suite", "logic"],
  timeout_s: 900
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
to promote incomplete CPU/probe evidence.

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

Once a pair's prerequisites and preflight pass, the worker reserves a shared
defaults job if the context has fewer than three independent baseline samples per performance workload (one for a
quality-only workload, which makes no speed claim).
Compatible candidates join that reservation and wait outside the GPU queue.
The defaults job measures only workloads still missing samples on each separate boot,
then releases all waiting candidates. There is no candidate/default flip for
every member of a new campaign just to build its noise floor. A single new
candidate still pays for that three-sample floor; sharing primarily benefits
multiple candidates and avoids the former first-pair incomplete result.

The reservation key includes revision, deployed sources, image, model/hardware
context, workload/environment, external inputs and ledger location. Failed
defaults block their consumers. Repeat the candidate with an explicit reason
to retry a failed shared reservation. Container ID plus StartedAt supplies
`boot_id`: repeated onepass runs on the same boot count once. Historical rows
without boot identity cannot fill the new shared reservation's three samples.

Open baseline reservations now combine different objectives and workload
subsets for the same attested serving configuration. Quality demands need one
usable boot; performance demands still need three, with the warm-compile rule
retained for TTFT. A reservation accumulates at most six evaluation requirements
and eight candidate dependencies. It collects for 0.5 seconds outside the GPU
hold and seals when execution starts. Running reservations accept only demands
already covered by their measurement plan, never additional workloads. Explicit
repeats remain separate. Each candidate checks its own baseline requirements
again before running; sharing does not turn repeated measurements into new boots.

Each managed pair measures its declared workloads on one attested serving boot
and makes one restore decision after the group. The legacy `pair.sh` remains
available for existing callers. Other
agents can enqueue their own pairs between submissions; long multi-candidate
experiments should be separate submissions. Candidate/baseline evidence stays
together and short probes can still yield through the existing fleet path.
There is no GPU preemption. A pair/chain's hold remains intact except for its
existing explicit short-probe yield points.

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
execution or proof/quality failure. Failed pairs/chains restore production when
the existing queue policy requires it, without spending another baseline sample.

## Probe manifest

Use `kind: "probe"`, `command: ["bash", "probes/...", "..."]`, the pinned GPU
`context`, external `inputs` and optional prerequisites. The standard probe
preflight/idle-serving rules apply. Generic probes return their log and process
exit status as `probe-log` evidence. An exit code alone stays `incomplete`.

For automatic CPU → numerical probe → pair progression, declare the numerical
contract before submission and add that probe ID to the pair's `depends_on`:

```json
{
  "kind": "probe",
  "revision": "FULL_40_CHARACTER_COMMIT_SHA",
  "hypothesis": "Strided QK normalization is bit-exact on every tested layout and regime",
  "command": ["bash", "probes/run_mk_probe.sh", "probes/qk_norm_strided_check.py"],
  "context": {
    "image": "sha256:IMMUTABLE_64_CHARACTER_LOCAL_IMAGE_ID",
    "model": "IMMUTABLE_MODEL_REVISION",
    "hardware": "PINNED_GPU_AND_DRIVER_IDENTITIES"
  },
  "depends_on": ["CPU_EXPERIMENT_ID"],
  "probe_contract": {
    "checks": {"mismatches": {"op": "eq", "value": 0}},
    "proof": ["strided_lane", "decline_guards"],
    "min_samples": 150
  },
  "estimate_min": 10
}
```

The existing QK probe now emits this report after its real comparisons and
fallback-guard checks. Other probes can call `bench/probe_report.py`'s
`write_report(metrics, proof, samples, device)` after completing their checks.
`run_mk_probe.sh` forwards the runner's fresh challenge and mounts a private
report directory into its container. Structured probes pin the immutable image
ID; custom wrappers must honor `IMAGE` and forward the report fields themselves.
Missing, stale, mismatched, nonfinite or insufficient reports never unlock the
next job. Threshold failures and unknown lane proof fail the probe. A successful
`gpu-probe` establishes only its declared numerical contract; the pair still
checks serving quality, proof, throughput and a matching noise floor.

## Queue policy and CPU content reuse

`bash bench/fleet.sh priority` explains the current ranking. At a free fleet
boundary, a ready job's score is `(1 + pending transitive dependents) /
estimate_min + wait_seconds / 1800`. At 30 minutes waiting, oldest-first takes
precedence, so a stream of tiny jobs cannot indefinitely starve a long one.
Explicit `front` and a chosen yielded probe retain their order; a yielded holder
resumes before other work. Ineligible probes wait for idle serving. Legacy jobs
participate with zero known dependents. Ranking never interrupts a live holder;
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

`stats` separates CPU/pair/probe/baseline counts, shared/reused requests, and p50/p95 time
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

The final holder uses `bench/fleet_restore.sh`: clean approved main, public
port 8000, profile defaults, warmup, no measurement leg. Candidate environment
overrides are removed. An already healthy public defaults arm of that approved
build avoids a duplicate boot. `FLEET_PRODUCTION_REPO` selects the production
checkout; it defaults to `/home/choiceoh/stkernel`. Restore failures return
nonzero and retain `restore-debt.json`; a subsequent supervised boot can recover
it before probes are admitted. SIGKILL/host loss cannot run a process's cleanup:
the debt remains visible for recovery; this is not a host-level watchdog.

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

## Share startup controls across a campaign

Commit/deploy one build containing the candidate toggles, then use:

```bash
REPO="$PWD" bash bench/fleet.sh startup startup-agent bench/startup-campaign.example.json 45
```

The example runs `PRIME, BASE1, FASTIOR1, SHAKEYR1, SHAKEYR2, FASTIOR2, BASE2`:
seven boots instead of two independent five-boot trials. Each candidate still
has two boots. Three candidates use nine instead of fifteen. These are boot
counts, not measured wall-clock savings. Both shared cache directories stay
fixed; every arm explicitly sets the same knob keys. Malformed, duplicate,
cache-off or changed-cache campaigns fail validation.

Every arm retains four-node cache receipts, canonical onepass response/quality
evidence and a distinct matching boot. Pack IO/key campaigns retain their GPU
checks on PRIME. The campaign pins source/profile/deployed manifest/workload,
rejects changed identities and compares its two controls for drift (10% default,
at most 25%). `campaign-result.json` contains health timings and paired candidate
summaries. It is exploration evidence with `promotion_ready: false`; drift
makes the result incomplete. Final promotion still needs the relevant direct
consumer metric and independent matched validation. Older unrelated builds'
baselines are never reused.

CPU validation of these changes uses the regular fleet suite, startup campaign
and receipt tests. Run `tests/test_boot_supervisor_linux.py` explicitly on Linux
for real shell admission, cancellation and handoff with fake system commands;
it never accesses GPUs, SSH or production containers.
