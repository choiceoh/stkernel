# Shared experiments for coding agents

Optimize the time from an agent's question to usable evidence. Submit once,
continue independent implementation, and read the shared result. `fleet.sh`
still owns GPU admission, preflight, short-probe yielding and production restore.
Submissions never deploy, change the queue order, or interrupt another holder.

## Agent workflow

Run the commands on the fleet head, from a **committed, clean checkout**. Each
new experiment gets a detached private checkout of that commit, so the agent
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
| `fleet` | Concurrent submissions, duplicate consumers, private source snapshots, CPU prerequisites, failures, timeouts, evidence compatibility and shell failure propagation |
| `startup` | Launcher/worker startup, file attestation, memory preflight and reclamation using local fakes |

Select by the changed contract. A scheduling change normally needs `fleet`;
a kernel math/layout change needs `logic`; a launcher change needs `startup`.
The repository's deployment gate still runs `logic`. Avoid running both `fleet`
and `logic` for the same revision unless investigating a new failure: `logic`
already includes `fleet`. Source checks and mocked dispatch tests cannot prove
device numerics, graph replay, race freedom, serving quality or throughput.

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

Each pair is one candidate plus the baseline/restore required by `pair.sh`. Other
agents can enqueue their own pairs between submissions; long multi-candidate
experiments should be separate submissions. Candidate/baseline evidence stays
together and short probes can still yield through the existing fleet path.
This adds no new priority scheduler or hard GPU preemption.

Pair results require fresh onepass records bearing the experiment ID, the
requested knobs, a matching revision/build/workload/runtime, complete quality
and corruption gates, known serving proof for every enabled non-default knob,
and a usable baseline noise floor on the same build. A `0/0` proof with unknown
enabled knobs, missing quality data, nonfinite timing, or an incompatible
baseline cannot become a successful result. A measured slowdown can be a valid
result; an effect within the noise floor remains inconclusive.

`result ID` rejudges an incomplete pair if a later matching baseline becomes
available, without rerunning the candidate. Existing `chain.sh` also publishes
each candidate's verdict before the next arm and stops dependent arms after
execution or proof/quality failure. Failed pairs/chains restore production when
the existing queue policy requires it, without spending another baseline sample.

## Probe manifest

Use `kind: "probe"`, `command: ["bash", "probes/...", "..."]`, the pinned GPU
`context`, external `inputs` and optional prerequisites. The standard probe
preflight/idle-serving rules apply. Generic probes return their log and process
exit status as `probe-log` evidence. A successful process exit is deliberately
`incomplete` until the agent reviews the probe's numerical and timing evidence;
it does not automatically unlock a dependent production experiment.

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

`stats` separates CPU/pair/probe counts, shared/reused requests, and p50/p95 time
to start and to a successful result. Queue time includes prerequisite waiting.
An incomplete/failed result is never counted as a successful fast result. The
initial implementation measures these timings; it makes no speedup claim until
matched live runs establish one.
