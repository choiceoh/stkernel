#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Durable, non-blocking experiment submissions on the fleet head.

The existing fleet.sh owns GPU admission. This module owns request identity,
CPU prerequisites, subscribers and results; it never deploys or steals a hold.
See bench/EXPERIMENTS.md for the manifest and agent workflow.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import sqlite3
import statistics
import subprocess
import sys
import time
import uuid

HERE = Path(__file__).resolve().parent
TERMINAL = {"succeeded", "failed", "blocked", "incomplete", "interrupted", "retired"}
RESERVED = {"HOME", "PATH", "PYTHONPATH", "BASH_ENV", "ENV", "REPO", "LOGD"}
BASE_ENV = ("HOME", "PATH", "USER", "LOGNAME", "LANG", "LC_ALL", "SSH_AUTH_SOCK", "TMPDIR")
ONEPASS_ONLY = ("GPU work is onepass-only: it runs through fleet.sh run --gpu, "
                "the ST bracket lanes, or the live onepass; it is not a stored experiment")
PAIR_RETIRED = "the pair lane retired with the vLLM overlay stack (2026-09-18); only CPU jobs and the ST lanes remain"


class RetiredJob(ValueError):
    """A request was withdrawn before execution; never revive it implicitly."""


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True,
                                   stderr=subprocess.PIPE, timeout=20).strip()


def snapshot(repo, spec):
    revision = git(repo, "rev-parse", "HEAD")
    if revision != spec["revision"]:
        raise ValueError("checkout revision changed; commit and submit the intended revision")
    if git(repo, "status", "--porcelain", "--untracked-files=normal"):
        raise ValueError("experiment checkout must be clean (including untracked inputs)")
    return {"revision": revision, "inputs": {p: digest(p) for p in spec["inputs"]},
            "host": platform.node(), "platform": platform.platform(),
            "python": sys.version, "runner": digest(__file__)}


def normalize(raw, repo):
    if not isinstance(raw, dict):
        raise ValueError("manifest must be a JSON object")
    kind = raw.get("kind")
    if kind == "probe":
        raise ValueError(ONEPASS_ONLY)
    if kind in {"pair", "baseline"}:
        raise ValueError(PAIR_RETIRED)
    allowed = {"kind", "revision", "hypothesis", "command", "inputs", "context",
               "env", "depends_on", "estimate_min", "timeout_s",
               "resources", "outputs"}
    if set(raw) - allowed:
        raise ValueError("unknown manifest fields: " + ", ".join(sorted(set(raw) - allowed)))
    if kind != "cpu":
        raise ValueError("kind must be cpu; GPU work runs through fleet.sh run --gpu, "
                         "the ST bracket lanes, or the live onepass")
    if not isinstance(raw.get("hypothesis"), str) or not raw["hypothesis"].strip():
        raise ValueError("hypothesis must explain what this experiment decides")
    revision = raw.get("revision", "")
    if not isinstance(revision, str) or not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("revision must be the full committed SHA, not a moving branch")
    env = raw.get("env", {})
    if not isinstance(env, dict) or any(
        not re.fullmatch(r"[A-Z_][A-Z0-9_]*", k) or not isinstance(v, str)
        or k in RESERVED or k.startswith(("FLEET_", "ONEPASS_"))
        for k, v in env.items()
    ):
        raise ValueError("env contains invalid values or runner control overrides")
    command = raw.get("command", [])
    if not isinstance(command, list) or any(not isinstance(v, str) or not v or "\0" in v for v in command):
        raise ValueError("command must be an argv array")
    from cpu_checks import canonical
    command = canonical(command, repo)
    if not command:
        raise ValueError("cpu takes command, not knobs")
    inputs = raw.get("inputs", [])
    if not isinstance(inputs, list) or any(not isinstance(p, str) for p in inputs):
        raise ValueError("inputs must list external input files to hash")
    inputs = sorted({str((repo / p).resolve()) for p in inputs})
    context = raw.get("context", {})
    if not isinstance(context, dict) or any(not isinstance(v, str) for v in context.values()):
        raise ValueError("context must map names to immutable environment identifiers")
    deps = raw.get("depends_on", [])
    if not isinstance(deps, list) or any(not isinstance(d, str) for d in deps):
        raise ValueError("depends_on must list existing experiment IDs")
    estimate = raw.get("estimate_min", 15)
    if type(estimate) is not int or not 1 <= estimate <= 720:
        raise ValueError("estimate_min must be an integer between 1 and 720")
    timeout = raw.get("timeout_s", 900)
    if type(timeout) is not int or not 1 <= timeout <= 86400:
        raise ValueError("timeout_s must be an integer between 1 and 86400 (CPU commands only)")
    from experiment_resources import normalize as resource_spec
    outputs = raw.get('outputs', [])
    if not isinstance(outputs, list) or any(not isinstance(p, str) for p in outputs):
        raise ValueError('outputs must be a list of CPU build output paths')
    from prepared_artifacts import output_path
    for output in outputs:
        output_path(repo, output)
    return dict(kind=kind, revision=revision, hypothesis=raw["hypothesis"], command=command,
                env=env, inputs=inputs, context=context,
                depends_on=sorted(set(deps)), estimate_min=estimate, timeout_s=timeout,
                resources=resource_spec(raw.get('resources')),
                outputs=sorted(set(outputs)))


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "experiments.sqlite3", timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, payload TEXT NOT NULL,
                state TEXT NOT NULL, created REAL NOT NULL, started REAL, finished REAL,
                result TEXT, worker_pid INTEGER, repeat_reason TEXT);
            CREATE INDEX IF NOT EXISTS fingerprints ON jobs(fingerprint, created);
            CREATE TABLE IF NOT EXISTS subscribers (
                job TEXT NOT NULL, session TEXT NOT NULL, attached REAL NOT NULL,
                PRIMARY KEY(job, session));
            CREATE INDEX IF NOT EXISTS subscriber_sessions ON subscribers(session, job);
            CREATE TABLE IF NOT EXISTS events (
                cursor INTEGER PRIMARY KEY AUTOINCREMENT, job TEXT NOT NULL,
                at REAL NOT NULL, kind TEXT NOT NULL, data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS dependencies (
                job TEXT NOT NULL, dependency TEXT NOT NULL, kind TEXT NOT NULL,
                PRIMARY KEY(job, dependency, kind));
            CREATE TABLE IF NOT EXISTS cpu_cache (
                key TEXT PRIMARY KEY, job TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS cpu_leases (
                job TEXT PRIMARY KEY, pid INTEGER NOT NULL, slots INTEGER NOT NULL, memory_mb INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS cpu_waiters (
                ticket INTEGER PRIMARY KEY AUTOINCREMENT, job TEXT NOT NULL UNIQUE,
                pid INTEGER NOT NULL, slots INTEGER NOT NULL, memory_mb INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS deliveries (
                session TEXT NOT NULL, cursor INTEGER NOT NULL, delivered REAL NOT NULL, acknowledged REAL,
                PRIMARY KEY(session, cursor));
            CREATE TABLE IF NOT EXISTS execution_groups (
                leader TEXT PRIMARY KEY, signature TEXT NOT NULL, sealed INTEGER NOT NULL,
                workloads TEXT NOT NULL, created REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS group_members (job TEXT PRIMARY KEY, leader TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS open_groups ON execution_groups(signature,sealed);
            CREATE TABLE IF NOT EXISTS withdrawals (
                job TEXT NOT NULL, session TEXT NOT NULL, replacement TEXT NOT NULL, reason TEXT NOT NULL,
                PRIMARY KEY(job,session));
            CREATE TABLE IF NOT EXISTS phase_open (
                job TEXT NOT NULL, phase TEXT NOT NULL, started REAL NOT NULL, PRIMARY KEY(job,phase));
            CREATE TABLE IF NOT EXISTS timings (
                job TEXT NOT NULL, phase TEXT NOT NULL, seconds REAL NOT NULL, at REAL NOT NULL, ok INTEGER NOT NULL);
            CREATE INDEX IF NOT EXISTS timing_phases ON timings(phase,at);
            CREATE TABLE IF NOT EXISTS retry_attempts (
                source TEXT NOT NULL, attempt TEXT NOT NULL, session TEXT NOT NULL,
                reason TEXT NOT NULL, created REAL NOT NULL, PRIMARY KEY(source,attempt,session));
            CREATE INDEX IF NOT EXISTS retry_attempt_ids ON retry_attempts(attempt);
        """)

    def event(self, job, kind, data):
        self.db.execute("INSERT INTO events(job,at,kind,data) VALUES(?,?,?,?)",
                        (job, time.time(), kind, encoded(data)))

    def get(self, job):
        row = self.db.execute("SELECT * FROM jobs WHERE id=?", (job,)).fetchone()
        if row is None:
            raise ValueError("unknown experiment: " + job)
        result = dict(row)
        result["payload"] = json.loads(result["payload"])
        result["result"] = json.loads(result["result"]) if result["result"] else None
        result["log"] = str(self.root / job / "run.log")
        return result

    def list_jobs(self, session=None, active=False, limit=100):
        if not 1 <= limit <= 1000:
            raise ValueError('jobs limit must be 1..1000')
        query = 'SELECT j.id,j.state,j.created,j.started,j.finished FROM jobs j'
        conditions, parameters = [], []
        if session is not None:
            query += ' JOIN subscribers s ON s.job=j.id'
            conditions += ['s.session=?', 'NOT EXISTS (SELECT 1 FROM withdrawals w '
                           'WHERE w.job=s.job AND w.session=s.session)']
            parameters.append(session)
        if active:
            conditions.append('j.state NOT IN (' + ','.join('?' for _ in TERMINAL) + ')')
            parameters.extend(sorted(TERMINAL))
        if conditions:
            query += ' WHERE ' + ' AND '.join(conditions)
        query += ' ORDER BY j.created DESC LIMIT ?'
        return [dict(row) for row in self.db.execute(query, [*parameters, limit])]

    @contextmanager
    def transaction(self):
        # A state transition inside retirement/group admission belongs to that
        # caller's transaction; it must not commit its partially updated state.
        if self.db.in_transaction:
            yield
        else:
            with self.db:
                self.db.execute('BEGIN IMMEDIATE')
                yield

    def state(self, job, state, result=None):
        with self.transaction():
            previous = self.get(job)
            if previous['state'] == 'retired' and state != 'retired':
                raise RetiredJob(job)
            from experiment_metrics import transition
            transition(self,previous,state)
            self.db.execute("UPDATE jobs SET state=?, result=COALESCE(?,result), "
                            "started=CASE WHEN ?='running' THEN COALESCE(started,?) ELSE started END, "
                            "finished=CASE WHEN ? THEN ? ELSE finished END WHERE id=?",
                            (state, encoded(result) if result is not None else None, state,
                             time.time(), state in TERMINAL, time.time(), job))
            self.event(job, state, result or {})

    def is_retry(self, job):
        return self.db.execute('SELECT 1 FROM retry_attempts WHERE attempt=? LIMIT 1', (job,)).fetchone() is not None

    def submit(self, session, payload, repeat=None, *, retry_failed=False):
        identity = {k: v for k, v in payload.items() if k != "repo"}
        identity["spec"] = {k: v for k, v in payload["spec"].items()
                            if k not in {"hypothesis", "estimate_min"}}
        identity["environment"] = {k: v for k, v in payload["environment"].items() if k != "SSH_AUTH_SOCK"}
        if 'prepared_artifacts' in identity:
            identity['prepared_artifacts'] = [{k:v for k,v in r.items() if k != 'path'} for r in identity['prepared_artifacts']]
        fingerprint = hashlib.sha256(encoded(identity).encode()).hexdigest()
        with self.transaction():
            for dep in payload["spec"]["depends_on"]:
                prerequisite = self.get(dep)  # existing IDs only: no dependency cycles
                if prerequisite["payload"]["spec"]["revision"] != payload["spec"]["revision"]:
                    raise ValueError("prerequisites must test the same committed revision")
                for path, sha in prerequisite["payload"].get("snapshot", {}).get("inputs", {}).items():
                    if payload.get("snapshot", {}).get("inputs", {}).get(path) != sha:
                        raise ValueError("prerequisite external inputs changed or are not pinned by this request: " + path)
            old = self.db.execute("SELECT id,state FROM jobs WHERE fingerprint=? ORDER BY created DESC LIMIT 1",
                                  (fingerprint,)).fetchone()
            retryable = old and old['state'] in {'failed', 'blocked', 'interrupted'}
            if old and old['state'] != 'retired' and not repeat and not (retry_failed and retryable):
                job = old["id"]
                disposition = "reused" if old["state"] in TERMINAL else "joined"
            else:
                job, disposition = uuid.uuid4().hex[:20], "submitted"
                self.db.execute("INSERT INTO jobs(id,fingerprint,payload,state,created,repeat_reason) "
                                "VALUES(?,?,?,'queued',?,?)",
                                (job, fingerprint, encoded(payload), time.time(), repeat))
            self.db.execute("INSERT OR IGNORE INTO subscribers VALUES(?,?,?)", (job, session, time.time()))
            self.db.execute('DELETE FROM withdrawals WHERE job=? AND session=?',(job,session))
            self.event(job, disposition, {"session": session, "repeat_reason": repeat,
                                           "hypothesis": payload["spec"]["hypothesis"]})
        return dict(id=job, disposition=disposition, state=self.get(job)["state"])

    def inbox(self, session, after=0):
        rows = self.db.execute("SELECT e.* FROM events e JOIN subscribers s ON s.job=e.job "
                               "WHERE s.session=? AND e.cursor>? AND NOT EXISTS "
                               "(SELECT 1 FROM withdrawals w WHERE w.job=s.job AND w.session=s.session) "
                               "ORDER BY e.cursor LIMIT 200",
                               (session, after)).fetchall()
        with self.db:
            self.db.executemany('INSERT OR IGNORE INTO deliveries(session,cursor,delivered) VALUES(?,?,?)',
                                [(session, r['cursor'], time.time()) for r in rows])
        events = [dict(cursor=r["cursor"], id=r["job"], at=r["at"], event=r["kind"],data=json.loads(r["data"])) for r in rows]
        from experiment_explain import explain
        for event in events:
            if event['event'] in TERMINAL:
                row = self.get(event['id'])
                row.update(state=event['event'],result=event['data'])
                event['explanation'] = explain(self,row)
        return events

    def acknowledge(self, session, cursor):
        if not self.db.execute('SELECT 1 FROM deliveries WHERE session=? AND cursor=?', (session, cursor)).fetchone():
            raise ValueError('only delivered events can be acknowledged')
        with self.db:
            self.db.execute('UPDATE deliveries SET acknowledged=COALESCE(acknowledged,?) WHERE session=? AND cursor<=?',
                            (time.time(), session, cursor))
        return dict(session=session, acknowledged=cursor)


def worker_lock(store, job):
    directory = store.root / job
    directory.mkdir(exist_ok=True)
    stream = (directory / "worker.lock").open("a")
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        return None
    return stream



def reject_legacy_gpu(store, job, spec):
    """Old queued custom GPU jobs cannot bypass current admission policy.

    The pair/baseline lane booted the overlay stack's launcher, which no longer
    exists: a queued pair job is blocked, not executed."""
    if spec.get("kind") == "cpu":
        return False
    if spec.get("kind") in {"pair", "baseline"}:
        store.state(job, "blocked", {"reason": PAIR_RETIRED,
                                     "evidence": "onepass-policy"})
        return True
    store.state(job, "blocked", {"reason": ONEPASS_ONLY,
                                 "evidence": "onepass-policy"})
    return True


def ensure_worker(store, job):
    if store.get(job)["state"] in TERMINAL:
        return
    with (store.root / (job + ".launch.lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        probe = worker_lock(store, job)
        if probe is None:
            return
        probe.close()
        state = store.get(job)["state"]
        # A worker may finish between the initial read and acquiring its lock.
        if state in TERMINAL:
            return
        if state not in {"queued", "waiting_dependencies", "waiting_cpu", "waiting_cpu_evidence"}:
            store.state(job, "interrupted", {"reason": "worker exited without a result; inspect log before an explicit repeat"})
            return
        payload = store.get(job)["payload"]
        if reject_legacy_gpu(store, job, payload["spec"]):
            return
        controller = None
        if payload.get('retry_controller'):
            from experiment_retry import controller_path
            controller = controller_path(payload)
        checkout = store.root / job / "checkout"
        if Path(payload["repo"]) != checkout:
            # Freeze the submitted commit so agents may keep editing their own
            # checkout. Each job has private build outputs as well as sources.
            if not checkout.exists():
                subprocess.run(["git", "-C", payload["repo"], "worktree", "add", "--detach",
                                str(checkout), payload["spec"]["revision"]], check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=60)
            payload["repo"] = str(checkout)
            with store.db:
                store.db.execute("UPDATE jobs SET payload=? WHERE id=?", (encoded(payload), job))
        with (store.root / job / "run.log").open("ab", buffering=0) as output:
            subprocess.Popen([sys.executable, str((controller or checkout) / "bench/experiments.py"), "--root", str(store.root), "worker", job],
                             stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                             start_new_session=True, close_fds=True)


def child_env(payload, store, job):
    env = dict(payload["environment"])
    env.update(payload["spec"]["env"])
    env.update(payload["paths"])
    env.update(REPO=payload["repo"], FLEET_EXPERIMENT_ID=job,
               FLEET_EXPERIMENT_ROOT=str(store.root), PYTHONUNBUFFERED="1",
               FLEET_CPU_REPORT=str(store.root / job / "cpu-report.json"),
               FLEET_CONTEXT=encoded(payload["spec"]["context"]))
    env['FLEET_CPU_SLOTS'] = str(payload['spec']['resources']['cpu_slots'])
    # Always execute the reviewed repo runners, not a stale log-directory copy.
    env["FLEET"] = str(Path(payload["repo"]) / "bench/fleet.sh")
    return env


def verify(payload):
    if payload.get('retry_controller'):
        from experiment_retry import controller_path
        controller_path(payload)
    current = snapshot(Path(payload["repo"]), payload["spec"])
    if current != payload["snapshot"]:
        raise ValueError("source, build, runner or external inputs changed while queued; resubmit after checking the new context")
    from prepared_artifacts import intact
    if not intact(payload.get('prepared_artifacts', [])):
        raise ValueError('prepared build artifacts changed')


def execute(store, job):
    """Called by fleet.sh only AFTER GO; revalidate before spending a boot."""
    row = store.get(job)
    if row['state'] == 'retired':
        return 0
    payload, spec = row["payload"], row["payload"]["spec"]
    if reject_legacy_gpu(store, job, spec):
        return 4
    verify(payload)
    command = spec['command']
    from experiment_resources import run_cpu
    rc, failure_reason = run_cpu(store, job, command, payload)
    if rc:
        result = {"returncode": rc, "evidence": "cpu-only",
                  "reason": failure_reason or "experiment failed; dependent jobs will not execute"}
        report = store.root / job / "cpu-report.json"
        if report.exists():
            result["checks"] = json.loads(report.read_text())
        store.state(job, "failed", result)
        return rc
    verify(payload)
    state, result = "succeeded", {"returncode": 0, "evidence": "cpu-only", "command": command,
                                 "revision": spec["revision"], "context": spec["context"]}
    from prepared_artifacts import capture
    result["artifacts"] = capture(payload)
    report = store.root / job / "cpu-report.json"
    if report.exists():
        result["checks"] = json.loads(report.read_text())
        if (result["checks"].get("passed") is not True
                or result["checks"].get("coverage_complete") is not True):
            state = "failed"
    from experiment_submission import cacheable_dependencies
    if (spec["kind"] == "cpu" and state == "succeeded" and payload.get("cpu_identity")
            and cacheable_dependencies(store,payload)
            and result.get("checks", {}).get("coverage_complete") is True):
        from cpu_evidence import identity
        if identity(Path(payload["repo"]), spec, payload["environment"]) != payload["cpu_identity"]:
            store.state(job, "failed", {"evidence": "cpu-only", "reason": "CPU environment changed during execution"})
            return 4
        with store.db:
            store.db.execute("INSERT OR REPLACE INTO cpu_cache VALUES(?,?)", (payload["cpu_identity"]["key"], job))
    store.state(job, state, result)
    return 0


def wait_dependencies(store, job, dependencies):
    """Observe completion promptly; recover workers at a separate, slower rate."""
    if not dependencies:
        return True
    store.state(job, 'waiting_dependencies')
    ids = list(dict.fromkeys([job, *dependencies]))
    recover_after = 0.0
    while True:
        # Read only indexed IDs/states, not every dependency's full pinned
        # payload. Chunk parameters for SQLite builds with lower bind limits.
        states = {}
        for offset in range(0, len(ids), 256):
            chunk = ids[offset:offset+256]
            states.update(store.db.execute('SELECT id,state FROM jobs WHERE id IN (' +
                ','.join('?' for _ in chunk) + ')', chunk))
        if len(states) != len(ids):
            raise ValueError('experiment dependency disappeared while waiting')
        if states[job] in TERMINAL:
            return False
        bad = [d for d in dependencies if states[d] in TERMINAL and states[d] != 'succeeded']
        if bad:
            store.state(job, 'blocked', dict(reason='prerequisite did not pass', dependencies=bad))
            return False
        pending = [d for d in dependencies if states[d] != 'succeeded']
        if not pending:
            return True
        if time.monotonic() >= recover_after:
            for dependency in pending:
                ensure_worker(store, dependency)
            recover_after = time.monotonic() + 1
        time.sleep(.05)


def worker(store, job):
    lock = worker_lock(store, job)
    if lock is None:
        return 0
    with lock, ExitStack() as resources:
        if store.get(job)["state"] in TERMINAL:
            return 0
        payload = store.get(job)["payload"]
        spec = payload["spec"]
        if reject_legacy_gpu(store, job, spec):
            return 4
        shared_fd = None
        with store.db:
            store.db.execute("UPDATE jobs SET worker_pid=? WHERE id=?", (os.getpid(), job))
        try:
            if not wait_dependencies(store, job, spec['depends_on']):
                return 0
            verify(payload)
            env = child_env(payload, store, job)
            fleet = env["FLEET"]
            # Preflight sees the ACTUAL payload. Hiding it behind execute would
            # defeat the existing --cpu classifier.
            actual = spec["command"]
            classification = subprocess.check_output([payload["bash"], fleet, "classify", *actual],
                                                     env=env, cwd=payload["repo"], text=True).strip()
            if classification == "gpu":
                raise ValueError("CPU experiment shows GPU use; correct the manifest")
            from experiment_submission import cacheable_dependencies
            if payload.get("cpu_identity") and cacheable_dependencies(store,payload):
                from cpu_evidence import identity
                if identity(Path(payload["repo"]), spec, payload["environment"]) != payload["cpu_identity"]:
                    raise ValueError("CPU environment changed while queued")
                from experiment_sharing import cpu_claim, joined_failure
                claim, joined = cpu_claim(store, job, payload)
                resources.enter_context(claim)
                shared_fd = claim.fileno()
                verify(payload)
                if identity(Path(payload['repo']), spec, payload['environment']) != payload['cpu_identity']:
                    raise ValueError('CPU environment changed while waiting for shared evidence')
                failure = joined_failure(store, joined, payload)
                retried_owner = joined and store.db.execute(
                    'SELECT 1 FROM retry_attempts WHERE attempt=? AND source=? LIMIT 1', (job, joined)).fetchone()
                if failure and not store.get(job)['repeat_reason'] and not retried_owner:
                    store.state(job, 'blocked', failure)
                    return 0
                hit = store.db.execute("SELECT job FROM cpu_cache WHERE key=?", (payload["cpu_identity"]["key"],)).fetchone()
                if hit and not store.get(job)["repeat_reason"]:
                    source = store.get(hit["job"])
                    from prepared_artifacts import intact
                    if (source["state"] == "succeeded" and intact((source["result"] or {}).get("artifacts", [])) and source["result"].get("checks", {}).get("coverage_complete") is True):
                        result = dict(source["result"], revision=spec["revision"], cache_source=source["id"],
                                      tested_revision=source["payload"]["spec"]["revision"],
                                      cache_identity=payload["cpu_identity"])
                        verify(payload)
                        store.state(job, "succeeded", result)
                        return 0
            if any((store.get(d)['result'] or {}).get('artifacts') for d in spec['depends_on']):
                from experiment_resources import readiness
                from prepared_artifacts import materialize
                from experiment_metrics import timed
                with timed(store,job,'preparation'):
                    payload['prepared_artifacts'] = materialize(store, payload)
                    nodes = readiness(spec['resources'])
                with store.db:
                    store.db.execute('UPDATE jobs SET payload=? WHERE id=?', (encoded(payload), job))
                    store.event(job, 'prepared', {'artifacts': payload['prepared_artifacts'],
                                                'nodes': nodes})
            store.state(job, "queued_fleet")
            from experiment_metrics import predict
            estimate = predict(store.db,store.get(job)['payload'])
            with store.db:
                store.event(job,'duration_estimate',estimate)
            command = [payload["bash"], fleet, "run", "--cpu", "exp-" + job,
                       str(estimate['minutes']), "experiment " + job, "--", sys.executable,
                       str(HERE / "experiments.py"), "--root", str(store.root), "execute", job]
            # The child retains the lock if the supervisor crashes. Recovery
            # cannot launch a second copy while the first is still in flight.
            rc = subprocess.call(command, cwd=payload["repo"], env=env,
                                 pass_fds=(lock.fileno(),) + ((shared_fd,) if shared_fd is not None else ()))
            if store.get(job)["state"] not in TERMINAL:
                store.state(job, "failed", {"returncode": rc, "reason": "runner exited without an experiment result"})
        except RetiredJob:
            return 0
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            if store.get(job)['state'] == 'retired':
                return 0
            store.state(job, "failed", {"reason": str(exc)})
    return 0


def stats(store):
    groups = {}
    for row in store.db.execute("SELECT * FROM jobs"):
        kind = json.loads(row["payload"])["spec"]["kind"]
        group = groups.setdefault(kind, dict(jobs=0, valid_results=0, queue_s=[], result_s=[]))
        group["jobs"] += 1
        if row["started"]:
            group["queue_s"].append(row["started"] - row["created"])
        if row["state"] == "succeeded":
            group["valid_results"] += 1
            group["result_s"].append(row["finished"] - row["created"])
    for group in groups.values():
        for field in ("queue_s", "result_s"):
            values = sorted(group.pop(field))
            group[field + "_p50"] = round(statistics.median(values), 2) if values else None
            group[field + "_p95"] = round(values[max(0, math.ceil(.95 * len(values)) - 1)], 2) if values else None
    dispositions = {r["kind"]: r["n"] for r in store.db.execute(
        "SELECT kind,count(*) AS n FROM events WHERE kind IN ('submitted','joined','reused') GROUP BY kind")}
    latencies = {}
    for prefix, kinds in [('', ['succeeded']), ('terminal_', sorted(TERMINAL))]:
        for field, column in [('delivery_s', 'delivered'), ('acknowledgment_s', 'acknowledged')]:
            placeholders = ','.join('?' for _ in kinds)
            values = sorted(r[0] for r in store.db.execute(
                f"SELECT d.{column}-e.at FROM deliveries d JOIN events e ON e.cursor=d.cursor "
                f"WHERE e.kind IN ({placeholders}) AND d.{column} IS NOT NULL", kinds))
            latencies[prefix + field] = dict(n=len(values), p50=statistics.median(values) if values else None,
                                    p95=values[max(0, math.ceil(.95*len(values))-1)] if values else None)
    from experiment_metrics import summary
    return dict(by_kind=groups, requests=dispositions, result_consumption=latencies, phases=summary(store))


def collect(store, session, path, repo):
    """Archive private ledgers without converting unverified data into a pass."""
    data = path.read_bytes()
    if len(data) > 64 * 1024 * 1024:
        raise ValueError('collect one result file of at most 64 MiB')
    if path.suffix == '.jsonl':
        rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    else:
        value = json.loads(data)
        rows = value if isinstance(value, list) else [value]
    sha = hashlib.sha256(data).hexdigest()
    payload = dict(repo=str(repo), environment={}, snapshot={}, paths={},
                   spec=dict(kind='observation', revision=git(repo, 'rev-parse', 'HEAD'),
                             hypothesis='Collected result file: ' + path.name, depends_on=[],
                             artifact_sha256=sha, original_path=str(path.resolve())))
    answer = store.submit(session, payload)
    job = answer['id']
    destination = store.root / job / ('collected' + path.suffix)
    destination.parent.mkdir(exist_ok=True)
    destination.write_bytes(data)
    if answer['disposition'] == 'submitted':
        store.state(job, 'incomplete', dict(evidence='external-archive', records=len(rows),
                    artifact=str(destination), sha256=sha,
                    reason='archived for discovery; performance and provenance have not been adjudicated'))
    return dict(answer, artifact=str(destination), records=len(rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=os.environ.get("FLEET_EXPERIMENT_ROOT", str(
        Path(os.environ.get("FLEET_DIR", "/home/choiceoh/glm53-logs/fleet")) / "experiments")))
    sub = ap.add_subparsers(dest="action", required=True)
    submit = sub.add_parser("submit")
    submit.add_argument("session")
    submit.add_argument("manifest", type=Path)
    submit.add_argument("--repeat", metavar="REASON", help="request an additional sample, with a recorded reason")
    submit.add_argument('--supersedes',action='append',default=[],metavar='ID',help='withdraw your demand for an older request')
    batch = sub.add_parser('batch')
    batch.add_argument('session')
    batch.add_argument('manifest',type=Path)
    batch.add_argument('--repeat',metavar='REASON')
    plan = sub.add_parser('plan')
    plan.add_argument('session')
    plan.add_argument('manifest', type=Path)
    plan.add_argument('--base', help='compare committed changes with this explicit git ref to select CPU suites')
    mode = plan.add_mutually_exclusive_group()
    mode.add_argument('--submit', action='store_true')
    mode.add_argument('--prepare-only', action='store_true', help='submit the CPU stages; retain the resolved preparation receipts')
    retire = sub.add_parser('retire')
    retire.add_argument('session')
    retire.add_argument('id')
    retire.add_argument('--replacement',required=True)
    retire.add_argument('--reason',required=True)
    retry = sub.add_parser('retry', help='retry an unsuccessful request using its saved manifest and valid evidence')
    retry.add_argument('session')
    retry.add_argument('id')
    retry.add_argument('--reason', required=True)
    pending = sub.add_parser('pending')
    pending.add_argument('id')
    estimate = sub.add_parser('estimate')
    estimate.add_argument('id')
    for cmd in ("worker", "execute", "result", "wait"):
        p = sub.add_parser(cmd)
        p.add_argument("id")
        if cmd in {"result", "wait"}:
            p.add_argument("--details", action="store_true", help="include the full pinned environment and input hashes")
        if cmd == "wait":
            p.add_argument("--timeout", type=float, default=60)
    inbox = sub.add_parser("inbox")
    inbox.add_argument("session")
    inbox.add_argument("--after", type=int, default=0)
    inbox.add_argument('--wait', type=float, default=0, help='long poll for at most 60 seconds')
    ack = sub.add_parser('ack')
    ack.add_argument('session')
    ack.add_argument('cursor', type=int)
    collector = sub.add_parser('collect')
    collector.add_argument('session')
    collector.add_argument('path', type=Path)
    jobs = sub.add_parser("jobs")
    jobs.add_argument('--session', help='show only your subscribed requests, excluding withdrawn demand')
    jobs.add_argument('--active', action='store_true', help='exclude completed, failed, and retired requests')
    jobs.add_argument('--limit', type=int, default=100, help='maximum results, 1..1000 (default: 100)')
    sub.add_parser("stats")
    args = ap.parse_args()
    store = Store(args.root)
    if getattr(args, 'session', None) is not None and not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", args.session):
        raise ValueError('session must be a short alphanumeric name')
    if args.action == "submit":
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", args.session):
            raise ValueError("session must be a short alphanumeric name")
        if args.repeat is not None and not args.repeat.strip():
            raise ValueError("repeat needs a reason")
        repo = Path(os.environ.get("REPO", HERE.parent)).resolve()
        spec = normalize(json.loads(args.manifest.read_text()), repo)
        from experiment_retirement import subscribed
        if any(not subscribed(store,args.session,old) or store.get(old)['payload']['spec']['kind'] != spec['kind']
               for old in args.supersedes):
            raise ValueError('supersedes must name your subscribed requests of the same kind')
        from experiment_submission import submit_many
        answer = submit_many(store,args.session,[dict(name='request',manifest=spec)],repo,
                             repeat=args.repeat,launch=False)['requests'][0]
        answer.pop('name')
        if args.supersedes:
            from experiment_retirement import retire
            answer['superseded'] = [retire(store,args.session,old,answer['id'],'Replaced by newer submission')
                                     for old in args.supersedes if old != answer['id']]
        ensure_worker(store, answer["id"])
    elif args.action == 'batch':
        from experiment_submission import submit_many
        answer = submit_many(store,args.session,json.loads(args.manifest.read_text()),
                             Path(os.environ.get('REPO',HERE.parent)).resolve(),repeat=args.repeat)
    elif args.action == 'plan':
        from experiment_plan import run
        answer = run(args, store, Path(os.environ.get('REPO', HERE.parent)).resolve())
    elif args.action == 'ack':
        answer = store.acknowledge(args.session, args.cursor)
    elif args.action == 'pending':
        return 3 if store.get(args.id)['state'] == 'retired' else 0
    elif args.action == 'estimate':
        from experiment_metrics import predict
        answer = predict(store.db,store.get(args.id)['payload'])
    elif args.action == 'retire':
        from experiment_retirement import retire
        answer = retire(store,args.session,args.id,args.replacement,args.reason)
    elif args.action == 'retry':
        from experiment_retry import retry
        answer = retry(store, args.session, args.id, args.reason,
                       Path(os.environ.get('REPO', HERE.parent)).resolve())
    elif args.action == 'collect':
        answer = collect(store, args.session, args.path, Path(os.environ.get('REPO', HERE.parent)).resolve())
    elif args.action == "worker":
        return worker(store, args.id)
    elif args.action == "execute":
        return execute(store, args.id)
    elif args.action in {"result", "wait"}:
        deadline = time.monotonic() + (max(0, min(args.timeout, 60)) if args.action == "wait" else 0)
        while True:
            ensure_worker(store, args.id)
            answer = store.get(args.id)
            if answer["state"] in TERMINAL or time.monotonic() >= deadline:
                break
            time.sleep(.2)
        from experiment_explain import explain
        answer['explanation'] = explain(store,answer)
        if not args.details:
            payload = answer.pop("payload")
            answer.update(revision=payload["spec"]["revision"], checkout=payload["repo"],
                          hypothesis=payload["spec"]["hypothesis"])
    elif args.action == "inbox":
        deadline = time.monotonic() + max(0, min(args.wait, 60))
        while True:
            events = store.inbox(args.session, args.after)
            if events or time.monotonic() >= deadline:
                break
            time.sleep(.2)
        answer = dict(events=events, cursor=events[-1]["cursor"] if events else args.after)
    elif args.action == "stats":
        answer = stats(store)
    else:
        answer = store.list_jobs(args.session, args.active, args.limit)
    print(encoded(answer))
    return 2 if args.action == "plan" and "error" in answer else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, sqlite3.Error, subprocess.SubprocessError) as exc:
        print(encoded({"error": str(exc)}), file=sys.stderr)
        raise SystemExit(2)
