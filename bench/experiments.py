#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Durable, non-blocking experiment submissions on the fleet head.

The existing fleet.sh owns GPU admission. This module owns request identity,
CPU prerequisites, subscribers and results; it never deploys or steals a hold.
See bench/EXPERIMENTS.md for the manifest and agent workflow.
"""
from __future__ import annotations

import argparse
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
TERMINAL = {"succeeded", "failed", "blocked", "incomplete", "interrupted"}
RESERVED = {"HOME", "PATH", "PYTHONPATH", "BASH_ENV", "ENV", "REPO", "LOGD",
            "LEVER", "SKIP_BOOT", "LEGS", "MK_OVERLAY_STAMP", "MK_COLD_COMPILE"}
BASE_ENV = ("HOME", "PATH", "USER", "LOGNAME", "LANG", "LC_ALL", "SSH_AUTH_SOCK", "TMPDIR")


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


def snapshot(repo, spec, stamp):
    revision = git(repo, "rev-parse", "HEAD")
    if revision != spec["revision"]:
        raise ValueError("checkout revision changed; commit and submit the intended revision")
    if git(repo, "status", "--porcelain", "--untracked-files=normal"):
        raise ValueError("experiment checkout must be clean (including untracked inputs)")
    result = {"revision": revision, "inputs": {p: digest(p) for p in spec["inputs"]},
              "host": platform.node(), "platform": platform.platform(),
              "python": sys.version, "runner": digest(__file__)}
    if spec["kind"] != "cpu":
        build = Path(stamp).read_text().strip()
        if not re.fullmatch(r"[a-fA-F0-9]{12,64}", build):
            raise ValueError("GPU submissions need a valid deployed overlay stamp")
        result["build"] = build
        if spec["kind"] in {"pair", "baseline"}:
            profile = dict(line.split("=", 1) for line in (repo / "profiles/glm53.env").read_text().splitlines()
                           if "=" in line and not line.startswith("#"))
            overlay = Path(profile.get("PROFILE_OVERLAY_DIR", "/home/choiceoh/overlays/glm53").strip('"\''))
            manifest = overlay / "manifest.tsv"
            lines = manifest.read_text().splitlines()
            if "# source_commit=" + revision not in lines or digest(manifest) != build:
                raise ValueError("deployed manifest is not this committed build; deploy through the fleet before submitting a pair")
            result["overlays"] = {line.split("\t")[0]: digest(overlay / line.split("\t")[0])
                                  for line in lines if line and not line.startswith("#")}
            modules = profile.get("MODULES", "").strip('"\'').split()
            for name, sha in result["overlays"].items():
                sources = [repo / "overlay/modules" / module / name for module in modules
                           if (repo / "overlay/modules" / module / name).is_file()]
                if len(sources) != 1 or digest(sources[0]) != sha:
                    raise ValueError("deployed overlay differs from committed source: " + name)
            result["image"] = subprocess.check_output(
                ["docker", "image", "inspect", spec["context"]["image"], "--format", "{{.Id}}"],
                text=True, stderr=subprocess.PIPE, timeout=15).strip()
            if result["image"] != spec["context"]["image"]:
                raise ValueError("context.image must pin the local immutable sha256 image ID")
        elif spec.get("probe_contract"):
            result["image"] = subprocess.check_output(
                ["docker", "image", "inspect", spec["context"]["image"], "--format", "{{.Id}}"],
                text=True, stderr=subprocess.PIPE, timeout=15).strip()
            if result["image"] != spec["context"]["image"]:
                raise ValueError("structured probe requires the pinned local immutable image ID")
    return result


def normalize(raw, repo):
    if not isinstance(raw, dict):
        raise ValueError("manifest must be a JSON object")
    allowed = {"kind", "revision", "hypothesis", "command", "knobs", "inputs", "context",
               "env", "depends_on", "estimate_min", "timeout_s", "probe_contract", "evaluations",
               "resources", "outputs", "api_port"}
    if set(raw) - allowed:
        raise ValueError("unknown manifest fields: " + ", ".join(sorted(set(raw) - allowed)))
    kind = raw.get("kind")
    if kind not in {"cpu", "pair", "probe"}:
        raise ValueError("kind must be cpu, pair or probe")
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
    command, knobs = raw.get("command", []), raw.get("knobs", {})
    if not isinstance(command, list) or any(not isinstance(v, str) or not v or "\0" in v for v in command):
        raise ValueError("command must be an argv array")
    if kind == 'cpu':
        from cpu_checks import canonical
        command = canonical(command, repo)
    if not isinstance(knobs, dict) or any(
        not re.fullmatch(r"VLLM_[A-Z0-9_]+", k) or not isinstance(v, str)
        or not re.fullmatch(r"[A-Za-z0-9_.,:/+%-]+", v) for k, v in knobs.items()
    ):
        raise ValueError("knobs must contain literal VLLM_* values")
    if kind == "pair" and (command or not knobs):
        raise ValueError("pair takes knobs and runs the standard onepass pair, not a custom command")
    if kind != "pair" and (not command or knobs):
        raise ValueError("cpu/probe takes command, not knobs")
    # Effective launcher settings must be declared in the pair, not inherited
    # from an agent's shell. CPU/probe environments are part of their identity.
    if kind == "pair" and set(env) - {"QUALITY_CTX", "HEALTH_BUDGET_S"}:
        raise ValueError("pair env supports QUALITY_CTX and HEALTH_BUDGET_S only; use knobs")
    if kind == "pair" and (not re.fullmatch(r"[0-9]+(?:,[0-9]+)*", env.get("QUALITY_CTX", "2000,32000,128000"))
                           or not env.get("HEALTH_BUDGET_S", "3000").isdigit()):
        raise ValueError("pair contexts and health budget must be numeric")
    inputs = raw.get("inputs", [])
    if not isinstance(inputs, list) or any(not isinstance(p, str) for p in inputs):
        raise ValueError("inputs must list external input files to hash")
    inputs = sorted({str((repo / p).resolve()) for p in inputs})
    context = raw.get("context", {})
    if not isinstance(context, dict) or any(not isinstance(v, str) for v in context.values()):
        raise ValueError("context must map names to immutable environment identifiers")
    if kind != "cpu" and not all(context.get(k) for k in ("image", "model", "hardware")):
        raise ValueError("GPU context must identify immutable image, model and hardware versions")
    if kind == "pair" and not re.fullmatch(r"sha256:[a-f0-9]{64}", context["image"]):
        raise ValueError("pair context.image must be an immutable sha256 image ID")
    deps = raw.get("depends_on", [])
    if not isinstance(deps, list) or any(not isinstance(d, str) for d in deps):
        raise ValueError("depends_on must list existing experiment IDs")
    estimate = raw.get("estimate_min", 15)
    if type(estimate) is not int or not 1 <= estimate <= 720:
        raise ValueError("estimate_min must be an integer between 1 and 720")
    timeout = raw.get("timeout_s", 900)
    if type(timeout) is not int or not 1 <= timeout <= 86400:
        raise ValueError("timeout_s must be an integer between 1 and 86400 (CPU commands only)")
    probe_contract = raw.get("probe_contract")
    if probe_contract is not None:
        from probe_report import contract
        if kind != "probe":
            raise ValueError("probe_contract is only valid for probes")
        if not re.fullmatch(r"sha256:[a-f0-9]{64}", context["image"]):
            raise ValueError("structured probe context.image must be an immutable sha256 image ID")
        contract(probe_contract)
    from measurement_contract import evaluations
    from experiment_resources import normalize as resource_spec
    evals = evaluations(raw) if kind == 'pair' else None
    if raw.get('evaluations') is not None and kind != 'pair':
        raise ValueError('evaluations only applies to pair')
    outputs = raw.get('outputs', [])
    if not isinstance(outputs, list) or any(not isinstance(p, str) for p in outputs) or (outputs and kind != 'cpu'):
        raise ValueError('outputs must be a list of CPU build output paths')
    from prepared_artifacts import output_path
    for output in outputs:
        output_path(repo, output)
    port = raw.get('api_port', 8000)
    if type(port) is not int or not 1024 <= port <= 65535 or (kind == 'cpu' and port != 8000):
        raise ValueError('GPU api_port must be 1024..65535')
    return dict(kind=kind, revision=revision, hypothesis=raw["hypothesis"], command=command,
                knobs=knobs, env=env, inputs=inputs, context=context,
                depends_on=sorted(set(deps)), estimate_min=estimate, timeout_s=timeout,
                probe_contract=probe_contract, evaluations=evals, resources=resource_spec(raw.get('resources')),
                outputs=sorted(set(outputs)), api_port=port)


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
            CREATE TABLE IF NOT EXISTS deliveries (
                session TEXT NOT NULL, cursor INTEGER NOT NULL, delivered REAL NOT NULL, acknowledged REAL,
                PRIMARY KEY(session, cursor));
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

    def state(self, job, state, result=None):
        with self.db:
            self.db.execute("UPDATE jobs SET state=?, result=COALESCE(?,result), "
                            "started=CASE WHEN ?='running' THEN COALESCE(started,?) ELSE started END, "
                            "finished=CASE WHEN ? THEN ? ELSE finished END WHERE id=?",
                            (state, encoded(result) if result is not None else None, state,
                             time.time(), state in TERMINAL, time.time(), job))
            self.event(job, state, result or {})

    def submit(self, session, payload, repeat=None):
        identity = {k: v for k, v in payload.items() if k != "repo"}
        identity["spec"] = {k: v for k, v in payload["spec"].items()
                            if k not in {"hypothesis", "estimate_min"}}
        identity["environment"] = {k: v for k, v in payload["environment"].items() if k != "SSH_AUTH_SOCK"}
        if 'prepared_artifacts' in identity:
            identity['prepared_artifacts'] = [{k:v for k,v in r.items() if k != 'path'} for r in identity['prepared_artifacts']]
        fingerprint = hashlib.sha256(encoded(identity).encode()).hexdigest()
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            for dep in payload["spec"]["depends_on"]:
                prerequisite = self.get(dep)  # existing IDs only: no dependency cycles
                if prerequisite["payload"]["spec"]["revision"] != payload["spec"]["revision"]:
                    raise ValueError("prerequisites must test the same committed revision")
                for path, sha in prerequisite["payload"].get("snapshot", {}).get("inputs", {}).items():
                    if payload.get("snapshot", {}).get("inputs", {}).get(path) != sha:
                        raise ValueError("prerequisite external inputs changed or are not pinned by this request: " + path)
            old = self.db.execute("SELECT id,state FROM jobs WHERE fingerprint=? ORDER BY created DESC LIMIT 1",
                                  (fingerprint,)).fetchone()
            if old and not repeat:
                job = old["id"]
                disposition = "reused" if old["state"] in TERMINAL else "joined"
            else:
                job, disposition = uuid.uuid4().hex[:20], "submitted"
                self.db.execute("INSERT INTO jobs(id,fingerprint,payload,state,created,repeat_reason) "
                                "VALUES(?,?,?,'queued',?,?)",
                                (job, fingerprint, encoded(payload), time.time(), repeat))
            self.db.execute("INSERT OR IGNORE INTO subscribers VALUES(?,?,?)", (job, session, time.time()))
            self.event(job, disposition, {"session": session, "repeat_reason": repeat,
                                           "hypothesis": payload["spec"]["hypothesis"]})
        return dict(id=job, disposition=disposition, state=self.get(job)["state"])

    def inbox(self, session, after=0):
        rows = self.db.execute("SELECT e.* FROM events e JOIN subscribers s ON s.job=e.job "
                               "WHERE s.session=? AND e.cursor>? ORDER BY e.cursor LIMIT 200",
                               (session, after)).fetchall()
        with self.db:
            self.db.executemany('INSERT OR IGNORE INTO deliveries(session,cursor,delivered) VALUES(?,?,?)',
                                [(session, r['cursor'], time.time()) for r in rows])
        return [dict(cursor=r["cursor"], id=r["job"], at=r["at"], event=r["kind"],
                     data=json.loads(r["data"])) for r in rows]

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
        if state not in {"queued", "waiting_dependencies", "waiting_baseline", "waiting_cpu"}:
            store.state(job, "interrupted", {"reason": "worker exited without a result; inspect log before an explicit repeat"})
            return
        payload = store.get(job)["payload"]
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
            subprocess.Popen([sys.executable, str(checkout / "bench/experiments.py"), "--root", str(store.root), "worker", job],
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
    if payload["spec"]["kind"] != "cpu":
        env["IMAGE"] = payload["spec"]["context"]["image"]
        env['GLM53_API_PORT'] = str(payload['spec'].get('api_port', 8000))
        env['HEAD_URL'] = 'http://127.0.0.1:' + env['GLM53_API_PORT']
    if payload['spec']['kind'] in {'pair', 'baseline'}:
        from measurement_contract import evaluations, environment
        env.update(environment(evaluations(payload['spec'])[0]))
    # Always execute the reviewed repo runners, not a stale log-directory copy.
    env["FLEET"] = str(Path(payload["repo"]) / "bench/fleet.sh")
    env["LEVER"] = str(Path(payload["repo"]) / "bench/ab-lever.sh")
    return env


def verify(payload):
    current = snapshot(Path(payload["repo"]), payload["spec"], payload["paths"]["MK_OVERLAY_STAMP"])
    if current != payload["snapshot"]:
        raise ValueError("source, build, runner or external inputs changed while queued; resubmit after checking the new context")
    from prepared_artifacts import intact
    if not intact(payload.get('prepared_artifacts', [])):
        raise ValueError('prepared build artifacts changed')


def knob_mismatch(record, knobs, repo):
    profile = {}
    for line in (Path(repo) / 'profiles/glm53.env').read_text().splitlines():
        line = line.strip()
        if line.startswith(('VLLM_', 'SPEC_K=')) and '=' in line:
            key, value = line.split('=', 1)
            profile['VLLM_GLM53_SPEC_K' if key == 'SPEC_K' else key] = value.strip().strip('"\'')
    expected = {k:v for k,v in knobs.items() if v != profile.get(k)}
    return record.get('knobs') != expected


def pair_result_one(payload, job, index=0):
    from baseline import load
    from judge import baselines_on, judge
    rows = load(payload["paths"]["ONEPASS_JSONL"])
    from measurement_contract import evaluations
    from serving_group import name_for
    from experiment_baselines import reference
    from judge import compatible
    evaluation = evaluations(payload["spec"])[index]
    name = name_for("EXP-" + job, payload["spec"], index)
    candidates = [r for r in rows if r.get("name") == name and r.get("experiment_id") == job
                  and not r.get("rehearsal")]
    if not candidates:
        return "incomplete", {"evidence": "none", "reason": "no fresh onepass record for this experiment"}
    cand = candidates[-1]
    if (cand.get("overlay") != payload["snapshot"]["build"][:12]
            or not payload["spec"]["revision"].startswith(cand.get("git") or "MISSING")
            or not compatible(cand, reference(payload, index))):
        return "incomplete", {"evidence": "unmatched", "reason": "record build/revision/workload does not match submission"}
    if knob_mismatch(cand, payload['spec']['knobs'], payload['repo']):
        return 'incomplete', dict(evidence='unmatched', reason='serving knobs do not exactly match the requested configuration')
    bases, _ = baselines_on(rows, cand)
    from measurement_contract import metric_compatible, metric_value
    obj = evaluation['objective']
    bases = [b for b in bases if metric_compatible(b, cand, obj)
             and (obj['metric'] == 'quality' or metric_value(b, obj) is not None)]
    verdict = judge(cand, bases[-1] if bases else None, rows, evaluation["objective"])
    result = dict(evidence="gpu-pair", verdict=verdict, candidate=cand,
                  baseline=bases[-1] if bases else None)
    return ("succeeded" if verdict["status"] == "valid" else "incomplete"), result


def pair_result(payload, job):
    from measurement_contract import evaluations
    results = [pair_result_one(payload, job, i) for i, _ in enumerate(evaluations(payload['spec']))]
    if len(results) == 1:
        return results[0]
    state = 'succeeded' if all(s == 'succeeded' for s, _ in results) else 'incomplete'
    return state, dict(evidence='gpu-pair', evaluations=[r for _, r in results])


def execute(store, job):
    """Called by fleet.sh only AFTER GO; revalidate before spending a boot."""
    row = store.get(job)
    payload, spec = row["payload"], row["payload"]["spec"]
    if spec["kind"] != "cpu":
        holder = Path(payload["paths"]["FLEET_DIR"]) / "holder"
        if not holder.exists() or holder.read_text().split("|", 1)[0] != "exp-" + job:
            raise ValueError("execute requires this experiment's fleet hold")
    verify(payload)
    if spec["kind"] != "cpu":
        from experiment_resources import readiness
        with store.db:
            store.event(job, "admission_ready", {"nodes": readiness(spec["resources"])})
    store.state(job, "running")
    if spec["kind"] == "baseline":
        from experiment_baselines import run
        state, result = run(store, job, payload)
        store.state(job, state, result)
        # Publish completions for earlier candidates even without result polls.
        for item in store.db.execute("SELECT id FROM jobs WHERE state='incomplete'").fetchall():
            refresh_result(store, item["id"])
        return 0 if state == "succeeded" else 4
    if spec["kind"] == "pair":
        from serving_group import run_pair
        state, result = run_pair(store, job, payload)
        store.state(job, state, result)
        return 0 if state in {'succeeded', 'incomplete'} else 4
    command = spec['command']
    nonce = uuid.uuid4().hex
    binding = dict(revision=spec["revision"], snapshot=payload["snapshot"], context=spec["context"])
    if spec["kind"] == "probe":
        report_path = store.root / job / "probe-report.json"
        report_path.unlink(missing_ok=True)
        os.environ.update(FLEET_PROBE_REPORT=str(report_path), FLEET_PROBE_NONCE=nonce,
                          FLEET_PROBE_BINDING=encoded(binding))
    failure_reason = None
    if spec["kind"] == "cpu":
        from experiment_resources import run_cpu
        rc, failure_reason = run_cpu(store, job, command, payload)
    else:
        rc = subprocess.call(command, cwd=payload["repo"])
    if rc:
        result = {"returncode": rc, "evidence": "cpu-only" if spec["kind"] == "cpu" else "process-exit",
                  "reason": failure_reason or "experiment failed; dependent jobs will not execute"}
        report = store.root / job / "cpu-report.json"
        if spec["kind"] == "cpu" and report.exists():
            result["checks"] = json.loads(report.read_text())
        store.state(job, "failed", result)
        return rc
    verify(payload)
    if spec["kind"] == "cpu":
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
    else:
        from probe_report import adjudicate
        state, result = adjudicate(report_path, spec.get("probe_contract"), job, nonce, binding)
    if (spec["kind"] == "cpu" and state == "succeeded" and payload.get("cpu_identity")
            and result.get("checks", {}).get("coverage_complete") is True):
        from cpu_evidence import identity
        if identity(Path(payload["repo"]), spec, payload["environment"]) != payload["cpu_identity"]:
            store.state(job, "failed", {"evidence": "cpu-only", "reason": "CPU environment changed during execution"})
            return 4
        with store.db:
            store.db.execute("INSERT OR REPLACE INTO cpu_cache VALUES(?,?)", (payload["cpu_identity"]["key"], job))
    store.state(job, state, result)
    return 0


def worker(store, job):
    lock = worker_lock(store, job)
    if lock is None:
        return 0
    with lock:
        if store.get(job)["state"] in TERMINAL:
            return 0
        payload = store.get(job)["payload"]
        spec = payload["spec"]
        with store.db:
            store.db.execute("UPDATE jobs SET worker_pid=? WHERE id=?", (os.getpid(), job))
        try:
            if spec["depends_on"]:
                store.state(job, "waiting_dependencies")
            while spec["depends_on"]:
                deps = [refresh_result(store, d) for d in spec["depends_on"]]
                bad = [d["id"] for d in deps if d["state"] in TERMINAL and d["state"] != "succeeded"]
                if bad:
                    store.state(job, "blocked", {"reason": "prerequisite did not pass", "dependencies": bad})
                    return 0
                if all(d["state"] == "succeeded" for d in deps):
                    break
                for dep in deps:
                    ensure_worker(store, dep["id"])
                time.sleep(1)
            verify(payload)
            env = child_env(payload, store, job)
            fleet = env["FLEET"]
            # Preflight sees the ACTUAL payload. Hiding it behind execute would
            # defeat the existing --cpu classifier / knob checks.
            actual = ([payload["bash"], str(Path(payload["repo"]) / "bench/pair.sh"),
                       "EXP-" + job, " ".join(k + "=" + v for k, v in sorted(spec["knobs"].items()))]
                      if spec["kind"] == "pair" else
                      [payload["bash"], str(Path(payload["repo"]) / "bench/ab-lever.sh"), "EXP-" + job + "-BASE", ""]
                      if spec["kind"] == "baseline" else spec["command"])
            if spec["kind"] != "cpu":
                pf = [payload["bash"], fleet, "preflight"]
                if spec["kind"] == "probe":
                    pf.append("--probe")
                rc = subprocess.call([*pf, "exp-" + job, "--", *actual], env=env, cwd=payload["repo"])
                if rc:
                    raise ValueError("fleet preflight refused the experiment (see run.log)")
            else:
                classification = subprocess.check_output([payload["bash"], fleet, "classify", *actual],
                                                         env=env, cwd=payload["repo"], text=True).strip()
                if classification == "gpu":
                    raise ValueError("CPU experiment shows GPU use; correct the manifest")
                if payload.get("cpu_identity"):
                    from cpu_evidence import identity
                    if identity(Path(payload["repo"]), spec, payload["environment"]) != payload["cpu_identity"]:
                        raise ValueError("CPU environment changed while queued")
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
            if spec["kind"] != "cpu":
                from experiment_resources import readiness
                from prepared_artifacts import materialize
                payload['prepared_artifacts'] = materialize(store, payload)
                with store.db:
                    store.db.execute('UPDATE jobs SET payload=? WHERE id=?', (encoded(payload), job))
                    store.event(job, 'prepared', {'artifacts': payload['prepared_artifacts'],
                                                'nodes': readiness(spec['resources'])})
            if spec["kind"] == "pair":
                from experiment_baselines import reserve, samples, ready
                if not ready(payload):
                    baseline = reserve(store, job)
                    store.state(job, "waiting_baseline", {"baseline_job": baseline})
                    while True:
                        ensure_worker(store, baseline)
                        base = store.get(baseline)
                        if base["state"] in TERMINAL:
                            if base["state"] != "succeeded":
                                store.state(job, "blocked", {"reason": "shared baseline did not pass", "baseline_job": baseline})
                                return 0
                            break
                        time.sleep(.2)
                    verify(payload)
                    if not ready(payload):
                        raise ValueError("shared baseline evidence changed; resubmit after checking its ledger")
            elif spec["kind"] == "baseline":
                from experiment_baselines import samples, ready
                bases = samples(payload)
                if ready(payload):
                    store.state(job, "succeeded", {"evidence": "gpu-baseline", "samples": len(bases), "baseline": bases})
                    return 0
            store.state(job, "queued_fleet")
            lane = ["--cpu"] if spec["kind"] == "cpu" else ["--gpu"]
            if spec["kind"] == "probe":
                lane.append("--probe")
            command = [payload["bash"], fleet, "run", *lane, "exp-" + job,
                       str(spec["estimate_min"]), "experiment " + job, "--", sys.executable,
                       str(HERE / "experiments.py"), "--root", str(store.root), "execute", job]
            # The child retains the lock if the supervisor crashes. Recovery
            # cannot launch a second copy while the first is still in flight.
            rc = subprocess.call(command, cwd=payload["repo"], env=env, pass_fds=(lock.fileno(),))
            if store.get(job)["state"] not in TERMINAL:
                store.state(job, "failed", {"returncode": rc, "reason": "runner exited without an experiment result"})
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            store.state(job, "failed", {"reason": str(exc)})
    return 0


def refresh_result(store, job):
    """A later matching baseline can complete an earlier pair without a boot."""
    row = store.get(job)
    if (row["state"] == "incomplete" and row["payload"]["spec"]["kind"] == "pair"
            and (row["result"] or {}).get("evidence") == "gpu-pair"):
        state, result = pair_result(row["payload"], job)
        if result != row["result"]:
            store.state(job, state, result)
    return store.get(job)


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
    return dict(by_kind=groups, requests=dispositions, result_consumption=latencies)


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
    plan = sub.add_parser('plan')
    plan.add_argument('session')
    plan.add_argument('manifest', type=Path)
    plan.add_argument('--base', help='compare committed changes with this explicit git ref to select CPU suites')
    mode = plan.add_mutually_exclusive_group()
    mode.add_argument('--submit', action='store_true')
    mode.add_argument('--prepare-only', action='store_true', help='submit CPU stages before deployment; retain the resolved GPU manifest')
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
    sub.add_parser("jobs")
    sub.add_parser("stats")
    args = ap.parse_args()
    store = Store(args.root)
    if hasattr(args, 'session') and not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", args.session):
        raise ValueError('session must be a short alphanumeric name')
    if args.action == "submit":
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", args.session):
            raise ValueError("session must be a short alphanumeric name")
        if args.repeat is not None and not args.repeat.strip():
            raise ValueError("repeat needs a reason")
        repo = Path(os.environ.get("REPO", HERE.parent)).resolve()
        spec = normalize(json.loads(args.manifest.read_text()), repo)
        logd = Path(os.environ.get("LOGD", "/home/choiceoh/glm53-logs"))
        paths = {"LOGD": str(logd), "FLEET_DIR": os.environ.get("FLEET_DIR", str(logd / "fleet")),
                 "ONEPASS_JSONL": os.environ.get("ONEPASS_JSONL", str(logd / "bracket-onepass.jsonl")),
                 "ONEPASS_VERDICTS": os.environ.get("ONEPASS_VERDICTS", str(logd / "verdicts.jsonl")),
                 "MK_OVERLAY_STAMP": os.environ.get("MK_OVERLAY_STAMP", str(Path.home() / "glm53-cache/.overlay-sha"))}
        payload = dict(spec=spec, repo=str(repo), paths=paths,
                       bash=shutil.which("bash"), environment={k: os.environ[k] for k in BASE_ENV if k in os.environ},
                       snapshot=snapshot(repo, spec, paths["MK_OVERLAY_STAMP"]))
        if spec["kind"] == "cpu":
            from cpu_evidence import identity
            payload["cpu_identity"] = identity(repo, spec, payload["environment"])
        answer = store.submit(args.session, payload, args.repeat)
        ensure_worker(store, answer["id"])
    elif args.action == 'plan':
        from experiment_plan import run
        answer = run(args, store, Path(os.environ.get('REPO', HERE.parent)).resolve())
    elif args.action == 'ack':
        answer = store.acknowledge(args.session, args.cursor)
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
            answer = refresh_result(store, args.id)
            if answer["state"] in TERMINAL or time.monotonic() >= deadline:
                break
            time.sleep(.2)
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
        answer = [dict(r) for r in store.db.execute(
            "SELECT id,state,created,started,finished FROM jobs ORDER BY created DESC LIMIT 100")]
    print(encoded(answer))
    return 2 if args.action == "plan" and "error" in answer else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, sqlite3.Error, subprocess.SubprocessError) as exc:
        print(encoded({"error": str(exc)}), file=sys.stderr)
        raise SystemExit(2)
