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
import signal
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
        if spec["kind"] == "pair":
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
    return result


def normalize(raw, repo):
    if not isinstance(raw, dict):
        raise ValueError("manifest must be a JSON object")
    allowed = {"kind", "revision", "hypothesis", "command", "knobs", "inputs", "context",
               "env", "depends_on", "estimate_min", "timeout_s"}
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
    return dict(kind=kind, revision=revision, hypothesis=raw["hypothesis"], command=command,
                knobs=knobs, env=env, inputs=inputs, context=context,
                depends_on=sorted(set(deps)), estimate_min=estimate, timeout_s=timeout)


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
        return [dict(cursor=r["cursor"], id=r["job"], at=r["at"], event=r["kind"],
                     data=json.loads(r["data"])) for r in rows]


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
        if state not in {"queued", "waiting_dependencies"}:
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
    if payload["spec"]["kind"] == "pair":
        env["IMAGE"] = payload["spec"]["context"]["image"]
    # Always execute the reviewed repo runners, not a stale log-directory copy.
    env["FLEET"] = str(Path(payload["repo"]) / "bench/fleet.sh")
    env["LEVER"] = str(Path(payload["repo"]) / "bench/ab-lever.sh")
    return env


def verify(payload):
    current = snapshot(Path(payload["repo"]), payload["spec"], payload["paths"]["MK_OVERLAY_STAMP"])
    if current != payload["snapshot"]:
        raise ValueError("source, build, runner or external inputs changed while queued; resubmit after checking the new context")


def pair_result(payload, job):
    from baseline import load
    from judge import baselines_on, judge
    rows = load(payload["paths"]["ONEPASS_JSONL"])
    name = "EXP-" + job
    candidates = [r for r in rows if r.get("name") == name and r.get("experiment_id") == job
                  and not r.get("rehearsal")]
    if not candidates:
        return "incomplete", {"evidence": "none", "reason": "no fresh onepass record for this experiment"}
    cand = candidates[-1]
    if (cand.get("overlay") != payload["snapshot"]["build"][:12]
            or not payload["spec"]["revision"].startswith(cand.get("git") or "MISSING")
            or not cand.get("workload")):
        return "incomplete", {"evidence": "unmatched", "reason": "record build/revision/workload does not match submission"}
    profile = dict(line.split("=", 1) for line in (Path(payload["repo"]) / "profiles/glm53.env").read_text().splitlines()
                   if line.startswith("VLLM_") and "=" in line)
    for k, v in payload["spec"]["knobs"].items():
        actual = (cand.get("knobs") or {}).get(k, profile.get(k, "").strip().strip('"'))
        if actual != v:
            return "incomplete", {"evidence": "unmatched", "reason": "requested knob not attested: " + k}
    bases, _ = baselines_on(rows, cand)
    verdict = judge(cand, bases[-1] if bases else None, rows)
    result = dict(evidence="gpu-pair", verdict=verdict, candidate=cand,
                  baseline=bases[-1] if bases else None)
    return ("succeeded" if verdict["status"] == "valid" else "incomplete"), result


def execute(store, job):
    """Called by fleet.sh only AFTER GO; revalidate before spending a boot."""
    row = store.get(job)
    payload, spec = row["payload"], row["payload"]["spec"]
    if spec["kind"] != "cpu":
        holder = Path(payload["paths"]["FLEET_DIR"]) / "holder"
        if not holder.exists() or holder.read_text().split("|", 1)[0] != "exp-" + job:
            raise ValueError("execute requires this experiment's fleet hold")
    verify(payload)
    store.state(job, "running")
    if spec["kind"] == "pair":
        command = [payload["bash"], str(Path(payload["repo"]) / "bench/pair.sh"),
                   "EXP-" + job, " ".join(k + "=" + v for k, v in sorted(spec["knobs"].items()))]
    else:
        command = spec["command"]
    if spec["kind"] == "cpu":
        proc = subprocess.Popen(command, cwd=payload["repo"], start_new_session=True)
        try:
            rc = proc.wait(timeout=spec["timeout_s"])
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            store.state(job, "failed", {"returncode": 124, "evidence": "cpu-only",
                                       "reason": "CPU time budget exceeded; dependent jobs will not execute"})
            return 124
    else:
        rc = subprocess.call(command, cwd=payload["repo"])
    if rc:
        result = {"returncode": rc, "evidence": "cpu-only" if spec["kind"] == "cpu" else "process-exit",
                  "reason": "experiment failed; dependent jobs will not execute"}
        report = store.root / job / "cpu-report.json"
        if spec["kind"] == "cpu" and report.exists():
            result["checks"] = json.loads(report.read_text())
        store.state(job, "failed", result)
        return rc
    verify(payload)
    if spec["kind"] == "pair":
        state, result = pair_result(payload, job)
    elif spec["kind"] == "cpu":
        state, result = "succeeded", {"returncode": 0, "evidence": "cpu-only", "command": command,
                                     "revision": spec["revision"], "context": spec["context"]}
        report = store.root / job / "cpu-report.json"
        if report.exists():
            result["checks"] = json.loads(report.read_text())
    else:
        state, result = "incomplete", {"returncode": 0, "evidence": "probe-log",
                                      "reason": "probe exited successfully; inspect its numeric and timing evidence before promotion"}
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
                deps = [store.get(d) for d in spec["depends_on"]]
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
                      if spec["kind"] == "pair" else spec["command"])
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
    return dict(by_kind=groups, requests=dispositions)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=os.environ.get("FLEET_EXPERIMENT_ROOT", str(
        Path(os.environ.get("FLEET_DIR", "/home/choiceoh/glm53-logs/fleet")) / "experiments")))
    sub = ap.add_subparsers(dest="action", required=True)
    submit = sub.add_parser("submit")
    submit.add_argument("session")
    submit.add_argument("manifest", type=Path)
    submit.add_argument("--repeat", metavar="REASON", help="request an additional sample, with a recorded reason")
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
    sub.add_parser("jobs")
    sub.add_parser("stats")
    args = ap.parse_args()
    store = Store(args.root)
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
        answer = store.submit(args.session, payload, args.repeat)
        ensure_worker(store, answer["id"])
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
        events = store.inbox(args.session, args.after)
        answer = dict(events=events, cursor=events[-1]["cursor"] if events else args.after)
    elif args.action == "stats":
        answer = stats(store)
    else:
        answer = [dict(r) for r in store.db.execute(
            "SELECT id,state,created,started,finished FROM jobs ORDER BY created DESC LIMIT 100")]
    print(encoded(answer))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, sqlite3.Error, subprocess.SubprocessError) as exc:
        print(encoded({"error": str(exc)}), file=sys.stderr)
        raise SystemExit(2)
