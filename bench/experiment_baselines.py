#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One shared defaults reservation per immutable pair context.

Build the independent noise sample set once, before admitting candidate pairs.
The reservation goes through normal fleet preflight/admission and never holds
the GPU while waiting for a CPU/probe prerequisite.
"""
import copy
from pathlib import Path
import subprocess


def reference(payload):
    return dict(overlay=payload["snapshot"]["build"][:12], git=payload["spec"]["revision"],
                harness=39, doc_lang="ko", thinking=True, runtime=payload["spec"]["context"],
                workload=dict(ctx=[int(c) for c in payload["spec"]["env"].get(
                    "QUALITY_CTX", "2000,32000,128000").split(",")], seed=7,
                    max_tokens=400, combine_min_ctx=32000))


def samples(payload):
    from baseline import load
    from judge import baselines_on, record_errors
    rows, _ = baselines_on(load(payload["paths"]["ONEPASS_JSONL"]), reference(payload))
    # Actual container ID + StartedAt distinguishes independent boots. Replays
    # of one boot and historical records without this evidence cannot pad n.
    distinct = {}
    for r in rows:
        if isinstance(r.get("boot_id"), str) and r["boot_id"] and not record_errors(r):
            distinct[r["boot_id"]] = r
    return list(distinct.values())


def reserve(store, job):
    existing = store.db.execute("SELECT dependency FROM dependencies WHERE job=? AND kind='baseline'", (job,)).fetchone()
    if existing:
        return existing[0]
    row = store.get(job)
    payload = copy.deepcopy(row["payload"])
    payload["spec"].update(kind="baseline", knobs={}, depends_on=[], command=[],
                           hypothesis="Prepare shared independent defaults samples", estimate_min=30)
    # CPU/probe contracts gate their own candidate. This reservation is only
    # created AFTER those gates pass, and is otherwise candidate-independent.
    payload["spec"].pop("probe_contract", None)
    payload["baseline_samples"] = 3
    request = store.submit("baseline-" + job, payload, repeat=row["repeat_reason"])
    with store.db:
        store.db.execute("INSERT OR IGNORE INTO dependencies VALUES(?,?,?)", (job, request["id"], "baseline"))
        store.event(job, "baseline_reserved", request)
    return request["id"]


def run(store, job, payload):
    from baseline import load
    from judge import compatible, is_baseline, record_errors
    from experiments import verify
    target = payload["baseline_samples"]
    ref = reference(payload)
    for index in range(target):
        before = samples(payload)
        if len(before) >= target:
            break
        verify(payload)
        name = f"EXP-{job}-BASE-{index + 1}"
        command = [payload["bash"], str(Path(payload["repo"]) / "bench/ab-lever.sh"), name, ""]
        rc = subprocess.call(command, cwd=payload["repo"])
        if rc:
            return "failed", dict(evidence="gpu-baseline", returncode=rc, reason="shared defaults arm failed")
        verify(payload)
        fresh = [r for r in load(payload["paths"]["ONEPASS_JSONL"])
                 if r.get("name") == name and r.get("experiment_id") == job and not r.get("rehearsal")]
        if (len(fresh) != 1 or not compatible(fresh[0], ref) or not is_baseline(fresh[0])[0]
                or record_errors(fresh[0]) or not fresh[0].get("boot_id")
                or fresh[0]["boot_id"] in {r["boot_id"] for r in before}
                or not payload["spec"]["revision"].startswith(fresh[0].get("git") or "MISSING")):
            return "failed", dict(evidence="gpu-baseline", reason="defaults sample lacks fresh independent boot/gate evidence")
        with store.db:
            store.event(job, "baseline_sample", fresh[0])
    bases = samples(payload)
    return ("succeeded" if len(bases) >= target else "incomplete"), dict(
        evidence="gpu-baseline", samples=len(bases), baseline=bases, scope="same build/workload/runtime")
