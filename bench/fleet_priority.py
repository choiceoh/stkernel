#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Rank waiting jobs at a free fleet boundary; never preempt a holder.

Small tickets go as a batch: in each lane the small ones (an estimate of SMALL_MAX_MIN or
less) run oldest-first ahead of anything larger, up to BATCH_CAP_MIN of small work per
ranking -- like a CPU pipeline draining the short instructions queued behind a long one
(operator, 2026-09-13). Past 30 minutes of waiting a ticket goes oldest-first ahead of the
unaged; past STARVE_S nothing but the explicit front and a yielded probe passes it. Otherwise
unblock more experiments per estimated minute, with gradual aging. The estimate is the
experiment DB's prediction, else the ledger's median of that session's (then its family's)
last five holds, else what the ticket declared. Failures to read the experiment DB simply
give every job zero dependents.
"""
import argparse
import json
import os
from pathlib import Path
import re
import sqlite3
import statistics
import time

TERMINAL = {"succeeded", "failed", "blocked", "incomplete", "interrupted", "retired"}
SMALL_MAX_MIN = float(os.environ.get("FLEET_SMALL_MAX_MIN", 5))     # a small ticket: this many estimated minutes or fewer
BATCH_CAP_MIN = float(os.environ.get("FLEET_BATCH_CAP_MIN", 15))    # small work that may pass the larger tickets per ranking
AGED_S = 1800                                                        # oldest-first from here (the rule before the batch)
# From here only front/yield passes a ticket. The batch may keep an aged ticket waiting for at most the
# batch's own allowance beyond the old 30 minutes -- so a stream of small tickets costs a long one 15
# minutes at most, and never starves it.
STARVE_S = float(os.environ.get("FLEET_STARVE_S", AGED_S + BATCH_CAP_MIN * 60))
HISTORY_HOLDS = 5


def family(session: str) -> str:
    """A session's family: its name with the digits out -- `c4-rows2-decode-profile` and `c4-rows-decode-profile`
    are one family, as are the numbered reruns of any ticket."""
    return re.sub(r"\d+", "", session)


def history_estimates(ledger, sessions):
    """From the ledger (ts, session, kind, note, held minutes, ...): the median of the last five holds of the
    same session, else of its family -- {session: {minutes, source}} for the sessions history knows."""
    path = Path(ledger)
    if not path.exists():
        return {}
    exact, families = {}, {}
    try:
        for line in path.read_text().splitlines():
            cells = line.split("\t")
            if len(cells) < 5:
                continue
            try:
                held = float(cells[4])
            except ValueError:
                continue
            if held <= 0:
                continue
            exact.setdefault(cells[1], []).append(held)
            families.setdefault(family(cells[1]), []).append(held)
    except OSError:
        return {}
    out = {}
    for s in sessions:
        holds, source = exact.get(s), "history"
        if not holds:
            holds, source = families.get(family(s)), "family"
        if holds:
            # the lower median, as bench/fleet.sh expected_min prints it
            out[s] = dict(minutes=statistics.median_low(holds[-HISTORY_HOLDS:]), source=source)
    return out


def downstream(db):
    if not Path(db).exists():
        return {}
    try:
        with sqlite3.connect("file:" + str(Path(db).resolve()) + "?mode=ro", uri=True, timeout=.2) as conn:
            jobs = {r[0]: (r[1], json.loads(r[2])) for r in conn.execute("SELECT id,state,payload FROM jobs")}
            edges = [(j, d) for j, (state, p) in jobs.items() if state not in TERMINAL
                     for d in p["spec"].get("depends_on", [])]
            if conn.execute("SELECT 1 FROM sqlite_master WHERE name='dependencies'").fetchone():
                edges += [(a, b) for a, b in conn.execute("SELECT job,dependency FROM dependencies")
                          if a in jobs and jobs[a][0] not in TERMINAL]
    except (sqlite3.Error, ValueError, KeyError):
        return {}
    children = {}
    for a, b in edges:
        children.setdefault(b, set()).add(a)
    counts = {}
    for root in jobs:
        seen, todo = {root}, list(children.get(root, []))
        while todo:
            j = todo.pop()
            if j not in seen:
                seen.add(j)
                todo.extend(children.get(j, []))
        counts["exp-" + root] = len(seen) - 1
    return counts


def rank(lines, counts, now, front="", yielded="", probes_ready=True, estimates=None):
    rows = []
    for index, line in enumerate(lines):
        cells = line.rstrip("\n").split("|")
        if len(cells) < 6:
            raise ValueError("malformed queue row")
        session, created, estimate = cells[1], float(cells[2]), max(1, float(cells[3]))
        prediction = (estimates or {}).get(session,{})
        estimate = max(1,prediction.get('minutes',estimate))
        age = max(0, now - created)
        dependents = counts.get(session, 0)
        score = (1 + dependents) / estimate + age / 1800
        rows.append(dict(session=session, age_s=round(age), dependents=dependents,
                         estimate_min=estimate,estimate_source=prediction.get('source','declared'),
                         score=round(score, 4), line=line.rstrip("\n"), index=index, created=created,
                         kind=cells[5], lane="single" if cells[5] == "single" else "fleet",
                         small=estimate <= SMALL_MAX_MIN))
    # the batch: each lane's small tickets, oldest first, as far as the cap lets them pass the larger ones
    batch = set()
    for lane in ("single", "fleet"):
        total = 0.0
        for r in sorted((r for r in rows if r["lane"] == lane and r["small"]), key=lambda r: (r["created"], r["index"])):
            if total + r["estimate_min"] > BATCH_CAP_MIN:
                break
            total += r["estimate_min"]
            batch.add(r["session"])
    for r in rows:
        session, age = r["session"], r["age_s"]
        if session == yielded:
            key = (0, 0)
        elif session == front:
            key = (1, 0)
        elif age >= STARVE_S:
            key = (2, r["created"])          # starved: only front and yield pass it now
        elif session in batch:
            key = (3, r["created"])          # the small batch, oldest first
        elif age >= AGED_S:
            key = (4, r["created"])          # aged: oldest first among the rest
        else:
            key = (5, -r["score"])
        if not probes_ready and r["kind"] == "probe":
            key = (6, r["index"])  # preserve safety and admit an eligible boot
        r["key"] = (*key, r["index"])
        r["batch"] = session in batch
        for drop in ("index", "created", "kind", "small"):
            r.pop(drop)
    return sorted(rows, key=lambda r: r["key"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory", type=Path)
    ap.add_argument("--apply", action="store_true", help="caller must hold fleet .lock and check holder first")
    ap.add_argument("--boot-only", action="store_true", help="serving is not idle; probes are temporarily ineligible")
    args = ap.parse_args()
    directory = args.directory
    def marker(name):
        path = directory / name
        return path.read_text().strip() if path.exists() else ""
    queue = directory / "queue"
    db = Path(os.environ.get('FLEET_EXPERIMENT_ROOT',directory/'experiments'))/'experiments.sqlite3'
    from experiment_metrics import estimates
    from fleet_handoff import identity
    lines = queue.read_text().splitlines()
    if Path('/proc').exists():
        lines = [line for line in lines if len(line.split('|')) < 7 or not line.split('|')[6]
                 or identity(int(line.split('|')[6]))]
    from fleet_pause import paused
    paused_lines = [line for line in lines if paused(directory, line.split('|')[1], line.split('|'))]
    if args.apply and paused_lines:
        from fleet_pause import reconcile
        for line in paused_lines:
            reconcile(directory, line.split('|')[1])
    lines = [line for line in lines if line not in paused_lines]
    # the experiment DB's predictions first, the ledger's history for every other ticket
    predicted = history_estimates(directory / "ledger.tsv", [line.split("|")[1] for line in lines])
    predicted.update(estimates(db))
    rows = rank(lines, downstream(db),
                time.time(), marker("priority-front"), marker("priority-yield"), not args.boot_only, predicted)
    # Recovery is central and begins only after five idle minutes. Legacy
    # restore debt must not change readiness or reorder runnable experiments.
    if args.apply:
        temporary = queue.with_suffix(".priority.tmp")
        temporary.write_text("".join(r["line"] + "\n" for r in rows))
        temporary.replace(queue)
    else:
        from fleet_pause import parked
        answer = [{k: v for k, v in r.items() if k not in {"line", "key", "lane"}} for r in rows]
        answer.extend(dict(session=v['session'], state='paused') for v in parked(directory))
        print(json.dumps(answer))


if __name__ == "__main__":
    main()
