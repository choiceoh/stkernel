#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Rank waiting jobs at a free fleet boundary; never preempt a holder.

Oldest first after 30 minutes. Otherwise unblock more experiments per estimated
minute, with gradual aging. Explicit front and a yielded probe retain priority.
Failures to read the experiment DB simply give every job zero dependents.
"""
import argparse
import json
from pathlib import Path
import sqlite3
import time

TERMINAL = {"succeeded", "failed", "blocked", "incomplete", "interrupted"}


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


def rank(lines, counts, now, front="", yielded="", probes_ready=True):
    rows = []
    for index, line in enumerate(lines):
        cells = line.rstrip("\n").split("|")
        if len(cells) < 6:
            raise ValueError("malformed queue row")
        session, created, estimate = cells[1], float(cells[2]), max(1, float(cells[3]))
        age = max(0, now - created)
        dependents = counts.get(session, 0)
        score = (1 + dependents) / estimate + age / 1800
        key = ((0, 0) if session == yielded else (1, 0) if session == front
               else (2, created) if age >= 1800 else (3, -score))
        if not probes_ready and cells[5] == "probe":
            key = (4, index)  # preserve safety and admit an eligible boot
        rows.append(dict(session=session, age_s=round(age), dependents=dependents,
                         score=round(score, 4), line=line.rstrip("\n"), key=(*key, index)))
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
    rows = rank(queue.read_text().splitlines(), downstream(directory / "experiments/experiments.sqlite3"),
                time.time(), marker("priority-front"), marker("priority-yield"), not args.boot_only)
    if args.apply:
        temporary = queue.with_suffix(".priority.tmp")
        temporary.write_text("".join(r["line"] + "\n" for r in rows))
        temporary.replace(queue)
    else:
        print(json.dumps([{k: v for k, v in r.items() if k not in {"line", "key"}} for r in rows]))


if __name__ == "__main__":
    main()
