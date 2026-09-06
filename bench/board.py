#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The board: every session's recent verdicts, today's boot accounting, and
what the fleet is doing -- one screen for "어떻게 되고 있어" (39차, operator "5").

    python3 bench/board.py [--n 12] [--days 1]
    fleet.sh board [n]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os

LOGD = os.environ.get("LOGD", "/home/choiceoh/glm53-logs")
FLEET_DIR = os.environ.get("FLEET_DIR", os.path.join(LOGD, "fleet"))
VERDICTS = os.environ.get("ONEPASS_VERDICTS", os.path.join(LOGD, "verdicts.jsonl"))
LEDGER = os.path.join(FLEET_DIR, "ledger.tsv")


def rows(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return [json.loads(l) for l in fh if l.strip()]
    except Exception:
        return []


def fleet_lines():
    out = []
    h = os.path.join(FLEET_DIR, "holder")
    q = os.path.join(FLEET_DIR, "queue")
    try:
        with open(h) as fh:
            s, pid, host, t0, est, note, *rest = fh.read().strip().split("|") + [""]
            since = dt.datetime.fromtimestamp(int(t0)).strftime("%H:%M")
            out.append(f"holder: {s} since {since} est {est}m  {note}")
    except Exception:
        out.append("holder: none")
    try:
        with open(q) as fh:
            for i, line in enumerate(l for l in fh if l.strip()):
                p = line.rstrip("\n").split("|")
                kind = p[5] if len(p) > 5 and p[5] else "boot"
                out.append(f"  {i + 1}. {p[1]} [{kind}] est {p[3]}m  {p[4]}")
    except Exception:
        pass
    return out


def ledger_today(days):
    since = (dt.date.today() - dt.timedelta(days=days - 1)).isoformat()
    agg = {}
    try:
        with open(LEDGER, encoding="utf-8") as fh:
            for line in fh:
                p = line.rstrip("\n").split("\t")
                if len(p) < 8 or p[0][:10] < since:
                    continue
                a = agg.setdefault(p[1], [0, 0, 0, 0, 0])
                a[0] += 1
                for i, j in ((1, 4), (2, 5), (3, 6), (4, 7)):
                    try:
                        a[i] += int(p[j])
                    except ValueError:
                        pass
    except Exception:
        pass
    return agg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--days", type=int, default=1)
    a = ap.parse_args()
    print("fleet:")
    for l in fleet_lines():
        print("  " + l)
    vs = rows(VERDICTS)[-a.n:]
    print(f"\nverdicts (last {len(vs)}):")
    print(f"  {'when':<12}{'session':<10}{'arm -> base':<26}{'build':<13}{'delta':>7}{'floor':>7}{'proof':>6}  verdict")
    for v in vs:
        d = v.get("delta")
        f = v.get("floor")
        print(f"  {(v.get('t') or '')[5:16]:<12}{(v.get('session') or '-'):<10}"
              f"{(v.get('cand') or '?') + ' -> ' + (v.get('base') or '-'):<26}{(v.get('build') or '-'):<13}"
              f"{(f'{100 * d:+.1f}%' if d is not None else '-'):>7}{(f'±{100 * f:.1f}%' if f else '-'):>7}"
              f"{(v.get('proof_ok') or '-'):>6}  {(v.get('verdict') or '')[:60]}")
    if not vs:
        print("  (none yet)")
    agg = ledger_today(a.days)
    print(f"\nboots (last {a.days} day{'s' if a.days > 1 else ''}):")
    print(f"  {'session':<12}{'holds':>6}{'min':>6}{'boots':>7}{'records':>9}{'wasted':>8}")
    for s, (h, m, b, r, w) in sorted(agg.items()):
        print(f"  {s:<12}{h:>6}{m:>6}{b:>7}{r:>9}{w:>8}")
    if not agg:
        print("  (no holds recorded)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
