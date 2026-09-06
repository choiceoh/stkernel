#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Does a baseline already exist for the build now deployed?

The fleet's scarce resource is a boot, and the most-repeated boot is the one
that re-measures the profile defaults on a build somebody already measured
(39차, operator: "기존 빌드의 기준점이 될만한 측정이 이미 있으면 그걸 알려주는
장치"). bench/fleet.sh calls this when a session takes a turn, so the answer
arrives before the boot is spent, not after.

A record is a BASELINE of its build when onepass recorded no knob that differs
from the profile (`knobs: {}`), and the build is the deployed overlay stamp --
not the git sha, because a bench that skipped the deploy runs the previous
build whatever the checkout says. Records written before onepass carried those
fields fall back to the git sha and a name heuristic, and say so.

    python3 bench/baseline.py                 # for the build deployed here
    python3 bench/baseline.py --brief         # one line, for fleet.sh
    python3 bench/baseline.py --build <sha>   # for a named build
"""
from __future__ import annotations

import argparse
import json
import os

JSONL = os.environ.get("ONEPASS_JSONL",
                       "/home/choiceoh/glm53-logs/bracket-onepass.jsonl")
STAMP = os.environ.get("MK_OVERLAY_STAMP",
                       "/home/choiceoh/glm53-cache/.overlay-sha")
LEGACY_BASE_PREFIXES = ("BASE", "PROD")   # the naming convention before `knobs`


def deployed_build() -> str | None:
    try:
        with open(STAMP) as fh:
            return fh.read().strip()[:12]
    except Exception:
        return None


def deployed_git(repo: str | None = None) -> str | None:
    """The checkout's sha -- the fallback identity for records written before
    onepass carried the overlay stamp."""
    import subprocess
    cands = [repo] if repo else [
        os.environ.get("MK_REPO"),
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        os.getcwd(),                     # the tool may be run from a copy (/tmp)
    ]
    for c in [c for c in cands if c]:
        try:
            r = subprocess.run(["git", "-C", c, "rev-parse", "--short", "HEAD"],
                               capture_output=True, text=True, timeout=10)
            sha = (r.stdout or "").strip()
            if sha:
                return sha
        except Exception:
            pass
    return None


def load(path: str = JSONL) -> list[dict]:
    try:
        with open(path) as fh:
            return [json.loads(l) for l in fh if l.strip()]
    except Exception:
        return []


BUILDS = os.environ.get("FLEET_BUILDS", "/home/choiceoh/glm53-logs/fleet/builds.tsv")


def builds() -> list[tuple[str, str, str, str]]:
    """(ts, stamp, sha, session) rows written by `fleet.sh deploy`."""
    try:
        with open(BUILDS) as fh:
            return [tuple(l.rstrip("\n").split("\t"))[:4] for l in fh if l.strip() and not l.startswith("#")]
    except Exception:
        return []


def sha_of_stamp(stamp: str | None) -> str | None:
    for _, st, sha, _ in reversed(builds()):
        if stamp and st.startswith(stamp[:12]):
            return sha
    return None


def is_baseline(rec: dict) -> tuple[bool, str]:
    """(baseline?, how we know). `knobs` is authoritative; the name is a guess.
    A rehearsal row (ab-lever under FLEET_REHEARSE=1) is never a baseline."""
    if rec.get("rehearsal"):
        return (False, "rehearsal")
    if "knobs" in rec:
        return (not rec["knobs"], "knobs")
    name = (rec.get("name") or "").upper()
    return (name.startswith(LEGACY_BASE_PREFIXES), "name")


def for_build(rows: list[dict], build: str | None, git: str | None = None) -> list[dict]:
    """Records of that build: by overlay stamp when they carry one, else by the
    checkout's git sha (rows written before the stamp field existed)."""
    out = [r for r in rows if build and r.get("overlay") == build]
    if out:
        return out
    git = git or sha_of_stamp(build)          # a deploy registry row names the sha
    if not git:
        return out
    return [r for r in rows if (r.get("git") or "").startswith(git[:7])]


def line(rec: dict) -> str:
    d = rec.get("decode") or {}
    q = rec.get("quality") or {}
    k = rec.get("korean") or {}
    return (f"{rec.get('name','?'):<14}{d.get('windows_med') or 0:7.2f} step/s"
            f"{d.get('tokens_per_step') or 0:7.2f} tok/step"
            f"{100 * (d.get('acc_raw') or 0):7.1f}% acc"
            f"   quality {q.get('ok','?')}/{q.get('total','?')}"
            f"   korean {k.get('dirty','?')}/{k.get('n','?')}"
            f"   {rec.get('t','')}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default=None, help="overlay stamp (default: the deployed one)")
    ap.add_argument("--brief", action="store_true", help="one line for fleet.sh")
    ap.add_argument("--jsonl", default=JSONL)
    ap.add_argument("--count-for", help="count usable baselines compatible with this candidate's workload/runtime")
    ap.add_argument("--knobs", default=None,
                    help="K=V,K=V of the arm you are about to boot: report a record with the SAME set on this build")
    args = ap.parse_args()

    if args.count_for:
        from judge import baselines_on, record_errors
        rows = load(args.jsonl)
        cand = next((r for r in reversed(rows) if r.get("name") == args.count_for and not r.get("rehearsal")), None)
        bases = baselines_on(rows, cand)[0] if cand else []
        print(sum(not record_errors(r) for r in bases))
        return 0

    build = args.build or deployed_build()
    git = args.build or deployed_git()      # an explicit --build names the build, full stop
    rows = load(args.jsonl)
    mine = for_build(rows, build, git)
    label = build if any(r.get("overlay") for r in mine) else (git or build)
    bases = [r for r in mine if is_baseline(r)[0]]
    guessed = bases and all(is_baseline(r)[1] == "name" for r in bases)

    if args.knobs:
        want = dict(kv.split("=", 1) for kv in args.knobs.split(",") if "=" in kv)
        want = {k: v for k, v in want.items() if v not in ("0", "", "off")}
        same = [r for r in mine if "knobs" in r and
                {k: v for k, v in (r["knobs"] or {}).items() if v not in ("0", "", "off")} == want]
        if same:
            r = same[-1]; d = r.get("decode") or {}
            print(f"already measured on this build: {r.get('name')} at {r.get('t','')} -- "
                  f"{d.get('windows_med') or 0:.2f} step/s, proof {r.get('proof_ok','?')}"
                  f" -- a repeat only buys a second sample")
        elif want:
            print(f"this arm ({','.join(sorted(want))}) has no record on build {label}")
    if args.brief:
        if not build and not git:
            print("baseline: unknown build (no overlay stamp and no checkout here)")
        elif not mine:
            print(f"baseline: NONE for build {label} -- a defaults arm is worth its boot")
        elif not bases:
            print(f"baseline: none for build {label}; {len(mine)} candidate run(s) on it")
        else:
            b = bases[-1]
            d = b.get("decode") or {}
            how = " (by name, pre-`knobs` record)" if guessed else ""
            print(f"baseline: {b.get('name')} on build {label} -- "
                  f"{d.get('windows_med') or 0:.2f} step/s, "
                  f"{d.get('tokens_per_step') or 0:.2f} tok/step, "
                  f"{100 * (d.get('acc_raw') or 0):.1f}% acc, {b.get('t','')}{how}"
                  f" -- skip a defaults arm and compare against it")
        return 0

    sha = sha_of_stamp(build)
    print(f"deployed build: {label or 'unknown'}{' = ' + sha if sha and sha != label else ''}   records on it: {len(mine)}")
    if not mine:
        print("no measurement of this build yet: the first arm pays for its own baseline")
        return 0
    print("\nbaselines (profile defaults):")
    for r in bases or []:
        print("  " + line(r))
    if not bases:
        print("  (none)")
    others = [r for r in mine if r not in bases]
    if others:
        print("\ncandidate arms on the same build:")
        for r in others:
            knobs = r.get("knobs")
            tag = ",".join(sorted(knobs)) if knobs else "knobs not recorded"
            print("  " + line(r))
            print(f"      {tag}")
    if guessed:
        print("\nNOTE: these predate onepass's `knobs` field, so 'baseline' is a "
              "guess from the name. Re-measure if the decision is close.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
