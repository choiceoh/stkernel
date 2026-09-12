#!/usr/bin/env python3
"""Did this branch break anything? Answered as a difference, because the absolute number lies.

A suite that has files which cannot run on the box you are on (no GPU, a missing wheel) makes "N failures" useless:
you cannot tell your N from the N that was already there. So this checks out a reference -- `origin/main` by default --
beside the working tree, runs `tools/check.py` in both, and prints ONLY what changed.

Usage
    python3 tools/regress.py                 # against origin/main
    python3 tools/regress.py --ref HEAD~1
    python3 tools/regress.py --pattern 'test_*'

Exit 0 when nothing got worse. A file that goes ok -> FAILED, or ok -> CANNOT RUN, is worse; the reverse is not.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
RANK = {"ok": 0, "CANNOT RUN": 1, "FAILED": 2}


def verdicts(tree: pathlib.Path, pattern: str, jobs: int, gpu: bool) -> "dict[str, str]":
    cmd = [sys.executable, str(tree / "tools" / "check.py"), "--pattern", pattern, "--jobs", str(jobs), "--list"]
    if gpu:
        cmd.append("--gpu")
    out = subprocess.run(cmd, cwd=tree, capture_output=True, text=True).stdout
    found = {}
    for line in out.splitlines():
        for state in ("CANNOT RUN", "FAILED", "ok"):                 # longest first: "ok" is a prefix of nothing here
            head = line.strip()
            if head.startswith(state):
                found[head[len(state):].split()[0]] = state
                break
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", default="origin/main", help="what to compare against (default origin/main)")
    ap.add_argument("--pattern", default="test_engine_*")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--json", action="store_true", help="machine-readable difference")
    a = ap.parse_args()

    subprocess.run(["git", "fetch", "origin", "--quiet"], cwd=ROOT, check=False)
    with tempfile.TemporaryDirectory(prefix="regress-") as tmp:
        base = pathlib.Path(tmp)
        archive = subprocess.run(["git", "archive", a.ref], cwd=ROOT, capture_output=True, check=True).stdout
        subprocess.run(["tar", "-x", "-C", str(base)], input=archive, check=True)
        if not (base / "tools" / "check.py").exists():                # the reference predates this tool: lend it ours
            (base / "tools").mkdir(exist_ok=True)
            (base / "tools" / "check.py").write_bytes((ROOT / "tools" / "check.py").read_bytes())
            (base / "tests" / "__init__.py").write_bytes((ROOT / "tests" / "__init__.py").read_bytes())
        print(f"  running both trees ({a.ref} and the working tree) ...", flush=True)
        was, now = verdicts(base, a.pattern, a.jobs, a.gpu), verdicts(ROOT, a.pattern, a.jobs, a.gpu)

    worse, better, new, gone = [], [], [], []
    for module in sorted(set(was) | set(now)):
        before, after = was.get(module), now.get(module)
        if before is None:
            new.append((module, after))
        elif after is None:
            gone.append((module, before))
        elif before != after:
            (worse if RANK[after] > RANK[before] else better).append((module, before, after))

    if a.json:
        print(json.dumps({"worse": worse, "better": better, "new": new, "gone": gone}, indent=2))
    else:
        for module, before, after in worse:
            print(f"  WORSE   {module}: {before} -> {after}")
        for module, before, after in better:
            print(f"  better  {module}: {before} -> {after}")
        for module, state in new:
            print(f"  new     {module}: {state}")
        for module, state in gone:
            print(f"  gone    {module} (was {state})")
        if not (worse or better or new or gone):
            print(f"  no difference against {a.ref}: {len(now)} files judged the same")
    return 1 if worse else 0


if __name__ == "__main__":
    sys.exit(main())
