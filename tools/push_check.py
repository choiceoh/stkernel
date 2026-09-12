#!/usr/bin/env python3
"""Is it still worth pushing this branch? Run it before you push, not after you wonder.

A branch whose pull request is already MERGED still accepts pushes. Nothing fails, nothing warns, and the commit
simply never reaches main. It happened twice on 2026-09-12 -- PR #633 and again on PR #694, the second time with
a note in memory saying not to -- so it is a check and not a thing to remember.

    python3 tools/push_check.py                 # the current branch
    python3 tools/push_check.py --branch foo

Exit 1 when pushing would be pointless or when the branch has nothing main does not already have.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--branch", default=None)
    ap.add_argument("--base", default="origin/main")
    a = ap.parse_args()
    branch = a.branch or git("branch", "--show-current")
    if not branch:
        print("  detached head: nothing to push")
        return 1
    subprocess.run(["git", "fetch", "origin", "--quiet"], check=False)

    out = subprocess.run(["gh", "pr", "list", "--head", branch, "--state", "all", "--limit", "5",
                          "--json", "number,state,mergedAt"], capture_output=True, text=True)
    prs = json.loads(out.stdout or "[]")
    ahead = int(git("rev-list", "--count", f"{a.base}..{branch}") or 0)
    behind = int(git("rev-list", "--count", f"{branch}..{a.base}") or 0)
    print(f"  {branch}: {ahead} commit(s) {a.base} does not have, {behind} it has that you do not")
    code, lines = verdict(prs, ahead, behind, base=a.base, tip=git("rev-parse", "--short", branch))
    for line in lines:
        print("  " + line)
    return code


def verdict(prs, ahead: int, behind: int, base: str = "origin/main", tip: str = "<sha>"):
    """(exit code, lines). Pure, so the case that caused this can be a test and not a story."""
    merged = [p for p in prs if p["state"] == "MERGED"]
    open_ = [p for p in prs if p["state"] == "OPEN"]
    if merged and not open_:
        numbers = ", ".join(f"#{p['number']} (merged {p.get('mergedAt', '')[:16].replace('T', ' ')})" for p in merged)
        return 1, [f"STOP. This branch's pull request is already merged: {numbers}",
                   "A push lands nowhere. Branch from the base and cherry-pick:",
                   f"    git checkout -B <new-branch> {base} && git cherry-pick {tip}"]
    if ahead == 0:
        return 1, [f"nothing to push: {base} already has everything on this branch"]
    out = ([f"open pull request: {', '.join('#' + str(p['number']) for p in open_)} -- a push updates it"]
           if open_ else ["no pull request yet -- a push will need `gh pr create`"])
    if behind:
        out.append(f"note: {base} has moved {behind} commit(s) ahead; rebase before asking for a merge")
    return 0, out


if __name__ == "__main__":
    sys.exit(main())
