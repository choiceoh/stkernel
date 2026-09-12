"""A release directory for one commit: the tree the ST engine mounts at /repo, cut from git, never a
working tree that may be mid-edit.

Shared by launchers/st-deploy-watch.py (main, deployed when the engine is quiet) and
bench/st_bracket.sh (a ticket's arm: one committed sha). One shape of release, so a candidate
that wins is promoted by pointing production at the very directory the bracket booted -- no
second cut, no "the tree on the nodes is no longer known".

    python3 launchers/st_release.py cut <sha> [--source DIR] [--releases DIR]   # prints the directory
    python3 launchers/st_release.py resolve <sha> [--source DIR]                # the full sha (fetched if absent)
    python3 launchers/st_release.py deployed [--state FILE]                     # the sha production runs

stdlib only, like base/fleet_lease.py: it runs on the head node outside any virtualenv.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

HOME = Path(os.environ.get("ST_HOME", Path.home()))
SOURCE = Path(os.environ.get("ST_SOURCE", HOME / "stkernel"))          # where commits are archived from
RELEASES = Path(os.environ.get("ST_RELEASES", HOME / "st-releases"))
STATE = Path(os.environ.get("ST_DEPLOY_STATE", RELEASES / "deploy-state.json"))
META = HOME / "st-engine" / "st-glm53-meta"                              # the chat templates and tokenizer the door needs
CARRY = ("engine", "launchers", "tests", "probes", "build")   # what a release has to hold to boot and be judged
SHA = re.compile(r"[0-9a-f]{7,40}")


def run(cmd, cwd=None, timeout=1800, env=None):
    out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout,
                         env={**os.environ, **(env or {})})
    return out.returncode, out.stdout, out.stderr


def resolve(sha: str, *, source: Path = SOURCE) -> str:
    """The full commit id for `sha`, fetching it from origin when the source tree has not seen it.

    An arm is a commit origin has: a bracket measures what can be cited, not a working tree.
    """
    if not SHA.fullmatch(sha):
        raise ValueError(f"not a commit id: {sha!r}")
    code, out, _ = run(["git", "-C", str(source), "rev-parse", "--verify", "--quiet", sha + "^{commit}"], timeout=60)
    if code:
        run(["git", "-C", str(source), "fetch", "--quiet", "origin", sha], timeout=600)
        code, out, _ = run(["git", "-C", str(source), "rev-parse", "--verify", "--quiet", sha + "^{commit}"], timeout=60)
        if code:
            raise ValueError(f"{sha} is not a commit {source} or origin has")
    return out.strip()


def cut(sha: str, *, source: Path = SOURCE, releases: Path = RELEASES, log=print, meta: Path = META) -> "Path | None":
    """A release directory for `sha`: a checkout, not a copy of a working tree that may be mid-edit."""
    target = Path(releases) / sha[:12]
    if target.exists():
        log(f"  release {target} is already cut")
        return target
    Path(releases).mkdir(parents=True, exist_ok=True)
    staging = target.with_suffix(".partial")
    run(["rm", "-rf", str(staging)])
    staging.mkdir(parents=True)
    for part in CARRY:
        # pipefail: without it the exit code is tar's, and tar is happy to extract the prefix of a
        # stream that died halfway -- a release that looks complete and is not
        code, _, err = run(["bash", "-c", "set -o pipefail; "
                            f"git -C {source} archive {sha} {part} | tar -x -C {staging}"])
        if code and part != "build":                      # `build` is the tokenizer meta: not in git on every tree
            log(f"  ABORT: {sha[:12]} has no {part} ({err.strip()[:80]})")
            run(["rm", "-rf", str(staging)])
            return None
    if meta and Path(meta).is_dir():                      # the chat templates and tokenizer the door needs
        (staging / "build").mkdir(parents=True, exist_ok=True)
        run(["rsync", "-a", f"{meta}/", str(staging / "build" / "st-glm53-meta") + "/"])
    staging.rename(target)
    log(f"  cut {target}")
    return target


def deployed(state: Path = STATE) -> str:
    """The sha deploy-watch recorded as running, or '' when nothing is recorded."""
    try:
        return str(json.loads(Path(state).read_text()).get("deployed") or "")
    except (OSError, ValueError):
        return ""


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("action", choices=("cut", "resolve", "deployed"))
    ap.add_argument("sha", nargs="?", default="")
    ap.add_argument("--source", default=str(SOURCE))
    ap.add_argument("--releases", default=str(RELEASES))
    ap.add_argument("--state", default=str(STATE))
    a = ap.parse_args(argv)
    try:
        if a.action == "deployed":
            sha = deployed(Path(a.state))
            if not sha:
                print(f"nothing recorded as deployed in {a.state}", file=sys.stderr)
                return 1
            print(sha)
            return 0
        full = resolve(a.sha, source=Path(a.source))
        if a.action == "resolve":
            print(full)
            return 0
        target = cut(full, source=Path(a.source), releases=Path(a.releases), log=lambda m: print(m, file=sys.stderr))
        if target is None:
            return 1
        print(target)
        return 0
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
