#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prune the queue's own debris. Nothing else does, and nothing else should.

`fleet_observation_cleanup` removes a holder's diagnostic clones; this removes what the
QUEUE leaves behind: a heartbeat per session ever seen, a launch record per detached run,
a preparation per prepared input, and a pinned runner per distinct bench/probe source.
Six days of ordinary use left 97 heartbeats, 303 launches, 181 preparations and 62 pinned
runners taking 62 MB (2026-09-12 census).

Two rules keep this safe to run while the queue is live:
  * anything a live session could still be using is never touched -- the current holder,
    every queued and paused session, and every runner those name;
  * everything else must also be OLDER than `--days`, so a race with a session that has
    just enqueued cannot lose its files.
The default is a dry run: it prints what it would remove and removes nothing.
"""
import argparse
import json
import shutil
import time
from pathlib import Path

KEEP_DAYS = 14
GROUPS = ("hb.*", "launches/*", "preparations/*", "runners/*", "run-logs/*", "validation/*")


def live_sessions(directory: Path) -> set:
    """Sessions the queue still knows about: the holder, the queue, and anything paused."""
    names = set()
    holder = directory / "holder"
    if holder.is_file():
        line = holder.read_text().strip()
        if line:
            names.add(line.split("|")[0])
    queue = directory / "queue"
    if queue.is_file():
        for line in queue.read_text().splitlines():
            parts = line.split("|")
            if len(parts) > 1:
                names.add(parts[1])
    for path in (directory / "pending").glob("*"):
        names.add(path.stem)
    return {n for n in names if n}


def live_runners(directory: Path) -> set:
    """Runner keys named by anything still in flight; a pinned runner outlives its ticket."""
    keys = set()
    for path in list((directory / "launches").glob("*")) + list((directory / "pending").glob("*")):
        try:
            text = path.read_text()
        except OSError:
            continue
        for token in text.replace('"', " ").replace("/", " ").split():
            if len(token) == 64 and all(c in "0123456789abcdef" for c in token):
                keys.add(token)
    return keys


def plan(directory: Path, days: int = KEEP_DAYS):
    """(path, reason) for everything prunable, oldest first. Pure: it removes nothing."""
    cutoff = time.time() - days * 86400
    sessions, runners = live_sessions(directory), live_runners(directory)
    out = []
    for pattern in GROUPS:
        for path in sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime if p.exists() else 0):
            try:
                age = path.stat().st_mtime
            except OSError:
                continue
            if age > cutoff:
                continue
            name = path.name
            if pattern == "hb.*" and name[3:] in sessions:
                continue
            if pattern == "runners/*" and name in runners:
                continue
            if any(name.startswith(s) or s in name for s in sessions):
                continue
            out.append((path, pattern.split("/")[0].rstrip(".*")))
    return out


def prune(directory: Path, days: int = KEEP_DAYS, apply: bool = False):
    removed, freed = [], 0
    for path, group in plan(directory, days):
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.is_dir() else (
            path.stat().st_size if path.is_file() else 0)
        if apply:
            try:
                shutil.rmtree(path) if path.is_dir() else path.unlink()
            except OSError:
                continue
        removed.append((str(path), group, size))
        freed += size
    return dict(removed=len(removed), freed_bytes=freed, applied=apply,
                groups=sorted({g for _, g, _ in removed}))


def _selfcheck() -> None:
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        for sub in ("launches", "preparations", "runners", "pending", "run-logs", "validation"):
            (d / sub).mkdir()
        (d / "holder").write_text("live-session|123|srv2|0|30|note|boot\n")
        (d / "queue").write_text("1|queued-session|123|30|note|boot\n")
        old = time.time() - 30 * 86400
        for name in ("hb.live-session", "hb.queued-session", "hb.ancient"):
            (d / name).write_text("x"); 
        (d / "launches/ancient.json").write_text("{}")
        (d / "runners" / ("a" * 64)).mkdir()
        (d / "launches/keeps-runner.json").write_text(json.dumps({"runner": "b" * 64}))
        (d / "runners" / ("b" * 64)).mkdir()
        for path in d.rglob("*"):
            import os
            os.utime(path, (old, old))
        names = {p.name for p, _ in plan(d)}
        assert "hb.ancient" in names and "ancient.json" in names, names
        assert "hb.live-session" not in names and "hb.queued-session" not in names, names
        assert "a" * 64 in names and "b" * 64 not in names, "a runner a launch still names is kept"
        before = len(list(d.rglob("*")))
        assert prune(d, apply=False)["removed"] and len(list(d.rglob("*"))) == before, "a dry run removes nothing"
        report = prune(d, apply=True)
        assert report["applied"] and not (d / "hb.ancient").exists() and (d / "hb.live-session").exists()
        assert (d / "runners" / ("b" * 64)).exists() and not (d / "runners" / ("a" * 64)).exists()
    print("  fleet_prune: keeps the holder, the queue, the paused and every runner they name; "
          "removes only what is both unreferenced and old; dry by default OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", default="/home/choiceoh/glm53-logs/fleet")
    parser.add_argument("--days", type=int, default=KEEP_DAYS)
    parser.add_argument("--apply", action="store_true", help="actually remove (default: report only)")
    parser.add_argument("--selfcheck", action="store_true")
    a = parser.parse_args()
    if a.selfcheck:
        _selfcheck()
    else:
        print(json.dumps(prune(Path(a.directory), a.days, a.apply), indent=2))
