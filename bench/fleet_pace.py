#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The queue's pace, read off its own log: how soon after one ticket the next one tends to come.

Production is restored only after the queue has been quiet for a grace; a grace that is a constant
(300 s) restores production into the next ticket's face on a busy night -- 2026-09-13: the next
boot request came a median 6.3 min after the last ticket ended, 34 of 42 within 15 min, and 17 of
46 production boots were followed by a ticket within 15 min. So the grace follows the record: the
75th percentile of the last hours' gaps, clamped to [FLOOR, CEILING]. A session that knows it is in
a campaign says so (`fleet.sh window s MINUTES`) and the grace is at least that long while the
window is open. Both the production supervisor and deploy-watch read what this writes.

    fleet_pace.py grace  DIR             recompute DIR/restore-grace.json from DIR/log and print it
    fleet_pace.py window DIR s MINUTES   open a campaign window for session s (MINUTES from now)
    fleet_pace.py window DIR s off       close it
    fleet_pace.py show   DIR             the effective grace now: adaptive, window, floor
"""
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

FLOOR = int(os.environ.get("FLEET_GRACE_FLOOR_S", 300))       # never less than the old constant
CEILING = int(os.environ.get("FLEET_GRACE_CEILING_S", 1200))  # twenty minutes: beyond that production has waited long enough
HOURS = float(os.environ.get("FLEET_GRACE_HOURS", 6))         # the record the percentile is taken over
PERCENTILE = 0.75
LINE = re.compile(r"^(\d{4}-\d\d-\d\d_\d\d:\d\d:\d\d) (request (\S+) est=\d+m .*\[(boot|probe)\]|release (\S+)|cancel (\S+))\s*$")


def parse(text: str, now: float, hours: float = HOURS):
    """(release times, request times) of boot tickets in the last `hours`, as unix seconds."""
    since = now - hours * 3600
    ends, reqs = [], []
    for line in text.splitlines():
        m = LINE.match(line)
        if not m:
            continue
        t = datetime.strptime(m.group(1), "%Y-%m-%d_%H:%M:%S").timestamp()
        if t < since:
            continue
        if m.group(3) and m.group(4) == "boot":
            reqs.append(t)
        elif m.group(5):
            ends.append(t)
    return ends, reqs


def gaps(ends, reqs):
    """Seconds from each ticket's end to the next boot request after it."""
    reqs = sorted(reqs)
    out = []
    for e in ends:
        nxt = next((r for r in reqs if r > e), None)
        if nxt is not None:
            out.append(nxt - e)
    return out


def percentile(values, p):
    values = sorted(values)
    if not values:
        return None
    return values[min(len(values) - 1, int(p * (len(values) - 1)))]


def adaptive(text: str, now: float) -> dict:
    ends, reqs = parse(text, now)
    g = gaps(ends, reqs)
    p = percentile(g, PERCENTILE)
    seconds = FLOOR if p is None else int(min(CEILING, max(FLOOR, p)))
    return dict(seconds=seconds, basis=dict(gaps=len(g), p75_s=None if p is None else int(p), hours=HOURS,
                                             floor_s=FLOOR, ceiling_s=CEILING), at=now)


def write_grace(directory: Path, now=None) -> dict:
    now = time.time() if now is None else now
    try:
        text = (directory / "log").read_text(errors="replace")
    except OSError:
        text = ""
    value = adaptive(text, now)
    tmp = directory / ".restore-grace.tmp"
    tmp.write_text(json.dumps(value, sort_keys=True) + "\n")
    tmp.replace(directory / "restore-grace.json")
    return value


def window(directory: Path, session: str, minutes, now=None) -> dict:
    now = time.time() if now is None else now
    path = directory / "window.json"
    if minutes in ("off", 0, "0"):
        path.unlink(missing_ok=True)
        return dict(session=session, until=None)
    minutes = int(minutes)
    if minutes < 1 or minutes > 240:
        raise ValueError("a window is 1..240 minutes")
    value = dict(session=session, until=now + 60 * minutes, opened=now, minutes=minutes)
    tmp = directory / ".window.tmp"
    tmp.write_text(json.dumps(value, sort_keys=True) + "\n")
    tmp.replace(path)
    return value


def effective(directory: Path, now=None) -> dict:
    """What a restorer should wait for: the adaptive grace, or the open window's remainder if longer."""
    now = time.time() if now is None else now
    try:
        grace = json.loads((directory / "restore-grace.json").read_text()).get("seconds", FLOOR)
    except (OSError, ValueError):
        grace = FLOOR
    try:
        w = json.loads((directory / "window.json").read_text())
        left = int(w.get("until", 0) - now)
        window_s, session = (left, w.get("session")) if left > 0 else (0, None)
    except (OSError, ValueError):
        window_s, session = 0, None
    return dict(grace_s=int(max(grace, window_s)), adaptive_s=int(grace), window_s=window_s, window_session=session)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 2 or argv[0] not in ("grace", "window", "show"):
        print(__doc__, file=sys.stderr)
        return 2
    directory = Path(argv[1])
    try:
        if argv[0] == "grace":
            print(json.dumps(write_grace(directory), sort_keys=True))
        elif argv[0] == "window":
            if len(argv) < 4:
                print("window DIR s MINUTES|off", file=sys.stderr)
                return 2
            print(json.dumps(window(directory, argv[2], argv[3]), sort_keys=True))
        else:
            print(json.dumps(effective(directory), sort_keys=True))
    except (OSError, ValueError) as exc:
        print("ABORT: " + str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
