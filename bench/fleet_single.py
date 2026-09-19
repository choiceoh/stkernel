#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The single-GPU lane: a check that needs one GPU runs on ONE Spark beside production, not on the fleet.

The queue (bench/fleet.sh) has two lanes. `boot` and `probe` take the fleet -- four Sparks,
one holder file. `single` takes ONE GPU and has its own holder (`holder-single`). Where that
GPU is, is FLEET_SINGLE_GPU_HOST: by default srv4, a Spark that serves production at the
same time. So the lane's evidence is not "is the GPU free" -- beside production it never is --
but "is there ROOM beside production": the box's MemAvailable, less this check's budget, must
stay above the floor a --test boot keeps (engine/profiles/glm53/boot.py TEST_FLOOR_GIB). GB10
has one pool for host and device, earlyoom's floor is absolute and the engine is a preferred
kill target: on 2026-09-11 a smoke test beside production killed the fleet's worker, not
itself. One probe container per box at a time, and a host that cannot answer is not free (D3).

The same rule serves a box of its own (ost-97x, the operator's Windows PC on the tailnet,
once it has sshd and an x86_64 image): point FLEET_SINGLE_GPU_HOST at its ssh alias. The
controller's ~/.ssh/config owns the alias -- address, user, port -- and nothing here
overrides it. Whether the host is one of the fleet's own boxes (so a fleet boot and a
single check must not share it) is `on_fleet`, from the name. The FLOOR, though, is a
Spark's: 16 GiB of a GB10's one pool, held for an engine earlyoom would otherwise pick.
A box of its own owes production nothing and may not even have one pool -- ost-97x has
31 GiB of host RAM beside a discrete 8 GiB card -- so FLEET_SINGLE_GPU_FLOOR_GIB says
what that box owes itself instead (bench/OST_97X_LANE.md).

A third lane, `check`, runs beside both on such a box (operator, 2026-09-19: two one-GPU lanes
at once). It is the same rule with its own holder (`holder-check`) and its own host,
FLEET_CHECK_GPU_HOST: by default ost-97x, the RTX 5050 in WSL2 -- sm_120, not a GB10, so a
verdict there is that card's and never a number (CHARTER D5). What a box of its own owes
itself, and which image and flashinfer it runs, are facts about that box: HOSTS, below.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

KIND = 'single'
DEFAULT_HOST = 'srv4'           # bench/fleet.sh carries the same default; tests/test_fleet_single.py pins that
DEFAULT_GPU = 'GB10'
CHECK_KIND = 'check'            # the second one-GPU lane: a box of its own, checks only (D5)
CHECK_DEFAULT_HOST = 'ost-97x'  # bench/fleet.sh carries the same default; tests/test_fleet_single.py pins that
CHECK_DEFAULT_GPU = 'RTX5050'
# lane -> (the host's variable, its default, the card's variable, its default); empty host turns a lane off
LANES = {KIND: ('FLEET_SINGLE_GPU_HOST', DEFAULT_HOST, 'FLEET_SINGLE_GPU_NAME', DEFAULT_GPU),
         CHECK_KIND: ('FLEET_CHECK_GPU_HOST', CHECK_DEFAULT_HOST, 'FLEET_CHECK_GPU_NAME', CHECK_DEFAULT_GPU)}
# What a box of its own owes itself and what it runs -- by its ssh alias (bench/OST_97X_LANE.md). A Spark
# has no entry: its floor is FLOOR_GIB and its image is the one production runs there.
#   floor_gib   the room a check leaves the box: host RAM beside a discrete card, not a GB10's one pool
#   budget_gib  what a check takes of that RAM when the submitter names none: the card's memory is its own
#   image       the check image (never the production tag -- that one is ARM64, sm_121a)
#   vendored    under the box's home: the Sparks' flashinfer unpacked, mounted over `site` (the b12x path
#               imports a staticmethod only the vendored build has; bench/compile_sm121a.sh)
HOSTS = {'ost-97x': dict(floor_gib=4.0, budget_gib=4.0, image='st-engine:glm53-sm120-x86',
                         vendored='st-x86-flashinfer/vendored', site='/usr/local/lib/python3.12/site-packages')}
FLOOR_GIB = 16.0                # = engine/profiles/glm53/boot.py TEST_FLOOR_GIB, what a --test boot leaves the box
FLOOR_ENV = 'FLEET_SINGLE_GPU_FLOOR_GIB'
DEFAULT_BUDGET_GIB = 8.0        # what one ST check may take beside production; ST_PROBE_GIB raises or lowers it
BUDGET_ENV = 'ST_PROBE_GIB'
CACHE = '.single-gpu-evidence'
TTL_S = 20.0                    # _try_hold asks once a second per waiter; one ssh per TTL is enough
SSH = ('ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=4', '-o', 'StrictHostKeyChecking=accept-new')
# One round trip: the box's memory as earlyoom counts it, and whether a probe is already there.
QUERY = "cat /proc/meminfo; echo ---; docker ps --filter name=st-probe- --format '{{.Names}}'"
FLEET_HOST = re.compile(r'^(srv[1-4]|spark[a-z0-9]*|10\.10\.(?:0|1|10|11)\.[1-4])(\..*)?$')
GIB = 2 ** 20                   # /proc/meminfo counts kB


def host(environ=None, lane=KIND) -> str:
    """The lane's host; '' when the lane is off (its FLEET_*_GPU_HOST set to nothing)."""
    env = os.environ if environ is None else environ
    variable, default, _, _ = LANES[lane]
    return env.get(variable, default).strip()


def gpu(environ=None, lane=KIND) -> str:
    env = os.environ if environ is None else environ
    _, _, variable, default = LANES[lane]
    return env.get(variable, default).strip() or default


def facts(name: str) -> dict:
    """HOSTS' entry for that box, by its alias (a `user@` prefix is not part of the name); {} for a Spark."""
    return HOSTS.get((name or '').rpartition('@')[2].strip().lower(), {})


def image(name: str) -> str:
    """The check image a box of its own runs; '' where production's image is the one (a Spark)."""
    return facts(name).get('image', '')


def budget_gib(environ=None, name=None) -> float:
    """This check's memory budget beside production, in GiB: ST_PROBE_GIB, else what that box says a check
    takes (HOSTS), else a kernel check's."""
    env = os.environ if environ is None else environ
    default = float(facts(name).get('budget_gib', DEFAULT_BUDGET_GIB))
    try:
        value = float(env.get(BUDGET_ENV, default))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def floor_gib(environ=None, name=None) -> float:
    """The room a check must leave that box, in GiB (FLEET_SINGLE_GPU_FLOOR_GIB).

    The default is a Spark's, and it is a Spark's for a reason: 16 GiB is what a --test
    boot leaves a GB10 so earlyoom does not pick the engine, on a box with ONE pool for
    host and device. A box of its own owes production nothing, and the pool may not even
    be one -- ost-97x has 31 GiB of host RAM and a discrete 8 GiB card, so a check's
    device memory is not drawn from what this floor guards and 16 GiB of host RAM refuses
    every check the box could otherwise run (2026-09-15: MemAvailable 13.6 GiB, so even a
    zero budget was refused). What such a box owes itself instead is this: the variable when
    it is set, else that box's own entry in HOSTS.
    """
    env = os.environ if environ is None else environ
    default = float(facts(name).get('floor_gib', FLOOR_GIB))
    try:
        value = float(env.get(FLOOR_ENV, default))
    except (TypeError, ValueError):
        return default
    return value if value >= 0 else default


def on_fleet(name: str, environ=None) -> bool:
    """Is that host one of the fleet's own boxes? Then a fleet boot and a single check must not share it.

    From the name (srv1-4, the Sparks' fabric addresses, the spark* aliases); FLEET_SINGLE_GPU_ON_FLEET=0/1
    says otherwise for a box the name does not tell.
    """
    env = os.environ if environ is None else environ
    forced = env.get('FLEET_SINGLE_GPU_ON_FLEET', '').strip()
    if forced in ('0', '1'):
        return forced == '1'
    return bool(FLEET_HOST.match(name.rpartition('@')[2].strip().lower()))


def label(environ=None, lane=KIND) -> str:
    name = host(environ, lane)
    if not name:
        return 'off'
    return f'{gpu(environ, lane)} on {name}' + (' beside production' if on_fleet(name, environ) else '')


def target(name: str) -> str:
    """The ssh target, exactly as configured: the controller's ~/.ssh/config owns the alias.

    Nothing here forces the fleet's user or address onto it -- a Spark answers as the
    invoking user, a box of its own (ost-97x) as whatever its alias says.
    """
    return name


def parse(text: str):
    """(MemAvailable GiB or None, the probe containers there) out of QUERY's answer."""
    memory, containers = None, []
    head, sep, tail = text.partition('\n---')
    for line in head.splitlines():
        if line.startswith('MemAvailable:'):
            try:
                memory = int(line.split()[1]) / GIB
            except (IndexError, ValueError):
                memory = None
    if sep:
        containers = [line.strip() for line in tail.splitlines() if line.strip().startswith('st-probe-')]
    return memory, containers


# Run ON the box, right before its container starts. MemAvailable counts clean page cache that the
# kernel reclaims for an anonymous allocation but not, on this UMA box, for a device one -- the
# engine's own arena admission learned that (engine/base/arena.py touch_pages), and the lane's
# first real ticket proved it the other way: a kernel check OOMed on its first tiny tensor while
# MemAvailable said 26 GiB, minutes after another session's boot had faulted the box's free pages.
# So: if MemFree already covers the budget, nothing to do; else, within the floor, hold the
# budget as anonymous pages for an instant (MAP_POPULATE) and give it back, which evicts cache
# and leaves that many pages immediately free. `short` after that means the box is still
# faulting (a boot, most likely): wait, do not start.
RECLAIM = r'''
import mmap, sys
from pathlib import Path
budget, floor = float(sys.argv[1]) * 2 ** 30, float(sys.argv[2]) * 2 ** 30
def meminfo():
    return {k: int(v.split()[0]) * 1024 for k, v in (l.split(':', 1) for l in Path('/proc/meminfo').read_text().splitlines())}
m = meminfo()
if m['MemFree'] >= budget:
    print('free', round(m['MemFree'] / 2 ** 30, 1)); raise SystemExit(0)
if m['MemAvailable'] - budget < floor:
    print('no-room', round(m['MemAvailable'] / 2 ** 30, 1)); raise SystemExit(2)
page = mmap.PAGESIZE
n = -(-int(budget) // page) * page
flags = mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS | getattr(mmap, 'MAP_POPULATE', 0)
region = mmap.mmap(-1, n, flags=flags)
try:
    if not getattr(mmap, 'MAP_POPULATE', 0):
        for off in range(0, n, page):
            region[off] = 1
finally:
    region.close()
m = meminfo()
ok = m['MemFree'] >= budget
print('reclaimed' if ok else 'short', round(m['MemFree'] / 2 ** 30, 1)); raise SystemExit(0 if ok else 3)
'''


def reclaim(name: str, budget: float = None, *, run=subprocess.run, timeout: float = 120.0,
            floor: float = None) -> list:
    """Make `budget` GiB immediately free on that box, or say why not; [] means it is free now."""
    if not name:
        return ['the single-GPU lane is off (FLEET_SINGLE_GPU_HOST is empty)']
    budget = budget_gib(name=name) if budget is None else float(budget)
    floor = floor_gib(name=name) if floor is None else float(floor)
    import base64
    code = base64.b64encode(RECLAIM.encode()).decode()
    command = f'python3 -c "import base64,sys;exec(base64.b64decode(\'{code}\'))" {budget} {floor}'
    try:
        done = run([*SSH, target(name), command], capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return [f'{name}: unreachable ({type(exc).__name__}) -- could not reclaim room']
    words = (done.stdout or '').split()
    state = words[0] if words else ''
    value = words[1] if len(words) > 1 else '?'
    if done.returncode == 0 and state in ('free', 'reclaimed'):
        return []
    if state == 'no-room':
        return [f'{name}: no room beside production -- MemAvailable {value} GiB, this check\'s budget {budget:.1f} GiB, floor {floor:.1f}']
    if state == 'short':
        return [f'{name}: only {value} GiB immediately free after reclaiming for a {budget:.1f} GiB budget -- the box is still faulting (a boot?)']
    detail = [line for line in (done.stderr or done.stdout or '').splitlines() if line.strip()]
    why = f': {detail[-1].strip()[:120]}' if detail else ''
    return [f'{name}: could not reclaim room (rc {done.returncode}{why})']


def evidence(name: str, budget: float = None, *, run=subprocess.run, timeout: float = 8.0,
             floor: float = None) -> list:
    """Every reason to believe that box has no room for this check; [] means it has. Not knowing is a reason."""
    if not name:
        return ['the single-GPU lane is off (FLEET_SINGLE_GPU_HOST is empty)']
    budget = budget_gib(name=name) if budget is None else float(budget)
    floor = floor_gib(name=name) if floor is None else float(floor)
    try:
        done = run([*SSH, target(name), QUERY], capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return [f'{name}: unreachable ({type(exc).__name__}) -- this queue cannot say it has room']
    if done.returncode:
        detail = [line for line in (done.stderr or done.stdout or '').splitlines() if line.strip()]
        why = f': {detail[-1].strip()[:120]}' if detail else ''
        return [f'{name}: cannot read its memory (rc {done.returncode}{why}) -- this queue cannot say it has room']
    memory, containers = parse(done.stdout)
    reasons = []
    if containers:
        reasons.append(f'{name}: a probe is already running there ({", ".join(containers)})')
    if memory is None:
        reasons.append(f'{name}: cannot read MemAvailable -- this queue cannot say it has room')
    elif memory - budget < floor:
        reasons.append(f'{name}: no room beside production -- MemAvailable {memory:.1f} GiB, this check\'s budget '
                       f'{budget:.1f} GiB, floor {floor:.1f}: {memory - budget:.1f} GiB would be left')
    return reasons


def cached_evidence(name: str, directory, budget: float = None, *, ttl: float = TTL_S, now=time.time,
                    run=subprocess.run, floor: float = None) -> list:
    """`evidence`, remembered for `ttl` seconds under the fleet directory (one ssh per TTL), one file a host:
    two lanes asking about two boxes must not overwrite each other's answer every poll."""
    budget = budget_gib(name=name) if budget is None else float(budget)
    floor = floor_gib(name=name) if floor is None else float(floor)   # part of the key: a changed floor must not read an answer computed under the old one
    path = Path(directory) / cache_name(name)
    try:
        value = json.loads(path.read_text())
        if (isinstance(value, dict) and value.get('host') == name and value.get('budget') == budget
                and value.get('floor') == floor
                and isinstance(value.get('reasons'), list)
                and isinstance(value.get('at'), (int, float)) and 0 <= now() - value['at'] <= ttl):
            return list(value['reasons'])
    except (OSError, ValueError):
        pass
    reasons = evidence(name, budget, run=run, floor=floor)
    try:
        temporary = path.with_name(f'{path.name}.{os.getpid()}')
        temporary.write_text(json.dumps(dict(host=name, budget=budget, floor=floor, reasons=reasons, at=now())) + '\n')
        temporary.replace(path)
    except OSError:
        pass
    return reasons


def cache_name(name: str) -> str:
    """The evidence cache for one host: `.single-gpu-evidence` for the single lane's default host (as before),
    `.single-gpu-evidence.<host>` for any other."""
    return CACHE if name == DEFAULT_HOST else CACHE + '.' + re.sub(r'[^A-Za-z0-9_.=-]', '_', name)


def report_name(session: str) -> str:
    """The file a ticket's check writes what it measured to (probes/probe_report.py), under the lane host's
    ~/.cache/st -- the container's /cache. One name a ticket, in the characters `collect` copies back."""
    return 'probe-report-' + re.sub(r'[^A-Za-z0-9_.=-]', '_', session) + '.json'


def read_report(path, session: str):
    """(state, detail) for the report a ticket's check left: passed | failed | unreadable.

    Read here, on the controller, from what `collect` copied back. `passed` is recomputed from the proof
    markers rather than believed, and a report written for another ticket is unreadable, not evidence:
    the lane host's ~/.cache/st outlives every ticket.
    """
    try:
        report = json.loads(Path(path).read_text())
        if not isinstance(report, dict) or report.get('schema') != 2:
            raise ValueError('not a schema-2 probe report')
        if report.get('session') != session:
            raise ValueError(f"written for ticket {report.get('session')!r}, not {session!r}")
        proof, metrics, samples, device = (report.get(k) for k in ('proof', 'metrics', 'samples', 'device'))
        if (not isinstance(proof, dict) or not proof or any(type(v) is not bool for v in proof.values())
                or not isinstance(metrics, dict) or not metrics or type(samples) is not int or samples < 1
                or not isinstance(device, str) or not device.strip()):
            raise ValueError('proof, metrics, samples or device missing')
    except (OSError, ValueError) as exc:
        return 'unreadable', str(exc)
    what = f'{samples} sample(s) on {device.strip()}, {len(metrics)} metric(s)'
    failed = sorted(k for k, v in proof.items() if not v)
    return ('failed', what + '; proof not held: ' + ', '.join(failed)) if failed else ('passed', what)


RESULT_FIND = ("find .cache/st -maxdepth 1 -type f -newermt @{since} "
               "\\( -name '*.json' -o -name '*.jsonl' -o -name '*.log' -o -name '*.tsv' \\) -print")


def collect(name: str, since: float, into, run=subprocess.run):
    """The check's fresh files on the lane host -- what it wrote under ~/.cache/st (the container's /cache)
    since `since` -- copied into `into` on the controller, so a session reads its results here instead of
    fetching them box to box by hand. Returns the files copied; a host that cannot answer copies nothing."""
    listing = run([*SSH, target(name), RESULT_FIND.format(since=int(since))], capture_output=True, text=True, timeout=30)
    if listing.returncode != 0:
        raise OSError(f'{name}: could not list its results ({listing.stderr.strip() or listing.returncode})')
    # only what find printed: paths under the cache, one name deep -- never an answer that merely has lines
    files = [line.strip() for line in listing.stdout.splitlines()
             if re.fullmatch(r'\.cache/st/[A-Za-z0-9][A-Za-z0-9_.,=-]*', line.strip())]
    if not files:
        return []
    into = Path(into)
    into.mkdir(parents=True, exist_ok=True)
    copied = []
    for remote in files:
        done = run(['scp', '-q', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=4', f'{target(name)}:{remote}', str(into)],
                   capture_output=True, text=True, timeout=300)
        if done.returncode == 0:
            copied.append(Path(remote).name)
    return copied


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('evidence', 'reclaim', 'host', 'label', 'on-fleet', 'budget', 'floor', 'collect',
                                           'image', 'vendored'))
    parser.add_argument('--lane', choices=tuple(LANES), default=KIND,
                        help='whose host --host defaults to: single (FLEET_SINGLE_GPU_HOST) or check (FLEET_CHECK_GPU_HOST)')
    parser.add_argument('--host', default=None, help='defaults to the lane\'s host: ' + DEFAULT_HOST + ' or ' + CHECK_DEFAULT_HOST)
    parser.add_argument('--gib', type=float, default=None, help=f'this check\'s budget; defaults to {BUDGET_ENV}, then {DEFAULT_BUDGET_GIB}')
    parser.add_argument('--floor', type=float, default=None,
                        help=f'the room a check must leave that box; defaults to {FLOOR_ENV}, then {FLOOR_GIB} (a Spark\'s)')
    parser.add_argument('--cache', help='fleet directory; remembers the answer for --ttl seconds')
    parser.add_argument('--ttl', type=float, default=TTL_S)
    parser.add_argument('--since', type=float, default=None, help='collect: files the lane host wrote after this epoch second')
    parser.add_argument('--into', default=None, help='collect: the controller directory the files go to')
    parser.add_argument('--session', default=None, help='collect: the ticket, to read the report its check left')
    args = parser.parse_args(argv)
    name = host(lane=args.lane) if args.host is None else args.host.strip()
    if args.action == 'host':
        print(name)
        return 0
    if args.action == 'image':
        print(image(name))
        return 0
    if args.action == 'vendored':
        entry = facts(name)
        print(f"{entry['vendored']} {entry['site']}" if entry.get('vendored') else '')
        return 0
    if args.action == 'collect':
        if args.since is None or not args.into or not name:
            parser.error('collect needs --since, --into and a lane host')
        copied = collect(name, args.since, args.into)
        if args.session is None:
            print(len(copied))
            return 0
        # the queue's log line: how many files came back, and what the check's own report says
        line = f'{len(copied)} file(s)'
        if report_name(args.session) in copied:
            state, detail = read_report(Path(args.into) / report_name(args.session), args.session)
            line += f', report {state} ({detail})'
        print(line)
        return 0
    if args.action == 'label':
        print(label(lane=args.lane))
        return 0
    if args.action == 'budget':
        print(budget_gib(name=name) if args.gib is None else args.gib)
        return 0
    if args.action == 'floor':
        print(floor_gib(name=name) if args.floor is None else args.floor)
        return 0
    if args.action == 'on-fleet':
        return 0 if on_fleet(name) else 1
    if args.action == 'reclaim':
        reasons = reclaim(name, args.gib, floor=args.floor)
    else:
        reasons = (cached_evidence(name, args.cache, args.gib, ttl=args.ttl, floor=args.floor) if args.cache
                   else evidence(name, args.gib, floor=args.floor))
    for reason in reasons:
        print(reason)
    return 1 if reasons else 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        # Never an empty answer from a failure: the shell reads "nothing printed" as room.
        print(f'{host() or "single-GPU lane"}: evidence failed ({exc}) -- this queue cannot say it has room')
        raise SystemExit(1)
