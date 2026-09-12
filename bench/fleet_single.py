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
single check must not share it) is `on_fleet`, from the name.
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
FLOOR_GIB = 16.0                # = engine/profiles/glm53/boot.py TEST_FLOOR_GIB, what a --test boot leaves the box
DEFAULT_BUDGET_GIB = 8.0        # what one ST check may take beside production; ST_PROBE_GIB raises or lowers it
BUDGET_ENV = 'ST_PROBE_GIB'
CACHE = '.single-gpu-evidence'
TTL_S = 20.0                    # _try_hold asks once a second per waiter; one ssh per TTL is enough
SSH = ('ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=4', '-o', 'StrictHostKeyChecking=accept-new')
# One round trip: the box's memory as earlyoom counts it, and whether a probe is already there.
QUERY = "cat /proc/meminfo; echo ---; docker ps --filter name=st-probe- --format '{{.Names}}'"
FLEET_HOST = re.compile(r'^(srv[1-4]|spark[a-z0-9]*|10\.10\.(?:0|1|10|11)\.[1-4])(\..*)?$')
GIB = 2 ** 20                   # /proc/meminfo counts kB


def host(environ=None) -> str:
    """The lane's host; '' when the lane is off (FLEET_SINGLE_GPU_HOST set to nothing)."""
    env = os.environ if environ is None else environ
    return env.get('FLEET_SINGLE_GPU_HOST', DEFAULT_HOST).strip()


def gpu(environ=None) -> str:
    env = os.environ if environ is None else environ
    return env.get('FLEET_SINGLE_GPU_NAME', DEFAULT_GPU).strip() or DEFAULT_GPU


def budget_gib(environ=None) -> float:
    """This check's memory budget beside production, in GiB (ST_PROBE_GIB; the default is a kernel check's)."""
    env = os.environ if environ is None else environ
    try:
        value = float(env.get(BUDGET_ENV, DEFAULT_BUDGET_GIB))
    except (TypeError, ValueError):
        return DEFAULT_BUDGET_GIB
    return value if value > 0 else DEFAULT_BUDGET_GIB


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


def label(environ=None) -> str:
    name = host(environ)
    if not name:
        return 'off'
    return f'{gpu(environ)} on {name}' + (' beside production' if on_fleet(name, environ) else '')


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


def evidence(name: str, budget: float = None, *, run=subprocess.run, timeout: float = 8.0,
             floor: float = FLOOR_GIB) -> list:
    """Every reason to believe that box has no room for this check; [] means it has. Not knowing is a reason."""
    if not name:
        return ['the single-GPU lane is off (FLEET_SINGLE_GPU_HOST is empty)']
    budget = budget_gib() if budget is None else float(budget)
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
                    run=subprocess.run) -> list:
    """`evidence`, remembered for `ttl` seconds under the fleet directory (one ssh per TTL)."""
    budget = budget_gib() if budget is None else float(budget)
    path = Path(directory) / CACHE
    try:
        value = json.loads(path.read_text())
        if (isinstance(value, dict) and value.get('host') == name and value.get('budget') == budget
                and isinstance(value.get('reasons'), list)
                and isinstance(value.get('at'), (int, float)) and 0 <= now() - value['at'] <= ttl):
            return list(value['reasons'])
    except (OSError, ValueError):
        pass
    reasons = evidence(name, budget, run=run)
    try:
        temporary = path.with_name(f'{path.name}.{os.getpid()}')
        temporary.write_text(json.dumps(dict(host=name, budget=budget, reasons=reasons, at=now())) + '\n')
        temporary.replace(path)
    except OSError:
        pass
    return reasons


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('evidence', 'host', 'label', 'on-fleet', 'budget'))
    parser.add_argument('--host', default=None, help='defaults to FLEET_SINGLE_GPU_HOST, then ' + DEFAULT_HOST)
    parser.add_argument('--gib', type=float, default=None, help=f'this check\'s budget; defaults to {BUDGET_ENV}, then {DEFAULT_BUDGET_GIB}')
    parser.add_argument('--cache', help='fleet directory; remembers the answer for --ttl seconds')
    parser.add_argument('--ttl', type=float, default=TTL_S)
    args = parser.parse_args(argv)
    name = host() if args.host is None else args.host.strip()
    if args.action == 'host':
        print(name)
        return 0
    if args.action == 'label':
        print(label())
        return 0
    if args.action == 'budget':
        print(budget_gib() if args.gib is None else args.gib)
        return 0
    if args.action == 'on-fleet':
        return 0 if on_fleet(name) else 1
    reasons = (cached_evidence(name, args.cache, args.gib, ttl=args.ttl) if args.cache
               else evidence(name, args.gib))
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
