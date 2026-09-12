#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The single-GPU lane: a check that needs one GPU runs on ost-97x's 5050, not the four Sparks.

The queue (bench/fleet.sh) has two lanes. `boot` and `probe` take the fleet -- four
Sparks, one holder file. `single` takes ONE GPU on another host and has its own holder
(`holder-single`), so a kernel check behind a queued boot does not wait for the fleet,
and a boot behind a queued kernel check does not wait for the 5050. WHICH lane a command
belongs to is decided at admission -- fleet_onepass.validate says how many GPUs the
entry needs -- and this module knows only WHERE that lane runs and whether the GPU there
is ours to take.

Evidence before a grant, as everywhere in this queue: that host's own GPU process list,
over ssh. A host that cannot answer is not free (D3).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

KIND = 'single'
DEFAULT_HOST = 'ost-97x'        # bench/fleet.sh carries the same default; tests/test_fleet_single.py pins that
DEFAULT_GPU = '5050'
USER = 'choiceoh'               # the fleet's user everywhere else (choiceoh@10.10.10.N)
CACHE = '.single-gpu-evidence'
TTL_S = 20.0                    # _try_hold asks once a second per waiter; one ssh per TTL is enough
SSH = ('ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=4', '-o', 'StrictHostKeyChecking=accept-new')
QUERY = 'nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits'


def host(environ=None) -> str:
    """The lane's host; '' when the lane is off (FLEET_SINGLE_GPU_HOST set to nothing)."""
    env = os.environ if environ is None else environ
    return env.get('FLEET_SINGLE_GPU_HOST', DEFAULT_HOST).strip()


def gpu(environ=None) -> str:
    env = os.environ if environ is None else environ
    return env.get('FLEET_SINGLE_GPU_NAME', DEFAULT_GPU).strip() or DEFAULT_GPU


def label(environ=None) -> str:
    name = host(environ)
    return f'{gpu(environ)} on {name}' if name else 'off'


def target(name: str) -> str:
    """[user@]host for ssh: the fleet's user unless the name already carries one."""
    return name if '@' in name else f'{USER}@{name}'


def evidence(name: str, *, run=subprocess.run, timeout: float = 8.0) -> list:
    """Every reason to believe that GPU is not ours; [] means free. Not knowing is a reason."""
    if not name:
        return ['the single-GPU lane is off (FLEET_SINGLE_GPU_HOST is empty)']
    try:
        done = run([*SSH, target(name), QUERY], capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return [f'{name}: unreachable ({type(exc).__name__}) -- this queue cannot say its GPU is free']
    if done.returncode:
        detail = [line for line in (done.stderr or done.stdout or '').splitlines() if line.strip()]
        why = f': {detail[-1].strip()[:120]}' if detail else ''
        return [f'{name}: cannot read its GPU (rc {done.returncode}{why}) -- this queue cannot say it is free']
    rows = [line.strip() for line in done.stdout.splitlines() if line.strip()]
    if not rows:
        return []
    first = [cell.strip() for cell in rows[0].split(',')]
    who = f'pid {first[0]}' + (f' ({first[1]})' if len(first) > 1 and first[1] else '')
    more = f' and {len(rows) - 1} more' if len(rows) > 1 else ''
    return [f'{name}: busy outside this queue -- {who}{more}']


def cached_evidence(name: str, directory, *, ttl: float = TTL_S, now=time.time, run=subprocess.run) -> list:
    """`evidence`, remembered for `ttl` seconds under the fleet directory (one ssh per TTL)."""
    path = Path(directory) / CACHE
    try:
        value = json.loads(path.read_text())
        if (isinstance(value, dict) and value.get('host') == name and isinstance(value.get('reasons'), list)
                and isinstance(value.get('at'), (int, float)) and 0 <= now() - value['at'] <= ttl):
            return list(value['reasons'])
    except (OSError, ValueError):
        pass
    reasons = evidence(name, run=run)
    try:
        temporary = path.with_name(f'{path.name}.{os.getpid()}')
        temporary.write_text(json.dumps(dict(host=name, reasons=reasons, at=now())) + '\n')
        temporary.replace(path)
    except OSError:
        pass
    return reasons


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('evidence', 'host', 'label'))
    parser.add_argument('--host', default=None, help='defaults to FLEET_SINGLE_GPU_HOST, then ' + DEFAULT_HOST)
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
    reasons = cached_evidence(name, args.cache, ttl=args.ttl) if args.cache else evidence(name)
    for reason in reasons:
        print(reason)
    return 1 if reasons else 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        # Never an empty answer from a failure: the shell reads "nothing printed" as free.
        print(f'{host() or "single-GPU lane"}: evidence failed ({exc}) -- this queue cannot say it is free')
        raise SystemExit(1)
