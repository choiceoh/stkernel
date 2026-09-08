#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Admit canonical onepass GPU workflows before reserving the fleet.

This is an operator workflow contract, not a security sandbox. Custom shell
payloads cannot establish that their only GPU workload is onepass, so use the
standard pair, chain or live onepass command instead.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import json
import os
from pathlib import Path
import re
import sqlite3
import sys

from fleet_prepare import command_environment

POLICY = 'GPU work is onepass-only; use fleet.sh pair, chain or onepass'
SHELL_ENTRIES = ('bench/pair.sh', 'bench/chain.sh', 'bench/ab-lever.sh',
                 'probes/run_ar_consumer_campaign.sh')
PYTHON_ENTRIES = ('bench/onepass.py', 'bench/experiments.py')


def _path(value, cwd):
    return (Path(cwd) / value).resolve()


def _same(path, relative, repo):
    """A familiar filename alone never authorizes an arbitrary wrapper."""
    expected = Path(repo) / relative
    try:
        if path.read_bytes() == expected.read_bytes():
            return
    except OSError as exc:
        raise ValueError(f'{POLICY}; cannot verify {relative}: {exc}') from exc
    raise ValueError(f'{POLICY}; {path} differs from the current canonical {relative}')


def _name(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', value):
        raise ValueError('onepass arm name must be a literal name of at most 128 characters')


def _knobs(value):
    for knob in value.split():
        if not re.fullmatch(r'[A-Z_][A-Z0-9_]*=[A-Za-z0-9_.,:/+%=-]+', knob):
            raise ValueError('onepass knobs must be literal NAME=value assignments')
        if knob.split('=', 1)[0] in {'LEGS', 'LEVER', 'FLEET', 'REPO', 'BASH_ENV', 'ENV', 'PYTHONPATH'}:
            raise ValueError('onepass knobs cannot replace workload control settings')
        if knob.startswith('PREFILL_WARMUP=') and knob != 'PREFILL_WARMUP=0':
            raise ValueError(POLICY + '; separate prefill warmup requests are disabled')


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError('invalid canonical onepass arguments: ' + message)


def _onepass_args(arguments):
    # Mirror only the public, literal argv grammar; never import the GPU or
    # serving environment to validate a submission.
    parser = _Parser(add_help=False, allow_abbrev=False)
    parser.add_argument('--name', default='onepass')
    parser.add_argument('--ctx', default='2000,32000,128000')
    parser.add_argument('--out')
    parser.add_argument('--max-tokens', type=int, default=400)
    parser.add_argument('--num-spec', type=int, default=7)
    parser.add_argument('--combine-min-ctx', type=int, default=32000)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--fixed-decode-tokens', type=int, default=0)
    parser.add_argument('--fixed-decode-reps', type=int, default=3)
    parser.add_argument('--require-exclusive', action='store_true')
    value = parser.parse_args(arguments)
    _name(value.name)
    from measurement_contract import from_args
    from_args(value)
    if value.num_spec < 0:
        raise ValueError('onepass num-spec must be nonnegative')


def _experiment(arguments, cwd, repo, environment):
    if (len(arguments) != 4 or arguments[0] != '--root' or
            arguments[2] != 'execute' or not re.fullmatch(r'[A-Za-z0-9_-]+', arguments[3])):
        raise ValueError(POLICY + '; expected experiments.py --root ROOT execute JOB')
    database = _path(arguments[1], cwd) / 'experiments.sqlite3'
    try:
        with closing(sqlite3.connect(database.as_uri() + '?mode=ro', uri=True, timeout=5)) as connection:
            row = connection.execute('SELECT payload FROM jobs WHERE id=?', (arguments[3],)).fetchone()
        if row is None:
            raise ValueError('unknown experiment')
        payload = json.loads(row[0])
        spec = payload['spec']
        source = _path(payload['repo'], cwd)
        if spec['kind'] not in {'pair', 'baseline'} or spec.get('command'):
            raise ValueError('experiment must use a standard pair or shared onepass baseline')
        if source != Path(cwd).resolve() or source != _path(environment.get('REPO', str(cwd)), cwd):
            raise ValueError('experiment source does not match its execution repository')
        if Path(payload.get('bash', 'bash')).name != 'bash':
            raise ValueError('experiment requires the standard bash interpreter')
        _knobs(' '.join(k + '=' + v for k, v in spec.get('knobs', {}).items()))
        for key in ('LEGS', 'LEVER', 'FLEET', 'REPO', 'BASH_ENV', 'ENV', 'PYTHONPATH'):
            if key in spec.get('env', {}):
                raise ValueError('experiment overrides a workload control setting: ' + key)
        for relative in ('bench/experiments.py', 'bench/serving_group.py',
                         'bench/experiment_baselines.py', 'bench/ab-lever.sh',
                         'bench/onepass.py', 'bench/onepass_deploy.py'):
            _same(source / relative, relative, repo)
    except (OSError, sqlite3.Error, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(POLICY + '; cannot verify recorded experiment: ' + str(exc)) from exc


def validate(command, cwd, repo, environment=None, *, kind='boot', rehearsal_only=False):
    """Validate effective argv and canonical source bytes; return entry metadata.

    ``repo`` is the trusted controller checkout or pinned runner, whose bench
    files (and the approved campaign wrapper) supply the expected bytes.
    ``cwd`` and REPO may name the candidate checkout, with matching controls.
    Recovery does not enter this GPU experiment lane: fleet_idle.authorize
    separately owns its boot-only maintenance action.
    """
    if kind not in {'boot', 'probe'}:
        raise ValueError('unknown GPU lane')
    if (not isinstance(command, (list, tuple)) or not command or
            any(not isinstance(v, str) or '\0' in v for v in command)):
        raise ValueError(POLICY + '; command must be literal argv')
    command, effective = command_environment(command, environment)
    if rehearsal_only and effective.get('FLEET_REHEARSE') != '1':
        raise ValueError('CPU rehearsal requires effective FLEET_REHEARSE=1 after env prefixes')
    if len(command) < 2 or command[1].startswith('-'):
        raise ValueError(POLICY + '; custom commands and interpreter code strings are disabled')
    interpreter = Path(command[0]).name
    entries = (SHELL_ENTRIES if interpreter == 'bash' else PYTHON_ENTRIES
               if re.fullmatch(r'python(?:3(?:\.[0-9]+)?)?', interpreter) else ())
    path = _path(command[1], cwd)
    relative = next((entry for entry in entries if path.as_posix().endswith('/' + entry)), None)
    if relative is None:
        raise ValueError(POLICY + '; custom GPU scripts and standalone checks are disabled')
    if kind == 'probe' and relative != 'bench/onepass.py':
        raise ValueError(POLICY + '; the live-serving lane accepts only bench/onepass.py')
    if effective.get('LEGS', 'onepass') != 'onepass':
        raise ValueError(POLICY + '; LEGS must be onepass')
    if relative in SHELL_ENTRIES and effective.get('PREFILL_WARMUP', '0') != '0':
        raise ValueError(POLICY + '; separate prefill warmup requests are disabled')
    for key in ('BASH_ENV', 'ENV', 'PYTHONPATH'):
        if effective.get(key):
            raise ValueError(POLICY + '; workload code injection setting is unsupported: ' + key)
    for key, target in (('LEVER', 'bench/ab-lever.sh'), ('FLEET', 'bench/fleet.sh')):
        if effective.get(key):
            _same(_path(effective[key], cwd), target, repo)
    _same(path, relative, repo)
    source = _path(effective.get('REPO', str(cwd)), cwd)
    dependencies = ('bench/ab-lever.sh', 'bench/onepass.py', 'bench/onepass_deploy.py') if relative in SHELL_ENTRIES else ('bench/onepass.py',)
    if relative == 'probes/run_ar_consumer_campaign.sh':
        dependencies += ('bench/pair.sh',)
    for dependency in dependencies:
        _same(source / dependency, dependency, repo)
    args = command[2:]
    if relative in ('bench/pair.sh', 'bench/ab-lever.sh'):
        if len(args) not in (1, 2):
            raise ValueError(POLICY + '; expected NAME and optional literal knobs')
        _name(args[0])
        _knobs(args[1] if len(args) == 2 else '')
    elif relative == 'bench/chain.sh':
        if not args:
            raise ValueError('onepass chain needs at least one NAME=KNOBS arm')
        seen = set()
        for arm in args:
            name, equal, knobs = arm.partition('=')
            if not equal or arm.startswith('-'):
                raise ValueError(POLICY + '; chain accepts only NAME=KNOBS, without --after or --legs')
            _name(name)
            if name in seen:
                raise ValueError('onepass chain arm names must be unique')
            seen.add(name)
            _knobs(knobs)
    elif relative == 'bench/onepass.py':
        _onepass_args(args)
    elif relative == 'bench/experiments.py':
        _experiment(args, cwd, repo, effective)
    else:
        while args:
            if args[0] == '--baseline-only':
                args = args[1:]
            elif args[0] == '--gpu-evidence' and len(args) > 1 and args[1] and not args[1].startswith('--'):
                args = args[2:]
            else:
                raise ValueError('AR onepass campaign accepts only --baseline-only or --gpu-evidence DIR')
    if rehearsal_only and relative not in {'bench/pair.sh', 'bench/chain.sh', 'bench/ab-lever.sh'}:
        raise ValueError('CPU rehearsal supports only the canonical pair, chain and ab-lever helpers')
    return dict(policy='onepass-only', entry=relative, kind=kind)


def authorize_wait(directory, session, pid):
    """Only the registered supervisor may enter the queue's wait/hold loop."""
    import fleet_handoff
    import fleet_pending
    from fleet_idle import descendant
    record = fleet_pending.read_record(Path(directory), session)
    if (not record or record.get('pid') != pid or record.get('state') not in {'queued', 'paused'}
            or not record.get('prepare_manifest') or not fleet_handoff.live(record)):
        raise ValueError(POLICY + '; bare request/wait is disabled; submit a supervised command')
    if not descendant(os.getpid(), pid):
        raise ValueError('onepass wait requires the owning supervisor process')
    return dict(policy='onepass-only', session=session, owner=pid)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--cwd', default=os.getcwd())
    parser.add_argument('--kind', choices=('boot', 'probe'), default='boot')
    parser.add_argument('--rehearsal-only', action='store_true')
    parser.add_argument('--wait-owner', nargs=2, metavar=('SESSION', 'PID'))
    parser.add_argument('--directory')
    args, command = parser.parse_known_args(argv)
    if command[:1] == ['--']:
        command.pop(0)
    try:
        if args.wait_owner:
            if not args.directory or command:
                raise ValueError('wait owner requires a fleet directory and no payload')
            print(json.dumps(authorize_wait(args.directory, args.wait_owner[0], int(args.wait_owner[1]))))
            return 0
        result = validate(command, args.cwd, args.repo, kind=args.kind, rehearsal_only=args.rehearsal_only)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
