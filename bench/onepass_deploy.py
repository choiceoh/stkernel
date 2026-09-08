#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Bind a canonical onepass arm to its submitted source before spending a boot.

A matching deployment is reused. Missing or older source is deployed through the
existing admission-checked deployer; this helper never changes the checkout.
The live-serving mode only verifies identity and never deploys or boots.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys


def source_revision(repo):
    revision = subprocess.check_output(
        ['git', '-C', str(repo), 'rev-parse', '--verify', 'HEAD'], text=True, timeout=10).strip()
    if not re.fullmatch(r'[0-9a-f]{40,64}', revision):
        raise ValueError('cannot identify the submitted onepass source')
    if subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain',
                                '--untracked-files=normal'], text=True, timeout=10).strip():
        raise ValueError('onepass source checkout is dirty; preserve a committed candidate')
    return revision


def overlay_directory(repo, environment):
    values = re.findall(r'^PROFILE_OVERLAY_DIR=(.+)$',
                        (repo / 'profiles/glm53.env').read_text(), re.M)
    words = shlex.split(values[0], comments=True) if len(values) == 1 else []
    if len(words) != 1 or not Path(words[0]).is_absolute() or '$' in words[0]:
        raise ValueError('onepass needs a literal absolute PROFILE_OVERLAY_DIR')
    directory = Path(words[0])
    for key in ('OVERLAY_DIR', 'PROFILE_OVERLAY_DIR'):
        if environment.get(key) and Path(environment[key]).resolve() != directory.resolve():
            raise ValueError('onepass overlay directory differs from its submitted profile')
    return directory


def selected_modules(repo):
    values = re.findall(r'^MODULES=(.+)$', (repo / 'profiles/glm53.env').read_text(), re.M)
    words = shlex.split(values[0], comments=True) if len(values) == 1 else []
    modules = words[0].split() if len(words) == 1 else []
    if not modules or len(set(modules)) != len(modules) or any(
            not re.fullmatch(r'[A-Za-z0-9_-]+', name) for name in modules):
        raise ValueError('onepass needs literal selected profile MODULES')
    return modules


def selected_manifest(repo, modules):
    """Mirror the composer's literal module rows and relative-target expansion."""
    values = re.findall(r'^TARGET_PREFIX=(.+)$', (repo / 'profiles/glm53.env').read_text(), re.M)
    if not values:
        prefix = '/opt/venv/lib/python3.12/site-packages/'
    else:
        words = shlex.split(values[0], comments=True) if len(values) == 1 else []
        if len(words) != 1 or not words[0].startswith('/') or '$' in words[0]:
            raise ValueError('onepass needs a literal absolute TARGET_PREFIX')
        prefix = words[0]
    rows, targets = {}, set()
    for module in modules:
        manifest = repo / 'overlay/modules' / module / 'manifest.tsv'
        if manifest.resolve() != manifest:
            raise ValueError('selected module manifest cannot redirect outside its source')
        for line in manifest.read_text().splitlines():
            if not line or line.startswith('#'):
                continue
            fields = line.split('\t')
            if len(fields) != 3:
                raise ValueError('malformed selected module manifest: ' + module)
            name, target, base = fields
            target = target if target.startswith('/') else prefix + target
            if (not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*', name)
                    or not fields[1] or '..' in Path(target).parts
                    or not (base == 'absent' or re.fullmatch(r'[0-9a-f]{64}', base))
                    or name in rows or target in targets):
                raise ValueError('invalid or duplicate selected module row: ' + module)
            rows[name] = (target, base)
            targets.add(target)
    if not rows:
        raise ValueError('selected module manifest is empty')
    return rows


def deployed(repo, directory, revision):
    path = directory / 'manifest.tsv'
    try:
        data = path.read_bytes()
        lines = data.decode().splitlines()
        commits = [line.removeprefix('# source_commit=') for line in lines
                   if line.startswith('# source_commit=')]
        rows = [line for line in lines if line and not line.startswith('#')]
        if commits != [revision] or not rows:
            return None
        modules = selected_modules(repo)
        expected = selected_manifest(repo, modules)
        seen = set()
        for line in rows:
            fields = line.split('\t')
            if (len(fields) != 3 or not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*', fields[0])
                    or not fields[1].startswith('/') or not (directory / fields[0]).is_file()
                    or fields[0] in seen or expected.get(fields[0]) != tuple(fields[1:])):
                return None
            seen.add(fields[0])
            sources = [repo / 'overlay/modules' / module / fields[0] for module in modules
                       if (repo / 'overlay/modules' / module / fields[0]).is_file()]
            # Selected source must remain inside its committed module. A
            # familiar basename or a source_commit comment alone is insufficient.
            if (len(sources) != 1 or sources[0].resolve() != sources[0]
                    or (directory / fields[0]).resolve().parent != directory.resolve()
                    or sources[0].read_bytes() != (directory / fields[0]).read_bytes()):
                return None
        if seen != expected.keys():
            return None
        return dict(manifest=str(path), manifest_sha256=hashlib.sha256(data).hexdigest())
    except (OSError, UnicodeError):
        return None


def live_binding(container, directory, identity, stamp):
    """Bind deployment proof to the running container, not just mutable disk."""
    if not container or not container.get('State', {}).get('Running'):
        raise ValueError('live onepass requires an existing running server')
    try:
        started = datetime.fromisoformat(container['State']['StartedAt'].replace('Z', '+00:00'))
        if started.tzinfo is None:
            raise ValueError('missing boot timezone')
        started_ns = int(started.timestamp() * 1_000_000_000)
        mounts = {row['Destination']: Path(row['Source']).resolve()
                  for row in container['Mounts'] if row.get('Type') == 'bind'}
        inputs = [Path(identity['manifest']), stamp]
        for line in Path(identity['manifest']).read_text().splitlines():
            if not line or line.startswith('#'):
                continue
            name, target, _base = line.split('\t')
            source = directory / name
            if mounts.get(target) != source.resolve():
                raise ValueError('running container does not bind the submitted overlay: ' + name)
            inputs.append(source)
        if any(path.stat().st_mtime_ns > started_ns for path in inputs):
            raise ValueError('deployment changed after the running container boot; submit a pair')
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError('live onepass lacks container boot and overlay binding proof') from exc
    return str(container.get('Id', '')) + '|' + container['State']['StartedAt']


def ensure(repo, *, live=False, environment=None):
    repo = Path(repo).resolve()
    environment = dict(os.environ if environment is None else environment)
    revision = source_revision(repo)
    directory = overlay_directory(repo, environment)
    identity = deployed(repo, directory, revision)
    reused = identity is not None
    if live:
        if not identity:
            raise ValueError('live onepass source differs from deployed serving; submit a pair to deploy it')
        stamp = Path(environment.get('MK_OVERLAY_STAMP', '/home/choiceoh/glm53-cache/.overlay-sha'))
        if stamp.read_text().strip() != identity['manifest_sha256']:
            raise ValueError('live onepass deployment has not been booted; submit a pair')
        import fleet_entry
        container = fleet_entry.inspect()
        identity['boot_id'] = live_binding(container, directory, identity, stamp)
    elif not reused:
        # Do not let a standalone lever publish overlays before its launcher
        # discovers that it has no fleet ownership. Recovery retains its stronger
        # process-bound authorization in this same existing authority check.
        import fleet_idle
        fleet_idle.boot_authorize(environment.get('FLEET_DIR', '/home/choiceoh/glm53-logs/fleet'),
                                  environment.get('FLEET_SESSION', ''))
        subprocess.run(['bash', str(repo / 'launchers/deploy-overlays.sh'), 'glm53'],
                       cwd=repo, env=dict(environment, PROFILE='glm53'), check=True,
                       stdout=sys.stderr, timeout=600)
        if source_revision(repo) != revision:
            raise ValueError('onepass source changed while publishing overlays')
        identity = deployed(repo, directory, revision)
        if not identity:
            raise ValueError('deployment did not attest the submitted onepass source')
    return dict(source_commit=revision, overlay_dir=str(directory), reused=reused,
                live=live, **identity)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args(argv)
    try:
        print(json.dumps(ensure(args.repo, live=args.live), sort_keys=True))
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print('ABORT: ' + str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
