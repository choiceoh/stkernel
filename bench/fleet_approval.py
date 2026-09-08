#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Bind managed deployment to the candidate and main accepted before queueing."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import fleet_prepared
import fleet_source


def git(repo, *args):
    from fleet_prepare import run
    return run(['git', *args], repo)


def commit(repo, ref):
    value = git(repo, 'rev-parse', '--verify', ref + '^{commit}')
    if not re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}', value):
        raise ValueError('deployment approval requires a full commit identity')
    return value


def freeze(value):
    """Only called before admission; never advance an existing signed approval."""
    if value.get('deployment_approvals'):
        validate(value)
        return
    approvals, fetched, bases = [], set(), {}
    for target in value['deployment_targets']:
        repo = target['repo']
        if repo not in bases:
            if git(repo, 'status', '--porcelain', '--untracked-files=normal'):
                raise ValueError('deployment approval requires a clean committed checkout: ' + repo)
            git(repo, 'fetch', '--quiet', 'origin', 'main')
            fetched.add((repo, 'main'))
            bases[repo] = commit(repo, 'origin/main')
        approval = dict(target, base=bases[repo], candidate=commit(repo, 'HEAD'))
        fleet_source.require_base(repo, approval['base'], protected_paths=value.get('protected_paths', []))
        approvals.append(approval)
    if not approvals:
        raise ValueError('deployment approval requires at least one target')
    for check in value['checks']:
        if check['kind'] not in ('ancestor', 'source-base'):
            continue
        repo = check['repo']
        if check.get('fetch') and (repo, check['fetch']) not in fetched:
            git(repo, 'fetch', '--quiet', 'origin', check['fetch'])
            fetched.add((repo, check['fetch']))
        check['accepted_ref'] = commit(repo, check['ref'])
    value['deployment_approvals'] = approvals


def validate_approval(approval, protected_paths=()):
    repo = approval['repo']
    for name in ('base', 'candidate'):
        if not re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}', approval.get(name, '')):
            raise ValueError('deployment approval has no fixed ' + name + ' commit')
    if commit(repo, 'HEAD') != approval['candidate']:
        raise ValueError('approved deployment candidate changed; edit or prepare again: ' + repo)
    if git(repo, 'status', '--porcelain', '--untracked-files=normal'):
        raise ValueError('approved deployment checkout is dirty; edit or prepare again: ' + repo)
    # The authenticated approval already proves the base relationship. Exact
    # candidate identity and a clean checkout preserve that proof without
    # repeating the source comparison (or touching a remote) during the hold.
    return approval


def validate(value):
    for approval in value.get('deployment_approvals', []):
        validate_approval(approval, value.get('protected_paths', []))


def verify(directory, session, repo, profile, environment=None):
    """The live reservation selects the receipt, never a caller-supplied SHA."""
    import fleet_handoff
    import fleet_idle
    import fleet_pending
    environment = os.environ if environment is None else environment
    directory = Path(directory)
    fleet_idle.boot_authorize(directory, session)
    record = fleet_pending.read_record(directory, session)
    holder = (directory / 'holder').read_text().strip().split('|')
    if (not record or record.get('kind') != 'boot' or record.get('state') != 'running'
            or not fleet_handoff.live(record) or len(holder) != 7
            or holder[:2] != [session, str(record['pid'])]):
        raise ValueError('deployment approval requires the current running boot reservation')
    path = record.get('prepare_manifest')
    if (not path or not environment.get('FLEET_PREPARE_MANIFEST')
            or Path(environment['FLEET_PREPARE_MANIFEST']).resolve() != Path(path).resolve()):
        raise ValueError('deployment receipt is not the accepted reservation revision')
    value = fleet_prepared.read(directory, path)
    if (value.get('session') != session or value.get('command') != record.get('command')
            or value.get('cwd') != record.get('cwd')):
        raise ValueError('deployment receipt does not match the accepted reservation')
    repo = str(Path(repo).resolve())
    matches = [a for a in value.get('deployment_approvals', [])
               if a.get('repo') == repo and a.get('profile') == profile]
    # Image/model overrides are part of the target approved by admission.
    matches = [a for a in matches if (a.get('image') or '') == environment.get('IMAGE', '')
               and (a.get('model') or '') == (str((Path(repo) / environment['MODEL_HOST_PATH']).resolve())
                    if environment.get('MODEL_HOST_PATH') else '')]
    if len(matches) != 1:
        raise ValueError('deployment target has no unique prequeue approval; prepare again')
    return validate_approval(matches[0], value.get('protected_paths', []))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['verify'])
    parser.add_argument('--repo', required=True)
    parser.add_argument('--profile', required=True)
    args = parser.parse_args(argv)
    try:
        result = verify(os.environ.get('FLEET_DIR', '/home/choiceoh/glm53-logs/fleet'),
                        os.environ.get('FLEET_SESSION', ''), args.repo, args.profile)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print('DEPLOY APPROVAL REFUSED: ' + str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
