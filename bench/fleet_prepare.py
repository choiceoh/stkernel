#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate and bind raw run inputs before queueing; cheap checks repeat before GO."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid

MAX_FILE = 8 * 1024 * 1024


def run(argv, cwd, timeout=15):
    try:
        result = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise ValueError('prepare timed out: ' + ' '.join(argv[:3])) from exc
    if result.returncode:
        raise ValueError('prepare failed: ' + ' '.join(argv[:4]) + '\n' + (result.stderr or result.stdout)[-4000:])
    return result.stdout.strip()


def digest(path):
    if not path.is_file():
        raise ValueError('required file is missing: ' + str(path))
    if path.stat().st_size > MAX_FILE:
        raise ValueError('source exceeds 8 MiB; declare large inputs in required_paths: ' + str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(value, cwd):
    return str((Path(cwd) / value).resolve())


def sources(command, cwd):
    """Only bind literal file arguments; never execute or expand shell text."""
    result = []
    interpreter = Path(command[0]).name
    if re.fullmatch(r'(?:ba|z|da)?sh|python(?:[0-9.]+)?', interpreter):
        for arg in command[1:]:
            if arg in ('-c', '-m', '-') or (arg.startswith('-') and 'c' in arg[1:]):
                break
            if arg.startswith('-'):
                continue
            path = Path(cwd) / arg
            if not path.is_file():
                raise ValueError('entrypoint script is missing: ' + str(path))
            result.append(path.resolve())
            break
    for arg in command:
        path = Path(cwd) / arg
        try:
            if path.is_file() and path.suffix in ('.py', '.sh', '.json', '.toml', '.yaml', '.yml'):
                result.append(path.resolve())
        except (OSError, ValueError):
            continue
    executable = Path(cwd) / command[0]
    if '/' in command[0] and executable.is_file():
        with executable.open('rb') as stream:
            if stream.read(2) == b'#!':
                result.append(executable.resolve())
    return list(dict.fromkeys(result))


def discover(command, cwd):
    checks = []
    for path in sources(command, cwd):
        if path.suffix != '.sh':
            continue
        text = path.read_text(errors='replace')
        # Recognize literal, read-only repository guards used by campaign scripts.
        # Dynamic shell expressions remain the campaign's responsibility.
        try:
            execution_cwd = path.parent.parent if 'dirname "$0")/..' in text and 'cd "$REPO"' in text else Path(cwd)
            repo = run(['git', '-C', str(execution_cwd), 'rev-parse', '--show-toplevel'], cwd, 3)
        except ValueError:
            continue
        for line_number, line in enumerate(text.splitlines(), 1):
            if line.lstrip().startswith('#'):
                continue
            match = re.search(r'git merge-base --is-ancestor ([A-Za-z0-9_./-]+) HEAD(?:\s|$)', line)
            if match:
                ref = match[1]
                check = dict(kind='ancestor', repo=repo, ref=ref, source=str(path), line=line_number)
                if ref.startswith('origin/') and re.search(r'\bgit fetch origin ' + re.escape(ref[7:]) + r'(?:\s|$)', text):
                    check['fetch'] = ref[7:]
                checks.append(check)
            if 'git status --porcelain' in line and re.search(r'-z\s+"?\$\(git status --porcelain(?: --untracked-files=(?:normal|all))?\)', line):
                checks.append(dict(kind='clean', repo=repo, source=str(path), line=line_number))
    return checks


def spec_read(path):
    spec = json.loads(Path(path).read_text()) if path else {}
    allowed = {'required_paths', 'absent_paths', 'git', 'images', 'cpu_command', 'timeout_seconds'}
    if not isinstance(spec, dict) or set(spec) - allowed:
        raise ValueError('unknown preparation fields; expected ' + ', '.join(sorted(allowed)))
    for key in ('required_paths', 'absent_paths', 'images', 'cpu_command'):
        if key in spec and (not isinstance(spec[key], list) or not all(isinstance(x, str) and x and '\0' not in x for x in spec[key])):
            raise ValueError(key + ' must be an array of nonempty strings')
    if 'git' in spec and (not isinstance(spec['git'], dict) or set(spec['git']) - {'repo', 'ancestor', 'clean'}):
        raise ValueError('git accepts repo, ancestor and clean')
    timeout = spec.get('timeout_seconds', 120)
    if isinstance(timeout, bool) or not isinstance(timeout, int) or not 1 <= timeout <= 300:
        raise ValueError('timeout_seconds must be 1..300')
    return spec


def validate(value, *, refresh=False, external=True):
    cwd = value['cwd']
    if not Path(cwd).is_dir():
        raise ValueError('working directory disappeared: ' + cwd)
    exe = value['command'][0]
    actual = shutil.which(resolve(exe, cwd) if '/' in exe else exe)
    if not actual:
        raise ValueError('executable missing: ' + exe)
    if value.get('executable') and executable_identity(actual) != value['executable']:
        raise ValueError('queued executable changed; prepare the intended runtime again: ' + actual)
    for filename, expected in value['files'].items():
        if digest(Path(filename)) != expected:
            raise ValueError('queued input changed; edit the reservation to accept the new input: ' + filename)
    for filename in value['required_paths']:
        if not Path(filename).exists():
            raise ValueError('required path is missing: ' + filename)
    for filename in value['absent_paths']:
        if Path(filename).exists():
            raise ValueError('fresh output path already exists: ' + filename)
    if value.get('head'):
        repo, expected = value['head']
        if run(['git', 'rev-parse', 'HEAD'], repo, 3) != expected:
            raise ValueError('queued checkout revision changed; edit or submit the intended revision: ' + repo)
    fetched = set()
    for check in value['checks']:
        repo = check['repo']
        source = f"{check.get('source', repo)}:{check.get('line', 0)}"
        if refresh and check.get('fetch') and (repo, check['fetch']) not in fetched:
            run(['git', 'fetch', '--quiet', 'origin', check['fetch']], repo)
            fetched.add((repo, check['fetch']))
        if check['kind'] == 'clean' and run(['git', 'status', '--porcelain'], repo, 3):
            raise ValueError(source + ': campaign requires a clean checkout')
        if check['kind'] == 'ancestor':
            try:
                run(['git', 'merge-base', '--is-ancestor', check['ref'], 'HEAD'], repo, 3)
            except ValueError as exc:
                raise ValueError(source + ': candidate must include ' + check['ref'] + '; update the candidate and its CPU evidence before queueing') from exc
    if external:
        for image in value['images']:
            run(['docker', 'image', 'inspect', image], cwd, 5)
    return value


def executable_identity(filename):
    path = Path(filename).resolve()
    stat = path.stat()
    return dict(path=str(path), device=stat.st_dev, inode=stat.st_ino,
                size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def prepare(directory, session, command, cwd, *, spec_path=None, fleet=None, execute_cpu=True):
    if not command or not command[0] or not all(isinstance(x, str) and '\0' not in x for x in command):
        raise ValueError('prepare requires a command argv')
    cwd = str(Path(cwd).resolve())
    spec = spec_read(spec_path)
    files = sources(command, cwd) + sources(spec['cpu_command'], cwd) if spec.get('cpu_command') else sources(command, cwd)
    if spec_path:
        files.append(Path(spec_path).resolve())
    value = dict(version=1, session=session, command=command, cwd=cwd,
                 files={str(p):digest(p) for p in files}, checks=discover(command, cwd),
                 required_paths=[resolve(p,cwd) for p in spec.get('required_paths', [])],
                 absent_paths=[resolve(p,cwd) for p in spec.get('absent_paths', [])],
                 images=spec.get('images', []), spec_path=str(Path(spec_path).resolve()) if spec_path else None,
                 prepared_at=time.time())
    try:
        repo = run(['git', 'rev-parse', '--show-toplevel'], cwd, 3)
        value['head'] = [repo, run(['git', 'rev-parse', 'HEAD'], repo, 3)]
    except ValueError:
        pass
    git = spec.get('git', {})
    repo = resolve(git.get('repo', cwd), cwd)
    if git.get('ancestor'):
        if not isinstance(git['ancestor'], str) or not re.fullmatch(r'[A-Za-z0-9_./-]+', git['ancestor']):
            raise ValueError('git ancestor must be a literal ref')
        check = dict(kind='ancestor', repo=repo, ref=git['ancestor'])
        if git['ancestor'].startswith('origin/'):
            check['fetch'] = git['ancestor'][7:]
        value['checks'].append(check)
    if git.get('clean'):
        value['checks'].append(dict(kind='clean', repo=repo))
    exe = command[0]
    actual = shutil.which(resolve(exe, cwd) if '/' in exe else exe)
    if not actual:
        raise ValueError('executable missing: ' + exe)
    value['executable'] = executable_identity(actual)
    validate(value, refresh=True)
    for path in files:
        if path.suffix == '.sh':
            run(['bash', '-n', str(path)], cwd, 5)
        elif path.suffix == '.py':
            compile(path.read_bytes(), str(path), 'exec')
    if execute_cpu and spec.get('cpu_command'):
        if not fleet:
            raise ValueError('CPU preparation requires the fleet classifier')
        command_class = run(['bash', str(fleet), 'classify', *spec['cpu_command']], cwd, 15)
        if command_class == 'gpu':
            raise ValueError('CPU preparation command shows GPU use; run classify --explain to locate it')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
        try:
            process = subprocess.Popen(spec['cpu_command'], cwd=cwd, env=env,
                                       stdout=sys.stderr, stderr=sys.stderr, start_new_session=True)
            process.wait(timeout=spec.get('timeout_seconds',120))
        except subprocess.TimeoutExpired as exc:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise ValueError('CPU preparation timed out') from exc
        if process.returncode:
            raise ValueError('CPU preparation failed with code ' + str(process.returncode))
        validate(value)
    root = Path(directory) / 'preparations'
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = root / (uuid.uuid4().hex + '.json')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as out:
        json.dump(value, out, ensure_ascii=False, sort_keys=True)
        out.write('\n')
    return path


def check_pending(directory, session, *, refresh=False, external=True, withdraw_failed=False):
    import fleet_pending
    record = fleet_pending.read_record(Path(directory), session)
    if not record or record.get('state') != 'queued' or not record.get('prepare_manifest'):
        return  # Compatibility: already queued older controllers retain their contract.
    try:
        value = json.loads(Path(record['prepare_manifest']).read_text())
        if value['command'] != record['command'] or value['cwd'] != record['cwd']:
            raise ValueError('preparation does not match the accepted command revision')
        validate(value, refresh=refresh, external=external)
    except (ValueError, OSError, subprocess.SubprocessError):
        if withdraw_failed:
            # Slow checks are outside .lock. A valid edit committed while we
            # checked an older revision must survive the older check's failure.
            import fleet_handoff
            with fleet_pending.lock(Path(directory)):
                current = fleet_pending.read_record(Path(directory), session)
                keys = ('ticket', 'pid', 'start', 'revision', 'prepare_manifest')
                if not current or any(current.get(k) != record.get(k) for k in keys):
                    return
                rows = fleet_handoff.rows(Path(directory))
                row = next((r for r in rows if r[1] == session), None)
                if not row or row[0] != record['ticket'] or row[6] != str(record['pid']):
                    return
                temp = Path(directory) / 'queue.prepare.tmp'
                temp.write_text(''.join('|'.join(r) + '\n' for r in rows if r != row))
                temp.replace(Path(directory) / 'queue')
                for name in ('priority-front', 'priority-yield'):
                    marker = Path(directory) / name
                    if marker.exists() and marker.read_text().strip() == session:
                        marker.unlink()
        raise


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['create', 'check'])
    ap.add_argument('session')
    ap.add_argument('--spec')
    ap.add_argument('--fleet')
    ap.add_argument('--refresh', action='store_true')
    ap.add_argument('--local', action='store_true')
    ap.add_argument('--withdraw-failed', action='store_true')
    args, command = ap.parse_known_args(argv)
    if command[:1] == ['--']:
        command.pop(0)
    try:
        directory = Path(os.environ['FLEET_DIR'])
        if args.action == 'create':
            print(prepare(directory,args.session,command,os.getcwd(),spec_path=args.spec,fleet=args.fleet))
        else:
            check_pending(directory,args.session,refresh=args.refresh,external=not args.local,withdraw_failed=args.withdraw_failed)
    except (ValueError, OSError, SyntaxError, subprocess.SubprocessError) as exc:
        print('PREPARE REFUSED (no GPU hold): ' + str(exc), file=sys.stderr)
        return 3
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
