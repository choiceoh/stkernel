#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Private, authenticated CPU preparation receipts and conservative identities."""
import hashlib
import hmac
import json
import os
from pathlib import Path
import stat
import subprocess
import uuid

# These variables belong to the controller, not the CPU preparation process.
# Remove them from execution too: ignoring an input without doing so is unsafe.
BOOKKEEPING = {'_', 'SHLVL', 'OLDPWD', 'FLEET_PREPARE_MANIFEST', 'FLEET_RUN_KIND',
               'FLEET_LAUNCH_ID', 'FLEET_LAUNCH_REQUEST', 'FLEET_RUNNER_REPO', 'FLEET_PID',
               'FLEET_VALIDATION_STORE', 'FLEET_VALIDATION_REQUIRED', 'FLEET_VALIDATION_LEVEL', 'FLEET_RECOVERY_RECEIPT',
               'FLEET_RESTORE_MANAGED', 'FLEET_TIMEOUT_MIN', 'FLEET_NO_RESTORE_CHECK', 'FLEET'}
TRANSPORT = {'SSH_CLIENT', 'SSH_CONNECTION', 'SSH_TTY', 'TERM_PROGRAM', 'TERM_PROGRAM_VERSION',
             'LC_TERMINAL', 'LC_TERMINAL_VERSION'}


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()


def request_environment(cwd, session):
    env = {k:v for k,v in os.environ.items() if k not in BOOKKEEPING | TRANSPORT}
    env.update(PWD=str(Path(cwd).resolve()), FLEET_SESSION=session)
    return env


def cpu_environment(cwd, session):
    return dict(request_environment(cwd, session), CUDA_VISIBLE_DEVICES='')


def payload_environment(environment):
    """Payload launch removes the same transport-only metadata as its receipt."""
    return {k:v for k,v in environment.items() if k not in TRANSPORT}


def key(directory, *, create=False):
    root = Path(directory) / 'preparations'
    if create:
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if root.is_symlink():
        raise ValueError('preparation receipt directory must not be a symlink')
    if create:
        root.chmod(0o700)
    path = root / '.receipt-key'
    if create:
        temporary = root / ('.receipt-key.' + uuid.uuid4().hex)
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(os.urandom(32))
            try:
                os.link(temporary, path)
            except FileExistsError:
                pass
        finally:
            temporary.unlink()
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
        with os.fdopen(fd, 'rb') as stream:
            meta = os.fstat(stream.fileno())
            if not stat.S_ISREG(meta.st_mode) or meta.st_mode & 0o077 or meta.st_uid != os.getuid():
                raise ValueError('preparation receipt key must be a private regular file')
            value = stream.read(33)
    except OSError as exc:
        raise ValueError('prepared receipt authentication key is missing or unreadable; prepare again') from exc
    if len(value) != 32:
        raise ValueError('prepared receipt authentication key is invalid; prepare again')
    return value


def private_digest(secret, value):
    return hmac.new(secret, encoded(value), hashlib.sha256).hexdigest()


def sign(value, secret):
    value['receipt'] = private_digest(secret, {k:v for k,v in value.items() if k != 'receipt'})


def read(directory, path):
    path = Path(path).resolve()
    root = (Path(directory) / 'preparations').resolve()
    if path.parent != root or path.suffix != '.json':
        raise ValueError('prepared receipt must belong to this fleet preparation directory')
    try:
        value = json.loads(path.read_text())
    except (ValueError, OSError) as exc:
        raise ValueError('prepared receipt is unreadable; prepare again') from exc
    if not isinstance(value, dict) or not isinstance(value.get('receipt'), str):
        raise ValueError('prepared manifest has no authenticated receipt; prepare again')
    expected = private_digest(key(directory), {k:v for k,v in value.items() if k != 'receipt'})
    if not hmac.compare_digest(value['receipt'], expected):
        raise ValueError('prepared receipt was modified or forged; prepare again')
    return value



def cpu_identity(repo, spec, command, cwd, env, required_paths, protected_paths):
    """Only the reviewed suite runner has an established dependency contract."""
    import cpu_evidence
    if not repo:
        raise ValueError('CPU preparation reuse requires a Git checkout and an audited CPU suite')
    if any(not Path(p).is_file() for p in required_paths):
        raise ValueError('CPU preparation reuse requires explicit file inputs; directory dependencies are unaudited')
    request = dict(command=spec['cpu_command'], env={}, context={'fleet_prepare': 2, 'cwd': cwd, 'command': command},
                   inputs=required_paths, timeout_s=spec.get('timeout_seconds', 120))
    identity = cpu_evidence.identity(Path(repo), request, env)
    if identity is None:
        raise ValueError('CPU preparation command has no audited dependency contract; prepare again or use bench/cpu_checks.py')
    source = cpu_source_identity(repo, identity['scope'], protected_paths)
    return dict(key=identity['key'], scope=identity['scope'], source=source)


def cpu_source_identity(repo, scope, protected_paths):
    import fleet_source
    pins = list(protected_paths)
    if not scope.startswith('audited-'):
        # An unknown/changed suite has full-tree scope, including dirty docs.
        pins.extend(p.decode() for p in subprocess.check_output(
            ['git', '-C', str(repo), 'ls-files', '-z', '--cached', '--others', '--exclude-standard']).split(b'\0') if p)
    return fleet_source.identity(repo, protected_paths=pins)


def difference(expected, actual):
    paths = sorted(k for k in set(expected) | set(actual) if expected.get(k) != actual.get(k))
    return ', '.join(paths[:5]) + (' (and more)' if len(paths) > 5 else '')
