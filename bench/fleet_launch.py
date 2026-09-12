#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Detach an ordinary fleet run after a durable, process-bound startup receipt."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys
import time
import uuid

import fleet_handoff as handoff

MAX_TAIL = 32 * 1024


def identity(pid):
    if sys.platform.startswith('linux'):
        return handoff.identity(pid)
    # Production uses /proc's start token. This fallback lets CPU-only process
    # fixtures and local launch inspection run on macOS as well.
    try:
        value = subprocess.check_output(['ps', '-o', 'stat=,lstart=', '-p', str(pid)],
                                        text=True, timeout=2).strip()
        state, started = value.split(None, 1)
        return None if state.startswith('Z') else 'ps:' + started
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def read(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def write(path, value):
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def ack_path(request):
    return request.with_suffix('.ack.json')


@contextmanager
def locked(path, deadline):
    with path.open('a') as stream:
        while True:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError('another launcher is still preparing this session; retry the same command')
                time.sleep(.05)
        yield


def live(value):
    return bool(value.get('pid') and value.get('start') and
                value.get('host') == socket.gethostname() and identity(value['pid']) == value['start'])


def request_for(session, pid):
    nonce, filename = os.environ.get('FLEET_LAUNCH_ID'), os.environ.get('FLEET_LAUNCH_REQUEST')
    if not nonce or not filename:
        return None
    path = Path(filename)
    value = read(path)
    # Nested fleet commands inherit the parent's environment but may not claim
    # the parent's launch receipt.
    if not value or value.get('session') != session:
        return None
    if value.get('launch_id') != nonce or value.get('pid') != pid or not live(value):
        raise ValueError('detached launch receipt does not match this process')
    return path, value


def acknowledge(directory, session, ticket, *, pid, start):
    """Called after pending.register, while its owner holds the fleet lock."""
    request = request_for(session, pid)
    if not request:
        return False
    path, value = request
    if start is not None and start != value['start']:
        raise ValueError('detached reservation start token changed')
    rows = handoff.rows(Path(directory))
    if not any(row[0] == ticket and row[1] == session and row[6] == str(pid) for row in rows):
        raise ValueError('detached reservation has not registered its queue owner')
    write(ack_path(path), dict(session=session, launch_id=value['launch_id'], pid=pid,
                              start=value['start'], ticket=ticket, state='queued', acknowledged_at=time.time()))
    return True


def cpu_event(session, returncode=None):
    request = request_for(session, os.getppid())
    if not request:
        return
    path, value = request
    previous = read(ack_path(path)) or {}
    answer = dict(session=session, launch_id=value['launch_id'], pid=value['pid'], start=value['start'],
                  state='running-cpu' if returncode is None else 'finished-cpu',
                  started_at=previous.get('started_at', time.time()))
    if returncode is not None:
        answer.update(returncode=returncode, finished_at=time.time())
    write(ack_path(path), answer)


def tail(path):
    try:
        with path.open('rb') as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                return ''
            stream.seek(max(0, os.fstat(stream.fileno()).st_size - MAX_TAIL))
            return stream.read(MAX_TAIL).decode(errors='replace')
    except OSError:
        return ''


def pending_record(directory, request, ticket=None):
    from fleet_pending import read_record
    value = read_record(directory, request['session'], ticket)
    if value and all(value.get(key) == request.get(key) for key in ('session', 'pid', 'start', 'launch_id')):
        return value
    return None


def receipt(directory, path, request):
    value = read(ack_path(path))
    if not value or any(value.get(key) != request.get(key) for key in ('session', 'pid', 'start', 'launch_id')):
        return None
    if value.get('state') not in ('queued', 'running-cpu', 'finished-cpu'):
        return None
    if value['state'] == 'finished-cpu':
        return value
    pending = pending_record(directory, request, value.get('ticket'))
    if value['state'] == 'queued':
        if not pending or pending.get('ticket') != value.get('ticket'):
            return None
        if pending.get('state') in ('finished', 'cancelled'):
            return dict(value, state=pending.get('outcome', pending['state']),
                        returncode=pending.get('returncode', 1), log_path=pending.get('log_path'))
    if live(request):
        return dict(value, **({'log_path':pending['log_path']} if pending and pending.get('log_path') else {}))
    return dict(value, state='interrupted', returncode=1)


def answer(request, *, disposition, state, accepted=False, **details):
    return dict(session=request['session'], launch_id=request['launch_id'], pid=request.get('pid'),
                disposition=disposition, state=state, accepted=accepted,
                startup_log=request['startup_log'], log_path=request['startup_log'],
                **details)


def worker(path):
    # Parent publishes PID + start before this process may run preflight. If
    # the parent dies before publishing, this shim exits without a reservation.
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        value = read(path)
        if value and value.get('pid') == os.getpid() and live(value):
            os.execvpe('bash', ['bash', value['fleet'], *value['argv']], os.environ)
        time.sleep(.02)
    raise ValueError('detached launcher did not publish its process identity')


def start(fleet, session, argv, timeout=30):
    if not re.fullmatch(r'[A-Za-z0-9_.-]{1,80}', session):
        raise ValueError('session must be a short alphanumeric name')
    if not argv or argv[0] != 'run' or '--detach' in argv[:argv.index('--') if '--' in argv else len(argv)]:
        raise ValueError('detached launch requires run arguments with --detach removed')
    fleet = Path(fleet).resolve()
    if not fleet.is_file():
        raise ValueError('fleet script does not exist')
    directory = Path(os.environ['FLEET_DIR']).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    launches = directory / 'launches'
    launches.mkdir(mode=0o700, exist_ok=True)
    if launches.is_symlink():
        raise ValueError('launch directory must be a real directory')
    launches.chmod(0o700)
    digest = hashlib.sha256(session.encode()).hexdigest()
    path = launches / (digest + '.json')
    deadline = time.monotonic() + max(.1, min(float(timeout), 60))
    with locked(launches / (digest + '.lock'), deadline):
        request = read(path)
        signature = dict(fleet=str(fleet), session=session, argv=argv, cwd=str(Path.cwd()))
        disposition = 'existing' if request else 'started'
        process = None
        if request:
            if any(request.get(key) != value for key, value in signature.items()):
                raise ValueError('session already has a detached launch with different arguments; edit its waiting reservation or use a fresh session name')
            if request.get('terminal'):
                result = request['terminal']
                return dict(result, disposition='existing'), result.get('returncode', 0)
        else:
            with locked(directory / '.lock', deadline):
                held = [row[0] for row in handoff.holders(directory).values()]
                rows = handoff.rows(directory) if (directory / 'queue').exists() else []
                if any(row[1] == session for row in rows) or session in held:
                    raise ValueError('session is already queued or running; inspect or edit that reservation instead')
            nonce = uuid.uuid4().hex
            log = launches / (digest + '.' + nonce + '.log')
            request = dict(signature, launch_id=nonce, host=socket.gethostname(), created_at=time.time(),
                           startup_log=str(log), pid=None, start=None)
            write(path, request)
            env = dict(os.environ, FLEET_LAUNCH_ID=nonce, FLEET_LAUNCH_REQUEST=str(path))
            fd = os.open(log, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            try:
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'worker', str(path)],
                                           env=env, stdin=subprocess.DEVNULL, stdout=fd,
                                           stderr=subprocess.STDOUT, start_new_session=True)
            finally:
                os.close(fd)
            request.update(pid=process.pid, start=identity(process.pid))
            if not request['start']:
                process.terminate()
                raise ValueError('could not identify detached worker; no reservation acknowledged')
            write(path, request)
        while True:
            acknowledged = receipt(directory, path, request)
            if acknowledged:
                state = acknowledged['state']
                details = {key:acknowledged[key] for key in ('ticket', 'returncode') if key in acknowledged}
                result = answer(request, disposition=disposition, state=state,
                                accepted=state != 'interrupted', **details)
                if acknowledged.get('log_path'):
                    result['log_path'] = acknowledged['log_path']
                rc = acknowledged.get('returncode', 0)
                if 'returncode' in acknowledged:
                    request['terminal'] = result
                    write(path, request)
                return result, rc
            rc = process.poll() if process else None
            unpublished = not request.get('pid') and time.time() - request['created_at'] > 11
            if rc is not None or (request.get('pid') and not live(request)) or unpublished:
                rc = rc if rc is not None else 1
                result = answer(request, disposition=disposition, state='startup-failed',
                                returncode=rc or 2, log_tail=tail(Path(request['startup_log'])))
                request['terminal'] = result
                write(path, request)
                return result, result['returncode']
            if time.monotonic() >= deadline:
                return answer(request, disposition=disposition, state='starting',
                              error='startup is not yet acknowledged; retry the identical command to inspect this same launch',
                              log_tail=tail(Path(request['startup_log']))), 124
            time.sleep(.05)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='action', required=True)
    launch = sub.add_parser('start')
    launch.add_argument('fleet'); launch.add_argument('session')
    launch.add_argument('command', nargs=argparse.REMAINDER)
    child = sub.add_parser('worker'); child.add_argument('request', type=Path)
    cpu = sub.add_parser('cpu-started'); cpu.add_argument('session')
    complete = sub.add_parser('complete'); complete.add_argument('session'); complete.add_argument('returncode', type=int)
    args = ap.parse_args(argv)
    if args.action == 'worker':
        return worker(args.request)
    if args.action in ('cpu-started', 'complete'):
        cpu_event(args.session, None if args.action == 'cpu-started' else args.returncode)
        return 0
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    value, rc = start(args.fleet, args.session, command, os.environ.get('FLEET_LAUNCH_TIMEOUT', '30'))
    print(json.dumps(value, ensure_ascii=False))
    return rc


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(json.dumps(dict(error=str(exc), accepted=False), ensure_ascii=False), file=sys.stderr)
        raise SystemExit(2)
