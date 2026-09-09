#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Only the idle controller may restore production, after 300 quiet seconds."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import stat
import subprocess
import sys
import time
import uuid

IDLE_SECONDS = 300
ROOT = Path(__file__).resolve().parents[1]
NODES = ('10.10.10.1', '10.10.10.2', '10.10.10.3', '10.10.10.4')


def read(path, default=None):
    try:
        if path.is_symlink() or path.stat().st_uid != os.getuid():
            raise ValueError('idle controller state must be an owned regular file')
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(value, stream, sort_keys=True)
        stream.write('\n')
    temporary.replace(path)


def boot_id():
    try:
        return Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    except FileNotFoundError:
        if sys.platform == 'darwin':
            # Local CPU fixtures also exercise enqueue/release on macOS. The
            # actual systemd watcher and GPU observations remain Linux-only.
            return subprocess.check_output(['sysctl', '-n', 'kern.boottime'], text=True).strip()
        raise


def clock():
    return time.monotonic()


@contextmanager
def lock(directory, name='.lock', *, blocking=True):
    directory.mkdir(parents=True, exist_ok=True)
    fd = os.open(directory / name, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield


def activity(directory, reason):
    """Caller owns fleet .lock. Every enqueue/acquire/release resets idle age."""
    directory = Path(directory)
    state = dict(version=1, boot_id=boot_id(), since=clock(), phase='waiting',
                 reason=reason, generation=uuid.uuid4().hex, updated_at=time.time())
    write(directory / 'idle-recovery.json', state)
    return state


def process(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else (int(fields[1]), fields[19])
    except (FileNotFoundError, ProcessLookupError):
        return None


def descendant(pid, ancestor):
    seen = set()
    while pid > 1 and pid not in seen:
        if pid == ancestor:
            return True
        seen.add(pid)
        value = process(pid)
        if value is None:
            return False
        pid = value[0]
    return False


def authorize(directory, session):
    """An environment flag alone cannot grant session restoration authority."""
    directory = Path(directory)
    lease = read(directory / 'idle-recovery-owner.json')
    holder = (directory / 'holder').read_text().split('|')
    identity = process(lease['pid']) if isinstance(lease, dict) else None
    if (not lease or lease.get('boot_id') != boot_id() or lease.get('session') != session
            or holder[:2] != [session, str(lease['pid'])]
            or not identity or identity[1] != lease.get('start')
            or not descendant(os.getpid(), lease['pid'])):
        raise ValueError('session restore is disabled; only the 5-minute idle controller may restore')
    return lease


def boot_authorize(directory, session):
    if os.environ.get('FLEET_BOOT_INTENT') == 'recovery' or os.environ.get('FLEET_DEPLOY_RECOVERY_RECEIPT'):
        return authorize(directory, session)
    row = (Path(directory) / 'holder').read_text().strip().split('|')
    if (len(row) != 7 or row[0] != session or row[6] != 'boot' or not row[1].isdigit()
            or row[2] not in (socket.gethostname(), socket.gethostname().split('.')[0])
            or not process(int(row[1])) or not descendant(os.getpid(), int(row[1]))):
        raise ValueError('boot requires an owned fleet experiment or the idle recovery controller')
    return dict(session=session, intent='experiment')


def holder_live(directory):
    path = directory / 'holder'
    if not path.exists() or not path.read_text().strip():
        return False
    row = path.read_text().strip().split('|')
    if len(row) != 7 or row[2] not in (socket.gethostname(), socket.gethostname().split('.')[0]):
        return True  # Unknown ownership is busy, not permission to take over.
    return bool(row[1].isdigit() and process(int(row[1])))


def runnable_queue(directory, serving_stopped):
    from fleet_pause import paused
    path = directory / 'queue'
    for line in path.read_text().splitlines() if path.exists() else ():
        if not line:
            continue
        row = line.split('|')
        if len(row) != 7:
            raise ValueError('cannot determine idle state from a malformed queue')
        if paused(directory, row[1], row):
            continue
        if row[6] and (not row[6].isdigit() or not process(int(row[6]))):
            continue
        # A probe needing absent serving cannot run until this controller boots.
        if row[5] == 'probe' and serving_stopped:
            continue
        return True
    return False


def legacy_work():
    # Read only process commands, never environments. GPU kernels and unmanaged
    # boot waiters must also finish before a maintenance hold can be acquired.
    output = subprocess.check_output(['ps', '-eo', 'args'], text=True)
    import re
    pattern = r'^(?:bash\s+\S*(?:ab-lever|lever-chain|onepass-after|start-glm53|orchestrate)[^\s]*\.sh|python(?:3(?:\.\d+)?)?\s+(?:\S*/)?(?:bench/(?:onepass|bracket)\.py|probes/))'
    return any(re.search(pattern, line.strip()) for line in output.splitlines())


def _local_head(host):
    """Match the fixed fleet head against kernel-reported local addresses."""
    if host != '10.10.10.2':
        return False
    # This is the same local-address requirement as the canonical launcher.
    # Do not use DNS, environment overrides, or retry a failed SSH locally.
    interfaces = json.loads(subprocess.check_output(
        ['ip', '-j', '-4', 'address', 'show'], text=True, timeout=4))
    if not isinstance(interfaces, list):
        raise ValueError('cannot establish local fleet head identity')
    local = False
    for interface in interfaces:
        if not isinstance(interface, dict) or not isinstance(interface.get('addr_info'), list):
            raise ValueError('cannot establish local fleet head identity')
        for address in interface['addr_info']:
            if not isinstance(address, dict):
                raise ValueError('cannot establish local fleet head identity')
            if address.get('family') == 'inet' and address.get('local') == host:
                local = True
    return local


def node_idle(host):
    # Recovery shares these hosts with resident inference services. A foreign
    # CUDA context is neither fleet work nor evidence of insufficient memory.
    # Check node/driver reachability here; GLM requests, experiments and leases
    # are checked separately. The launcher sizes memory after reclaiming only
    # the old GLM containers and refuses if that preflight cannot establish a
    # budget. Never require another service to unload just to restore GLM.
    code = '''import json,subprocess
s=lambda a:subprocess.check_output(a,text=True,stderr=subprocess.DEVNULL).strip()
pids=[p.strip() for p in s(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits']).splitlines() if p.strip()]
print(json.dumps({'idle':all(p.isascii() and p.isdecimal() and int(p)>0 for p in pids)}))'''
    command = ([sys.executable, '-B', '-c', code] if _local_head(host) else
               ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=4',
                'choiceoh@' + host, 'python3 -c ' + shlex.quote(code)])
    result = subprocess.run(command, capture_output=True, text=True, timeout=12)
    value = json.loads(result.stdout) if result.returncode == 0 else None
    if not isinstance(value, dict) or value.get('idle') is not True:
        raise ValueError('GPU process inventory is unavailable or invalid on ' + host)


def observe():
    import fleet_entry
    container = fleet_entry.inspect()
    metrics = fleet_entry.idle(container, 'http://127.0.0.1:8000')
    if legacy_work():
        raise ValueError('unmanaged experiment or boot is still running')
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(node_idle, NODES))
    # Cumulative request counters catch traffic completed between timer ticks.
    counters = '\n'.join(sorted(line for line in metrics.splitlines()
                              if line.startswith(('vllm:request_success_total',
                                                  'vllm:e2e_request_latency_seconds_count'))))
    return dict(stopped=metrics == 'stopped', traffic=hashlib.sha256(counters.encode()).hexdigest())


def recovery(directory):
    import fleet_recovery
    import fleet_validation
    configured = directory / 'production-repo'
    repo = Path(configured.read_text().strip()) if configured.exists() else ROOT
    # This controller never starts a release suite. Operators explicitly prime
    # or refresh recovery; an idle tick only consumes approved existing evidence.
    original = fleet_recovery.source_release
    def refuse(*args, **kwargs):
        raise ValueError('no approved recovery receipt; prepare recovery outside the idle controller')
    fleet_recovery.source_release = refuse
    try:
        return fleet_validation.prepare_recovery(repo, directory / 'validation')
    finally:
        fleet_recovery.source_release = original


def restore(directory, receipt, session):
    import fleet_validation
    env = fleet_validation.environment()
    env.update(FLEET_DIR=str(directory), FLEET_SESSION=session, FLEET_RUNNER_REPO=str(ROOT),
               FLEET_RECOVERY_RECEIPT=receipt, FLEET_VALIDATION_STORE=str(directory / 'validation'),
               FLEET_VALIDATION_REQUIRED='1', FLEET_RESTORE_MANAGED='1',
               FLEET_BOOT_INTENT='recovery')
    logdir = directory / 'idle-recovery-logs'
    logdir.mkdir(mode=0o700, exist_ok=True)
    with (logdir / (session + '.log')).open('w') as output:
        child = subprocess.Popen(['bash', str(ROOT / 'bench/fleet_restore.sh')], env=env,
                                 stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return child.wait(timeout=3600)
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                # A shell can exit before descendants that ignore TERM. Reap
                # the remaining group before releasing the recovery hold.
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


def _tick(directory):
    directory = Path(directory).resolve()
    with lock(directory, '.idle-controller.lock', blocking=False):
        with lock(directory):
            if holder_live(directory):
                return activity(directory, 'holder active')
            state = read(directory / 'idle-recovery.json')
            if (not state or state.get('boot_id') != boot_id() or not isinstance(state.get('since'), (int, float))
                    or state['since'] > clock() or state.get('phase') == 'recovering'):
                state = activity(directory, 'idle observation started')
            generation = state['generation']
        try:
            observed = observe()
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            with lock(directory):
                return activity(directory, 'not idle: ' + str(exc))
        with lock(directory):
            if holder_live(directory) or runnable_queue(directory, observed['stopped']):
                return activity(directory, 'runnable work')
            current = read(directory / 'idle-recovery.json')
            if current['generation'] != generation:
                return current
            if state.get('traffic', observed['traffic']) != observed['traffic']:
                state = activity(directory, 'serving traffic')
            state.update(traffic=observed['traffic'], idle_seconds=max(0, clock() - state['since']))
            write(directory / 'idle-recovery.json', state)
            if state['idle_seconds'] < IDLE_SECONDS:
                return state
            generation = state['generation']
        # Receipt verification/network work happens outside the scheduler lock.
        try:
            approved = recovery(directory)
            import fleet_entry
            if fleet_entry.production_current(Path(approved['repo']), fleet_entry.inspect()):
                with lock(directory):
                    state = read(directory / 'idle-recovery.json')
                    if state['generation'] != generation or holder_live(directory):
                        return state
                    state = activity(directory, 'approved defaults already healthy')
                    state.update(phase='healthy', traffic=observed['traffic'])
                    write(directory / 'idle-recovery.json', state)
                    return state
            observed = observe()  # New GPU processes/requests can appear meanwhile.
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            with lock(directory):
                return activity(directory, 'recovery deferred: ' + str(exc))
        with lock(directory):
            state = read(directory / 'idle-recovery.json')
            if (holder_live(directory) or runnable_queue(directory, observed['stopped'])
                    or state['generation'] != generation or clock() - state['since'] < IDLE_SECONDS):
                return activity(directory, 'work arrived before recovery')
            if observed['traffic'] != state.get('traffic'):
                return activity(directory, 'serving traffic before recovery')
            session = 'idle-recovery-' + uuid.uuid4().hex[:12]
            identity = process(os.getpid())
            lease = dict(session=session, pid=os.getpid(), start=identity[1], boot_id=boot_id(),
                         receipt=approved['receipt'], acquired_at=time.time())
            write(directory / 'idle-recovery-owner.json', lease)
            holder = f'{session}|{os.getpid()}|{socket.gethostname().split(".")[0]}|{int(time.time())}|60|automatic recovery after 5 idle minutes|boot\n'
            (directory / 'holder').write_text(holder)
            (directory / 'restore-debt.json').unlink(missing_ok=True)
            state.update(phase='recovering', session=session, reason='5-minute idle threshold reached')
            write(directory / 'idle-recovery.json', state)
        rc = 1
        try:
            rc = restore(directory, approved['receipt'], session)
        finally:
            with lock(directory):
                if (directory / 'holder').exists() and (directory / 'holder').read_text() == holder:
                    (directory / 'holder').unlink()
                (directory / 'idle-recovery-owner.json').unlink(missing_ok=True)
                state = activity(directory, 'idle recovery completed' if rc == 0 else 'idle recovery failed')
                state.update(phase='healthy' if rc == 0 else 'retry', returncode=rc)
                write(directory / 'idle-recovery.json', state)
        return state


def tick(directory):
    try:
        return _tick(directory)
    except BlockingIOError:
        raise
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        # A failed observation is not part of five proven idle minutes.
        with lock(Path(directory)):
            return activity(directory, 'idle observation failed: ' + str(exc))


def legacy_defer(directory, session):
    """No-boot migration endpoint for an already running old supervisor."""
    directory = Path(directory)
    with lock(directory):
        row = (directory / 'holder').read_text().strip().split('|')
        if (len(row) != 7 or row[0] != session or not row[1].isdigit()
                or not descendant(os.getpid(), int(row[1]))):
            raise ValueError('legacy restore deferral requires the owning session')
        state = activity(directory, 'legacy session restore deferred')
        with (directory / 'lifecycle.jsonl').open('a') as output:
            output.write(json.dumps(dict(t=time.time(), session=session, event='restore-deferred',
                                         reason='central idle recovery policy')) + '\n')
        return dict(state, deferred=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['tick', 'activity', 'authorize', 'boot-authorize', 'legacy-defer', 'status'])
    parser.add_argument('directory', type=Path)
    parser.add_argument('detail', nargs='?')
    args = parser.parse_args()
    try:
        if args.action == 'activity':
            result = activity(args.directory, args.detail or 'fleet activity')
        elif args.action == 'authorize':
            result = authorize(args.directory, args.detail or os.environ.get('FLEET_SESSION', ''))
        elif args.action == 'boot-authorize':
            result = boot_authorize(args.directory, args.detail or os.environ.get('FLEET_SESSION', ''))
        elif args.action == 'legacy-defer':
            result = legacy_defer(args.directory, args.detail or os.environ.get('FLEET_SESSION', ''))
        elif args.action == 'status':
            result = read(args.directory / 'idle-recovery.json', {})
        else:
            import fleet_validation
            fleet_validation.bootstrap_python(args.directory / 'validation')
            result = tick(args.directory)
        print(json.dumps(result, sort_keys=True))
        return 0
    except BlockingIOError:
        return 0  # Another controller tick already owns the maintenance decision.
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        print('ABORT: ' + str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
