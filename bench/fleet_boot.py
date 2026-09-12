#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Supervise a boot payload and finish once, including nested/failed runners."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import queue
import select
import signal
import stat
import subprocess
import sys
import threading
import time

import fleet_handoff as handoff
import fleet_pending as pending
from fleet_prepared import payload_environment


class Supervisor:
    def __init__(self, fleet, session, estimate, note, command):
        self.fleet, self.session, self.estimate, self.note, self.command = fleet, session, estimate, note, command
        self.directory = Path(os.environ['FLEET_DIR'])
        self.kind = os.environ.get('FLEET_RUN_KIND', 'boot')
        self.repo = Path(__file__).resolve().parent.parent
        self.env = dict(payload_environment(os.environ), FLEET_PID=str(os.getpid()), FLEET_SESSION=session,
                        FLEET_RESTORE_MANAGED='1' if self.kind == 'boot' else '0', FLEET_RUNNER_REPO=str(self.repo),
                        FLEET=fleet, FLEET_NO_RESTORE_CHECK='1')
        # The queue takes the fleet lease as queue/<session> at GO; the payload's launcher and
        # probe runner VERIFY that record instead of taking one of their own (one record, one
        # owner). This process is the record's pid, so its death frees the fleet at once.
        if self.kind == 'boot':
            self.env['ST_LEASE_OWNER'] = 'queue/' + session
            self.env['ST_LEASE_PATH'] = os.environ.get('FLEET_LEASE_PATH', '/home/choiceoh/glm53-logs/st-fleet.lock')
        self.child = None
        self.child_interruptible = True
        self.stopping = 0
        self.log_fd = None
        self.log_error = None

    def warning(self, message):
        try:
            fd = sys.stderr.fileno()
        except (AttributeError, OSError, ValueError):
            try:
                print(message, file=sys.stderr)
            except (OSError, ValueError):
                pass
        else:
            def emit():
                try:
                    os.write(fd, (message + '\n').encode(errors='replace'))
                except OSError:
                    pass
            # A full caller pipe must not block recovery or a log reader.
            threading.Thread(target=emit, daemon=True).start()

    def open_log(self, reservation):
        """Keep each ticket's output, independently of its caller's terminal."""
        try:
            directory = self.directory / 'run-logs'
            directory.mkdir(mode=0o700, exist_ok=True)
            if directory.is_symlink():
                raise OSError('run-logs must be a real directory')
            directory.chmod(0o700)
            digest = hashlib.sha256((self.session + '\0' + reservation['ticket']).encode()).hexdigest()
            path = directory / (digest + '.log')
            self.log_fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
            if not stat.S_ISREG(os.fstat(self.log_fd).st_mode):
                raise OSError('output log must be a regular file')
            os.fchmod(self.log_fd, 0o600)
            self.mark_pending(None, log_path=str(path.resolve()))
        except OSError as exc:
            self.log_error = f'output capture unavailable: {exc}'
            if self.log_fd is not None:
                os.close(self.log_fd)
                self.log_fd = None
            self.warning(f'output capture unavailable: {exc}')

    def capture(self, stream, stopped):
        # A slow/disconnected caller must not stall the payload or recovery.
        # Keep durable output first; live forwarding has a bounded queue.
        live = queue.Queue(maxsize=128)
        complete = threading.Event()
        def forward():
            try:
                output_fd = sys.stdout.fileno()
            except (AttributeError, OSError, ValueError):
                output_fd = None
            while True:
                try:
                    chunk = live.get(timeout=.1)
                except queue.Empty:
                    if complete.is_set():
                        return
                    continue
                try:
                    if output_fd is not None:
                        # Never own Python's buffered stdout lock in a daemon:
                        # a blocked caller must not hang interpreter shutdown.
                        remaining = memoryview(chunk)
                        while remaining:
                            written = os.write(output_fd, remaining)
                            if not written:
                                return
                            remaining = remaining[written:]
                    else:
                        target = getattr(sys.stdout, 'buffer', sys.stdout)
                        target.write(chunk if hasattr(sys.stdout, 'buffer') else chunk.decode(errors='replace'))
                        target.flush()
                except (OSError, ValueError):
                    return
        forwarding = threading.Thread(target=forward, daemon=True)
        forwarding.start()
        drain_deadline = None
        try:
            while True:
                if stopped.is_set():
                    if drain_deadline is None:
                        drain_deadline = time.monotonic() + .2
                    elif time.monotonic() >= drain_deadline:
                        self.log_error = 'output capture stopped: descendants retained the completed command output pipe'
                        break
                readable, _, _ = select.select([stream], [], [], .1)
                if not readable:
                    if stopped.is_set():
                        break
                    continue
                chunk = os.read(stream.fileno(), 65536)
                if not chunk:
                    break
                if self.log_fd is not None:
                    try:
                        remaining = memoryview(chunk)
                        while remaining:
                            written = os.write(self.log_fd, remaining)
                            if not written:
                                raise OSError('output log write made no progress')
                            remaining = remaining[written:]
                    except OSError as exc:
                        self.log_error = f'output capture failed: {exc}'
                        self.warning(f'output capture failed: {exc}')
                        failed_fd = self.log_fd
                        self.log_fd = None
                        try:
                            os.close(failed_fd)
                        except OSError:
                            pass
                try:
                    live.put_nowait(chunk)
                except queue.Full:
                    pass
        except (OSError, ValueError) as exc:
            self.log_error = f'output reader failed: {exc}'
            self.warning(f'output reader failed: {exc}')
        finally:
            stream.close()
            complete.set()
            forwarding.join(timeout=.2)

    @contextmanager
    def lock(self):
        with (self.directory / '.lock').open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            yield

    def event(self, event, **details):
        try:
            with self.lock():
                with (self.directory / 'lifecycle.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(t=time.time(), session=self.session, event=event,
                                                 runner=str(self.repo), **details)) + '\n')
        except OSError as exc:
            self.warning(f'lifecycle record {event}: {exc}')

    def call(self, *args):
        # Control/recovery calls must drain output too, but a cancellation of
        # the payload must not interrupt release, admission or recovery work.
        return self.execute(['bash', self.fleet, *args], self.env, interruptible=False)

    def held(self):
        try:
            return (self.directory / 'holder').read_text().split('|')[0] == self.session
        except FileNotFoundError:
            return False

    def signal(self, signum, _frame):
        self.stopping = 128 + signum
        if self.child and self.child_interruptible and self.child.poll() is None:
            os.killpg(self.child.pid, signum)

    def execute(self, command, env, cwd=None, *, interruptible=True):
        self.child_interruptible = interruptible
        self.child = subprocess.Popen(command, env=env, cwd=cwd, start_new_session=True,
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        stopped = threading.Event()
        reader = threading.Thread(target=self.capture, args=(self.child.stdout, stopped), daemon=True)
        reader.start()
        deadline = None
        while self.child.poll() is None:
            if self.stopping and interruptible:
                if deadline is None:
                    os.killpg(self.child.pid, signal.SIGTERM)
                    deadline = time.monotonic() + 20
                elif time.monotonic() >= deadline:
                    os.killpg(self.child.pid, signal.SIGKILL)
            try:
                self.child.wait(timeout=.2)
            except subprocess.TimeoutExpired:
                pass
            (self.directory / ('hb.' + self.session)).touch()
        rc = self.child.returncode
        self.child = None
        self.child_interruptible = True
        stopped.set()
        # Descendants may retain the pipe; never wait for them to close it.
        reader.join(timeout=.5)
        return rc if rc >= 0 else 128 - rc

    def restore(self):
        """Compatibility entrypoint: sessions can only defer recovery."""
        self.event('restore-deferred', reason='central controller waits for 300 seconds of idle fleet')
        self.mark_pending('finishing', phase='release', recovery_policy='idle-controller',
                          recovery_deferred=True)
        return 0

    def cleanup_observation(self):
        if self.env.get('FLEET_OBSERVATION_CLONES')!='1':
            return 0
        old=self.stopping
        self.stopping=0
        handlers=[signal.signal(sig,signal.SIG_IGN) for sig in (signal.SIGINT,signal.SIGTERM)]
        self.event('observation-cleanup-start')
        try:
            while self.held():
                rc=self.execute(['python3',str(self.repo/'bench/fleet_observation_cleanup.py')],self.env)
                self.event('observation-cleanup-finished',rc=rc)
                if rc==0:return 0
                # Cleanup is idempotent. Keep the ticket and retry teardown;
                # never hand GPUs with remaining clones to the next workload.
                self.event('observation-cleanup-retry',delay_s=10)
                time.sleep(10)
            return 1
        finally:
            self.stopping=old
            for sig,handler in zip((signal.SIGINT,signal.SIGTERM),handlers):signal.signal(sig,handler)

    def finish(self):
        # Recovery belongs to the central idle controller. Releasing immediately
        # lets any queued workload continue, including probes and cancelled peers.
        self.event('restore-deferred', reason='central controller waits for 300 seconds of idle fleet')
        self.mark_pending('finishing', phase='release', recovery_policy='idle-controller',
                          recovery_deferred=True)
        return self.call('release', self.session)

    def mark_pending(self, state, **details):
        # A damaged edit record must not strand GPU ownership at teardown.
        try:
            with self.lock():
                pending.transition(self.directory, self.session, state, **details)
            return 0
        except (OSError, ValueError) as exc:
            self.warning(f'pending record {state}: {exc}')
            return 1

    def cleanup_reservation(self):
        # A failed/cancelled waiter can finish after its session name is reused.
        # Only its own receipt and queue row belong to this cleanup operation.
        pid = os.getpid()
        try:
            with self.lock():
                receipt = handoff.receipt(self.directory, self.session)
                value = handoff.read(receipt)
                if (isinstance(value, dict) and value.get('session') == self.session
                        and value.get('pid') == pid and handoff.live(value)):
                    receipt.unlink(missing_ok=True)
        except (OSError, ValueError) as exc:
            self.warning(f'ready receipt cleanup: {exc}')
        return self.call('withdraw', self.session, '--pid', str(pid))

    def accepted_payload(self, accepted):
        from fleet_prepare import command_environment
        payload, environment = command_environment(accepted['command'], self.env)
        # A literal env -i/-u may select payload settings, but the owning
        # supervisor still supplies fleet and recovery context.
        for key in ('FLEET_DIR', 'FLEET_SESSION', 'FLEET_PID', 'FLEET_RUNNER_REPO',
                    'FLEET_RESTORE_MANAGED', 'FLEET_NO_RESTORE_CHECK', 'FLEET',
                    'FLEET_VALIDATION_STORE', 'FLEET_VALIDATION_REQUIRED', 'FLEET_VALIDATION_LEVEL', 'FLEET_RECOVERY_RECEIPT',
                    'ST_LEASE_OWNER', 'ST_LEASE_PATH', 'FLEET_LEASE_PATH'):
            if key in self.env:
                environment[key] = self.env[key]
        # Edits may replace the prepared receipt while this supervisor waits.
        # Only the admitted record owns that selection, never the original
        # process environment or an env prefix supplied by the payload.
        if accepted.get('prepare_manifest'):
            environment['FLEET_PREPARE_MANIFEST'] = accepted['prepare_manifest']
        else:
            environment.pop('FLEET_PREPARE_MANIFEST', None)
        return payload, payload_environment(environment)

    def run(self):
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self.signal)
        with self.lock():
            reservation = pending.register(self.directory, self.session, self.command, self.fleet, self.kind)
            if self.kind == 'boot':
                handoff.ready(self.directory, self.session, os.getpid())
        rc = 1
        try:
            self.open_log(reservation)
            self.event('ready', protocol=handoff.PROTOCOL, source_sha256={name:hashlib.sha256((self.repo/'bench'/name).read_bytes()).hexdigest()
                       for name in ('fleet.sh', 'fleet_boot.py', 'fleet_handoff.py')})
            # Wait is interruptible and its parent identity is this supervisor.
            rc = self.execute(['bash', self.fleet, 'wait', self.session,
                               os.environ.get('FLEET_TIMEOUT_MIN', '720')], self.env)
            if not rc and not self.stopping:
                with self.lock():
                    accepted = pending.transition(self.directory, self.session, 'running',
                                                  phase='payload', started_at=time.time())
                self.event('accepted', revision=accepted['revision'])
                if self.kind == 'boot' and self.call('nodes') and os.environ.get('FLEET_NODES') == 'strict':
                    rc = 4
                elif not self.stopping:
                    from fleet_onepass import validate as validate_onepass
                    contract = validate_onepass(accepted['command'], accepted['cwd'], self.repo,
                                                environment=self.env, kind=self.kind)
                    payload, payload_env = self.accepted_payload(accepted)
                    if contract['entry'] == 'bench/onepass.py':
                        from onepass_deploy import ensure
                        ensure(Path(payload_env.get('REPO', accepted['cwd'])), live=True,
                               environment=payload_env)
                    rc = self.execute(payload, payload_environment(payload_env), accepted['cwd'])
                    self.mark_pending('running', phase='payload', payload_returncode=rc,
                                      payload_finished_at=time.time())
                self.event('payload-finished', rc=rc)
        except Exception as exc:
            rc = rc or 1
            self.warning(f'fleet execution failed: {exc}')
            self.mark_pending('finishing', phase='finishing', error=str(exc))
        finally:
            if self.mark_pending('finishing', phase='finishing'):
                rc = rc or 1
            cleanup_complete = True
            if self.held() and self.kind == 'boot':
                try:
                    cleanup_complete = False
                    if self.cleanup_observation():
                        raise RuntimeError('observation cleanup lost fleet ownership')
                    cleanup_complete = True
                    if self.finish():
                        rc = rc or 1
                except Exception as exc:
                    self.event('finish-error', reason=str(exc))
                    rc = rc or 1
            if self.held() and cleanup_complete:
                self.call('release', self.session)
            self.cleanup_reservation()
            result = self.stopping or rc
            if self.mark_pending('cancelled' if self.stopping else 'finished', phase='finished',
                                 finished_at=time.time(), returncode=result,
                                 outcome='cancelled' if self.stopping else 'failed' if result else 'succeeded',
                                 **({'log_error':self.log_error} if self.log_error else {})):
                rc = rc or 1
            if self.log_fd is not None:
                os.close(self.log_fd)
                self.log_fd = None
        return self.stopping or rc


if __name__ == '__main__':
    raise SystemExit(Supervisor(*sys.argv[1:5], sys.argv[5:]).run())
