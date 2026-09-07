#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Supervise a boot payload and finish once, including nested/failed runners."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import fleet_handoff as handoff


class Supervisor:
    def __init__(self, fleet, session, estimate, note, command):
        self.fleet, self.session, self.estimate, self.note, self.command = fleet, session, estimate, note, command
        self.directory = Path(os.environ['FLEET_DIR'])
        self.repo = Path(__file__).resolve().parent.parent
        self.env = dict(os.environ, FLEET_PID=str(os.getpid()), FLEET_SESSION=session,
                        FLEET_RESTORE_MANAGED='1', FLEET_RUNNER_REPO=str(self.repo),
                        FLEET=fleet, FLEET_NO_RESTORE_CHECK='1')
        self.child = None
        self.stopping = 0

    @contextmanager
    def lock(self):
        with (self.directory / '.lock').open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            yield

    def event(self, event, **details):
        with self.lock():
            with (self.directory / 'lifecycle.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(t=time.time(), session=self.session, event=event,
                                             runner=str(self.repo), **details)) + '\n')

    def call(self, *args):
        return subprocess.call(['bash', self.fleet, *args], env=self.env)

    def held(self):
        try:
            return (self.directory / 'holder').read_text().split('|')[0] == self.session
        except FileNotFoundError:
            return False

    def signal(self, signum, _frame):
        self.stopping = 128 + signum
        if self.child and self.child.poll() is None:
            os.killpg(self.child.pid, signum)

    def execute(self, command, env):
        self.child = subprocess.Popen(command, env=env, start_new_session=True)
        deadline = None
        while self.child.poll() is None:
            if self.stopping:
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
        return rc if rc >= 0 else 128 - rc

    def restore(self):
        start = time.monotonic()
        self.event('restore-start')
        # Cancellation stops the experiment, never the final recovery command.
        old = self.stopping
        self.stopping = 0
        handlers = [signal.signal(sig, signal.SIG_IGN) for sig in (signal.SIGINT, signal.SIGTERM)]
        try:
            rc = self.execute(['bash', str(self.repo / 'bench/fleet_restore.sh')], self.env)
        finally:
            self.stopping = old
            for sig, handler in zip((signal.SIGINT, signal.SIGTERM), handlers):
                signal.signal(sig, handler)
        self.event('restore-finished', rc=rc, seconds=time.monotonic() - start)
        if rc == 0:
            with self.lock():
                handoff.clear(self.directory, self.session)
        return rc

    def finish(self):
        with self.lock():
            handoff.claim_held(self.directory, self.session, os.getpid())
            target = handoff.offer(self.directory, self.session)
        if not target:
            return self.restore()
        self.event('handoff-offered', successor=target['session'])
        if self.call('release', self.session):
            return self.restore()
        # Keep the donor alive until acceptance, including cancel-before-GO.
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            with self.lock():
                debt = handoff.read(self.directory / 'restore-debt.json')
            if not debt or debt['owner']['session'] != self.session:
                self.event('handoff-accepted', successor=target['session'])
                return 0
            if not handoff.live(target) or not any(r[1] == target['session'] for r in handoff.rows(self.directory)):
                break
            time.sleep(.2)
        # No receiver: reclaim responsibility through the same queue/legacy
        # checks. The debt blocks probes until a boot supervisor accepts it.
        self.event('handoff-reclaim', successor=target['session'])
        with self.lock():
            debt = handoff.read(self.directory / 'restore-debt.json')
            if not debt or debt['owner']['session'] != self.session:
                return 0
            debt.pop('target', None)
            handoff.write(self.directory / 'restore-debt.json', debt)
        if self.call('request', self.session, '15', 'recover unaccepted handoff'):
            return 1
        self.call('front', self.session)
        if self.call('wait', self.session, '5'):
            return 1
        return self.restore()

    def run(self):
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self.signal)
        with self.lock():
            handoff.ready(self.directory, self.session, os.getpid())
        self.event('ready', protocol=handoff.PROTOCOL, source_sha256={name:hashlib.sha256((self.repo/'bench'/name).read_bytes()).hexdigest()
                   for name in ('fleet.sh', 'fleet_boot.py', 'fleet_handoff.py')})
        rc = 1
        try:
            # Wait is interruptible and its parent identity is this supervisor.
            rc = self.execute(['bash', self.fleet, 'wait', self.session,
                               os.environ.get('FLEET_TIMEOUT_MIN', '720')], self.env)
            if not rc and not self.stopping:
                self.event('accepted')
                if self.call('nodes') and os.environ.get('FLEET_NODES') == 'strict':
                    rc = 4
                elif not self.stopping:
                    rc = self.execute(self.command, self.env)
                self.event('payload-finished', rc=rc)
        finally:
            if self.held():
                try:
                    if self.finish():
                        rc = rc or 1
                except Exception as exc:
                    self.event('finish-error', reason=str(exc))
                    rc = rc or 1
                    self.restore()
                if self.held():
                    self.call('release', self.session)
            with self.lock():
                handoff.receipt(self.directory, self.session).unlink(missing_ok=True)
            # Remove a cancelled waiter's row without signalling ourselves.
            self.call('withdraw', self.session)
        return self.stopping or rc


if __name__ == '__main__':
    raise SystemExit(Supervisor(*sys.argv[1:5], sys.argv[5:]).run())
