"""A failed component cannot leave live compiler children beside the next GPU test."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from probes.engine_decode_bundle import ComponentStartError, run_component


@unittest.skipUnless(os.name == 'posix', 'fleet component groups require POSIX')
class ComponentProcessTests(unittest.TestCase):
    @staticmethod
    def command(pidfile):
        child = (
            "import os,signal,time; from pathlib import Path; "
            "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
            f"Path({str(pidfile)!r}).write_text(str(os.getpid())); time.sleep(60)"
        )
        parent = (
            "import subprocess,sys,time; "
            f"subprocess.Popen([sys.executable,'-c',{child!r}]); time.sleep(60)"
        )
        return [sys.executable, '-c', parent]

    @staticmethod
    def alive(pid):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        # Orphan grandchildren may briefly await PID 1's reap inside a container.
        try:
            return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0] != 'Z'
        except FileNotFoundError:
            return True

    def assert_stopped(self, pid):
        deadline = time.monotonic() + 3
        while self.alive(pid) and time.monotonic() < deadline:
            time.sleep(.02)
        self.assertFalse(self.alive(pid), f'component left child {pid} alive')

    def clean_child(self, path):
        if path.exists():
            pid = int(path.read_text())
            if self.alive(pid):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def test_timeout_kills_grandchild_and_preserves_unrelated_process_and_handlers(self):
        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'child.pid'
            outsider = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],
                                        start_new_session=True)
            try:
                code = run_component(self.command(path), cwd=directory, timeout=2)
                self.assertEqual(code, 124)
                self.assertTrue(path.exists(), 'child did not reach its signal handler')
                self.assert_stopped(int(path.read_text()))
                self.assertIsNone(outsider.poll())
                self.assertEqual(handlers, {sig: signal.getsignal(sig) for sig in handlers})
                self.assertEqual(run_component([sys.executable, '-c', 'raise SystemExit(7)'],
                                               cwd=directory, timeout=2), 7)
            finally:
                self.clean_child(path)
                outsider.kill()
                outsider.wait()

    def test_launch_error_restores_signal_handlers(self):
        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ComponentStartError):
                run_component([str(Path(directory) / 'missing')], cwd=directory, timeout=2)
        self.assertEqual(handlers, {sig: signal.getsignal(sig) for sig in handlers})

    def test_bundle_cancellation_kills_child_group_and_exits_without_next_component(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            path, later = Path(directory) / 'child.pid', Path(directory) / 'later'
            script = (
                'from pathlib import Path; from probes.engine_decode_bundle import run_component; '
                f'run_component({self.command(path)!r},cwd={directory!r},timeout=60); '
                f'Path({str(later)!r}).touch()'
            )
            process = subprocess.Popen([sys.executable, '-c', script], cwd=root, start_new_session=True)
            try:
                deadline = time.monotonic() + 5
                while not path.exists() and time.monotonic() < deadline:
                    time.sleep(.02)
                self.assertTrue(path.exists(), 'component did not start')
                process.send_signal(signal.SIGTERM)
                self.assertEqual(process.wait(timeout=5), 128 + signal.SIGTERM)
                self.assert_stopped(int(path.read_text()))
                self.assertFalse(later.exists())
            finally:
                self.clean_child(path)
                if process.poll() is None:
                    process.kill()
                    process.wait()


if __name__ == '__main__':
    unittest.main()
