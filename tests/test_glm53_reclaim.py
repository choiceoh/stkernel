"""Exercise the launcher's reclaim stage with fake Docker/SSH only."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "launchers/start-glm53-nvfp4-tp4.sh"
BASH = shutil.which("bash") or "/bin/bash"
NODES = ("head", "w1", "w2", "w3")
MOCK = r'''
import json, os
from pathlib import Path
import subprocess, sys, time
root = Path(os.environ['MOCK_ROOT'])
node = os.environ.get('MOCK_NODE', 'head')
if Path(sys.argv[0]).name == 'ssh':
    node = sys.argv[-2].split('@')[-1]
    if os.environ.get('MOCK_SSH_FAIL') == node:
        sys.exit(255)
    sys.exit(subprocess.call([os.environ['MOCK_BASH'], '-c', sys.argv[-1]],
                            env=dict(os.environ, MOCK_NODE=node)))
cmd = sys.argv[1]
def record(stage):
    (root / (node + '.' + stage)).write_text(str(time.monotonic()))
if cmd == 'rm':
    record('stop.start')
    time.sleep(float(os.environ.get('MOCK_DELAY', '.15')))
    record('stop.done')
    sys.exit(1 if os.environ.get('MOCK_MISSING') == node else 0)
elif cmd == 'ps':
    if os.environ.get('MOCK_DAEMON_FAIL') == node:
        sys.exit(13)
    if '-q' in sys.argv:
        if os.environ.get('MOCK_PRIOR_BUILDER') == node:
            print('old-unnamed-builder')
    elif os.environ.get('MOCK_REMAINS') == node:
        print('glm53-worker')
    else:
        assert 'name=^/(glm53|glm53-worker)$' in sys.argv
elif cmd == 'inspect':
    assert sys.argv[-1] == 'old-unnamed-builder'
    if os.environ.get('MOCK_INSPECT_FAIL') == node:
        sys.exit(15)
    print(os.environ.get('MOCK_BUILDER_CACHE', '/mock/cache'))
elif cmd == 'run':
    assert (root / (node + '.stop.done')).exists(), 'cleared locks before stop'
    record('lock.start')
    (root / (node + '.argv')).write_text(json.dumps(sys.argv[1:]))
    time.sleep(float(os.environ.get('MOCK_DELAY', '.15')))
    record('lock.done')
    sys.exit(14 if os.environ.get('MOCK_LOCK_FAIL') == node else 0)
else:
    raise AssertionError(sys.argv)
'''


def run_reclaim(source=None, prebuild="0", terminate=False, **settings):
    source = source or LAUNCHER.read_text()
    start = source.index('\n', source.index('if [ "${SKIP_PREFLIGHT:-0}"')) + 1
    end = source.index('  echo "== memfree preflight =="', start)
    block = source[start:end]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for tool in ("ssh", "docker"):
            path = root / tool
            path.write_text("#!" + sys.executable + "\n" + MOCK)
            path.chmod(0o755)
        env = dict(os.environ, PATH=str(root) + os.pathsep + os.environ["PATH"],
                   MOCK_ROOT=directory, MOCK_BASH=BASH, PREBUILD=prebuild, **settings)
        setup = '''set -euo pipefail
HEAD_IP=head
WORKER_IPS=(w1 w2 w3)
NAME_HEAD=glm53
NAME_WORKER=glm53-worker
CACHE_HOST_PATH=/mock/cache
IMAGE=mock-image
SSHOPT=""
'''
        start_time = time.monotonic()
        proc = subprocess.Popen([BASH, "-c", setup + block + '\necho SIZE_READY'],
                                env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            if terminate:
                deadline = time.monotonic() + 4
                while not all((root / (node + ".stop.start")).exists() for node in NODES):
                    if time.monotonic() >= deadline:
                        raise AssertionError("nodes did not start")
                    time.sleep(.01)
                proc.send_signal(signal.SIGTERM)
            out, err = proc.communicate(timeout=15)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.communicate()
        files = {p.name: p.read_text() for p in root.iterdir() if "." in p.name}
        return (subprocess.CompletedProcess(proc.args, proc.returncode, out, err),
                files, time.monotonic() - start_time)


class ReclaimTests(unittest.TestCase):
    def test_nodes_overlap_and_stop_before_clearing_local_locks(self):
        result, files, _ = run_reclaim()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("SIZE_READY", result.stdout)
        self.assertLess(max(float(files[n + ".stop.start"]) for n in NODES),
                        min(float(files[n + ".lock.done"]) for n in NODES))
        for node in NODES:
            self.assertLess(float(files[node + ".stop.done"]), float(files[node + ".lock.start"]))
            argv = json.loads(files[node + ".argv"])
            self.assertIn("/mock/cache:/cache", argv)
            self.assertEqual(argv[-1], "rm -f /cache/mk_build/*/lock /cache/osar_build/*/lock")

    def test_absent_containers_are_successful_reclaim(self):
        result, files, _ = run_reclaim(MOCK_MISSING="w1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("w1.lock.done", files)

    def test_failed_node_blocks_sizing_and_joins_other_nodes(self):
        for key in ("MOCK_SSH_FAIL", "MOCK_DAEMON_FAIL", "MOCK_REMAINS", "MOCK_LOCK_FAIL"):
            with self.subTest(key=key):
                result, files, _ = run_reclaim(**{key: "w1"})
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertNotIn("SIZE_READY", result.stdout)
                for node in ("head", "w2", "w3"):
                    self.assertIn(node + ".lock.done", files)
                if key != "MOCK_LOCK_FAIL":
                    self.assertNotIn("w1.lock.start", files)

    def test_prebuild_owns_its_locks(self):
        result, files, _ = run_reclaim(prebuild="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        for node in NODES:
            self.assertIn(node + ".stop.done", files)
            self.assertNotIn(node + ".lock.start", files)

    def test_signal_joins_every_reclaim_without_starting_sizing(self):
        result, files, _ = run_reclaim(terminate=True, MOCK_DELAY=".3")
        self.assertEqual(result.returncode, 143, result.stderr)
        self.assertNotIn("SIZE_READY", result.stdout)
        for node in NODES:
            self.assertIn(node + ".lock.done", files)

    def test_previous_prebuild_survives_default_launcher_without_losing_locks(self):
        result, files, _ = run_reclaim(prebuild="0", MOCK_PRIOR_BUILDER="w2")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("retaining extension build locks", result.stdout)
        self.assertNotIn("w2.lock.start", files)
        for node in ("head", "w1", "w3"):
            self.assertIn(node + ".lock.done", files)

    def test_unrelated_cache_user_does_not_block_lock_recovery(self):
        result, files, _ = run_reclaim(MOCK_PRIOR_BUILDER="w2", MOCK_BUILDER_CACHE="/other/cache")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("w2.lock.done", files)

    def test_unknown_cache_ownership_blocks_cleanup_and_sizing(self):
        result, files, _ = run_reclaim(MOCK_PRIOR_BUILDER="w2", MOCK_INSPECT_FAIL="w2")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertNotIn("SIZE_READY", result.stdout)
        self.assertNotIn("w2.lock.start", files)
        for node in ("head", "w1", "w3"):
            self.assertIn(node + ".lock.done", files)


if __name__ == "__main__":
    unittest.main()
