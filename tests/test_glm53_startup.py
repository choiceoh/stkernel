"""Exercise the actual launch block with local fake Docker/SSH, no fleet access."""
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

MOCK = r'''
import base64, json, os
from pathlib import Path
import subprocess, sys, time
tool = Path(sys.argv[0]).name
node = os.environ.get("MOCK_NODE", "head")
root = Path(os.environ["MOCK_ROOT"])
if tool == "base64":
    sys.stdout.buffer.write(base64.b64encode(sys.stdin.buffer.read()))
elif tool == "mkdir":
    sys.exit(7 if node == os.environ.get("MOCK_MKDIR_FAIL") else 0)
elif tool == "ssh":
    node = sys.argv[-2].split("@")[-1]
    if node == os.environ.get("MOCK_SSH_FAIL"):
        sys.exit(255)
    env = dict(os.environ, MOCK_NODE=node)
    sys.exit(subprocess.call([os.environ["MOCK_BASH"], "-c", sys.argv[-1]], env=env))
elif tool == "docker" and sys.argv[1] == "run":
    (root / (node + ".start")).write_text(str(time.monotonic()))
    (root / (node + ".argv")).write_text(json.dumps(sys.argv[1:]))
    if node != "head":
        if os.environ.get("MOCK_BARRIER") == "1":
            deadline = time.monotonic() + 3
            while not all((root / (n + ".start")).exists() for n in ("w1", "w2", "w3")):
                if time.monotonic() > deadline:
                    sys.exit(8)
                time.sleep(.01)
        time.sleep(float(os.environ.get("MOCK_DELAY", ".15")))
        (root / (node + ".done")).write_text(str(time.monotonic()))
        if node == os.environ.get("MOCK_DOCKER_FAIL"):
            sys.exit(9)
    else:
        (root / "head.joined").write_text(str(all((root / (n + ".done")).exists() for n in ("w1", "w2", "w3"))))
'''


def run_launch(source=None, terminate=False, **settings):
    source = source or LAUNCHER.read_text()
    block = source[source.index("# Workers first"):source.index("# deneb fork (prefill-warmup)")]
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        for tool in ("ssh", "docker", "mkdir", "base64"):
            path = root / tool
            path.write_text("#!" + sys.executable + "\n" + MOCK)
            path.chmod(0o755)
        env = dict(os.environ, PATH=str(root) + os.pathsep + os.environ["PATH"],
                   MOCK_ROOT=str(root), MOCK_BASH=BASH, **settings)
        setup = '''set -euo pipefail
WORKER_IPS=(w1 w2 w3)
SSHOPT=""
NAME_WORKER=glm53-worker
NAME_HEAD=glm53
COMMON="-d -v CACHEDIR:/cache"
ENVV='-e VLLM_TEST={"m":1,"n":2}'
CACHE_HOST_PATH=/mock/cache
LOG_HOST_DIR=/mock/logs
CT_GID_PRELUDE='export NCCL_IB_GID_INDEX=3'
SERVE_ARGS='model --master-addr head --nnodes 4'
IMAGE=mock-image
HEAD_IP=head
PORT=8000
'''
        start = time.monotonic()
        proc = subprocess.Popen([BASH, "-c", setup + block], env=env,
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            if terminate:
                deadline = time.monotonic() + 5
                while not all((root / (n + ".start")).exists() for n in ("w1", "w2", "w3")):
                    if time.monotonic() >= deadline:
                        raise AssertionError("workers did not start")
                    time.sleep(.01)
                proc.send_signal(signal.SIGTERM)
            stdout, stderr = proc.communicate(timeout=15)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.communicate()
        result = subprocess.CompletedProcess(proc.args, proc.returncode, stdout, stderr)
        elapsed = time.monotonic() - start
        files = {p.name: p.read_text() for p in root.iterdir() if "." in p.name}
        return result, files, elapsed


class WorkerLaunchTests(unittest.TestCase):
    def test_workers_overlap_and_all_finish_before_head(self):
        result, files, _ = run_launch(MOCK_BARRIER="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(files["head.joined"], "True")
        self.assertLess(max(float(files[n + ".start"]) for n in ("w1", "w2", "w3")),
                        min(float(files[n + ".done"]) for n in ("w1", "w2", "w3")))
        for rank, node in enumerate(("w1", "w2", "w3"), 1):
            argv = json.loads(files[node + ".argv"])
            self.assertIn('VLLM_TEST={"m":1,"n":2}', argv)
            self.assertIn("VLLM_HOST_IP=" + node, argv)
            import base64
            payload = base64.b64decode(argv[-1].split()[1]).decode()
            self.assertIn(f"--node-rank {rank} --headless", payload)
            self.assertLess(payload.index("export NCCL_IB_GID_INDEX"), payload.index("vllm serve"))

    def test_failed_worker_prevents_head_and_joins_other_starts(self):
        result, files, _ = run_launch(MOCK_DOCKER_FAIL="w1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("worker rank=1 @w1 failed", result.stderr)
        self.assertNotIn("head.start", files)
        for node in ("w1", "w2", "w3"):
            self.assertIn(node + ".done", files)

    def test_ssh_or_directory_failure_prevents_head(self):
        for setting in ("MOCK_SSH_FAIL", "MOCK_MKDIR_FAIL"):
            with self.subTest(setting=setting):
                result, files, _ = run_launch(**{setting: "w2"})
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("w2.start", files)
                self.assertNotIn("head.start", files)
                self.assertIn("w1.done", files)
                self.assertIn("w3.done", files)

    def test_termination_joins_inflight_starts_and_does_not_launch_head(self):
        result, files, _ = run_launch(terminate=True, MOCK_DELAY="0.5")
        self.assertEqual(result.returncode, 143, result.stderr)
        self.assertNotIn("head.start", files)
        for node in ("w1", "w2", "w3"):
            self.assertIn(node + ".done", files)


if __name__ == "__main__":
    unittest.main()
