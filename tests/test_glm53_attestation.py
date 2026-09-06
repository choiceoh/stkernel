"""Check the real GLM file-attestation block without SSH or Docker access."""
import json
import os
from pathlib import Path
import shlex
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
WORKERS = ("w1", "w2", "w3")
START = "# Verify the head once, then the independent workers concurrently."
OLD_START = "# Refuse to start over a live dsv4/q38 stack. Skipped under DRY_RUN:"
END = "# The manifest pins each target's sha256 as the image ships it."

MOCK = r'''
import hashlib, json, os
from pathlib import Path
import shutil, subprocess, sys, time
root = Path(os.environ["ATTEST_ROOT"])
node = os.environ.get("ATTEST_NODE", "head")
command = Path(sys.argv[0]).name
def event(kind):
    with (root / "events").open("a") as stream:
        stream.write(f"{kind} {node}\n")
if command == "sha256sum":
    event("hash")
    rc = 0
    for name in sys.argv[1:]:
        try:
            digest = hashlib.sha256(Path(name).read_bytes()).hexdigest()
            print(f"{digest}  {name}")
        except OSError:
            rc = 1
    raise SystemExit(23 if node == os.environ.get("ATTEST_HASH_FAIL") else rc)
if command == "rm":
    path = Path(sys.argv[-1]).resolve()
    assert path.parent == (root / "tmp").resolve(), path
    event("cleanup")
    shutil.rmtree(path)
    raise SystemExit(0)
if command == "bash":
    argv = sys.argv[1:]
    if node != "head" and len(argv) >= 5 and argv[2] == "sh":
        assert argv[3] == os.environ["ATTEST_MODEL"], argv
        assert argv[4] == os.environ["ATTEST_OVERLAY"], argv
        argv[3] = str(root / node / "model")
        argv[4] = str(root / node / "overlay")
    raise SystemExit(subprocess.call([os.environ["ATTEST_BASH"], *argv]))
assert command == "ssh", command
node = sys.argv[-2].split("@")[-1]
event("start")
(root / (node + ".start")).touch()
if os.environ.get("ATTEST_BARRIER") == "1":
    deadline = time.monotonic() + 3
    while not all((root / (n + ".start")).exists() for n in ("w1", "w2", "w3")):
        if time.monotonic() >= deadline:
            raise SystemExit(98)
        time.sleep(.005)
time.sleep(float(os.environ.get("ATTEST_DELAY", "0")))
if node == os.environ.get("ATTEST_SSH_FAIL"):
    rc = 255
else:
    env = dict(os.environ, ATTEST_NODE=node)
    rc = subprocess.call([os.environ["ATTEST_BASH"], "-c", sys.argv[-1]], env=env)
event("finish")
raise SystemExit(rc)
'''


def run_attestation(*, source=None, dry_run=False, terminate=False,
                    missing_model=None, missing_overlay=None, skew=None,
                    quoted_paths=False, **settings):
    source = source or LAUNCHER.read_text()
    start = START if START in source else OLD_START
    block = source[source.index(start):source.index(END)]
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        bindir = root / "bin"
        bindir.mkdir()
        (root / "tmp").mkdir()
        for command in ("ssh", "bash", "sha256sum", "rm"):
            path = bindir / command
            path.write_text("#!" + sys.executable + "\n" + MOCK)
            path.chmod(0o755)
        suffix = " path ' $()" if quoted_paths else ""
        model = root / ("model" + suffix)
        overlay = root / ("overlay" + suffix)
        for node in ("head", *WORKERS):
            node_model = model if node == "head" else root / node / "model"
            node_overlay = overlay if node == "head" else root / node / "overlay"
            node_model.mkdir(parents=True)
            node_overlay.mkdir(parents=True)
            if node != missing_model:
                (node_model / "config.json").write_text("{}")
            (node_overlay / "manifest.tsv").write_text("model.py\ttarget\tabsent\n")
            if node != missing_overlay:
                (node_overlay / "model.py").write_text("skew" if node == skew else "same")
        env = dict(os.environ, PATH=str(bindir) + os.pathsep + os.environ["PATH"],
                   TMPDIR=str(root / "tmp"), ATTEST_ROOT=str(root), ATTEST_BASH=BASH,
                   ATTEST_MODEL=str(model), ATTEST_OVERLAY=str(overlay), **settings)
        setup = f'''set -euo pipefail
HEAD_IP=head
WORKER_IPS=(w1 w2 w3)
SSHOPT=""
MODEL_HOST_PATH={shlex.quote(str(model))}
OVERLAY_DIR={shlex.quote(str(overlay))}
OVFILES=(model.py)
DRY_RUN={int(dry_run)}
'''
        start_time = time.monotonic()
        proc = subprocess.Popen([BASH, "-c", setup + block + "printf 'verified\\n'\n"],
                                env=env, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        try:
            if terminate:
                deadline = time.monotonic() + 5
                while not all((root / (n + ".start")).exists() for n in WORKERS):
                    if time.monotonic() >= deadline or proc.poll() is not None:
                        raise AssertionError("worker checks did not start")
                    time.sleep(.005)
                proc.send_signal(signal.SIGTERM)
            stdout, stderr = proc.communicate(timeout=15)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.communicate()
        elapsed = time.monotonic() - start_time
        events = (root / "events").read_text().splitlines() if (root / "events").exists() else []
        result = subprocess.CompletedProcess(proc.args, proc.returncode, stdout, stderr)
        return result, events, elapsed, list((root / "tmp").iterdir())


class AttestationTests(unittest.TestCase):
    def assert_joined(self, events):
        cleanup = events.index("cleanup head")
        for node in WORKERS:
            self.assertLess(events.index("finish " + node), cleanup)

    def test_head_hashed_once_workers_overlap_and_join(self):
        result, events, _, leftovers = run_attestation(ATTEST_BARRIER="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "verified\n")
        self.assertEqual(events.count("hash head"), 1)
        self.assertEqual(sum(e.startswith("start ") for e in events), 3)
        self.assertLess(max(events.index("start " + n) for n in WORKERS),
                        min(events.index("finish " + n) for n in WORKERS))
        self.assert_joined(events)
        self.assertEqual(leftovers, [])

    def test_missing_head_model_or_overlay_stops_before_ssh(self):
        for key in ("missing_model", "missing_overlay"):
            with self.subTest(key=key):
                result, events, _, _ = run_attestation(**{key: "head"})
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("verified", result.stdout)
                self.assertFalse(any(e.startswith("start ") for e in events))

    def test_worker_errors_join_all_checks_and_prevent_progress(self):
        for key in ("missing_model", "missing_overlay", "skew",
                    "ATTEST_SSH_FAIL", "ATTEST_HASH_FAIL"):
            with self.subTest(key=key):
                result, events, _, leftovers = run_attestation(**{key: "w1"})
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("verified", result.stdout)
                self.assertIn("w1", result.stderr)
                self.assert_joined(events)
                self.assertEqual(leftovers, [])

    def test_head_hash_command_failure_is_not_accepted_as_valid_output(self):
        result, events, _, _ = run_attestation(ATTEST_HASH_FAIL="head")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("verified", result.stdout)
        self.assertFalse(any(e.startswith("start ") for e in events))

    def test_model_and_overlay_paths_survive_remote_shell_quoting(self):
        result, _, _, _ = run_attestation(quoted_paths=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "verified\n")

    def test_dry_run_does_not_hash_or_contact_nodes(self):
        result, events, _, _ = run_attestation(dry_run=True, missing_model="head")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(events, [])

    def test_term_waits_for_existing_checks_before_exit(self):
        result, events, _, leftovers = run_attestation(
            terminate=True, ATTEST_BARRIER="1", ATTEST_DELAY="0.5")
        self.assertEqual(result.returncode, 143, result.stderr)
        self.assertNotIn("verified", result.stdout)
        self.assert_joined(events)
        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
