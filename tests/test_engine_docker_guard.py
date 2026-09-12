#!/usr/bin/env python3
"""The docker in front of docker: a leased rank cannot be removed by reflex.

Run against a fake `docker`, so this needs no daemon and no fleet. What it pins is the
judgement -- which commands the guard looks at, which containers it protects, and the two
ways through it -- because the failure it exists for is silent: one `docker rm -f` on one
node and the other three ranks die at the rendezvous with nothing to say why.
"""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
GUARD = ROOT / "launchers/docker-fleet-guard.sh"

# `docker inspect --format '{{.State.Running}}{{println}}{{range .Config.Env}}...'` is the
# only shape the guard asks for; everything else the fake reports as passed through.
FAKE = """#!/bin/bash
if [ "${1:-}" = inspect ]; then
  for a in "$@"; do
    case "$a" in
      running-leased)  printf 'true\\nPATH=/usr/bin\\nST_LEASE_OWNER=someone@srv1/42\\n'; exit 0;;
      stopped-leased)  printf 'false\\nST_LEASE_OWNER=someone@srv1/42\\n'; exit 0;;
      running-plain)   printf 'true\\nPATH=/usr/bin\\n'; exit 0;;
    esac
  done
  exit 1
fi
echo "PASSTHROUGH $*"
"""


class DockerGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fake = Path(self.tmp.name) / "docker"
        self.fake.write_text(FAKE)
        self.fake.chmod(0o755)

    def run_guard(self, *args, **environment):
        env = dict(os.environ, ST_DOCKER_REAL=str(self.fake))
        env.update(environment)
        return subprocess.run(["bash", str(GUARD), *args], capture_output=True, text=True, env=env)

    def assertPassedThrough(self, done, *, contains=""):
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("PASSTHROUGH", done.stdout)
        if contains:
            self.assertIn(contains, done.stdout)

    def test_a_running_leased_rank_is_not_removable_by_hand(self):
        """The 2026-09-12 shape: `ssh srv1 "docker rm -f st-glm53"` on someone else's fleet."""
        for command in (["rm", "-f", "running-leased"], ["kill", "running-leased"],
                        ["stop", "-t", "1", "running-leased"], ["restart", "running-leased"]):
            with self.subTest(command=command):
                done = self.run_guard(*command)
                self.assertEqual(done.returncode, 125)
                self.assertNotIn("PASSTHROUGH", done.stdout)
                self.assertIn("someone@srv1/42", done.stderr)
                self.assertIn("start-st-glm53.sh stop", done.stderr)     # what to run instead
                self.assertIn("ST_FLEET_OK=", done.stderr)               # and the way through

    def test_naming_the_owner_you_evict_is_the_way_through(self):
        done = self.run_guard("rm", "-f", "running-leased", ST_FLEET_OK="someone@srv1/42")
        self.assertPassedThrough(done, contains="rm -f running-leased")
        wrong = self.run_guard("rm", "-f", "running-leased", ST_FLEET_OK="somebody-else")
        self.assertEqual(wrong.returncode, 125)

    def test_a_stopped_rank_is_forensics_and_stays_removable(self):
        """Every boot clears the previous container before it starts; a guard that blocked
        that would stop the fleet from ever coming back."""
        self.assertPassedThrough(self.run_guard("rm", "-f", "stopped-leased"))

    def test_containers_without_a_lease_are_not_this_guard_s_business(self):
        for target in ("running-plain", "does-not-exist"):
            with self.subTest(target=target):
                self.assertPassedThrough(self.run_guard("rm", "-f", target))

    def test_everything_that_is_not_a_destructive_verb_goes_straight_through(self):
        for command in (["ps"], ["images", "-q"], ["inspect", "running-leased", "--format", "{{.Id}}"],
                        ["logs", "--tail", "5", "running-leased"], ["run", "-d", "img"]):
            with self.subTest(command=command):
                self.assertEqual(self.run_guard(*command).returncode, 0)

    def test_a_bare_rm_inside_a_run_payload_is_not_the_verb(self):
        """`docker run img rm -rf /` must not be read as `docker rm`."""
        self.assertPassedThrough(self.run_guard("run", "--rm", "img", "rm", "-rf", "/"))

    def test_global_flags_before_the_verb_are_skipped_not_guessed(self):
        done = self.run_guard("-H", "unix:///var/run/docker.sock", "rm", "-f", "running-leased")
        self.assertEqual(done.returncode, 125, done.stdout)
        self.assertPassedThrough(self.run_guard("--debug", "ps"))

    def test_it_never_becomes_the_outage_it_prevents(self):
        """A guard that cannot find the real docker must not be the reason docker stops
        working -- and a container it cannot inspect is not a container it may refuse."""
        env = dict(os.environ, ST_DOCKER_REAL=str(Path(self.tmp.name) / "not-here"),
                   PATH=self.tmp.name + os.pathsep + os.environ["PATH"])
        done = subprocess.run(["bash", str(GUARD), "rm", "-f", "running-leased"],
                              capture_output=True, text=True, env=env)
        self.assertEqual(done.returncode, 0)
        self.assertIn("PASSTHROUGH", done.stdout)


class InstallationTests(unittest.TestCase):
    def setUp(self):
        self.installer = (ROOT / "launchers/install-docker-fleet-guard.sh").read_text()
        self.launcher = (ROOT / "launchers/start-st-glm53.sh").read_text()

    def test_it_is_installed_where_ssh_actually_looks(self):
        """The incident's shape was `ssh srv1 "docker rm -f ..."`, and that PATH is
        /usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin -- so /usr/local/bin, not ~/.local/bin."""
        self.assertIn("TARGET=/usr/local/bin/docker", self.installer)
        self.assertIn("--uninstall", self.installer)

    def test_the_shim_answers_ps_before_it_is_installed(self):
        """It shadows docker on four nodes that serve: a broken one is an outage."""
        self.assertIn("bash /tmp/.docker-guard.$$ ps -q >/dev/null", self.installer)
        self.assertIn("sudo install -m 0755", self.installer)

    def test_the_sanctioned_stop_names_the_owner_it_evicts(self):
        self.assertIn("ST_FLEET_OK=", self.launcher)
        self.assertIn("refusing to stop another fleet owner's lease", self.launcher)


if __name__ == "__main__":
    unittest.main()
