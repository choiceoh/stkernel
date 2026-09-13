"""Exercise fleet ownership using real shell control flow and an isolated fake fleet."""
import os
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class FleetHarness(unittest.TestCase):
    """The isolated fake fleet: stubs for hostname, ssh, docker and curl, a copied launchers/ tree."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name)
        self.bin = self.home / "bin"
        self.bin.mkdir()
        self.repo = self.home / "repo"
        shutil.copytree(ROOT / "launchers", self.repo / "launchers")
        (self.repo / 'engine/base').mkdir(parents=True)
        shutil.copy2(ROOT / 'engine/base/fleet_lease.py', self.repo / 'engine/base/fleet_lease.py')
        self.lock = self.home / "fleet.lock"
        self.events = self.home / "events"
        self.fleet_dir = self.home / "fleet"
        self.fleet_dir.mkdir()
        # A boot says what it is (2026-09-12): these are a session's own boots by hand. The
        # supervisor overrides this with production for its own.
        self.env = dict(os.environ, PATH=f"{self.bin}:{os.environ['PATH']}",
                        FAKE_HOME=str(self.home), ST_FORENSICS=str(self.home / "forensics"),
                        ST_REPO=str(self.repo), ST_SUPERVISOR_ONCE="1", ST_LEASE_KIND="session",
                        CKPT=str(self.home / "missing-checkpoint"), FLEET_DIR=str(self.fleet_dir))
        self.script("hostname", "#!/bin/sh\necho 192.0.2.1\n")
        self.script("ssh", '''#!/usr/bin/env python3
import os, pathlib, subprocess, sys
h = pathlib.Path(os.environ['FAKE_HOME'])
cmd = sys.argv[-1].replace('/home/choiceoh/glm53-logs/st-fleet.lock', str(h / 'fleet.lock'))
cmd = cmd.replace('/home/choiceoh/st-fleet.lock', str(h / 'fleet.lock'))
if os.environ.get('FAKE_SSH_FAIL'):
    sys.exit(255)
if 'python3 - acquire' in cmd and os.environ.get('FAKE_RACE'):
    (h / 'fleet.lock').write_text('other-runner won the race')
sys.exit(subprocess.call(['/bin/bash', '-c', cmd], env=os.environ))
''')
        self.script("docker", '''#!/usr/bin/env python3
import os, pathlib, sys
a = sys.argv[1:]
if a[:2] == ['rm', '-f']:
    with (pathlib.Path(os.environ['FAKE_HOME']) / 'events').open('a') as f:
        f.write('stop\\n')
elif a and a[0] == 'ps':
    if '-q' in a:
        print('container-id')
    else:
        print(os.environ.get('FAKE_CONTAINERS', ''))
''')
        self.script("curl", '#!/bin/sh\necho \'{"data":[{"id":"glm-5.3-flash"}],"choices":[{}]}\'\n')

    def script(self, name, body):
        p = self.bin / name
        p.write_text(body)
        p.chmod(0o755)

    def run_script(self, name, *args):
        return subprocess.run(["bash", str(self.repo / "launchers" / name), *args],
                              env=self.env, text=True, capture_output=True, timeout=15)



class FleetOps(FleetHarness):
    def test_stop_preserves_foreign_owner_and_containers(self):
        self.lock.write_text("st-replay-other-session")
        result = self.run_script("start-st-glm53.sh", "stop")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.lock.read_text(), "st-replay-other-session")
        self.assertFalse(self.events.exists())

    def test_stop_releases_production_fleet(self):
        self.lock.write_text("choiceoh@srv2 st-glm53 2026-09-12")
        result = self.run_script("start-st-glm53.sh", "stop")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.events.read_text().splitlines(), ["stop"] * 4)
        self.assertFalse(self.lock.exists())

    def test_atomic_acquisition_preserves_race_winner(self):
        self.env["FAKE_RACE"] = "1"
        result = self.run_script("start-st-glm53.sh")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.lock.read_text(), "other-runner won the race")
        self.assertFalse(self.events.exists())

    def test_metadata_failure_releases_only_our_lock(self):
        result = self.run_script("start-st-glm53.sh")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.lock.exists())
        self.assertFalse(self.events.exists())

    def test_json_lease_is_owned_by_its_container_for_stop_and_supervision(self):
        self.lock.write_text(json.dumps(dict(owner='prod-release', container='st-glm53')))
        self.env['FAKE_CONTAINERS'] = 'st-glm53'
        result = self.run_script('st-glm53-supervisor.sh')
        self.assertEqual(result.stdout.strip(), 'healthy', result.stderr)
        result = self.run_script('start-st-glm53.sh', 'stop')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.lock.exists())
        self.assertEqual(self.events.read_text().splitlines(), ['stop'] * 4)

    def test_json_foreign_lease_is_preserved(self):
        text = json.dumps(dict(owner='other-task', container='st-probe'))
        self.lock.write_text(text)
        result = self.run_script('start-st-glm53.sh', 'stop')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.lock.read_text(), text)
        self.assertFalse(self.events.exists())

    def test_supervisor_dumps_no_forensics_while_the_fleet_is_someone_elses(self):
        """The forensics ring keeps ten dumps. A dump every 30 s while a session's tickets held the
        fleet (srv2, 2026-09-13 02:49-02:55) evicted the evidence of the last real failure, and the
        log said "fleet taken" twice a minute. The loop now asks fleet_taken before it dumps."""
        text = (Path(__file__).resolve().parents[1] / 'launchers/st-glm53-supervisor.sh').read_text()
        loop = text[text.rindex('while :; do'):]
        self.assertLess(loop.index('reason=$(wait_reason)'), loop.index('\n  forensics\n'), 'asked before the dump')
        self.assertIn('no forensics, no launch attempt', loop)
        self.assertIn('wait_logged=$key', loop, 'said once per reason')

    def test_supervisor_waits_for_foreign_lock(self):
        self.lock.write_text("st-replay-other-session")
        result = self.run_script("st-glm53-supervisor.sh")
        self.assertIn("fleet taken: lock:", result.stdout)
        self.assertEqual(self.lock.read_text(), "st-replay-other-session")
        self.assertFalse(self.events.exists())

    def test_supervisor_waits_for_other_st_container(self):
        self.env["FAKE_CONTAINERS"] = "st-glm53\nst-probe"
        result = self.run_script("st-glm53-supervisor.sh")
        self.assertIn("fleet taken:", result.stdout)
        self.assertIn("st-probe", result.stdout)
        self.assertFalse(self.events.exists())

    def test_supervisor_adopts_healthy_production(self):
        self.lock.write_text("choiceoh@srv2 st-glm53 2026-09-12")
        self.env["FAKE_CONTAINERS"] = "st-glm53\nunrelated"
        result = self.run_script("st-glm53-supervisor.sh")
        self.assertEqual(result.stdout.strip(), "healthy", result.stderr)
        self.assertFalse(self.events.exists())

    def test_supervisor_waits_through_a_deliberate_handover(self):
        """A drain is not a dead engine. It refuses new work on purpose while it parks what it
        holds and lets the lease go, and `base/serve.catalog` says so with an empty list and a
        503. Counting that as three failed health checks relaunches into the next holder -- and
        the window where the lease is released but not yet taken is where `fleet_taken` is blind
        too (45차 §56)."""
        self.lock.write_text("choiceoh@srv2 st-glm53 2026-09-12")
        self.env["FAKE_CONTAINERS"] = "st-glm53"
        self.script("curl", '#!/bin/sh\necho \'{"object": "list", "data": [], "status": "draining", '
                            '"handing_over_to": "another-session"}\'\n')
        result = self.run_script("st-glm53-supervisor.sh")
        self.assertEqual(result.stdout.strip(), "handing over: waiting", result.stderr)
        self.assertFalse(self.events.exists(), "nothing was stopped or relaunched")

    def test_a_door_that_is_simply_down_still_reads_as_a_launch(self):
        # The guard must not swallow a real outage: no draining status, no waiting.
        self.lock.write_text("choiceoh@srv2 st-glm53 2026-09-12")
        self.env["FAKE_CONTAINERS"] = "st-glm53"
        self.script("curl", '#!/bin/sh\nexit 7\n')
        result = self.run_script("st-glm53-supervisor.sh")
        self.assertIn("would launch", result.stdout)

    def test_supervisor_preserves_unreachable_fleet(self):
        self.env["FAKE_SSH_FAIL"] = "1"
        result = self.run_script("st-glm53-supervisor.sh")
        self.assertIn("head unreachable", result.stdout)
        self.assertFalse(self.events.exists())


GNU_DATE = subprocess.run(['date', '-d', '@0', '+%s'], capture_output=True).returncode == 0


@unittest.skipUnless(GNU_DATE, "the supervisor's container age needs GNU date: run this inside a Linux container")
class SupervisorLoopTests(FleetHarness):
    """The loop itself, two iterations with no sleep, against a stateful fake door and a fake launcher.
    What srv2 taught on 2026-09-13 02:49-04:20: a waiting loop must dump nothing and say things once,
    a boot in progress is adopted rather than restarted, a launch is done when a chat answers, and a
    fleet the queue just let go is not restored under the next ticket's feet."""

    def setUp(self):
        super().setUp()
        launcher = self.home / "launcher.sh"
        launcher.write_text('#!/bin/sh\necho "launch $*" >> "$FAKE_HOME/events"\ntouch "$FAKE_HOME/containers-up"\nexit 0\n')
        self.script("docker", '''#!/usr/bin/env python3
import os, pathlib, sys
a = sys.argv[1:]
h = pathlib.Path(os.environ['FAKE_HOME'])
up = os.environ.get('FAKE_CONTAINERS', '') or ((h / 'containers-up').exists() and 'st-glm53')
if a[:1] == ['inspect']:
    print(os.environ.get('FAKE_STARTED_AT', '2000-01-01T00:00:00Z')); sys.exit(0 if up else 1)
if a and a[0] == 'ps':
    if '-q' in a:
        if up: print('container-id')
    else:
        print(up or '')
''')
        self.script("curl", '''#!/usr/bin/env python3
import os, pathlib, sys
h = pathlib.Path(os.environ['FAKE_HOME'])
url = [a for a in sys.argv[1:] if a.startswith('http')][0]
def count(name):
    f = h / ('count-' + name); n = (int(f.read_text()) if f.exists() else 0) + 1; f.write_text(str(n)); return n
if '/v1/models' in url:
    if count('door') <= int(os.environ.get('FAKE_DOOR_DOWN_CALLS', '0')): sys.exit(22)
    print('{"data":[{"id":"glm-5.3-flash"}]}'); sys.exit(0)
if '/v1/chat/completions' in url:
    if count('chat') <= int(os.environ.get('FAKE_CHAT_FAIL_CALLS', '0')): sys.exit(22)
    print('{"choices":[{}]}'); sys.exit(0)
print('{}')
''')
        self.env.update(ST_SUPERVISOR_ONCE="0", ST_SUPERVISOR_LOOPS="2", ST_SUPERVISOR_SLEEP="0", ST_BOOT_POLL="0",
                        BOOT_GRACE="3", CHAT_TIMEOUT="1", FLEET_DIR=str(self.fleet_dir), ST_LAUNCHER=str(launcher))

    def loop(self, **env):
        self.env.update({k: str(v) for k, v in env.items()})
        return self.run_script("st-glm53-supervisor.sh")

    def activity(self, ago):
        import time
        (self.fleet_dir / "idle-recovery.json").write_text(json.dumps(dict(updated_at=time.time() - ago, reason="release")))

    def launches(self):
        return self.events.read_text().splitlines() if self.events.exists() else []

    def test_a_taken_fleet_is_waited_for_once_with_nothing_dumped_and_no_crash(self):
        self.lock.write_text("st-replay-other-session")
        out = self.loop(FAKE_DOOR_DOWN_CALLS=99)                 # production is down: no door, no chat
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertIn("loop bound reached", out.stdout, "the loop ran to its bound: no unbound variable killed it")
        self.assertEqual(out.stdout.count("no forensics, no launch attempt"), 1, out.stdout)
        self.assertNotIn("health check failed", out.stdout)
        self.assertEqual(self.launches(), [])
        self.assertFalse(any((self.home / "forensics").glob("2*")), "no dump while the fleet is someone else's")

    def test_a_fleet_that_is_booting_is_adopted_not_relaunched(self):
        import time
        started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        out = self.loop(FAKE_CONTAINERS="st-glm53", FAKE_STARTED_AT=started, FAKE_DOOR_DOWN_CALLS=4)   # health, handover, booting, one adoption poll
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertIn("adopting it, not relaunching it", out.stdout)
        self.assertIn("adoption: door up after", out.stdout)
        self.assertIn("adoption: healthy after", out.stdout)
        self.assertEqual(self.launches(), [], "the launcher was never called")

    def test_a_fleet_the_queue_just_let_go_is_not_restored_under_its_next_ticket(self):
        self.activity(ago=5)
        out = self.loop(FAKE_DOOR_DOWN_CALLS=99)                 # production is down: the queue just let the fleet go
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertIn("the queue was active 5s ago", out.stdout)
        self.assertEqual(out.stdout.count("no forensics, no launch attempt"), 1, "said once, not every 30 s")
        self.assertEqual(self.launches(), [])

    def test_a_node_whose_census_stalls_is_not_a_dead_ring_when_a_chat_answers(self):
        """containers_up needs four ssh answers; a stalled one made the old loop count a failure
        while the engine answered chats. A chat that answers is health; the census is reported."""
        self.activity(ago=1000)
        out = self.loop(FAKE_CONTAINERS="", FAKE_CHAT_FAIL_CALLS=0)      # no census at all, yet the door chats
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertIn("existing ST fleet healthy -- adopting", out.stdout)
        self.assertEqual(self.launches(), [])
        self.assertNotIn("health check failed", out.stdout)

    def test_a_failed_check_says_which_probe_failed(self):
        self.activity(ago=1000)
        self.env.update(FAKE_CONTAINERS="st-glm53", FAKE_STARTED_AT="2000-01-01T00:00:00Z")   # an old container, no boot in progress
        out = self.loop(FAKE_CHAT_FAIL_CALLS=999, ST_BOOT_POLL=1, BOOT_GRACE=2)   # a launch whose chat never answers gives up in 2 s
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertIn("launch: boot grace exceeded (2s)", out.stdout)
        self.assertIn("health check failed (1/3): containers=yes door=yes chat=no", out.stdout)
        self.assertIn("health check failed (2/3): containers=yes door=yes chat=no", out.stdout)

    def test_the_grace_follows_the_queue_s_pace_and_an_open_window(self):
        """300 s restored production into the next ticket's face 17 times in one night; the queue now
        writes what its record says (bench/fleet_pace.py) and a session can hold a window open."""
        import time
        self.activity(ago=400)                                        # past the old constant
        (self.fleet_dir / "restore-grace.json").write_text(json.dumps({"seconds": 900}))
        out = self.loop(FAKE_DOOR_DOWN_CALLS=99)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertIn("900s of quiet queue", out.stdout)
        self.assertEqual(self.launches(), [])
        (self.fleet_dir / "restore-grace.json").write_text(json.dumps({"seconds": 300}))
        (self.fleet_dir / "window.json").write_text(json.dumps({"session": "campaign", "until": time.time() + 1800}))
        out = self.loop(FAKE_DOOR_DOWN_CALLS=99)
        self.assertIn("of quiet queue", out.stdout)
        self.assertEqual(self.launches(), [], "a window keeps production down between a session's tickets")
        (self.fleet_dir / "window.json").unlink()
        # This case tests when restoration starts. Make the fake door healthy
        # when its launcher runs, instead of spending 99 zero-delay polls on it.
        self.script("curl", '#!/bin/sh\n[ -f "$FAKE_HOME/containers-up" ] || exit 22\n'
                            'echo \'{"data":[{"id":"glm-5.3-flash"}],"choices":[{}]}\'\n')
        out = self.loop(ST_RESTORE_GRACE_S=100)   # the operator's constant wins
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertEqual(self.launches(), ["launch stop", "launch "])

    def test_a_fleet_taken_while_launching_is_not_a_failed_attempt(self):
        """05:30 and 06:25 on 2026-09-13: the launcher's stop let the lease go, a waiting ticket took it,
        the launcher's start was refused -- and the loop counted attempts 1 and 5 and HELD production."""
        self.activity(ago=1000)
        launcher = self.home / "launcher.sh"
        launcher.write_text('#!/bin/sh\necho "launch $*" >> "$FAKE_HOME/events"\n'
                            'echo st-replay-other-session > "$FAKE_HOME/fleet.lock"\nexit 1\n')   # refused: a ticket took the fleet
        out = self.loop(FAKE_DOOR_DOWN_CALLS=99)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertIn("the fleet was taken while launching", out.stdout)
        self.assertNotIn("launch attempt 1/5", out.stdout)
        text = (Path(__file__).resolve().parents[1] / 'launchers/st-glm53-supervisor.sh').read_text()
        self.assertIn("a foreign window resets the launch count", text, "and a window ends the row of failures the hold counts")

    def test_a_launch_is_done_when_a_chat_answers_and_is_not_repeated(self):
        self.activity(ago=1000)                                   # the grace has passed
        out = self.loop(FAKE_CHAT_FAIL_CALLS=2)                  # the first two chats after the door fail: warmup
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertEqual(self.launches(), ["launch stop", "launch "], "one launch, no relaunch while the chat warmed up")
        self.assertIn("launch: door up after 0s", out.stdout)
        self.assertIn("launch: healthy after", out.stdout)
        self.assertNotIn("launch attempt", out.stdout)
        self.assertNotIn("health check failed", out.stdout)


if __name__ == "__main__":
    unittest.main()
