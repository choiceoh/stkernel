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



class LaunchHarness(FleetHarness):
    """The fake fleet with everything the real launcher needs to reach `docker run` on all four nodes."""
    NODES = ("10.10.10.2", "10.10.10.1", "10.10.10.3", "10.10.10.4")

    def setUp(self):
        super().setUp()
        models = self.home / "models"
        ckpt, ranks, drafter = models / "ckpt", models / "ranks", models / "drafter"
        for d in (ckpt, ranks, drafter):
            d.mkdir(parents=True)
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json",
                     "processor_config.json"):
            (ckpt / name).write_text("{}")
        for name in [f"rank{r}of4.safetensors" for r in range(4)] + ["vision.safetensors"]:
            (ranks / name).write_bytes(b"x")
        (drafter / "model.safetensors").write_bytes(b"x")
        engine_dir = self.home / "st-engine"
        (engine_dir / "engine/runtime").mkdir(parents=True)
        (engine_dir / "engine/runtime/build.sh").write_text("exit 0\n")
        # the node's copy of the broker (the tree push is a no-op here): it says it started, for which rank dir
        (engine_dir / "launchers").mkdir()
        (engine_dir / "launchers/st-reclaim-broker.sh").write_text(
            'echo "${FAKE_NODE:-?} broker:$1:$(basename "${2:-}")" >> "$FAKE_HOME/events"\n'
            '[ "$1" = start ] && [ -n "${FAKE_BROKER_FAIL:-}" ] && { echo "reclaim broker did not start in $2"; exit 1; }\n'
            'echo "reclaim broker serving $2 (pid 4242)"\n')
        meminfo = self.home / "meminfo"
        meminfo.write_text("MemTotal: 125000000 kB\nMemFree: 35000000 kB\nMemAvailable: 110000000 kB\n"
                           "Cached: 75000000 kB\nCommitLimit: 79477760 kB\nCommitted_AS: 4718592 kB\n")
        overcommit = self.home / "overcommit_memory"
        overcommit.write_text("2\n")
        self.env.update(CKPT=str(ckpt), RANKS_DIR=str(ranks), DRAFTER=str(drafter), ST_ENGINE_DIR=str(engine_dir),
                        ST_MEMINFO=str(meminfo), ST_OVERCOMMIT=str(overcommit))
        # ssh tells the stubs which node a command ran on; the tree push and the flush do nothing here
        self.script("ssh", '''#!/usr/bin/env python3
import os, pathlib, subprocess, sys
h = pathlib.Path(os.environ['FAKE_HOME'])
node = next((a.split('@', 1)[1] for a in sys.argv[1:] if a.startswith('choiceoh@')), 'local')
cmd = sys.argv[-1].replace('/home/choiceoh/glm53-logs/st-fleet.lock', str(h / 'fleet.lock'))
cmd = cmd.replace('/home/choiceoh/st-fleet.lock', str(h / 'fleet.lock'))
sys.exit(subprocess.call(['/bin/bash', '-c', cmd], env=dict(os.environ, FAKE_NODE=node)))
''')
        self.script("docker", '''#!/usr/bin/env python3
import os, pathlib, sys
a = sys.argv[1:]
h = pathlib.Path(os.environ['FAKE_HOME'])
def note(what, path='events'):
    with (h / path).open('a') as f:
        f.write(os.environ.get('FAKE_NODE', '?') + ' ' + what + '\\n')
if a[:2] == ['rm', '-f']:
    note('rm')
elif a[:1] == ['run']:
    note('run')
    note(' '.join(a), 'runs')
elif a and a[0] == 'ps':
    print(os.environ.get('FAKE_CONTAINERS', ''))
''')
        self.script("sudo", '#!/bin/sh\necho "${FAKE_NODE:-?} reclaim" >> "$FAKE_HOME/events"\nexit "${FAKE_SUDO_FAIL:-0}"\n')
        self.script("rsync", "#!/bin/sh\nexit 0\n")
        self.script("sync", "#!/bin/sh\nexit 0\n")
        self.script("timeout", '#!/bin/sh\nshift\nexec "$@"\n')

    def launch(self):
        result = self.run_script("start-st-glm53.sh")
        steps = {}
        for line in (self.events.read_text().splitlines() if self.events.exists() else []):
            node, what = line.split(" ", 1)
            steps.setdefault(node, []).append(what)
        return result, steps

    def boot_commands(self):
        runs = self.home / "runs"
        return runs.read_text().splitlines() if runs.exists() else []


RANK_OF = {"10.10.10.2": 0, "10.10.10.1": 1, "10.10.10.3": 2, "10.10.10.4": 3}


class FileCacheReturnTests(LaunchHarness):
    """Each node returns its clean file cache from the host the moment before its container starts (2026-09-13).

    No production boot came up from 19:29 to 19:48 on 2026-09-13. srv2's strict overcommit refused the
    engine's 76.47 GiB reclaim mapping outright, and on srv4 the same fault would have crossed the box's
    SIGTERM line. Inside its container the engine sees only its own checkpoint. So the launcher drops each
    node's cache from the host, after the rsync and the image build and right before `docker run`, with
    the lease held and the fleet idle. The fake fleet runs the real launcher all the way to `docker run`.
    """

    def test_every_node_returns_its_cache_after_the_old_container_goes_and_before_the_new_one_starts(self):
        result, steps = self.launch()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(steps, {ip: ["rm", "reclaim", f"broker:start:rank{RANK_OF[ip]}", "run"] for ip in self.NODES})
        for ip in self.NODES:
            self.assertIn(f"{ip}: file cache returned: MemFree 33.4 GiB, MemAvailable 104.9 GiB, Cached 71.5 GiB -> ",
                          result.stdout)
            self.assertIn(f"{ip}: started", result.stdout)
        self.assertEqual(result.stdout.count("; commit: overcommit_memory 2, CommitLimit 75.8 GiB, Committed_AS 4.5 GiB"), 4,
                         "every node says how much commit room a strict node has")
        self.assertTrue(self.lock.exists(), "the boot went on to hold its lease")

    def test_a_node_that_cannot_return_it_says_so_and_the_boot_goes_on(self):
        self.env["FAKE_SUDO_FAIL"] = "1"
        result, steps = self.launch()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(steps, {ip: ["rm", "reclaim", f"broker:start:rank{RANK_OF[ip]}", "run"] for ip in self.NODES})
        self.assertEqual(result.stdout.count("file cache NOT returned (sudo -n refused): MemFree 33.4 GiB"), 4)
        self.assertEqual(result.stdout.count("starting anyway, the engine's admission decides"), 4)

    def test_zero_turns_it_off(self):
        self.env["ST_RECLAIM_FILE_CACHE"] = "0"
        result, steps = self.launch()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(steps, {ip: ["rm", "run"] for ip in self.NODES})
        self.assertNotIn("file cache", result.stdout)
        self.assertFalse([c for c in self.boot_commands() if "ST_RECLAIM_DIR" in c], "no broker, so no directory to ask")

    def test_a_foreign_container_or_a_typo_returns_nothing_and_starts_nothing(self):
        for setting, busy in (("1", "st-other"), ("typo", "")):
            with self.subTest(setting=setting, busy=busy):
                self.env.update(ST_RECLAIM_FILE_CACHE=setting, FAKE_CONTAINERS=busy)
                result, steps = self.launch()
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(steps, {})
                self.assertFalse(self.lock.exists())

    def test_the_return_comes_after_the_tree_and_the_image_so_nothing_refills_it_first(self):
        text = (ROOT / "launchers/start-st-glm53.sh").read_text()
        rank = text[text.index("start_rank() {"):text.index("pids=()")]
        self.assertLess(rank.index("push_tree"), rank.index("st-return-file-cache.sh"))
        self.assertLess(rank.index("engine/runtime/build.sh"), rank.index("st-return-file-cache.sh"))
        self.assertLess(rank.index('docker rm -f $NAME >/dev/null 2>&1 || true"'), rank.index("st-return-file-cache.sh"))
        self.assertLess(rank.index("st-return-file-cache.sh"), rank.index("docker run -d --name $NAME"))
        self.assertNotIn("memfree-preflight.sh", text, "the vLLM sizing preflight no longer gates an ST boot")


class ReclaimBrokerLaunchTests(LaunchHarness):
    """Each rank's node runs a broker the engine asks when it is short of immediately free memory (2026-09-13).

    srv2 runs strict overcommit at ratio 50 (CommitLimit 75.8 GiB), and the engine's own reclaim is an anonymous
    mapping that never fits there. The host's drop needs no commit room, so the launcher starts a broker per
    rank and hands the container its directory."""

    def test_every_rank_gets_its_own_broker_and_its_container_is_told_where(self):
        result, steps = self.launch()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for ip in self.NODES:
            self.assertEqual(steps[ip][2], f"broker:start:rank{RANK_OF[ip]}", "after the return, before the container")
            self.assertIn(f"reclaim broker serving /home/choiceoh/glm53-logs/st-reclaim/rank{RANK_OF[ip]} (pid 4242)", result.stdout)
        commands = dict(line.split(" ", 1) for line in self.boot_commands())
        for ip, command in commands.items():
            self.assertIn(f"-e ST_RECLAIM_DIR=/home/choiceoh/glm53-logs/st-reclaim/rank{RANK_OF[ip]} ", command)
            self.assertIn(f"-e RANK={RANK_OF[ip]} ", command, "the directory and the rank agree")

    def test_a_broker_that_does_not_start_leaves_the_container_its_own_reclaim(self):
        self.env["FAKE_BROKER_FAIL"] = "1"
        result, steps = self.launch()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual({ip: s[-1] for ip, s in steps.items()}, {ip: "run" for ip in self.NODES})
        self.assertEqual(result.stdout.count("the reclaim broker did not start -- admission falls back to its own reclaim"), 4)
        self.assertFalse([c for c in self.boot_commands() if "ST_RECLAIM_DIR" in c])

    def test_stop_ends_every_broker_after_the_containers(self):
        self.lock.write_text("choiceoh@srv2 st-glm53 2026-09-12")
        result = self.run_script("start-st-glm53.sh", "stop")
        self.assertEqual(result.returncode, 0, result.stderr)
        steps = {}
        for line in self.events.read_text().splitlines():
            node, what = line.split(" ", 1)
            steps.setdefault(node, []).append(what)
        self.assertEqual(steps, {ip: ["rm", "broker:stop-all:st-reclaim"] for ip in self.NODES})


class WorkspaceCeilingLaunchTests(LaunchHarness):
    """ST_WORKSPACE_GIB: a shape that spends more than the profile's ceiling raises it for every rank (2026-09-13)."""

    def test_the_profile_ceiling_is_the_default_and_the_variable_reaches_every_rank(self):
        result, _ = self.launch()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(self.boot_commands()), 4)
        self.assertFalse([c for c in self.boot_commands() if "--workspace-gib" in c], "no flag: budget.WORKSPACE_GIB")
        (self.home / "runs").unlink()
        self.events.unlink()
        self.lock.unlink()
        self.env["ST_WORKSPACE_GIB"] = "10.5"
        result, _ = self.launch()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        commands = self.boot_commands()
        self.assertEqual(len(commands), 4)
        # no ST_KV_GIB is not "no --kv-gib" since PR #1042: the launcher pins the budget production
        # serves on, because a value that lives only in the box's env file does not survive a deploy.
        self.assertTrue(all("--kv-gib 14.0 --workspace-gib 10.5 --port" in c for c in commands), commands)

    def test_a_ceiling_that_is_not_a_positive_number_starts_nothing(self):
        for bad in ("0", "0.0", "-1", "ten", "1e3", "10.5GiB"):
            with self.subTest(bad=bad):
                self.env["ST_WORKSPACE_GIB"] = bad
                result, steps = self.launch()
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn("ST_WORKSPACE_GIB must be a positive GiB ceiling", result.stderr)
                self.assertEqual((steps, self.boot_commands()), ({}, []))
                self.assertFalse(self.lock.exists())


class ReturnScriptTests(unittest.TestCase):
    """launchers/st-return-file-cache.sh on its own: one line, exit 0 when returned, 3 when it could not be."""

    def run_return(self, *, sudo_exit=0, meminfo="MemFree: 1048576 kB\nMemAvailable: 3145728 kB\nCached: 2097152 kB\n"
                   "CommitLimit: 79477760 kB\nCommitted_AS: 4718592 kB\n"):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            (home / "bin").mkdir()
            for name, body in (("sudo", f'#!/bin/sh\necho "$*" > "{home}/sudo"\nexit {sudo_exit}\n'),
                               ("sync", "#!/bin/sh\nexit 0\n"), ("timeout", '#!/bin/sh\nshift\nexec "$@"\n')):
                (home / "bin" / name).write_text(body)
                (home / "bin" / name).chmod(0o755)
            info = home / "meminfo"
            if meminfo is not None:
                info.write_text(meminfo)
            (home / "overcommit_memory").write_text("2\n")
            env = dict(os.environ, PATH=f"{home / 'bin'}:{os.environ['PATH']}", ST_MEMINFO=str(info),
                       ST_OVERCOMMIT=str(home / "overcommit_memory"))
            result = subprocess.run(["bash", str(ROOT / "launchers/st-return-file-cache.sh")], env=env, text=True,
                                    capture_output=True, timeout=15)
            asked = (home / "sudo").read_text().strip() if (home / "sudo").exists() else None
        return result, asked

    def test_it_drops_the_clean_cache_and_says_what_was_there(self):
        result, asked = self.run_return()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(asked, "-n sh -c echo 3 > /proc/sys/vm/drop_caches")
        self.assertEqual(result.stdout.strip(), "file cache returned: MemFree 1.0 GiB, MemAvailable 3.0 GiB, "
                                                "Cached 2.0 GiB -> MemFree 1.0 GiB, MemAvailable 3.0 GiB, Cached 2.0 GiB; "
                                                "commit: overcommit_memory 2, CommitLimit 75.8 GiB, Committed_AS 4.5 GiB")

    def test_without_passwordless_sudo_it_says_so_and_exits_three(self):
        result, _ = self.run_return(sudo_exit=1)
        self.assertEqual(result.returncode, 3)
        self.assertEqual(result.stdout.strip(), "file cache NOT returned (sudo -n refused): MemFree 1.0 GiB, "
                                                "MemAvailable 3.0 GiB, Cached 2.0 GiB; commit: overcommit_memory 2, "
                                                "CommitLimit 75.8 GiB, Committed_AS 4.5 GiB")

    def test_a_meminfo_without_commit_counters_says_nothing_of_commit(self):
        result, _ = self.run_return(meminfo="MemFree: 1048576 kB\nMemAvailable: 3145728 kB\nCached: 2097152 kB\n")
        self.assertEqual(result.returncode, 0)
        self.assertNotIn("commit", result.stdout)

    def test_an_unreadable_meminfo_does_not_stop_the_return(self):
        result, asked = self.run_return(meminfo=None)
        self.assertEqual(result.returncode, 0)
        self.assertIsNotNone(asked)
        self.assertTrue(result.stdout.startswith("file cache returned: meminfo unreadable -> "), result.stdout)


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

    def warm_file(self, lines=('{"prompt": "a shared system prompt"}',)):
        path = self.home / "st-warm.jsonl"
        path.write_text("".join(l + "\n" for l in lines))
        probes = self.repo / "probes"
        probes.mkdir(parents=True, exist_ok=True)
        (probes / "st_prefix_warm.py").write_text(
            'import json, os, pathlib, sys\n'
            'h = pathlib.Path(os.environ["FAKE_HOME"])\n'
            'if os.environ.get("FAKE_WARM_FAIL"): sys.exit(3)\n'
            'n = sum(1 for l in open(sys.argv[1]) if l.strip())\n'
            '(h / "warm-argv").write_text(json.dumps(sys.argv[1:]))\n'
            'print(f"  warmed {n} prompts")\n')
        self.env["ST_WARM_FILE"] = str(path)
        return path

    def test_a_healthy_boot_warms_the_prefix_cache_it_started_empty(self):
        """`/v1/prefix/warm` and probes/st_prefix_warm.py existed all along with no caller.

        A boot begins with nothing cached, so every relaunch -- three on 2026-09-16 -- made the first
        conversation prefill the prompt every conversation shares. And the shared boundary is the one
        the tier could not keep either until PR #1045: it stops being a leaf as soon as anyone writes
        past it.
        """
        self.warm_file()
        out = self.loop(FAKE_DOOR_DOWN_CALLS=0)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertIn("prefix cache warmed -- warmed 1 prompts", out.stdout)
        argv = json.loads((self.home / "warm-argv").read_text())
        self.assertIn("--pin", argv, "pinned, so _victim keeps them behind everything else")
        self.assertIn("--url", argv)
        self.assertEqual(argv[0], str(self.home / "st-warm.jsonl"))

    def test_a_warm_that_fails_is_not_a_failed_boot(self):
        """The fleet is already healthy when this runs. A cache that did not warm is slower, not broken."""
        self.warm_file()
        out = self.loop(FAKE_DOOR_DOWN_CALLS=0, FAKE_WARM_FAIL="1")
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertIn("prefix warm did not finish (rc=3)", out.stdout)
        self.assertIn("healthy after", out.stdout)
        self.assertNotIn("health check failed", out.stdout)

    def test_no_warm_file_is_no_warm_and_no_complaint(self):
        """And an EMPTY ST_WARM_FILE is off, not the default: `${x-d}`, never `${x:-d}`.

        The tier learned the same lesson the same day (PR #1042) -- an operator who writes
        `ST_WARM_FILE=` means off, and a `:-` reads that as "unset" and hands back the default.
        """
        self.warm_file()                                     # the default path exists and would warm
        for value in (str(self.home / "absent.jsonl"), ""):
            with self.subTest(ST_WARM_FILE=value):
                (self.home / "warm-argv").unlink(missing_ok=True)
                out = self.loop(FAKE_DOOR_DOWN_CALLS=0, ST_WARM_FILE=value)
                self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
                self.assertNotIn("prefix", out.stdout)
                self.assertFalse((self.home / "warm-argv").exists(), "the probe must not have run")

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
        # The real clock can advance while the subprocess starts on a busy CPU.
        self.assertRegex(out.stdout, r"the queue was active [0-9]+s ago")
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
