"""Exercise fleet ownership using real shell control flow and an isolated fake fleet."""
import os
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class FleetOps(unittest.TestCase):
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
        self.env = dict(os.environ, PATH=f"{self.bin}:{os.environ['PATH']}",
                        FAKE_HOME=str(self.home), ST_FORENSICS=str(self.home / "forensics"),
                        ST_REPO=str(self.repo), ST_SUPERVISOR_ONCE="1",
                        CKPT=str(self.home / "missing-checkpoint"))
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

    def test_supervisor_preserves_unreachable_fleet(self):
        self.env["FAKE_SSH_FAIL"] = "1"
        result = self.run_script("st-glm53-supervisor.sh")
        self.assertIn("head unreachable", result.stdout)
        self.assertFalse(self.events.exists())


if __name__ == "__main__":
    unittest.main()
