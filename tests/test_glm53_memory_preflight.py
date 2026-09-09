"""Run the launcher's memory gate with isolated, inert reclaim commands."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / 'launchers/start-glm53-nvfp4-tp4.sh'


class LauncherMemoryTests(unittest.TestCase):
    def run_gate(self, *, gmu='0.73', pinned=False, budget='0.65',
                 status=0, missing=False, cleanup_status=0):
        source = LAUNCHER.read_text()
        # Execute the complete reclaim + sizing block used by the launcher.
        # No model imports, containers, SSH connections or real reclaim occur.
        block = source[source.index('\nPREFLIGHT=') + 1:source.index('# CUDA graph memory profiling')]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bindir = root / 'bin'
            bindir.mkdir()
            for name in ('bash', 'ssh'):
                command = bindir / name
                command.write_text(f'#!/bin/sh\nexit {cleanup_status}\n')
                command.chmod(0o755)
            if not missing:
                helper = root / 'memfree-preflight.sh'
                helper.write_text(f'#!/bin/sh\nprintf "%s\\n" {budget}\nexit {status}\n')
                helper.chmod(0o755)
            script = root / 'launcher.sh'
            script.write_text('set -euo pipefail\n' + '''
PREBUILD=0
HEAD_IP=10.10.10.2
WORKER_IPS=(10.10.10.1 10.10.10.3 10.10.10.4)
NAME_HEAD=glm53
NAME_WORKER=glm53-worker
CACHE_HOST_PATH=/unused
IMAGE=fixture
SSHOPT=''
''' + block + '\nprintf "FINAL_GMU=%s\\n" "$GMU"\n')
            env = {**os.environ, 'PATH': f'{bindir}:/usr/bin:/bin',
                   'GMU': gmu, '_GMU_PINNED': '1' if pinned else '',
                   'SKIP_PREFLIGHT': '0', 'DRY_RUN': '0'}
            return subprocess.run(['/bin/bash', str(script)], env=env,
                                  text=True, capture_output=True, timeout=10)

    def test_unpinned_budget_is_adopted_in_both_directions(self):
        for gmu in ('0.60', '0.73'):
            with self.subTest(gmu=gmu):
                result = self.run_gate(gmu=gmu)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('FINAL_GMU=0.65', result.stdout)

    def test_pinned_budget_at_or_below_measured_ceiling_is_preserved(self):
        for gmu in ('0.60', '0.65'):
            with self.subTest(gmu=gmu):
                result = self.run_gate(gmu=gmu, pinned=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('FINAL_GMU=' + gmu, result.stdout)

    def test_pinned_budget_above_measured_ceiling_aborts(self):
        result = self.run_gate(pinned=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('exceeds measured memory budget', result.stderr)
        self.assertNotIn('FINAL_GMU=', result.stdout)

    def test_failed_preflight_never_falls_back_to_configured_budget(self):
        for pinned in (False, True):
            with self.subTest(pinned=pinned):
                result = self.run_gate(pinned=pinned, status=1)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('memory preflight failed', result.stderr)
                self.assertNotIn('FINAL_GMU=', result.stdout)

    def test_missing_checkout_helper_aborts(self):
        result = self.run_gate(missing=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('memory preflight helper is unavailable', result.stderr)
        self.assertNotIn('FINAL_GMU=', result.stdout)

    def test_failed_glm_container_cleanup_aborts_before_sizing(self):
        result = self.run_gate(cleanup_status=23)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('container/lock cleanup failed', result.stderr)
        self.assertNotIn('FINAL_GMU=', result.stdout)


if __name__ == '__main__':
    unittest.main()
