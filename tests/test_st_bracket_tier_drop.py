"""An arm's prefix tier leaves with its boot.

The shell stop_arm/drop_tier run unchanged against a fake release launcher and a
fake docker on PATH (node_sh takes its local branch). No GPU, container,
network or fleet lease is used.
"""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SHA = '0123456789abcdef0123456789abcdef01234567'
NODES = '127.0.0.91 127.0.0.92'
DOCKER = '''#!/usr/bin/env bash
echo "$*" >> "$DOCKER_EVENTS"
case "$1" in
  ps) [ -z "${IN_USE:-}" ] || echo c1;;
  inspect) echo "[-lc boot.py --tier-dir $IN_USE --dump-dir /x]";;
  run)
    [ "${IMAGE_MISSING:-0}" != 1 ] || exit 125
    mount=""; while [ $# -gt 0 ]; do case "$1" in -v) mount=$2; shift 2;; *) last=$1; shift;; esac; done
    host=${mount%%:/tier}; exec rm -rf -- "$host/${last#/tier/}";;
esac
'''


class TierDropTests(unittest.TestCase):
    def execute(self, *, stop_rc=0, keep=False, in_use=False, image_missing=False, arm='ST-abc', driver='stop_arm'):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs = root / 'logs'
            tier = logs / 'st-bracket-tier' / ('test-' + arm)
            (tier / 'rank0').mkdir(parents=True)
            (tier / 'rank0' / 'seq-1.kv').write_bytes(b'kv')
            keep_other = logs / 'st-bracket-tier' / 'other-session'
            keep_other.mkdir()
            release = root / 'release'
            (release / 'launchers').mkdir(parents=True)
            (release / 'launchers/start-st-glm53.sh').write_text('exit %d\n' % stop_rc)
            bin_dir = root / 'bin'
            bin_dir.mkdir()
            (bin_dir / 'docker').write_text(DOCKER)
            (bin_dir / 'docker').chmod(0o755)
            source = (ROOT / 'bench/st_bracket.sh').read_text().rsplit('\ncase "${1:-}" in', 1)[0]
            source += '\nSELF_IPS=" %s "\nRELEASE=%s ARM=%s ARM_SHA=%s\n%s\n' % (NODES, release, arm, SHA, driver)
            script = root / 'runner.sh'
            script.write_text(source)
            events = root / 'docker-events'
            env = dict(os.environ, REPO=str(root), LOGD=str(logs), ST_NODES=NODES,
                       PATH=str(bin_dir) + os.pathsep + os.environ['PATH'],
                       ST_PRODUCTION_ENV=str(root / 'absent.env'), DOCKER_EVENTS=str(events),
                       FLEET_SESSION='test', FLEET_REHEARSE='0',
                       IN_USE=str(tier) if in_use else '', IMAGE_MISSING=str(int(image_missing)))
            if keep:
                env['ST_BRACKET_KEEP_TIER'] = '1'
            result = subprocess.run(['bash', str(script)], text=True, capture_output=True,
                                    timeout=20, env=env)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue(keep_other.is_dir(), "another session's tier is never touched")
            return result.stdout, tier.exists(), events.read_text() if events.exists() else ''

    def test_successful_stop_drops_the_arm_tier_on_every_node_with_the_arm_image(self):
        out, exists, docker = self.execute()
        self.assertFalse(exists)
        self.assertIn('st-engine:bracket-%s' % SHA[:12], docker)
        self.assertIn('--network none', docker)
        for ip in NODES.split():
            self.assertIn('tier test-ST-abc on %s: ' % ip, out)
        self.assertIn(': dropped', out)

    def test_failed_stop_keeps_the_tier(self):
        out, exists, docker = self.execute(stop_rc=1)
        self.assertTrue(exists)
        self.assertEqual(docker, '')
        self.assertIn('the tier stays until a stop succeeds', out)

    def test_keep_switch_keeps_the_tier(self):
        out, exists, docker = self.execute(keep=True)
        self.assertTrue(exists)
        self.assertEqual(docker, '')
        self.assertIn('ST_BRACKET_KEEP_TIER=1', out)

    def test_a_running_container_naming_the_tier_keeps_it(self):
        out, exists, docker = self.execute(in_use=True)
        self.assertTrue(exists)
        self.assertNotIn('run ', docker)
        self.assertIn(': in use', out)

    def test_missing_image_falls_back_to_a_plain_remove(self):
        out, exists, _ = self.execute(image_missing=True)
        self.assertFalse(exists, 'files the runner itself can remove still leave')

    def test_an_arm_name_that_escapes_the_tier_root_is_refused(self):
        for dir in ('"$LOGD/st-bracket-tier/test-ST-abc/../../escape"', '"$LOGD/st-bracket-tier"',
                    '"$LOGD/elsewhere/test-ST-abc"', '"$LOGD/st-bracket-tier/test-ST-abc/rank0"'):
            with self.subTest(dir=dir):
                out, exists, docker = self.execute(driver='drop_tier %s img' % dir)
                self.assertTrue(exists)
                self.assertNotIn('run ', docker)
                self.assertIn('tier not dropped', out)


if __name__ == '__main__':
    unittest.main()
