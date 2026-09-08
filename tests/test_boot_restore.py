"""Production checkout selection without Docker, network, or a fleet hold."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RestoreTests(unittest.TestCase):
    def test_config_without_newline_selects_main_and_preserves_candidate_branch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo, origin, fleet, runner = [root / name for name in ('production with spaces', 'origin', 'fleet', 'runner')]
            repo.mkdir(); fleet.mkdir(); (runner/'bench').mkdir(parents=True)
            def git(*args, cwd=repo):
                return subprocess.check_output(['git', *args], cwd=cwd, text=True, stderr=subprocess.PIPE).strip()
            git('init', '--bare', str(origin), cwd=root)
            git('init', '-b', 'main')
            git('config', 'user.name', 'Fixture'); git('config', 'user.email', 'fixture@example.invalid')
            (repo/'version').write_text('approved')
            git('add', '.'); git('commit', '-m', 'approved')
            approved = git('rev-parse', 'HEAD')
            git('remote', 'add', 'origin', str(origin)); git('push', 'origin', 'main')
            git('switch', '-c', 'candidate')
            (repo/'version').write_text('candidate')
            git('commit', '-am', 'candidate')
            candidate = git('rev-parse', 'HEAD')
            (fleet/'holder').write_text('fixture|1|host|0|1|test|boot\n')
            (fleet/'production-repo').write_text(str(repo))  # EOF without newline
            (runner/'bench/fleet_entry.py').write_text('raise SystemExit(0)\n')
            env = {k:v for k,v in os.environ.items() if not k.startswith('FLEET_')}
            env.update(FLEET_SESSION='fixture', FLEET_DIR=str(fleet), FLEET_RUNNER_REPO=str(runner))
            result = subprocess.run(['bash', str(ROOT/'bench/fleet_restore.sh')], env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertEqual(git('rev-parse', 'HEAD'), approved)
            self.assertEqual(git('rev-parse', 'candidate'), candidate)
            self.assertEqual(git('status', '--porcelain'), '')
            # A dirty tree must never be silently discarded on the next restore.
            (repo/'version').write_text('user edit')
            result = subprocess.run(['bash', str(ROOT/'bench/fleet_restore.sh')], env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 2)
            self.assertEqual((repo/'version').read_text(), 'user edit')


if __name__ == '__main__':
    unittest.main()
