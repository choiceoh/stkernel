"""The queue's pace: the grace production waits after the last ticket follows the record (45차, 2026-09-13)."""
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
import fleet_pace as pace                                                # noqa: E402


def line(t, text):
    return time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(t)) + ' ' + text


class PaceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.now = time.time()

    def log(self, *gaps_min, hours_ago=1.0):
        """A ticket ends, the next boot request comes `gap` minutes later, for each gap."""
        t = self.now - hours_ago * 3600
        rows = []
        for g in gaps_min:
            rows.append(line(t, 'release s'))
            t += g * 60
            rows.append(line(t, f'request s{int(t)} est=10m note [boot]'))
            t += 60
        (self.dir / 'log').write_text('\n'.join(rows) + '\n')

    def test_the_grace_is_the_p75_of_recent_gaps_within_the_floor_and_ceiling(self):
        self.log(2, 4, 6, 8, 12)
        v = pace.write_grace(self.dir, self.now)
        self.assertEqual(v['basis']['gaps'], 5)
        self.assertEqual(v['seconds'], 8 * 60)                            # p75 of 2,4,6,8,12
        self.assertEqual(json.loads((self.dir / 'restore-grace.json').read_text())['seconds'], 480)
        self.log(1, 1, 1)
        self.assertEqual(pace.write_grace(self.dir, self.now)['seconds'], pace.FLOOR, 'never below the old constant')
        self.log(60, 90, 120)
        self.assertEqual(pace.write_grace(self.dir, self.now)['seconds'], pace.CEILING, 'never beyond twenty minutes')

    def test_no_record_means_the_floor(self):
        self.assertEqual(pace.write_grace(self.dir, self.now)['seconds'], pace.FLOOR)
        (self.dir / 'log').write_text('garbage\n')
        self.assertEqual(pace.write_grace(self.dir, self.now)['seconds'], pace.FLOOR)

    def test_old_gaps_and_probe_requests_are_not_counted(self):
        self.log(20, 20, hours_ago=9)                                       # older than the window
        self.assertEqual(pace.write_grace(self.dir, self.now)['basis']['gaps'], 0)
        t = self.now - 600
        (self.dir / 'log').write_text(line(t, 'release s') + '\n' + line(t + 60, 'request p est=5m note [probe]') + '\n')
        self.assertEqual(pace.write_grace(self.dir, self.now)['basis']['gaps'], 0, 'a probe takes no fleet')

    def test_a_window_holds_the_grace_up_while_it_is_open(self):
        self.log(2, 2, 2)
        pace.write_grace(self.dir, self.now)
        self.assertEqual(pace.effective(self.dir, self.now)['grace_s'], pace.FLOOR)
        pace.window(self.dir, 'campaign', 30, self.now)
        e = pace.effective(self.dir, self.now + 60)
        self.assertEqual((e['grace_s'], e['window_session']), (29 * 60, 'campaign'))
        e = pace.effective(self.dir, self.now + 31 * 60)
        self.assertEqual((e['grace_s'], e['window_s']), (pace.FLOOR, 0), 'expired: the adaptive grace again')
        pace.window(self.dir, 'campaign', 'off', self.now)
        self.assertFalse((self.dir / 'window.json').exists())
        with self.assertRaises(ValueError):
            pace.window(self.dir, 'campaign', 500, self.now)

    def test_the_cli_speaks_json(self):
        import subprocess
        out = subprocess.run([sys.executable, str(Path(pace.__file__)), 'window', str(self.dir), 's', '10'], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(json.loads(out.stdout)['session'], 's')
        out = subprocess.run([sys.executable, str(Path(pace.__file__)), 'show', str(self.dir)], capture_output=True, text=True)
        self.assertGreaterEqual(json.loads(out.stdout)['grace_s'], 590)


if __name__ == '__main__':
    unittest.main()
