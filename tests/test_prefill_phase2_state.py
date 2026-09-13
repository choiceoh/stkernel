import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / 'measurements/st_prefill_phase2_20260913/diagnose_state.py'
SPEC = importlib.util.spec_from_file_location('phase2_state', PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class StateTests(unittest.TestCase):
    def row(self, rank, step, count):
        return dict(kind='host_step', rank=rank, request_token='prepare', step=step,
                    phase='decode', rows=[0], tokens=7, dispatch='_launch',
                    host_state={'before': [{'generated': count}]})

    def test_first_shared_divergence_is_retained(self):
        rows = [self.row(r, s, s + (r == 3 and s == 4))
                for s in range(1, 6) for r in range(4)]
        result = MODULE.compare(reversed(rows))
        self.assertEqual(result['status'], 'divergent_host_state')
        self.assertEqual(result['step'], 4)
        self.assertEqual(result['ranks'][3]['host_state']['before'][0]['generated'], 5)

    def test_missing_rank_and_old_format_cannot_establish_agreement(self):
        result = MODULE.compare([self.row(r, 1, 10) for r in range(3)])
        self.assertEqual(result['status'], 'insufficient_state_records')
        self.assertEqual(result['partial_dispatches'], 1)
        rows = [self.row(r, 1, 10) for r in range(4)]
        for row in rows:
            del row['host_state']
        result = MODULE.compare(rows)
        self.assertEqual(result['status'], 'insufficient_state_records')
        self.assertEqual(result['dispatches_without_state'], 4)


if __name__ == '__main__':
    unittest.main()
