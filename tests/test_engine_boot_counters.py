"""A boot's decode counters must outlive it, and two records must subtract."""
import ast
import json
import tempfile
import time
import types
import unittest
from pathlib import Path

BOOT = Path(__file__).resolve().parents[1] / 'engine/profiles/glm53/boot.py'


def namespace():
    tree = ast.parse(BOOT.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'write_boot_counters']
    ns = {'time': time, 'Path': Path, 'dict': dict}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(BOOT), 'exec'), ns)
    return ns


def fake_engine(**over):
    base = dict(accepted_per_step=[3, 2, 1, 0, 0, 0, 0, 5], accepted_total=17, drafted_total=56,
                steps=11, steps_verified=11, ceiling_positions=4, reachable_mass=1.5,
                covered_mass=0.75, decode_shape_counts={8: 9, 16: 2},
                lane_info={'spec_k': '7'},
                drafter=types.SimpleNamespace(k=7, fc_bias_status='missing'))
    base.update(over)
    return types.SimpleNamespace(**base)


class BootCountersTests(unittest.TestCase):
    def test_it_writes_counts_that_subtract(self):
        write = namespace()['write_boot_counters']
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'sub' / 'counters-rank2.json'
            record = write(fake_engine(), out, rank=2)
            self.assertTrue(out.is_file(), 'the parent directory is created')
            on_disk = json.loads(out.read_text())
        self.assertEqual(on_disk, json.loads(json.dumps(record, sort_keys=True, default=str)))
        self.assertEqual(on_disk['rank'], 2)
        self.assertEqual(on_disk['spec_k'], 7)
        self.assertEqual(on_disk['fc_bias_status'], 'missing')
        counters = on_disk['counters']
        self.assertEqual(counters['accepted_per_step'], [3, 2, 1, 0, 0, 0, 0, 5])
        self.assertEqual(counters['accepted_total'], 17)
        # counts, not rates: a later record minus an earlier one is the traffic between them
        self.assertEqual(counters['steps'], 11)

    def test_dictionary_keys_survive_json(self):
        write = namespace()['write_boot_counters']
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'c.json'
            write(fake_engine(), out, rank=0)
            shapes = json.loads(out.read_text())['counters']['decode_shape_counts']
        self.assertEqual(shapes, {'8': 9, '16': 2}, 'integer widths become strings, not a crash')

    def test_a_missing_counter_is_omitted_not_invented(self):
        write = namespace()['write_boot_counters']
        engine = fake_engine()
        del engine.covered_mass
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'c.json'
            write(engine, out, rank=1)
            counters = json.loads(out.read_text())['counters']
        self.assertNotIn('covered_mass', counters)
        self.assertIn('reachable_mass', counters)

    def test_a_drafterless_boot_still_writes(self):
        write = namespace()['write_boot_counters']
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'c.json'
            record = write(fake_engine(drafter=None), out, rank=0)
        self.assertIsNone(record['spec_k'])
        self.assertIsNone(record['fc_bias_status'])


class ShutdownWiringTests(unittest.TestCase):
    def test_the_shutdown_writes_it_and_never_fails_on_it(self):
        source = BOOT.read_text()
        self.assertIn('write_boot_counters(engine, Path(a.dump_dir) / f"counters-rank{comm.rank}.json"', source)
        after = source.split('write_boot_counters(engine,', 1)[1][:400]
        self.assertIn('except Exception', after, 'a shutdown never fails on its own record')

    def test_the_collector_arms_when_the_bias_is_missing(self):
        source = BOOT.read_text()
        self.assertIn('DRAFT_FC_CAPTURE = False', source)          # never forced on in main
        self.assertIn('DRAFT_FC_CAPTURE_WHEN_MISSING = True', source)
        self.assertIn('"fc_bias", None) is None', source)


if __name__ == '__main__':
    unittest.main()
