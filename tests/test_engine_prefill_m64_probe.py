"""probes/engine_moe_prefill_m64: the parts of the M64 gate that need no device.

The probe itself needs the ST image (moe_dispatch imports cutlass and flashinfer at module
scope) and an sm_121a card. What is checkable here is that the ladder it will spend a GPU
slot on actually covers the eligibility window, that its noise arithmetic is what it claims,
and that it names sources that exist -- the mistakes that waste a queue slot.
"""
import ast
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / 'probes/engine_moe_prefill_m64.py'


def probe():
    spec = importlib.util.spec_from_file_location('st_m64_probe', PROBE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)          # top-level imports are stdlib only
    return module


class LadderTests(unittest.TestCase):
    """`64 < m <= 8192` with tile_m 64 is the window the dispatcher admits."""

    def test_the_ladder_stays_inside_the_window_and_touches_both_edges(self):
        rows = probe().ROWS
        self.assertTrue(all(64 < m <= 8192 for m in rows), rows)
        self.assertEqual(min(rows), 65, 'the open edge is the first admitted row count')
        self.assertEqual(max(rows), 8192, 'the closed edge is the last')

    def test_the_ladder_exercises_tails_and_the_served_chunks(self):
        rows = set(probe().ROWS)
        self.assertTrue({2304, 4608, 6912} <= rows, 'the served prefill chunk sizes')
        self.assertTrue([m for m in rows if m % 64],
                        'a row count that is not a multiple of the tile, for the tail strips')
        self.assertTrue([m for m in rows if m % 128],
                        'a row count M128 also has to tail, so the arms tail together')

    def test_the_reference_bucket_is_small_enough_to_be_a_python_loop(self):
        module = probe()
        self.assertIn(module.ORACLE_ROWS, module.ROWS)
        self.assertLessEqual(module.ORACLE_ROWS, 1024)


class NoiseArithmeticTests(unittest.TestCase):
    """The gate is `across arms <= arm spread x factor`, so both halves must be right."""

    def setUp(self):
        self.m = probe()
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest('requires PyTorch')

    def test_relative_is_a_max_absolute_difference_over_the_expected_scale(self):
        import torch
        a = torch.tensor([1.0, 2.0, 4.0])
        self.assertAlmostEqual(self.m._relative(a, a), 0.0)
        self.assertAlmostEqual(self.m._relative(a + 0.4, a), 0.1, places=6)
        zero = torch.zeros(3)
        self.assertLess(self.m._relative(zero, zero), 1e-6, 'an all-zero expected must not divide by 0')

    def test_spread_is_the_widest_pair_not_the_first_one(self):
        import torch
        runs = [torch.tensor([10.0, 0.0]), torch.tensor([10.1, 0.0]), torch.tensor([9.1, 0.0])]
        # the widest pair is runs[1] vs runs[2]: |10.1 - 9.1| / 9.1
        self.assertAlmostEqual(self.m._spread(runs), 1.0 / 9.1, places=6)
        self.assertGreater(self.m._spread(runs), self.m._relative(runs[0], runs[1]))

    def test_spread_does_not_depend_on_the_order_the_repeats_arrived_in(self):
        import torch
        runs = [torch.tensor([10.0]), torch.tensor([9.0])]
        self.assertAlmostEqual(self.m._spread(runs), self.m._spread(runs[::-1]), places=9)
        self.assertAlmostEqual(self.m._spread(runs), 1.0 / 9.0, places=6)

    def test_a_single_repeat_has_no_floor_of_its_own(self):
        import torch
        self.assertEqual(self.m._spread([torch.ones(2)]), 0.0)
        self.assertEqual(self.m._spread([]), 0.0)


class GateRuleTests(unittest.TestCase):
    """The rule must fail a candidate that cannot reproduce itself.

    The first srv4 run passed a candidate whose own repeats differed by 107%: the rule was
    `across <= floor x factor`, and the floor was the candidate's own spread, so a broken
    arm raised the bar it was measured against. A self-agreement rule comes first.
    """

    def test_a_candidate_that_disagrees_with_itself_cannot_be_rescued_by_the_cross_arm_rule(self):
        m = probe()
        control, candidate, across = 0.0, 1.0674, 1.8522      # the measured m=1024 row
        self.assertFalse(candidate <= max(control * m.TOLERANCE_FACTOR, m.REPRODUCIBLE_CEILING),
                         'a 107% self-spread must fail the reproducibility rule')
        self.assertTrue(across <= max(max(control, candidate) * m.TOLERANCE_FACTOR, 1e-6),
                        'and the cross-arm rule alone would have passed it')

    def test_an_ordinary_reorder_still_passes(self):
        m = probe()
        control, candidate = 1.5e-4, 3.0e-4
        self.assertTrue(candidate <= max(control * m.TOLERANCE_FACTOR, m.REPRODUCIBLE_CEILING))

    def test_the_probe_checks_reproducibility_before_the_cross_arm_difference(self):
        text = PROBE.read_text(encoding='utf-8')
        self.assertIn('REPRODUCIBLE_CEILING', text)
        self.assertLess(text.index('if not reproducible:'), text.index('elif not within:'),
                        'self-agreement is the first gate')


class ContractTests(unittest.TestCase):
    def test_every_named_source_exists(self):
        for name in probe().SOURCES:
            self.assertTrue((ROOT / name).is_file(), name)

    def test_the_probe_reserves_nothing_and_flips_no_default(self):
        tree = ast.parse(PROBE.read_text(encoding='utf-8'))
        tree.body = [n for n in tree.body if not (isinstance(n, ast.Expr)
                                                  and isinstance(n.value, ast.Constant))]
        code = ast.unparse(tree)          # the docstring's usage line is not an action
        for forbidden in ('configure_static_v2', 'subprocess', 'fleet.sh', 'STK_'):
            self.assertNotIn(forbidden, code, f'a gate must not touch {forbidden}')
        names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        self.assertTrue({'cpu_check', 'gpu_check', 'eligibility', 'main'} <= names)

    def test_the_gpu_arm_asks_for_the_private_lane_through_the_served_entry(self):
        text = PROBE.read_text(encoding='utf-8')
        # It must OVERWRITE the keyword, not supply it as a default: b12x_fused_moe
        # forwards `_prefill_tile64=None` explicitly, and a call-time keyword beats
        # functools.partial -- a partial here lost the flag and both arms ran M128.
        self.assertIn("kw['_prefill_tile64'] = True", text, 'the candidate asks by name')
        self.assertNotIn('partial(real_moe', text, 'a partial is overridden by the callee')
        self.assertIn('launch_sm120_moe', text,
                      'the arms must differ at the served entry, not below it')
        self.assertNotIn('_prefill_m64_workspace', text,
                         'the dispatcher owns the workspace derivation; a probe that '
                         'rebuilds it measures a copy of the route, not the route')
        self.assertIn("get_device_capability() != (12, 1)", text,
                      'an sm_120 card must not be allowed to render an sm_121a verdict')

    def test_the_dispatcher_owns_the_m64_workspace_and_keeps_it_opt_in(self):
        dispatch = (ROOT / 'engine/kernels/b12x/moe_dispatch.py').read_text(encoding='utf-8')
        entry = dispatch.split('def launch_sm120_moe(')[1].split('\ndef ')[0]
        self.assertIn('_prefill_tile64: bool | None = None', entry,
                      'the served entry carries the switch, defaulting to off')
        self.assertIn('_prefill_m64_workspace(workspace, num_tokens)', entry,
                      'and derives the M64 workspace itself')
        lane = (ROOT / 'engine/kernels/b12x/b12x_moe.py').read_text(encoding='utf-8')
        self.assertIn('_prefill_tile64', lane, 'the lane-facing entry forwards it')
        # Nothing in the engine may pass it: a default flip needs the GPU verdict first.
        for name in ('engine/profiles/glm53/net.py', 'engine/profiles/glm53/lanes.py'):
            self.assertNotIn('_prefill_tile64', (ROOT / name).read_text(encoding='utf-8'), name)

    def test_the_gpu_mode_refuses_to_run_without_the_consumer_ranks(self):
        import subprocess
        import sys
        done = subprocess.run([sys.executable, str(PROBE), '--gpu'],
                              capture_output=True, text=True)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn('--ranks', done.stderr)


if __name__ == '__main__':
    unittest.main()
