"""Keep a short shape experiment isolated from serving defaults."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from probes.engine_decode_k7 import candidate_projections, verification_rows, w4_projection_keys


class K7ProbeTests(unittest.TestCase):
    def test_output_extension_keeps_three_independently_bounded_components(self):
        from probes import engine_decode_bundle as bundle
        commands = []
        def execute(command, **kwargs):
            commands.append((command, kwargs))
            return SimpleNamespace(returncode=1 if len(commands) == 2 else 0)
        with patch.object(bundle, 'require_current_probe'), patch.object(bundle.subprocess, 'run', side_effect=execute), \
                patch('builtins.print'), self.assertRaises(RuntimeError):
            bundle.check('/immutable/ranks', bundle='k7_output_bundle')
        self.assertEqual(len(commands), 3)
        self.assertEqual(commands[-1][0][-4:], ['--lanes', 'moe_output', '--ranks', '/immutable/ranks'])
        self.assertEqual([kw['timeout'] for _, kw in commands], [300, 300, 300])

    def test_combined_hold_retains_a_failed_commit_result_and_checks_dense(self):
        from probes import engine_decode_bundle as bundle
        commands = []
        def execute(command, **kwargs):
            commands.append((command, kwargs))
            return SimpleNamespace(returncode=1 if len(commands) == 1 else 0)
        with patch.object(bundle, 'require_current_probe'), patch.object(bundle.subprocess, 'run', side_effect=execute), \
                patch('builtins.print'):
            with self.assertRaisesRegex(RuntimeError, 'one or more decode candidates failed'):
                bundle.check('/immutable/ranks', bundle='k7_commit_bundle')
        self.assertEqual(len(commands), 2)
        self.assertIn('--commit-only', commands[0][0])
        self.assertNotIn('--ranks', commands[0][0])
        self.assertEqual(commands[1][0][-4:], ['--lanes', 'decode_k7', '--ranks', '/immutable/ranks'])
        self.assertEqual([kwargs['timeout'] for _, kwargs in commands], [300, 300])

    def test_real_weight_selection_follows_the_declared_mlp_layout(self):
        common = ['L0.kda.in_proj', 'L0.kda.o_proj', 'L3.mla.qkv_a',
                  'L3.mla.q_b', 'L3.mla.o_proj', 'L3.idx.wq_b']
        plain = ['L0.mlp.gate_up', 'L0.mlp.down']
        packed = ['L0.mlp.w13', 'L0.mlp.w2']
        for extras, expected, is_packed in ((plain, common + plain, False), (packed, common, True)):
            keys, actual = w4_projection_keys(SimpleNamespace(keys=lambda: common + extras))
            self.assertEqual(keys, expected)
            self.assertIs(actual, is_packed)
        for keys in (common, common + plain[:1], common + plain + packed, common[1:] + packed):
            with self.assertRaises(RuntimeError):
                w4_projection_keys(SimpleNamespace(keys=lambda: keys))

    def test_scope_rejects_stale_draft_depth(self):
        self.assertEqual(verification_rows(7), (8, 16, 24, 32))
        for k in (0, 6, 8, 7., True, '7'):
            with self.assertRaises(ValueError):
                verification_rows(k)

    def test_projection_override_retains_old_cells_and_unwinds_on_failure(self):
        original = (1, 6, 7, 14, 21, 28)
        module = SimpleNamespace(DECODE_ROWS=original)
        with self.assertRaisesRegex(RuntimeError, 'probe failed'):
            with candidate_projections(module, verification_rows(7)):
                self.assertTrue(set(original).issubset(module.DECODE_ROWS))
                self.assertTrue(set((8, 16, 24, 32)).issubset(module.DECODE_ROWS))
                with candidate_projections(module, (8, 16)):
                    self.assertEqual(module.DECODE_ROWS, (1, 6, 7, 8, 14, 16, 21, 24, 28, 32))
                raise RuntimeError('probe failed')
        self.assertIs(module.DECODE_ROWS, original)


if __name__ == '__main__':
    unittest.main()
