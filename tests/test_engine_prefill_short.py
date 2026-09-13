"""Short prefill must keep decode, raw scales and non-TP layouts separate."""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ShortPrefillTests(unittest.TestCase):
    def test_word_admission_never_captures_decode_or_raw_scale_cells(self):
        source = ROOT / 'engine/kernels/b12x/moe_dispatch.py'
        tree = ast.parse(source.read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == '_short_prefill_q0_word_unpack')
        ns = {}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), 'exec'), ns)
        select = ns[fn.name]
        for rows in (1, 7, 28, 64, 65, 128, 2121, 2128, 4095, 8192, 8193, 32256, True):
            for q0 in (False, True):
                for packed in (False, True):
                    for ep in (False, True):
                        got = select(m=rows, tp_sf6_q0=q0, reform_sf_pack=packed, ep_local=ep)
                        expected = rows in (65, 128, 2121, 2128, 4095, 8192) and q0 and packed and not ep
                        self.assertEqual(got, expected, (rows, q0, packed, ep))

    def test_only_scale_load_methods_are_overridden(self):
        source = ROOT / 'engine/kernels/b12x/moe_dynamic_gated_sf6_q0_words.py'
        cls = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef))
        self.assertEqual([ast.unparse(n) for n in cls.bases], ['MoEGatedDynamicKernelSF6Q0'])
        assignments = {ast.unparse(n.targets[0]): ast.unparse(n.value)
                       for n in cls.body if isinstance(n, ast.Assign)}
        self.assertEqual(assignments, {
            'load_fc1_tma_slice': 'MoEGatedDynamicKernelSF6Words.load_fc1_tma_slice',
            'load_fc2_tma_tile': 'MoEGatedDynamicKernelSF6Words.load_fc2_tma_tile'})
        self.assertEqual([n.name for n in cls.body if isinstance(n, ast.FunctionDef)], ['__init__'])


if __name__ == '__main__':
    unittest.main()
