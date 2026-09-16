"""The v2 dense GEMM's L2 slice prefetch is a bench knob, off on the served path (2026-09-16).

Read from the source, as the other native-dense structure tests do: the kernel asks for its slice from one
thread before the PDL wait and only when the captured context carries the knob; the context takes the knob
from the process global at its single construction site; the dense cells probe holds the knob only around
the captures of its *_l2 arms.
"""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]


class DenseL2PrefetchTests(unittest.TestCase):
    def test_kernel_prefetches_its_slice_from_one_thread_before_the_pdl_wait(self):
        source = (ROOT / 'engine/kernels/dense/kernels.cu').read_text()
        self.assertIn('"cp.async.bulk.prefetch.L2.global [%0], %1;"', source)
        self.assertIn('  int l2_prefetch = 0;     // bench knob', source)
        block = re.search(r'  if \(c\.l2_prefetch && threadIdx\.x == 0\) \{(.*?)\n  \}\n', source, re.S)
        self.assertIsNotNone(block, 'the prefetch block is missing or not gated on the context knob and thread 0')
        self.assertEqual(block.group(1).count('mk_prefetch_l2('), 2)   # W records and their scales
        after = source[block.end():block.end() + 400]
        self.assertIn('stage_raw(kb0 + d, (kb0 + d) % NB);', after)
        self.assertIn('griddepcontrol.wait', after)
        self.assertLess(source.index('if (c.l2_prefetch && threadIdx.x == 0)'),
                        source.index('  asm volatile("griddepcontrol.wait;" ::: "memory");\n  if constexpr (DIRECT)'))

    def test_context_takes_the_knob_once_and_the_served_default_is_off(self):
        source = (ROOT / 'engine/kernels/dense/kernels.cu').read_text()
        self.assertIn('int g_mk2_l2_prefetch = 0;', source)
        self.assertEqual(source.count('c2.l2_prefetch = g_mk2_l2_prefetch;'), 1)
        self.assertEqual(source.count('MKGemm2Ctx c2{};'), 1)
        self.assertIn('m.def("set_gemm2_l2_prefetch"', source)

    def test_probe_holds_the_knob_only_around_its_l2_arms(self):
        source = (ROOT / 'probes/engine_dense_cells.py').read_text()
        self.assertIn('ext.set_gemm2_l2_prefetch(1)', source)
        self.assertIn('ext.set_gemm2_l2_prefetch(0)', source)
        for route in ('bound_l2', 'generic_l2', 'pair_l2'):
            self.assertIn(f"'{route}': lambda ext, owners, x, d: _l2(ext,", source)
        # every cell's plan pairs the served route with its _l2 twin at each row count it lists
        import ast
        tree = ast.parse(source)
        cells = next(n for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'CELLS')
        for cell in cells.value.elts:
            plan = cell.elts[5]
            for rows, pairs in zip(plan.keys, plan.values):
                arms = [(a.value, b.value) for pair in pairs.elts for a, b in [pair.elts]]
                self.assertTrue(any(b == a + '_l2' for a, b in arms), (cell.elts[0].value, rows.value, arms))


if __name__ == '__main__':
    unittest.main()
