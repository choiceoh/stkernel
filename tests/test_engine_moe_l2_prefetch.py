"""The l<n> cell: the static MoE DMA warp prefetches B stages n ahead into L2 (2026-09-16).

A hint, not a layout: the MMA reads the same bytes. The parser admits it only over the tile-major M16 reform
(t,r), the cache key tells depths apart, and the kernel source issues one bulk L2 prefetch per box from the
DMA lane alone -- read from the source, as the other MoE config tests do, since the DSL is not importable here.
"""
import ast
from pathlib import Path
import re
import unittest

from tests.test_engine_moe_scatter_config import namespace

ROOT = Path(__file__).resolve().parents[1]


class L2PrefetchCellTests(unittest.TestCase):
    def test_parser_admits_l_over_the_reform_only(self):
        ns = namespace()
        parse = ns['_parse_glm53_static_v2']
        self.assertEqual(parse('t,r,sf6,batch')['l2_prefetch'], 0)
        for depth in (2, 4, 8):
            cfg = parse(f't,r,sf6,batch,l{depth}')
            self.assertEqual(cfg['l2_prefetch'], depth)
            self.assertTrue(cfg['tiled'] and cfg['decode_reform'])
        for spec in ('u,l4', 't,l4', 'v,l2', 't,lf4'):
            with self.assertRaisesRegex(ValueError, 'l<n> requires t,r'):
                parse(spec)
        with self.assertRaises(ValueError):
            parse('t,r,sf6,lx')
        # lf<n>: the FC2-only variant (the first ticket measured FC1 slower under its own prefetch)
        both, fc2 = parse('t,r,sf6,batch,l4'), parse('t,r,sf6,batch,lf4')
        self.assertEqual((both['l2_prefetch'], both['l2_prefetch_fc1']), (4, True))
        self.assertEqual((fc2['l2_prefetch'], fc2['l2_prefetch_fc1']), (4, False))
        self.assertNotEqual(ns['_static_v2_cache_key'](both, m=16), ns['_static_v2_cache_key'](fc2, m=16))

    def test_cache_key_tells_depths_apart_and_zero_is_the_served_key(self):
        ns = namespace()
        key = ns['_static_v2_cache_key']
        base = ns['_parse_glm53_static_v2']('t,r,sf6,batch')
        keys = {key(dict(base, l2_prefetch=d), m=16) for d in (0, 2, 4, 8)}
        self.assertEqual(len(keys), 4)
        served = dict(base)
        del served['l2_prefetch']
        self.assertEqual(key(served, m=16), key(base, m=16))

    def test_kernel_issues_the_prefetch_from_the_dma_lane_only(self):
        source = (ROOT / 'engine/kernels/b12x/moe_static_kernel_v4.py').read_text()
        tree = ast.parse(source)
        calls = [node for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and getattr(node.func, 'id', None) == '_bulk_prefetch_l2']
        # gate + up ahead, the FC2 box past FC1's end, and the FC2 box ahead inside the FC2 loop
        self.assertEqual(len(calls), 4)
        lines = source.splitlines()
        for call in calls:
            above = "\n".join(lines[max(0, call.lineno - 24):call.lineno])
            self.assertIn('if is_dma_lane0:', above, f'line {call.lineno}: the prefetch must be one lane')
            self.assertIn('self.l2_prefetch > 0', above, f'line {call.lineno}: the prefetch must be a compile-time cell')
        # the two FC1 boxes sit under the FC1 switch; the seam and FC2 boxes do not
        fc1_gated = [c for c in calls if 'self.l2_prefetch_fc1' in '\n'.join(lines[max(0, c.lineno - 8):c.lineno])]
        self.assertEqual(len(fc1_gated), 2)
        common = (ROOT / 'engine/kernels/b12x/moe_static_common.py').read_text()
        self.assertIn('"cp.async.bulk.prefetch.L2.global [$0], $1;"', common)
        # the raw storage tensors ride the kernel signature and both launch sites hand them over
        self.assertIn('b_w13_raw: cute.Tensor', source)
        self.assertEqual(len(re.findall(r'sfb2_packed,\n\s+b_w13,\n\s+b_down,\n\s+\)\.launch\(', source)), 1)
        v5 = (ROOT / 'engine/kernels/b12x/moe_static_kernel_v5.py').read_text()
        self.assertEqual(len(re.findall(r'sfb2_packed,\n\s+b_w13,\n\s+b_down,\n\s+\)\.launch\(', v5)), 1)

    def test_dispatcher_refuses_the_prefetch_off_the_256_chunk(self):
        source = (ROOT / 'engine/kernels/b12x/moe_dispatch.py').read_text()
        self.assertIn('if l2_prefetch and (not tiled or not reform or chunk != 256):', source)
        self.assertIn('l2_prefetch=l2_prefetch,', source)


if __name__ == '__main__':
    unittest.main()
