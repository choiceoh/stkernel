"""Cell z redone for the M16 reform tile (2026-09-17): the parser, the cache key, the host pre-swizzle and the
kernel's bulk B stages -- read from the source where the DSL is not importable, run where torch suffices.

The pre-swizzle is the stage's own byte order (probes/b12x_reform_layout_print.py enumerated it on the CPU):
FC1 stages are 128 rows x 128 B under Swizzle<3,4,3> (chunk c of row r lands at c ^ (r % 8)), FC2 stages are
256 rows x 64 B under Swizzle<2,4,3> (c ^ ((r // 2) % 4)). Both are involutions per row.
"""
import ast
from pathlib import Path
import unittest

import torch

from tests.test_engine_moe_scatter_config import namespace

ROOT = Path(__file__).resolve().parents[1]


def _swizzle():
    """_swizzle_tile_boxes lifted from the dispatcher source (the module imports the DSL)."""
    source = (ROOT / 'engine/kernels/b12x/moe_dispatch.py').read_text()
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_swizzle_tile_boxes')
    ns = {'torch': torch, 'Tuple': tuple}
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'moe_dispatch.py', 'exec'), ns)
    return ns['_swizzle_tile_boxes']


class BulkCellTests(unittest.TestCase):
    def test_parser_admits_z_over_the_reform_and_the_key_tells_it_apart(self):
        ns = namespace()
        parse, key = ns['_parse_glm53_static_v2'], ns['_static_v2_cache_key']
        served = parse('t,r,sf6,batch')
        self.assertFalse(served['bulk_b'])
        z = parse('t,r,sf6,batch,z')
        self.assertTrue(z['bulk_b'])
        self.assertNotEqual(key(served, m=16), key(z, m=16))
        for spec in ('u,z', 't,z'):
            with self.assertRaisesRegex(ValueError, 'z requires t,r'):
                parse(spec)

    def test_pre_swizzle_is_the_enumerated_stage_order_and_an_involution(self):
        swizzle = _swizzle()
        e, kt, rows = 2, 3, 256
        w13 = torch.randint(0, 256, (e, kt, rows, 128), dtype=torch.uint8)
        w2 = torch.randint(0, 256, (e, 4, 512, 64), dtype=torch.uint8)
        w13_z, w2_z = swizzle(w13, w2)
        self.assertEqual(w13_z.shape, w13.shape)
        self.assertEqual(w2_z.shape, w2.shape)
        # FC1: within every 128-row box, row r's chunk c came from chunk c ^ (r % 8); bytes inside a chunk stay
        for box in range(rows // 128):
            for r in (0, 1, 7, 8, 13, 127):
                for c in range(8):
                    want = w13[1, 2, box * 128 + r, (c ^ (r % 8)) * 16:(c ^ (r % 8)) * 16 + 16]
                    self.assertTrue(torch.equal(w13_z[1, 2, box * 128 + r, c * 16:c * 16 + 16], want), (box, r, c))
        # FC2: within every 256-row box, chunk c of row r came from c ^ ((r // 2) % 4)
        for box in range(2):
            for r in (0, 1, 2, 3, 8, 255):
                for c in range(4):
                    src = c ^ ((r // 2) % 4)
                    want = w2[0, 3, box * 256 + r, src * 16:src * 16 + 16]
                    self.assertTrue(torch.equal(w2_z[0, 3, box * 256 + r, c * 16:c * 16 + 16], want), (box, r, c))
        again13, again2 = swizzle(w13_z, w2_z)
        self.assertTrue(torch.equal(again13, w13))
        self.assertTrue(torch.equal(again2, w2))
        with self.assertRaises(ValueError):
            swizzle(torch.zeros(1, 1, 128, 256, dtype=torch.uint8), w2)   # the 512 chunk has no 128 B rows
        # the diagnostics' other two orders: the nibble swizzle moves 8 B units by (2r + half) % 8 (FC1) and r % 4
        # (FC2) and is an involution too; plain is the identity
        n13, n2 = swizzle(w13, w2, kind='nibble')
        for r in (0, 1, 3, 4, 127):
            for u in range(16):
                src = u ^ ((2 * r + (u >> 3)) % 8)
                self.assertTrue(torch.equal(n13[0, 0, r, u * 8:u * 8 + 8], w13[0, 0, r, src * 8:src * 8 + 8]), (r, u))
        for r in (0, 1, 2, 3, 255):
            for u in range(8):
                src = u ^ (r % 4)
                self.assertTrue(torch.equal(n2[1, 1, r, u * 8:u * 8 + 8], w2[1, 1, r, src * 8:src * 8 + 8]), (r, u))
        back13, back2 = swizzle(n13, n2, kind='nibble')
        self.assertTrue(torch.equal(back13, w13) and torch.equal(back2, w2))
        p13, p2 = swizzle(w13, w2, kind='plain')
        self.assertTrue(torch.equal(p13, w13) and torch.equal(p2, w2))
        with self.assertRaises(ValueError):
            swizzle(w13, w2, kind='other')

    def test_kernel_lands_each_b_stage_with_one_bulk_copy_from_the_dma_lane(self):
        source = (ROOT / 'engine/kernels/b12x/moe_static_kernel_v4.py').read_text()
        self.assertIn('sb1_base_addr = shared_ptr_to_u32(cute.recast_ptr(storage.sB1.data_ptr(), dtype=cutlass.Uint8))', source)
        self.assertIn('sb2_base_addr = shared_ptr_to_u32(cute.recast_ptr(storage.sB2.data_ptr(), dtype=cutlass.Uint8))', source)
        self.assertEqual(source.count('if cutlass.const_expr(self.bulk_b):'), 2)
        for stage, box in (('sb1_base_addr + fc1_prod_state.index * fc1_box_bytes', 'fc1_box_bytes, shared_ptr_to_u32(bar))'),
                           ('sb2_base_addr + fc2_prod_state.index * fc2_box_bytes', 'fc2_box_bytes, shared_ptr_to_u32(bar2))')):
            self.assertIn(stage, source)
            self.assertIn(box, source)
        lines = source.splitlines()
        for i, line in enumerate(lines):
            if 'if cutlass.const_expr(self.bulk_b):' in line:
                self.assertIn('if is_dma_lane0:', lines[i + 1] + lines[i + 2] + lines[i + 3] + lines[i + 4])
        dispatch = (ROOT / 'engine/kernels/b12x/moe_dispatch.py').read_text()
        self.assertIn('cell z and pre-swizzled expert storage must agree', dispatch)
        self.assertIn('    swizzled: bool = False', dispatch)


if __name__ == '__main__':
    unittest.main()
