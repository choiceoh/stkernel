"""C2 expert tiles preserve capacity and every other row-count's kernel identity."""
import unittest
from pathlib import Path

from tests.test_engine_moe_scatter_config import namespace


class BatchReformTests(unittest.TestCase):
    def test_only_declared_k7_batch_shapes_change_compiled_handles(self):
        ns = namespace()
        parse, choose, key = (ns[name] for name in
            ('_parse_glm53_static_v2', '_static_v2_decode_config', '_static_v2_cache_key'))
        base, candidate = parse('t,r,sf6'), parse('t,r,sf6,batch')
        for rows in range(129):
            old, new = choose(base, rows), choose(candidate, rows)
            changed = rows == 16
            self.assertEqual(key(old, m=rows) != key(new, m=rows), changed)
            self.assertEqual(choose(new, rows), new, 'capture/compile normalize twice')
            for feature in ('decode_reform', 'fc1_reuse_a', 'compact_staging', 'sf6_registers'):
                self.assertEqual(new[feature], 1 <= rows <= 8 or changed)
            self.assertEqual(new['c2_direct_scatter'], changed)
            self.assertEqual(new['c2_scatter_reuse'], changed)
            self.assertEqual(new['c2_fc2_prefetch'], changed)
            tile_only = choose(dict(candidate, c2_direct_scatter=False), rows)
            self.assertEqual(key(tile_only, m=rows) != key(new, m=rows), changed)
            self.assertFalse(tile_only['c2_scatter_reuse'])
            self.assertFalse(tile_only['c2_fc2_prefetch'])
            direct_only = choose(dict(candidate, c2_scatter_reuse=False), rows)
            self.assertEqual(key(direct_only, m=rows) != key(new, m=rows), changed)
            self.assertFalse(direct_only['c2_fc2_prefetch'])
            retained = choose(dict(candidate, c2_fc2_prefetch=False), rows)
            self.assertEqual(key(retained, m=rows) != key(new, m=rows), changed)

    def test_the_companion_lane_keeps_the_m16_cell_without_sf6(self):
        """A mixed-provenance checkpoint builds a reform_sf_pack=False lane beside every sf6 lane.

        At m=16 `batch` turns direct register scatter on for that companion too. The kernel used to
        refuse the pair and the boot died in warmup_decode_experts
        (measurements/st_hybrid_boot_block_20260916); #1056 bought the boot by refusing the companion
        the scatter. It now keeps it: private scatter writes from the MMA registers to the packed
        FP32 output and reads no scale state, so reform_sf_pack -- the FC1 *scale* packing -- was
        never its business. What still needs sf6 is the reuse ring, and that stays off.
        """
        ns = namespace()
        choose = ns['_static_v2_decode_config']
        served = ns['_parse_glm53_static_v2']('t,r,sf6,batch')
        companion = dict(served, reform_sf_pack=False)
        for rows in range(129):
            with self.subTest(rows=rows):
                c = choose(companion, rows)
                self.assertEqual(c['c2_direct_scatter'], rows == 16)
                self.assertEqual(choose(served, rows)['c2_direct_scatter'], rows == 16)
                # the reuse ring and its prefetch read sf6 registers: never the companion's
                self.assertFalse(c['c2_scatter_reuse'])
                self.assertFalse(c['c2_fc2_prefetch'])
                self.assertFalse(c['sf6_registers'])
                self.assertEqual(choose(c, rows), c, 'capture/compile normalize twice')

    def test_the_kernel_asks_private_scatter_only_for_the_output_it_writes(self):
        """The guard that refused the companion, pinned so it cannot be re-tightened.

        `_validate_direct_scatter_layout` binds register pairs to `epi_tile = (tile_m, fc2_tile_n)`
        and touches no scale state, so direct scatter's requirement is the packed FP32 output and an
        unsplit epilogue. route_scatter re-indexes the output by route and has only been built on the
        sf6 tile, so it keeps the original pairing.
        """
        source = (Path(__file__).resolve().parents[1]
                  / 'engine/kernels/b12x/moe_static_kernel_v4.py').read_text()
        body = source.split('def __init__(', 1)[1].split('\n    def ', 1)[0]
        code = ' '.join(l.split('#')[0] for l in body.splitlines())   # comments name it to explain it
        direct = code.split('if self.direct_scatter and not (', 1)[1].split(')', 1)[0]
        route = code.split('if self.route_scatter and not (', 1)[1].split(')', 1)[0]
        self.assertIn('scatter_fp32', direct)
        self.assertIn('not split', direct)
        self.assertNotIn('reform_sf_pack', direct)
        self.assertIn('reform_sf_pack', route)

    def test_recipe_requires_the_existing_packed_reform_contract(self):
        parse = namespace()['_parse_glm53_static_v2']
        for recipe in ('batch', 'u,batch', 't,batch', 't,r,batch', 't,r,sf6,batch,f4'):
            with self.subTest(recipe=recipe), self.assertRaises(ValueError):
                parse(recipe)
        self.assertTrue(parse('batch,sf6,t,r')['batch_reform'])

    def test_runtime_control_preserves_independent_operand_comparisons(self):
        ns = namespace()
        base = ns['_parse_glm53_static_v2']('t,r,sf6,batch')
        for rows in (16,):
            for feature in ('sf6_registers', 'compact_staging', 'fc1_reuse_a'):
                chosen = ns['_static_v2_decode_config'](dict(base, **{feature: False}), rows)
                self.assertFalse(chosen[feature])
                self.assertFalse(chosen['sf6_registers'])
                self.assertFalse(chosen['c2_scatter_reuse'])
                self.assertFalse(chosen['c2_fc2_prefetch'])
                self.assertTrue(chosen['decode_reform'])

    def test_prefetch_preserves_explicit_stage_controls(self):
        ns = namespace()
        base = ns['_parse_glm53_static_v2']('t,r,sf6,batch')
        for stages in (1, 2, 3):
            chosen = ns['_static_v2_decode_config'](dict(base, fc2=stages), 16)
            self.assertEqual(chosen['c2_fc2_prefetch'], stages == 2)
            self.assertEqual(chosen['fc2'], stages)
            self.assertEqual(ns['_static_v2_decode_config'](chosen, 16), chosen)


if __name__ == '__main__':
    unittest.main()
