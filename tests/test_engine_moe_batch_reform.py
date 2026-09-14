"""C2 expert tiles preserve capacity and every other row-count's kernel identity."""
import unittest

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
                self.assertTrue(chosen['decode_reform'])


if __name__ == '__main__':
    unittest.main()
