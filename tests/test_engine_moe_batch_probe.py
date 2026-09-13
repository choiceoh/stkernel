"""An experimental wider tile must neither alias nor enable a served handle."""
import unittest

from tests.test_engine_moe_waves import namespace


class BatchReformConfigTests(unittest.TestCase):
    def setUp(self):
        self.ns = namespace()
        self.base = self.ns['_parse_glm53_static_v2']('t,r,sf6')
        self.probe = dict(self.base, probe_batch_reform=True)

    def test_served_wide_rows_keep_original_geometry(self):
        choose = self.ns['_static_v2_decode_config']
        for rows in (14, 21, 28):
            self.assertFalse(choose(self.base, rows)['decode_reform'])
            self.assertTrue(choose(self.probe, rows)['decode_reform'])
        for rows in (1, 6, 7, 8):
            self.assertTrue(choose(self.base, rows)['decode_reform'])
        self.assertNotIn('probe_batch_reform', self.base)

    def test_unqualified_shapes_and_layouts_are_refused(self):
        choose = self.ns['_static_v2_decode_config']
        for rows in (0, 1, 7, 8, 13, 15, 16, 20, 22, 27, 29, 32, 128):
            with self.assertRaisesRegex(ValueError, '14/21/28'):
                choose(self.probe, rows)
        for field in ('decode_reform', 'tiled', 'reform_sf_pack'):
            with self.assertRaisesRegex(ValueError, 'packed'):
                choose(dict(self.probe, **{field: False}), 28)
        with self.assertRaises(ValueError):
            choose(dict(self.probe, even=True), 28)

    def test_independent_compilation_orders_keep_distinct_handles(self):
        choose, key = (self.ns[name] for name in ('_static_v2_decode_config', '_static_v2_cache_key'))
        for rows in (14, 21, 28):
            configs = [choose(config, rows) for config in (self.base, self.probe)]
            for order in (configs, list(reversed(configs))):
                cache = {key(config, m=rows): config['decode_reform'] for config in order}
                self.assertEqual(len(cache), 2)
                self.assertFalse(cache[key(configs[0], m=rows)])
                self.assertTrue(cache[key(configs[1], m=rows)])
        for suffix in ('batch', 'batch_reform', 'probe_batch_reform'):
            with self.assertRaises(ValueError):
                self.ns['_parse_glm53_static_v2']('t,r,sf6,' + suffix, probe=True)


if __name__ == '__main__':
    unittest.main()
