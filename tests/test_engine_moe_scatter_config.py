"""Private larger-output kernels must never alias a served graph handle."""
import unittest
from tests.test_engine_moe_waves import namespace


class ScatterConfigTests(unittest.TestCase):
    def test_cache_abi_and_parser_isolation(self):
        ns = namespace()
        base = ns['_parse_glm53_static_v2']('t,r,sf6')
        key = ns['_static_v2_cache_key']
        keys = {key(dict(base, probe_route_scatter=r, probe_direct_scatter=d), m=7)
                for r in (False, True) for d in (False, True)}
        self.assertEqual(len(keys), 4)
        self.assertEqual(key(base, m=7), key(dict(base, probe_route_scatter=False, probe_direct_scatter=False), m=7))
        for token in ('probe_route_scatter', 'probe_direct_scatter'):
            with self.assertRaises(ValueError):
                ns['_parse_glm53_static_v2']('t,r,sf6,' + token, probe=True)

    def test_padded_batch_geometry_is_retained_without_explicit_reform(self):
        ns = namespace()
        config = dict(ns['_parse_glm53_static_v2']('t,r,sf6'), probe_route_scatter=True, probe_direct_scatter=True)
        for rows in (7, 14, 21, 28):
            chosen = ns['_static_v2_decode_config'](config, rows)
            self.assertEqual(chosen['decode_reform'], rows == 7)
        for rows in (0, 1, 6, 8, 29, 128):
            with self.assertRaises(ValueError):
                ns['_static_v2_decode_config'](config, rows)
        for field in ('tiled', 'reform_sf_pack', 'decode_reform'):
            with self.assertRaises(ValueError):
                ns['_static_v2_decode_config'](dict(config, **{field: False}), 7)
        for field in ('split', 'skip_a', 'skip_sf', 'even'):
            with self.assertRaises(ValueError):
                ns['_static_v2_decode_config'](dict(config, **{field: True}), 7)


if __name__ == '__main__':
    unittest.main()
