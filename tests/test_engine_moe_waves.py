"""Keep the resident-wave experiment out of served configs and baseline caches."""
import ast
from pathlib import Path
import unittest


def namespace():
    path = Path(__file__).resolve().parents[1] / 'engine/kernels/b12x/moe_dispatch.py'
    tree = ast.parse(path.read_text())
    names = {'_parse_glm53_static_v2', '_static_v2_decode_config', '_static_v2_cache_key'}
    constants = {'_STATIC_V2_DEFAULT', '_STATIC_SUNSET_TOKENS', '_GLM53_B12X_STATIC_V2_ENV'}
    nodes = [node for node in tree.body
             if (isinstance(node, ast.FunctionDef) and node.name in names)
             or (isinstance(node, ast.Assign) and any(
                 isinstance(target, ast.Name) and target.id in constants for target in node.targets))]
    ns = dict(Tuple=tuple, _static_kernel_cache_key=lambda **fields: tuple(sorted(fields.items())))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
    return ns


class WaveConfigTests(unittest.TestCase):
    def setUp(self):
        self.ns = namespace()
        self.base = self.ns['_parse_glm53_static_v2']('t,r,sf6')
        self.probe = dict(self.base, even=True)

    def test_served_and_legacy_parsers_do_not_admit_probe(self):
        self.assertFalse(self.base.get('even', False))
        for suffix in ('e', 'k', 'even'):
            for probe in (False, True):
                with self.assertRaises(ValueError):
                    self.ns['_parse_glm53_static_v2']('t,r,sf6,' + suffix, probe=probe)

    def test_cache_handles_stay_distinct_in_both_compile_orders(self):
        key = self.ns['_static_v2_cache_key']
        self.assertEqual(key(self.base, m=7), key(dict(self.base, even=False), m=7))
        for order in ((self.base, self.probe), (self.probe, self.base)):
            cache = {key(config, m=7): config.get('even', False) for config in order}
            self.assertEqual(len(cache), 2)
            self.assertFalse(cache[key(self.base, m=7)])
            self.assertTrue(cache[key(self.probe, m=7)])
        self.assertNotEqual(key(self.probe, m=7), key(self.probe, m=8))

    def test_ineligible_probe_shapes_fail_before_compilation(self):
        choose = self.ns['_static_v2_decode_config']
        for rows in (1, 6, 7, 8):
            self.assertTrue(choose(self.probe, rows)['even'])
        for rows in (0, 9, 24, 128):
            with self.assertRaisesRegex(ValueError, '1..8 tokens'):
                choose(self.probe, rows)
        for field in ('decode_reform', 'tiled', 'reform_sf_pack'):
            with self.assertRaisesRegex(ValueError, 't,r,sf6'):
                choose(dict(self.probe, **{field: False}), 7)
        self.assertFalse(choose(self.base, 24)['decode_reform'])
        self.assertTrue(self.base['decode_reform'])


if __name__ == '__main__':
    unittest.main()
