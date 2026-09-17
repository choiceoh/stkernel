"""Private larger-output kernels must never alias a served graph handle."""
import unittest
import ast
from pathlib import Path


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


class ScatterConfigTests(unittest.TestCase):
    def test_input_reuse_scope_and_cache_identity(self):
        ns = namespace()
        parse, select, key = (ns[n] for n in ('_parse_glm53_static_v2', '_static_v2_decode_config', '_static_v2_cache_key'))
        base = parse('t,r,sf6,batch')
        for rows in (8, 16):
            configs = [select(dict(base, input_reuse=mode), rows) for mode in (0, 1, 2)]
            self.assertEqual(len({key(c, m=rows) for c in configs}), 3)
            self.assertEqual(key(configs[0], m=rows), key(select(base, rows), m=rows))
            for config in configs:
                self.assertEqual(select(config, rows), config)
        for rows in (1, 7, 12, 24, 32, 128):
            with self.assertRaisesRegex(ValueError, 'input reuse'):
                select(dict(base, input_reuse=1), rows)
        for changed in ({'input_reuse': 3}, {'input_reuse': 1, 'input_vec16': False}):
            with self.assertRaisesRegex(ValueError, 'input reuse'):
                select(dict(base, **changed), 8)

    def test_vector_input_scope_rollback_and_cache_identity(self):
        ns = namespace()
        parse, select, key = (ns[n] for n in ('_parse_glm53_static_v2', '_static_v2_decode_config', '_static_v2_cache_key'))
        for recipe in ('t,r,sf6,batch', 't,r,sf6', 't'):
            for rows in (1, 6, 7, 8, 12, 16, 24, 32):
                cfg = select(parse(recipe), rows)
                enabled = cfg['decode_reform'] and rows in (8, 16)
                self.assertEqual(cfg['input_vec16'], enabled)
                self.assertEqual(select(cfg, rows), cfg)
                control = select(dict(parse(recipe), input_vec16=False), rows)
                self.assertFalse(control['input_vec16'])
                self.assertEqual(key(cfg, m=rows) != key(control, m=rows), enabled)

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
            # The actual launch normalizes once; the compiler normalizes again.
            # Wide rows intentionally have decode_reform=False after pass one.
            twice = ns['_static_v2_decode_config'](chosen, rows)
            self.assertEqual(twice, chosen)
            self.assertEqual(ns['_static_v2_cache_key'](twice, m=rows),
                             ns['_static_v2_cache_key'](chosen, m=rows))
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
