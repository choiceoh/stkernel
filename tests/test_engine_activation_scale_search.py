"""CPU gates for activation-search eligibility and preserving the old ss recipe."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'engine/kernels/b12x/moe_dispatch.py'
TREE = ast.parse(SOURCE.read_text())


def policy():
    names = {'_parse_glm53_static_v2', '_activation_scale_search_for',
             '_glm_tp_scatter_shape', '_glm_tp_scatter_fp32'}
    constants = {'_STATIC_V2_DEFAULT', '_STATIC_SUNSET_TOKENS', '_GLM53_B12X_STATIC_V2_ENV'}
    nodes = [copy.deepcopy(n) for n in TREE.body if
             isinstance(n, ast.FunctionDef) and n.name in names or
             isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in n.targets)]
    ns = dict(_STATIC_V2_OVERRIDE=None, _GLM53_B12X_STATIC_V2=None, _ACTIVATION_SCALE_SEARCH_RADIUS=None,
              _admitted_moe=lambda: SimpleNamespace(experts_local=288, hidden=4096,
                inter_local=512, dense_inter_local=3072, topk=8, quant='nvfp4',
                activation='swigluoai_uninterleave', swiglu_limit=10.))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), ns)
    return ns


class ActivationSearchTests(unittest.TestCase):
    def test_ep_compact_pairs_keep_the_bound_precision_and_explicit_control(self):
        ns = policy()
        ns['_admitted_moe'] = lambda: SimpleNamespace(experts=512, experts_local=128, hidden=2560,
            inter_local=640, dense_inter_local=160, topk=10, quant='nvfp4', activation='silu', swiglu_limit=None)
        ns['_ACTIVATION_SCALE_SEARCH_RADIUS'] = 2
        geom = dict(state_E=128, weight_E=128, k=2560, n=640, num_topk=10, quant_mode='nvfp4',
                    activation='silu', swiglu_alpha=1., swiglu_beta=0., swiglu_limit=None)
        for topk in (1, 10):
            row = geom | dict(num_topk=topk)
            self.assertTrue(ns['_glm_tp_scatter_fp32'](**row))
            self.assertEqual(ns['_activation_scale_search_for'](**row), 2)
        for change in (dict(num_topk=2), dict(n=512), dict(k=4096), dict(activation='relu2'), dict(state_E=512)):
            self.assertEqual(ns['_activation_scale_search_for'](**(geom | change)), 0)
        ns['_STATIC_V2_OVERRIDE'] = dict(activation_scale_search=0)
        self.assertEqual(ns['_activation_scale_search_for'](**geom), 0)

    def test_old_recipe_is_static_fc2_only_and_all_recipe_covers_dense(self):
        ns = policy()
        geometry = dict(state_E=288, weight_E=288, k=4096, n=512, num_topk=8,
            quant_mode='nvfp4', activation='swigluoai_uninterleave',
            swiglu_alpha=1., swiglu_beta=0., swiglu_limit=10.)
        select = ns['_activation_scale_search_for']
        parse = ns['_parse_glm53_static_v2']
        for token, radius in [('ss1', 0), ('ss2', 0), ('as1', 1), ('as2', 2)]:
            ns['_GLM53_B12X_STATIC_V2'] = parse('t,r,sf6,batch,' + token)
            self.assertEqual(select(**geometry), radius)
            self.assertEqual(select(**(geometry | dict(state_E=1, weight_E=1, n=3072, num_topk=1))), radius)
            for change in [dict(quant_mode='mxfp4'), dict(quant_mode='w4a16'),
                           dict(num_topk=7), dict(state_E=72), dict(k=2048),
                           dict(n=511), dict(activation='silu'), dict(swiglu_alpha=1.702),
                           dict(swiglu_beta=1.), dict(swiglu_limit=9.)]:
                with self.subTest(token=token, change=change):
                    self.assertEqual(select(**(geometry | change)), 0)
        ns['_STATIC_V2_OVERRIDE'] = parse('t,r,sf6,batch,ss1')
        self.assertEqual(select(**geometry), 0)
        for token in ['as0', 'as3', 'as-1', 'ss1,as1', 'as1,ss2']:
            with self.subTest(token=token), self.assertRaises(ValueError):
                parse('t,r,sf6,batch,' + token)

    def test_all_quantizer_branches_keep_the_original_fast_and_exact_fallback(self):
        # A zero mode must keep both original arithmetic paths, including expert-specific scales.
        names = ['moe_static_kernel_v4.py', 'moe_static_kernel.py', 'moe_micro_kernel.py',
                 '_moe_dynamic/gated.py', '_moe_dynamic/generic.py',
                 'moe_dynamic_gated_sf6_q0.py', 'moe_dynamic_gated_sf6_prefill.py',
                 'moe_dynamic_prefill_packets.py']
        seen = 0
        for name in names:
            tree = ast.parse((ROOT/'engine/kernels/b12x'/name).read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.If) or ast.unparse(node.test) != 'cutlass.const_expr(self.activation_scale_search > 0)':
                    continue
                searched = node.body[0]
                original = node.orelse[0]
                self.assertEqual(ast.unparse(original.test), 'self.fast_math')
                fast, exact = original.body[0], original.orelse[0]
                self.assertEqual(ast.dump(searched.targets[0]), ast.dump(fast.targets[0]))
                self.assertEqual(ast.dump(fast.targets[0]), ast.dump(exact.targets[0]))
                self.assertEqual(ast.unparse(fast.value.func), 'quantize_block_fp4_fast')
                self.assertEqual(ast.unparse(exact.value.func), 'quantize_block_fp4')
                self.assertEqual([ast.unparse(a) for a in searched.value.args[:3]],
                                 [ast.unparse(a) for a in fast.value.args])
                self.assertEqual([ast.unparse(a) for a in fast.value.args],
                                 [ast.unparse(a) for a in exact.value.args])
                seen += 1
        self.assertEqual(seen, 23)


if __name__ == '__main__':
    unittest.main()
