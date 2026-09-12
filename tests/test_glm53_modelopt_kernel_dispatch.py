"""Native ModelOpt dense and short-prefill accumulation contracts, without CUDA."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import test_glm53_ep_scatter_fp32 as ep

DISPATCH = Path(__file__).resolve().parents[1] / 'engine/kernels/b12x/moe_dispatch.py'


def functions(*names):
    tree = ast.parse(DISPATCH.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(DISPATCH), 'exec'), namespace)
    return namespace


class ModelOptKernelDispatchTests(unittest.TestCase):
    def test_dense_and_routed_glm_partial_sums_have_fp32_storage(self):
        ns = functions('_glm_tp_scatter_shape', '_glm_tp_scatter_fp32')
        gate = ns['_glm_tp_scatter_fp32']
        common = dict(k=4096, quant_mode='nvfp4', activation='swigluoai_uninterleave',
                      swiglu_alpha=1., swiglu_beta=0., swiglu_limit=10.)
        for experts, width, topk in ((1,3072,1),(288,512,8)):
            args = dict(common, state_E=experts, weight_E=experts, n=width, num_topk=topk)
            self.assertTrue(gate(**args))
            self.assertFalse(gate(**dict(args, n=width+128)))
            self.assertFalse(gate(**dict(args, swiglu_limit=9.)))

    def test_short_prefill_uses_the_same_q0_fp32_abi_as_large_prefill(self):
        gate = functions('_tp_sf6_q0_eligible')['_tp_sf6_q0_eligible']
        args = dict(enabled=True, E=288, k=4096, n=512, num_topk=8, tile_m=128,
                    quant_mode='nvfp4', tiled=True, reform_sf_pack=True,
                    activation='swigluoai_uninterleave', swiglu_alpha=1., swiglu_beta=0.,
                    swiglu_limit=10., share_input_across_experts=False)
        for rows in (1,17,64,129,4095,4096,8192):
            self.assertTrue(gate(**args,m=rows))
        for rows in (0,-1,8193,True,8192.):
            self.assertFalse(gate(**args,m=rows))
        self.assertFalse(gate(**dict(args,enabled=False),m=129))
        # Scale compressibility cannot change the FP32 accumulation ABI.
        self.assertTrue(gate(**dict(args,reform_sf_pack=False),m=129))

    def test_short_tp_scatter_reuses_and_grows_storage_without_loosening_ep(self):
        with patch.object(ep,'MD',DISPATCH):
            ns,events,allocations = ep.RuntimeTests().namespace()
        fn=ns['_ep_local_scatter_buffer']
        ws=SimpleNamespace(device='cuda:0',ep_scatter_fp32=None)
        with self.assertRaises(ValueError):
            fn(ws,ep.Tensor((129,4096)),129,4096)
        first=fn(ws,ep.Tensor((129,4096)),129,4096,tp=True)
        smaller=fn(ws,ep.Tensor((17,4096)),17,4096,tp=True)
        self.assertEqual(first.data_ptr(),smaller.data_ptr())
        self.assertEqual(len(allocations),1)
        bigger=fn(ws,ep.Tensor((4096,4096)),4096,4096,tp=True)
        self.assertNotEqual(first.data_ptr(),bigger.data_ptr())
        self.assertEqual(len(allocations),2)


if __name__ == '__main__':
    unittest.main()
