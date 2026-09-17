"""Native ModelOpt dense and short-prefill accumulation contracts, without CUDA."""
import ast
import copy
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from engine.base.kernel_shape import MEASURED

DISPATCH = Path(__file__).resolve().parents[1] / 'engine/kernels/b12x/moe_dispatch.py'
CELL = MEASURED.moe     # the admitted MoE cell the gates compare against (engine/base/kernel_shape)


class Tensor:
    """The CPU torch stub the EP scatter oracle runs against (no CUDA here)."""
    def __init__(self, shape=(1,), dtype='bf16', device='cuda:0', *, events=None, parent=None):
        self.shape, self.dtype, self.device, self.ndim = shape,dtype,device,len(shape)
        self.events = events if events is not None else []
        self.parent = parent or self; self.contiguous = True
    def data_ptr(self): return id(self.parent)
    def is_contiguous(self): return self.contiguous
    def numel(self): return math.prod(self.shape)
    def view(self,*a): return self
    def to(self,*a): return self
    def record_stream(self, stream): self.events.append(('record',stream,self.data_ptr()))
    def __getitem__(self, s): return Tensor((s.stop,self.shape[1]),self.dtype,self.device,events=self.events,parent=self.parent)
    def copy_(self, other): self.events.append(('copy',other.data_ptr())); return self


def ep_scatter_oracle():
    """Extract the engine's own _ep_local_scatter_buffer under the stub torch."""
    fn = copy.deepcopy(next(n for n in ast.walk(ast.parse(DISPATCH.read_text()))
                            if isinstance(n, ast.FunctionDef) and n.name == '_ep_local_scatter_buffer'))
    fn.decorator_list = []
    events, allocations = [], []
    def empty(shape, **kw):
        result = Tensor(shape,events=events,**kw); allocations.append(result); return result
    ns = dict(torch=SimpleNamespace(bfloat16='bf16',float32='f32',int32='i32',empty=empty,
                                    cuda=SimpleNamespace(current_stream=lambda dev:'side-stream')))
    exec(compile(ast.Module(body=[ast.parse('from __future__ import annotations').body[0], fn],
                            type_ignores=[]), str(DISPATCH), 'exec'), ns)
    return ns, events, allocations


def functions(*names):
    tree = ast.parse(DISPATCH.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(DISPATCH), 'exec'), namespace)
    return namespace


class ModelOptKernelDispatchTests(unittest.TestCase):
    def test_w4a16_dispatch_preserves_glm_swiglu_parameters(self):
        tree = ast.parse(DISPATCH.read_text())
        fn = next(node for node in tree.body
                  if isinstance(node, ast.FunctionDef)
                  and node.name == '_launch_sm120_w4a16_moe')
        calls = [node for node in ast.walk(fn)
                 if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name)
                 and node.func.id == 'run_w4a16_moe']
        self.assertEqual(len(calls), 1)
        forwarded = {kw.arg for kw in calls[0].keywords}
        self.assertTrue({'swiglu_limit', 'swiglu_alpha', 'swiglu_beta'} <= forwarded)

    def test_dense_and_routed_glm_partial_sums_have_fp32_storage(self):
        ns = functions('_glm_tp_scatter_shape', '_glm_tp_scatter_fp32')
        gate = ns['_glm_tp_scatter_fp32']
        common = dict(k=4096, quant_mode='nvfp4', activation='swigluoai_uninterleave',
                      swiglu_alpha=1., swiglu_beta=0., swiglu_limit=10., cell=CELL)
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
                    swiglu_limit=10., share_input_across_experts=False, cell=CELL)
        for rows in (1,17,64,129,4095,4096,8192):
            self.assertTrue(gate(**args,m=rows))
        for rows in (0,-1,8193,True,8192.):
            self.assertFalse(gate(**args,m=rows))
        self.assertFalse(gate(**dict(args,enabled=False),m=129))
        # Scale compressibility cannot change the FP32 accumulation ABI.
        self.assertTrue(gate(**dict(args,reform_sf_pack=False),m=129))

    def test_short_tp_scatter_reuses_and_grows_storage_without_loosening_ep(self):
        ns,events,allocations = ep_scatter_oracle()
        fn=ns['_ep_local_scatter_buffer']
        ws=SimpleNamespace(device='cuda:0',ep_scatter_fp32=None)
        with self.assertRaises(ValueError):
            fn(ws,Tensor((129,4096)),129,4096)
        first=fn(ws,Tensor((129,4096)),129,4096,tp=True)
        smaller=fn(ws,Tensor((17,4096)),17,4096,tp=True)
        self.assertEqual(first.data_ptr(),smaller.data_ptr())
        self.assertEqual(len(allocations),1)
        bigger=fn(ws,Tensor((4096,4096)),4096,4096,tp=True)
        self.assertNotEqual(first.data_ptr(),bigger.data_ptr())
        self.assertEqual(len(allocations),2)


if __name__ == '__main__':
    unittest.main()
