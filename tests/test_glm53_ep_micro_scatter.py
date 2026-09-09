"""Actual-source CPU oracles for the two pinned EP micro FP32 sum planes."""
import ast
import copy
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'overlay/modules/glm53_moe/moe_dispatch.py'
EXACT = dict(state_E=72, weight_E=72, m=8, k=4096, n=2048, num_topk=8,
             max_rows=64, skip_zero_weight_expert_id=72, quant_mode='nvfp4',
             activation='swigluoai_uninterleave', swiglu_alpha=1.,
             swiglu_beta=0., swiglu_limit=10.)


def load(name, ns):
    fn = copy.deepcopy(next(n for n in ast.parse(SOURCE.read_text()).body
                            if isinstance(n, ast.FunctionDef) and n.name == name))
    module = ast.Module(body=[ast.parse('from __future__ import annotations').body[0], fn],
                        type_ignores=[])
    exec(compile(module, str(SOURCE), 'exec'), ns)
    return ns[name]


class Tensor:
    def __init__(self, shape, dtype='bf16', device='cuda:0', *, events=None):
        self.shape, self.dtype, self.device = tuple(shape), dtype, device
        self.events = events if events is not None else []
        self.contiguous = True
    def numel(self): return math.prod(self.shape)
    def is_contiguous(self): return self.contiguous
    def data_ptr(self): return id(self)
    def view(self, *args): return self
    def permute(self, *args): return self
    def to(self, *args): return self
    def __getitem__(self, index): return self
    def record_stream(self, stream): self.events.append(('record', stream, id(self)))
    def copy_(self, other): self.events.append(('copy', id(other))); return self


def fake_torch(events, allocations):
    def allocate(*shape, **kw):
        if len(shape) == 1 and isinstance(shape[0], tuple): shape = shape[0]
        result = Tensor(shape, events=events, **kw); allocations.append(result); return result
    return SimpleNamespace(float32='f32', bfloat16='bf16', int32='i32', uint8='u8',
        float4_e2m1fn_x2='fp4x2', empty=allocate, zeros=allocate,
        arange=lambda n, **kw: allocate(n, **kw), device=lambda d:d,
        cuda=SimpleNamespace(current_stream=lambda d:'current-stream'))


class AdmissionTests(unittest.TestCase):
    def test_only_fixed_top1_and_sentinel_top8_are_selected(self):
        gate = load('_ep_micro_scatter_fp32', {})
        self.assertTrue(gate(**EXACT))
        fixed = EXACT | dict(num_topk=1, max_rows=8, skip_zero_weight_expert_id=None)
        self.assertTrue(gate(**fixed))
        changes = dict(state_E=(71, 288), weight_E=(71, 288), m=(1, 6, 7, 9),
            k=(2048, 4095), n=(1024, 2047), quant_mode=('mxfp4',),
            activation=('silu', 'relu2'), swiglu_alpha=(1.1,), swiglu_beta=(1.,),
            swiglu_limit=(None, 9.), num_topk=(2, 7), max_rows=(8, 63, 128),
            skip_zero_weight_expert_id=(None, 71, 73))
        for field, values in changes.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    self.assertFalse(gate(**(EXACT | {field:value})))
        for change in (dict(max_rows=64), dict(skip_zero_weight_expert_id=72)):
            self.assertFalse(gate(**(fixed | change)))

    def test_compile_constructor_output_dtype_and_cache_namespace_agree(self):
        # Reuse the existing fake compiler, executing the actual constructor
        # and get-kernel source. No CuTe/Torch import or device compilation.
        from test_glm53_ep_micro_tile import Harness
        h = Harness(); load('_ep_micro_scatter_fp32', h.ns)
        h.get(); kernel, args, _ = h.compiles[-1]
        self.assertTrue(kernel.scatter_fp32); self.assertEqual(args[21].dtype, 'Float32')
        self.assertTrue(kernel.ep_direct_scatter)
        key = next(reversed(h.ns['_MICRO_KERNEL_CACHE']))
        self.assertEqual(key[17], 72)
        self.assertEqual(key[22:], ('glm53_ep_micro_scatter_fp32_v1',
                                    'glm53_ep_micro_direct_scatter_v1'))
        h.get(num_topk=1, max_rows=8, skip_zero_weight_expert_id=None)
        kernel, args, _ = h.compiles[-1]
        self.assertTrue(kernel.scatter_fp32); self.assertEqual(args[21].dtype, 'Float32')
        self.assertFalse(kernel.ep_direct_scatter)
        fixed_key = next(reversed(h.ns['_MICRO_KERNEL_CACHE']))
        self.assertIsNone(fixed_key[17])
        self.assertEqual(fixed_key[22:], ('glm53_ep_micro_scatter_fp32_v1',))
        h.get(skip_zero_weight_expert_id=None)
        kernel, args, _ = h.compiles[-1]
        self.assertFalse(kernel.scatter_fp32); self.assertEqual(args[21].dtype, 'BFloat16')
        self.assertFalse(kernel.ep_direct_scatter)
        self.assertEqual(len(next(reversed(h.ns['_MICRO_KERNEL_CACHE']))), 22)
        count = len(h.compiles); h.get()
        self.assertEqual(len(h.compiles), count)

    def test_direct_scatter_rejects_every_geometry_and_execution_mode_mismatch(self):
        ns = {}; load('_ep_micro_scatter_fp32', ns)
        gate = load('_ep_micro_direct_scatter', ns)
        exact = EXACT | dict(mma_tiler_mn=(32,128), share_input_across_experts=False,
                             share_expert_scales=False, single_token=False)
        self.assertTrue(gate(**exact))
        changes = dict(state_E=(71,73,288), weight_E=(71,73,288), m=(1,6,7,9),
            k=(2048,4095,4097), n=(1024,2047,2049), num_topk=(1,7,9),
            max_rows=(8,63,65,128), skip_zero_weight_expert_id=(None,-1,71,73),
            quant_mode=('mxfp4',), activation=('silu','relu2'),
            swiglu_alpha=(1.702,float('nan')), swiglu_beta=(1.,float('nan')),
            swiglu_limit=(None,9.,11.,float('nan')), mma_tiler_mn=((64,128),(32,256)),
            share_input_across_experts=(True,), share_expert_scales=(True,),
            single_token=(True,))
        for field, values in changes.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    self.assertFalse(gate(**(exact | {field:value})))
        # The independent six-call top1 control still uses FP32 but must keep
        # its buffered scatter and its existing kernel/cache identity.
        fixed = exact | dict(num_topk=1,max_rows=8,skip_zero_weight_expert_id=None,
                             mma_tiler_mn=(64,128))
        self.assertFalse(gate(**fixed))

    def test_buffered_fp32_cache_cannot_satisfy_direct_scatter(self):
        from test_glm53_ep_micro_tile import Harness
        h = Harness(); h.get()
        direct_key = next(iter(h.ns['_MICRO_KERNEL_CACHE']))
        # Explicit old ABI/order oracle, including the sentinel at index17.
        buffered_key = ('micro','nvfp4',72,72,8,4096,2048,8,64,48,(32,128),
                        'int32',False,True,False,False,False,72,
                        'swigluoai_uninterleave',1.,0.,10.,
                        'glm53_ep_micro_scatter_fp32_v1')
        self.assertEqual(direct_key[:-1], buffered_key)
        stale = (object(),48)
        h.ns['_MICRO_KERNEL_CACHE'] = {buffered_key:stale}
        result = h.get()
        self.assertNotEqual(result,stale)
        self.assertEqual(len(h.compiles),2)
        self.assertEqual(set(h.ns['_MICRO_KERNEL_CACHE']),{buffered_key,direct_key})
        self.assertEqual(h.get(),result)
        self.assertEqual(len(h.compiles),2)

    def test_compile_execution_modes_keep_existing_rejection_or_buffered_fp32(self):
        from test_glm53_ep_micro_tile import Harness
        for flag in ('share_input_across_experts','share_expert_scales','single_token',None):
            with self.subTest(flag=flag):
                h = Harness()
                if flag is None:
                    h.ns['_select_micro_mma_tiler_mn'] = lambda **kw:(64,128)
                if flag in ('share_input_across_experts','single_token'):
                    with self.assertRaisesRegex(ValueError,'zero-weight expert skip'):
                        h.get(**{flag:True})
                    self.assertEqual(h.compiles,[])
                    self.assertEqual(h.ns['_MICRO_KERNEL_CACHE'],{})
                    continue
                h.get(**({flag:True} if flag else {}))
                kernel,args,_ = h.compiles[-1]
                self.assertFalse(kernel.ep_direct_scatter)
                self.assertTrue(kernel.scatter_fp32)
                self.assertEqual((args[21].dtype,args[21].shape),('Float32',(8,4096)))
                self.assertEqual(next(iter(h.ns['_MICRO_KERNEL_CACHE']))[22:],
                                 ('glm53_ep_micro_scatter_fp32_v1',))

    def test_actual_allocator_pins_only_matching_capacity_before_any_launch(self):
        events, allocations = [], []
        torch = fake_torch(events, allocations)
        ns = dict(torch=torch, _normalize_activation_precision=lambda x:x,
            _normalize_quant_mode=lambda *x:x[0], _sf_params_for_quant_mode=lambda x:(16,'sf'),
            _align_up=lambda x,y:(x+y-1)//y*y, _check_memref_limit=lambda *a:None,
            Sm120StaticMoEWorkspace=lambda **kw:SimpleNamespace(ep_micro_scatter_fp32=None, **kw),
            make_ptr=lambda *a,**kw:None, cute=SimpleNamespace(AddressSpace=SimpleNamespace(gmem=0)),
            _direct_micro_candidate=lambda *a:False)
        allocate = load('allocate_sm120_static_workspace', ns)
        base = dict(state_E=72, weight_E=72, max_rows=8, k=4096, n=2048,
                    num_topk=1, device='cuda:0')
        for change, selected in (({},True),(dict(num_topk=8,max_rows=64),True),
                (dict(state_E=288,weight_E=288),False),(dict(max_rows=128),False),
                (dict(num_topk=8,max_rows=40),False),(dict(n=1024),False)):
            ws = allocate(**(base | change)); buf = ws.ep_micro_scatter_fp32
            self.assertEqual(buf is not None, selected)
            if selected:
                self.assertEqual((buf.shape,buf.dtype), ((8,4096),'f32'))
                self.assertEqual(buf.numel()*4, 128*1024)
        self.assertEqual(events, [])


class LaunchTests(unittest.TestCase):
    def harness(self, *, candidate=True, experts=72, failure=False):
        events, allocations = [], []
        torch = fake_torch(events, allocations)
        ws = SimpleNamespace(state_E=experts, max_rows=64 if candidate else 8,
                             device='cuda:0', dm_barrier_count=None)
        for name in ('compact_topk_ids','packed_a_view','packed_input_scale','packed_a_flat',
                     'scale_flat','barrier_count','barrier_epoch','row_counts','active_expert_count',
                     'weight_expert_ids','global_to_local_expert','token_map','token_weights'):
            setattr(ws,name,Tensor((72,)))
        ws.ep_micro_scatter_fp32 = Tensor((8,4096),'f32',events=events)
        out = Tensor((8,4096),events=events)
        def compiled(*args):
            events.append(('launch', id(args[21]), args[21].dtype))
            if failure: raise RuntimeError('micro launch failed')
        ns = dict(__package__='ep_micro_test', torch=torch,
            _normalize_activation_precision=lambda x:x, _normalize_quant_mode=lambda *x:x[0],
            _check_memref_limit=lambda *a:None, _expand_to_experts=lambda x,n:x,
            _FORCED_BACKEND=None, _GLM53_B12X_FORCE_BACKEND=None,
            _MICRO_SHARE_INPUT_ACROSS_EXPERTS=False, _MICRO_MAX_TOKENS=8,
            _DIRECT_MICRO_CUTOVER_PAIRS=0, _DIRECT_MICRO_MAX_N=1024,
            _MICRO_COMPACT_CUTOVER_PAIRS=40, _MICRO_COMPACT_CUTOVER_PAIRS_MULTI_TOPK=40,
            _B12X_EP_ZERO_WEIGHT_MICRO=candidate,
            _b12x_ep_zero_weight_micro_expert_id=lambda **kw:72 if candidate else None,
            get_num_sm=lambda _:48, get_max_active_clusters=lambda _:48,
            _STATIC_MAC_LADDER=(), _GLM53_B12X_STATIC_MAC_LADDER=None,
            _MICRO_MAC_LADDER=(), _GLM53_B12X_MICRO_MAC_LADDER=None,
            _lookup_mac_ladder=lambda *a:None, _scale_runtime_addresses=lambda *a,**kw:(1,2),
            _get_micro_kernel=lambda *a,**kw:(compiled,48))
        load('_ep_micro_scatter_fp32',ns);load('_ep_micro_scatter_buffer',ns)
        launch = load('launch_sm120_static_moe',ns)
        weights=SimpleNamespace(tiled=False,w13_fp4=Tensor((1,)),down_fp4=Tensor((1,)),
                                w1_alpha=Tensor((72,),'f32'),w2_alpha=Tensor((72,),'f32'))
        args=dict(workspace=ws,weights=weights,a=Tensor((8,4096)),topk_ids=Tensor((8,8 if candidate else 1),'i32'),
            topk_weights=Tensor((8,8 if candidate else 1),'bf16'),input_gs=Tensor((72,)),
            down_input_scale=Tensor((72,)),scatter_output=out,num_experts=experts,
            num_tokens=8,k=4096,n=2048,top_k=8 if candidate else 1,
            activation='swigluoai_uninterleave',swiglu_alpha=1.,swiglu_beta=0.,swiglu_limit=10.)
        module=SimpleNamespace(compact_topk_ids=lambda *a:events.append(('compact',)))
        def run():
            with patch.dict(sys.modules,{'ep_micro_test.triton_compact':module}):
                return launch(**args)
        return run,events,allocations,ws,out,torch

    def test_actual_selected_launches_copy_after_kernel_with_same_pinned_pointer(self):
        for candidate in (False,True):
            run,events,allocations,ws,out,torch=self.harness(candidate=candidate)
            pinned=ws.ep_micro_scatter_fp32
            for stream in ('default','side','capture-replay'):
                torch.cuda.current_stream=lambda d,s=stream:s
                self.assertIs(run(),out)
                self.assertIs(ws.ep_micro_scatter_fp32,pinned)
                self.assertEqual(events[-4:],[('record',stream,id(pinned)),('compact',),
                    ('launch',id(pinned),'f32'),('copy',id(pinned))])
            self.assertEqual(allocations,[])

    def test_tp_micro_preserves_output_abi_without_new_copy(self):
        run,events,allocations,ws,out,_=self.harness(candidate=False,experts=288)
        self.assertIs(run(),out)
        self.assertEqual(events,[('compact',),('launch',id(out),'bf16')])
        self.assertEqual(allocations,[])

    def test_missing_or_wrong_buffer_refuses_before_launch_and_never_reallocates(self):
        for invalid in (None,Tensor((8,4096),'bf16'),Tensor((16,4096),'f32'),
                        Tensor((8,4096),'f32',device='cuda:1')):
            run,events,allocations,ws,_,_=self.harness();ws.ep_micro_scatter_fp32=invalid
            with self.assertRaises(ValueError):run()
            self.assertEqual(events,[]);self.assertEqual(allocations,[])

    def test_output_contract_and_launch_failure_do_not_publish_stale_output(self):
        for dtype,shape in (('f32',(8,4096)),('bf16',(6,4096))):
            run,events,_,_,out,_=self.harness();out.dtype=dtype;out.shape=shape
            with self.assertRaises(ValueError):run()
            self.assertEqual(events,[])
        run,events,_,_,_,_=self.harness(failure=True)
        with self.assertRaisesRegex(RuntimeError,'micro launch failed'):run()
        self.assertEqual([e[0] for e in events],['record','compact','launch'])


if __name__ == '__main__':unittest.main()
