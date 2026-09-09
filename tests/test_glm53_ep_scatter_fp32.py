"""CPU source/launch oracles for EP's BF16-contribution/FP32-sum variant."""
import ast
import copy
import gzip
import hashlib
import math
from pathlib import Path
import struct
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / 'overlay/modules/glm53_moe/moe_dispatch.py'
KERNEL = ROOT / 'overlay/modules/glm53_moe/moe_dynamic_ep_local.py'
CANARY = ROOT / 'overlay/modules/glm53_moe/glm53_ep_local_selftest.py'
STOCK = ROOT / 'measurements/glm53_ep_local_20260908/cpu13/stock-gated.py.gz'


def node(path, name):
    return next(n for n in ast.walk(ast.parse(path.read_text()))
                if isinstance(n, ast.FunctionDef) and n.name == name)


def extract(path, name, ns):
    fn = copy.deepcopy(node(path, name)); fn.decorator_list = []
    exec(compile(ast.Module(body=[ast.parse('from __future__ import annotations').body[0], fn],
                            type_ignores=[]), str(path), 'exec'), ns)
    return ns[name]


def asm_of(fn):
    call = next(n for n in ast.walk(fn) if isinstance(n, ast.Call)
                and ast.unparse(n.func) == 'llvm.inline_asm')
    return call.args[2].value, call


class SourceTests(unittest.TestCase):
    def test_stock_contribution_and_entire_scatter_addressing_are_preserved(self):
        raw = gzip.decompress(STOCK.read_bytes())
        self.assertEqual(hashlib.sha256(raw).hexdigest(),
                         '993783308233288ddfa77293e9dbabdc825ba5bfdcc4dcc41e842a895ec33445')
        tree = ast.parse(raw)
        stock = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                     and n.name == 'scatter_add_weighted_bf16x8_packed_alpha')
        old, _ = asm_of(stock)
        new, call = asm_of(node(KERNEL, 'scatter_add_weighted_bf16x8_to_f32'))
        start = ' ld.shared.v4.u32'
        stop = ' mul.rn.bf16x2 p3, p3, w2;'
        self.assertEqual(old[old.index(start):old.index(stop)+len(stop)],
                         new[new.index(start):new.index(stop)+len(stop)])
        self.assertEqual(new.count('cvt.f32.bf16'), 8)
        self.assertEqual(new.count('red.global.add.L2::cache_hint.v4.f32'), 2)
        self.assertIn('[$0], {f0,f1,f2,f3}', new)
        self.assertIn('add.u64 next, $0, 16;', new)
        self.assertIn('[next], {f4,f5,f6,f7}', new)
        self.assertTrue(next(k.value.value for k in call.keywords if k.arg == 'has_side_effects'))
        actual = copy.deepcopy(node(KERNEL, 'scatter_sC_to_gmem'))
        for n in ast.walk(actual):
            if isinstance(n, ast.Name) and n.id == 'scatter_add_weighted_bf16x8_to_f32':
                n.id = 'scatter_add_weighted_bf16x8_packed_alpha'
        expected = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                        and n.name == 'scatter_sC_to_gmem')
        self.assertEqual(ast.dump(actual), ast.dump(expected))

    def test_fp32_zero_covers_every_row_and_keeps_grid_barrier(self):
        fn = node(KERNEL, 'initialize_route_q0_and_publish')
        assignments = {n.targets[0].id: n.value for n in fn.body if isinstance(n, ast.Assign)
                       and isinstance(n.targets[0], ast.Name)}
        self.assertEqual(ast.unparse(assignments['cols_u32']), 'cols')
        self.assertEqual(ast.unparse(assignments['output_bytes_per_row']), 'cols // Int32(2)')
        start = next(i for i,n in enumerate(fn.body) if isinstance(n, ast.Assign)
                     and ast.unparse(n.targets[0]) == 'scatter_total_u32')
        end = next(i for i,n in enumerate(fn.body[start:], start) if isinstance(n, ast.If))
        code = compile(ast.Module(body=fn.body[start:end], type_ignores=[]), str(KERNEL), 'exec')
        for rows in (4096, 4097, 6912, 8192, 16384):
            # Execute actual address/length expressions at domain boundaries;
            # use one vector per synthetic thread to avoid T*K CPU allocation.
            vectors = []
            class Tail:
                def __setitem__(self, index, value): raise AssertionError('unexpected K4096 scalar tail')
            stride = rows*4096//4
            for tid in (0, 1, stride//2, stride-1):
                ns = dict(num_tokens=rows, cols_u32=4096, flat_tid=tid, flat_stride=stride,
                          Int32=int, Int64=int, Uint32=int, scatter_base=0,
                          scatter_output_u32=Tail(),
                          st_global_v4_u32=lambda addr,*zero: vectors.append(addr))
                exec(code, ns)
            self.assertEqual(vectors,[0,16,(stride//2)*16,rows*4096*4-16])
        bulk = next(n for n in fn.body if isinstance(n,ast.While)
                    and ast.unparse(n.test) == 'zv < scatter_vecs')
        self.assertEqual(ast.unparse(bulk.body[-1]), 'zv += flat_stride')
        text = ast.unparse(fn)
        self.assertLess(text.index('while zv < scatter_vecs'), text.index('self.resident_grid_barrier'))

    def test_ep_cache_and_fake_dtype_are_distinct_without_changing_stock_key(self):
        key = extract(MD, '_dynamic_kernel_cache_key', {})
        args = dict(activation_precision='fp4', quant_mode='nvfp4', E=72, k=4096, n=2048,
            num_topk=8, mac=48, mma_tiler_mn=(128,128), topk_ids_dtype='i32',
            input_scales_are_reciprocal=False, fast_math=True, activation='swigluoai_uninterleave',
            swiglu_alpha=1., swiglu_beta=0., swiglu_limit=10., share_input_across_experts=False)
        stock = key(**args); ep = key(**args, ep_local_prefill=True)
        self.assertEqual(ep, stock + ('glm53_ep_prefill_local_fp32_v2',))
        self.assertNotEqual(ep, stock + ('glm53_ep_prefill_local_v1',))
        fn = node(MD, '_get_dynamic_kernel')
        dtype = next(n.value for n in fn.body if isinstance(n, ast.Assign)
                     and ast.unparse(n.targets[0]) == 'scatter_dtype')
        for selected in (None, object()):
            self.assertEqual(eval(compile(ast.Expression(dtype), str(MD), 'eval'),
                dict(cutlass=SimpleNamespace(Float32='f32'), ep_local_cls=selected, a_dtype='bf16')),
                'f32' if selected is not None else 'bf16')
        entry = ast.unparse(node(KERNEL, '__call__'))
        self.assertIn('scatter_output.element_type != cutlass.Float32', entry)

    def test_onepass9_bf16_fixture_and_legacy_scratch_dtype_are_explicit(self):
        tree = ast.parse(CANARY.read_text())
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and ast.unparse(n.func) in ('torch.rand', 'torch.linspace'):
                self.assertEqual(ast.unparse(next(k.value for k in n.keywords if k.arg == 'dtype')), 'torch.bfloat16')
        ns = dict()
        scratch = extract(CANARY, '_scratch', ns)
        for dtype in ('f32','bf16','f16'):
            torch = SimpleNamespace(int32='i32', float32='f32', int64='i64', bool='bool',
                empty=lambda shape,**kw: (shape, kw['dtype'], kw['device']))
            obj = SimpleNamespace(); scratch(obj, torch, 6, 'cuda:0', weights_dtype=dtype)
            self.assertEqual(obj._ep_scales, ((6,8),dtype,'cuda:0'))
        prep = ast.unparse(node(CANARY, '_prepare_check'))
        self.assertLess(prep.index('candidate.dtype != reference.dtype'), prep.index('_tensor_bytes(candidate)'))
        case_node = node(CANARY, '_case'); case = ast.unparse(case_node)
        self.assertIn('for dtype in (torch.float32, torch.float16)', case)
        self.assertTrue(any(isinstance(n, ast.Try) and any(
            ast.unparse(stmt) == '_scratch(obj, torch, rows, device, weights_dtype=weights.dtype)'
            for stmt in n.finalbody) for n in ast.walk(case_node)))


class Tensor:
    registry = {}
    def __init__(self, shape=(1,), dtype='bf16', device='cuda:0', *, events=None, parent=None):
        self.shape, self.dtype, self.device, self.ndim = shape,dtype,device,len(shape)
        self.events = events if events is not None else []
        self.parent = parent or self; self.contiguous = True
        self.registry[id(self.parent)] = self.parent
    def data_ptr(self): return id(self.parent)
    def is_contiguous(self): return self.contiguous
    def numel(self): return math.prod(self.shape)
    def view(self,*a): return self
    def to(self,*a): return self
    def record_stream(self, stream): self.events.append(('record',stream,self.data_ptr()))
    def __getitem__(self, s): return Tensor((s.stop,self.shape[1]),self.dtype,self.device,events=self.events,parent=self.parent)
    def copy_(self, other): self.events.append(('copy',other.data_ptr())); return self


class RuntimeTests(unittest.TestCase):
    def namespace(self):
        events, allocations = [], []
        def empty(shape, **kw):
            result = Tensor(shape,events=events,**kw); allocations.append(result); return result
        torch = SimpleNamespace(bfloat16='bf16',float32='f32',int32='i32',empty=empty,
                                cuda=SimpleNamespace(current_stream=lambda dev:'side-stream'))
        ns = dict(torch=torch)
        extract(MD, '_ep_local_scatter_buffer', ns)
        return ns,events,allocations

    def test_shared_buffer_reuses_grows_and_records_nondefault_stream(self):
        ns,events,allocations = self.namespace(); fn=ns['_ep_local_scatter_buffer']
        ws=SimpleNamespace(device='cuda:0',ep_scatter_fp32=None)
        first=fn(ws,Tensor((8192,4096)),8192,4096)
        short=fn(ws,Tensor((4096,4096)),4096,4096)
        self.assertEqual(first.data_ptr(),short.data_ptr())
        self.assertEqual(len(allocations),1)
        bigger=fn(ws,Tensor((16384,4096)),16384,4096)
        self.assertNotEqual(first.data_ptr(),bigger.data_ptr())
        self.assertEqual(len(allocations),2)
        self.assertEqual(len(events),3)
        self.assertTrue(all(item[1]=='side-stream' for item in events))
        self.assertEqual(8192*4096*4,128*1024*1024)

    def test_incompatible_output_or_cached_storage_is_rejected(self):
        ns,_,allocations=self.namespace(); fn=ns['_ep_local_scatter_buffer']
        for out in (Tensor((8192,4096),'f32'),Tensor((8192,2048)),Tensor((8192,4096),device='cuda:1')):
            with self.assertRaises(ValueError):fn(SimpleNamespace(device='cuda:0',ep_scatter_fp32=None),out,8192,4096)
        ws=SimpleNamespace(device='cuda:0',ep_scatter_fp32=Tensor((8192,4096),'bf16'))
        with self.assertRaises(ValueError):fn(ws,Tensor((8192,4096)),8192,4096)
        self.assertEqual(allocations,[])

    def launch(self, *, ep, failure=False):
        ns,events,allocations=self.namespace()
        def compiled(*args):
            events.append(('launch',args[25]))
            if failure: raise RuntimeError('kernel failed')
        ns.update(_normalize_activation_precision=lambda x:x,_check_memref_limit=lambda *a:None,
            _normalize_quant_mode=lambda q,a:q,_expand_to_experts=lambda x,n:x,
            _scale_runtime_addresses=lambda *a,**kw:(11,12),
            _ep_local_prefill_kernel=lambda **kw:object() if ep else None,
            _get_dynamic_kernel=lambda *a,**kw:(compiled,48),_sf_pack_dummy=lambda device:Tensor())
        fn=extract(MD,'launch_sm120_dynamic_moe',ns)
        ws=SimpleNamespace(device='cuda:0',ep_scatter_fp32=None,tile_m=128,max_rows=8192,
            physical_tiles_capacity=64,task_capacity=256)
        for key in ('packed_a_view','packed_input_scale','packed_a_flat','scale_flat','barrier_count',
            'barrier_epoch','pair_head','task_head','task_tail','task_expert','task_valid_rows',
            'row_counts','expert_write_rows','expert_tile_base','token_map','token_weights'):
            setattr(ws,key,Tensor())
        weights=SimpleNamespace(reform_scales=None,w13_fp4=Tensor(),down_fp4=Tensor(),w1_alpha=Tensor(),w2_alpha=Tensor())
        out=Tensor((8192,4096),events=events)
        args=dict(workspace=ws,weights=weights,a=Tensor((8192,4096)),topk_ids=Tensor((8192,8),'i32'),
            topk_weights=Tensor((8192,8),'f32'),input_gs=Tensor((72,)),down_input_scale=Tensor((72,)),
            scatter_output=out,num_experts=72,num_tokens=8192,k=4096,n=2048,top_k=8)
        if failure:
            with self.assertRaisesRegex(RuntimeError,'kernel failed'):fn(**args)
        else:self.assertIs(fn(**args),out)
        return events,allocations,ws,out

    def test_actual_launcher_uses_fp32_then_copy_without_sync_and_stock_is_unchanged(self):
        events,allocations,ws,out=self.launch(ep=True)
        self.assertEqual([e[0] for e in events],['record','launch','copy'])
        self.assertEqual(events[1][1],ws.ep_scatter_fp32.data_ptr())
        self.assertNotEqual(events[1][1],out.data_ptr())
        self.assertEqual(events[2][1],ws.ep_scatter_fp32.data_ptr())
        events,allocations,ws,out=self.launch(ep=False)
        self.assertEqual(events,[('launch',out.data_ptr())]);self.assertEqual(allocations,[])

    def test_kernel_failure_cannot_copy_stale_or_claim_output(self):
        events,_,_,_=self.launch(ep=True,failure=True)
        self.assertEqual([e[0] for e in events],['record','launch'])


if __name__ == '__main__':unittest.main()
