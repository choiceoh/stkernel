"""CPU-only contracts for the actual EP tile-major source and launch ABI.

No accelerator modules are imported. CuTe lowering, graph replay and actual
numerics remain separate fleet/startup gates; mocks do not establish them.
"""
import ast
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'overlay/modules/glm53_moe/moe_static_ep_tiled.py'
STOCK = SOURCE.with_name('moe_static_kernel_v4.py')
V5 = SOURCE.with_name('moe_static_kernel_v5.py')
ORACLE = ROOT/'measurements/glm53_ep_local_20260908/micro-stock-oracle'


def function(name, path=SOURCE):
    return copy.deepcopy(next(n for n in ast.walk(ast.parse(path.read_text()))
                              if isinstance(n, ast.FunctionDef) and n.name == name))


def extract(name, ns):
    f = function(name); f.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom('__future__', [ast.alias('annotations')], 0), f],
                        type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), 'exec'), ns)
    return ns[name]


def constants():
    return {n.targets[0].id: ast.literal_eval(n.value)
            for n in ast.parse(SOURCE.read_text()).body
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
            and isinstance(n.value, ast.Constant)}


class Tensor:
    next_pointer = 4096
    def __init__(self, shape, dtype, *, device=None, strides=None, events=None):
        self.shape = tuple(shape); self.dtype = dtype; self.element_type = dtype
        self.device = device or types.SimpleNamespace(type='cuda')
        self._strides = strides; self.events = events if events is not None else []
        self.pointer = Tensor.next_pointer; Tensor.next_pointer += 4096
    def numel(self): return math.prod(self.shape)
    def element_size(self): return 1
    def is_contiguous(self): return self._strides is None
    def stride(self): return self._strides
    def data_ptr(self): return self.pointer
    def view(self, *shape):
        out = Tensor((self.numel(),) if shape == (-1,) else shape, self.dtype,
                     device=self.device, events=self.events)
        out.pointer = self.pointer; return out
    def __getitem__(self, item):
        assert isinstance(item, slice) and item.start is None and item.step is None
        out = Tensor((item.stop, *self.shape[1:]), self.dtype,
                     device=self.device, events=self.events)
        out.pointer = self.pointer; return out
    def record_stream(self, stream): self.events.append(('record', self.pointer, stream))
    def copy_(self, source): self.events.append(('copy', source.pointer, source.dtype)); return self


def fake_torch(events=None):
    t = types.ModuleType('torch')
    for k in ('int32','int64','float32','bfloat16','uint8','float4_e2m1fn_x2'):
        setattr(t,k,k)
    t.cuda = types.SimpleNamespace(is_current_stream_capturing=lambda:False,
                                  current_stream=lambda device:'current-stream')
    t.empty = lambda shape,device,dtype:Tensor(shape,dtype,device=device,events=events)
    t.zeros = t.empty
    return t


class EPTiledStaticTests(unittest.TestCase):
    def test_stock_source_pin_and_only_three_kernel_transformations(self):
        c=constants()
        self.assertEqual(hashlib.sha256(STOCK.read_bytes()).hexdigest(), c['STOCK_V4_SHA256'])
        self.assertEqual(hashlib.sha256(V5.read_bytes()).hexdigest(), c['STOCK_V5_SHA256'])
        actual = function('kernel'); expected=function('kernel',STOCK)
        routes=next(n for n in ast.walk(actual) if isinstance(n,ast.While)
                    and ast.unparse(n.test)=='pair_idx < total_pairs')
        guard=routes.body[1]
        self.assertEqual(ast.unparse(guard.test), 'expert_id >= Int32(0) and expert_id < num_experts')
        self.assertEqual(ast.unparse(guard.body[1].test), 'weight != cutlass.Float32(0.0)')
        restored=guard.body[1].body
        # Restore precisely the original load position; no other route change.
        self.assertEqual(ast.unparse(restored[0]), 'token_idx = pair_idx // num_topk')
        restored.insert(1,guard.body[0])
        routes.body=routes.body[:1]+restored+routes.body[2:]
        class RestoreABI(ast.NodeTransformer):
            def visit_Name(self,n):
                if n.id=='scatter_add_v4_bf16x2_to_f32': n.id='scatter_add_v4_bf16x2'
                return n
            def visit_Assign(self,n):
                if ast.unparse(n.targets[0])=='scatter_output[j // cols, j % cols]':
                    self.assert_zero = ast.unparse(n.value)
                    if self.assert_zero!='cutlass.Float32(0.0)': raise AssertionError(self.assert_zero)
                    n.value=ast.parse('cutlass.BFloat16(0.0)',mode='eval').body
                return self.generic_visit(n)
        actual=RestoreABI().visit(actual)
        self.assertEqual(ast.dump(actual,include_attributes=False),ast.dump(expected,include_attributes=False))

    def test_actual_route_guard_ignores_poison_and_preserves_nan_signed_zero_duplicates(self):
        route=next(n for n in ast.walk(function('kernel')) if isinstance(n,ast.While)
                   and ast.unparse(n.test)=='pair_idx < total_pairs')
        guard=copy.deepcopy(route.body[1]); guard.body[1].body=[ast.Return(ast.Tuple(
            elts=[ast.Name('expert_id',ast.Load()),ast.Name('weight',ast.Load())],ctx=ast.Load()))]
        fn=ast.FunctionDef(name='select',args=ast.arguments(posonlyargs=[],args=[ast.arg(x) for x in
            ('expert_id','topk_weights','pair_idx','num_experts')],kwonlyargs=[],kw_defaults=[],defaults=[]),
            body=[guard,ast.Return(ast.Constant(None))],decorator_list=[])
        ns={'Int32':int,'cutlass':types.SimpleNamespace(Float32=float)}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(SOURCE),'exec'),ns)
        class Value(float):
            def to(self,dtype): return dtype(self)
        class Poison:
            def __getitem__(self,i): raise AssertionError('invalid route read a weight')
        select=ns['select']
        for e in (-2**31,-1,72,73,287,2**31-1):self.assertIsNone(select(e,Poison(),0,72))
        for e in range(72):
            for z in (0.,-0.):self.assertIsNone(select(e,[Value(z)],0,72))
            self.assertEqual(select(e,[Value(.75)],0,72),(e,.75))
            self.assertTrue(math.isnan(select(e,[Value(float('nan'))],0,72)[1]))
        for m in range(1,33):
            selected=[select(71,[Value(1.)],0,72) for _ in range(m*8)]
            self.assertEqual(len(selected),m*8)
            self.assertLessEqual(len(selected),256)
        # The final per-route barrier executes even for invalid/zero routes.
        self.assertEqual(ast.unparse(route.body[-2]),'cute.arch.sync_threads()')
        self.assertEqual(ast.unparse(route.body[-1]),'pair_idx += Int32(gdim_z)')

    def test_geometry_full_native_band_and_capacity_bounds(self):
        g=extract('ep_tiled_geometry',{})
        for m in range(1,33):
            c=g(m,256,48)
            self.assertEqual(c['fc1'],(16,128,256) if m<=8 else (32,64,512))
            self.assertEqual(c['fc2'],(16,256,128) if m<=8 else (32,128,128))
            self.assertEqual(c['fc2'][2],128)
        for args in [(0,256,48),(33,256,48),(32,128,48),(6,64,48),(6,384,48),(6,256,49),(6,256,0)]:
            with self.assertRaises(ValueError):g(*args)
        for args in [(True,256,48),(6,256.,48),(6,256,True)]:
            with self.assertRaises(TypeError):g(*args)

    def test_exact_stock_bf16_conversion_then_widened_float_scatter(self):
        raw=gzip.decompress((ORACLE/'fp4_common.py.gz').read_bytes())
        ident=json.loads((ORACLE/'identity.json').read_text())
        self.assertEqual(hashlib.sha256(raw).hexdigest(),ident['source_sha256'])
        stock=next(n for n in ast.walk(ast.parse(raw)) if isinstance(n,ast.FunctionDef)
                   and n.name=='scatter_add_v4_bf16x2')
        def asm(fn):
            call=next(n for n in ast.walk(fn) if isinstance(n,ast.Call)
                      and ast.unparse(n.func)=='llvm.inline_asm')
            return ast.literal_eval(call.args[2])
        original=asm(stock); new=asm(function('scatter_add_v4_bf16x2_to_f32'))
        old_cvt=[x.strip() for x in original.split(';') if 'cvt.' in x]
        new_cvt=[x.strip() for x in new.split(';') if 'cvt.rn.' in x]
        self.assertEqual(new_cvt,old_cvt)
        self.assertEqual(new.count('cvt.f32.bf16'),8)
        self.assertEqual(new.count('red.global.add.v4.f32'),2)
        self.assertIn('add.u64 pnext,$0,16',new)
        self.assertNotIn('red.global.add.noftz.v4.bf16x2',new)

    def test_real_compile_factory_all_shapes_fp32_and_exact_tiled_descriptors(self):
        t=fake_torch(); dtype_names=('BFloat16','Float4E2M1FN','Float32','Float8E4M3FN',
                                    'Int32','Int64','Uint8')
        cutlass=types.SimpleNamespace(**{x:x for x in dtype_names})
        cute=types.SimpleNamespace(runtime=types.SimpleNamespace(
            make_fake_compact_tensor=lambda dtype,shape,**kw:Tensor(shape,dtype),
            make_fake_stream=lambda **kw:('fake-stream',kw)),AddressSpace=types.SimpleNamespace(gmem='global'))
        fu=types.ModuleType('flashinfer.cute_dsl.utils');fu.make_ptr=lambda *a,**kw:('ptr',a,kw)
        ns={'cutlass':cutlass,'cute':cute,'STAMP_SLOTS':71,'REFORM_SF_STAGE':1552,
            'EP_TILED_CACHE_TAG':constants()['EP_TILED_CACHE_TAG'],
            'MoEStaticEPTiledKernel':lambda **kw:kw}
        extract('ep_tiled_geometry',ns);extract('ep_tiled_scale_mode',ns)
        factory=extract('ep_tiled_compile_spec',ns)
        modules={'torch':t,'flashinfer':types.ModuleType('flashinfer'),
                 'flashinfer.cute_dsl':types.ModuleType('flashinfer.cute_dsl'),
                 'flashinfer.cute_dsl.utils':fu}
        with patch.dict(sys.modules,modules):
            keys=set()
            for m in range(1,33):
                for dtype in (t.int32,t.int64):
                    for sf6 in (False,True):
                        kernel,args,key=factory(num_tokens=m,topk_ids_dtype=dtype,reform_sf_pack=sf6)
                        self.assertEqual(len(args),30)
                        self.assertEqual((args[9].shape,args[11].shape),((4096,512,8,72),(4096,128,16,72)))
                        self.assertEqual((args[21].dtype,args[21].shape),('Float32',(m,4096)))
                        self.assertEqual(args[3].shape,(256,4096,72))
                        self.assertEqual(args[13].shape,(72,))
                        self.assertEqual(args[22].shape,(72,256))
                        self.assertEqual(args[-2],48)
                        self.assertEqual(kernel['num_tokens'],m)
                        self.assertIs(kernel['reform_sf_pack'],sf6)
                        self.assertEqual(key[10],'sf6_v1' if sf6 else 'raw_mma_scales')
                        self.assertEqual(args[26].shape,(72,512,1552) if sf6 else (1,1,16))
                        self.assertEqual(args[27].shape,(72,256,1552) if sf6 else (1,1,16))
                        self.assertEqual(args[26].dtype,'Uint8')
                        self.assertEqual(args[27].dtype,'Uint8')
                        keys.add(key)
            self.assertEqual(len(keys),128)
            with self.assertRaises(TypeError):factory(num_tokens=6,topk_ids_dtype=t.float32)
            for invalid in (None,0,1,'sf6'):
                with self.assertRaises(TypeError):factory(num_tokens=6,reform_sf_pack=invalid)

    def _launch_fixture(self,m=6):
        events=[];t=fake_torch(events);dev=types.SimpleNamespace(type='cuda')
        def tensor(shape,dtype,**kw):return Tensor(shape,dtype,device=dev,events=events,**kw)
        ws=types.SimpleNamespace(state_E=72,weight_E=72,k=4096,n=2048,num_topk=8,
             max_rows=256,activation_precision='fp4',quant_mode='nvfp4',device=dev)
        for name,shape,dtype in [('row_counts',(72,),t.int32),('token_map',(72,256),t.int32),
            ('token_weights',(72,256),t.float32),('packed_input',(72,256,2048),t.uint8),
            ('packed_input_scale',(72,256,256),t.uint8),('barrier_count',(1,),t.int32),
            ('barrier_epoch',(1,),t.int32),('active_expert_count',(1,),t.int32),
            ('weight_expert_ids',(72,),t.int32),('global_to_local_expert',(72,),t.int32)]:
            setattr(ws,name,tensor(shape,dtype))
        ws.packed_a_view=ws.packed_input;ws.packed_a_flat=ws.packed_input.view(-1)
        ws.scale_flat=ws.packed_input_scale.view(-1)
        weights=types.SimpleNamespace(tiled=True,packed_only=False,reform_scales=None,
             w13_fp4=tensor((4096,256,8,72),t.float4_e2m1fn_x2,strides=(256,1,1048576,8388608)),
             down_fp4=tensor((4096,64,16,72),t.float4_e2m1fn_x2,strides=(64,1,262144,4194304)),
             _w13_sf_storage=tensor((72*4096*256,),t.uint8),
             _down_sf_storage=tensor((72*4096*128,),t.uint8),
             w1_alpha=tensor((72,),t.float32),w2_alpha=tensor((72,),t.float32))
        scratch=types.SimpleNamespace(scatter_fp32=tensor((32,4096),t.float32),
             stamps=tensor((48,71),t.int64),counter=tensor((1,),t.int32),
             dummy_scales=tensor((1,1,16),t.uint8),max_tokens=32,max_active_clusters=48)
        ns={'STAMP_SLOTS':71,'REFORM_SF_STAGE':1552};extract('ep_tiled_geometry',ns)
        def compiled(*args):events.append(('compiled',args))
        compile_options=[]
        def get(**kw):compile_options.append(kw);return compiled,48
        ns['get_ep_tiled_decode_kernel']=get
        launch=extract('launch_ep_tiled_decode',ns)
        launch.compile_options=compile_options
        kwargs=dict(workspace=ws,weights=weights,a=tensor((m,4096),t.bfloat16),
            topk_ids=tensor((m,8),t.int32),topk_weights=tensor((m,8),t.float32),
            input_gs=tensor((72,),t.float32),down_input_scale=tensor((72,),t.float32),
            output=tensor((m,4096),t.bfloat16),scratch=scratch)
        return t,events,launch,kwargs

    def test_actual_launch_uses_same_weights_pinned_fp32_and_current_stream_then_copy(self):
        t,events,launch,kw=self._launch_fixture()
        with patch.dict(sys.modules,{'torch':t}):
            for _ in range(2):self.assertIs(launch(**kw),kw['output'])
        self.assertEqual([e[0] for e in events],['record','compiled','copy']*2)
        for e in (events[1],events[4]):
            args=e[1];self.assertEqual(len(args),28)
            self.assertIs(args[9],kw['weights'].w13_fp4)
            self.assertIs(args[11],kw['weights'].down_fp4)
            self.assertEqual(args[21].pointer,kw['scratch'].scatter_fp32.pointer)
            self.assertEqual(args[21].shape,(6,4096))
            self.assertEqual(args[21].dtype,t.float32)
            self.assertEqual(args[10],kw['weights']._w13_sf_storage.pointer)
        self.assertEqual(events[2][1],events[5][1])
        self.assertTrue(all(not kw['reform_sf_pack'] for kw in launch.compile_options))

    def _sf6_fixture(self,m=6):
        t,events,launch,kw=self._launch_fixture(m)
        weights=kw['weights'];weights.packed_only=True
        weights._w13_sf_storage=weights._down_sf_storage=None
        weights.sfb1_packed=Tensor((72,512,1552),t.uint8,device=kw['a'].device)
        weights.sfb2_packed=Tensor((72,256,1552),t.uint8,device=kw['a'].device)
        weights.reform_scales=types.SimpleNamespace(enabled=True,
            fc1=weights.sfb1_packed,fc2=weights.sfb2_packed)
        return t,events,launch,kw

    def test_sf6_launch_uses_only_shared_packed_owner_for_both_native_geometries(self):
        for m in (1,6,8,9,12,24,32):
            t,events,launch,kw=self._sf6_fixture(m)
            with patch.dict(sys.modules,{'torch':t}):
                self.assertIs(launch(**kw),kw['output'])
            self.assertEqual([e[0] for e in events],['record','compiled','copy'])
            args=events[1][1];owner=kw['weights'].reform_scales
            self.assertIs(args[26],owner.fc1);self.assertIs(args[27],owner.fc2)
            self.assertEqual(args[10],owner.fc1.data_ptr())
            self.assertEqual(args[12],owner.fc2.data_ptr())
            self.assertIs(args[9],kw['weights'].w13_fp4)
            self.assertIs(args[11],kw['weights'].down_fp4)
            self.assertEqual(args[21].pointer,kw['scratch'].scatter_fp32.pointer)
            self.assertIs(launch.compile_options[0]['reform_sf_pack'],True)
        for mutation in ('disabled','retainedraw','rawowner','fc1shape','fc2shape','dtype','alias'):
            t,events,launch,kw=self._sf6_fixture()
            w=kw['weights']
            if mutation=='disabled':w.reform_scales.enabled=False
            if mutation=='retainedraw':w._w13_sf_storage=object()
            if mutation=='rawowner':w.packed_only=False
            if mutation=='fc1shape':w.reform_scales.fc1.shape=(72,128,1552)
            if mutation=='fc2shape':w.reform_scales.fc2.shape=(72,64,1552)
            if mutation=='dtype':w.reform_scales.fc1.dtype=t.float32
            if mutation=='alias':w.sfb2_packed=object()
            with patch.dict(sys.modules,{'torch':t}),self.assertRaises(ValueError):launch(**kw)
            self.assertEqual(events,[]);self.assertEqual(launch.compile_options,[])

    def test_sf6_constructor_preserves_t_and_tr_geometry_and_warmup_mode(self):
        class Base:
            def __init__(self,**kwargs):self.options=kwargs
        init=function('__init__');init.decorator_list=[]
        cls=ast.ClassDef(name='MoEStaticEPTiledKernel',bases=[ast.Name('Base',ast.Load())],
                        keywords=[],body=[init],decorator_list=[])
        ns={'Base':Base,'ep_tiled_source_contract':lambda:None}
        extract('ep_tiled_geometry',ns);extract('ep_tiled_scale_mode',ns)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),str(SOURCE),'exec'),ns)
        for m in range(1,33):
            for mode in (False,True):
                k=ns['MoEStaticEPTiledKernel'](num_tokens=m,max_rows=256,
                    max_active_clusters=48,reform_sf_pack=mode)
                self.assertIs(k.options['reform_sf_pack'],mode)
                self.assertIs(k.options['decode_reform'],m<=8)
                self.assertEqual(k.options['output_tile_count_n'],16)
                self.assertEqual((k.options['fc1_stages'],k.options['fc2_stages']),(2,2))
        calls=[];ns['get_ep_tiled_decode_kernel']=lambda **kw:calls.append(kw)
        warm=extract('warm_ep_tiled_decode',ns)
        self.assertEqual(warm(reform_sf_pack=True),tuple(range(1,33)))
        self.assertEqual(len(calls),32)
        self.assertTrue(all(call['reform_sf_pack'] is True for call in calls))

    def test_bad_layout_scale_or_scratch_fails_before_any_kernel(self):
        for mutation in ('row-major','sf6','badstrides','rawscale','smalloutput','routeweight'):
            t,events,launch,kw=self._launch_fixture()
            if mutation=='row-major':kw['weights'].tiled=False
            if mutation=='sf6':kw['weights'].packed_only=True
            if mutation=='badstrides':kw['weights'].down_fp4._strides=(1,2,3,4)
            if mutation=='rawscale':kw['weights']._w13_sf_storage.shape=(16,)
            if mutation=='smalloutput':kw['scratch'].max_tokens=5
            if mutation=='routeweight':kw['topk_weights'].dtype=t.bfloat16
            with patch.dict(sys.modules,{'torch':t}),self.assertRaises(ValueError):launch(**kw)
            self.assertEqual(events,[])

    def test_entry_reuses_only_v5_tma_and_allocator_rejects_capture(self):
        call=function('__call__');calls=[ast.unparse(n.func) for n in ast.walk(call) if isinstance(n,ast.Call)]
        self.assertEqual(calls,['self._check_ep_call','MoEStaticKernelV5.__call__'])
        body=function('allocate_ep_tiled_decode_scratch')
        t=fake_torch();t.cuda.is_current_stream_capturing=lambda:True
        ns={};extract('ep_tiled_geometry',ns)
        alloc=extract('allocate_ep_tiled_decode_scratch',ns)
        with patch.dict(sys.modules,{'torch':t}),self.assertRaises(RuntimeError):alloc(device=types.SimpleNamespace(type='cuda'))


if __name__=='__main__':unittest.main()
