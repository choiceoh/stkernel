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


def shard_helpers(ns):
    """Load actual pure geometry helpers for AST-extracted production functions."""
    path = SOURCE.with_name('glm53_ep_shard_geometry.py')
    tree = ast.parse(path.read_text())
    wanted = {'ep_shard_geometry', 'ep_shard_cache_suffix', 'require_hybrid_mode'}
    nodes = [copy.deepcopy(n) for n in tree.body if
        (isinstance(n, ast.FunctionDef) and n.name in wanted) or
        (isinstance(n, ast.Assign) and len(n.targets) == 1 and
         isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'HYBRID_TAG')]
    assert {n.name for n in nodes if isinstance(n, ast.FunctionDef)} == wanted
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 str(path), 'exec'), ns)
    return ns


def route_helpers(ns):
    """Load actual declaration/key helpers, not a fabricated local key stub."""
    shard_helpers(ns)
    ns['EP_TILED_ROUTE_CACHE_TAG'] = constants()['EP_TILED_ROUTE_CACHE_TAG']
    ns['EP_TILED_DECODE_OPT_CACHE_TAG'] = constants()['EP_TILED_DECODE_OPT_CACHE_TAG']
    ns.setdefault('_EP_TILED_DECODE_OPT', False)
    extract('ep_tiled_decode_opt', ns)
    extract('ep_tiled_decode_opt_enabled', ns)
    extract('ep_tiled_route_metadata', ns)
    extract('ep_tiled_route_key', ns)
    return ns


def baseline_kernel_text():
    """Recover original bytes by selecting only the exact new opt=False arm."""
    source=SOURCE.read_text(); node=function('kernel'); lines=source.splitlines(keepends=True)
    branches=[n for n in ast.walk(node) if isinstance(n,ast.If)
              and ast.unparse(n.test)=='cutlass.const_expr(self.ep_decode_opt)']
    # Admit the exact five register-max additions, not just a branch count.
    # Both staging and quantization keep the complete old fallback bodies.
    assert len(branches)==5
    setup, scratch, registers, staging, quant = sorted(branches, key=lambda n:n.lineno)
    for branch, expected in ((setup, [
            'self._check_ep_storage(Storage)',
            'self._check_ep_q1_register_layout(tiled_mma1, epi1_smem_staged, a2_smem_layout, sfa2_smem_layout)']),
            (scratch, ['q1_max_scratch = shared_ptr_to_u32(storage.sC1.data_ptr())']),
            (registers, ['tRS_q1_halfmax = cute.make_rmem_tensor((4,), Float32)'])):
        assert not branch.orelse and [ast.unparse(n) for n in branch.body] == expected
    for branch, predicate, call in ((staging, 'epi_m_valid <= Int32(8)',
            'self._ep_q1_register_max(tRS_rD1_out, tRS_q1_halfmax, epi_m_valid, tidx, q1_max_scratch)'),
            (quant, 'epi_rows <= Int32(8)',
            'self._ep_q1_register_quantize(tRS_rD1_out, tRS_q1_halfmax, epi_rows, tidx, gs_value, q1_max_scratch, a2_base_addr, a2_smem_layout, sfa2_base_addr)')):
        assert len(branch.body)==1 and isinstance(branch.body[0],ast.If)
        bounded = branch.body[0]
        assert ast.unparse(bounded.test)==predicate
        assert [ast.unparse(n) for n in bounded.body] == [call]
        assert [ast.dump(n,include_attributes=False) for n in bounded.orelse] == [
            ast.dump(n,include_attributes=False) for n in branch.orelse]
    assert [ast.unparse(n) for n in staging.orelse] == [
        'cute.copy(tiled_copy_r2s1, tRS_rD1_out, tRS_sD1[None, None, None, 0])']
    assert len(quant.orelse)==2 and isinstance(quant.orelse[1],ast.While)
    assert ast.unparse(quant.orelse[0])=='quant_idx = Int32(tidx)'
    changes=[]
    for branch in branches:
        replacement=[]
        if branch.orelse:
            replacement=[line[4:] if line.startswith('    ') else line
                         for line in lines[branch.orelse[0].lineno-1:branch.end_lineno]]
        changes.append((branch.lineno-1,branch.end_lineno,replacement))
    fields=[n for n in ast.walk(node) if isinstance(n,ast.AnnAssign)
            and ast.unparse(n.target)=='sf2_packed_source']
    assert len(fields)==1
    field=fields[0]
    assert ast.unparse(field.annotation)=='cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, 16], 16]'
    comments=lines[field.lineno-3:field.lineno-1]
    assert 'zero-sized MemRange' in comments[0] and 'existing header padding' in comments[1]
    changes.append((field.lineno-3,field.end_lineno,[]))
    for start,end,replacement in sorted(changes,reverse=True):lines[start:end]=replacement
    restored=''.join(lines)
    restored_node=next(n for n in ast.walk(ast.parse(restored))
                       if isinstance(n,ast.FunctionDef) and n.name=='kernel')
    return ast.get_source_segment(restored,restored_node)


def local_reference_function(name):
    """Select exactly the declared local route branch; reject other changes."""
    node = ast.parse(baseline_kernel_text()).body[0] if name == 'kernel' else function(name)
    if name == 'kernel':
        node.decorator_list = function(name).decorator_list
    expression = "cutlass.const_expr(self.ep_route_mode == 'global')"
    branches = [n for n in ast.walk(node) if isinstance(n, ast.If)
                and ast.unparse(n.test) == expression]
    assert len(branches) == 1, 'missing or duplicated global route admission'
    branch = branches[0]
    if name == 'kernel':
        assert [ast.unparse(n) for n in branch.body] == [
            'expert_id = self._global_route_id(topk_ids, pair_idx, expert_map)']
        assert [ast.unparse(n) for n in branch.orelse] == [
            'expert_id = topk_ids[pair_idx].to(Int32)']
    else:
        assert name == '__call__' and not branch.orelse
        assert isinstance(branch.body[-1], ast.Return)
        assert ast.unparse(branch.body[-1].value.func) == 'self._call_global'
    class LocalOnly(ast.NodeTransformer):
        def visit_If(self, n):
            if ast.unparse(n.test) == expression:
                return n.orelse
            return self.generic_visit(n)
    node = LocalOnly().visit(node)
    assert node.args.args[-1].arg == 'expert_map'
    assert ast.unparse(node.args.args[-1].annotation) == 'cute.Tensor'
    assert len(node.args.defaults) == 1 and ast.unparse(node.args.defaults[0]) == 'None'
    node.args.args.pop(); node.args.defaults.pop()
    return node


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
        actual = local_reference_function('kernel'); expected=function('kernel',STOCK)
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
            def __init__(self, selected):self.selected,self.branches=selected,0
            def visit_If(self,n):
                if ast.unparse(n.test)=='cutlass.const_expr(self.scatter_bf16)':
                    self.branches+=1
                    return [self.visit(x) for x in (n.body if self.selected else n.orelse)]
                return self.generic_visit(n)
            def visit_Name(self,n):
                if n.id=='scatter_add_v4_bf16x2_to_f32': n.id='scatter_add_v4_bf16x2'
                return n
            def visit_Assign(self,n):
                if ast.unparse(n.targets[0])=='scatter_output[j // cols, j % cols]':
                    self.assert_zero = ast.unparse(n.value)
                    wanted='cutlass.BFloat16(0.0)' if self.selected else 'cutlass.Float32(0.0)'
                    if self.assert_zero!=wanted: raise AssertionError(self.assert_zero)
                    n.value=ast.parse('cutlass.BFloat16(0.0)',mode='eval').body
                return self.generic_visit(n)
        for selected in (False,True):
            transform=RestoreABI(selected)
            restored=transform.visit(copy.deepcopy(actual))
            self.assertEqual(transform.branches,2)
            self.assertEqual(ast.dump(restored,include_attributes=False),ast.dump(expected,include_attributes=False))

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
        # The BF16 branch uses that same pinned helper and the same weighted
        # BF16 contributions; only the accumulation precision is different.
        scatter=next(n for n in ast.walk(function('kernel')) if isinstance(n,ast.If)
                     and ast.unparse(n.test)=='cutlass.const_expr(self.scatter_bf16)'
                     and isinstance(n.body[0],ast.Expr))
        self.assertEqual(ast.unparse(scatter.body[0].value.func),'scatter_add_v4_bf16x2')
        self.assertEqual(ast.dump(scatter.body[0].value.args[0]),ast.dump(scatter.orelse[0].value.args[0]))
        self.assertEqual([ast.dump(x) for x in scatter.body[0].value.args],
                         [ast.dump(x) for x in scatter.orelse[0].value.args])
        self.assertEqual(original.count('cvt.rn.satfinite.bf16x2.f32'),4)
        self.assertEqual(original.count('red.global.add.noftz.v4.bf16x2'),1)
        # Validate the real CPU receipt reader against the full pinned helper
        # file, including rejection of mutated bytes and a substituted path.
        probe=ROOT/'probes/glm53_ep_tiled_compile.py'
        nodes=[n for n in ast.parse(probe.read_text()).body if isinstance(n,ast.FunctionDef)
               and n.name in ('scatter_helper_receipt','validate_scatter_helper_receipt')]
        ns=dict(Path=Path,json=json,hashlib=hashlib)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(probe),'exec'),ns)
        actual_path=Path(ident['source_path'])
        with patch.object(Path,'read_bytes',autospec=True,return_value=raw):
            receipt=ns['scatter_helper_receipt'](ROOT,actual_path)
            self.assertEqual(receipt['sha256'],ident['source_sha256'])
            with self.assertRaises(AssertionError):ns['scatter_helper_receipt'](ROOT,actual_path.with_name('other.py'))
        with patch.object(Path,'read_bytes',autospec=True,return_value=raw[:-1]+bytes([raw[-1]^1])):
            with self.assertRaises(AssertionError):ns['scatter_helper_receipt'](ROOT,actual_path)

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
            'EP_TILED_A_RING_CACHE_TAG':constants()['EP_TILED_A_RING_CACHE_TAG'],
            'EP_TILED_SF6_WORD_CACHE_TAG':constants()['EP_TILED_SF6_WORD_CACHE_TAG'],
            'EP_TILED_BF16_SCATTER_CACHE_TAG':constants()['EP_TILED_BF16_SCATTER_CACHE_TAG'],
            'MoEStaticEPTiledKernel':lambda **kw:kw}
        extract('ep_tiled_geometry',ns);extract('ep_tiled_scale_mode',ns);route_helpers(ns)
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
                        selected=sf6 and m<=8
                        self.assertEqual((args[21].dtype,args[21].shape),
                                         ('BFloat16' if selected else 'Float32',(m,4096)))
                        self.assertEqual(args[3].shape,(256,4096,72))
                        self.assertEqual(args[13].shape,(72,))
                        self.assertEqual(args[22].shape,(72,256))
                        self.assertEqual(args[-2],48)
                        self.assertEqual(kernel['num_tokens'],m)
                        self.assertIs(kernel['reform_sf_pack'],sf6)
                        self.assertEqual(key[10],'sf6_v1' if sf6 else 'raw_mma_scales')
                        self.assertEqual(len(key),19 if selected else 16)
                        self.assertEqual(key[15],'bf16_scatter' if selected else 'fp32_scatter')
                        if sf6 and m<=8:
                            self.assertEqual(key[-3:], (constants()['EP_TILED_A_RING_CACHE_TAG'],
                                constants()['EP_TILED_SF6_WORD_CACHE_TAG'],constants()['EP_TILED_BF16_SCATTER_CACHE_TAG']))
                        else:
                            self.assertEqual(key[-1], 'fp32_scatter')
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
        ns={'STAMP_SLOTS':71,'REFORM_SF_STAGE':1552};extract('ep_tiled_geometry',ns);route_helpers(ns)
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
            selected=m<=8
            self.assertEqual([e[0] for e in events],['record','compiled']+([] if selected else ['copy']))
            args=events[1][1];owner=kw['weights'].reform_scales
            self.assertIs(args[26],owner.fc1);self.assertIs(args[27],owner.fc2)
            self.assertEqual(args[10],owner.fc1.data_ptr())
            self.assertEqual(args[12],owner.fc2.data_ptr())
            self.assertIs(args[9],kw['weights'].w13_fp4)
            self.assertIs(args[11],kw['weights'].down_fp4)
            if selected:
                self.assertIs(args[21],kw['output'])
                self.assertEqual(args[21].dtype,t.bfloat16)
                self.assertEqual(events[0][1],kw['output'].pointer)
            else:
                self.assertEqual(args[21].pointer,kw['scratch'].scatter_fp32.pointer)
                self.assertEqual(args[21].dtype,t.float32)
            self.assertIs(launch.compile_options[0]['reform_sf_pack'],True)
        for mutation in ('disabled','retainedraw','rawowner','fc1shape','fc2shape','dtype','alias',
                         'outputdtype','outputalignment'):
            t,events,launch,kw=self._sf6_fixture()
            w=kw['weights']
            if mutation=='disabled':w.reform_scales.enabled=False
            if mutation=='retainedraw':w._w13_sf_storage=object()
            if mutation=='rawowner':w.packed_only=False
            if mutation=='fc1shape':w.reform_scales.fc1.shape=(72,128,1552)
            if mutation=='fc2shape':w.reform_scales.fc2.shape=(72,64,1552)
            if mutation=='dtype':w.reform_scales.fc1.dtype=t.float32
            if mutation=='alias':w.sfb2_packed=object()
            if mutation=='outputdtype':kw['output'].dtype=t.float32
            if mutation=='outputalignment':kw['output'].pointer+=2
            with patch.dict(sys.modules,{'torch':t}),self.assertRaises(ValueError):launch(**kw)
            self.assertEqual(events,[]);self.assertEqual(launch.compile_options,[])

    def test_sf6_constructor_preserves_t_and_tr_geometry_and_warmup_mode(self):
        class Base:
            def __init__(self,**kwargs):self.options=kwargs
        init=function('__init__');init.decorator_list=[]
        cls=ast.ClassDef(name='MoEStaticEPTiledKernel',bases=[ast.Name('Base',ast.Load())],
                        keywords=[],body=[init],decorator_list=[])
        ns={'Base':Base,'ep_tiled_source_contract':lambda:None}
        extract('ep_tiled_geometry',ns);extract('ep_tiled_scale_mode',ns);route_helpers(ns)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),str(SOURCE),'exec'),ns)
        for m in range(1,33):
            for mode in (False,True):
                k=ns['MoEStaticEPTiledKernel'](num_tokens=m,max_rows=256,
                    max_active_clusters=48,reform_sf_pack=mode)
                self.assertIs(k.options['reform_sf_pack'],mode)
                self.assertIs(k.options['decode_reform'],m<=8)
                self.assertIs(k.scatter_bf16,mode and m<=8)
                self.assertEqual((k.ep_route_mode,k.ep_route_map_len,k.ep_local_expert_offset),
                                 ('local',None,0))
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
        call=local_reference_function('__call__');calls=[ast.unparse(n.func) for n in ast.walk(call) if isinstance(n,ast.Call)]
        self.assertEqual(calls,['self._check_ep_call','MoEStaticKernelV5.__call__'])
        # Execute the real host dispatch against inert callees. This also
        # binds the extra map's exact position before any CuTe TMA setup runs.
        events=[]
        def inherited(*args):events.append(('local',args))
        ns=dict(cutlass=types.SimpleNamespace(const_expr=bool,Int32='i32',Int64='i64'),
                MoEStaticKernelV5=types.SimpleNamespace(__call__=inherited))
        entry=extract('__call__',ns)
        map_check=extract('_check_ep_route_map',ns)
        self.assertEqual(function('_check_ep_route_map').decorator_list,[])
        self.assertFalse(any(isinstance(n,ast.Raise) for n in ast.walk(function('__call__'))))
        global_branch=next(n for n in ast.walk(function('__call__')) if isinstance(n,ast.If)
            and ast.unparse(n.test)=="cutlass.const_expr(self.ep_route_mode == 'global')")
        self.assertEqual(ast.unparse(global_branch.body[0]),'self._check_ep_route_map(expert_map)')
        owner=types.SimpleNamespace(ep_route_mode='local',ep_route_map_len=None,
            _check_ep_call=lambda *a:events.append(('check',a)),
            _call_global=lambda *a:events.append(('global',a)))
        owner._check_ep_route_map=types.MethodType(map_check,owner)
        operands=[object() for _ in range(30)]
        entry(owner,*operands)
        self.assertEqual([e[0] for e in events],['check','local'])
        self.assertEqual(events[-1][1],(owner,*operands))
        owner.ep_route_mode='global'
        for length,dtype in ((None,'i32'),(0,'i32'),(288,'i32'),(288,'i64')):
            events.clear();owner.ep_route_map_len=length
            mapping=Tensor((length or 1,),dtype)
            entry(owner,*operands,mapping)
            self.assertEqual([e[0] for e in events],['check','global'])
            self.assertEqual(events[-1][1],(*operands,mapping))
        for mapping in (None,Tensor((1,),'i32'),Tensor((288,),'f32')):
            events.clear()
            with self.assertRaises(ValueError):entry(owner,*operands,mapping)
            self.assertEqual([e[0] for e in events],['check'])
        body=function('allocate_ep_tiled_decode_scratch')
        t=fake_torch();t.cuda.is_current_stream_capturing=lambda:True
        ns={};extract('ep_tiled_geometry',ns)
        alloc=extract('allocate_ep_tiled_decode_scratch',ns)
        with patch.dict(sys.modules,{'torch':t}),self.assertRaises(RuntimeError):alloc(device=types.SimpleNamespace(type='cuda'))

    def test_real_entry_rejects_mixed_bf16_fp32_fake_output_abis(self):
        types_=('BFloat16','Float32','Int32','Int64','Float4E2M1FN')
        ns=dict(cutlass=types.SimpleNamespace(**{x:x for x in types_}))
        check=extract('_check_ep_call',ns)
        for m in range(1,33):
            for sf6 in (False,True):
                selected=sf6 and m<=8
                owner=types.SimpleNamespace(ep_num_tokens=m,ep_max_rows=256,scatter_bf16=selected,
                                            ep_num_experts=72,ep_intermediate_size=2048)
                args=[Tensor((m,4096),'BFloat16'),Tensor((m*8,),'Int32'),Tensor((m*8,),'Float32'),
                      Tensor((4096,512,8,72),'Float4E2M1FN'),Tensor((4096,128,16,72),'Float4E2M1FN'),
                      Tensor((72,),'Int32'),Tensor((72,256),'Int32'),
                      Tensor((m,4096),'BFloat16' if selected else 'Float32')]
                check(owner,*args)
                for index,replacement in ((7,Tensor((m,4096),'Float32' if selected else 'BFloat16')),
                                           (7,Tensor((m+1,4096),args[7].dtype)),
                                           (3,Tensor(args[3].shape,'BFloat16'))):
                    bad=args.copy();bad[index]=replacement
                    with self.subTest(m=m,sf6=sf6,index=index),self.assertRaises(ValueError):check(owner,*bad)

    def test_entire_output_is_zeroed_once_before_routes_for_both_abis(self):
        body=function('kernel').body
        start=next(i for i,n in enumerate(body) if isinstance(n,ast.Assign)
                   and ast.unparse(n.targets[0])=='scatter_total')
        self.assertEqual(ast.unparse(body[start+3]),'cute.arch.sync_threads()')
        self.assertEqual(ast.unparse(body[start+4].value.func),'self._resident_grid_barrier')
        fn=ast.FunctionDef(name='zero',args=ast.arguments(posonlyargs=[],args=[ast.arg(x) for x in
            ('self','num_tokens','cols','flat_tid','flat_stride','scatter_output')],kwonlyargs=[],kw_defaults=[],defaults=[]),
            body=copy.deepcopy(body[start:start+3]),decorator_list=[])
        ns=dict(cutlass=types.SimpleNamespace(const_expr=bool,Float32=lambda x:('f32',x),
                                              BFloat16=lambda x:('bf16',x)))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(SOURCE),'exec'),ns)
        class Output:
            def __init__(self,m,selected):self.m,self.dtype,self.count=m,'bf16' if selected else 'f32',bytearray((m+1)*4096)
            def __setitem__(self,index,value):
                row,col=index
                assert 0<=row<self.m and 0<=col<4096 and value==(self.dtype,0.)
                self.count[row*4096+col]+=1
        cases=[(m,mode,1) for m in range(1,33) for mode in (False,True)]+[(6,True,48),(32,False,48)]
        for m,sf6,grid in cases:
            selected=sf6 and m<=8;output=Output(m,selected)
            for tid in range(grid*160):
                ns['zero'](types.SimpleNamespace(scatter_bf16=selected),m,4096,tid,grid*160,output)
            self.assertEqual(output.count[:m*4096],bytes([1])*(m*4096))
            self.assertEqual(output.count[m*4096:],bytes(4096))


if __name__=='__main__':unittest.main()
