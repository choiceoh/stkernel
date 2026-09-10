"""CPU source/oracle contracts; actual CuTe lowering and numerics remain fleet gates."""
import ast
import copy
import hashlib
import math
from pathlib import Path
import struct
import sys
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
import test_glm53_ep_tiled_static as static_tests
from test_glm53_ep_tiled_static import (
    Tensor, constants, extract, fake_torch, function, baseline_kernel_text,
)

SOURCE = Path(__file__).resolve().parents[1] / "overlay/modules/glm53_moe/moe_static_ep_tiled.py"
V5 = SOURCE.with_name("moe_static_kernel_v5.py")
# Exact pre-fusion kernel source at eedd2e84; checked after removing only the
# declared optional operand and selecting the unchanged local route branch.
LOCAL_KERNEL_SHA256 = "f9af0f29945985cba066a9abf4dcab4417e23806631c7850073867408d6cb048"


def route_ns():
    ns = static_tests.shard_helpers(constants())
    ns["_EP_TILED_DECODE_OPT"] = False
    extract("ep_tiled_decode_opt", ns)
    extract("ep_tiled_route_metadata", ns)
    extract("ep_tiled_route_key", ns)
    return ns


class I32(int):
    def __new__(cls, x=0):
        return int.__new__(cls, (int(x) + (1 << 31)) % (1 << 32) - (1 << 31))
    def __sub__(self, x): return I32(int(self) - int(x))
    def to(self, dtype): return dtype(self)


class I64(int):
    def __new__(cls, x=0):
        return int.__new__(cls, (int(x) + (1 << 63)) % (1 << 64) - (1 << 63))
    def to(self, dtype): return dtype(self)


class Guarded:
    def __init__(self, values=(), dtype=I64):
        self.values, self.dtype, self.reads = values, dtype, []
    def __getitem__(self, i):
        assert 0 <= i < len(self.values), "out-of-range/poisoned read"
        self.reads.append(int(i))
        return self.dtype(self.values[int(i)])


def admission():
    ns = dict(Int32=I32, Int64=I64, cutlass=types.SimpleNamespace(const_expr=lambda x:x))
    return extract("_global_route_id", ns)


def reference(expert, mapping, offset):
    # Independent scalar version of remap -> local native admission. In map
    # mode, sign/range precedes narrowing; offset mode narrows before subtract.
    signed = lambda x: (int(x) + 2147483648) % 4294967296 - 2147483648
    if mapping is not None:
        if not 0 <= expert < len(mapping) or mapping[expert] < 0: return 72
        local = signed(mapping[expert])
    else:
        local = signed(signed(expert) - offset)
        if expert < 0: return 72
    return local if 0 <= local < 72 else 72


class EPTiledRouteFusionTests(unittest.TestCase):
    def test_actual_admission_matches_narrowing_maps_offsets_and_bounds(self):
        run = admission()
        values = [-2**63, -2**32, -1, 0, 1, 70, 71, 72, 73, 287, 288,
                  2**31-1, 2**31, 2**32-1, 2**32, 2**32+71, 2**63-1]
        for dtype in (I32, I64):
            actual = [int(dtype(x)) for x in values]
            for offset in (0, 1, 72, 216, 2**31-1):
                owner = types.SimpleNamespace(ep_num_experts=72, ep_route_map_len=None, ep_local_expert_offset=offset)
                for e in actual:
                    ids, mapping = Guarded([e], dtype), Guarded()
                    self.assertEqual(run(owner,ids,I32(0),mapping), reference(e,None,offset))
                    self.assertEqual(mapping.reads, [])
        mapvalues = [-2**63, -2**32, -1, 0, 1, 71, 72, 73, 2**31-1,
                     2**31, 2**32-1, 2**32, 2**32+71, 2**63-1]
        for dtype in (I32,I64):
            mapping = [int(dtype(x)) for x in mapvalues]
            owner = types.SimpleNamespace(ep_num_experts=72, ep_route_map_len=len(mapping), ep_local_expert_offset=0)
            for e in range(-2,len(mapping)+2):
                ids, arr = Guarded([e]), Guarded(mapping,dtype)
                self.assertEqual(run(owner,ids,I32(0),arr), reference(e,mapping,0))
                self.assertEqual(arr.reads, [e] if 0<=e<len(mapping) else [])
            for e in (-2**63,2**32,2**63-1):
                arr=Guarded(mapping,dtype)
                self.assertEqual(run(owner,Guarded([e]),I32(0),arr),72)
                self.assertEqual(arr.reads,[])

    def test_empty_and_remote_routes_never_read_poison_and_weight_bits_survive(self):
        run=admission(); owner=types.SimpleNamespace(ep_num_experts=72, ep_route_map_len=0,ep_local_expert_offset=0)
        ids,mapping=Guarded(),Guarded()
        self.assertEqual(run(owner,ids,I32(100),mapping),72)
        self.assertEqual(ids.reads+mapping.reads,[])
        route=next(n for n in ast.walk(function('kernel')) if isinstance(n,ast.While)
                   and ast.unparse(n.test)=='pair_idx < total_pairs')
        guard=copy.deepcopy(route.body[1])
        guard.body[1].body=[ast.Return(ast.Tuple([ast.Name('expert_id',ast.Load()),
                                                ast.Name('weight',ast.Load())],ast.Load()))]
        fn=ast.FunctionDef(name='select',args=ast.arguments(posonlyargs=[],args=[ast.arg(x) for x in
            ('expert_id','topk_weights','pair_idx','num_experts')],kwonlyargs=[],kw_defaults=[],defaults=[]),
            body=[guard,ast.Return(ast.Constant(None))],decorator_list=[])
        class F32:
            def __init__(self,bits): self.bits=bits
            def to(self,dtype): return self
            def __ne__(self,other): return struct.unpack('<f',struct.pack('<I',self.bits))[0] != other
        ns=dict(Int32=I32,cutlass=types.SimpleNamespace(Float32=float))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(SOURCE),'exec'),ns)
        for e in (-2**31,-1,72,73,2**31-1):
            self.assertIsNone(ns['select'](e,Guarded(),0,72))
        for bits in (0x00000000,0x80000000):
            self.assertIsNone(ns['select'](0,[F32(bits)],0,72))
        for bits in (0x3f800000,0xbf800000,0x7fc00042,0xffc01234,0x7f800001):
            value=F32(bits)
            out=ns['select'](71,[value],0,72)
            self.assertIs(out[1],value)
            self.assertEqual(out[1].bits,bits)
        # Duplicate routes and maximum M32x8 capacity retain every selection.
        for m in (1,4,6,8,12,16,24,32):
            self.assertEqual(sum(ns['select'](71,[F32(0x3f800000)],0,72) is not None
                                 for _ in range(m*8)),m*8)

    def test_local_device_body_and_global_tma_host_are_exactly_preserved(self):
        raw=baseline_kernel_text()
        branch='            if cutlass.const_expr(self.ep_route_mode == "global"):\n                expert_id = self._global_route_id(topk_ids, pair_idx, expert_map)\n            else:\n                expert_id = topk_ids[pair_idx].to(Int32)'
        self.assertEqual(raw.count(branch),1)
        raw=raw.replace(branch,'            expert_id = topk_ids[pair_idx].to(Int32)')
        raw=raw.replace('        expert_map: cute.Tensor = None,\n','')
        self.assertEqual(hashlib.sha256(raw.encode()).hexdigest(),LOCAL_KERNEL_SHA256)
        actual, expected=function('_call_global'),function('__call__',V5)
        actual.name='__call__'
        self.assertEqual(actual.args.args[-1].arg,'expert_map')
        actual.args.args.pop()
        call=next(n for n in ast.walk(actual) if isinstance(n,ast.Call)
                  and ast.unparse(n.func)=='self.kernel')
        self.assertEqual(ast.unparse(call.args[-1]),'expert_map');call.args.pop()
        self.assertEqual(ast.dump(actual,include_attributes=False),ast.dump(expected,include_attributes=False))
        self.assertEqual(hashlib.sha256(V5.read_bytes()).hexdigest(),constants()['STOCK_V5_SHA256'])

    def test_metadata_key_separates_global_variants_and_canonicalizes_unused_offset(self):
        t=fake_torch();ns=route_ns()
        with patch.dict(sys.modules,{'torch':t}):
            key=ns['ep_tiled_route_key']
            self.assertEqual(key(),())
            baseline=key(route_mode='global',expert_map_len=288,expert_map_dtype=t.int32)
            self.assertEqual(baseline,(constants()['EP_TILED_ROUTE_CACHE_TAG'],288,'int32',0))
            for offset in (0,72,216,2**31-1):
                self.assertEqual(key(route_mode='global',expert_map_len=288,
                    expert_map_dtype=t.int32,local_expert_offset=offset),baseline)
            self.assertNotEqual(key(route_mode='global',local_expert_offset=0),
                                key(route_mode='global',local_expert_offset=72))
            self.assertNotEqual(key(route_mode='global',expert_map_len=288,expert_map_dtype=t.int64),baseline)
            self.assertEqual(key(route_mode='global',expert_map_len=0),
                             key(route_mode='global',expert_map_len=0,expert_map_dtype=t.int64))
            self.assertEqual(key(route_mode='global',expert_map_len=0,expert_map_dtype=t.int32),
                             key(route_mode='global',expert_map_len=0,expert_map_dtype=t.int64))
            for kw in ({'route_mode':'unknown'}, {'route_mode':'local','expert_map_len':0},
                {'route_mode':'global','expert_map_len':-1},
                {'route_mode':'global','expert_map_len':2**31},
                {'route_mode':'global','expert_map_len':True},
                {'route_mode':'global','local_expert_offset':True},
                {'route_mode':'global','local_expert_offset':-1},
                {'route_mode':'global','local_expert_offset':2**31},
                {'route_mode':'global','expert_map_dtype':t.int32},
                {'route_mode':'global','expert_map_len':1,'expert_map_dtype':t.float32}):
                with self.assertRaises((ValueError,TypeError)): key(**kw)

    def test_real_compile_factory_and_exact_actual_warm_configuration(self):
        t=fake_torch();ns=route_ns()
        cutlass=types.SimpleNamespace(**{x:x for x in
            ('BFloat16','Float4E2M1FN','Float32','Float8E4M3FN','Int32','Int64','Uint8')})
        ns.update(cutlass=cutlass,STAMP_SLOTS=71,REFORM_SF_STAGE=1552,
                  MoEStaticEPTiledKernel=lambda **kw:kw,
                  cute=types.SimpleNamespace(runtime=types.SimpleNamespace(
                    make_fake_compact_tensor=lambda dtype,shape,**kw:Tensor(shape,dtype),
                    make_fake_stream=lambda **kw:'stream'),AddressSpace=types.SimpleNamespace(gmem='global')))
        extract('ep_tiled_geometry',ns);extract('ep_tiled_scale_mode',ns)
        factory=extract('ep_tiled_compile_spec',ns)
        fu=types.ModuleType('flashinfer.cute_dsl.utils');fu.make_ptr=lambda *a,**kw:('ptr',a)
        with patch.dict(sys.modules,{'torch':t,'flashinfer.cute_dsl.utils':fu}):
            for m in (1,4,6,8,9,12,16,24,32):
                for sf6 in (False,True):
                    for ids_dtype in (t.int32,t.int64):
                        _,localargs,localkey=factory(num_tokens=m,reform_sf_pack=sf6,topk_ids_dtype=ids_dtype)
                        self.assertEqual(len(localargs),30)
                        for length,dtype in ((None,None),(0,t.int64),(288,t.int32),(288,t.int64)):
                            kernel,args,key=factory(num_tokens=m,reform_sf_pack=sf6,topk_ids_dtype=ids_dtype,
                                route_mode='global',expert_map_len=length,expert_map_dtype=dtype,local_expert_offset=216)
                            self.assertEqual(key[:-4],localkey)
                            self.assertEqual(len(args),31)
                            self.assertEqual(args[21].dtype,localargs[21].dtype)
                            self.assertEqual(args[-1].shape,(length or 1,))
                            self.assertEqual(args[-1].dtype,'Int64' if length and dtype==t.int64 else 'Int32')
                            self.assertEqual(args[28],48)
                            self.assertEqual(kernel['route_mode'],'global')
            calls=[];ns['get_ep_tiled_decode_kernel']=lambda **kw:calls.append(kw)
            warm=extract('warm_ep_tiled_decode',ns)
            rows=warm(reform_sf_pack=True,route_mode='global',topk_ids_dtype=t.int64,
                expert_map_len=288,expert_map_dtype=t.int32,local_expert_offset=216)
            self.assertEqual(rows,tuple(range(1,33)))
            self.assertEqual(len(calls),32)
            self.assertTrue(all(c['topk_ids_dtype']==t.int64 and c['expert_map_len']==288
                                and c['expert_map_dtype']==t.int32 for c in calls))

    def test_actual_launch_map_operand_local_abi_no_copy_and_invalid_metadata_abort(self):
        for m in (4,6,8,12,32):
            for length,dtype in ((None,None),(0,'int64'),(288,'int32'),(288,'int64')):
                fixture=static_tests.EPTiledStaticTests()
                t,events,_,kw=fixture._sf6_fixture(m)
                ns=route_ns();ns.update(STAMP_SLOTS=71,REFORM_SF_STAGE=1552)
                extract('ep_tiled_geometry',ns)
                options=[]
                def get(**opts):
                    options.append(opts)
                    return (lambda *args:events.append(('compiled',args))),48
                ns['get_ep_tiled_decode_kernel']=get
                launch=extract('launch_ep_tiled_decode',ns)
                mapping=None if length is None else Tensor((length,),dtype,device=kw['a'].device)
                kw.update(route_mode='global',expert_map=mapping,local_expert_offset=216)
                kw['topk_ids'].dtype=t.int64
                with patch.dict(sys.modules,{'torch':t}): self.assertIs(launch(**kw),kw['output'])
                args=next(e[1] for e in events if e[0]=='compiled')
                self.assertEqual(len(args),29)
                self.assertIs(args[-1],mapping if length else kw['scratch'].counter)
                self.assertEqual(options[0]['expert_map_len'],length)
                self.assertEqual(options[0]['topk_ids_dtype'],t.int64)
                self.assertEqual([e[0] for e in events],['record','compiled']+([] if m<=8 else ['copy']))
        for mutation in ('shape','dtype','device','offset','localmap','idsdtype'):
            t,events,_,kw=static_tests.EPTiledStaticTests()._sf6_fixture()
            ns=route_ns();ns.update(STAMP_SLOTS=71,REFORM_SF_STAGE=1552)
            extract('ep_tiled_geometry',ns)
            ns['get_ep_tiled_decode_kernel']=lambda **kw: (_ for _ in ()).throw(AssertionError('compiled invalid metadata'))
            launch=extract('launch_ep_tiled_decode',ns)
            mapping=Tensor((288,),t.int32,device=kw['a'].device)
            kw.update(route_mode='global',expert_map=mapping)
            if mutation=='shape':mapping.shape=(144,2)
            if mutation=='dtype':mapping.dtype=t.float32
            if mutation=='device':mapping.device=types.SimpleNamespace(type='cpu')
            if mutation=='offset':kw['local_expert_offset']=-1
            if mutation=='localmap':kw['route_mode']='local'
            if mutation=='idsdtype':kw['topk_ids'].dtype=t.float32
            with patch.dict(sys.modules,{'torch':t}),self.assertRaises((ValueError,TypeError)):launch(**kw)
            self.assertEqual(events,[])


if __name__ == '__main__':
    unittest.main()
