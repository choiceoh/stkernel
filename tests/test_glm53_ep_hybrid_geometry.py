"""Pure CPU source/shape contracts. No CuTe import, lowering, CUDA or numerics."""
import ast
import copy
import gzip
import hashlib
import importlib.util
import math
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT=Path(__file__).resolve().parents[1]
MODULES=ROOT/'overlay/modules/glm53_moe'
BASE_DIR=ROOT/'measurements/glm53_ep_tiled_20260909/ep76_onepass5/source'
sys.path.insert(0,str(ROOT/'tests'))
from test_glm53_ep_tiled_static import Tensor, fake_torch
from test_glm53_ep_tiled_route_fusion import I32,I64,Guarded
from test_glm53_ep_route_scale_cache import single_warp_projection

spec=importlib.util.spec_from_file_location('private_geometry',MODULES/'glm53_ep_shard_geometry.py')
geom=importlib.util.module_from_spec(spec);spec.loader.exec_module(geom)
NATIVE=MODULES/'moe_static_ep_tiled.py'
BASE=BASE_DIR/'moe_static_ep_tiled.py.gz'
DYNAMIC=MODULES/'moe_dynamic_ep_local.py'
BASE_DYNAMIC=BASE_DIR/'moe_dynamic_ep_local.py.gz'
STOCK_GATED=ROOT/'measurements/glm53_ep_local_20260908/cpu13/stock-gated.py.gz'
BASE_DISPATCH=ROOT/'measurements/glm53_ep_tiled_20260909/onepass1/source/moe_dispatch.py.gz'
PINNED_SOURCES={
    BASE:'39cc387e7ded15c77e94d72de9235e3d3f757bdbbc293b08823205e7931199ba',
    BASE_DYNAMIC:'89bab8e22514d4ce036e4cc6755a5dcfaeb8b12277766394dcdd101db8751595',
    STOCK_GATED:'993783308233288ddfa77293e9dbabdc825ba5bfdcc4dcc41e842a895ec33445',
    # This older full source has an exactly identical pre-hybrid dynamic key
    # function to ca076; native/dynamic math uses the actual ca076 archives.
    BASE_DISPATCH:'40bdc0e9c38c3e406fd7dd7350e14077b80b5edf8b5e9fc31d323753b9bfbadf',
}

def read_source(path):
    raw=path.read_bytes()
    if path.suffix=='.gz':raw=gzip.decompress(raw)
    if path in PINNED_SOURCES:
        assert hashlib.sha256(raw).hexdigest()==PINNED_SOURCES[path],path
    return raw.decode()


def node(path,name,cls=None):
    tree=ast.parse(read_source(path))
    if cls:tree=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
    matches=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name]
    assert len(matches)==1,(name,len(matches))
    return copy.deepcopy(matches[0])

def extract(path,name,ns,cls=None):
    n=node(path,name,cls);n.decorator_list=[]
    mod=ast.Module(body=[ast.ImportFrom('__future__',[ast.alias('annotations')],0),n],type_ignores=[])
    exec(compile(ast.fix_missing_locations(mod),str(path),'exec'),ns)
    return ns[name]

def ns_for(path):
    ns={n.targets[0].id:ast.literal_eval(n.value) for n in ast.parse(read_source(path)).body
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)
        and isinstance(n.value,ast.Constant)}
    ns.update(ep_shard_geometry=geom.ep_shard_geometry,
              ep_shard_cache_suffix=geom.ep_shard_cache_suffix,
              require_hybrid_mode=geom.require_hybrid_mode,_EP_TILED_DECODE_OPT=False)
    for name in ('ep_tiled_geometry','ep_tiled_scale_mode','ep_tiled_decode_opt',
                 'ep_tiled_route_metadata','ep_tiled_route_key'):
        extract(path,name,ns)
    return ns

def factory(path):
    t=fake_torch();ns=ns_for(path)
    ns['cutlass']=types.SimpleNamespace(**{x:x for x in
        ('BFloat16','Float4E2M1FN','Float32','Float8E4M3FN','Int32','Int64','Uint8')})
    ns['cute']=types.SimpleNamespace(runtime=types.SimpleNamespace(
        make_fake_compact_tensor=lambda dtype,shape,**kw:Tensor(shape,dtype),
        make_fake_stream=lambda **kw:('fake-stream',kw)),AddressSpace=types.SimpleNamespace(gmem='global'))
    ns.update(STAMP_SLOTS=71,REFORM_SF_STAGE=1552,MoEStaticEPTiledKernel=lambda **kw:kw)
    util=types.ModuleType('flashinfer.cute_dsl.utils');util.make_ptr=lambda *a,**kw:('ptr',a,kw)
    modules={'torch':t,'flashinfer':types.ModuleType('flashinfer'),
             'flashinfer.cute_dsl':types.ModuleType('flashinfer.cute_dsl'),
             'flashinfer.cute_dsl.utils':util}
    return extract(path,'ep_tiled_compile_spec',ns),modules,ns

class HybridContracts(unittest.TestCase):
    def test_exact_admission_shapes_memory_and_mode(self):
        old,new=geom.ep_shard_geometry(),geom.ep_shard_geometry(144,1024)
        self.assertEqual(new['raw_w13'],(144,2048,2048))
        self.assertEqual(new['raw_down'],(144,4096,512))
        self.assertEqual(new['sf6_fc1'],(144,256,1552))
        self.assertEqual(new['sf6_fc2'],(144,128,1552))
        self.assertEqual(new['native_slices'],8);self.assertEqual(new['dynamic_groups'],2)
        a,b=geom.memory_contract(old),geom.memory_contract(new)
        for k in ('routed_weight_bytes','sf6_bytes'):self.assertEqual(a[k],b[k])
        self.assertEqual(a['routed_weight_bytes'],864*2**20)
        self.assertEqual(a['sf6_bytes'],85819392)
        self.assertEqual((a['static_q0_bytes'],b['static_q0_bytes']),(42467328,84934656))
        self.assertEqual(geom.ep_shard_cache_suffix(old),())
        for e,i in ((72,1024),(144,2048),(288,512),(True,2048),(144.,1024)):
            with self.assertRaises(ValueError):geom.ep_shard_geometry(e,i)
        geom.require_hybrid_mode(new,True,False)
        for sf,opt in ((False,False),(True,None),(True,True),(True,0)):
            with self.assertRaises(ValueError):geom.require_hybrid_mode(new,sf,opt)

    def test_actual_factories_all_native_rows_preserve_baseline_keys_and_abi(self):
        old,mods,_=factory(BASE);new,_,_=factory(NATIVE)
        with patch.dict(sys.modules,mods):
            for m in range(1,33):
                for sf in (False,True):
                    for route in ('local','global'):
                        kw=dict(num_tokens=m,reform_sf_pack=sf,decode_opt=False,route_mode=route)
                        if route=='global':kw.update(expert_map_len=288,expert_map_dtype='int32')
                        _,a,ka=old(**kw);_,b,kb=new(**kw)
                        self.assertEqual(ka,kb)
                        self.assertEqual([(x.shape,x.dtype) if isinstance(x,Tensor) else x for x in a],
                                         [(x.shape,x.dtype) if isinstance(x,Tensor) else x for x in b])
                k,args,key=new(num_tokens=m,reform_sf_pack=True,decode_opt=False,
                              num_local_experts=144,intermediate_size=1024,
                              route_mode='global',expert_map_len=288,expert_map_dtype='int32')
                self.assertEqual(len(args),31)
                self.assertEqual(args[9].shape,(2048,512,8,144))
                self.assertEqual(args[11].shape,(4096,128,8,144))
                self.assertEqual(args[13].shape,(144,))
                self.assertEqual(args[22].shape,(144,256))
                self.assertEqual(args[26].shape,(144,256,1552))
                self.assertEqual(args[27].shape,(144,128,1552))
                self.assertEqual(args[21].dtype,'BFloat16' if m<=8 else 'Float32')
                self.assertEqual(key[-3:],(geom.HYBRID_TAG,144,1024))
                self.assertEqual((k['num_local_experts'],k['intermediate_size']),(144,1024))

    def test_actual_constructor_preserves_tiles_and_only_halves_item_count(self):
        class Parent:
            def __init__(self,**kw):self.options=kw
        ns=ns_for(NATIVE);ns.update(Parent=Parent,ep_tiled_source_contract=lambda:None)
        f=node(NATIVE,'__init__');f.decorator_list=[]
        c=ast.ClassDef(name='Kernel',bases=[ast.Name('Parent',ast.Load())],keywords=[],body=[f],decorator_list=[])
        exec(compile(ast.fix_missing_locations(ast.Module(body=[c],type_ignores=[])),str(NATIVE),'exec'),ns)
        for m in range(1,33):
            a=ns['Kernel'](num_tokens=m,max_rows=256,max_active_clusters=48,reform_sf_pack=True,decode_opt=False)
            b=ns['Kernel'](num_tokens=m,max_rows=256,max_active_clusters=48,reform_sf_pack=True,
                           decode_opt=False,num_local_experts=144,intermediate_size=1024)
            oa,ob=dict(a.options),dict(b.options)
            self.assertEqual(oa.pop('output_tile_count_n'),16)
            self.assertEqual(ob.pop('output_tile_count_n'),8)
            self.assertEqual(oa,ob)
            self.assertFalse(a.ep_decode_opt);self.assertFalse(b.ep_decode_opt)
            self.assertEqual((a.scatter_bf16,b.scatter_bf16),(m<=8,m<=8))

    def test_actual_route_bounds_signed_narrowing_and_no_poison_reads(self):
        run=extract(NATIVE,'_global_route_id',dict(Int32=I32,Int64=I64,
                    cutlass=types.SimpleNamespace(const_expr=bool)))
        signed=lambda x:int(I32(x))
        for e in (72,144):
            for offset in (0,144,2**31-1):
                owner=types.SimpleNamespace(ep_num_experts=e,ep_route_map_len=None,ep_local_expert_offset=offset)
                for raw in (-2**63,-1,0,e-1,e,287,288,2**31,2**32+e-1,2**63-1):
                    local=signed(signed(raw)-offset)
                    want=local if raw>=0 and 0<=local<e else e
                    self.assertEqual(run(owner,Guarded([raw]),I32(0),Guarded()),want)
            for dtype in (I32,I64):
                values=[int(dtype(x)) for x in (-1,0,e-1,e,2**32+e-1,2**31)]
                owner=types.SimpleNamespace(ep_num_experts=e,ep_route_map_len=len(values),ep_local_expert_offset=0)
                for raw in range(-1,len(values)+1):
                    mapped=values[raw] if 0<=raw<len(values) else -1
                    val=signed(mapped);want=val if mapped>=0 and 0<=val<e else e
                    arr=Guarded(values,dtype)
                    self.assertEqual(run(owner,Guarded([raw]),I32(0),arr),want)
                    self.assertEqual(arr.reads,[raw] if 0<=raw<len(values) else [])
            owner=types.SimpleNamespace(ep_num_experts=e,ep_route_map_len=0,ep_local_expert_offset=0)
            self.assertEqual(run(owner,Guarded(),I32(999),Guarded()),e)

    def test_device_math_barriers_and_dynamic_publishers_unchanged(self):
        for f in ('kernel','__call__','_call_global','_sf_expand_stage'):
            self.assertEqual(ast.dump(node(NATIVE,f)),ast.dump(node(BASE,f)))
        base=BASE_DYNAMIC
        for f in ('initialize_route_q0_and_publish','scatter_sC_to_gmem','publish_ep_local_uniform_tasks'):
            actual = node(DYNAMIC,f)
            if f == 'initialize_route_q0_and_publish':
                actual = single_warp_projection(actual)
            self.assertEqual(ast.dump(actual),ast.dump(node(base,f)))
        # Execute actual existing vector dispatcher and its actual stock scalar fallback.
        ns=dict(Int32=int,Uint32=int,Int64=int,get_ptr_as_int64=lambda a,i:i*4,
                st_global_v4_u32=lambda *x:None)
        stock=extract(STOCK_GATED,'publish_uniform_deferred_tasks',ns)
        publish=extract(DYNAMIC,'publish_ep_local_uniform_tasks',ns)
        owner=types.SimpleNamespace(publish_uniform_deferred_tasks=lambda *a:stock(None,*a))
        ex,rows={},{}
        for tile in range(11):publish(owner,ex,rows,8,4,143,tile,1+tile)
        self.assertEqual(set(ex),set(range(22)))
        for slot in range(22):
            self.assertEqual(ex[slot]&0xffff,143)
            self.assertEqual(ex[slot]>>16,slot//2)
            self.assertEqual((rows[slot]>>8)&255,(slot%2)*4)
            self.assertEqual(rows[slot]>>20,4)

    def test_actual_dynamic_shape_guard_and_sf6_stage_coverage(self):
        check=extract(DYNAMIC,'_check_ep_shard_weights',dict(ep_shard_geometry=geom.ep_shard_geometry))
        index=extract(MODULES/'moe_dynamic_gated_sf6.py','dynamic_sf6_stage_index',{})
        for e,i in ((72,2048),(144,1024)):
            g=geom.ep_shard_geometry(e,i)
            check(None,Tensor(g['cute_w13'],'fp4'),Tensor(g['cute_down'],'fp4'),Tensor((e,),'i32'),sf6=True)
            for kind,rows,k,shape in (('fc1',2*i,4096,g['sf6_fc1']),('fc2',4096,i,g['sf6_fc2'])):
                pairs=set()
                for expert in (0,e-1):
                    for r in range(rows//128):
                        for kk in range(k//128):
                            stage,half=index(kind,rows,k,expert,r,kk)
                            self.assertTrue(expert*shape[1]<=stage<(expert+1)*shape[1])
                            pairs.add((stage,half))
                self.assertEqual(len(pairs),4*shape[1])
        for w,d,r in (((2048,512,8,144),(4096,128,16,144),(144,)),
                      ((4096,512,8,144),(4096,128,8,144),(144,))):
            with self.assertRaises(ValueError):check(None,Tensor(w,'x'),Tensor(d,'x'),Tensor(r,'x'),sf6=True)
        g=geom.ep_shard_geometry(144,1024)
        with self.assertRaises(ValueError):check(None,Tensor(g['cute_w13'],'x'),Tensor(g['cute_down'],'x'),Tensor((144,),'x'),sf6=False)

    def test_actual_launch_shapes_stream_output_and_rejection(self):
        for e,i in ((72,2048),(144,1024)):
            g=geom.ep_shard_geometry(e,i)
            for m in (1,6,8,9,16,32):
                events=[];t=fake_torch(events);dev=types.SimpleNamespace(type='cuda')
                tensor=lambda shape,dtype,**kw:Tensor(shape,dtype,device=dev,events=events,**kw)
                ws=types.SimpleNamespace(state_E=e,weight_E=e,k=4096,n=i,num_topk=8,max_rows=256,
                                        activation_precision='fp4',quant_mode='nvfp4',device=dev)
                for name,shape,dtype in [('row_counts',(e,),t.int32),('token_map',(e,256),t.int32),
                    ('token_weights',(e,256),t.float32),('packed_input',(e,256,2048),t.uint8),
                    ('packed_input_scale',(e,256,256),t.uint8),('barrier_count',(1,),t.int32),
                    ('barrier_epoch',(1,),t.int32),('active_expert_count',(1,),t.int32),
                    ('weight_expert_ids',(e,),t.int32),('global_to_local_expert',(e,),t.int32)]:
                    setattr(ws,name,tensor(shape,dtype))
                ws.packed_a_view=ws.packed_input;ws.packed_a_flat=ws.packed_input.view(-1)
                ws.scale_flat=ws.packed_input_scale.view(-1)
                sf1,sf2=tensor(g['sf6_fc1'],t.uint8),tensor(g['sf6_fc2'],t.uint8)
                w=types.SimpleNamespace(tiled=True,packed_only=True,_w13_sf_storage=None,_down_sf_storage=None,
                    reform_scales=types.SimpleNamespace(enabled=True,fc1=sf1,fc2=sf2),sfb1_packed=sf1,sfb2_packed=sf2,
                    w13_fp4=tensor(g['torch_w13'],t.float4_e2m1fn_x2,strides=g['torch_w13_stride']),
                    down_fp4=tensor(g['torch_down'],t.float4_e2m1fn_x2,strides=g['torch_down_stride']),
                    w1_alpha=tensor((e,),t.float32),w2_alpha=tensor((e,),t.float32))
                scratch=types.SimpleNamespace(scatter_fp32=tensor((32,4096),t.float32),stamps=tensor((48,71),t.int64),
                    counter=tensor((1,),t.int32),dummy_scales=tensor((1,1,16),t.uint8),max_tokens=32,max_active_clusters=48)
                ns=ns_for(NATIVE);opts=[]
                def get(**kw):opts.append(kw);return (lambda *args:events.append(('compiled',args))),48
                ns.update(STAMP_SLOTS=71,REFORM_SF_STAGE=1552,get_ep_tiled_decode_kernel=get)
                launch=extract(NATIVE,'launch_ep_tiled_decode',ns)
                kw=dict(workspace=ws,weights=w,a=tensor((m,4096),t.bfloat16),topk_ids=tensor((m,8),t.int32),
                    topk_weights=tensor((m,8),t.float32),input_gs=tensor((e,),t.float32),down_input_scale=tensor((e,),t.float32),
                    output=tensor((m,4096),t.bfloat16),scratch=scratch,decode_opt=False)
                with patch.dict(sys.modules,{'torch':t}):self.assertIs(launch(**kw),kw['output'])
                self.assertEqual([x[0] for x in events],['record','compiled']+([] if m<=8 else ['copy']))
                self.assertEqual((opts[0]['num_local_experts'],opts[0]['intermediate_size']),(e,i))
                self.assertIs(events[1][1][26],sf1);self.assertIs(events[1][1][27],sf2)
                if e==144:
                    events.clear();opts.clear();w.down_fp4._strides=(1,2,3,4)
                    with patch.dict(sys.modules,{'torch':t}),self.assertRaises(ValueError):launch(**kw)
                    self.assertEqual(events,[]);self.assertEqual(opts,[])

    def test_rank_partition_pairing_and_work_budget_without_runtime_claim(self):
        # Every global expert/intermediate channel belongs to exactly one rank.
        seen=set()
        for rank in range(4):
            ep,sub=rank//2,rank%2
            for local in range(144):
                expert=ep*144+local
                for j in range(1024):
                    key=(expert,sub*1024+j)
                    self.assertNotIn(key,seen);seen.add(key)
        self.assertEqual(len(seen),288*2048)
        # Slice-count bounds only, not estimates of latency or generated tokens.
        for counts in ((0,0,0,0),(1,1,1,1),(6,0,0,0),(8,7,1,0),(1,7,0,8)):
            old=16*max(counts);new=8*max(counts[0]+counts[1],counts[2]+counts[3])
            self.assertLessEqual(new,old);self.assertGreaterEqual(new,old/2)
            self.assertEqual(16*sum(counts),2*8*sum(counts))

    def test_actual_native_getter_cache_and_warm_metadata_match_factory(self):
        build,mods,_=factory(NATIVE);ns=ns_for(NATIVE)
        ns.update(__package__='private_pkg',_EP_TILED_KERNEL_CACHE={})
        jit=types.ModuleType('flashinfer.jit.cute_dsl_core')
        jit.build_and_load_cute_dsl_kernel=lambda *a,**k: self.fail('unexpected lowering')
        pkg=types.ModuleType('private_pkg');pkg.__path__=[]
        pkg.moe_dispatch=types.ModuleType('private_pkg.moe_dispatch')
        mods.update({'private_pkg':pkg,'private_pkg.moe_dispatch':pkg.moe_dispatch,
                     'flashinfer.jit.cute_dsl_core':jit})
        get=extract(NATIVE,'get_ep_tiled_decode_kernel',ns)
        with patch.dict(sys.modules,mods):
            for e,i in ((72,2048),(144,1024)):
                for m in (1,6,8,12,32):
                    kw=dict(num_tokens=m,num_local_experts=e,intermediate_size=i,
                        reform_sf_pack=True,decode_opt=False,route_mode='global',
                        expert_map_len=288,expert_map_dtype='int32')
                    _,_,key=build(**kw);cached=object();ns['_EP_TILED_KERNEL_CACHE'][key]=cached
                    self.assertIs(get(**kw)[0],cached)
        calls=[];ns['get_ep_tiled_decode_kernel']=lambda **kw:calls.append(kw)
        warm=extract(NATIVE,'warm_ep_tiled_decode',ns)
        warm(num_local_experts=144,intermediate_size=1024,reform_sf_pack=True,decode_opt=False)
        self.assertEqual(len(calls),32)
        self.assertTrue(all((k['num_local_experts'],k['intermediate_size'],k['decode_opt'])==(144,1024,False) for k in calls))

    def test_dynamic_selector_and_cache_preserve_old_geometry(self):
        ns=dict(__package__='private_pkg',_GLM53_EP_TILED=True,_GLM53_EP_PREFILL_LOCAL=False,
                _FORCED_BACKEND=None,_FORCE_MOE_W4A16_ENV='IGNORED',os=types.SimpleNamespace(environ={}),
                torch=types.SimpleNamespace(cuda=types.SimpleNamespace(get_device_capability=lambda:(12,1))),
                select_sm120_moe_backend=lambda **kw:'dynamic')
        pkg=types.ModuleType('private_pkg');pkg.__path__=[]
        module=types.ModuleType('private_pkg.moe_dynamic_ep_local')
        module.MoEGatedEPLocalKernel=object();module.stock_contract_matches=lambda:True
        path=MODULES/'moe_dispatch.py'
        select=extract(path,'_ep_local_prefill_kernel',ns)
        with patch.dict(sys.modules,{'private_pkg':pkg,'private_pkg.moe_dynamic_ep_local':module}):
            for e,i in ((72,2048),(144,1024)):
                for m in (33,4096,8192,16384):
                    self.assertIs(select(E=e,m=m,k=4096,n=i,num_topk=8,tile_m=128,
                        activation='swigluoai_uninterleave',swiglu_alpha=1.,swiglu_beta=0.,swiglu_limit=10.,
                        quant_mode='nvfp4',tiled=True),module.MoEGatedEPLocalKernel)
        old=extract(BASE_DISPATCH,'_dynamic_kernel_cache_key',{})
        new=extract(path,'_dynamic_kernel_cache_key',{})
        kw=dict(activation_precision='fp4',quant_mode='nvfp4',E=72,k=4096,n=2048,
                num_topk=8,mac=48,mma_tiler_mn=(128,128),topk_ids_dtype='int32',
                input_scales_are_reciprocal=False,fast_math=True,activation='swigluoai_uninterleave',
                swiglu_alpha=1.,swiglu_beta=0.,swiglu_limit=10.,share_input_across_experts=False,
                ep_local_prefill=True,tiled=True,reform_sf_pack=True)
        self.assertEqual(old(**kw),new(**kw))
        self.assertEqual(new(**dict(kw,E=144,n=1024))[-1],geom.HYBRID_TAG)
        self.assertIn('glm53_ep_shard_geometry.py',ast.unparse(node(path,'_get_dynamic_kernel')))

if __name__=='__main__':unittest.main(verbosity=2)
