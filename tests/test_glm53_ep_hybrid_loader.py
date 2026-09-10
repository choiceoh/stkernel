"""Hybrid loader host contract tests; actual pinned AST + sparse tensor mocks.
No torch/CUDA/vLLM import, model allocation, or numeric-kernel claim.
"""
import ast
import copy
from dataclasses import dataclass
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace as NS, ModuleType
from unittest import TestCase, main
from unittest.mock import patch

REPO=Path(__file__).resolve().parents[1]
FIXTURE=REPO/'tests/fixtures/glm53_ep_hybrid/runtime'
MODULE=REPO/'overlay/modules/glm53_model/glm53_ep_hybrid.py'
spec=importlib.util.spec_from_file_location('_glm53_hybrid_loader_contract',MODULE)
hybrid=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=hybrid
spec.loader.exec_module(hybrid)
HybridShard=hybrid.HybridShard
SimpleNamespace=NS
BASE_MODULE='vllm.model_executor.layers.fused_moe.routed_experts'
QUANT_MODULE='vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors'
METHOD_MODULE='vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_nvfp4'

@dataclass
class PC:
    tp_size:int=1;tp_rank:int=0;ep_size:int=4;ep_rank:int=0
    dp_size:int=1;pcp_size:int=1;sp_size:int=1
    use_ep:bool=True;enable_eplb:bool=False;use_all2all_kernels:bool=False

class Manager:
    placement_strategy='linear'
    def __init__(self,p):self.moe_parallel_config=p;self.local_num_experts=72
    def update(self,p,e):
        assert e==288
        self.moe_parallel_config=p;self.local_num_experts=144

SHAPES={
 'w13_weight_packed':(144,2048,2048),'w2_weight_packed':(144,4096,512),
 'w13_weight_scale':(144,2048,256),'w2_weight_scale':(144,4096,64),
 'w13_weight_global_scale':(144,2),'w2_weight_global_scale':(144,),
 'w13_input_global_scale':(144,2),'w2_input_global_scale':(144,),
}

class HybridLoaderTests(TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.runtime=Path(self.temp.name).resolve()/'vllm';self.runtime.mkdir()
        self.pins=json.loads((FIXTURE/'source_pins.json').read_text())
        self.assertEqual(self.pins,hybrid._RUNTIME_SOURCE_PINS)
        for rel,sha in self.pins.items():
            data=gzip.decompress((FIXTURE/(rel+'.gz')).read_bytes())
            self.assertEqual(hashlib.sha256(data).hexdigest(),sha)
            dest=self.runtime/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
        self.modules={}
        for name,rel in [(BASE_MODULE,'model_executor/layers/fused_moe/routed_experts.py'),(QUANT_MODULE,'model_executor/layers/quantization/compressed_tensors/compressed_tensors.py')]:
            mod=ModuleType(name);mod.__file__=str(self.runtime/rel);self.modules[name]=mod
        self.patch=patch.dict(sys.modules,self.modules);self.patch.start();self.addCleanup(self.patch.stop)
        hybrid._RUNTIME_SOURCE_CACHE.clear()
        self.config=NS(hidden_size=4096,n_routed_experts=288,moe_intermediate_size=2048,num_experts_per_token=8,hidden_act='silu',swiglu_limit=10)
        self.parallel=NS(enable_expert_parallel=True,enable_eplb=False,use_sequence_parallel_moe=False,expert_placement_strategy='linear',tensor_parallel_size=4,pipeline_parallel_size=1,data_parallel_size=1,prefill_context_parallel_size=1,decode_context_parallel_size=1,eplb_config=NS(num_redundant_experts=0))
        self.quant=type('CompressedTensorsConfig',(),{'__module__':QUANT_MODULE,'get_name':lambda s:'compressed-tensors'})()
        self.env={'VLLM_GLM53_EP_HYBRID_TP2':'1','VLLM_GLM53_EP_TILED':'1','VLLM_GLM53_B12X_STATIC_V2':'t,r,sf6'}
        self.allocations=[]
        self.method=type('CompressedTensorsW4A4Nvfp4MoEMethod',(),{'__module__':METHOD_MODULE,'group_size':16,'use_global_sf':True,'nvfp4_backend':NS(name='FLASHINFER_B12X')})()
        test=self
        class RoutedExperts:
            def _get_quant_method(self,*args):return test.method
            def __init__(self,n,d,c,q,expert_map_manager,**kw):
                assert c.moe_parallel_config is expert_map_manager.moe_parallel_config
                self.quant_method=self._get_quant_method(n,q,c)
                test.allocations.append((c.num_local_experts,c.intermediate_size_per_partition))
                for key,shape in SHAPES.items():setattr(self,key,NS(shape=shape))
        RoutedExperts.__module__=BASE_MODULE
        self.base=RoutedExperts
    def kwargs(self,rank=0,**kw):
        return hybrid.hybrid_factory_kwargs(self.config,self.parallel,self.quant,physical_tp_rank=rank,physical_tp_size=4,env=self.env,base_cls=self.base,**kw)
    def construct(self,rank=0):
        kw=self.kwargs(rank)
        original=PC(ep_rank=rank);manager=Manager(original)
        mc=NS(moe_parallel_config=original,num_experts=288,num_local_experts=72,hidden_dim=4096,intermediate_size=2048,intermediate_size_per_partition=2048,intermediate_size_per_partition_unpadded=2048,experts_per_token=8,has_bias=False,is_lora_enabled=False,skip_final_all_reduce=False,moe_backend='flashinfer_b12x',in_dtype='torch.bfloat16',is_act_and_mul=True)
        result=kw['routed_experts_cls']('model.layers.3.mlp.experts','bf16',mc,self.quant,manager,**kw['routed_experts_args'])
        return original,mc,result
    def test_flag_zero_has_no_binding_or_io(self):
        self.assertEqual(hybrid.hybrid_factory_kwargs(None,None,None,physical_tp_rank=None,physical_tp_size=None,env={}),{})
        self.assertEqual(hybrid._RUNTIME_SOURCE_CACHE,{})
    def test_exact_four_ranks_allocation_hook_and_immutable_identity(self):
        for rank in range(4):
            original,mc,result=self.construct(rank)
            self.assertEqual((original.tp_size,original.ep_size),(1,4))
            self.assertEqual((mc.moe_parallel_config.tp_rank,mc.moe_parallel_config.ep_rank),(rank%2,rank//2))
            hybrid.validate_hybrid_loader_identity(mc._glm53_hybrid_loader_identity,rank)
            self.assertEqual(len(mc._glm53_hybrid_runtime_sources),15)
            for field in range(1,13):
                bad=list(mc._glm53_hybrid_loader_identity);bad[field]+=1
                with self.subTest(rank=rank,field=field),self.assertRaises(ValueError):hybrid.validate_hybrid_loader_identity(tuple(bad),rank)
        self.assertEqual(self.allocations,[(144,1024)]*4)
        # Actual base has quant selection before size rounding and allocation.
        tree=ast.parse((self.runtime/'model_executor/layers/fused_moe/routed_experts.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RoutedExperts')
        init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
        calls={n.func.attr:n.lineno for n in ast.walk(init) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in {'_get_quant_method','maybe_roundup_sizes','create_weights'}}
        self.assertLess(calls['_get_quant_method'],calls['maybe_roundup_sizes']);self.assertLess(calls['maybe_roundup_sizes'],calls['create_weights'])
    def test_exact_prebinding_gates_reject(self):
        mutations=[(self.env,'VLLM_GLM53_EP_HYBRID_TP2','true'),(self.env,'VLLM_GLM53_EP_TILED','0'),(self.env,'VLLM_GLM53_EP_DECODE_OPT','1'),(self.env,'VLLM_GLM53_TP_SF6_Q0','1'),(self.env,'VLLM_GLM53_B12X_STATIC_V2','t,r'),(self.config,'moe_intermediate_size',1024),(self.parallel,'enable_expert_parallel',False),(self.parallel,'use_sequence_parallel_moe',True),(self.parallel,'tensor_parallel_size',2),(self.parallel,'enable_eplb',True),(self.quant,'get_name',lambda:'modelopt_fp4')]
        for obj,key,value in mutations:
            with self.subTest(key=key):
                if isinstance(obj,dict):
                    with patch.dict(obj,{key:value}),self.assertRaises(ValueError):self.kwargs()
                else:
                    with patch.object(obj,key,value),self.assertRaises(ValueError):self.kwargs()
        with self.assertRaises(ValueError):self.kwargs(main_layer=False)
        self.assertEqual(self.allocations,[])
    def test_actual_selected_ct_class_backend_guard_precedes_allocation(self):
        for field,value in [('group_size',32),('use_global_sf',False),('nvfp4_backend',NS(name='FLASHINFER_CUTLASS'))]:
            with self.subTest(field=field),patch.object(self.method,field,value),self.assertRaises(RuntimeError):self.construct()
        for field,value in [('__name__','ModelOptNvFp4FusedMoE'),('__module__','foreign.module')]:
            cls=type(self.method);old=getattr(cls,field)
            try:
                setattr(cls,field,value)
                with self.subTest(field=field),self.assertRaises(RuntimeError):self.construct()
            finally:setattr(cls,field,old)
        self.assertEqual(self.allocations,[])
    def test_runtime_source_pins_real_lookup_and_receipt_copy(self):
        actual=hybrid.hybrid_runtime_sources(self.base,self.quant)
        self.assertEqual(len(actual),15)
        for rel,entry in actual.items():self.assertEqual(entry,{'path':str(self.runtime/rel),'sha256':self.pins[rel]})
        actual[next(iter(actual))]['sha256']='poison'
        self.assertNotIn('poison',str(hybrid.hybrid_runtime_sources(self.base,self.quant)))
    def test_runtime_pins_fail_closed_on_source_location_and_class(self):
        rel='model_executor/layers/fused_moe/config.py';path=self.runtime/rel;original=path.read_bytes()
        path.write_bytes(original+b'\n# changed')
        with self.assertRaises(RuntimeError):hybrid.hybrid_runtime_sources(self.base,self.quant)
        path.unlink()
        with self.assertRaises(FileNotFoundError):hybrid.hybrid_runtime_sources(self.base,self.quant)
        outside=Path(self.temp.name)/'redirect.py';outside.write_bytes(original);path.symlink_to(outside)
        with self.assertRaises(RuntimeError):hybrid.hybrid_runtime_sources(self.base,self.quant)
        path.unlink();path.write_bytes(original)
        with patch.object(self.base,'__module__','foreign'),self.assertRaises(RuntimeError):hybrid.hybrid_runtime_sources(self.base,self.quant)
        with patch.object(self.modules[QUANT_MODULE],'__file__',str(outside)),self.assertRaises(RuntimeError):hybrid.hybrid_runtime_sources(self.base,self.quant)
    def test_actual_gate_up_down_loader_slices(self):
        _check_raw_loader(self.runtime)
    def test_actual_ct_q0_alpha_and_single_shared_routed_sum(self):
        _check_ct_contract(self.runtime,REPO)


def _check_raw_loader(runtime):
    source=(runtime/'model_executor/layers/fused_moe/routed_experts.py').read_text()
    root=ast.parse(source)
    cls=next(n for n in root.body if isinstance(n,ast.ClassDef) and n.name=='RoutedExperts')
    defs=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('_load_w13','_load_w2')]
    ns={};exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+defs,type_ignores=[])),source,'exec'),ns)
    class View:
        def __init__(self,shape,start=None,root=None):
            self.shape=tuple(shape);self.ndim=len(shape);self.start=tuple(start or [0]*len(shape));self.root=root or self;self.copies=[]
        def narrow(self,dim,start,length):
            assert 0<=start<=self.shape[dim] and 0<=length<=self.shape[dim]-start
            shape=list(self.shape);shape[dim]=length;s=list(self.start);s[dim]+=start
            return View(shape,s,self.root)
        def copy_(self,other):
            assert self.shape==other.shape
            self.root.copies.append((self.start,other.start,self.shape))
    class Loader:
        _load_w13=ns['_load_w13'];_load_w2=ns['_load_w2']
        def __init__(self):self.moe_config=SimpleNamespace(is_act_and_mul=True,moe_parallel_config=SimpleNamespace(tp_size=2))
        def _get_hidden_dim(self,shard_dim,ndim):return 1-shard_dim
        def _narrow_expert_data_for_padding(self,a,b,**kw):assert a.shape==b.shape;return a
    for rank in range(4):
        s=HybridShard.for_rank(rank);load=Loader()
        # FP4 weight bytes and raw per-16 block scales both use the same loader.
        for k in (2048,256):
            target=View((2048,k));full=View((2048,k))
            load._load_w13(target,0,'w1',full,s.tp_rank)
            load._load_w13(target,0,'w3',full,s.tp_rank)
            assert target.copies==[((0,0),(s.intermediate_start,0),(1024,k)),((1024,0),(s.intermediate_start,0),(1024,k))]
        for divide in (2,16):
            target=View((4096,1024//divide));full=View((4096,2048//divide))
            load._load_w2(target,1,full,s.tp_rank)
            assert target.copies==[((0,0),(0,s.intermediate_start//divide),(4096,1024//divide))]
    # Every (expert, I index) is owned once, while each expert has two TP halves.
    for e in range(288):
        owners=[s for s in map(HybridShard.for_rank,range(4)) if s.expert_start<=e<s.expert_stop]
        assert [(s.intermediate_start,s.intermediate_stop) for s in owners]==[(0,1024),(1024,2048)]


def _check_ct_contract(runtime,repo):
    BASE=DEP=CON=runtime / 'model_executor/layers'
    CT=BASE/'quantization/compressed_tensors'
    WT=repo
    def extract(path,cls,names,ns,strip_local_imports=False):
        tree=ast.parse(path.read_text());nodes=tree.body
        if cls:nodes=next(n for n in nodes if isinstance(n,ast.ClassDef) and n.name==cls).body
        defs=list({n.name:copy.deepcopy(n) for n in nodes if isinstance(n,ast.FunctionDef) and n.name in names}.values())
        for n in defs:
            n.decorator_list=[]
            if strip_local_imports:n.body=[x for x in n.body if not isinstance(x,(ast.Import,ast.ImportFrom))]
        exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+defs,type_ignores=[])),str(path),'exec'),ns)

    class Scalar:
        shape=();ndim=0
        def __init__(self,x):self.value=x
        def to(self,*a):return self
        def reshape(self,s):assert s==();return self
        def repeat(self,n):return Array((n,),values={(i,):self.value for i in range(n)})
    class Array:
        device='cpu'
        def __init__(self,shape,values=None,prefix=(),dtype='f32',expr=None):
            self.shape=tuple(shape);self.ndim=len(shape);self.values=values if values is not None else {};self.prefix=prefix;self.dtype=dtype;self.expr=expr or ('array',self.shape)
        def __getitem__(self,i):
            if isinstance(i,tuple):
                assert i[0]==slice(None) and type(i[1]) is int and self.ndim==2
                return Array((self.shape[0],),{(r,):self.values[(r,i[1])] for r in range(self.shape[0])},dtype=self.dtype)
            if self.ndim>1:return Array(self.shape[1:],self.values,self.prefix+(i,),dtype=self.dtype)
            return self.values.get(self.prefix+(i,),0)
        def __setitem__(self,i,v):self.values[self.prefix+(i,)]=v.value if isinstance(v,Scalar) else v
        def size(self,i):return self.shape[i]
        def contiguous(self):return self
        def float(self):return self
        def to(self,dtype):return Array(self.shape,dict(self.values),dtype=dtype,expr=self.expr)
        def max(self):return Scalar(max(self.values.values()))
        def view(self,*shape):
            assert shape==(-1,1,1) and self.ndim==1
            return Array((self.shape[0],1,1),dict(self.values),expr=('view',self.expr))
        def fill_(self,x):
            assert self.ndim==1
            self.values.clear();self.values.update({(i,):x for i in range(self.shape[0])});return self
        def __rtruediv__(self,x):return Array(self.shape,{k:x/v for k,v in self.values.items()},dtype=self.dtype,expr=('div',x,self.expr))
        def __mul__(self,other):
            other=other.data if isinstance(other,Param) else other
            assert other.shape==(self.shape[0],1,1)
            return Array(self.shape,dtype=self.dtype,expr=('bake',self.expr,tuple(sorted(other.values.items()))))
    class Param:
        def __init__(self,data,**kw):self.data=data
        def __getattr__(self,n):return getattr(self.data,n)
        def __getitem__(self,i):return self.data[i]
        def __rtruediv__(self,x):return x/self.data
    class Torch:
        uint8='u8';float8_e4m3fn='e4m3';float32='f32';nn=NS(Parameter=Param)
        @staticmethod
        def empty(*shape,**kw):return Array(shape,dtype=kw.get('dtype','f32'))
        @staticmethod
        def allclose(a,b):return a.values==b.values

    def setattrs(p,d):p.__dict__.update(d)
    Enum=NS(**{k:NS(value=k.lower()) for k in ('BLOCK','GROUP','TENSOR','CHANNEL')})
    ns=dict(torch=Torch,FusedMoeWeightScaleSupported=Enum,set_weight_attrs=setattrs,replace_parameter=lambda l,n,v:setattr(l,n,Param(v)),logger=NS(warning_once=lambda *a:None))
    ctfile=CT/'compressed_tensors_moe/compressed_tensors_moe_w4a4_nvfp4.py'
    extract(ctfile,'CompressedTensorsW4A4Nvfp4MoEMethod',{'create_weights','process_weights_after_loading','get_fused_moe_quant_config'},ns)
    extract(BASE/'fused_moe/routed_experts.py','RoutedExperts',{'weight_loader','_to_scalar','_load_single_value','_load_per_tensor_weight_scale'},ns)
    class Loader:
        weight_loader=ns['weight_loader'];_to_scalar=staticmethod(ns['_to_scalar']);_load_single_value=ns['_load_single_value'];_load_per_tensor_weight_scale=ns['_load_per_tensor_weight_scale']
        quant_config=NS(get_name=lambda:'compressed-tensors')
        quant_method=type('CompressedTensorsW4A4Nvfp4MoEMethod',(),{'use_global_sf':True})()
        def __init__(self,start,E):self.start=start;self.E=E
        def _map_global_expert_id_to_local_expert_id(self,e):return e-self.start if self.start<=e<self.start+self.E else -1
    backend=NS(**{k:k for k in ['VLLM_CUTLASS','FLASHINFER_CUTLASS','FLASHINFER_TRTLLM','FLASHINFER_CUTEDSL_BATCHED','FLASHINFER_CUTEDSL','FLASHINFER_B12X','HUMMING','MARLIN','EMULATION']})
    reorders=[]
    ns.update(NvFp4MoeBackend=backend,FLASHINFER_NVFP4_MOE_BACKENDS=[backend.FLASHINFER_B12X],is_global_sf_supported_for_nvfp4_backend=lambda b:b==backend.FLASHINFER_B12X,
     swizzle_blockscale=lambda a:a,reorder_w1w3_to_w3w1=lambda w,s:(reorders.append((w.shape,s.shape)) or (w,s)),nvfp4_moe_quant_config=lambda **kw:NS(**kw))
    extract(CON/'quantization/utils/quant_utils.py',None,{'amax_for_moe_activation_quant'},ns)
    extract(DEP/'quantization/utils/flashinfer_fp4_moe.py',None,{'prepare_nvfp4_moe_layer_for_fi_or_cutlass'},ns,True)
    extract(DEP/'fused_moe/oracle/nvfp4.py',None,{'convert_to_nvfp4_moe_kernel_format','make_nvfp4_moe_quant_config'},ns)
    extract(CON/'fused_moe/modular_kernel.py','FusedMoEExperts',{'g1_alphas','g2_alphas','a1_gscale','a2_gscale'},ns)
    # Execute the exact active wrapper's normalization statements, excluding owner/relayout.
    wrapper=WT/'overlay/modules/glm53_moe/flashinfer_b12x_moe.py'
    tree=ast.parse(wrapper.read_text())
    method=next(n for c in tree.body if isinstance(c,ast.ClassDef) for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='process_weights_after_loading' and any('w13_weight_scale.data' in ast.unparse(x) for x in n.body))
    start=next(i for i,n in enumerate(method.body) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='layer.w13_weight_scale.data')
    end=next(i for i,n in enumerate(method.body[start:],start) if isinstance(n,ast.If) and ast.unparse(n.test)=='self._use_ep')
    norm=ast.FunctionDef(name='normalize',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='self'),ast.arg(arg='layer')],kwonlyargs=[],kw_defaults=[],defaults=[]),body=copy.deepcopy(method.body[start:end]),decorator_list=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[norm],type_ignores=[])),str(wrapper),'exec'),ns)
    class Owner:
        g1_alphas=property(ns['g1_alphas']);g2_alphas=property(ns['g2_alphas']);a1_gscale=property(ns['a1_gscale']);a2_gscale=property(ns['a2_gscale'])
        def __init__(self,q):self.quant_config=q
        def process_weights_after_loading(self,layer):
            self.pre_a1=tuple(layer.w13_input_scale.values.values())
            ns['normalize'](self,layer)
            self.after=(tuple(self.g1_alphas.values.values()),tuple(self.g2_alphas.values.values()),tuple(self._fc2_input_scale.values.values()))
            assert self.g1_alphas is layer.w13_weight_scale_2
            assert self.g2_alphas is layer.w2_weight_scale_2
            assert all(set(v)=={1.0} for v in self.after)
    owners=[]
    def makekernel(**kw):
        owner=Owner(kw['moe_quant_config']);owners.append(owner);return NS(fused_experts=owner)
    ns['make_nvfp4_moe_kernel']=makekernel
    results=[]
    for E,I,start_e in [(72,2048,0),(72,2048,72),(144,1024,0),(144,1024,144)]:
        for variant in (1,11):
            loader=Loader(start_e,E);layer=NS(activation=NS(is_gated=True),moe_config=NS(moe_parallel_config=NS(enable_eplb=False)),_expert_routing_tables=lambda:None)
            layer.register_parameter=lambda n,p:setattr(layer,n,p)
            q=NS(group_size=16,moe=NS(is_act_and_mul=True),use_global_sf=True,nvfp4_backend=backend.FLASHINFER_B12X,experts_cls=object)
            q.get_fused_moe_quant_config=lambda l:ns['get_fused_moe_quant_config'](q,l)
            ns['create_weights'](q,layer,E,4096,I,'bf16',weight_loader=loader.weight_loader,global_num_experts=288)
            assert layer.w13_input_global_scale.shape==(E,2) and layer.w2_input_global_scale.shape==(E,)
            for e in range(288):
                for sid,val in [('w1',1/(variant*(e+1))),('w3',1/(variant*(e+.5)))]:
                    loaded=loader.weight_loader(layer.w13_input_global_scale,Scalar(val),'w13_input_global_scale',sid,e,True)
                    assert loaded==(start_e<=e<start_e+E)
                    loader.weight_loader(layer.w13_weight_global_scale,Scalar(e+2),'w13_weight_global_scale',sid,e)
                loader.weight_loader(layer.w2_input_global_scale,Scalar(1/(variant*(2*e+1))),'w2_input_global_scale','w2',e)
                loader.weight_loader(layer.w2_weight_global_scale,Scalar(e+3),'w2_weight_global_scale','w2',e)
            assert len(layer.w13_input_global_scale.values)==2*E and len(layer.w2_input_global_scale.values)==E
            ns['process_weights_after_loading'](q,layer)
            owner=owners[-1]
            assert len(set(owner.pre_a1))==1 and abs(owner.pre_a1[0]-variant*(start_e+E))<1e-8, (E,variant,owner.pre_a1[0])
            result={'E':E,'I':I,'start':start_e,'input_variant':variant,'a1_reciprocal_amax':owner.pre_a1[0], 'a1_gscale':next(iter(owner.a1_gscale.values.values())), 'Q0_input_gs':1.0,'FC1_weight_alpha':1.0,'FC2_input_gs':1.0,'FC2_weight_alpha':1.0,'w13_bake':repr(layer.w13_weight_scale.expr),'w2_bake':repr(layer.w2_weight_scale.expr)}
            if variant==11:
                prior=results[-1]
                assert result['a1_gscale']!=prior['a1_gscale']
                assert result['w13_bake']==prior['w13_bake'] and result['w2_bake']==prior['w2_bake']
            results.append(result)
    # Verify actual native/dynamic owner callsites retain g1, never a1. This is an
    # ABI selection check; no symbolic CUDA operation is substituted for numerics.
    ownerfile=WT/'overlay/modules/glm53_moe/glm53_ep_tiled.py'
    calls=[n for n in ast.walk(ast.parse(ownerfile.read_text())) if isinstance(n,ast.Call) and any(k.arg=='input_gs' for k in n.keywords)]
    assert len(calls)==2
    assert all(next(ast.unparse(k.value) for k in n.keywords if k.arg=='input_gs')=='owner.g1_alphas' for n in calls)
    assert all(next(ast.unparse(k.value) for k in n.keywords if k.arg=='down_input_scale')=='owner._fc2_input_scale' for n in calls)
    # Actual CT matcher uses case-insensitive class substring, so subclass name works.
    import re
    ns.update(re=re,MappingProxyType=dict)
    extract(CON/'quantization/compressed_tensors/utils.py',None,{'_is_equal_or_regex_match','_find_first_match','find_matched_target'},ns)
    ns['_match_fused_layer']=lambda *a:None
    ns['MappingProxyType']=dict
    assert ns['find_matched_target']('model.layers.3.mlp.experts.0.gate_proj',type('HybridRoutedExperts',(),{})(),['RoutedExperts'])=='RoutedExperts'
    assert ns['find_matched_target']('model.layers.3.mlp.experts.0.gate_proj',object(),['RoutedExperts']) is None
    # Execute the existing runner's exact selection methods. There is no routed
    # output transform and no early-reduced fused output. Each physical rank owns
    # its original TP4 shared-output quarter, not a replica within its EP pair.
    calls=[]
    class Terms(dict):
        def __add__(self,other):
            result=Terms(self)
            for k,v in other.items():result[k]=result.get(k,0)+v
            return result

    def global_tp4_sum(x):calls.append(x);return x
    ns={'tensor_model_parallel_all_reduce':global_tp4_sum}
    extract(BASE/'fused_moe/runner/moe_runner.py','MoERunner',{'_maybe_reduce_shared_expert_output','_maybe_reduce_routed_output_before_transform','_maybe_reduce_final_output'},ns)
    partials=[]
    for rank in range(4):
        runner=NS(moe_config=NS(is_sequence_parallel=False,skip_final_all_reduce=False,tp_size=2,ep_size=2),routed_output_transform=None)
        shared=Terms({('shared_TP4_quarter',rank):1});fused=Terms({('routed_expert_set',rank//2,'Ihalf',rank%2):1})
        fused,reduced=ns['_maybe_reduce_routed_output_before_transform'](runner,fused,False)
        shared=ns['_maybe_reduce_shared_expert_output'](runner,shared,reduced)
        ns['_maybe_reduce_final_output'](runner,shared+fused,None,reduced)
        partials.append(shared+fused)
    assert len(calls)==4 # one collective call per rank, not four sequential sums
    combined=Terms()
    for x in partials:combined=combined+x
    assert len(combined)==8 and set(combined.values())=={1}


if __name__=='__main__':
    main()
