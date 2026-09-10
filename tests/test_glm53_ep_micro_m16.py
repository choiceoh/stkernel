"""No-device M16 geometry/routing oracles; real CuTe and GPU remain gates."""
import ast
import copy
import gzip
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import runpy
import struct
from types import SimpleNamespace
import unittest

ROOT=Path(__file__).resolve().parents[1]
MICRO=ROOT/'overlay/modules/glm53_moe/moe_micro_kernel.py'
ORACLE=ROOT/'measurements/glm53_ep_local_20260908/micro-stock-oracle'
STOCK_SHA='68760d0bbab7aa761d880d03c18314ad8adf18fc3a955a6f893e42cec96b0c68'


def function(name,raw=None):
    return next(n for n in ast.walk(ast.parse(raw or MICRO.read_text()))
                if isinstance(n,ast.FunctionDef) and n.name==name)


def extract(fn,namespace):
    fn=copy.deepcopy(fn);fn.decorator_list=[]
    module=ast.Module(body=[ast.ImportFrom('__future__',[ast.alias('annotations')],0),fn],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),str(MICRO),'exec'),namespace)
    return namespace[fn.name]


def code(nodes):
    return compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes),type_ignores=[])),str(MICRO),'exec')


@lru_cache(maxsize=1)
def initializer():
    ns=dict(cutlass=SimpleNamespace(Float32='f32'),DenseGemmKernel=object,
            is_gated_activation=lambda _:True,
            utils=SimpleNamespace(get_smem_capacity_in_bytes=lambda _:101376),
            pipeline=SimpleNamespace(NamedBarrier=lambda **kw:kw))
    return extract(function('__init__'),ns)


def constructor(**override):
    obj=SimpleNamespace()
    kwargs=dict(sf_vec_size=16,mma_tiler_mn=(16,128),output_tile_count_n=16,
                activation='swigluoai_uninterleave',swiglu_alpha=1.,swiglu_beta=0.,swiglu_limit=10.,
                skip_zero_weight_expert_id=72,scatter_fp32=True,ep_direct_scatter=True,
                shared_fc1_a=True,ep_m16=True)
    initializer()(obj,**{**kwargs,**override})
    return obj


def points(tid):
    # Independent m16n128 warp-M1/N2 logical ownership candidate. Actual CuTe
    # partition_D must independently pass the source host validator at compile.
    lane,warp=tid%32,tid//32
    values=[]
    for pair in range(16):
        row=(lane>>2)+8*(pair&1);k=pair>>1
        col=2*(lane&3)+16*warp+8*(k&1)+32*((k>>1)&1)+64*((k>>2)&1)
        values.extend(((row,col,0),(row,col+1,0)))
    return values


def bf16(x):
    bits=struct.unpack('<I',struct.pack('<f',x))[0]
    word=((bits+0x7fff+((bits>>16)&1))>>16)&0xffff
    return struct.unpack('<f',struct.pack('<I',word<<16))[0]


class M16Tests(unittest.TestCase):
    def test_false_entire_source_matches_cpu19_oracle(self):
        raw=gzip.decompress((ORACLE/'moe_micro_kernel_cpu19.py.gz').read_bytes())
        meta=json.loads((ORACLE/'micro-kernel-cpu19-identity.json').read_text())
        self.assertEqual(hashlib.sha256(raw).hexdigest(),STOCK_SHA)
        self.assertEqual(meta['source_sha256'],STOCK_SHA)
        self.assertEqual(meta['source_revision'],'cd293e19c146bd52c3419b064af209b57b2555d9')
        self.assertEqual(hashlib.sha256((ORACLE/'moe_micro_kernel_cpu19.py.gz').read_bytes()).hexdigest(),meta['gzip_sha256'])
        class FalseM16(ast.NodeTransformer):
            def visit_If(self,n):
                if ast.unparse(n.test)=='self.ep_m16':return []
                if ast.unparse(n.test)=='cutlass.const_expr(not self.ep_m16)':
                    return [self.visit(x) for x in n.body]
                return self.generic_visit(n)
            def visit_IfExp(self,n):
                return self.visit(n.orelse) if ast.unparse(n.test)=='self.ep_m16' else self.generic_visit(n)
            def visit_Assign(self,n):
                if ast.unparse(n.targets[0])=='self.ep_m16':
                    assert ast.unparse(n.value)=='bool(ep_m16)'
                    return None
                return self.generic_visit(n)
            def visit_FunctionDef(self,n):
                if n.name=='__init__':
                    idx=next(i for i,a in enumerate(n.args.kwonlyargs) if a.arg=='ep_m16')
                    assert isinstance(n.args.kw_defaults[idx],ast.Constant) and n.args.kw_defaults[idx].value is False
                    del n.args.kwonlyargs[idx];del n.args.kw_defaults[idx]
                return self.generic_visit(n)
        self.assertEqual(ast.dump(FalseM16().visit(ast.parse(MICRO.read_text()))),ast.dump(ast.parse(raw)))
        # All three M16 warps share one warpgroup, so neither role may issue a
        # different warpgroup-collective setmaxnreg. The original full-module
        # comparison above retains both original calls when the flag is false.
        class TrueM16(ast.NodeTransformer):
            def visit_If(self,n):
                if ast.unparse(n.test)=='cutlass.const_expr(not self.ep_m16)':
                    return [self.visit(x) for x in n.orelse]
                return self.generic_visit(n)
        def register_calls(tree):
            return [ast.unparse(n) for n in ast.walk(tree) if isinstance(n,ast.Call)
                    and isinstance(n.func,ast.Attribute) and n.func.attr.startswith('setmaxregister_')]
        self.assertEqual(register_calls(ast.parse(raw)),[
            'cute.arch.setmaxregister_increase(self.mma_register_requirement)',
            'cute.arch.setmaxregister_decrease(self.load_register_requirement)'])
        self.assertEqual(register_calls(TrueM16().visit(ast.parse(MICRO.read_text()))),[])

    def test_constructor_narrow_admission_thread_barriers_and_physical_rows(self):
        obj=constructor()
        self.assertEqual(obj.tile_shape_mnk,(16,128,128))
        self.assertEqual(obj.epi_tile,(16,128))
        self.assertEqual((obj.num_mma_warps,obj.tma_load_warp_id,obj.threads_per_cta),(2,2,96))
        self.assertEqual(obj.epilog_sync_barrier,dict(barrier_id=1,num_threads=64))
        self.assertEqual(obj.pass_sync_barrier,dict(barrier_id=2,num_threads=96))
        self.assertEqual(obj.sa_tile_shape_mk,obj.sfa_tile_shape_mk)
        self.assertEqual(obj.sa_tile_shape_mk,(128,128))
        self.assertEqual((obj.sa_tiles_per_block,obj.sfa_tiles_per_block),(8,8))
        old=constructor(ep_m16=False,mma_tiler_mn=(32,128))
        self.assertEqual((old.num_mma_warps,old.threads_per_cta),(4,160))
        for key,bad in [('ep_m16',False),('shared_fc1_a',False),('ep_direct_scatter',False),
                        ('scatter_fp32',False),('mma_tiler_mn',(32,128)),('mma_tiler_mn',(16,256)),
                        ('sf_vec_size',32),('output_tile_count_n',8),('single_token',True),
                        ('share_input_across_experts',True),('share_expert_scales',True),
                        ('skip_zero_weight_expert_id',None),('activation','silu'),
                        ('swiglu_alpha',1.702),('swiglu_beta',1.),('swiglu_limit',9.)]:
            with self.subTest(key=key,bad=bad),self.assertRaises(ValueError):constructor(**{key:bad})

    def test_actual_setup_selects_nonzero_mma_fragments_and_warp_layout(self):
        fn=function('_setup_attributes')
        end=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='sfa_smem')
        nodes=[n for n in fn.body[:end] if not isinstance(n,(ast.Import,ast.ImportFrom))]
        for enabled,shape,atom in ((True,(16,128,128),(1,2,1)),(False,(32,128,128),(2,2,1))):
            obj=constructor(ep_m16=enabled,mma_tiler_mn=shape[:2]);calls=[]
            obj.a_dtype='fp4';obj.acc_dtype='fp32';obj.sf_dtype='ue4m3'
            def tiled(op,layout,**kw):calls.append((op,layout,kw));return object()
            ns=dict(self=obj,hidden_size=4096,
                    sm120_utils=SimpleNamespace(get_permutation_mnk=lambda s,sf,swap:(s,sf,swap)),
                    cute=SimpleNamespace(nvgpu=SimpleNamespace(warp=SimpleNamespace(MmaMXF4NVF4Op=lambda *a:'nvfp4',MmaMXF4Op=lambda *a:'mxfp4')),
                        make_layout=lambda x:x,make_tiled_mma=tiled,make_mma_atom=lambda x:x))
            exec(code(nodes),ns)
            self.assertEqual(calls,[('nvfp4',atom,{'permutation_mnk':(shape,16,False)})])
            self.assertEqual((obj.num_m_tiles,obj.num_n_tiles,obj.num_k_blocks),(1,8,2))

    def test_actual_host_coordinate_validator_covers_all_64_lanes_and_rejects_bad_partition(self):
        class Partition:
            def __init__(self,values,stages=1):self.values,self.stages=values,stages
            def __getitem__(self,key):return self if isinstance(key,tuple) else self.values[key]
        class Thread:
            def __init__(self,tid,mutate,stages):self.tid,self.mutate,self.stages=tid,mutate,stages
            def partition_D(self,identity):
                return Partition(points(self.tid),self.stages) if identity=='staged' else Partition(self.mutate(self.tid,points(self.tid)))
            def partition_S(self,_):return (32,1,1,1)
        def run(mutate=lambda tid,p:p,stages=1):
            seen=[]
            def slice(tid):seen.append(tid);return Thread(tid,mutate,stages)
            obj=constructor();obj.c_layout=SimpleNamespace(is_m_major_c=lambda:False)
            obj.tiled_mma=None;obj.epi_smem_layout_staged=SimpleNamespace(outer='staged')
            api=SimpleNamespace(get_slice=slice)
            cute=SimpleNamespace(nvgpu=SimpleNamespace(CopyUniversalOp=lambda:None,warp=SimpleNamespace(StMatrix8x8x16bOp=lambda *a:None)),
                make_copy_atom=lambda *a:None,make_tiled_copy_C_atom=lambda *a:None,make_tiled_copy_S=lambda *a:api,
                make_identity_tensor=lambda x:x,shape=lambda x:x,make_layout=lambda x:x,
                size=lambda x,mode=None:x.stages if mode==[3] else len(x.values) if isinstance(x,Partition) else 32)
            extract(function('_validate_ep_direct_scatter_layout'),dict(cute=cute,cutlass=SimpleNamespace(BFloat16='bf16')))(obj)
            self.assertEqual(seen,list(range(64)))
        run()
        for stages in (0,2):
            with self.assertRaisesRegex(ValueError,'one epilogue buffer'):run(stages=stages)
        for mutation in [lambda tid,p:points(0) if tid==63 else p,
                         lambda tid,p:[(r+16,c,z) for r,c,z in p] if tid==0 else p,
                         lambda tid,p:[p[1],p[0],*p[2:]] if tid==0 else p,
                         lambda tid,p:p[:4]+p[6:8]+p[4:6]+p[8:] if tid==0 else p]:
            with self.assertRaisesRegex(ValueError,'direct scatter|row grouping'):run(mutation)

    def test_actual_scheduler_and_physical_subtile_address_cover_rows_once(self):
        schedule=extract(function('_compact_static_get_work_tile'),dict(Int32=int))
        obj=constructor();counts_cases=[[n] for n in range(65)]+[[0,1,15,16,17,31,32,33,48,63,64]]
        for counts in counts_cases:
            expected=[(m,n,e) for e,count in enumerate(counts) for m in range((count+15)//16) for n in range(16)]
            observed=[];expert=accum=0
            for work in range(len(expected)+1):
                tile,valid,expert,accum=schedule(counts,[len(counts)],tile_m=16,num_tiles_n=16,
                    cluster_shape_mn=(1,1),current_work_linear_idx=work,current_local_expert_idx=expert,
                    accum_tile_m=accum,cta_id_in_cluster=(0,0,0))
                if valid:observed.append(tile)
                else:self.assertEqual(work,len(expected))
            self.assertEqual(observed,expected)
            for e,count in enumerate(counts):
                physical=[]
                for m in range((count+15)//16):
                    # Same active physical M128 TMA block; rows are M16 subtiles.
                    self.assertEqual(m//obj.sa_tiles_per_block,0)
                    base=(m%obj.sa_tiles_per_block)*obj.tile_shape_mnk[0]
                    valid=max(0,min(16,count-m*16))
                    self.assertLessEqual(base+valid,128)
                    physical.extend(base+r for r in range(valid))
                self.assertEqual(physical,list(range(count)))

    def test_q1_64_lane_work_distribution_and_scale_layout_never_touch_padding(self):
        fn=function('kernel')
        loop=next(n for n in ast.walk(fn) if isinstance(n,ast.While) and ast.unparse(n.test)=='quant_idx < epi_rows * sf_blocks_per_row')
        # Execute the actual loop's row/scale-address expressions; the expensive
        # numeric quantizer is intentionally outside this indexing oracle.
        starts=loop.body[:4]
        sf_start=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='outer_m_idx')
        end=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='st_shared_u8')
        indexed=copy.deepcopy(loop);indexed.body=starts+loop.body[sf_start:end]+[ast.Expr(value=ast.Call(func=ast.Name(id='record',ctx=ast.Load()),args=[ast.Name(id='local_row',ctx=ast.Load()),ast.Name(id='row',ctx=ast.Load()),ast.Name(id='sf_block',ctx=ast.Load()),ast.Name(id='sf_raw_idx',ctx=ast.Load())],keywords=[])),loop.body[-1]]
        for count in range(65):
            seen=[]
            for m in range(4):
                valid=max(0,min(16,count-m*16));base=m*16
                for tid in range(64):
                    def record(local,row,sf,offset):
                        self.assertTrue(0<=local<valid)
                        self.assertTrue(base<=row<base+valid)
                        reference=(sf//4)*512+(row%32)*16+(row//32)*4+(sf%4)
                        self.assertEqual(offset,reference);self.assertTrue(0<=offset<1024)
                        seen.append((row,sf))
                    exec(code([indexed]),dict(quant_idx=tid,epi_rows=valid,sf_blocks_per_row=8,
                        sa_row_base=base,rows_offset=0,Int32=int,self=constructor(),record=record))
            self.assertCountEqual(seen,[(r,sf) for r in range(count) for sf in range(8)])

    def test_actual_grouped_scatter_handles_four_partial_tiles_and_poisoned_rows(self):
        guard=next(n for n in ast.walk(function('kernel')) if isinstance(n,ast.If)
                   and ast.unparse(n.test)=='cutlass.const_expr(self.shared_fc1_a)'
                   and any(isinstance(c,ast.For) and ast.unparse(c.target)=='ep_parity' for c in n.body))
        compiled=code(guard.body)
        for count in (0,1,6,8,15,16,17,31,32,33,47,48,49,63,64):
            observed=[];expected=[]
            for m in range(4):
                valid=max(0,min(16,count-m*16));base=m*16
                for tid in range(64):
                    p=points(tid);loads=[]
                    class Values:
                        def __getitem__(self,index):
                            row,col,_=p[index];assert row<valid,'inactive register'
                            return bf16((base+row+1)*.137+(col+1)*.0031)
                    def load(address,offset):
                        row=(address-offset)//4;assert 0<=row<valid,'poisoned metadata';loads.append(row)
                        return (base+row)%8 if offset==1000 else [1.,.5,-2.,0.,-0.,2.,-.5,1.][(base+row)%8]
                    def scatter(address,a,b):
                        observed.extend(((address,struct.pack('<f',bf16(a))),(address+1,struct.pack('<f',bf16(b)))))
                    exec(compiled,dict(ep_coords=p,epi_m=0,valid_tile_rows=valid,self=constructor(),Int32=int,
                        cutlass=SimpleNamespace(range_constexpr=range,Float32=float),tRS_rD_out=Values(),
                        scatter_tok_base_addr=1000,scatter_weight_base_addr=2000,_ld_shared_i32=lambda a:load(a,1000),
                        _ld_shared_f32=lambda a:load(a,2000),scatter_output=None,scatter_N=4096,tile_n_base_cur=3968,
                        get_ptr_as_int64=lambda _,index:index,scatter_add_bf16x2_to_f32=scatter))
                    self.assertEqual(len(loads),2*sum(p[2*k][0]<valid for k in (0,1)))
                for row in range(valid):
                    token=(base+row)%8;weight=[1.,.5,-2.,0.,-0.,2.,-.5,1.][token]
                    for col in range(128):
                        v=bf16((base+row+1)*.137+(col+1)*.0031)
                        expected.append((token*4096+3968+col,struct.pack('<f',bf16(weight*v))))
            self.assertEqual(len(observed),count*128)
            self.assertCountEqual(observed,expected)


if __name__=='__main__':unittest.main()
