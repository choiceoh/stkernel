"""Actual-source CPU schedules, immutable false-path oracle, and route bounds.

These do not execute CuTe/CUDA. Actual lowering, races and serving remain gates.
"""
import ast
import copy
import gzip
import hashlib
import json
from pathlib import Path
import struct
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
MICRO = ROOT / 'overlay/modules/glm53_moe/moe_micro_kernel.py'
ORACLE = ROOT / 'measurements/glm53_ep_local_20260908/micro-stock-oracle'
STOCK_SHA = 'fbf64e979f313b5f460094be5158055992f44607542c4209d73b4108c69ced07'


def function(name, source=None):
    return next(n for n in ast.walk(ast.parse(source or MICRO.read_text()))
                if isinstance(n, ast.FunctionDef) and n.name == name)


class SelectShared(ast.NodeTransformer):
    def __init__(self, enabled=False): self.enabled = enabled
    def condition(self, n):
        text = ast.unparse(n)
        if text in ('self.shared_fc1_a', 'cutlass.const_expr(self.shared_fc1_a)'):
            return self.enabled
        if text in ('not self.shared_fc1_a', 'cutlass.const_expr(not self.shared_fc1_a)'):
            return not self.enabled
        return None
    def visit_If(self, n):
        value = self.condition(n.test)
        if value is None: return self.generic_visit(n)
        result = []
        for item in n.body if value else n.orelse:
            item = self.visit(item)
            result.extend(item if isinstance(item, list) else [item])
        return result
    def visit_IfExp(self, n):
        value = self.condition(n.test)
        return self.generic_visit(n) if value is None else self.visit(n.body if value else n.orelse)


def block(loop_name):
    return next(n for n in ast.walk(function('kernel')) if isinstance(n, ast.If)
                and ast.unparse(n.test) == 'cutlass.const_expr(self.shared_fc1_a)'
                and any(isinstance(k, ast.For) and ast.unparse(k.target) == loop_name
                        for k in n.body))


def code(nodes):
    class Ranges(ast.NodeTransformer):
        def visit_Call(self, n):
            if isinstance(n.func, ast.Name) and n.func.id == 'range':
                n.keywords = []
            return self.generic_visit(n)
    tree = ast.Module(body=copy.deepcopy(nodes), type_ignores=[])
    return compile(ast.fix_missing_locations(Ranges().visit(tree)), str(MICRO), 'exec')


def init():
    fn = copy.deepcopy(function('__init__')); fn.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom('__future__',[ast.alias('annotations')],0),fn],type_ignores=[])
    ns = dict(cutlass=SimpleNamespace(Float32='f32'), DenseGemmKernel=object,
              is_gated_activation=lambda _:True,
              utils=SimpleNamespace(get_smem_capacity_in_bytes=lambda _:101376),
              pipeline=SimpleNamespace(NamedBarrier=lambda **k:k))
    exec(compile(ast.fix_missing_locations(module),str(MICRO),'exec'),ns)
    return ns['__init__']


class State:
    def __init__(self, stages): self.stages,self.count = stages,0
    @property
    def index(self): return self.count % self.stages
    def reset_count(self): self.count = 0
    def advance(self): self.count += 1


class Ref:
    def __init__(self, name, keys=()): self.name,self.keys=name,keys
    def __getitem__(self,key): return Ref(self.name, self.keys+(key,))
    @property
    def iterator(self): return self
    def fill(self,value): assert value == 0.


def producer_trace(stages, tiles=32):
    state=State(stages); events=[]; batch=[]
    def copy_tma(kind, src, dst, *, tma_bar_ptr):
        assert src.keys == ((None,state.count),)
        assert dst.keys == ((None,state.index),)
        assert tma_bar_ptr == ('ml',state.count)
        batch.append((kind,src.name,dst.name))
    def acquire(s):
        assert not batch
        events.append(('acquire',s.count,s.index))
    def commit(s):
        events.append(('commit',s.count,s.index,tuple(batch))); batch.clear()
    pipe=SimpleNamespace(producer_acquire=acquire, producer_commit=commit,
                         producer_get_barrier=lambda s:('ml',s.count))
    ns=dict(prod_state=state,ml_pipeline=pipe,fc1_k_tile_cnt=tiles,
            cute=SimpleNamespace(copy=copy_tma),
            self=SimpleNamespace(pass_sync_barrier=SimpleNamespace(
                arrive_and_wait=lambda:events.append(('end_fc1',state.count)))))
    names=('tAgA_mk','tAgSFA_mk','tBgB_w13_gate_nk','tBgSFB_w13_gate_nk',
           'tBgB_w13_up_nk','tBgSFB_w13_up_nk','tAsA','tAsSFA','tBsB_w13',
           'tBsSFB_w13','tBsB_w13_up','tBsSFB_w13_up')
    ns.update({n:Ref(n) for n in names})
    ns.update({n:n for n in ('tma_a','tma_sfa','tma_b_w13','tma_sfb_w13')})
    exec(code(block('k_tile').body),ns)
    return events


def consumer_trace(stages, tiles=32):
    state=State(stages); events=[]; registers={}; scales={}; current=[None]
    groups={'crA_tile':'A','tCrA_tile':'A','crSFA_tile':'SFA','tCrSFA_tile':'SFA',
            'crB':'B','tCrB':'B','crSFB_tile':'SFB','tCrSFB_tile':'SFB'}
    names=('csA_tile','csB','csB_up','csSFA_tile','csSFB_tile','csSFB_up_tile')
    def copy_smem(kind,src,dst):
        assert len(src.keys)==2 and src.keys[0][-1]==state.index
        kb=src.keys[-1][-1]
        assert kb in (0,1) and current[0]==state.count
        assert dst.keys[-1][-1]==kb
        registers[(groups[dst.name],kb)]=(src.name,state.count,kb)
        events.append(('read',state.count,kb,src.name))
    def wait(s,p):
        assert p==s.count
        current[0]=s.count;events.append(('wait',s.count,s.index))
    def set_sf(kind,ref): scales[kind]=registers[(groups[ref.name],ref.keys[-1][-1])]
    def gemm(atom,out,a,b,previous):
        assert out.name==previous.name
        mt,nt=out.keys[-1][1:]
        kb=a.keys[-1][-1]
        assert b.keys[-1][-1]==kb
        plane='gate' if out.name=='gate_acc' else 'up'
        assert registers[('A',kb)]==('csA_tile',state.count,kb)
        assert scales['SFA']==('csSFA_tile',state.count,kb)
        assert registers[('B',kb)]==('csB' if plane=='gate' else 'csB_up',state.count,kb)
        assert scales['SFB']==('csSFB_tile' if plane=='gate' else 'csSFB_up_tile',state.count,kb)
        events.append(('mma',state.count,kb,plane,mt,nt))
    def release(s):
        # Both K blocks and all fragments have consumed gate/up before reuse.
        actual=[e[2:] for e in events if e[:2]==('mma',s.count)]
        expected=[(kb,plane,0,nt) for kb in range(2) for plane in ('gate','up') for nt in range(8)]
        assert actual==expected
        events.append(('release',s.count,s.index))
    pipe=SimpleNamespace(consumer_try_wait=lambda s:s.count, consumer_wait=wait,consumer_release=release)
    ns={n:Ref(n) for n in (*names,*groups,'gate_acc','up_acc')}
    ns.update({n:n for n in ('smem_copy_A','smem_copy_B','smem_copy_SFA','smem_copy_SFB')})
    ns.update(cons_state=state,ml_pipeline=pipe,fc1_k_tile_cnt=tiles,num_k_blocks=2,
              cute=SimpleNamespace(copy=copy_smem,filter_zeros=lambda r:r,gemm=gemm),
              cutlass=SimpleNamespace(range_constexpr=range),mma_atom=SimpleNamespace(set=set_sf),
              WarpField=SimpleNamespace(SFA='SFA',SFB='SFB'),
              self=SimpleNamespace(num_m_tiles=1,num_n_tiles=8,pass_sync_barrier=SimpleNamespace(
                  arrive_and_wait=lambda:events.append(('end_fc1',state.count)))))
    exec(code(block('_k_tile').body),ns)
    # CuTe merges this name with the legacy FC1 branch before the common FC2
    # loop. Both paths must supply an int, including the shared path.
    assert type(ns['k_next']) is int and ns['k_next'] == 0
    return events


def coords(tid):
    lane,warp=tid%32,tid//32
    out=[]
    for pair in range(16):
        row=(lane>>2)+16*(warp&1)+8*(pair&1); k=pair>>1
        col=2*(lane&3)+16*(warp>>1)+8*(k&1)+32*((k>>1)&1)+64*((k>>2)&1)
        out.extend([(row,col,0),(row,col+1,0)])
    return out


def bf16(value):
    bits=struct.unpack('<I',struct.pack('<f',value))[0]
    rounded=(bits+0x7fff+((bits>>16)&1))>>16
    return struct.unpack('<f',struct.pack('<I',(rounded&0xffff)<<16))[0]


class SharedFC1Tests(unittest.TestCase):
    def test_false_path_matches_hash_bound_cpu17_entire_kernel_and_layout(self):
        raw=gzip.decompress((ORACLE/'moe_micro_kernel_cpu17.py.gz').read_bytes())
        identity=json.loads((ORACLE/'micro-kernel-cpu17-identity.json').read_text())
        self.assertEqual(hashlib.sha256(raw).hexdigest(),STOCK_SHA)
        self.assertEqual(identity['source_sha256'],STOCK_SHA)
        self.assertEqual(identity['source_revision'],'0331579f4b16b8b811f2ca7e5099f8f461507c67')
        self.assertEqual(hashlib.sha256((ORACLE/'moe_micro_kernel_cpu17.py.gz').read_bytes()).hexdigest(),identity['gzip_sha256'])
        for name in ('kernel','__call__','_setup_attributes','_shared_storage_size_bytes','_validate_ep_direct_scatter_layout'):
            current=SelectShared().visit(function(name))
            self.assertEqual(ast.dump(current),ast.dump(function(name,raw)),name)

    def test_constructor_is_default_off_and_reuses_all_exact_direct_gates(self):
        make=init(); kwargs=dict(sf_vec_size=16,mma_tiler_mn=(32,128),output_tile_count_n=16,
            activation='swigluoai_uninterleave',swiglu_alpha=1.,swiglu_beta=0.,swiglu_limit=10.,
            scatter_fp32=True,ep_direct_scatter=True,skip_zero_weight_expert_id=72)
        base=SimpleNamespace();make(base,**kwargs);self.assertFalse(base.shared_fc1_a)
        enabled=SimpleNamespace();make(enabled,**kwargs,shared_fc1_a=True)
        self.assertTrue(enabled.shared_fc1_a)
        for key,bad in (('ep_direct_scatter',False),('sf_vec_size',32),('mma_tiler_mn',(64,128)),
                        ('output_tile_count_n',8),('scatter_fp32',False),('skip_zero_weight_expert_id',None),
                        ('single_token',True),('share_input_across_experts',True),('share_expert_scales',True),
                        ('activation','silu'),('swiglu_alpha',1.702),('swiglu_beta',1.),('swiglu_limit',9.)):
            with self.subTest(key=key),self.assertRaises(ValueError):
                make(SimpleNamespace(),**{**kwargs,key:bad},shared_fc1_a=True)

    def test_actual_producer_six_distinct_transfers_one_epoch_and_exact_bytes(self):
        expected=(('tma_a','tAgA_mk','tAsA'),('tma_sfa','tAgSFA_mk','tAsSFA'),
                  ('tma_b_w13','tBgB_w13_gate_nk','tBsB_w13'),('tma_sfb_w13','tBgSFB_w13_gate_nk','tBsSFB_w13'),
                  ('tma_b_w13','tBgB_w13_up_nk','tBsB_w13_up'),('tma_sfb_w13','tBgSFB_w13_up_nk','tBsSFB_w13_up'))
        for stages in (1,2):
            events=producer_trace(stages)
            self.assertEqual(events[-1],('end_fc1',32))
            self.assertEqual([e for e in events if e[0]=='commit'],
                             [('commit',k,k%stages,expected) for k in range(32)])
        fn=function('kernel'); start=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='tma_copy_bytes')
        sizes={'A':8192,'B':8192,'SFA':1024,'SFB':1024}
        for enabled,total in ((False,18432),(True,27648)):
            ns=dict(self=SimpleNamespace(a_dtype='a',b_dtype='b',sf_dtype='sf',shared_fc1_a=enabled),
                    cute=SimpleNamespace(size_in_bytes=lambda dtype,layout:sizes[layout]),
                    cutlass=SimpleNamespace(const_expr=lambda x:x),
                    a_smem_one='A',b_smem_one='B',sfa_smem_one='SFA',sfb_smem_one='SFB')
            exec(code(fn.body[start:start+3]),ns)
            self.assertEqual(ns['tma_copy_bytes'],total)
            self.assertEqual(ns['phase2_tma_copy_bytes'],9216)
        self.assertEqual(32*(27648+9216),1179648)
        self.assertEqual(32*(2*18432+9216),1474560)

    def test_actual_consumer_keeps_every_accumulator_k_order_and_reads_A_once(self):
        shared = block('_k_tile').body
        initialization = next(n for n in shared if isinstance(n,ast.Assign)
                              and ast.unparse(n.targets[0]) == 'k_next')
        self.assertEqual(ast.dump(initialization.value),ast.dump(ast.Constant(0)))
        self.assertLess(shared.index(initialization),
                        next(i for i,n in enumerate(shared) if isinstance(n,ast.For)))
        for stages in (1,2):
            events=consumer_trace(stages)
            self.assertEqual(events[-1],('end_fc1',32))
            for plane in ('gate','up'):
                actual=[(e[1],e[2],e[4],e[5]) for e in events if e[0]=='mma' and e[3]==plane]
                self.assertEqual(actual,[(k,b,0,n) for k in range(32) for b in range(2) for n in range(8)])
            for name in ('csA_tile','csSFA_tile','csB','csSFB_tile','csB_up','csSFB_up_tile'):
                self.assertEqual(sum(e[0]=='read' and e[3]==name for e in events),64)
            for k in range(32):
                at_release=next(i for i,e in enumerate(events) if e[:2]==('release',k))
                self.assertTrue(all(i<at_release for i,e in enumerate(events) if e[:2] in [('read',k),('mma',k)]))

    def test_end_fc1_barrier_and_no_independent_up_pipeline_or_tail(self):
        selected=SelectShared(True).visit(function('kernel'))
        text=ast.unparse(selected)
        for operation in ('up_pipeline.consumer_','up_pipeline.producer_','up_cons_state.reset','up_prod_state.reset'):
            self.assertNotIn(operation,text)
        for name,target in (('up_pipeline','ml_pipeline'),('up_prod_state','prod_state'),('up_cons_state','cons_state')):
            assignment=next(n for n in ast.walk(selected) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])==name)
            self.assertEqual(ast.unparse(assignment.value),target)
        for loop in ('_k_tile','k_tile'):
            self.assertEqual(ast.unparse(block(loop).body[-1]),'self.pass_sync_barrier.arrive_and_wait()')
        # This same full CTA barrier remains before both activation and FC2 DMA,
        # and the original final task barrier is preserved by the false oracle.
        dma=block('k_tile')
        parent=next(n for n in ast.walk(function('kernel')) if isinstance(n,ast.While)
                    and any(k.lineno==dma.lineno for k in n.body))
        position=next(i for i,n in enumerate(parent.body) if n.lineno==dma.lineno)
        following=parent.body[position+1:]
        self.assertEqual(ast.unparse(following[0]),'phase2_prod_state.reset_count()')
        self.assertTrue(any(ast.unparse(n)=='self.pass_sync_barrier.arrive_and_wait()' for n in following))

    def test_grouped_scatter_same_contributions_with_poisoned_inactive_rows(self):
        branch=block('ep_parity'); compiled=code(branch.body)
        class Slice:
            def __init__(self,points):self.points=points
            def __getitem__(self,index):return self.points[index]
        for count in (0,1,8,31,32,33,48,63,64):
            for col_base in (0,896,3968):
                observed=[];expected=[];loads=[]
                for tile_base in (0,32):
                    valid=max(0,min(32,count-tile_base))
                    weights=[1.,.5,-2.,0.,-0.,2.,-.5,1.]
                    for tid in range(128):
                        points=coords(tid)
                        class Values:
                            def __getitem__(self,index):
                                row,col,_=points[index]
                                assert row<valid,'inactive register read'
                                return bf16((tile_base+row+1)*.137+(col+1)*.0031)
                        def load(address,base):
                            row=(address-base)//4
                            assert 0<=row<valid,'unwritten metadata read'
                            loads.append((tid,row,base))
                            return (tile_base+row)%8 if base==1000 else weights[(tile_base+row)%8]
                        def scatter(address,a,b):
                            observed.extend(((address,struct.pack('<f',bf16(a))),(address+1,struct.pack('<f',bf16(b)))))
                        ns=dict(ep_coords=Slice(points),epi_m=0,valid_tile_rows=valid,
                                self=SimpleNamespace(epi_tile=(32,128)),Int32=int,
                                cutlass=SimpleNamespace(range_constexpr=range,Float32=float),
                                tRS_rD_out=Values(),scatter_tok_base_addr=1000,scatter_weight_base_addr=2000,
                                _ld_shared_i32=lambda a:load(a,1000),_ld_shared_f32=lambda a:load(a,2000),
                                scatter_N=4096,tile_n_base_cur=col_base,scatter_output=None,
                                get_ptr_as_int64=lambda _,index:index,scatter_add_bf16x2_to_f32=scatter)
                        before=len(loads);exec(compiled,ns)
                        self.assertEqual(len(loads)-before,2*sum(points[2*p][0]<valid for p in (0,1)))
                    for row in range(valid):
                        tok=(tile_base+row)%8
                        for col in range(128):
                            value=bf16((tile_base+row+1)*.137+(col+1)*.0031)
                            expected.append((tok*4096+col_base+col,struct.pack('<f',bf16(weights[tok]*value))))
                self.assertEqual(len(observed),count*128)
                self.assertCountEqual(observed,expected)

    def test_actual_coordinate_row_group_guard_rejects_legal_pair_permutation(self):
        fn=function('_validate_ep_direct_scatter_layout')
        guard=next(n for n in ast.walk(fn) if isinstance(n,ast.If) and ast.unparse(n.test)=='self.shared_fc1_a')
        for tid in range(128):
            points=coords(tid)
            ns=dict(self=SimpleNamespace(shared_fc1_a=True),cute=SimpleNamespace(size=len),coords=points)
            exec(code([guard]),ns)
            # Swap two complete pairs across row parity: still unique/adjacent
            # and full coverage, but metadata hoisting would now be wrong.
            bad=points.copy();bad[4:8]=points[6:8]+points[4:6]
            with self.assertRaisesRegex(ValueError,'row grouping'):
                exec(code([guard]),{**ns,'coords':bad})


if __name__=='__main__':unittest.main()
