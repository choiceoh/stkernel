"""CPU source execution for hybrid dual-warp Q0 ownership; no GPU/DSL imports.

Quantizer calls carry symbolic payloads here. The unchanged per-block math is
separately compared with the pinned baseline AST; these are not GPU numerics.
"""
import ast
import copy
import gzip
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

from test_glm53_ep_route_scale_cache import FloatBits, ExpertId, Uint32, f32, single_warp_projection

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'overlay/modules/glm53_moe/moe_dynamic_ep_local.py'
BASE = ROOT / 'measurements/glm53_ep_tiled_20260909/ep76_onepass6/source/moe_dynamic_ep_local.py.gz'
BASE_SHA = '579515f00dbd459f49a63b7b3fa801c30ebcc7fdbc06e96ca2b5a51c22faac2e'


def method(name, *, baseline=False, cls='MoEGatedEPLocalKernel'):
    raw = gzip.decompress(BASE.read_bytes()) if baseline else SOURCE.read_bytes()
    if baseline:
        assert hashlib.sha256(raw).hexdigest() == BASE_SHA
    tree = ast.parse(raw)
    c = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    return copy.deepcopy(next(n for n in c.body if isinstance(n, ast.FunctionDef) and n.name == name))


def compile_nodes(nodes, ns):
    tree = ast.Module(body=[ast.ImportFrom('__future__', [ast.alias('annotations')], 0)] + copy.deepcopy(nodes), type_ignores=[])
    return compile(ast.fix_missing_locations(tree), str(SOURCE), 'exec')


def spelling(node):
    return ast.dump(node, include_attributes=False)


class UniqueStores(dict):
    def __setitem__(self, key, value):
        if key in self:
            raise AssertionError(('duplicate writer', key))
        super().__setitem__(key, value)


class Batch:
    """Execute the actual route phase, publication point, and Q0 consumers."""
    def __init__(self, dual, *, tail=4, local_routes=4, unequal=False, changed=False, fast=True, weight_bits=None, scale_bits=None):
        fn = method('initialize_route_q0_and_publish')
        loop = next(n for n in ast.walk(fn) if isinstance(n, ast.While) and ast.unparse(n.test) == 'produce_active > Int32(0)')
        dispatch = next(n for n in loop.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'batch_base >= producer_limit')
        body = dispatch.orelse
        begin = next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0]) == 'q0_row_idx')
        publish = next(i for i,n in enumerate(body) if isinstance(n,ast.If) and ast.unparse(n.test) == 'cutlass.const_expr(self.q0_dual_warp)' and any(isinstance(x,ast.Call) and ast.unparse(x.func) == 'cute.arch.sync_threads' for x in ast.walk(n)))
        # The new barrier is a direct child of the uniform live-batch branch.
        self.publish_node = body[publish]
        producer = compile_nodes(body[begin:publish], {})
        barrier = compile_nodes([body[publish]], {})
        consumer = compile_nodes(body[publish+1:], {})
        self.shared, self.alloc, self.global_writes = {}, [0]*144, UniqueStores()
        self.packed, self.scales = UniqueStores(), UniqueStores()
        self.loads, self.quantizers, self.barriers, self.warp_syncs = [], [], [], []
        self.weights_read = []
        self.tail, self.dual = tail, dual
        base_token = 32
        # Duplicate first expert, high local expert, sentinel, negative, zero.
        choices = [0, 0, 143, 71, 5, 142, 10, 20]
        ids = [ExpertId(144)]*(base_token*8)
        weight_values = [f32(0)]*(base_token*8)
        for row in range(tail):
            ids.extend(ExpertId(choices[s] if s<local_routes else (144 if s%2 else -1)) for s in range(8))
            weight_values.extend(FloatBits(weight_bits[s]) if weight_bits is not None else
                                 f32((-.125 if changed else .25)*(s+1)) for s in range(8))
        for expert in range(144):
            self.shared[2304+expert*4] = (scale_bits[expert % len(scale_bits)] if scale_bits is not None
                                       else f32(1+(expert%3 if unequal else 0)).bits)
        outer = self
        class Weights:
            def __getitem__(self, i):
                if not 0 <= ids[i] < 144: raise AssertionError('remote weight read')
                outer.weights_read.append(i)
                return weight_values[i]
        def load(addr):
            value = self.shared[addr]
            return value if value < 0x80000000 else value-0x100000000
        def store(addr, value): self.shared[addr] = value & 0xFFFFFFFF
        def atomic(pointer, value):
            name, expert = pointer
            assert name == 'rows' and value == 1
            old=self.alloc[expert];self.alloc[expert]+=1
            return old
        def global_store(pointer, value):
            self.global_writes[pointer] = value.bits if isinstance(value, FloatBits) else value
        current = [None]
        def read_values(address):
            row, byte = divmod(address, 8192)
            assert 0 <= row < tail and byte % 32 == 0 and byte < 8192
            self.loads.append((current[0],row,byte//32))
            sign=-1 if changed else 1
            return [sign*(row*4096+byte//2+i+1) for i in range(16)]
        def quant(values, maximum, gs):
            payload=(tuple(values),maximum,gs.bits,fast)
            self.quantizers.append(payload)
            return payload, (gs.bits & 255)
        def packed_store(pointer, payload):
            assert pointer[0]=='packed'
            self.packed[pointer[1]]=payload
        ns=dict(Int32=int,Int64=int,Uint32=Uint32,Uint64=int,Uint8=int,
            self=SimpleNamespace(q0_dual_warp=dual,fast_math=fast,tile_shape_mnk=(128,128,128)),
            cutlass=SimpleNamespace(const_expr=bool,Float32=f32,range_constexpr=range),
            cute=SimpleNamespace(make_rmem_tensor=lambda shape,dtype:[None]*shape[0],arch=SimpleNamespace(
                sync_threads=lambda:self.barriers.append(current[0]),sync_warp=lambda:self.warp_syncs.append(current[0]))),
            batch_base=base_token,producer_batch_tokens=4,num_tokens=base_token+tail,
            num_topk=8,num_experts=144,num_k_tiles=64,cols=4096,sf_blocks_per_row=256,
            output_bytes_per_row=2048,q0_input_stage_base_addr=0,q0_bulk_barrier_addr=777,q0_bulk_phase=0,
            topk_ids=ids,topk_weights=Weights(),expert_write_rows='rows',expert_tile_base=[e*8 for e in range(144)],
            token_map='map',token_weights='weights',route_phys_rows_addr=0,route_scales_addr=1152,
            route_expert_ids_addr=1152,expert_scales_addr=2304,
            _ld_shared_i32=load,_st_shared_i32=store,get_ptr_as_int64=lambda t,i:(t,i),
            atomic_add_global_i32=atomic,st_global_i32=global_store,st_global_f32=global_store,
            q0_bulk_try_wait=lambda *a:1,load_shared_bf16x16_to_f32x16=read_values,
            fabs_f32=abs,fmax_f32=lambda a,b:max(a.number() if isinstance(a,FloatBits) else a,b),
            quantize_block_fp4_fast=quant,quantize_block_fp4=quant,
            packed_a_storage='packed',st_global_u64=packed_store,scale_storage=self.scales)
        contexts=[]
        for tid in range(288):
            current[0]=tid
            context=dict(ns,tidx=tid,warp_idx=tid//32,lane_id=tid%32)
            exec(producer,context);contexts.append(context)
        for tid,context in enumerate(contexts):
            current[0]=tid;exec(barrier,context)
        if dual: assert self.barriers == list(range(288))
        # Non-leaders consume first to expose reliance on producer registers.
        order=sorted(range(288),key=lambda t:(t%64==0,t))
        for tid in order:
            current[0]=tid;exec(consumer,contexts[tid])
        self.contexts=contexts


class DualWarpTests(unittest.TestCase):
    def test_constructor_and_actual_weight_gate(self):
        fn=method('__init__',cls='MoEGatedEPLocalKernelSF6');fn.decorator_list=[]
        class Parent:
            def __init__(self,*args,**kwargs):
                self.__dict__.update(num_mma_warps=8,threads_per_cta=288,tile_shape_mnk=(128,128,128),
                    sf_vec_size=16,share_input_across_experts=False,activation='swigluoai_uninterleave',
                    swiglu_alpha=1.,swiglu_beta=0.,swiglu_limit=10.)
                self.__dict__.update(kwargs.pop('mutate',{}))
        cls=ast.ClassDef('Candidate',[ast.Name('Parent',ast.Load())],[],[fn],[])
        ns={'Parent':Parent};exec(compile_nodes([cls],ns),ns); C=ns['Candidate']
        self.assertFalse(C().q0_dual_warp);self.assertTrue(C(q0_dual_warp=True).q0_dual_warp)
        for bad in (0,1,None,'1'):
            with self.subTest(bad=bad),self.assertRaises(TypeError):C(q0_dual_warp=bad)
        for key,value in dict(num_mma_warps=4,threads_per_cta=256,tile_shape_mnk=(64,128,128),sf_vec_size=32,
                              share_input_across_experts=True,activation='silu',swiglu_alpha=1.7,swiglu_beta=1.,swiglu_limit=9.).items():
            with self.subTest(key=key),self.assertRaises(ValueError):C(q0_dual_warp=True,mutate={key:value})
        with self.assertRaises(ValueError):C(q0_dual_warp=True,reform_sf_pack=False)
        spec=importlib.util.spec_from_file_location('q0_geometry',ROOT/'overlay/modules/glm53_moe/glm53_ep_shard_geometry.py')
        geometry=importlib.util.module_from_spec(spec);spec.loader.exec_module(geometry)
        gate=method('_check_ep_shard_weights');gate.decorator_list=[]
        ns={'ep_shard_geometry':geometry.ep_shard_geometry};exec(compile_nodes([gate],ns),ns)
        for E,I,sf6,dual,ok in ((144,1024,True,True,True),(72,2048,True,True,False),(72,2048,True,False,True),(144,1024,False,True,False)):
            shard=geometry.ep_shard_geometry(E,I)
            args=[SimpleNamespace(q0_dual_warp=dual),SimpleNamespace(shape=shard['cute_w13']),SimpleNamespace(shape=shard['cute_down']),SimpleNamespace(shape=(E,))]
            if ok:ns[gate.name](*args,sf6=sf6)
            else:
                with self.assertRaises(ValueError):ns[gate.name](*args,sf6=sf6)

    def test_false_projection_preserves_entire_original_q0_method(self):
        # Stronger than numeric samples: constant-off work, quantization order,
        # output clear, histogram, task publication and all old barriers match.
        original=method('initialize_route_q0_and_publish',baseline=True)
        projected=single_warp_projection(method('initialize_route_q0_and_publish'))
        self.assertEqual(spelling(projected),spelling(original))
        # Compare the live dual block itself, excluding only its lane-stride
        # increment and replacing the separately tested staging-row variable.
        current=method('initialize_route_q0_and_publish')
        loops=[next(n for n in ast.walk(tree) if isinstance(n,ast.While) and ast.unparse(n.test)=='sf_idx < sf_blocks_per_row') for tree in (current,original)]
        class RowName(ast.NodeTransformer):
            def visit_Name(self,n):
                if n.id=='q0_row_idx':n.id='warp_idx'
                return n
        actual=RowName().visit(ast.Module(body=copy.deepcopy(loops[0].body[:-1]),type_ignores=[]))
        expected=ast.Module(body=loops[1].body[:-1],type_ignores=[])
        self.assertEqual(spelling(actual),spelling(expected))

    def test_dual_owns_every_sf_block_once_and_halves_blocks_per_thread(self):
        for dual in (False,True):
            batch=Batch(dual)
            self.assertEqual(sorted((row,sf) for _,row,sf in batch.loads),[(r,s) for r in range(4) for s in range(256)])
            threads={tid for tid,_,_ in batch.loads}
            self.assertEqual(len(threads),256 if dual else 128)
            self.assertTrue(all(sum(t==tid for t,_,_ in batch.loads)==(4 if dual else 8) for tid in threads))
            self.assertFalse(any(t>=256 for t in threads))
            self.assertEqual(len(batch.weights_read),16)

    def test_all_tail_warps_and_dma_reach_publication_before_consume(self):
        for tail in (1,2,3,4):
            b=Batch(True,tail=tail)
            self.assertEqual(b.barriers,list(range(288)))
            self.assertEqual(len({t for t,_,_ in b.loads}),tail*64)
            self.assertEqual(sum(b.alloc),tail*4)
            self.assertEqual(len(b.weights_read),tail*4)
            self.assertEqual(b.warp_syncs,[])

    def test_packed_scale_and_route_payloads_equal_for_zero_to_eight_routes(self):
        for count in range(9):
            for unequal in (False,True):
                a,b=Batch(False,local_routes=count,unequal=unequal),Batch(True,local_routes=count,unequal=unequal)
                self.assertEqual(a.global_writes,b.global_writes)
                self.assertEqual(a.packed,b.packed);self.assertEqual(a.scales,b.scales)
                self.assertEqual(a.alloc,b.alloc)
                self.assertEqual(len(a.quantizers),len(b.quantizers))

    def test_changed_input_and_slow_math_use_same_exact_operands(self):
        for fast in (False,True):
            for changed in (False,True):
                a,b=Batch(False,unequal=True,changed=changed,fast=fast),Batch(True,unequal=True,changed=changed,fast=fast)
                self.assertEqual(a.packed,b.packed);self.assertEqual(a.scales,b.scales)
        self.assertNotEqual(Batch(True).packed,Batch(True,changed=True).packed)

    def test_signed_zero_nan_scale_and_zero_weight_routing_match(self):
        bits=[0x00000000,0x80000000,0x3f800000,0x7fc00021,0x3e800000,0xbf800000,0x3f000000,0x3f800000]
        for scales in ([0x00000000,0x80000000],[0x7fc00031],[0x3f800000,0x7fc00041]):
            a,b=(Batch(flag,local_routes=8,weight_bits=bits,scale_bits=scales) for flag in (False,True))
            self.assertEqual(a.alloc,b.alloc);self.assertEqual(sum(b.alloc),24)
            self.assertEqual(a.global_writes,b.global_writes)
            self.assertEqual(a.packed,b.packed);self.assertEqual(a.scales,b.scales)
            self.assertEqual(len(a.quantizers),len(b.quantizers))

    def test_staging_capacity_and_uniform_barrier_scope(self):
        fn=method('initialize_route_q0_and_publish')
        batch=next(n for n in ast.walk(fn) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='producer_batch_tokens' for t in n.targets))
        self.assertEqual(eval(compile(ast.Expression(batch.value),str(SOURCE),'eval'),{'Int32':int,'self':SimpleNamespace(tile_shape_mnk=(128,128,128)),'cols':4096}),4)
        loop=next(n for n in ast.walk(fn) if isinstance(n,ast.While) and ast.unparse(n.test)=='produce_active > Int32(0)')
        branch=next(n for n in loop.body if isinstance(n,ast.If) and ast.unparse(n.test)=='batch_base >= producer_limit')
        direct=[n for n in branch.orelse if isinstance(n,ast.If) and ast.unparse(n.test)=='cutlass.const_expr(self.q0_dual_warp)' and any(isinstance(x,ast.Call) and ast.unparse(x.func)=='cute.arch.sync_threads' for x in ast.walk(n))]
        self.assertEqual(len(direct),1)
        self.assertEqual(len(direct[0].body),1)
        self.assertEqual(ast.unparse(direct[0].body[0]),'cute.arch.sync_threads()')
        self.assertFalse(any(isinstance(n,(ast.Break,ast.Continue,ast.Return)) for n in ast.walk(loop)))
        self.assertFalse(any('sync_threads' in ast.unparse(n) for n in branch.body))


if __name__=='__main__':unittest.main()
