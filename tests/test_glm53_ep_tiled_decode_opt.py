"""CPU source/ownership contracts; actual CuTe layout and GPU gates are separate."""
import ast
import copy
import random
import types
import unittest
from unittest.mock import patch
from test_glm53_ep_tiled_static import (
    SOURCE, STOCK, constants, extract, fake_torch, function, route_helpers,
)
from test_glm53_ep_tiled_sf6_word_unpack import u32, i32, word_function


def separate_stage(payload, seed):
    source = 1136
    dest = 86016  # A disjoint aligned raw SF stage, not a claimed CuTe offset.
    memory = bytearray(dest + 2048)
    memory[source:source+1552] = payload
    node = function('_sf_expand_fc2_out_of_place')
    class Steps(ast.NodeTransformer):
        def visit_Expr(self, n):
            if isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='_st_shared_i32':
                return [n, ast.Expr(ast.Yield(ast.Constant('store')))]
            if ast.unparse(n.value)=='self.sf_expand_barrier.arrive_and_wait()':
                return ast.Expr(ast.Yield(ast.Constant('publish')))
            return self.generic_visit(n)
    node=Steps().visit(node)
    loads=[];stores=[]
    def load(addr):
        loads.append(addr)
        assert source <= addr <= source+1548 and addr%4==0
        return int.from_bytes(memory[addr:addr+4],'little',signed=True)
    def store(addr,val):
        assert dest <= addr <= dest+2044 and addr%4==0
        stores.append(addr);memory[addr:addr+4]=u32(val).to_bytes(4,'little')
    ns=dict(Int32=i32,cutlass=types.SimpleNamespace(Uint32=u32),
            _ld_shared_i32_volatile=load,_st_shared_i32=store,
            _sf6_unpack_word=word_function())
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(SOURCE),'exec'),ns)
    threads=[ns[node.name](object(),source,dest,t) for t in range(128)]
    active=list(range(128));rng=random.Random(seed)
    while active:
        tid=rng.choice(active)
        event=next(threads[tid])
        if event=='publish':active.remove(tid)
    assert len(loads)==512 and sorted(stores)==list(range(dest,dest+2048,4))
    assert memory[source:source+1552]==payload
    return bytes(memory[dest:dest+2048])


class DecodeOptimizationTests(unittest.TestCase):
    def test_actual_word_unpack_is_identical_except_disjoint_reads_and_one_barrier(self):
        old=function('_sf_expand_stage');new=function('_sf_expand_fc2_out_of_place')
        # Retain exact byte arithmetic and original volatile read operations.
        old.body=old.body[2:]  # existing fallback and exact block-size guard
        old.args=copy.deepcopy(new.args);old.name=new.name
        class Normalize(ast.NodeTransformer):
            def visit_Call(self,n):
                if ast.unparse(n.func)=='_ld_shared_i32_volatile':
                    for x in ast.walk(n.args[0]):
                        if isinstance(x,ast.Name) and x.id=='stage_addr':x.id='source_addr'
                return self.generic_visit(n)
        old=Normalize().visit(old)
        barriers=[n for n in old.body if isinstance(n,ast.Expr)
                  and ast.unparse(n.value)=='self.sf_expand_barrier.arrive_and_wait()']
        self.assertEqual(len(barriers),2);old.body.remove(barriers[0])
        self.assertEqual(ast.dump(old,include_attributes=False),ast.dump(new,include_attributes=False))
        rng=random.Random(72)
        for base in range(256):
            payload=bytearray(rng.randbytes(1552));payload[1536]=base
            expected=bytes((base+((payload[i//2]>>(4*(i%2)))&15)
                         +16*((payload[1024+i//4]>>(2*(i%4)))&3))&255 for i in range(2048))
            self.assertEqual(separate_stage(payload,base),expected)

    def test_source_struct_offsets_keep_baseline_and_fit_exact_one_cta_cap(self):
        struct=next(n for n in ast.walk(function('kernel')) if isinstance(n,ast.ClassDef) and n.name=='Storage')
        original=next(n for n in ast.walk(function('kernel',STOCK)) if isinstance(n,ast.ClassDef) and n.name=='Storage')
        # Model documented struct alignment from actual field annotations.
        class Mem:
            def __class_getitem__(cls, values):return (values[0]*values[1],max(1,values[0]))
        class Align:
            def __class_getitem__(cls, values):return (values[0][0],values[1])
        cute=types.SimpleNamespace(struct=types.SimpleNamespace(Align=Align,MemRange=Mem),cosize=lambda x:x)
        cutlass=types.SimpleNamespace(Int32=4,Int64=8,Uint8=1,BFloat16=2,Float32=4)
        sizes=dict(a1_smem_staged=4096,b1_smem_staged=32768,sfa1_smem_staged=4096,
                   sfb1_smem_staged=4096,b2_smem_staged=32768,sfb2_smem_staged=4096,
                   a2_smem_layout=1024,sfa2_smem_layout=1024,epi1_smem_staged=2048,epi_smem_staged=4096)
        def layout(node,opt):
            owner=types.SimpleNamespace(fc1_stages=2,fc2_stages=2,ep_decode_opt=opt,tile_m=16,
                    a_dtype=1,b_dtype=1,sf_dtype=1,buffer_align_bytes=1024)
            ns=dict(self=owner,cute=cute,cutlass=cutlass,_COMPACT_STATIC_TILE_M=128,**sizes)
            offset=0;out={}
            for field in node.body:
                size,align=eval(compile(ast.Expression(field.annotation),str(SOURCE),'eval'),ns)
                offset=(offset+align-1)//align*align;out[field.target.id]=(offset,size);offset+=size
            return out,offset
        base,size=layout(original,False);off,off_size=layout(struct,False);on,on_size=layout(struct,True)
        self.assertEqual((size,off_size,on_size),(98304,98304,100352))
        self.assertEqual({k:v for k,v in off.items() if k!='sf2_packed_source'},base)
        self.assertEqual(on['sf2_packed_source'],(240,3104))
        source_start,source_size=on['sf2_packed_source']
        for name,(start,length) in on.items():
            if name!='sf2_packed_source':self.assertTrue(start+length<=source_start or start>=source_start+source_size,name)
        self.assertEqual(on['scatter_tok_cache'][1],16*4)
        self.assertEqual(on['scatter_weight_cache'][1],16*4)
        # All duplicate routes up to native8*top8=64 span four M16 tiles.
        # The actual epilogue clips rows before indexing either cache.
        for count in range(65):
            for tile in range(4):
                valid=max(0,min(16,count-16*tile))
                consumed=set()
                for lane in range(32):
                    for vec in range(lane,valid*8,32):consumed.add(vec//8)
                self.assertEqual(consumed,set(range(valid)))
                self.assertTrue(all(row<16 for row in consumed))
        self.assertEqual(on_size+1024,101376)
        self.assertEqual(on['sSFB2'][1],4096)
        for slot in range(2):
            self.assertEqual((source_start+1552*slot)%16,0)
        self.assertGreater(size,101376//2)

    def test_actual_capacity_guard_checks_cute_struct_size_before_allocation(self):
        check=extract('_check_ep_storage',{})
        owner=types.SimpleNamespace(smem_bytes=100352,smem_capacity=101376,threads_per_cta=160)
        check(owner,types.SimpleNamespace(size_in_bytes=lambda:100352))
        self.assertEqual(owner.ep_storage_bytes,100352)
        for actual in (98304,101408,102400):
            with self.assertRaises(ValueError):check(owner,types.SimpleNamespace(size_in_bytes=lambda:actual))
        owner.smem_capacity=101375
        with self.assertRaises(ValueError):check(owner,types.SimpleNamespace(size_in_bytes=lambda:100352))
        body=function('kernel').body
        allocation=next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and ast.unparse(n.value)=='smem.allocate(Storage)')
        self.assertEqual(ast.unparse(body[allocation-1].body[0]),'self._check_ep_storage(Storage)')
        self.assertFalse(any(isinstance(n,ast.Raise) for n in ast.walk(function('kernel'))))

    def test_exact_admission_latch_overrides_and_key_append_contract(self):
        ns=constants();route_helpers(ns);resolve=ns['ep_tiled_decode_opt']
        for latched in (False,True):
            ns['_EP_TILED_DECODE_OPT']=latched
            self.assertIs(ns['ep_tiled_decode_opt_enabled'](),latched)
            for m in range(1,33):
                for sf6 in (False,True):
                    self.assertEqual(resolve(m,sf6,None),latched and sf6 and m<=8)
                    self.assertEqual(resolve(m,sf6,True),sf6 and m<=8)
                    self.assertFalse(resolve(m,sf6,False))
        for bad in (0,1,'1',[],object()):
            with self.assertRaises(TypeError):resolve(6,True,bad)
        factory=function('ep_tiled_compile_spec');get=function('get_ep_tiled_decode_kernel')
        self.assertEqual(ast.literal_eval(factory.args.kw_defaults[-1]),False)
        for node in (factory,get):
            branches=[n for n in ast.walk(node) if isinstance(n,ast.If) and ast.unparse(n.test)=='decode_opt']
            self.assertEqual(len(branches),1)
            self.assertEqual(ast.unparse(branches[0].body[0]),'key += (EP_TILED_DECODE_OPT_CACHE_TAG,)')

    def test_fc2_consumers_and_producer_use_same_slot_without_changing_credit_or_release(self):
        text=ast.unparse(function('kernel'))
        self.assertIn('fc2_tma_bytes += self.sf2_stage_bytes',text)
        self.assertIn('sf2_dest = sf2_source_base_addr + fc2_prod_state.index * Int32(1552)',text)
        self.assertIn('self._sf_expand_fc2_out_of_place(sf2_source_base_addr + fc2_cons_state.index * Int32(1552), sfb2_base_addr + fc2_cons_state.index * Int32(2048), Int32(tidx))',text)
        fc2=next(n for n in ast.walk(function('kernel')) if isinstance(n,ast.For)
                 and ast.unparse(n.target)=='output_tile_idx'
                 and any(isinstance(x,ast.Call) and ast.unparse(x.func)=='self._sf_expand_fc2_out_of_place' for x in ast.walk(n)))
        src=ast.unparse(fc2)
        self.assertLess(src.index('fc2_pipeline.consumer_wait'),src.index('self._sf_expand_fc2_out_of_place'))
        self.assertLess(src.index('self._sf_expand_fc2_out_of_place'),src.index('cute.copy(smem_copy_SFB'))
        self.assertLess(src.index('cute.copy(smem_copy_SFB'),src.index('fc2_pipeline.consumer_release'))
        self.assertEqual(src.count('self.epilog_sync_barrier.arrive_and_wait()'),2)
        self.assertIn('_bulk_g2s(sf2_dest, sf2_source, Int32(1552), shared_ptr_to_u32(bar2))',text)
        # The baseline full-body oracle in existing static/route tests also
        # verifies the untouched FC1, FC2 math, release/tail and grid barriers.

    def test_actual_estimator_adds_only_absorbed_header_padding(self):
        class Parent:
            def _smem_bytes_estimate(self):return 98304
        ns=dict(Parent=Parent,_COMPACT_STATIC_TILE_M=128)
        cls=ast.ClassDef(name='Candidate',bases=[ast.Name('Parent',ast.Load())],keywords=[],
                        body=[function('_smem_bytes_estimate')],decorator_list=[])
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),str(SOURCE),'exec'),ns)
        obj=ns['Candidate']();obj.fc1_stages=obj.fc2_stages=2;obj.tile_m=16
        obj.ep_decode_opt=False;self.assertEqual(obj._smem_bytes_estimate(),98304)
        obj.ep_decode_opt=True;self.assertEqual(obj._smem_bytes_estimate(),100352)


if __name__=='__main__':unittest.main()
