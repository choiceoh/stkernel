"""Packed KV storage, bounded selected decode and runtime ABI; CPU only."""
import ast
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'overlay/modules/dsv41_model'))
sys.path.insert(0,str(ROOT/'probes'))
import dsv41_packed_kv as core
from dsv41_packed_kv_diff import arithmetic_unpack,scalar_decode_bits,oracle_quantize,e4m3_rne
from dsv41_dual_sparse_diff import online64,equal_output


class PackedKVContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads=torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def cache(self):
        cache=core.PackedKVCache(2,16,'cpu',20,1)
        y=torch.arange(2*9*256).remainder(256).to(torch.uint8).reshape(2,9,256)
        sf=torch.arange(2*9*32).remainder(80).to(torch.uint8).reshape(2,9,32)
        core.write_packed_kv(cache,y,sf,start_slot=0)
        return cache

    def inputs(self):
        rng=torch.Generator().manual_seed(54901)
        q=torch.randn(2,3,16,1024,generator=rng,dtype=torch.float32).to(torch.bfloat16)[...,::2]
        window=torch.randn(2,256,1024,generator=rng,dtype=torch.float32).to(torch.bfloat16)[:,::2,::2]
        sink=torch.linspace(-2,2,32,dtype=torch.float32)[::2]
        ids=torch.tensor([0,999,127,999,128,999,136,999,-1,999],dtype=torch.int32).expand(2,3,-1)[...,::2]
        return q,window,self.cache(),sink,ids

    def expected(self,args):
        q,w,cache,sink,ids=args
        dense=arithmetic_unpack(cache.packed[:,:9],cache.scales[:,:9])
        return online64(q,torch.cat((w,dense),1),sink,ids,512**-.5)

    def test_allocation_initial_zero_and_official_prefix_write(self):
        cache=core.PackedKVCache(2,7,'cpu',8,2)
        self.assertEqual(cache.storage_bytes,2*7*288)
        zero=core.unpack_kv_tile(cache.packed,cache.scales)
        self.assertTrue(torch.equal(zero.view(torch.int16),torch.zeros_like(zero,dtype=torch.int16)))
        y=torch.full((1,3,256),0xA3,dtype=torch.uint8)
        sf=torch.full((1,3,32),0x38,dtype=torch.uint8)
        core.write_packed_kv(cache,y,sf,start_slot=2,rows=2)
        self.assertEqual(cache.generation,1)
        self.assertTrue(torch.equal(cache.packed[0,2:4],y[0,:2]))
        self.assertEqual(int(cache.packed[1].sum()),0)
        self.assertEqual(int(cache.packed[0,4:].sum()),0)
        core.write_packed_kv(cache,y,sf,start_slot=7,rows=0)
        self.assertEqual(cache.generation,1)

    def test_raw_signed_scale_and_zero_bits_match_arithmetic(self):
        for code,scale in ((0,0xB8),(8,0xB8),(1,1),(15,126),(7,0xFE),(0,127),(15,255)):
            with self.subTest(code=code,scale=scale):
                y=torch.full((1,256),code|(code<<4),dtype=torch.uint8)
                sf=torch.full((1,32),scale,dtype=torch.uint8)
                actual=core.unpack_kv_tile(y,sf)
                expected=torch.tensor([scalar_decode_bits(code,scale)],dtype=torch.int32).to(torch.int16).view(torch.bfloat16).expand(1,512)
                equal_output(actual,expected,'raw decoder arithmetic')
        self.assertEqual(e4m3_rne(464),126)
        self.assertEqual(e4m3_rne(465),127)

    def test_nan_and_satfinite_producer_policies_both_restore(self):
        source=torch.full((1,2,512),torch.finfo(torch.bfloat16).max,dtype=torch.bfloat16)
        for policy,nan in (('nan',True),('satfinite',False)):
            with self.subTest(policy=policy):
                y,s,expected=oracle_quantize(source,overflow=policy)
                cache=core.PackedKVCache(1,2,'cpu',20,1)
                core.write_packed_kv(cache,y,s,start_slot=0)
                actual=core.unpack_kv_tile(cache.packed,cache.scales)
                equal_output(actual,expected,'declared CPU producer policy')
                self.assertEqual(bool(actual.isnan().all()),nan)

    def test_owner_metadata_external_mutation_and_generation_rejected(self):
        mutations=(lambda c:setattr(c,'owner_layer',8),lambda c:setattr(c,'ratio',2),
                   lambda c:setattr(c,'batch_size',1),lambda c:setattr(c,'generation',True),
                   lambda c:setattr(c,'packed',c.packed.clone()),lambda c:c.scales.fill_(56))
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                cache=self.cache();mutate(cache)
                with self.assertRaises(RuntimeError):cache.validate()

    def test_invalid_or_aliasing_writes_leave_owner_unchanged(self):
        cache=self.cache();before=cache.packed.clone()
        with self.assertRaisesRegex(ValueError,'alias'):
            core.write_packed_kv(cache,cache.packed[:,:1],cache.scales[:,:1],start_slot=0)
        with self.assertRaisesRegex(ValueError,'capacity'):
            core.write_packed_kv(cache,torch.zeros(2,9,256,dtype=torch.uint8),torch.zeros(2,9,32,dtype=torch.uint8),start_slot=8)
        with self.assertRaises(TypeError):
            core.write_packed_kv(cache,torch.zeros(2,1,256,dtype=torch.bfloat16),torch.zeros(2,1,32,dtype=torch.uint8),start_slot=0)
        with self.assertRaises(TypeError):
            core.write_packed_kv(cache,torch.zeros(2,1,256,dtype=torch.uint8),torch.zeros(2,1,32,dtype=torch.float8_e8m0fnu),start_slot=0)
        self.assertTrue(torch.equal(cache.packed,before))

    def test_strided_sources_and_active_prefix_ignore_nan_capacity_tail(self):
        args=list(self.inputs());cache=args[2]
        # A NaN tail is a valid official-byte write, but not part of width9.
        core.write_packed_kv(cache,torch.full((2,7,256),0xFF,dtype=torch.uint8),
                            torch.full((2,7,32),127,dtype=torch.uint8),start_slot=9)
        expected=self.expected(args)
        for chunk in (1,2,32):
            actual=core.packed_sparse_attn(*args,512**-.5,width=9,query_chunk_size=chunk)
            equal_output(actual,expected,'strided actual prefix')
            self.assertTrue(actual.is_contiguous())
        # Capacity row9+ is outside the original active index domain, even
        # though it has allocated and deliberately poisonous contents.
        bad=torch.tensor([[[137,-2,2**31-1]]],dtype=torch.int32).expand(2,3,-1)
        result=core.packed_sparse_attn(*args[:-1],bad,512**-.5,width=9)
        self.assertTrue((result==0).all())

    def test_actual_unpack_scratch_is_tile_bounded_and_never_full_cache(self):
        args=self.inputs();expected=self.expected(args);calls=[]
        original=core.unpack_kv_tile
        def inspect(packed,scales):
            self.assertLessEqual(packed.shape[0],2)
            self.assertEqual(packed.shape[1:],(64,256))
            calls.append(tuple(packed.shape))
            return original(packed,scales)
        with patch.object(core,'unpack_kv_tile',side_effect=inspect),patch.object(torch,'cat',side_effect=AssertionError('full cache concat')):
            actual=core.packed_sparse_attn(*args,512**-.5,width=9,query_chunk_size=2)
        equal_output(actual,expected,'bounded selected unpack')
        self.assertEqual(len(calls),3)
        with self.assertRaisesRegex(ValueError,'bounded'):
            core.unpack_kv_tile(torch.zeros(2049,256,dtype=torch.uint8),torch.zeros(2049,32,dtype=torch.uint8))

    def test_same_storage_writer_update_and_bf16_default_are_observed(self):
        args=list(self.inputs());cache=args[2]
        first=core.packed_sparse_attn(*args,512**-.5,width=9)
        pointers=(cache.packed.data_ptr(),cache.scales.data_ptr())
        y=cache.packed[:,:9].clone()^0x88
        sf=cache.scales[:,:9].clone()
        core.write_packed_kv(cache,y,sf,start_slot=0)
        previous=torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            second=core.packed_sparse_attn(*args,512**-.5,width=9)
        finally:torch.set_default_dtype(previous)
        equal_output(second,self.expected(args),'same address updated bytes')
        self.assertEqual(pointers,(cache.packed.data_ptr(),cache.scales.data_ptr()))
        self.assertFalse(torch.equal(first,second))

    def test_empty_and_all_invalid_preserve_sink_and_nan_qk_contract(self):
        q,w,cache,sink,ids=self.inputs()
        for slots in (0,65):
            selected=torch.full((2,3,slots),-1,dtype=torch.int32)
            for value in (0.,-float('inf'),-3e30,float('inf')):
                sinks=torch.full_like(sink,value)
                actual=core.packed_sparse_attn(q,w[:,:0],cache,sinks,selected,512**-.5,width=0)
                expected=online64(q,w[:,:0],sinks,selected,512**-.5)
                equal_output(actual,expected,'empty sink denominator')
        q[0,0,0,0]=torch.nan
        selected=torch.full((2,3,1),-1,dtype=torch.int32)
        actual=core.packed_sparse_attn(q,w,cache,sink,selected,512**-.5,width=9)
        self.assertTrue(actual[0,0,0].isnan().all())

    def test_width_dtype_and_query_chunk_guards(self):
        args=self.inputs()
        for width in (-1,True,17):
            with self.subTest(width=width),self.assertRaises(ValueError):
                core.packed_sparse_attn(*args,512**-.5,width=width)
        for kwargs in ({'query_chunk_size':0},{'query_chunk_size':True},{'query_chunk_size':33},{'backend':'auto'}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                core.packed_sparse_attn(*args,512**-.5,width=9,**kwargs)
        for index,value in ((0,args[0].float()),(3,args[3].to(torch.bfloat16)),(4,args[4].long())):
            copy=list(args);copy[index]=value
            with self.assertRaises((TypeError,ValueError)):core.packed_sparse_attn(*copy,512**-.5,width=9)

    def test_runtime_width_i64_strides_and_offline_signature_agree(self):
        tree=ast.parse((ROOT/'overlay/modules/dsv41_model/dsv41_packed_kv_triton.py').read_text())
        assigned={n.targets[0].id:n.value for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
        dims=ast.literal_eval(assigned['_RUNTIME_DIMS']);strides=ast.literal_eval(assigned['_RUNTIME_STRIDES'])
        self.assertEqual(set(dims),{'WIDTH_WINDOW','WIDTH_COMP','QUERIES','TOPK'})
        kernel=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_packed_sparse_kernel')
        actual=next(k.value for k in kernel.decorator_list[0].keywords if k.arg=='do_not_specialize')
        self.assertEqual(ast.unparse(actual),"_RUNTIME_INTS + ['SCALE']")
        annotations={a.arg:ast.unparse(a.annotation) if a.annotation else None for a in kernel.args.args}
        self.assertTrue(all(annotations[name]=='tl.int32' for name in dims))
        self.assertTrue(all(annotations[name]=='tl.int64' for name in strides))
        offline=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='offline_compile')
        text=ast.unparse(offline)
        self.assertIn("{name: 'i32' for name in _RUNTIME_DIMS}",text)
        self.assertIn("{name: 'i64' for name in _RUNTIME_STRIDES}",text)
        # Code-generation boundary only: this is not GPU numerical proof.
        # The BF16 register move must remain opaque and bit preserving, with
        # the original selected KV feeding both unmodified seeded dot calls.
        loop=next(n for n in kernel.body if isinstance(n,ast.For))
        kv_assignments=[n for n in loop.body if isinstance(n,ast.Assign)
                        and any(isinstance(t,ast.Name) and t.id=='kv' for t in n.targets)]
        expected=ast.parse("""
kv = tl.where(in_window[:, None], window, compressed)
kv = tl.inline_asm_elementwise('mov.b16 $0, $1;', constraints='=h,h', args=[kv], dtype=tl.bfloat16, is_pure=False, pack=1)
""").body
        self.assertEqual([ast.dump(n) for n in kv_assignments],[ast.dump(n) for n in expected])
        moves=[n for n in ast.walk(kernel) if isinstance(n,ast.Call)
               and ast.unparse(n.func)=='tl.inline_asm_elementwise']
        self.assertEqual(len(moves),1)
        dots=[ast.unparse(n) for statement in loop.body for n in ast.walk(statement) if isinstance(n,ast.Call)
              and ast.unparse(n.func)=='tl.dot']
        self.assertEqual(dots,["tl.dot(q, tl.trans(kv), initial, out_dtype=tl.float32)",
                               "tl.dot(probability.to(tl.bfloat16), kv, accumulated, out_dtype=tl.float32)"])


if __name__=='__main__':unittest.main()
