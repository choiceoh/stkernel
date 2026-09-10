"""Packed-key owner, collective and specialization boundaries; CPU only."""
import ast
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "overlay/modules/dsv41_model"))
from dsv41_packed_index import (PackedIndexCache, allocate_packed_index_cache,
                                packed_index_scores, unpack_index_tile, write_packed_index)


class PackedIndexerContracts(unittest.TestCase):
    def cache(self):
        cache = allocate_packed_index_cache(2, 16, device="cpu", owner_layer=20, ratio=1)
        y = torch.arange(2*9*64).remainder(256).to(torch.uint8).reshape(2,9,64)
        sf = torch.arange(2*9*4).remainder(4).add(124).to(torch.uint8).reshape(2,9,4)
        write_packed_index(cache, y, sf, start_slot=0)
        return cache

    def inputs(self):
        rng = torch.Generator().manual_seed(92010)
        # Strided q/weights/IDs must not be treated as contiguous. Each owner
        # has capacity16 while only9 positions belong to the active step.
        q = torch.randn(2,3,8,256,generator=rng).to(torch.bfloat16)[...,::2]
        w = torch.randn(2,3,16,generator=rng).to(torch.bfloat16)[...,::2]
        raw = torch.tensor([0,99,8,99,-1,99,2**31-1,99],dtype=torch.int32)
        ids = raw.expand(2,3,8)[...,::2]
        return q, self.cache(), w, ids

    def test_official_byte_write_offsets_and_zero_initialization(self):
        cache = allocate_packed_index_cache(2,7,device="cpu",owner_layer=8,ratio=2)
        self.assertEqual(cache.storage_bytes,2*7*68)
        self.assertTrue(torch.equal(unpack_index_tile(cache.packed,cache.scales),
                                    torch.zeros(2,7,128,dtype=torch.bfloat16)))
        y=torch.full((1,2,64),0xAB,dtype=torch.uint8)
        sf=torch.full((1,2,4),125,dtype=torch.uint8)
        write_packed_index(cache,y,sf,start_slot=3)
        self.assertEqual(cache.generation,1)
        self.assertTrue(torch.equal(cache.packed[0,3:5],y[0]))
        self.assertEqual(int(cache.packed[1].sum()),0)
        self.assertEqual(int(cache.packed[0,:3].sum()),0)
        write_packed_index(cache,y,sf,start_slot=7,rows=0)
        self.assertEqual(cache.generation,1)

    def test_owner_identity_generation_and_outside_writes_are_detected(self):
        mutations=(lambda c:setattr(c,"owner_layer",14),
                   lambda c:setattr(c,"generation",c.generation+1),
                   lambda c:setattr(c,"packed",c.packed.clone()),
                   lambda c:c.scales.fill_(127))
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                cache=self.cache(); mutate(cache)
                with self.assertRaises(RuntimeError): cache.validate()
        owner=torch.zeros(2*16*68,dtype=torch.uint8)
        with self.assertRaisesRegex(ValueError,"disjoint"):
            PackedIndexCache(owner[:2*16*64].view(2,16,64),owner[-2*16*4:].view(2,16,4),20,1)

    def test_invalid_writes_and_strided_owner_are_refused(self):
        cache=self.cache()
        before=cache.packed.clone()
        with self.assertRaisesRegex(ValueError,"alias"):
            write_packed_index(cache,cache.packed[:,:1],cache.scales[:,:1],start_slot=1)
        with self.assertRaises(ValueError):
            write_packed_index(cache,torch.zeros(2,9,64,dtype=torch.uint8),
                               torch.ones(2,9,4,dtype=torch.uint8),start_slot=8)
        with self.assertRaises(TypeError):
            write_packed_index(cache,torch.zeros(2,1,64,dtype=torch.bfloat16),
                               torch.ones(2,1,4,dtype=torch.uint8),start_slot=0)
        self.assertTrue(torch.equal(before,cache.packed))
        with self.assertRaisesRegex(ValueError,"contiguous"):
            PackedIndexCache(torch.zeros(2,32,64,dtype=torch.uint8)[:,::2],cache.scales,20,1)

    def test_strided_inputs_preserve_bf16_rounding_and_invalid_ids_zero(self):
        q,cache,w,ids=self.inputs()
        key=unpack_index_tile(cache.packed[:,:9],cache.scales[:,:9])
        full=(torch.einsum("bqhd,bsd->bqhs",q,key).relu()*w.unsqueeze(-1)).sum(2)
        valid=(ids>=0)&(ids<9)
        expected=full.gather(-1,ids.clamp(0,8).long()).masked_fill(~valid,0)
        for chunks in ((1,1),(2,3),(4,256)):
            with self.subTest(chunks=chunks):
                actual=packed_index_scores(q,cache,w,width=9,ids=ids,
                    query_chunk_size=chunks[0],position_chunk_size=chunks[1])
                self.assertTrue(torch.equal(actual,expected))

    def test_collective_once_inplace_and_zero_width_shape(self):
        q,cache,w,ids=self.inputs(); calls=[]
        original=packed_index_scores(q,cache,w,width=9,ids=ids)
        def reduce(value):
            calls.append((tuple(value.shape),value.dtype,value.is_contiguous()))
            value.mul_(2)
            return value
        actual=packed_index_scores(q,cache,w,width=9,ids=ids,reduce_fn=reduce)
        self.assertEqual(calls,[((2,3,4),torch.bfloat16,True)])
        self.assertTrue(torch.equal(actual,original*2))
        empty=[]
        value=packed_index_scores(q,cache,w,width=0,reduce_fn=lambda x:empty.append(tuple(x.shape)))
        self.assertEqual(tuple(value.shape),(2,3,0)); self.assertEqual(empty,[(2,3,0)])

    def test_collective_replacement_async_metadata_and_cache_mutation_fail(self):
        callbacks=(lambda v:v.clone(),lambda v:object(),lambda v:v.resize_(1),
                   lambda v:v.set_(v.clone()),lambda v:v.transpose_(0,1))
        for callback in callbacks:
            with self.subTest(callback=callback):
                q,cache,w,ids=self.inputs()
                with self.assertRaises((TypeError,ValueError)):
                    packed_index_scores(q,cache,w,width=9,ids=ids,reduce_fn=callback)
        q,cache,w,ids=self.inputs()
        def bad_cache(value): cache.packed.zero_()
        with self.assertRaisesRegex(RuntimeError,"outside"):
            packed_index_scores(q,cache,w,width=9,ids=ids,reduce_fn=bad_cache)

    def test_dtype_shapes_range_and_scratch_boundaries(self):
        q,cache,w,ids=self.inputs()
        for kwargs in ({"width":17},{"width":True},{"width":9,"query_chunk_size":True},
                       {"width":9,"position_chunk_size":0},
                       {"width":9,"query_chunk_size":4,"position_chunk_size":513},
                       {"width":9,"backend":"auto"}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                packed_index_scores(q,cache,w,**kwargs)
        for a,b,c in ((q.float(),w,ids),(q,w.float(),ids),(q,w,ids.long())):
            with self.assertRaises(ValueError): packed_index_scores(a,cache,b,width=9,ids=c)
        with self.assertRaisesRegex(ValueError,"bounded"):
            unpack_index_tile(torch.zeros(2049,64,dtype=torch.uint8),torch.ones(2049,4,dtype=torch.uint8))

    def test_decode_dimensions_are_unspecialized_runtime_arguments(self):
        tree=ast.parse((ROOT/"overlay/modules/dsv41_model/dsv41_packed_index_triton.py").read_text())
        runtime=next(n.value for n in tree.body if isinstance(n,ast.Assign)
                     and any(isinstance(t,ast.Name) and t.id=="_RUNTIME_INTS" for t in n.targets))
        names=ast.literal_eval(runtime)
        kernel=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="_packed_scores_kernel")
        decorator=kernel.decorator_list[0]
        configured=next(v.value for v in decorator.keywords if v.arg=="do_not_specialize")
        self.assertEqual(ast.unparse(configured),"_RUNTIME_INTS")
        for name in ("WIDTH","OUT_COLS","QUERIES"):
            self.assertIn(name,names)
            self.assertIsNone(next(a for a in kernel.args.args if a.arg==name).annotation)
        offline=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="offline_compile")
        text=ast.unparse(offline)
        self.assertIn("for name in _RUNTIME_INTS",text)
        self.assertIn("'i32'",text)


if __name__=="__main__":
    unittest.main()
