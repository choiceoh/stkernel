"""Dual-pool sparse arithmetic, bounded storage and ABI contracts; CPU only."""
import ast
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "overlay/modules/dsv41_model"))
sys.path.insert(0, str(ROOT / "probes"))
import dsv41_dual_sparse as core
from dsv41_dual_sparse_diff import online64, equal_output, check_actual_gather, reference_kernel


class DualSparseContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def inputs(self, *, batch=1, queries=1, heads=16, slots=65):
        generator = torch.Generator().manual_seed(5329)
        def bf(shape):
            return torch.randn(shape, generator=generator, dtype=torch.float32).to(torch.bfloat16)
        q, window, compressed = bf((batch, queries, heads, 512)), bf((batch, 128, 512)), bf((batch, 7, 512))
        sink = torch.linspace(-1, 1, heads, dtype=torch.float32)
        ids = torch.randint(135, (batch, queries, slots), generator=generator, dtype=torch.int32)
        ids[..., ::7] = -1
        return q, window, compressed, sink, ids

    def oracle(self, args):
        q, w, c, sink, ids = args
        return online64(q, torch.cat((w, c), 1), sink, ids, 512**-.5)

    def test_boundaries_and_noncontiguous_batch_query_head_inputs(self):
        q, w, c, sink, ids = self.inputs(batch=2, queries=3)
        # Each real row has a different physical batch/row stride from concat.
        wide = torch.empty((2, 260, 1024), dtype=torch.bfloat16)
        window = wide[:, 2:258:2, ::2]; window.copy_(w)
        comp_owner = torch.empty((2, 20, 1024), dtype=torch.bfloat16)
        compressed = comp_owner[:, 1:15:2, ::2]; compressed.copy_(c)
        q_owner = torch.empty((*q.shape[:-1], 1024), dtype=torch.bfloat16)
        query = q_owner[..., ::2]; query.copy_(q)
        ids_owner = torch.zeros((*ids.shape[:-1], ids.shape[-1]*2), dtype=torch.int32)
        selected = ids_owner[..., ::2]; selected.copy_(ids)
        sink_owner = torch.zeros(32, dtype=torch.float32)
        sinks = sink_owner[::2]; sinks.copy_(sink)
        for slots in (63, 64, 65):
            values = (query, window, compressed, sinks, selected[..., :slots])
            with self.subTest(slots=slots):
                check_actual_gather(core, window, compressed, selected[..., :slots])
                expected = self.oracle(values)
                for chunk in (1, 2, 32):
                    actual = core.dual_sparse_attn(*values, 512**-.5, query_chunk_size=chunk)
                    equal_output(actual, expected, "strided ABI/chunks")
                    self.assertTrue(actual.is_contiguous())

    def test_bf16_global_default_retains_fp32_accumulators(self):
        old = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            args = self.inputs()
            equal_output(core.dual_sparse_attn(*args, 512**-.5), self.oracle(args), "BF16 global default")
        finally:
            torch.set_default_dtype(old)
        self.assertEqual(torch.get_default_dtype(), old)

    def test_sink_is_once_after_staged_bf16_probability_rounding(self):
        args = self.inputs(slots=129)
        actual = core.dual_sparse_attn(*args, 512**-.5)
        equal_output(actual, self.oracle(args), "staged recurrence")
        q, w, c, sink, ids = args
        joined = torch.cat((w,c), 1)
        keys = joined[0, ids[0,0].clamp_min(0).long()].float()
        scores = (q[0,0].float() @ keys.T) * (512**-.5)
        scores[:, ids[0,0] == -1] = -torch.inf
        # A mathematically equivalent single full softmax changes rounding.
        probabilities = torch.softmax(torch.cat((scores,sink[:,None]), 1), dim=1)[:,:-1]
        wrong = (probabilities @ keys).to(torch.bfloat16)
        self.assertFalse(torch.equal(actual[0,0], wrong), "fixture must distinguish full-softmax substitution")

    def test_duplicates_are_not_deduplicated_and_order_is_retained(self):
        q, w, c, sink, _ = self.inputs()
        q.zero_()
        repeated = torch.tensor([[[0, 128, 128, 128, 1]]], dtype=torch.int32)
        unique = torch.tensor([[[0, 128, 1]]], dtype=torch.int32)
        args = (q,w,c,sink,repeated)
        actual = core.dual_sparse_attn(*args, 512**-.5)
        equal_output(actual,self.oracle(args),"duplicate slot weights")
        self.assertFalse(torch.equal(actual,core.dual_sparse_attn(q,w,c,sink,unique,512**-.5)))
        check_actual_gather(core,w,c,repeated)

    def test_empty_pools_and_empty_slots_preserve_extreme_sink_domain(self):
        q, w, c, sink, ids = self.inputs()
        for slots in (0, 65):
            selected = torch.full((1,1,slots),-1,dtype=torch.int32)
            for value, nan in ((0.,False),(-float("inf"),True),(-3e30,True),(float("inf"),False)):
                with self.subTest(slots=slots,sink=value):
                    sinks = torch.full_like(sink,value)
                    actual = core.dual_sparse_attn(q,w[:,:0],None,sinks,selected,512**-.5)
                    expected = online64(q,w[:,:0],sinks,selected,512**-.5)
                    equal_output(actual,expected,"empty sink contract")
                    self.assertEqual(bool(actual.isnan().all()),nan)

    def test_invalid_qk_seed_keeps_nan_times_zero_behavior(self):
        q,w,c,sink,ids = self.inputs(slots=1)
        q[0,0,0,0] = torch.nan
        ids.fill_(-1)
        actual = core.dual_sparse_attn(q,w,c,sink,ids,512**-.5)
        equal_output(actual,self.oracle((q,w,c,sink,ids)),"invalid seeded QK")
        self.assertTrue(actual[0,0,0].isnan().all())
        self.assertTrue((actual[0,0,1:] == 0).all())

    def test_extra_safe_invalid_ids_do_not_claim_original_equivalence(self):
        args = list(self.inputs(slots=5))
        args[-1] = torch.tensor([[[-2,2**31-1,135,0,128]]],dtype=torch.int32)
        safe = args[-1].clone(); safe[...,:3] = -1
        actual = core.dual_sparse_attn(*args,512**-.5)
        expected = core.dual_sparse_attn(*args[:-1],safe,512**-.5)
        equal_output(actual,expected,"extra-domain safe masking")
        with self.assertRaisesRegex(ValueError,"defined official"):
            self.oracle(args)

    def test_gather_scratch_is_bounded_and_no_full_pool_cat_occurs(self):
        args = self.inputs(batch=2,queries=3,slots=129)
        expected = self.oracle(args)
        calls=[]
        original=core._gather_tile
        def inspect(w,c,batch_ids,selected):
            self.assertLessEqual(selected.shape[0],2)
            self.assertEqual(selected.shape[1],64)
            out=original(w,c,batch_ids,selected)
            self.assertEqual(tuple(out[0].shape),(*selected.shape,512))
            calls.append(tuple(selected.shape))
            return out
        with patch.object(core,"_gather_tile",side_effect=inspect), patch.object(torch,"cat",side_effect=AssertionError("full pool concat")):
            actual=core.dual_sparse_attn(*args,512**-.5,query_chunk_size=2)
        equal_output(actual,expected,"bounded gather")
        self.assertEqual(len(calls),9)

    def test_same_address_mutation_is_visible_without_input_writes(self):
        args=list(self.inputs())
        before=[v.clone() for v in args]
        first=core.dual_sparse_attn(*args,512**-.5)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(args,before)))
        pointers=[v.data_ptr() for v in args]
        args[1].mul_(-1); args[2].add_(1); args[0].mul_(.5); args[3].add_(2)
        args[4][...,1]=128
        second=core.dual_sparse_attn(*args,512**-.5)
        equal_output(second,self.oracle(args),"current cache contents")
        self.assertEqual(pointers,[v.data_ptr() for v in args])
        self.assertFalse(torch.equal(first,second))
        self.assertNotIn(second.data_ptr(),pointers)

    def test_invalid_shapes_types_and_scalar_guards_fail_before_dispatch(self):
        values=self.inputs()
        variants=((0,values[0].float()),(1,values[1].float()),(3,values[3].to(torch.bfloat16)),
                  (4,values[4].long()),(0,values[0][...,:511]),(2,values[2][:,:,:511]))
        for index,value in variants:
            args=list(values); args[index]=value
            with self.subTest(index=index),self.assertRaises((TypeError,ValueError)):
                core.dual_sparse_attn(*args,512**-.5)
        for scale in (True,0.,-1.,float("nan"),float("inf"),1e100,1e-100):
            with self.subTest(scale=scale),self.assertRaises((TypeError,ValueError)):
                core.dual_sparse_attn(*values,scale)
        for kwargs in ({"backend":"auto"},{"query_chunk_size":True},{"query_chunk_size":0},{"query_chunk_size":33}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                core.dual_sparse_attn(*values,512**-.5,**kwargs)
        # Meta tensors exercise the actual signed-counter boundary without
        # allocating billions of IDs or executing a multi-million-tile loop.
        meta = [tensor.to("meta") for tensor in values]
        last_safe = 2**31 - 64
        meta[-1] = torch.empty((1, 1, last_safe), dtype=torch.int32, device="meta")
        core._contract(*meta, 512**-.5, 1)
        for slots in (last_safe + 1, 2**31 - 1):
            meta[-1] = torch.empty((1, 1, slots), dtype=torch.int32, device="meta")
            with self.subTest(slots=slots), self.assertRaisesRegex(ValueError, "selected slot count"):
                core.dual_sparse_attn(*meta, 512**-.5, backend="triton")

    def test_triton_widths_and_strides_are_runtime_not_prefill_specializations(self):
        tree=ast.parse((ROOT/"overlay/modules/dsv41_model/dsv41_dual_sparse_triton.py").read_text())
        assignments={n.targets[0].id:n.value for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
        dims=ast.literal_eval(assignments["_RUNTIME_DIMS"])
        strides=ast.literal_eval(assignments["_RUNTIME_STRIDES"])
        self.assertEqual(set(dims),{"WIDTH_WINDOW","WIDTH_COMP","QUERIES","TOPK"})
        kernel=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="_dual_sparse_kernel")
        no_specialize=next(k.value for k in kernel.decorator_list[0].keywords if k.arg=="do_not_specialize")
        self.assertEqual(ast.unparse(no_specialize),"_RUNTIME_INTS + ['SCALE']")
        annotations={a.arg:ast.unparse(a.annotation) if a.annotation else None for a in kernel.args.args}
        self.assertTrue(all(annotations[name]=="tl.int32" for name in dims))
        self.assertTrue(all(annotations[name]=="tl.int64" for name in strides))
        offline=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="offline_compile")
        self.assertIn("{name: 'i32' for name in _RUNTIME_DIMS}",ast.unparse(offline))
        self.assertIn("{name: 'i64' for name in _RUNTIME_STRIDES}",ast.unparse(offline))

    def test_untrusted_reference_source_is_refused_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/"kernel.py"
            source.write_text("raise AssertionError('must never run')\n")
            with self.assertRaisesRegex(ValueError,"SHA mismatch"):
                reference_kernel(source)


if __name__=="__main__":unittest.main()
