"""Short conv reads/writes only the chosen slot, including graph rollback."""
import importlib.util
import unittest

from tests.image_kernels import PRESENT, REASON

if importlib.util.find_spec('torch'):
    import torch
else:
    torch = None


@unittest.skipUnless(torch is not None and torch.cuda.is_available() and PRESENT, 'requires CUDA; ' + REASON)
class ConvRingTests(unittest.TestCase):
    def setUp(self):
        from engine.kernels.causal_conv_ring import causal_conv1d_ring
        from engine.kernels.causal_conv_single import causal_conv1d_single
        self.run, self.old = causal_conv1d_ring, causal_conv1d_single
        torch.manual_seed(129620)

    def inputs(self, t, c=6144, k=4, dtype=None):
        dtype=dtype or torch.bfloat16
        x=torch.randn(t,c+272,device='cuda',dtype=dtype)[:,:c]
        w=torch.randn(c,k,device='cuda')
        storage=torch.randn(3*(c*8+64)+64,device='cuda',dtype=dtype)
        ring=storage.as_strided((3,c,8),(c*8+64,8,1),64)
        return x,w,storage,ring

    def exact(self, x, y):
        self.assertEqual(x.shape,y.shape)
        self.assertEqual(x.dtype,y.dtype)
        self.assertTrue(torch.equal(x.contiguous().view(torch.uint8),y.contiguous().view(torch.uint8)))

    def expected(self, x, w, storage, ring, slot, ctx):
        pos=ctx+torch.arange(-(w.shape[1]-1),0,device='cuda')
        hist=ring[slot,:,pos.clamp_min(0)%8].masked_fill((pos<0)[None],0)
        out,_=self.old(x,w,hist)
        target=storage.clone()
        view=target.as_strided(ring.shape,ring.stride(),ring.storage_offset())
        for i in range(x.shape[0]):view[slot,:,(ctx+i)%8]=x[i]
        return out,target,hist

    def test_projection_slices_all_short_lengths_and_ring_wrap(self):
        for t in range(1,9):
            x,w,storage,ring=self.inputs(t)
            for slot,ctx in ((0,0),(1,1),(2,2),(1,3),(2,7),(1,8),(2,32768)):
                with self.subTest(tokens=t,slot=slot,context=ctx):
                    expected,raw,_=self.expected(x,w,storage,ring,slot,ctx)
                    before=(x.clone(),w.clone())
                    actual=self.run(x,w,ring,slot,ctx)
                    self.exact(actual,expected);self.exact(storage,raw)
                    self.exact(x,before[0]);self.exact(w,before[1])

    def test_strides_tails_widths_types_and_independent_reference(self):
        from engine.modules.causal_conv import causal_conv1d
        for dtype in (torch.bfloat16,torch.float16,torch.float32):
            for k in (2,3,4):
                for t in (1,6):
                    x,w,storage,ring=self.inputs(t,67,k,dtype)
                    x=torch.randn(2*t,134,device='cuda',dtype=dtype)[::2,::2]
                    w=torch.randn(134,2*k,device='cuda')[::2,::2]
                    expected,raw,hist=self.expected(x,w,storage,ring,1,7)
                    actual=self.run(x,w,ring,1,7)
                    self.exact(actual,expected);self.exact(storage,raw)
                    reference,_=causal_conv1d(x,w,initial_state=hist)
                    torch.testing.assert_close(actual.float(),reference.float(),atol=2e-3,rtol=.008)
        for dtype in (torch.bfloat16,torch.float16):
            x,w,storage,ring=self.inputs(6,67,dtype=dtype);w=w.to(dtype)
            expected,raw,_=self.expected(x,w,storage,ring,1,2)
            self.exact(self.run(x,w,ring,1,2),expected);self.exact(storage,raw)

    def test_graph_slot_context_and_rejected_suffix(self):
        for t in (1,6):
            x,w,storage,ring=self.inputs(t)
            slot=torch.tensor(1,device='cuda',dtype=torch.int64)
            ctx=torch.tensor(0,device='cuda',dtype=torch.int64)
            self.run(x,w,ring,slot,ctx)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):actual=self.run(x,w,ring,slot,ctx)
            try:
                for physical,context in ((2,0),(2,t),(2,t+1),(2,t+2),(1,1),(1,4095),(2,4096)):
                    x.normal_();w.normal_();slot.fill_(physical);ctx.fill_(context)
                    expected,raw,_=self.expected(x,w,storage,ring,physical,context)
                    graph.replay();self.exact(actual,expected);self.exact(storage,raw)
            finally:graph.reset()

    def test_zero_context_nan_and_bf16_patterns(self):
        x,w,storage,ring=self.inputs(6)
        ring.fill_(float('nan'))
        expected,raw,_=self.expected(x,w,storage,ring,1,0)
        actual=self.run(x,w,ring,torch.tensor([1],device='cuda',dtype=torch.int32),
                        torch.tensor(0,device='cuda',dtype=torch.int32))
        self.exact(actual,expected);self.exact(storage,raw)
        x,w,storage,ring=self.inputs(8,8192)
        values=torch.arange(65536,device='cuda',dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
        x.copy_(torch.where(torch.isfinite(values),values,0.).view_as(x))
        w.copy_(torch.tensor([1.,-.5,.25,-.125],device='cuda'))
        expected,raw,_=self.expected(x,w,storage,ring,1,0)
        self.exact(self.run(x,w,ring,1,0),expected);self.exact(storage,raw)
        for value in (-100.,-88.,-0.,0.,88.,100.):
            x.fill_(value)
            expected,raw,_=self.expected(x,w,storage,ring,1,8)
            self.exact(self.run(x,w,ring,1,8),expected);self.exact(storage,raw)

    def test_rows_fold_matches_one_row_launches_and_replays(self):
        """net._kda folds a captured step's rows into one conv launch (45차, the C=4 question): byte-equal to the
        one-row kernel on each row in turn, replayed with whatever the slot and context vectors hold."""
        from engine.kernels.causal_conv_ring import causal_conv1d_ring_rows
        rows=3
        for t in (1,6,8):
            x,w,storage,ring=self.inputs(t*rows)

            def one_by_one(slots,contexts):
                raw=storage.clone()
                view=raw.as_strided(ring.shape,ring.stride(),ring.storage_offset())
                return torch.cat([self.run(x[i*t:(i+1)*t],w,view,slots[i],contexts[i]) for i in range(rows)]),raw

            slots,contexts=[1,0,2],[0,t+1,32768]
            expected,raw=one_by_one(slots,contexts)
            dev=tuple(torch.tensor(v,device='cuda',dtype=torch.int64) for v in (slots,contexts))
            actual=causal_conv1d_ring_rows(x,w,ring,*dev)
            self.exact(actual,expected);self.exact(storage,raw)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):actual=causal_conv1d_ring_rows(x,w,ring,*dev)
            try:
                for slots,contexts in (([2,1,0],[3,0,4095]),([0,2,1],[t,1,t+2])):
                    x.normal_();w.normal_()
                    dev[0].copy_(torch.tensor(slots,device='cuda'));dev[1].copy_(torch.tensor(contexts,device='cuda'))
                    expected,raw=one_by_one(slots,contexts)
                    graph.replay();self.exact(actual,expected);self.exact(storage,raw)
            finally:graph.reset()
        with self.assertRaises(ValueError):
            x,w,_,ring=self.inputs(7)
            causal_conv1d_ring_rows(x,w,ring,torch.tensor([0,1],device='cuda'),torch.tensor([0,0],device='cuda'))

    def test_disjoint_slots_on_two_streams(self):
        x,w,storage,ring=self.inputs(6)
        other=x.neg()
        original=storage.clone();self.run(x,w,ring,1,7);storage.copy_(original)
        expected1,raw,_=self.expected(x,w,storage,ring,1,7)
        view=raw.as_strided(ring.shape,ring.stride(),ring.storage_offset())
        expected2,raw,_=self.expected(other,w,raw,view,2,2)
        streams=[torch.cuda.Stream(),torch.cuda.Stream()]
        for s in streams:s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(streams[0]):actual1=self.run(x,w,ring,1,7)
        with torch.cuda.stream(streams[1]):actual2=self.run(other,w,ring,2,2)
        for s in streams:torch.cuda.current_stream().wait_stream(s)
        self.exact(actual1,expected1);self.exact(actual2,expected2);self.exact(storage,raw)

    def test_reject_invalid_contract_and_input_aliases(self):
        x,w,_,ring=self.inputs(6,67)
        for bad in (ring.float(),ring.transpose(-1,-2),ring[...,:4],ring.cpu()):
            with self.assertRaises(ValueError):self.run(x,w,bad,1,0)
        for slot,ctx in ((3,0),(-1,0),(1,-1),(1,torch.tensor(0,device='cuda')),
                         (torch.tensor(1),torch.tensor(0))):
            with self.assertRaises(ValueError):self.run(x,w,ring,slot,ctx)
        for bad in (x[:0],x.repeat(2,1),x.to(torch.int32),ring[1,:,:6].T):
            with self.assertRaises(ValueError):self.run(bad,w,ring,1,0)

    def test_declared_reference_conv_disables_direct_ring(self):
        from engine.profiles.glm53.lanes import served,reference
        self.assertIsNone(reference().conv_ring)
        self.assertIsNone(served(reference_for=('expert','conv_prefill')).conv_ring)
        self.assertIsNone(served(reference_for=('expert','conv_ring')).conv_ring)


if __name__=='__main__':unittest.main()
