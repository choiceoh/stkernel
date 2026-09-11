"""Direct KDA ring ownership, rollback, graph addressing and functional parity."""
import importlib.util
import unittest

if importlib.util.find_spec('torch'):
    import torch
else:
    torch = None


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), 'requires CUDA')
class KdaRingTests(unittest.TestCase):
    def setUp(self):
        from engine.profiles.glm53.lanes import served, reference
        from engine.kernels.kda.ring import recurrent_kda_ring
        self.run = recurrent_kda_ring
        self.functional = served(reference_for=('expert',)).kda_recurrent
        self.reference = reference().kda_recurrent
        torch.manual_seed(129612)

    def inputs(self, t, h=16, hv=16, k=128, v=128, dtype=None):
        dtype = dtype or torch.bfloat16
        # Exercise real merged-conv token strides and sliced projection beta.
        def strided(heads, dim):
            return torch.randn(1,t,heads,dim*3,device='cuda',dtype=dtype)[..., :dim]
        q, kk, vv = strided(h,k), strided(h,k), strided(hv,v)
        g = torch.randn(1,t,hv,k,device='cuda',dtype=dtype)
        beta = torch.randn(1,t,hv*3,device='cuda',dtype=dtype)[..., :hv]
        a, bias = torch.randn(h,device='cuda')*.2, torch.randn(h*k,device='cuda')*.1
        width = hv*k*v
        backing = torch.randn(3*(6*width+64)+64,device='cuda')*.1
        shape, stride = (3,6,hv,k,v), (6*width+64,width,k*v,v,1)
        ring = backing.as_strided(shape,stride,64)
        return (q,kk,vv,g,beta,a,bias), backing, ring

    def equal(self, x, y):
        self.assertTrue(torch.equal(x.contiguous().view(torch.uint8),y.contiguous().view(torch.uint8)))

    def expected(self, args, backing, ring, slot, ctx, lb=-5.):
        expected = backing.clone()
        target = expected.as_strided(ring.shape,ring.stride(),ring.storage_offset())
        initial = ring[slot,(ctx-1)%6][None] if ctx else None
        out,states = self.functional(*args,initial,lb)
        for i,state in enumerate(states): target[slot,(ctx+i)%6].copy_(state)
        return out,expected,states

    def test_every_snapshot_and_padding_with_wrapping_initial_row(self):
        for t in range(1,7):
            args,backing,ring = self.inputs(t)
            for slot,ctx in ((0,0),(1,1),(2,5),(1,6),(2,32768)):
                with self.subTest(tokens=t,slot=slot,context=ctx):
                    out,expected,_ = self.expected(args,backing,ring,slot,ctx)
                    actual = self.run(*args,ring,slot,ctx,-5.)
                    self.equal(actual,out); self.equal(backing,expected)

    def test_graph_mutable_slot_context_and_rejected_drafts(self):
        for t in (1,6):
            args,backing,ring = self.inputs(t)
            slot,ctx = (torch.tensor(x,device='cuda',dtype=torch.int64) for x in (1,0))
            self.run(*args,ring,slot,ctx,-5.)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph): actual = self.run(*args,ring,slot,ctx,-5.)
            try:
                # Retry at accepted prefix positions, including overwritten last
                # initial row when T==R. State written after EVERY token is used.
                for physical,context in ((2,0),(2,t),(2,t+2),(1,1),(2,t+3),(1,4095),(1,4096)):
                    for x in args[:5]: x.normal_()
                    slot.fill_(physical);ctx.fill_(context)
                    out,expected,_ = self.expected(args,backing,ring,physical,context)
                    graph.replay()
                    self.equal(actual,out);self.equal(backing,expected)
            finally: graph.reset()

    def test_tail_dimensions_grouped_heads_and_float_types(self):
        for dtype in (torch.bfloat16,torch.float16,torch.float32):
            for t in (1,6):
                args,backing,ring = self.inputs(t,2,4,33,17,dtype)
                for context in (0,5):
                    out,expected,states = self.expected(args,backing,ring,1,context)
                    initial = ring[1,(context-1)%6][None].clone() if context else None
                    actual = self.run(*args,ring,1,context,-5.)
                    self.equal(actual,out);self.equal(backing,expected)
                    q,k,v,g,beta,a,bias = args
                    ref,ref_states = self.reference(q.repeat_interleave(2,dim=2),k.repeat_interleave(2,dim=2),
                        v,g,beta,a.repeat_interleave(2),bias.view(2,33).repeat_interleave(2,dim=0).flatten(),initial,-5.)
                    # Independent torch recurrence, including every FP32 state.
                    for x,y,limit in ((actual,ref,.01),(states,ref_states,3e-6)):
                        rel=(x.float()-y.float()).abs().max()/y.float().abs().max().clamp_min(1e-6)
                        self.assertLess(rel.item(),limit)

    def test_zero_context_masks_nan_and_device_int32(self):
        args,backing,ring = self.inputs(6)
        ring.fill_(float('nan'))
        out,expected,_ = self.expected(args,backing,ring,1,0)
        actual = self.run(*args,ring,torch.tensor([1],device='cuda',dtype=torch.int32),
                          torch.tensor(0,device='cuda',dtype=torch.int32),-5.)
        self.equal(actual,out);self.equal(backing,expected)

    def test_disjoint_slots_on_two_cuda_streams(self):
        args,backing,ring = self.inputs(6)
        other,_,_ = self.inputs(6)
        # JIT before concurrent execution, then restore the starting arena.
        original = backing.clone()
        self.run(*args,ring,1,5,-5.)
        backing.copy_(original)
        out1,expected,_ = self.expected(args,backing,ring,1,5)
        expected_ring = expected.as_strided(ring.shape,ring.stride(),ring.storage_offset())
        out2,expected,_ = self.expected(other,expected,expected_ring,2,32768)
        streams = [torch.cuda.Stream(),torch.cuda.Stream()]
        for stream in streams:stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(streams[0]):actual1 = self.run(*args,ring,1,5,-5.)
        with torch.cuda.stream(streams[1]):actual2 = self.run(*other,ring,2,32768,-5.)
        for stream in streams:torch.cuda.current_stream().wait_stream(stream)
        self.equal(actual1,out1);self.equal(actual2,out2);self.equal(backing,expected)

    def test_invalid_layout_indices_and_overlapping_inputs(self):
        args,backing,ring = self.inputs(6)
        for bad in (ring.half(),ring.transpose(-1,-2),ring[:,:5],ring.cpu()):
            with self.assertRaises(ValueError):self.run(*args,bad,1,0,-5.)
        for slot,ctx in ((3,0),(-1,0),(1,-1),(1,torch.tensor(0,device='cuda')),
                         (torch.tensor(1),torch.tensor(0))):
            with self.assertRaises(ValueError):self.run(*args,ring,slot,ctx,-5.)
        invalid=list(args); invalid[3]=ring[1,:6,:,:,0].unsqueeze(0)
        with self.assertRaises(ValueError):self.run(*invalid,ring,1,0,-5.)

    def test_reference_bisect_disables_direct_ring_lane(self):
        from engine.profiles.glm53.lanes import served
        self.assertIsNone(served(reference_for=('expert','kda_recurrent')).kda_recurrent_ring)
        self.assertIsNone(served(reference_for=('expert','kda_recurrent_ring')).kda_recurrent_ring)


if __name__ == '__main__': unittest.main()
