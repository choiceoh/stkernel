"""Direct KDA ring ownership, rollback, graph addressing and functional parity."""
import importlib.util
import unittest

from tests.image_kernels import PRESENT, REASON

if importlib.util.find_spec('torch'):
    import torch
else:
    torch = None


@unittest.skipUnless(torch is not None and torch.cuda.is_available() and PRESENT, 'requires CUDA; ' + REASON)
class KdaRingTests(unittest.TestCase):
    def setUp(self):
        from engine.profiles.glm53.lanes import served, reference
        from engine.kernels.kda.ring import recurrent_kda_ring
        self.run = recurrent_kda_ring
        self.functional = served(reference_for=('expert',)).kda_recurrent
        self.reference = reference().kda_recurrent
        torch.manual_seed(129612)

    def inputs(self, t, h=16, hv=16, k=128, v=128, dtype=None, state_dtype=None, cells=6):
        dtype = dtype or torch.bfloat16
        # Exercise real merged-conv token strides and sliced projection beta.
        def strided(heads, dim):
            return torch.randn(1,t,heads,dim*3,device='cuda',dtype=dtype)[..., :dim]
        q, kk, vv = strided(h,k), strided(h,k), strided(hv,v)
        g = torch.randn(1,t,hv,k,device='cuda',dtype=dtype)
        beta = torch.randn(1,t,hv*3,device='cuda',dtype=dtype)[..., :hv]
        a, bias = torch.randn(h,device='cuda')*.2, torch.randn(h*k,device='cuda')*.1
        width = hv*k*v
        backing = torch.randn(3*(cells*width+64)+64,device='cuda',dtype=state_dtype or torch.float32)*.1
        shape, stride = (3,cells,hv,k,v), (cells*width+64,width,k*v,v,1)
        ring = backing.as_strided(shape,stride,64)
        return (q,kk,vv,g,beta,a,bias), backing, ring

    def equal(self, x, y, label=''):
        a, b = x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8)
        if not torch.equal(a, b):
            positions = ((a.flatten() != b.flatten()).nonzero().flatten()[:12]
                         // x.element_size()).unique()
            values = [(i, x.flatten()[i].item(), y.flatten()[i].item()) for i in positions.tolist()]
            self.fail(f'{label}: unequal storage; scalar samples {values}')

    def expected(self, args, backing, ring, slot, ctx, lb=-5.):
        expected = backing.clone()
        target = expected.as_strided(ring.shape,ring.stride(),ring.storage_offset())
        initial = ring[slot,(ctx-1)%ring.shape[1]][None].float() if ctx else None
        out,states = self.functional(*args,initial,lb)
        for i,state in enumerate(states): target[slot,(ctx+i)%ring.shape[1]].copy_(state)
        return out,expected,states

    def test_every_snapshot_and_padding_with_wrapping_initial_row(self):
        for t in range(1,8):
            args,backing,ring = self.inputs(t, cells=max(6,t))
            for slot,ctx in ((0,0),(1,1),(2,ring.shape[1]-1),(1,ring.shape[1]),(2,32768)):
                with self.subTest(tokens=t,slot=slot,context=ctx):
                    out,expected,_ = self.expected(args,backing,ring,slot,ctx)
                    actual = self.run(*args,ring,slot,ctx,-5.)
                    self.equal(actual,out); self.equal(backing,expected)

    def test_graph_mutable_slot_context_and_rejected_drafts(self):
        for t in (1,6,7):
            args,backing,ring = self.inputs(t, cells=max(6,t))
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

    def test_rows_fold_matches_one_row_launches_and_replays(self):
        """net._kda folds a captured step's rows into one launch (45차, the C=4 question): its output and its
        ring writes are byte-equal to the one-row kernel run on each row in turn, and a captured graph
        replays it with whatever the slot and context vectors hold."""
        from engine.kernels.kda.ring import recurrent_kda_ring_rows
        rows = 3
        for t in (1, 6, 7):
            args, backing, ring = self.inputs(t * rows, cells=max(6, t))
            core, params = args[:5], args[5:]

            def one_by_one(slots, contexts):
                expected = backing.clone()
                view = expected.as_strided(ring.shape, ring.stride(), ring.storage_offset())
                outs = [self.run(*(x[:, i * t:(i + 1) * t] for x in core), *params, view, slots[i], contexts[i], -5.)
                        for i in range(rows)]
                return torch.cat(outs, dim=1), expected

            slots, contexts = [1, 0, 2], [0, t + 3, 32768]
            out, expected = one_by_one(slots, contexts)
            dev = tuple(torch.tensor(v, device='cuda', dtype=torch.int64) for v in (slots, contexts))
            actual = recurrent_kda_ring_rows(*args, ring, *dev, -5.)
            self.equal(actual, out, f't={t}'); self.equal(backing, expected, f't={t} ring')
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = recurrent_kda_ring_rows(*args, ring, *dev, -5.)
            try:
                for slots, contexts in (([2, 1, 0], [5, 0, 4095]), ([0, 1, 2], [t, t + 1, 1])):
                    for x in core:
                        x.normal_()
                    dev[0].copy_(torch.tensor(slots, device='cuda')); dev[1].copy_(torch.tensor(contexts, device='cuda'))
                    out, expected = one_by_one(slots, contexts)
                    graph.replay()
                    self.equal(actual, out, f't={t} replay'); self.equal(backing, expected, f't={t} replay ring')
            finally:
                graph.reset()
        with self.assertRaises(ValueError):
            args, backing, ring = self.inputs(7)
            recurrent_kda_ring_rows(*args, ring, torch.tensor([0, 1], device='cuda'), torch.tensor([0, 0], device='cuda'), -5.)

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

    def test_fp16_storage_rounds_only_at_ring_writes_and_replays_rollback(self):
        for t in (1, 7):
            args, backing, ring = self.inputs(t, state_dtype=torch.float16, cells=7)
            slot, ctx = (torch.tensor(x, device='cuda', dtype=torch.int64) for x in (1, 0))
            self.run(*args, ring, slot, ctx, -5.)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = self.run(*args, ring, slot, ctx, -5.)
            try:
                for physical, context in ((2, 0), (2, 7), (2, 9), (1, 1), (2, 11), (1, 32768)):
                    for x in args[:5]:
                        x.normal_()
                    if context == 0:
                        ring.fill_(float('nan'))
                    slot.fill_(physical); ctx.fill_(context)
                    out, expected, states = self.expected(args, backing, ring, physical, context)
                    graph.replay()
                    # Start from the same FP16 bits, do an entire step in FP32,
                    # then allow one FP16 rounding plus the independently gated
                    # FP32 arithmetic tolerance. Pointer dtype changes compiler
                    # tiling: round-to-nearest ties need not choose identical
                    # FP16 bits when FP32 sums differ by a few ULPs.
                    # Both outputs have already rounded to BF16; allow its
                    # representable spacing plus the FP32 recurrence tolerance.
                    output_tolerance = out.float().abs() * torch.finfo(out.dtype).eps + out.float().abs().max() * 3e-6
                    self.assertTrue(((actual.float() - out.float()).abs() <= output_tolerance).all(),
                                    f'T={t} slot={physical} ctx={context}: output beyond rounding')
                    target = expected.as_strided(ring.shape, ring.stride(), ring.storage_offset())
                    for i, state in enumerate(states):
                        row = (context + i) % ring.shape[1]
                        stored = ring[physical, row]
                        self.assertTrue(torch.isfinite(stored).all())
                        tolerance = state.abs() * 2**-11 + 2**-25 + state.abs().max() * 3e-6
                        self.assertTrue(((stored.float() - state).abs() <= tolerance).all(),
                                        f'T={t} slot={physical} ctx={context} state={i}: beyond FP16 rounding')
                        target[physical, row].copy_(stored)
                    # Every byte outside the written rows (including NaNs and
                    # slot padding) must still match exactly.
                    self.equal(backing, expected, f'T={t} slot={physical} ctx={context} ring')
                    self.assertTrue(torch.isfinite(actual).all())
                    if context == 0:
                        # NaNs prove zero-context masking and untouched bytes,
                        # but the later physical slots/rollback rows must hold
                        # valid histories (T=1 did not initialize seven rows).
                        backing.normal_(std=.1)
            finally:
                graph.reset()

    def test_fp16_initial_state_continues_prefill_and_functional_decode_in_fp32(self):
        from engine.profiles.glm53.lanes import served
        chunk = served(reference_for=('expert',)).kda_chunk
        for tokens, lane in ((7, self.functional), (128, chunk)):
            args, _, ring = self.inputs(tokens, state_dtype=torch.float16)
            initial = ring[1, 0][None].contiguous()
            saved = initial.clone()
            expected = lane(*args, initial.float(), -5.)
            actual = lane(*args, initial, -5.)
            self.equal(initial, saved)
            self.assertEqual(actual[1].dtype, torch.float32)
            for a, b, limit in zip(actual, expected, (.002, 2e-6)):
                self.assertTrue(torch.isfinite(a).all())
                error = (a.float()-b.float()).abs().max()/b.float().abs().max().clamp_min(1e-6)
                self.assertLessEqual(error.item(), limit)

    def test_invalid_layout_indices_and_overlapping_inputs(self):
        args,backing,ring = self.inputs(6)
        for bad in (ring.bfloat16(),ring.transpose(-1,-2),ring[:,:5],ring.cpu()):
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
