"""Direct-ring DFlash attention: startup, wrap, GQA, dynamic graph inputs, and the split it is computed in.

The attention reads each KV head once for all of its queries and cuts the window into slices to fill the
machine (45차 §84). Two things have to hold for that to be the same attention: the slices' partial softmaxes
must combine into the whole one, and the answer must not depend on how many slices the window was cut into.
"""
import importlib.util
import unittest

torch = None
if importlib.util.find_spec("torch"):
    import torch


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "requires CUDA")
class DraftAttentionTests(unittest.TestCase):
    def test_captured_device_count_writes_only_the_accepted_prefix(self):
        from engine.kernels.draft_attention import write_draft_kv
        storage=torch.randn(3,5*2*127*8*128+512,device='cuda',dtype=torch.bfloat16)
        field=storage[:,:-512].view(3,5,2,127,8,128)
        slot=torch.tensor([1],device='cuda',dtype=torch.int64)
        positions=torch.arange(125,131,device='cuda',dtype=torch.int64)
        valid=torch.zeros((),device='cuda',dtype=torch.int64)
        k=torch.randn(6,2,128,device='cuda',dtype=torch.bfloat16)
        v=torch.randn_like(k)
        def call():write_draft_kv(field,slot,4,positions,k,v,valid=valid)
        call()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):call()
        try:
            for target,count in ((1,0),(2,1),(1,3),(2,6)):
                storage.normal_();saved=storage.clone();slot.fill_(target);valid.fill_(count)
                graph.replay()
                expected=saved[:,:-512].view_as(field)
                expected[target,4,0,positions[:count]%127,:2]=k[:count]
                expected[target,4,1,positions[:count]%127,:2]=v[:count]
                self.assertTrue(torch.equal(storage,saved))
        finally:graph.reset()

    def test_direct_arena_slot_changes_and_tp_head_writes(self):
        from engine.kernels.draft_attention import draft_attention, write_draft_kv
        # Real arena slots contain other state fields after the draft ring.
        storage=torch.randn(3,5*2*127*8*128+512,device='cuda',dtype=torch.bfloat16)
        field=storage[:,:-512].view(3,5,2,127,8,128)
        slot=torch.tensor([1],device='cuda',dtype=torch.int64)
        pos=torch.arange(3,device='cuda',dtype=torch.int64)
        context=torch.tensor(6,device='cuda',dtype=torch.int64)
        q=torch.randn(6,8,128,device='cuda',dtype=torch.bfloat16)
        k=torch.randn(6,2,128,device='cuda',dtype=torch.bfloat16)
        v=torch.randn_like(k)
        wk,wv=k[:3].contiguous(),v[:3].contiguous()
        def call():
            write_draft_kv(field,slot,3,pos,wk,wv)
            return draft_attention(q,k,v,field,context,slot=slot,layer=3)
        call()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):out=call()
        try:
            for target,start in ((2,125),(1,260),(2,7)):
                field.normal_();saved=field.clone();slot.fill_(target)
                pos.copy_(start+torch.arange(3,device='cuda'));context.fill_(start+3)
                wk.normal_();wv.normal_();graph.replay()
                expected=saved.clone()
                expected[target,3,0,pos%127,:2]=wk
                expected[target,3,1,pos%127,:2]=wv
                self.assertTrue(torch.equal(field,expected))
                ref=self.reference(q,k,v,field[target,3,:,:,:2].contiguous(),start+3)
                self.assertLess(((out.float()-ref.float()).norm()/ref.float().norm()).item(),.004)
        finally:graph.reset()

    def reference(self, q, k, v, ring, position):
        w = ring.shape[1]
        positions = torch.arange(position - w, position, device=q.device)
        keys = torch.cat((ring[0, positions % w], k)).repeat_interleave(q.shape[1] // k.shape[1], 1)
        values = torch.cat((ring[1, positions % w], v)).repeat_interleave(q.shape[1] // k.shape[1], 1)
        valid = torch.cat((positions >= 0, torch.ones(q.shape[0], device=q.device, dtype=torch.bool)))
        scores = torch.einsum("bhd,nhd->bhn", q.float(), keys.float()) * q.shape[2]**-.5
        scores.masked_fill_(~valid[None, None, :], -float("inf"))
        return torch.einsum("bhn,nhd->bhd", scores.softmax(-1), values.float()).bfloat16()

    def test_startup_wrap_graph_and_poisoned_unused_slots(self):
        from engine.kernels.draft_attention import draft_attention
        torch.manual_seed(413)
        for b, h, hk, w in ((1, 32, 8, 2048), (6, 32, 8, 2048), (24, 32, 8, 2048), (6, 8, 8, 127)):
            q = torch.randn(b, h, 128, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(b, hk, 128, device="cuda", dtype=torch.bfloat16)
            v = torch.randn_like(k)
            ring = torch.randn(2, w, hk, 128, device="cuda", dtype=torch.bfloat16)
            position = torch.tensor(0, device="cuda", dtype=torch.int64)
            draft_attention(q, k, v, ring, position)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = draft_attention(q, k, v, ring, position)
            try:
                for ctx in (0, 1, 7, w-1, w, w+1, 17*w+31):
                    position.fill_(ctx)
                    ring.normal_()
                    if ctx < w:
                        ring[:, ctx:] = float("nan")
                    saved = ring.clone()
                    graph.replay()
                    ref = self.reference(q, k, v, ring.nan_to_num(), ctx)
                    self.assertTrue(torch.isfinite(out).all().item())
                    rel = (out.float()-ref.float()).norm()/ref.float().norm()
                    self.assertLess(rel.item(), .004)
                    self.assertTrue(torch.allclose(ring, saved, equal_nan=True, rtol=0, atol=0))
            finally:
                graph.reset()


    def oracle(self, q, k, v, ring, ctx, layer, slot):
        """fp32 softmax attention over exactly the cells the kernel calls live."""
        b, h, dim = q.shape
        hk = k.shape[1]
        cells = ring.shape[3]
        band = ring[slot, layer]
        out = torch.zeros(b, h, dim, device=q.device, dtype=torch.float32)
        n = torch.arange(cells + b, device=q.device)
        absolute = ctx - cells + n
        live = (n >= cells) | (absolute >= 0)
        for head in range(h):
            kh = head // (h // hk)
            keys = torch.where((n < cells)[:, None],
                               band[0, (absolute + cells) % cells, kh].float(),
                               torch.cat([torch.zeros(cells, dim, device=q.device), k[:, kh].float()]))
            values = torch.where((n < cells)[:, None],
                                 band[1, (absolute + cells) % cells, kh].float(),
                                 torch.cat([torch.zeros(cells, dim, device=q.device), v[:, kh].float()]))
            scores = (keys[live] @ q[:, head].float().T) * dim ** -0.5           # [live, b]
            out[:, head] = (torch.softmax(scores, 0).T @ values[live])
        return out

    def test_the_slices_combine_into_the_attention_they_were_cut_from(self):
        from engine.kernels.draft_attention import draft_attention
        gen = torch.Generator(device="cuda").manual_seed(84)
        kind = dict(device="cuda", generator=gen, dtype=torch.float32)
        for b, h, hk, cells, ctx in ((6, 8, 2, 2056, 900), (6, 8, 2, 2056, 4), (1, 8, 2, 2056, 70_000),
                                     (6, 32, 8, 520, 300), (32, 4, 1, 72, 40)):
            with self.subTest(b=b, h=h, hk=hk, cells=cells, ctx=ctx):
                q = torch.randn(b, h, 128, **kind).bfloat16()
                k = torch.randn(b, hk, 128, **kind).bfloat16()
                v = torch.randn(b, hk, 128, **kind).bfloat16()
                ring = torch.randn(3, 2, 2, cells, hk, 128, **kind).bfloat16()
                position = torch.tensor(ctx, device="cuda", dtype=torch.int64)
                got = draft_attention(q, k, v, ring, position, slot=torch.tensor([1], device="cuda"), layer=1)
                want = self.oracle(q, k, v, ring, ctx, 1, 1)
                gap = (want - got.float()).norm(dim=-1) / want.norm(dim=-1).clamp_min(1e-6)
                self.assertLess(gap.max().item(), 2 ** -8)   # the answer is stored bf16: a step is the floor

    def test_a_whole_kv_head_of_queries_does_not_outgrow_a_cta(self):
        """One KV head can serve every query head: b * (h // hk) is 24 at TP=4 but 128 here, and an
        accumulator that wide will not fit. The tile is bounded and carried on the grid instead."""
        from engine.kernels.draft_attention import draft_attention
        gen = torch.Generator(device="cuda").manual_seed(85)
        kind = dict(device="cuda", generator=gen, dtype=torch.float32)
        q = torch.randn(32, 32, 128, **kind).bfloat16()
        k = torch.randn(32, 1, 128, **kind).bfloat16()
        v = torch.randn(32, 1, 128, **kind).bfloat16()
        ring = torch.randn(2, 1, 2, 136, 1, 128, **kind).bfloat16()
        got = draft_attention(q, k, v, ring, torch.tensor(90, device="cuda", dtype=torch.int64),
                              slot=torch.tensor([0], device="cuda"), layer=0)
        want = self.oracle(q, k, v, ring, 90, 0, 0)
        gap = (want - got.float()).norm(dim=-1) / want.norm(dim=-1).clamp_min(1e-6)
        self.assertLess(gap.max().item(), 2 ** -8)


class BatchedRingWriteTests(unittest.TestCase):
    """`observe_rows` is 21% of a decode step in production (45차: forward 63.5%, observe 21.4%, propose 15.1%)
    and its fast path wrote one row at a time, because the kernel required `slot.numel() == 1`. The batched form
    must put exactly the same bytes in the same cells -- a ring write that is merely close is a drafter that
    proposes from something the target never said."""

    @unittest.skipUnless(torch.cuda.is_available(), "the ring write is a device kernel")
    def test_it_writes_what_the_per_row_form_wrote(self):
        from engine.kernels.draft_attention import write_draft_kv, write_draft_kv_rows
        dev = torch.device("cuda")
        torch.manual_seed(5)
        S, L, CELLS, KV, D = 6, 3, 64, 4, 128
        n, t = 4, 6
        fresh = lambda: torch.zeros(S, L, 2, CELLS, KV, D, device=dev, dtype=torch.bfloat16)   # noqa: E731
        slots = torch.tensor([1, 3, 2, 5], device=dev)
        positions = torch.randint(0, 500, (n, t), device=dev, dtype=torch.int64)
        k = torch.randn(n, t, KV, D, device=dev, dtype=torch.bfloat16)
        v = torch.randn(n, t, KV, D, device=dev, dtype=torch.bfloat16)
        # a row that accepted nothing, one that accepted everything, and two in between
        valid = torch.tensor([6, 3, 0, 5], device=dev, dtype=torch.int64)
        one = fresh()
        for layer in range(L):
            for r in range(n):
                write_draft_kv(one, slots[r:r + 1], layer, positions[r], k[r], v[r], valid=valid[r])
        many = fresh()
        for layer in range(L):
            write_draft_kv_rows(many, slots, layer, positions, k, v, valid=valid)
        self.assertTrue(torch.equal(one, many))

    def test_it_refuses_a_shape_it_cannot_write(self):
        from engine.kernels.draft_attention import write_draft_kv_rows
        n, t, KV, D = 2, 3, 4, 8
        field = torch.zeros(4, 2, 2, 16, KV, D)
        slots = torch.zeros(n, dtype=torch.int64)
        positions = torch.zeros(n, t, dtype=torch.int64)
        k = v = torch.zeros(n, t, KV, D)
        valid = torch.zeros(n, dtype=torch.int64)
        for bad in (dict(slots=torch.zeros(n + 1, dtype=torch.int64)),
                    dict(valid=torch.zeros(n, dtype=torch.int32)),
                    dict(positions=torch.zeros(n, t, dtype=torch.int32)),
                    dict(layer=9)):
            args = dict(field=field, slots=slots, layer=0, positions=positions, k=k, v=v, valid=valid)
            args.update(bad)
            with self.assertRaises(ValueError):
                write_draft_kv_rows(args["field"], args["slots"], args["layer"], args["positions"],
                                    args["k"], args["v"], valid=args["valid"])


if __name__ == "__main__":
    unittest.main()
