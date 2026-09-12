"""Direct-ring DFlash attention: startup, wrap, GQA and dynamic graph inputs."""
import importlib.util
import unittest

torch = None
if importlib.util.find_spec("torch"):
    import torch


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "requires CUDA")
class DraftAttentionTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
