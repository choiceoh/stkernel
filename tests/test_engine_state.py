"""Direct slot addressing preserves rollback history and untouched arena bytes."""
import importlib.util
import unittest

torch = None
if importlib.util.find_spec("torch"):
    import torch


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "requires CUDA")
class StateGraphTests(unittest.TestCase):
    def test_history_masks_partial_tiles_and_zero_context_in_padded_slots(self):
        from engine.kernels.state import kda_history
        for width in (64, 1089, 16*128*128):
            storage = torch.randn(5*(6*width+64)+64, device="cuda")
            rec = storage.as_strided((5,6,1,width), (6*width+64,width,width,1), 64)
            conv = torch.randn(5,11,8, device="cuda", dtype=torch.bfloat16)
            slot = torch.tensor([3], device="cuda")
            for context in (0,1,6,32768):
                ctx = torch.tensor(context, device="cuda")
                hist, initial = kda_history(conv, rec, slot, ctx, 3)
                positions = context + torch.arange(-3,0, device="cuda")
                expected_hist = conv[3,:,positions.clamp_min(0)%8].masked_fill((positions<0)[None,:],0)
                expected_state = rec[3,(context-1)%6][None] if context else torch.zeros_like(initial)
                self.assertTrue(torch.equal(hist, expected_hist))
                self.assertTrue(torch.equal(initial, expected_state))
            rec.fill_(float("nan"))
            _, initial = kda_history(conv, rec, slot, torch.tensor(0, device="cuda"), 3)
            self.assertTrue(torch.equal(initial, torch.zeros_like(initial)))

    def test_ring_write_preserves_padding_with_strided_rows_and_long_inputs(self):
        from engine.kernels.state import write_ring
        width = 1089
        for tokens in (1,6,17):
            storage = torch.randn(5*(6*width+64)+64, device="cuda")
            expected = storage.clone()
            shape, stride = (5,6,width), (6*width+64,width,1)
            target = storage.as_strided(shape,stride,64)
            reference = expected.as_strided(shape,stride,64)
            source = torch.randn(tokens*2,width, device="cuda")[::2]
            slot = torch.tensor([3], device="cuda")
            ctx = torch.tensor(32768, device="cuda")
            for i in range(max(0,tokens-6), tokens):
                reference[3,(32768+i)%6].copy_(source[i])
            write_ring(source,target,slot,ctx)
            self.assertTrue(torch.equal(storage.view(torch.uint8), expected.view(torch.uint8)))

    def test_slot_remapping_zero_context_wrap_and_rejected_future_writes(self):
        from engine.base.arena import Arena
        from engine.profiles.glm53.caches import Glm53Caches, layout
        from engine.profiles.glm53.decode_graphs import GraphCaches
        from tests.test_engine_glm53 import tiny_facts
        F = tiny_facts()
        plan = layout(F, [0, 1])
        caches = [Glm53Caches(Arena(plan.nbytes(4, 3)), F, [0, 1], 4, 3) for _ in range(2)]
        actual, expected = caches
        for length in (1, 6):
            for field in actual._fields.values():
                field.copy_(torch.randn_like(field.float()).to(field.dtype))
            initial = actual.state.clone()
            seqs = torch.zeros(1, device="cuda", dtype=torch.int64)
            slot = torch.ones(1, device="cuda", dtype=torch.int64)
            ctx = torch.zeros((), device="cuda", dtype=torch.int64)
            inputs = torch.randn(length, 3*F.kda_heads_local*F.kda_dim, device="cuda", dtype=torch.bfloat16)
            states = torch.randn(length, F.kda_heads_local, F.kda_dim, F.kda_dim, device="cuda")
            keys = torch.randn(length, F.idx_dim, device="cuda", dtype=torch.bfloat16)
            gates = -keys
            view = GraphCaches(actual, seqs, slot, F.block*4)

            def step():
                hist, state = view.kda_history(0, 0, ctx)
                tail = view.tail(1, 0)
                view.write_conv(0, 0, ctx, inputs)
                view.write_rec(0, 0, ctx, states)
                view.write_tail(1, 0, ctx, keys, gates)
                return hist, state, tail

            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                step()
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                hist, state, tail = step()
            actual.state.copy_(initial); expected.state.copy_(initial)
            try:
                for physical, context in ((2, 0), (1, 2), (3, 7), (2, 8), (3, 8), (1, 4095), (1, 4096)):
                    conv, rec = expected.kda(0, physical)
                    positions = context + torch.arange(-(F.conv-1), 0, device="cuda")
                    ref_hist = conv[:, positions.clamp_min(0) % conv.shape[-1]].masked_fill((positions < 0)[None, :], 0)
                    ref_state = rec[(context-1) % rec.shape[0]][None].clone() if context else torch.zeros_like(state)
                    ref_tail = expected.tail(1, physical).clone()
                    slot.fill_(physical); ctx.fill_(context)
                    graph.replay()
                    self.assertTrue(torch.equal(hist, ref_hist), (length, physical, context, "conv history"))
                    self.assertTrue(torch.equal(state, ref_state), (length, physical, context, "initial state"))
                    self.assertTrue(torch.equal(tail, ref_tail))
                    for i in range(length):
                        conv[:, (context+i) % conv.shape[-1]] = inputs[i]
                        rec[(context+i) % rec.shape[0]] = states[i]
                        expected.tail(1, physical)[(context+i) % ref_tail.shape[0], 0] = keys[i]
                        expected.tail(1, physical)[(context+i) % ref_tail.shape[0], 1] = gates[i]
                    self.assertTrue(torch.equal(actual.state, expected.state), (length, physical, context))
                    self.assertFalse(hasattr(view, "fields"), "graph retained copies of complete state rings")
            finally:
                graph.reset()


if __name__ == "__main__":
    unittest.main()
