"""Direct slot addressing preserves rollback history and untouched arena bytes."""
import importlib.util
import unittest

torch = None
if importlib.util.find_spec("torch"):
    import torch


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "requires CUDA")
class StateGraphTests(unittest.TestCase):
    def test_slot_remapping_zero_context_wrap_and_rejected_future_writes(self):
        from engine.base.arena import Arena
        from engine.profiles.glm53.caches import Glm53Caches, layout
        from engine.profiles.glm53.decode_graphs import GraphCaches
        from test_engine_glm53 import tiny_facts
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
