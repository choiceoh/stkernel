"""Actual served burst capture/rebind/readback with small deterministic GPU graphs.

The target is a numerical toy, the commit and serving adapter are real. The
single-GPU transport oracle is deliberately not real TP4 or model-quality proof.
"""
from types import SimpleNamespace as NS
import unittest
import torch


def toy_engine():
    from engine.base.graphs import DecodeGraphs
    from engine.kernels.bounded_graph import append_child
    from tests.test_engine_burst_decode import engine
    e = engine()
    e.eos = {1000000}
    e.caches.device = torch.device('cuda')
    e.caches.reset = lambda: None
    observed = torch.zeros(5, dtype=torch.int64, device='cuda')
    e.caches.draft_field = lambda: observed
    e.drafter.propose_rows = lambda field, slots, anchors, ctx, alive=None: anchors.unsqueeze(1) + 1
    e.drafter.observe_rows = lambda field, slots, positions, aux, count: field.index_add_(0, slots, count)
    e.net.comm.transport.eligible_max = lambda t: t.is_cuda and t.dtype == torch.int64
    buffers = {}
    def inputs(n, t, capacity):
        buffers[n, t] = torch.zeros(n, t, dtype=torch.int64, device='cuda')
        return dict(ids=torch.zeros(n*t, dtype=torch.int64, device='cuda'),
                    ctx=torch.zeros(n, dtype=torch.int64, device='cuda'),
                    seqs=torch.zeros(n, dtype=torch.int64, device='cuda'),
                    slots=torch.ones(n, dtype=torch.int64, device='cuda'))
    def forward(b):
        n = len(b['ctx'])
        out = buffers[n, 2]
        out.copy_(b['ids'].view(n, 2) + 1)
        aux = b['ids'].float().view(n*2, 1)
        return aux, aux, out
    graphs = DecodeGraphs(forward, inputs, [(n, 2, 64) for n in range(4, 0, -1)], append_child=append_child)
    def run_inputs(shape, ids, ctx, seqs, slots):
        def fill(b):
            for key, value in zip(('ids', 'ctx', 'seqs', 'slots'), (ids, ctx, seqs, slots)):
                b[key].copy_(value)
        return graphs.run(shape, fill)
    e.decode_graphs = NS(graphs=graphs, shape_for=lambda n, end: (n, 2, 64), run_inputs=run_inputs,
                         run_device=lambda shape, step, *args: run_inputs(shape, *args))
    greedy = DecodeGraphs(lambda b: b.clone(), lambda n, t: buffers[n, t], [(n, 2) for n in range(4, 0, -1)],
                           append_child=append_child)
    e.sampling_graphs = NS(greedy=greedy)
    e.observed = observed
    return e


@unittest.skipUnless(torch.cuda.is_available(), 'requires admitted GB10')
class ServedBurstCudaTests(unittest.TestCase):
    def test_boot_capture_rebinding_and_shared_chain_match_ordinary_serving(self):
        from engine.profiles.glm53.burst_decode import BurstDecode, BurstPending
        from engine.profiles.glm53.pipeline import AsyncDecode
        for limit in (2, 4):
            e, ref = toy_engine(), toy_engine()
            p = None
            try:
                p, normal = BurstDecode(e, limit), AsyncDecode(ref)
                e.observed.zero_(); ref.observed.zero_()  # boot's real observation warmup was not a request
                for rows, slots in (([1], [1]), ([1, 2, 3, 4], [1, 2, 3, 4]), ([4, 2], [4, 2]), ([3], [3])):
                    for _ in range(2):
                        pending = p.launch(rows, slots)
                        for _ in range(limit):
                            before = dict(ref.ctx)
                            done = normal.launch(rows, slots).resolve()
                            if any(done) or any(ref.ctx[s] // ref.F.block != before[s] // ref.F.block for s in rows):
                                break
                        pending.resolve()
                        self.assertEqual(e.tokens, ref.tokens)
                        self.assertEqual(e.ctx, ref.ctx)
                        self.assertEqual((e.accepted_total, e.drafted_total, e.steps),
                                         (ref.accepted_total, ref.drafted_total, ref.steps))
                        torch.testing.assert_close(e.observed, ref.observed, rtol=0, atol=0)
                        self.assertTrue(all(t > 0 for t in pending.iteration_seconds))
                        for record in pending.iteration_records:
                            self.assertEqual(set(record['stages_us']), {'forward', 'sample', 'commit', 'boundaries', 'observe', 'propose'})
                            self.assertTrue(all(t > 0 for t in record['stages_us'].values()))
                e.ends[2] = ref.ends[2] = set(range(1000000, 1000020))
                p.invalidate([2]); normal.invalidate([2])
                pending = p.launch([1, 2], [1, 2])
                self.assertNotIsInstance(pending, BurstPending)
                pending.resolve(); normal.launch([1, 2], [1, 2]).resolve()
                pending = p.launch([1], [1])
                self.assertIsInstance(pending, BurstPending)
                for _ in range(limit):
                    before = ref.ctx[1]
                    done = normal.launch([1], [1]).resolve()
                    if any(done) or ref.ctx[1] // ref.F.block != before // ref.F.block:
                        break
                pending.resolve()
                self.assertEqual(e.tokens, ref.tokens)
                self.assertEqual(e.ctx, ref.ctx)
                torch.testing.assert_close(e.observed, ref.observed, rtol=0, atol=0)
            finally:
                if p is not None:
                    p.close()
                for engine in (e, ref):
                    engine.sampling_graphs.greedy.close()
                    engine.decode_graphs.graphs.close()


if __name__ == '__main__':
    unittest.main()
