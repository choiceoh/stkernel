"""CPU serving/capture contracts; the Triton numerical case is CUDA-only."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from engine.kernels.draft_select import walk_scores
from engine.modules.draft_boundary import EMPTY, tensor
from engine.profiles.glm53.draft_tuning import DraftTuning, prepare_store
from engine.profiles.glm53.draft_policy import DraftPolicy, decode_name
from engine.profiles.glm53.drafter import Drafter, DrafterFacts, store_name
from engine.profiles.glm53.decode_graphs import DrafterDecodeGraphs


def parts():
    return (torch.tensor([[[2., 0.], [0., 0.], [0., 0.]]]),
            torch.tensor([[[1, 2], [1, 2], [1, 2]]]), torch.tensor([0]),
            torch.ones(1, 3, 1), torch.tensor([[3.], [0.], [3.], [0.], [-3.]]),
            torch.tensor([[0.], [0.], [1.], [0.], [0.]]))


class WalkTests(unittest.TestCase):
    def test_fc_bias_norm_preserves_fp32_addition_and_signed_weight(self):
        from engine.kernels.common.norm_rope import norm
        x = torch.tensor([[1., 1.], [2., -1.]]).bfloat16()
        bias = torch.tensor([1 / 256, -1 / 256])
        gamma = torch.tensor([-2., 3.]).bfloat16()
        expected_input = x.double() + bias.double()
        expected = (expected_input * torch.rsqrt(expected_input.square().mean(-1, keepdim=True) + 1e-6))
        expected = expected.bfloat16() * gamma
        actual = norm(x, gamma, 1e-6, bias=bias)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(norm(x, gamma, 1e-6, bias=torch.zeros(2)), norm(x, gamma, 1e-6), rtol=0, atol=0)
        self.assertEqual(norm(x[:0], gamma, 1e-6, bias=bias).shape, (0, 2))
        for invalid in (bias.bfloat16(), bias[:1], torch.ones(4)[::2]):
            with self.assertRaisesRegex(ValueError, 'bias'):
                norm(x, gamma, 1e-6, bias=invalid)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA fused norm numerics; CPU CI never launches GPUs')
    def test_fc_bias_norm_cuda_graph_reads_the_owned_vector(self):
        from engine.kernels.common.norm_rope import norm
        gen = torch.Generator().manual_seed(19)
        x = torch.randn(28, 4096, generator=gen).bfloat16().cuda()
        gamma = torch.randn(4096, generator=gen).bfloat16().cuda()
        bias = (torch.randn(4096, generator=gen) * .05).cuda()
        for _ in range(3):
            norm(x, gamma, 1e-6, bias=bias)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = norm(x, gamma, 1e-6, bias=bias)
        try:
            for _ in range(2):
                graph.replay()
                expected = norm(x.cpu(), gamma.cpu(), 1e-6, bias=bias.cpu())
                torch.testing.assert_close(actual.cpu(), expected, atol=.04, rtol=.016)
                bias.zero_()
        finally:
            graph.reset()

    def test_position_alpha_and_forced_predecessor_are_used_in_the_actual_walk(self):
        args = parts()
        self.assertEqual(walk_scores(*args).tolist(), [[2, 2, 2]])
        self.assertEqual(walk_scores(*args, alpha=(.5, 1., 1.)).tolist(), [[1, 1, 1]])
        trace = (torch.zeros(3, 3, 2), torch.zeros(3, 3, 2))
        packet = tensor((0, 0, 4) + (-1,) * 32, 'cpu')
        got = walk_scores(*args, boundary=packet, trace=trace, trace_slots=torch.tensor([2]))
        self.assertEqual(got.tolist(), [[4, 1, 1]])
        self.assertEqual(trace[1][2, 1].tolist(), [0., -3.])
        self.assertEqual(trace[0][0].count_nonzero().item(), 0, 'trace writes only its arena slot')
        for alpha in ((float('nan'),), (1., 2.)):
            with self.assertRaises(ValueError):
                walk_scores(*args, alpha=alpha)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA selector numerics; CPU CI never launches GPUs')
    def test_cuda_controlled_walk_and_trace_match_cpu_and_replay_changed_packets(self):
        cpu = parts()
        gpu = tuple(t.cuda() for t in cpu)
        trace = (torch.zeros(3, 3, 2, device='cuda'), torch.zeros(3, 3, 2, device='cuda'))
        slot = torch.tensor([2], device='cuda')
        packet = tensor(None, 'cuda')
        for alpha in ((1., 1., 1.), (.5, 1., .75)):
            walk_scores(*gpu, alpha=alpha, boundary=packet, trace=trace, trace_slots=slot)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = walk_scores(*gpu, alpha=alpha, boundary=packet, trace=trace, trace_slots=slot)
            try:
                for value in (EMPTY, (0, 0, 4) + (-1,) * 32, EMPTY):
                    packet.copy_(tensor(value, 'cuda'))
                    graph.replay()
                    expected_trace = (torch.zeros(3, 3, 2), torch.zeros(3, 3, 2))
                    expected = walk_scores(*cpu, alpha=alpha, boundary=tensor(value, 'cpu'),
                                          trace=expected_trace, trace_slots=slot.cpu())
                    self.assertEqual(actual.cpu().tolist(), expected.tolist())
                    torch.testing.assert_close(trace[1][2].cpu(), expected_trace[1][2])
            finally:
                graph.reset()


class WiringTests(unittest.TestCase):
    def test_projection_precision_reaches_greedy_sampled_and_batched_paths(self):
        from engine.base.comm import Comm
        from engine.modules.draft_projection import project
        facts = DrafterFacts(layers=1, hidden=16, heads=2, kv_heads=1, head_dim=4,
            inter=16, rms_eps=1e-6, rope_theta=10000., window=8, block=4, mask_id=7,
            conv_taps=2, conv_group=4, sel_rank=4, sel_top_k=2, target_layers=(1,), k=3)
        target = SimpleNamespace(comm=Comm(), rank=0, vp=8,
            head_local=lambda h: torch.tensor([[0., 1., 2., 3., 4., 5., 6., 7.]] * len(h)))
        d = Drafter(facts, target, 8)
        d.block = d.block_rows = lambda *args: torch.zeros(4, 16).bfloat16()
        d.p = {'candidate_selector.hidden_projection.weight': torch.zeros(4, 16).bfloat16(),
               'candidate_selector.predecessor_codebook': torch.zeros(8, 4).bfloat16(),
               'candidate_selector.successor_codebook': torch.zeros(8, 4).bfloat16()}
        for enabled in (True, False):
            with patch('engine.modules.draft_projection.project', wraps=project) as projection:
                d.propose(0, 5, torch.zeros(2))
                d.propose_sampled_tensor(torch.tensor([0]), 5, torch.zeros(2), 1., torch.full((3,), .5), 8)
                d.propose_rows(torch.zeros(2), torch.tensor([0]), torch.tensor([0]), torch.tensor([5]))
            self.assertEqual(projection.call_count, 3)
            self.assertTrue(all(call.kwargs['fp32'] is enabled for call in projection.call_args_list))
            d.tuning = DraftTuning(selector_projection_fp32=False)

    def test_fc_bias_follows_decode_phase_even_with_shared_calibration(self):
        facts = DrafterFacts(layers=1, hidden=4, heads=1, kv_heads=1, head_dim=4,
            inter=4, rms_eps=1e-6, rope_theta=10000., window=8, block=2, mask_id=7,
            conv_taps=2, conv_group=4, sel_rank=4, sel_top_k=2, target_layers=(1,), k=1)
        d = Drafter(facts, SimpleNamespace(), 8)
        d.p = {'hidden_norm.weight': torch.ones(4).bfloat16(),
               'layers.0.self_attn.k_norm.weight': torch.ones(4).bfloat16()}
        d.fc_bias = torch.tensor([-1., 2., 0., 0.])
        d.context_kv = torch.cat([torch.eye(4), torch.eye(4)]).bfloat16()
        d.context_linear = lambda aux, *args, **kwargs: aux
        aux = torch.tensor([[3., 1., 2., 2.]]).bfloat16()
        corrected = torch.tensor([[2., 3., 2., 2.]]).bfloat16()
        from engine.kernels.common.norm_rope import norm
        expected = norm(corrected, d.p['hidden_norm.weight'], facts.rms_eps)
        ring = torch.zeros(1, 2, 8, 1, 4).bfloat16()
        d.observe(ring, torch.tensor([0]), aux)
        self.assertTrue(torch.equal(ring[0, 1, 0, 0], norm(aux, d.p['hidden_norm.weight'], facts.rms_eps)[0]))
        d.observe_committed(ring, torch.tensor([0]), aux)
        self.assertTrue(torch.equal(ring[0, 1, 0, 0], expected[0]))
        for n in (1, 4):
            ctx = d._project_context(torch.zeros(n, 2, dtype=torch.int64), aux.expand(n * 2, -1),
                                     torch.ones(n, dtype=torch.int64), observe=False)
            self.assertTrue(torch.equal(ctx[:, :, 0, 1, 0], expected.expand(n, 2, -1)))

    def test_capture_construction_binds_boundary_input_only_when_enabled(self):
        class Graph:
            def __init__(self, body, make, shapes, **kw):
                self.body = body
                self.inputs = {shape: make(*shape) for shape in shapes}
            def run(self, shape, fill):
                fill(self.inputs[shape])
                return self.body(self.inputs[shape])
            def close(self):
                pass
        field = torch.zeros(4, 8)
        caches = SimpleNamespace(_fields={('draft', -1): field}, device='cpu',
                                 pool=SimpleNamespace(max_seqs=3), reset=lambda: None)
        d = SimpleNamespace(k=3, F=SimpleNamespace(hidden=4), aux_layers=(0,), fast_attention=True)
        d.propose_tensor = lambda anchor, pos, ring, support_slot, boundary=None: (
            boundary.clone() if boundary is not None else None)
        for enabled in (False, True):
            d.request_boundaries = enabled
            with patch('engine.profiles.glm53.decode_graphs.DecodeGraphs', Graph):
                graphs = DrafterDecodeGraphs(d, caches)
            try:
                got = graphs.propose(1, 5, field[2])
                self.assertEqual(got.tolist() if got is not None else None, list(EMPTY) if enabled else None)
            finally:
                graphs.close()

    def test_actual_proposal_masks_before_topk_and_sampled_q_matches_the_forced_pick(self):
        from engine.base.comm import Comm
        facts = DrafterFacts(layers=1, hidden=16, heads=2, kv_heads=1, head_dim=4,
            inter=16, rms_eps=1e-6, rope_theta=10000., window=8, block=4, mask_id=7,
            conv_taps=2, conv_group=4, sel_rank=4, sel_top_k=2, target_layers=(1,), k=3)
        target = SimpleNamespace(comm=Comm(), rank=0, vp=8,
            head_local=lambda h: torch.tensor([[0., 1., 2., 3., 4., 5., 6., 100.]] * 3))
        d = Drafter(facts, target, 8)
        d.block = lambda *args: torch.zeros(4, 16).bfloat16()
        d.p = {'candidate_selector.hidden_projection.weight': torch.zeros(4, 16).bfloat16(),
               'candidate_selector.predecessor_codebook': torch.zeros(8, 4).bfloat16(),
               'candidate_selector.successor_codebook': torch.zeros(8, 4).bfloat16()}
        ring = torch.zeros(2)
        self.assertEqual(d.propose(0, 5, ring), [7, 7, 7])
        boundary = (1, 1, 4, 7) + (-1,) * 31
        self.assertEqual(d.propose(0, 5, ring, boundary=boundary), [6, 4, 7])
        tokens, q = d.propose_sampled_tensor(torch.tensor([0]), 5, ring, 1., torch.full((3,), .5), 8,
                                             boundary=boundary)
        self.assertEqual(q[0, 7].item(), 0.)
        self.assertEqual(q[1, 4].item(), 1.)
        self.assertEqual(tokens[1].item(), 4)
        torch.testing.assert_close(q.sum(-1), torch.ones(3))
        self.assertTrue(bool((q.gather(1, tokens[:, None]) > 0).all()))

    def test_packet_reuse_resets_to_empty_on_both_graph_entry_points(self):
        graphs = DrafterDecodeGraphs.__new__(DrafterDecodeGraphs)
        graphs.field = torch.zeros(4, 8)
        graphs.drafter = SimpleNamespace(k=3)
        inputs = dict(anchor=torch.zeros(1, dtype=torch.int64), position=torch.tensor(0),
                      slot=torch.zeros(1, dtype=torch.int64), boundary=tensor(None, 'cpu'))
        def run(shape, fill):
            self.assertEqual(shape, (1, 4))
            fill(inputs)
            return inputs['boundary'].clone()
        graphs.proposals = SimpleNamespace(run=run)
        boundary = (2, 0, 4, 7) + (-1,) * 31
        self.assertEqual(graphs.propose(1, 5, graphs.field[2], boundary=boundary).tolist(), list(boundary))
        self.assertEqual(inputs['slot'].item(), 2)
        self.assertEqual(graphs.propose(1, 6, graphs.field[3]).tolist(), list(EMPTY))
        graphs.propose(1, 7, graphs.field[2], boundary=boundary)
        self.assertEqual(graphs.propose_from(torch.tensor([1]), torch.tensor(8), torch.tensor([3])).tolist(), list(EMPTY))
        inputs.pop('boundary')
        with self.assertRaisesRegex(ValueError, 'not enabled before'):
            graphs.propose(1, 8, graphs.field[2], boundary=boundary)

    def test_trace_collection_cannot_run_ahead_and_overwrite_its_proposal(self):
        from engine.profiles.glm53.adapter import Glm53Engine
        e = Glm53Engine.__new__(Glm53Engine)
        e.drafter = SimpleNamespace(tuning=DraftTuning(trace_every=2))
        self.assertFalse(e._plain_ahead(1))
        self.assertEqual(e._blocked_by(1), 'draft_trace')

    def test_preparation_requires_real_statistics_and_binds_only_draft_names(self):
        reader, norm = 'layers.0.self_attn.qkv', 'layers.0.input_layernorm.weight'
        tuning = DraftTuning.from_dict(dict(version=1, smoothing_alpha={norm: .25},
                                           gptq_damping={reader: .02, 'fc.weight': .02}))
        store = SimpleNamespace(amax=lambda name: torch.ones(16), calibrated=lambda name: True,
                                gptq_damping={'target': .01})
        comm = SimpleNamespace(wait_prepared=lambda stage: None, gather_objects=lambda x: [x, x])
        prepare_store(tuning, store, DraftPolicy('fp8', 'decode', True), SimpleNamespace(hidden=16), comm)
        self.assertEqual(store.gptq_damping, {'target': .01, store_name(reader): .02,
            store_name('fc.weight'): .02, decode_name(store_name('fc.weight')): .02})
        store.amax = lambda name: None
        with self.assertRaisesRegex(ValueError, 'channel peaks'):
            prepare_store(tuning, store, DraftPolicy(), SimpleNamespace(hidden=16), comm)
        comm.gather_objects = lambda x: [None, 'peer missing statistics']
        with self.assertRaisesRegex(ValueError, 'rank 1: peer missing'):
            prepare_store(tuning, store, DraftPolicy(), SimpleNamespace(hidden=16), comm)


if __name__ == '__main__':
    unittest.main()
