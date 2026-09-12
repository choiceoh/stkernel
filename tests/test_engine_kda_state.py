"""Canonical recurrent state: numerical oracle, replay and rejected drafts."""
import importlib.util
import unittest

from tests.image_kernels import PRESENT, REASON

torch = None
if importlib.util.find_spec("torch"):
    import torch


@unittest.skipUnless(torch is not None and torch.cuda.is_available() and PRESENT, "requires CUDA; " + REASON)
class KdaStateTests(unittest.TestCase):
    def setUp(self):
        from engine.kernels.kda import fused_recurrent_kda
        from engine.profiles.glm53.lanes import reference, served
        self.kernel = fused_recurrent_kda
        self.ref = reference().kda_recurrent
        self.run = served(reference_for=("expert",)).kda_recurrent
        torch.manual_seed(92128)

    def inputs(self, tokens, heads=16, dk=128, dv=128, seeded=True):
        def rand(*shape):
            return torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
        # Real net projections pass strided q/k/v slices to the public lane.
        q, k = [rand(1, tokens, heads, 2*dk)[..., ::2] for _ in range(2)]
        v, g, beta = rand(1, tokens, heads, dv), rand(1, tokens, heads, dk), rand(1, tokens, heads)
        a, bias = rand(heads).float()*.2, rand(heads*dk).float()*.1
        state = torch.randn(1, heads, dk, dv, device="cuda")*.1 if seeded else None
        return (q, k, v, g, beta, a, bias, state, -5.)

    def close(self, actual, expected, state_limit=2e-6):
        for a, b, limit in zip(actual, expected, (.008, state_limit)):
            self.assertEqual(a.shape, b.shape)
            self.assertTrue(torch.isfinite(a).all())
            error = (a.float()-b.float()).abs().max()/b.float().abs().max().clamp_min(1e-6)
            self.assertLess(error.item(), limit)
        self.assertEqual(actual[1].dtype, torch.float32)
        self.assertTrue(actual[1].is_contiguous())

    def test_every_snapshot_against_independent_recurrence(self):
        for h, k, v in ((16,128,128), (3,64,32), (2,33,17)):
            for t in (1, 2, 6, 7, 12, 64):
                for seeded in (False, True):
                    with self.subTest(shape=(t,h,k,v), seeded=seeded):
                        args = self.inputs(t,h,k,v,seeded)
                        before = [x.clone() if isinstance(x, torch.Tensor) else x for x in args]
                        self.close(self.run(*args), self.ref(*args))
                        for x, saved in zip(args, before):
                            if isinstance(x, torch.Tensor): self.assertTrue(torch.equal(x, saved))

    def test_graph_replay_changed_state_and_rollback(self):
        for t in (1, 6):
            args = list(self.inputs(t))
            self.run(*args)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph): actual = self.run(*args)
            try:
                for iteration in range(8):
                    for x in args[:5]: x.normal_()
                    if iteration == 0: args[7].zero_()
                    elif iteration == 1: args[7].normal_(std=.1)
                    saved = args[7].clone()
                    graph.replay()
                    expected = self.ref(*args)
                    self.close(actual, expected)
                    self.assertTrue(torch.equal(args[7], saved))
                    # Accept a prefix; the next replay must ignore rejected
                    # future snapshots even though all of them were produced.
                    accepted = iteration % t + 1
                    args[7].copy_(actual[1][accepted-1:accepted])
            finally:
                graph.reset()

    def test_eager_zero_context_matches_graph_history_zeros_exactly(self):
        for t in (1, 6):
            args = list(self.inputs(t, seeded=False))
            eager = self.run(*args)
            args[7] = torch.zeros(1,16,128,128,device="cuda")
            self.run(*args)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph): replay = self.run(*args)
            graph.replay()
            try:
                for x,y in zip(eager,replay):
                    self.assertTrue(torch.equal(x.view(torch.uint8),y.view(torch.uint8)))
            finally:
                graph.reset()

    def test_retained_state_across_many_steps_and_legacy_layout(self):
        for lower_bound in (-5., -.01):
            args = list(self.inputs(6))
            args[-1] = lower_bound
            reference_state, legacy_state = args[7].clone(), args[7].transpose(-1,-2).contiguous()
            for _ in range(32):
                for x in args[:5]: x.normal_()
                actual = self.run(*args)
                expected = self.ref(*args[:7], reference_state, lower_bound)
                self.close(actual, expected, state_limit=2e-5)
                q,k,v,g,beta,a,bias,_,lb = args
                old = self.kernel(q,k,v,g,beta,initial_state=legacy_state,
                    inplace_final_state=False,sigmoid_beta=True,a_log=a,g_bias=bias,
                    compute_gate=True,lower_bound=lb)
                self.close(actual, (old[0],old[1].transpose(-1,-2)), state_limit=2e-5)
                args[7] = actual[1][-1:]
                reference_state = expected[1][-1:]
                legacy_state = old[1][-1:]

    def test_layout_contract_rejects_unsupported_tables_before_launch(self):
        q,k,v,g,beta,a,bias,state,lb = self.inputs(1)
        kwargs = dict(initial_state=state, inplace_final_state=False, state_layout="kv")
        for change in (dict(state_layout="invalid"), dict(inplace_final_state=True),
                       dict(initial_state=state.transpose(-1,-2)),
                       dict(initial_state=state.bfloat16()), dict(initial_state=state[0]),
                       dict(cu_seqlens=torch.tensor([0,1],device="cuda")),
                       dict(ssm_state_indices=torch.ones(1,device="cuda",dtype=torch.int32)),
                       dict(num_accepted_tokens=torch.ones(1,device="cuda",dtype=torch.int32))):
            with self.subTest(change=list(change)), self.assertRaises(ValueError):
                self.kernel(q,k,v,g,beta,**(kwargs|change))
        with self.assertRaises(ValueError):
            self.kernel(q.expand(2,-1,-1,-1),k.expand(2,-1,-1,-1),v.expand(2,-1,-1,-1),
                        g.expand(2,-1,-1,-1),beta.expand(2,-1,-1),**kwargs)


if __name__ == "__main__": unittest.main()
