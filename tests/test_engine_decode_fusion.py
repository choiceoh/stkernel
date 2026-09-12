"""Changed-input CUDA graph replay of fused token commit and state advance."""
import unittest

import torch


@unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA')
class CommitReplayTests(unittest.TestCase):
    def test_limits_eos_ghosts_and_sampled_acceptance_survive_replay(self):
        from engine.base.sampler import commit_batch
        from engine.kernels.decode_commit import advance
        for n, k in ((1, 0), (4, 1), (8, 5)):
            for sampled in (False, True):
                picks = torch.zeros(n, k+1, dtype=torch.int64, device='cuda')
                state = dict(drafts=picks[:, :k].clone(), ends=torch.zeros(n, 3, dtype=torch.int64, device='cuda'),
                             alive=torch.ones(n, dtype=torch.bool, device='cuda'))
                for name in ('generated', 'limit', 'ctx', 'anchor', 'real_slot', 'slot'):
                    state[name] = torch.zeros(n, dtype=torch.int64, device='cuda')
                accepted = torch.zeros(n, dtype=torch.int64, device='cuda') if sampled else None
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    advance(picks, state, accepted)
                torch.cuda.current_stream().wait_stream(side)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = advance(picks, state, accepted)
                try:
                    for trial in range(8):
                        picks.random_(0, 32); state['drafts'].copy_(picks[:, :k])
                        if k:
                            state['drafts'][::2, trial % k].add_(1)
                        state['ends'].random_(0, 32)
                        state['ends'][0, 0] = picks[0, 0]
                        state['alive'].fill_(True); state['alive'][1::2] = False
                        state['generated'].random_(0, 16); state['limit'].random_(0, 16)
                        state['ctx'].random_(0, 16384); state['anchor'].random_(0, 32)
                        state['real_slot'].copy_(torch.arange(1, n+1, device='cuda'))
                        state['slot'].copy_(state['real_slot'])
                        if accepted is not None:
                            accepted.random_(0, k+1)
                        before = {name: value.clone() for name, value in state.items()}
                        count, done, kept, _ = commit_batch(picks, state['drafts'], state['alive'],
                                                           state['generated'], state['limit'], state['ends'], accepted)
                        graph.replay()
                        for got, expected in zip(actual, (count, done, kept, picks, before['ctx'])):
                            torch.testing.assert_close(got, expected, rtol=0, atol=0)
                        self.assertTrue(torch.equal(state['ctx'], before['ctx']+count))
                        self.assertTrue(torch.equal(state['generated'], before['generated']+count))
                        self.assertTrue(torch.equal(state['alive'], before['alive'] & ~done))
                        self.assertTrue(torch.equal(state['slot'], torch.where(before['alive'] & ~done, before['real_slot'], 0)))
                finally:
                    graph.reset()


if __name__ == '__main__':
    unittest.main()
