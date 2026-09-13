"""Prefill output contraction preserves token order and the first-token hidden row."""
from types import SimpleNamespace as NS
import unittest

import torch

from engine.base.comm import LocalTP
from engine.profiles.glm53.net import Glm53Net, Step, rmsnorm
from engine.profiles.glm53.adapter import Glm53Engine


class TokenShards:
    def __init__(self, comm):
        self.comm = comm

    def all_gather(self, x):
        return self.comm.all_gather(x, dim=0)

    def reduce_scatter(self, x):
        return self.comm.all_reduce(x).chunk(self.comm.world_size)[self.comm.rank].contiguous()


def network(comm, *, sharded):
    """Small deterministic layers; the real forward owns all token partitioning."""
    net = Glm53Net.__new__(Glm53Net)
    net.F = NS(hidden=8, hc=2, rms_eps=1e-6, is_dsa=lambda L: True, is_moe=lambda L: False)
    net.comm, net.rank, net.probe = comm, comm.rank, None
    net.prefill_transport = TokenShards(comm) if sharded else None
    net.layers, net.p = (0, 1), {"norm": torch.arange(1, 9).float()}
    net.embed = lambda ids: ((ids[:, None] + torch.arange(8)[None, :]) % 17).float()
    post = lambda x, res, p, c: res + x[:, None, :] * p[:, :, None]
    net.lanes = NS(mhc_post=post)
    net._norm = rmsnorm

    def pre(L, res, which):
        n = len(res)
        return torch.ones(n, 2), torch.zeros(n, 2, 2), res.mean(1) + L + 1

    def post_pre(L, x, res, p, c, which):
        res = post(x, res, p, c)
        return res, *pre(L, res, which)

    net._hc_pre, net._hc_post_pre = pre, post_pre
    net._dsa = lambda L, x, step, caches, reduce: reduce(x / comm.world_size)
    net._dense = lambda L, x, reduce: reduce(x / comm.world_size)
    return net


class PrefillOutputTests(unittest.TestCase):
    def test_boot_warms_interior_marks_within_the_reserved_snapshot_slots(self):
        from unittest.mock import MagicMock
        caches = MagicMock()
        caches.device, caches.snapshots = 'cpu', 2
        caches.pool.rows_in_use, caches.pool.num_blocks = 0, 8
        caches.slots.owner = [-1, -1]
        caches.slots.take.return_value = 1
        calls = []
        engine = Glm53Engine.__new__(Glm53Engine)
        engine.caches, engine.F, engine.prefill_chunk = caches, NS(block=64), 256
        engine.max_context = 512
        engine.memory = MagicMock()
        engine.net = NS(comm=NS(all_reduce_max=lambda x: x), head=lambda x: x)
        def forward(step):
            calls.append(step)
            return torch.ones(1, 8), None
        engine._prefill_forward = forward
        engine._warmup_prefill_memory()
        self.assertEqual([step.marks for step in calls], [((64, 0), (128, 1))] * 2)
        self.assertEqual([step.segments[0].ctx for step in calls], [0, 256])
        caches.reset.assert_called_once()

    def test_warmup_uses_the_served_ceiling_and_releases_cache_at_each_boundary(self):
        from unittest.mock import MagicMock, call
        for ceiling in (128, 384, 1024):
            with self.subTest(ceiling=ceiling):
                caches = MagicMock(device='cpu', snapshots=0)
                caches.pool.rows_in_use, caches.pool.num_blocks = 0, 8
                caches.slots.owner = [-1, -1]
                caches.slots.take.return_value = 1
                engine = Glm53Engine.__new__(Glm53Engine)
                engine.caches, engine.F, engine.prefill_chunk = caches, NS(block=64), 256
                engine.max_context, engine.memory = ceiling, MagicMock()
                engine.net = NS(comm=NS(all_reduce_max=lambda x: x), head=lambda x: x)
                engine._prefill_forward = MagicMock(return_value=(torch.ones(1, 8), None))
                engine._warmup_prefill_memory()
                capacity, width = min(512, ceiling), min(256, ceiling)
                starts = sorted({0, capacity - width})
                steps = [c.args[0] for c in engine._prefill_forward.call_args_list]
                self.assertEqual([(s.segments[0].ctx, s.ids.numel()) for s in steps],
                                 [(start, width) for start in starts])
                caches.pool.reserve.assert_called_once_with(0, capacity)
                self.assertEqual(engine.memory.checkpoint.call_args_list,
                                 [c for start in starts for c in
                                  (call(f'prefill/{width}/{start}/before'),
                                   call(f'prefill/{width}/{start}/prepared', release_cache=True))])

    def test_kernel_warmup_respects_the_ceiling_and_releases_before_the_guard(self):
        from unittest.mock import MagicMock
        caches = MagicMock(device='cpu')
        caches.pool.rows_in_use, caches.pool.num_blocks = 0, 8
        caches.slots.owner = [-1, -1]
        caches.slots.take.return_value = 1
        engine = Glm53Engine.__new__(Glm53Engine)
        engine.caches, engine.F, engine.prefill_chunk = caches, NS(block=64), 256
        engine.max_context, engine.memory = 8, MagicMock()
        engine.memory.checkpoint.return_value = {'allocator_reclaimed_bytes': 0}
        engine.net = NS(head=lambda x: x)
        engine._forward = MagicMock(return_value=(torch.ones(1, 8), None))
        engine._warmup_serving_kernels()
        self.assertEqual([c.args[0].ids.numel() for c in engine._forward.call_args_list], [1, 8])
        caches.reset.assert_called_once()
        engine.memory.checkpoint.assert_called_once_with('warm kernels [1, 8]', release_cache=True)

    def test_last_hidden_and_aux_equal_full_output_on_every_rank(self):
        for n, sharded in ((7, False), (128, True), (260, True), (129, True)):
            with self.subTest(n=n, sharded=sharded):
                def rank(comm):
                    net = network(comm, sharded=sharded)
                    step = Step.prefill(torch.arange(n), 768, 3, 1)
                    full, aux = net.forward(step, None, aux_layers=(0, 1))
                    last, context = net.forward(step, None, aux_layers=(0, 1), last_hidden_only=True)
                    torch.testing.assert_close(last, full[-1:], rtol=0, atol=0)
                    torch.testing.assert_close(context, aux, rtol=0, atol=0)
                    self.assertEqual(tuple(last.shape), (1, 8))
                    return last, context

                results = LocalTP(4).run(rank)
                for h, c in results[1:]:
                    self.assertTrue(torch.equal(h, results[0][0]))
                    self.assertTrue(torch.equal(c, results[0][1]))

    def test_plain_target_prefill_needs_only_the_last_hidden_row(self):
        def rank(comm):
            net = network(comm, sharded=True)
            step = Step.prefill(torch.arange(256), 0, 0, 1)
            full = net.forward(step, None)
            last = net.forward(step, None, last_hidden_only=True)
            torch.testing.assert_close(last, full[-1:], rtol=0, atol=0)
        LocalTP(4).run(rank)


if __name__ == "__main__":
    unittest.main()
