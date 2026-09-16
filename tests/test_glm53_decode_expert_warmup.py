"""Rank-local expert preparation must finish before a peer enters decode."""
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch

from engine.profiles.glm53.adapter import Glm53Engine
from engine.profiles.glm53.net import Glm53Net


class DecodeExpertWarmupTests(unittest.TestCase):
    def test_bound_packed_and_fallback_layers_are_both_run_without_collectives(self):
        net = Glm53Net.__new__(Glm53Net)
        net.F = NS(hidden=4, topk_experts=2, is_moe=lambda layer: layer != 0)
        net.comm = Mock(side_effect=AssertionError("no TP during local preparation"))
        calls = []

        def expert(layer):
            def run(x, ids, weights):
                topk = 1 if layer == 0 else 2
                self.assertEqual(x.dtype, torch.bfloat16)
                self.assertEqual(ids.dtype, torch.int32)
                self.assertEqual(weights.dtype, torch.float32)
                self.assertEqual(tuple(ids.shape), (len(x), topk))
                self.assertTrue(torch.equal(ids[0], torch.arange(topk, dtype=torch.int32)))
                self.assertTrue(torch.equal(weights.sum(1), torch.ones(len(x))))
                calls.append((layer, len(x)))
                return x
            return run

        # Layers 1 and 2 have the same geometry but different packed-scale
        # admission. Neither may be skipped just because the shape was seen.
        net._experts = {layer: expert(layer) for layer in (0, 1, 2)}
        net.warmup_decode_experts((16, 8), 'cpu')
        self.assertEqual(calls, [(0, 16), (1, 16), (2, 16), (0, 8), (1, 8), (2, 8)])
        self.assertEqual(net.comm.mock_calls, [])
        for bad in ((0,), (-1,), (True,), (1.5,)):
            with self.subTest(rows=bad), self.assertRaises(ValueError):
                net.warmup_decode_experts(bad, 'cpu')

    def test_a_slow_fallback_rank_keeps_peers_on_the_host_until_ready(self):
        ready = set()
        rendezvous = threading.Barrier(4, timeout=5)
        fast_rank_arrived = threading.Event()
        lock = threading.Lock()
        reports = []

        def rank_work(rank):
            engine = Glm53Engine.__new__(Glm53Engine)
            engine.caches, engine.drafter = NS(device='cpu'), NS(k=7)
            engine.jit_windows = Mock()

            def prepare(rows, device):
                self.assertEqual((rows, device), ((16, 8), 'cpu'))
                if rank == 3:  # local fallback loading finishes after a peer is ready
                    self.assertTrue(fast_rank_arrived.wait(5))
                with lock:
                    ready.add(rank)

            def wait(phase, *, final):
                self.assertEqual(phase, 'decode-experts')
                self.assertTrue(final)
                if rank == 0:
                    fast_rank_arrived.set()
                rendezvous.wait()

            def vote(phase, **kw):
                with lock:
                    self.assertEqual(ready, {0, 1, 2, 3})
                    reports.append((rank, phase, kw))

            engine.net = NS(warmup_decode_experts=prepare, comm=NS(wait_prepared=wait))
            engine.memory = NS(checkpoint=vote)
            engine._warmup_decode_experts(2)

        with ThreadPoolExecutor(4) as pool:
            list(pool.map(rank_work, range(4)))
        self.assertEqual(len(reports), 4)
        self.assertTrue(all(phase == 'decode experts' and kw == {'release_cache': True}
                            for _, phase, kw in reports))

    def test_cuda_work_is_drained_before_loading_and_before_host_rendezvous(self):
        engine = Glm53Engine.__new__(Glm53Engine)
        engine.caches, engine.drafter = NS(device='cuda:0'), NS(k=0)
        engine.memory, engine.jit_windows = None, Mock()
        steps = []
        engine.net = NS(warmup_decode_experts=lambda rows, device: steps.append(('load', rows)),
                        comm=NS(wait_prepared=lambda phase, **kw: steps.append(('host', phase))))
        with patch.object(torch.cuda, 'synchronize', side_effect=lambda: steps.append('sync')):
            engine._warmup_decode_experts(3)
        self.assertEqual(steps, ['sync', ('load', (3, 2, 1)), 'sync', ('host', 'decode-experts')])


if __name__ == '__main__':
    unittest.main()
