"""Exact candidate exchange inside CUDA WHILE, with three distinct proxy peers."""
from contextlib import contextmanager
import queue
import threading
import time
from types import SimpleNamespace
import unittest

import torch


@contextmanager
def proxy(test, rank):
    from engine.kernels.mapped_staging import allocate
    from tests.test_engine_direct_producer_cuda import build_oracle
    ext = build_oracle()
    host, device = allocate(ext.bytes())
    ext.prepare(host, device, rank, 0, 0)
    requests, errors, stopping = queue.Queue(), [], threading.Event()

    def work():
        last = 0
        while not stopping.is_set():
            sequence = ext.published(host)
            if sequence <= last:
                time.sleep(.00005)
                continue
            try:
                expected, peers = requests.get(timeout=5)
                test.assertTrue(torch.equal(ext.payload(host, sequence), expected))
                if peers is None:
                    ext.land(host, sequence)
                else:
                    ext.land_peers(host, sequence, peers)
            except BaseException as error:
                errors.append(error)
                ext.land(host, sequence)  # let the GPU retire so the failure is reportable
            last = sequence

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    try:
        yield ext, requests
        torch.cuda.synchronize()
        test.assertFalse(errors, repr(errors))
        test.assertTrue(requests.empty())
        test.assertEqual(ext.tickets(host), ext.published(host) * 48)
    finally:
        torch.cuda.synchronize()
        stopping.set()
        worker.join(timeout=6)
        test.assertFalse(worker.is_alive())


def packets(values, rank):
    encoded = values.contiguous().view(torch.uint8).reshape(4, -1).cpu()
    return encoded[rank].clone(), encoded[[r for r in range(4) if r != rank]].contiguous()


@unittest.skipUnless(torch.cuda.is_available(), 'requires admitted GB10')
class IntegerGatherCudaTests(unittest.TestCase):
    def test_distinct_signed_peers_tails_replays_and_mixed_ticket_wrap(self):
        for rank in range(4):
            with proxy(self, rank) as (ext, requests):
                for keys in (1, 3, 63, 64, 96, 192, 384, 511, 512):
                    x = torch.empty(keys, device='cuda', dtype=torch.int64)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        out = ext.oneshot_gather_int64(x)
                    try:
                        for trial in range(3):
                            values = torch.arange(4*keys, dtype=torch.int64).reshape(4, keys) * (1 << 40) + trial
                            values[:, 0] = torch.tensor([-(2**63), 2**63-1, -1, 0])
                            x.copy_(values[rank])
                            requests.put(packets(values, rank))
                            graph.replay()
                            torch.testing.assert_close(out.cpu(), values, rtol=0, atol=0)
                            torch.testing.assert_close(x.cpu(), values[rank], rtol=0, atol=0)
                            # MAX and BF16 exchange share exactly the same publication counter.
                            maximum = torch.arange(3, device='cuda', dtype=torch.int64)
                            requests.put((maximum.cpu().view(torch.uint8), None))
                            ext.oneshot_max_int64(maximum)
                            torch.testing.assert_close(maximum.cpu(), torch.arange(3), rtol=0, atol=0)
                            bf16 = torch.full((1, 4096), trial+1, device='cuda', dtype=torch.bfloat16)
                            requests.put((bf16.cpu().view(torch.uint8).flatten(), None))
                            reduced = torch.empty_like(bf16)
                            ext.consume(ext.oneshot_packets(bf16), reduced)
                            torch.testing.assert_close(reduced, bf16*4, rtol=0, atol=0)
                    finally:
                        graph.reset()

    def test_actual_vocab_topk_inside_bounded_graph_preserves_ties_and_each_iteration(self):
        from engine.base.comm import Comm
        from engine.kernels.bounded_graph import BoundedGraph
        from engine.kernels.oneshot import OneShot
        from engine.kernels.common.vocab_candidates import pack, select
        from engine.modules.vocab import topk
        torch.manual_seed(913)
        width, k = 256, 16
        for rank in range(4):
            with proxy(self, rank) as (ext, requests):
                transport = SimpleNamespace(eligible_gather=OneShot.eligible_gather,
                                            gather=ext.oneshot_gather_int64)
                comm = Comm(4, rank, None, transport=transport)
                for rows in (6, 24):
                    local = torch.empty(rows, width, device='cuda', dtype=torch.bfloat16)
                    count = torch.zeros(1, device='cuda', dtype=torch.int64)
                    stop = torch.zeros_like(count)
                    histories = (torch.empty(4, rows, k, device='cuda'),
                                 torch.empty(4, rows, k, device='cuda', dtype=torch.int64))
                    # Compile the actual selection kernels before entering capture.
                    packets_local = select(pack(local.zero_(), rank*width, width), k)
                    from engine.kernels.common.vocab_candidates import restore
                    restore(packets_local.repeat(1, 4), width*4).topk(k, dim=-1)
                    for limit in (1, 2, 4):
                        graph = torch.cuda.CUDAGraph(keep_graph=True)
                        with torch.cuda.graph(graph):
                            values, ids = topk(local, comm, rank*width, k, width*4)
                            histories[0].index_copy_(0, count, values.unsqueeze(0))
                            histories[1].index_copy_(0, count, ids.unsqueeze(0))
                        loop = BoundedGraph(graph, count, stop, limit, owners=(local, histories, comm))
                        try:
                            for trial in range(2):
                                # Quantized values create abundant tied top-k scores across ranks.
                                logits = torch.randint(-3, 4, (4, rows, width), device='cuda').bfloat16()
                                packed = torch.stack([select(pack(logits[r], r*width, width), k) for r in range(4)])
                                local.copy_(logits[rank])
                                # The old NCCL gather contract is rank-ordered cat of these exact
                                # packets. Keep the existing CUDA tie policy as the reference.
                                reference_comm = SimpleNamespace(world_size=4,
                                    all_gather=lambda value, dim=-1: torch.cat(list(packed), dim=dim))
                                expected = topk(local, reference_comm, rank*width, k, width*4)
                                for h in histories:
                                    h.fill_(-999)
                                for _ in range(limit):
                                    requests.put(packets(packed, rank))
                                loop.replay()
                                self.assertEqual(count.item(), limit)
                                for iteration in range(limit):
                                    torch.testing.assert_close(histories[0][iteration], expected.values, rtol=0, atol=0)
                                    torch.testing.assert_close(histories[1][iteration], expected.indices, rtol=0, atol=0)
                                for h in histories:
                                    self.assertTrue((h[limit:] == -999).all().item())
                        finally:
                            loop.close()
                            graph.reset()

    def test_forbidden_event_node_reports_its_location(self):
        from engine.kernels.bounded_graph import BoundedGraph
        count = torch.zeros(1, device='cuda', dtype=torch.int64)
        stop = torch.zeros_like(count)
        external = torch.cuda.Event(external=True)
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            stop.zero_()
            external.record()
        try:
            with self.assertRaisesRegex(RuntimeError, r'node type [0-9]+ at body/'):
                BoundedGraph(graph, count, stop, 4, owners=(external,))
        finally:
            graph.reset()


if __name__ == '__main__':
    unittest.main()
