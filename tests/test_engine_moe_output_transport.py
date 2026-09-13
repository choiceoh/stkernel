"""Real fused publication/consume protocol with a CPU proxy, not a NIC benchmark."""
import queue
import threading
import time
import unittest
import torch

from tests.test_engine_direct_producer_cuda import build_oracle


@unittest.skipUnless(torch.cuda.is_available(), 'requires admitted GB10')
class MoeOutputTransportTests(unittest.TestCase):
    def test_rank_packets_ring_wrap_changed_inputs_and_mixed_collectives(self):
        from engine.kernels.mapped_staging import allocate
        ext = build_oracle()
        for rank, rows in enumerate((8, 16, 24, 32)):
            host, device = allocate(ext.bytes())
            ext.prepare(host, device, rank, 0, 0)
            expected, errors, stop = queue.Queue(), [], threading.Event()
            def proxy():
                last = 0
                while not stop.is_set():
                    sequence = ext.published(host)
                    if sequence > last:
                        try:
                            reference = expected.get(timeout=5)
                            self.assertTrue(torch.equal(ext.payload(host, sequence), reference))
                        except BaseException as error:
                            errors.append(error)
                        finally:
                            ext.land(host, sequence)
                            last = sequence
                    else:
                        time.sleep(.00005)
            worker = threading.Thread(target=proxy, daemon=True)
            worker.start()
            graph = None
            try:
                acc = torch.randn(rows, 4096, device='cuda')
                shared = torch.randn_like(acc, dtype=torch.bfloat16)
                template, out = torch.empty_like(shared), torch.empty_like(shared)
                def produce():
                    ext.consume(ext.moe_packets(acc, shared), out)
                reference = (acc.bfloat16()+shared).cpu()
                expected.put(reference.view(torch.uint8).flatten())
                produce()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    produce()
                for iteration in range(8):
                    if iteration == 0:
                        acc.fill_(1.+2**-8); shared.fill_(-1.)
                    else:
                        acc.normal_(); shared.normal_()
                    reference = (acc.bfloat16()+shared).cpu()
                    expected.put(reference.view(torch.uint8).flatten())
                    graph.replay()
                    torch.cuda.synchronize()
                    folded = (reference.float()*4).bfloat16()
                    self.assertTrue(torch.equal(out.cpu().view(torch.uint8), folded.view(torch.uint8)))
                    # Existing copy and integer lanes continue the same ticket sequence.
                    template.copy_(reference)
                    expected.put(reference.view(torch.uint8).flatten())
                    ext.consume(ext.oneshot_packets(template), out)
                    keys = torch.arange(rows, device='cuda', dtype=torch.int64)
                    expected.put(keys.cpu().view(torch.uint8).flatten())
                    ext.oneshot_max_int64(keys)
                    torch.cuda.synchronize()
                    self.assertEqual(keys.cpu().tolist(), list(range(rows)))
                self.assertEqual(ext.tickets(host), ext.published(host)*48)
                self.assertFalse(errors, repr(errors))
                self.assertTrue(expected.empty())
            finally:
                torch.cuda.synchronize()
                if graph is not None:
                    graph.reset()
                stop.set(); worker.join(timeout=5)
                self.assertFalse(worker.is_alive())


if __name__ == '__main__':
    unittest.main()
