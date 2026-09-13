"""Real GEMM/ring kernels and CPU proxy oracle; no NIC performance claim."""
from functools import cache
from pathlib import Path
import queue
import threading
import time
import unittest
import torch


@cache
def build_oracle():
    from torch.utils.cpp_extension import load
    from engine.kernels.common.native_cache import prepare_sources
    root = Path(__file__).resolve().parents[1]
    files = [root/'probes/oneshot_producer_oracle.cu',
             root/'engine/kernels/oneshot/dsv4_oneshot_ar.cu',
             root/'engine/kernels/oneshot/dsv4_oneshot_transport.h']
    flags = ['-O2', '-gencode', 'arch=compute_121a,code=sm_121a', '-DMAXEL=262144']
    key, directory, sources = prepare_sources(Path.home()/'.cache/st/producer-oracle', files,
                                               (flags, ['-libverbs'], torch.__version__, torch.version.cuda))
    return load(name='st_producer_oracle_'+key, sources=[sources[0]], extra_cuda_cflags=flags,
                extra_ldflags=['-libverbs'], build_directory=str(directory), verbose=False)


@unittest.skipUnless(torch.cuda.is_available(), 'requires admitted GB10')
class DirectProducerCudaTests(unittest.TestCase):
    def test_real_gemm_ring_wrap_and_mixed_packets_match_ordinary_output(self):
        from engine.kernels.dense import DenseLinear
        from engine.kernels.mapped_staging import allocate
        ext = build_oracle()
        torch.manual_seed(9313)
        for cols in (2048, 4096):
            layer = DenseLinear(torch.randn(4096, cols, device='cuda', dtype=torch.bfloat16)*.02, prefill=False)
            for rows in (1, 7, 28, 32):
                for private in (False, True):
                    layer.workspace = None
                    if private:
                        layer.isolate_workspace()
                    host, device = allocate(ext.bytes())
                    ext.prepare(host, device, rows % 4, 0, 0)
                    expected, errors = queue.Queue(), []
                    stop = threading.Event()
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
                        x = torch.randn(rows, cols, device='cuda', dtype=torch.bfloat16)
                        metadata = torch.empty(rows, 4096, device='cuda', dtype=torch.bfloat16)
                        out = torch.empty_like(metadata)
                        def produce():
                            slot = ext.reserve_packets(metadata)
                            layer._write_slot(x, slot)
                            descriptor = ext.publish_packets(metadata, slot)
                            ext.consume(descriptor, out)
                        reference = layer(x).cpu()
                        expected.put(reference.view(torch.uint8).flatten())
                        produce()
                        torch.cuda.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            produce()
                        for iteration in range(6):  # ring wrap with replay-time addresses
                            x.normal_()
                            reference = layer(x).cpu()
                            expected.put(reference.view(torch.uint8).flatten())
                            graph.replay()
                            torch.cuda.synchronize()
                            torch.testing.assert_close(out.cpu(), (reference.float()*4).bfloat16(), rtol=0, atol=0)
                            # Mix the original copy path and int64 MAX to check ticket continuity.
                            expected.put(reference.view(torch.uint8).flatten())
                            metadata.copy_(reference)
                            ext.consume(ext.oneshot_packets(metadata), out)
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

    def test_reservation_waits_for_ack_before_exposing_a_reused_slot(self):
        from engine.kernels.mapped_staging import allocate
        ext = build_oracle()
        host, device = allocate(ext.bytes())
        ext.prepare(host, device, 0, 4, 0)
        template = torch.empty(1, 4096, device='cuda', dtype=torch.bfloat16)
        slot = ext.reserve_packets(template)
        done = torch.cuda.Event(); done.record()
        try:
            time.sleep(.01)
            self.assertFalse(done.query())
        finally:
            ext.ack(host, 1)
            done.synchronize()
        self.assertEqual(slot.cpu()[1].item(), 5)
        self.assertEqual(ext.published(host), 4)  # reservation never publishes bytes


if __name__ == '__main__':
    unittest.main()
