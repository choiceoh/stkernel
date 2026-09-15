"""The PDL consumer sum against the ordinary kernel on one GB10, with a CPU proxy standing in for the NIC.

Both kernels come from one oracle build of the production source, so the ordinary kernel is the same-build control.
Every rank position, C=1 and C=2 row counts and both sides of the dispatch bound, scalar tails, three distinct peer
packets, changed inputs behind captured graphs, a late PDL producer and an early PDL successor around the sum, and
one ring ticket sequence shared with int64 MAX. Byte equality only; this is not a transport latency measurement.
"""
from contextlib import contextmanager
import math
import queue
import threading
import time
import unittest

import torch

from tests.test_engine_direct_producer_cuda import build_oracle

HIDDEN = 4096
# C=1's and C=2's verify sums and both sides of the consumer bound, a C=4 and a MAXEL sum, and element counts whose
# scalar tail, empty vector lanes or unfinished grid stride the row shapes never reach
SHAPES = ((1, HIDDEN), (7, HIDDEN), (8, HIDDEN), (16, HIDDEN), (17, HIDDEN), (32, HIDDEN), (64, HIDDEN),
          (5,), (65533,), (98311,))
LAND_DELAY_S = .003       # the proxy lands the peers late: an early-released successor must still wait
PRODUCER_CYCLES = 3 << 20  # ~2 ms of SM clock: the producer releases the sum before it writes the input
KERNELS = ('oneshot_ar', 'oneshot_ar_consumer')


@contextmanager
def proxy(test, rank):
    from engine.kernels.mapped_staging import allocate
    ext = build_oracle()
    host, device = allocate(ext.bytes())
    ext.prepare(host, device, rank, 0, 0)
    requests, errors, stopping = queue.Queue(), [], threading.Event()

    def work():
        last = 0
        while not stopping.is_set():
            published = ext.published(host)
            if published <= last:
                time.sleep(.00005)
                continue
            for sequence in range(last + 1, published + 1):
                try:
                    local, peers, delay = requests.get(timeout=5)
                    # The published payload is this rank's input: a sum that read before its producer finished
                    # would have published the poison written ahead of each replay.
                    test.assertTrue(torch.equal(ext.payload(host, sequence), local))
                    time.sleep(delay)
                    if peers is None:
                        ext.land(host, sequence)
                    else:
                        ext.land_peers(host, sequence, peers)
                except BaseException as error:
                    errors.append(error)
                    ext.land(host, sequence)  # let the GPU retire so the failure is reportable
            last = published

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


def packets(values, rank, delay=0.):
    encoded = values.contiguous().view(torch.uint8).reshape(4, -1).cpu()
    return encoded[rank].clone(), encoded[[r for r in range(4) if r != rank]].contiguous(), delay


def fixture(n, trial, generator):
    """[4, n] BF16 rank inputs. Trial 0 is the boot self-test's cancellation set, where local-first FP32 addition
    disagrees across ranks; later trials mix magnitudes so the FP32 sums round at the BF16 boundary."""
    if trial == 0:
        columns = torch.tensor(((2.**24, 256., 1., -1.), (-2.**24, -256., 2., 1.), (1., 2.**-16, 3., 2.**-24),
                                (1., 2.**-16, 4., -2.**-24)), device='cuda', dtype=torch.bfloat16)
        return columns.repeat(1, (n + 3) // 4)[:, :n].contiguous()
    exponent = torch.randint(-12, 13, (4, n), device='cuda', generator=generator)
    return (torch.randn(4, n, device='cuda', generator=generator) * torch.pow(2., exponent)).bfloat16()


def fold(values):
    """The transport's arithmetic, independently: global ranks 0, 1, 2, 3 in FP32, then one BF16 rounding."""
    total = values[0].float()
    for r in range(1, 4):
        total = total + values[r].float()
    return total.bfloat16()


def same_bytes(a, b):
    return a.shape == b.shape and torch.equal(a.view(torch.int16), b.view(torch.int16))


@unittest.skipUnless(torch.cuda.is_available(), 'requires admitted GB10')
class ConsumerSumCudaTests(unittest.TestCase):
    def test_consumer_and_ordinary_sums_are_the_same_bytes_at_every_rank_and_size(self):
        generator = torch.Generator(device='cuda').manual_seed(965)
        for rank in range(4):
            with proxy(self, rank) as (ext, requests):
                for shape in SHAPES:
                    n = math.prod(shape)
                    x = torch.empty(shape, device='cuda', dtype=torch.bfloat16)
                    for trial in range(3):
                        values = fixture(n, trial, generator)
                        expected = fold(values).view(shape)
                        x.copy_(values[rank].view(shape))
                        for name in KERNELS:
                            requests.put(packets(values, rank))
                            actual = getattr(ext, name)(x)
                            torch.cuda.synchronize()
                            self.assertTrue(same_bytes(actual, expected), (rank, shape, trial, name))
                        self.assertTrue(same_bytes(x, values[rank].view(shape)), 'the sum is out of place')
                    # int64 MAX continues the same publication ticket sequence
                    keys = torch.arange(3, device='cuda', dtype=torch.int64)
                    requests.put((keys.cpu().view(torch.uint8), None, 0.))
                    ext.oneshot_max_int64(keys)
                    torch.cuda.synchronize()
                    self.assertEqual(keys.cpu().tolist(), [0, 1, 2])

    def test_captured_sums_wait_for_a_late_producer_and_an_early_successor_waits_for_them(self):
        generator = torch.Generator(device='cuda').manual_seed(966)
        for rank in range(4):
            with proxy(self, rank) as (ext, requests):
                for shape in SHAPES:
                    n = math.prod(shape)
                    staged, x, out = (torch.empty(shape, device='cuda', dtype=torch.bfloat16) for _ in range(3))
                    for name in KERNELS:
                        reduce = getattr(ext, name)
                        values = fixture(n, 1, generator)
                        staged.copy_(values[rank].view(shape))
                        # The eager chain publishes once. Capture records the launches without running them, so it
                        # takes no proxy request, as in the other oracle tests; every replay then takes exactly one.
                        requests.put(packets(values, rank))
                        ext.staged_copy(staged, x, 0)
                        ext.staged_copy(reduce(x), out, 0)
                        torch.cuda.synchronize()
                        self.assertTrue(same_bytes(out, fold(values).view(shape)), (rank, shape, name, 'eager'))
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            ext.staged_copy(staged, x, PRODUCER_CYCLES)
                            summed = reduce(x)
                            ext.staged_copy(summed, out, 0)
                        try:
                            torch.cuda.synchronize()
                            for trial in range(3):
                                values = fixture(n, trial, generator)
                                staged.copy_(values[rank].view(shape))
                                for poisoned in (x, summed, out):
                                    poisoned.fill_(float('nan'))
                                torch.cuda.synchronize()
                                requests.put(packets(values, rank, LAND_DELAY_S))
                                graph.replay()
                                torch.cuda.synchronize()
                                expected = fold(values).view(shape)
                                self.assertTrue(same_bytes(summed, expected), (rank, shape, name, trial))
                                self.assertTrue(same_bytes(out, expected), (rank, shape, name, trial, 'successor'))
                        finally:
                            graph.reset()
                # One captured step's order of sums: C=2's consumer, a larger ordinary sum, C=1's consumer.
                shapes = ((16, HIDDEN), (17, HIDDEN), (8, HIDDEN))
                inputs = [torch.empty(shape, device='cuda', dtype=torch.bfloat16) for shape in shapes]
                kernels = (ext.oneshot_ar_consumer, ext.oneshot_ar, ext.oneshot_ar_consumer)
                rounds = [[fixture(math.prod(shape), trial, generator) for shape in shapes] for trial in range(3)]
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    outputs = [kernel(x) for kernel, x in zip(kernels, inputs)]
                try:
                    torch.cuda.synchronize()
                    for trial, step in enumerate(rounds):
                        for values, x in zip(step, inputs):
                            x.copy_(values[rank].view(x.shape))
                            requests.put(packets(values, rank, LAND_DELAY_S if trial else 0.))
                        graph.replay()
                        torch.cuda.synchronize()
                        for values, output in zip(step, outputs):
                            self.assertTrue(same_bytes(output, fold(values).view(output.shape)), (rank, trial))
                finally:
                    graph.reset()


if __name__ == '__main__':
    unittest.main()
