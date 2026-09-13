"""Real CPU/GPU publication and cancellation; no model-quality claim."""
import time
import unittest
import torch


@unittest.skipUnless(torch.cuda.is_available(), "requires admitted GB10 GPU")
class DecodeQueueCudaTests(unittest.TestCase):
    def payload(self, n, t, value):
        return dict(tokens=torch.full((n, t), value, device="cuda", dtype=torch.int64),
                    count=torch.full((n,), t, device="cuda", dtype=torch.int64),
                    done=torch.zeros(n, device="cuda", dtype=torch.bool),
                    accepted=torch.full((n,), t-1, device="cuda", dtype=torch.int64),
                    before=torch.full((n,), value, device="cuda", dtype=torch.int64))

    def test_cpu_reads_each_immutable_iteration_before_the_burst_retires(self):
        from engine.kernels.decode_queue import SharedDecodeQueue
        for n in (1, 4):
            q = SharedDecodeQueue(4, 7)
            payloads = [self.payload(n, 7, j+20) for j in range(4)]
            indices = [torch.tensor([j], device="cuda") for j in range(4)]
            torch.cuda.synchronize()
            q.begin()
            end = torch.cuda.Event()
            for payload, index in zip(payloads, indices):
                q.publish(payload, index)
                torch.cuda._sleep(50000000)
            end.record()
            early, rows = False, []
            deadline = time.monotonic() + 10
            for j in range(4):
                while True:
                    actual = q.take(j, n)
                    if actual is not None:
                        break
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(.00005)
                early |= not end.query()
                self.assertEqual(actual["tokens"], [[j+20]*7]*n)
                self.assertEqual(actual["count"], [7]*n)
                self.assertEqual(actual["accepted"], [6]*n)
                self.assertEqual(actual["before"], [j+20]*n)
                self.assertEqual(actual["done"], [False]*n)
                rows.append(actual)
            end.synchronize()
            self.assertTrue(early, "a shared queue must publish before final stream synchronization")
            self.assertEqual([q.take(j, n) for j in range(4)], rows)

    def test_system_atomic_cancellation_and_capture_replay_reset(self):
        from engine.kernels.decode_queue import SharedDecodeQueue
        q = SharedDecodeQueue(4, 7)
        interrupt = torch.zeros(1, device="cuda", dtype=torch.int64)
        payload, index = self.payload(4, 7, 11), torch.zeros(1, device="cuda", dtype=torch.int64)
        q.publish(payload, index)
        q.read_interrupt(interrupt)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            torch.cuda._sleep(50000000)
            q.read_interrupt(interrupt)
            q.publish(payload, index)
        try:
            for value, cancelled in ((21, True), (32, False), (45, True)):
                q.begin()
                payload["tokens"].fill_(value)
                graph.replay()
                if cancelled:
                    q.cancel()  # CPU release while GPU work is in flight
                torch.cuda.synchronize()
                self.assertEqual(interrupt.item(), int(cancelled))
                self.assertEqual(q.take(0, 4)["tokens"], [[value]*7]*4)
                self.assertIsNone(q.take(1, 4))
        finally:
            torch.cuda.synchronize()
            graph.reset()


if __name__ == "__main__":
    unittest.main()
