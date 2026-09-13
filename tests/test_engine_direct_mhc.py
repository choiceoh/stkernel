"""Packet lifetime and real four-rank arithmetic oracle for direct MHC order."""
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

from engine.base.comm import Comm, LocalTP
from engine.kernels.oneshot import OneShot, RankPackets
from engine.profiles.glm53.direct_mhc import decode_direct
from engine.profiles.glm53.execution import ExecutionPlan
from engine.profiles.glm53.net import Step
from tests.test_engine_execution_plans import model


class OracleTransport:
    """Real LocalTP fold with explicit exchange/consumer ownership."""
    def __init__(self, comm):
        self.comm, self.pending, self.count = comm, None, 0

    def assert_consumed(self):
        if self.pending is not None:
            raise RuntimeError("unconsumed packet")

    def exchange(self, x):
        self.assert_consumed()
        self.pending = self.comm.all_reduce(x.clone())
        self.count += 1
        return self.pending


def oracle_consume(net, layer, carry, side, packet):
    t = net.comm.transport
    assert t.pending is packet
    carry.res, carry.post, carry.comb, carry.x = net._hc_post_pre(
        layer, packet, carry.res, carry.post, carry.comb, side)
    t.pending = None


class DirectMhcTests(unittest.TestCase):
    def test_producer_reserves_before_writing_and_poison_prevents_fallback(self):
        events = []
        owner = OneShot.__new__(OneShot)
        owner.pending, owner.packet_failed, owner.closed = None, False, False
        owner.hidden, owner.world = 4096, 4
        owner.eligible = lambda t: True
        source, slot, descriptor = torch.empty(7, 4096), object(), torch.arange(4)
        owner.ext = NS(healthy=lambda: True,
                       reserve_packets=lambda t: events.append('reserve') or slot,
                       publish_packets=lambda t, r: events.append('publish') or descriptor)
        def producer(reservation):
            self.assertIs(reservation, slot)
            with self.assertRaises(RuntimeError):
                owner.assert_consumed()
            events.append('GEMM')
        with patch('torch.cuda.current_stream', return_value=NS(cuda_stream=19)):
            packet = owner.produce(source, producer)
            self.assertEqual(events, ['reserve', 'GEMM', 'publish'])
            self.assertIs(packet.consume(lambda t, d: d), descriptor)
            owner.assert_consumed()
            with self.assertRaisesRegex(ValueError, 'writer failed'):
                owner.produce(source, lambda r: (_ for _ in ()).throw(ValueError('writer failed')))
            with self.assertRaises(RuntimeError):
                owner.exchange(source)
            self.assertTrue(owner.packet_failed)

    def test_direct_output_preserves_multi_pack_and_observer_boundaries(self):
        from engine.kernels.dense import DenseLinear
        layer = DenseLinear.__new__(DenseLinear)
        layer.rows, layer.packs, layer.observer = 4096, (object(),), None
        for rows in (1, 7, 28, 32):
            self.assertIsNotNone(layer.slot_writer(rows))
        self.assertIsNone(layer.slot_writer(33))
        layer.observer = lambda x: None
        self.assertIsNone(layer.slot_writer(7))
        layer.observer, layer.packs = None, (object(), object())
        self.assertIsNone(layer.slot_writer(7))

    def test_four_rank_c1_c4_outputs_aux_and_state(self):
        torch.set_num_threads(1)
        def rank(comm):
            for count, producer in ((1, False), (4, False), (1, True), (4, True)):
                net, cache = model(("kda", "dsa", "kda"), comm=comm)
                chunks = []
                for seq in (2, 0, 3, 1)[:count]:
                    slot = cache.slots.take(seq)
                    cache.pool.reserve(seq, 7)
                    chunks.append(((torch.arange(7)+seq) % net.vp, 0, seq, slot))
                step = Step.decode(chunks)
                cache.prepare(step)
                state, paged = cache.state.clone(), cache.paged.clone()
                expected = net.forward(step, cache, aux_layers=[0, 2])
                after, paged_after = cache.state.clone(), cache.paged.clone()
                cache.state.copy_(state); cache.paged.copy_(paged)
                transport = OracleTransport(comm)
                if producer:
                    class Linear:
                        def __init__(self, weight):
                            self.weight = weight
                        def __call__(self, x):
                            return torch.nn.functional.linear(x, self.weight)
                        def slot_writer(self, rows):
                            return lambda x, slot: setattr(slot, 'value', self(x))
                    for name, weight in net.p.items():
                        if name.endswith('o_proj') or name.endswith('mlp.down'):
                            net.dense[name] = Linear(weight)
                    def produce(template, writer):
                        slot = NS(value=None)
                        writer(slot)
                        self.assertEqual(slot.value.shape, template.shape)
                        return transport.exchange(slot.value)
                    transport.produce = produce
                comm.transport = transport
                seen = []
                try:
                    actual = decode_direct(net, step, cache, [0, 2], seen.append, consumer=oracle_consume)
                finally:
                    comm.transport = None
                for a, b in zip(actual, expected):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                torch.testing.assert_close(cache.state, after, rtol=0, atol=0)
                torch.testing.assert_close(cache.paged, paged_after, rtol=0, atol=0)
                torch.testing.assert_close(seen[0], expected[1], rtol=0, atol=0)
                self.assertEqual(transport.count, 4)  # 3 attention + 1 non-aux FFN
                transport.assert_consumed()
            return actual[0]
        results = LocalTP(4, timeout_s=30).run(rank)
        for result in results[1:]:
            torch.testing.assert_close(result, results[0], rtol=0, atol=0)

    def test_packet_ownership_stream_failure_and_collective_guards(self):
        owner = NS(pending=None, packet_failed=False)
        source = torch.ones(4)
        with patch("torch.cuda.current_stream", return_value=NS(cuda_stream=19)) as stream:
            packet = RankPackets(owner, source, torch.arange(4))
            owner.pending = packet
            with self.assertRaises(RuntimeError):
                OneShot.assert_consumed(owner)
            for op in ("all_reduce", "all_reduce_max", "all_gather", "reduce_scatter_rows", "barrier"):
                with self.subTest(op=op), self.assertRaisesRegex(RuntimeError, "consume rank packets"):
                    fn = getattr(Comm(4, 0, transport=owner), op)
                    fn() if op == "barrier" else fn(source)
            stream.return_value = NS(cuda_stream=20)
            with self.assertRaisesRegex(RuntimeError, "exchange stream"):
                packet.consume(lambda x, p: x)
            self.assertIs(owner.pending, packet)
            stream.return_value = NS(cuda_stream=19)
            self.assertIs(packet.consume(lambda x, p: x), source)
            with self.assertRaisesRegex(RuntimeError, "stale"):
                packet.consume(lambda x, p: x)
            owner.pending = RankPackets(owner, source, torch.arange(4))
            with self.assertRaisesRegex(ValueError, "consumer failed"):
                owner.pending.consume(lambda x, p: (_ for _ in ()).throw(ValueError("consumer failed")))
            self.assertIsNone(owner.pending)
            with self.assertRaises(RuntimeError):
                OneShot.assert_consumed(owner)

    def test_default_and_unsupported_split(self):
        self.assertFalse(ExecutionPlan().direct_mhc)
        self.assertTrue(ExecutionPlan(direct_mhc=True).active)
        with self.assertRaisesRegex(ValueError, "unsplit"):
            ExecutionPlan(direct_mhc=True, overlap=True)
        with self.assertRaises(ValueError):
            ExecutionPlan(direct_mhc=1)


if __name__ == "__main__":
    unittest.main()
