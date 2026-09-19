"""Qwen3.8's MoE output finalized by the packet grid (engine/QWEN38_CARRY.md X2, GLM-5.3's MoE packets #904 #906): the
net sends a direct step's MoE layer to the transport's gated exchange, the transport checks what it is handed, and the
kernel's arithmetic is gated_sum's -- product then sum, each rounded. On a GB10 the TX bytes and the leave after them are
held to the finalizer and the consumer through the single-GPU oracle (probes/engine_qwen38_rank_packets `moe_checks`).

    docker exec -w <repo> stk-test python3 -m unittest tests.test_engine_qwen38_moe_packets
"""
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
torch = None
if importlib.util.find_spec("torch") is not None:
    import torch
READY = torch is not None and importlib.util.find_spec("triton") is not None


class Transport:
    def __init__(self):
        self.calls = []

    @staticmethod
    def eligible(t):
        return True

    def exchange_moe_gated(self, routed, shared, gate):
        self.calls.append(("gated", tuple(routed.shape), tuple(shared.shape), tuple(gate.shape)))
        return "packets"


@unittest.skipUnless(READY, "torch and triton required")
class FinishTests(unittest.TestCase):
    def net(self, direct, transport):
        reduced = []
        finished = []

        def moe_finish(routed, shared, gate):
            finished.append(tuple(gate.shape))
            return routed + shared

        comm = SimpleNamespace(transport=transport, _settled=lambda t: t,
                               all_reduce=lambda t: reduced.append(tuple(t.shape)) or t)
        net = SimpleNamespace(comm=comm, F=SimpleNamespace(hidden=8), _direct=direct,
                              lanes=SimpleNamespace(moe_finish=moe_finish))
        return net, reduced, finished

    def test_a_direct_step_hands_the_layer_to_the_gated_exchange(self):
        from engine.profiles.qwen38.net import Qwen38Net
        transport = Transport()
        net, reduced, finished = self.net(True, transport)
        routed, shared, gate = torch.zeros(4, 8).bfloat16(), torch.zeros(4, 8).bfloat16(), torch.zeros(4, 1)
        self.assertEqual(Qwen38Net._finish(net, routed, shared, gate), "packets")
        self.assertEqual(transport.calls, [("gated", (4, 8), (4, 8), (4,))])      # one gate a row, flat
        self.assertEqual((reduced, finished), ([], []))

    def test_otherwise_the_finalizer_then_the_sum(self):
        from engine.profiles.qwen38.net import Qwen38Net
        routed, shared, gate = torch.zeros(4, 8).bfloat16(), torch.zeros(4, 8).bfloat16(), torch.zeros(4, 1)
        for direct, transport, rows in ((False, Transport(), 4), (True, None, 4), (True, Transport(), 65)):
            net, reduced, finished = self.net(direct, transport)
            r, s, g = (torch.zeros(rows, 8).bfloat16(), torch.zeros(rows, 8).bfloat16(), torch.zeros(rows, 1))
            with self.subTest(direct=direct, transport=transport is not None, rows=rows):
                Qwen38Net._finish(net, r, s, g)
                self.assertEqual((finished, reduced), ([(rows, 1)], [(rows, 8)]))
                if transport is not None:
                    self.assertEqual(transport.calls, [])

    def test_the_moe_layer_finishes_through_it_forked_or_not(self):
        source = (ROOT / "engine/profiles/qwen38/net.py").read_text(encoding="utf-8")
        self.assertIn("return Qwen38Net._finish(self, routed, shared, gate)", source)
        self.assertIn("finish=lambda parts, shared: Qwen38Net._finish(self, parts[0], shared, parts[1]))", source)


@unittest.skipUnless(READY, "torch and triton required")
class TransportTests(unittest.TestCase):
    def owner(self):
        from engine.kernels.oneshot import OneShot
        owner = OneShot.__new__(OneShot)
        owner.pending, owner.packet_failed, owner.closed = None, False, False
        owner.hidden, owner.world = 8, 4
        owner.eligible = lambda t: True
        owner.ext = SimpleNamespace(healthy=lambda: True, moe_gated_packets=lambda r, s, g: torch.arange(4))
        return owner

    def test_the_gated_exchange_checks_what_it_is_handed(self):
        routed, shared, gate = torch.zeros(4, 8).bfloat16(), torch.zeros(4, 8).bfloat16(), torch.zeros(4)
        with mock.patch("torch.cuda.current_stream", return_value=SimpleNamespace(cuda_stream=7)):
            packet = self.owner().exchange_moe_gated(routed, shared, gate)
            self.assertEqual(packet.consume(lambda own, descriptor: descriptor.tolist()), [0, 1, 2, 3])
            for bad in ((routed.float(), shared, gate), (routed, shared, gate.bfloat16()), (routed, shared, gate[:3]),
                        (routed[:3], shared, gate), (torch.zeros(65, 8).bfloat16(), torch.zeros(65, 8).bfloat16(),
                                                    torch.zeros(65))):
                with self.assertRaises(ValueError):
                    self.owner().exchange_moe_gated(*bad)


class SourceTests(unittest.TestCase):
    def test_the_grid_computes_gated_sums_arithmetic_into_tx(self):
        source = (ROOT / "engine/kernels/oneshot/dsv4_oneshot_ar.cu").read_text(encoding="utf-8")
        body = source[source.index("if constexpr (MOE_GATED) {"):]
        self.assertIn("__fadd_rn(__bfloat162float(part.values[j]),\n"
                      "                                                         __fmul_rn(__bfloat162float(shared.values[j]), g))",
                      body)                                         # no fused multiply-add: gated_sum's rounding
        self.assertIn("const float g = __ldg(gate + (v << 3) / width);", body)
        self.assertIn("MOE_OUTPUT || MOE_GATED ? c->tx[slot] : src", source)   # the local packet is the TX slot
        self.assertIn("k_oneshot_impl<true, false, true, false, true, false, false, true>(", source)
        self.assertIn('m.def("moe_gated_packets", &py_moe_gated_packets);', source)
        oracle = (ROOT / "probes/oneshot_producer_oracle.cu").read_text(encoding="utf-8")
        self.assertIn('m.def("moe_gated_packets", &py_moe_gated_packets);', oracle)
        finisher = (ROOT / "engine/kernels/moe_output.py").read_text(encoding="utf-8")
        self.assertIn("(a + b * g).to(Destination.dtype.element_ty)", finisher)
        self.assertIn("enable_fp_fusion=False)", finisher)

    def test_the_lane_runs_the_oracle_checks(self):
        check = (ROOT / "probes/engine_kernel_check.py").read_text(encoding="utf-8")
        self.assertIn("args.lanes == 'qwen38_moe_packets'", check)
        self.assertIn("args.lanes == 'direct_producer_timing'", check)


if __name__ == "__main__":
    unittest.main()
