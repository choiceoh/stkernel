"""The ranks' packets folded by the leave after a sum (engine/QWEN38_CARRY.md H5, GLM-5.3's direct MHC #812): the leave's
fold against the one-shot consumer's rank-ordered sum byte for byte, the lanes consuming an exchange, the net sending
a captured step's sums as packets and every packet consumed by the next leave, and the knob down to the launcher. On a
GB10 the fold is held to the real consumer through the single-GPU oracle (tests/test_engine_qwen38_rank_packets_cuda).

    docker exec -e TRITON_INTERPRET=1 -w <repo> stk-test python3 -m unittest tests.test_engine_qwen38_rank_packets
"""
import importlib.util
import inspect
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
torch = None
if importlib.util.find_spec("torch") is not None:
    import torch
READY = torch is not None and importlib.util.find_spec("triton") is not None
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
HC, EPS = 4, 1e-6


def packets(rows, hidden, seed=0):
    """Four ranks' BF16 packets [rows, hidden] whose sum cancels: a fold out of rank order moves the result (the one-shot
    self-test's fixture, spread over random values)."""
    gen = torch.Generator().manual_seed(seed)
    ranks = [torch.randn(rows, hidden, generator=gen) for _ in range(4)]
    cancel = ((2.0 ** 24, 256.0, 1.0, -1.0), (-2.0 ** 24, -256.0, 2.0, 1.0), (1.0, 2.0 ** -16, 3.0, 2.0 ** -24),
              (1.0, 2.0 ** -16, 4.0, -2.0 ** -24))
    for r in range(4):
        ranks[r][:, :4] = torch.tensor(cancel[r])
    return [t.bfloat16() for t in ranks]


def rank_ordered(ranks):
    """engine/kernels/oneshot's osar_sum_rank_order: ((r0 + r1) + r2) + r3 in fp32, rounded once."""
    acc = ranks[0].float() + ranks[1].float()
    acc = acc + ranks[2].float()
    return (acc + ranks[3].float()).bfloat16()


@unittest.skipUnless(READY and INTERPRET, "TRITON_INTERPRET=1 with torch and triton")
class FoldTests(unittest.TestCase):
    def test_the_leave_folds_the_ranks_as_the_consumer_does(self):
        from engine.kernels import gated_residual as hcr
        from tests.test_engine_qwen38_kernels import served_kernels
        hidden = 16
        for rows in (1, 3):
            for rank in range(4):
                ranks = packets(rows, hidden, seed=rows * 10 + rank)
                descriptor = torch.tensor([t.data_ptr() for t in ranks], dtype=torch.int64)
                gen = torch.Generator().manual_seed(rank)
                h = torch.randn(rows, HC * hidden, generator=gen).bfloat16()
                inject = (torch.rand(rows, HC, generator=gen) * 2).bfloat16()
                w = (torch.randn(HC * hidden, generator=gen) * 0.1).bfloat16()
                with served_kernels():
                    want = hcr.leave_norm(h.clone(), rank_ordered(ranks), inject, w, EPS, HC)
                    got = hcr.leave_norm(h.clone(), ranks[rank], inject, w, EPS, HC, packets=descriptor)
                    left = hcr.leave(h.clone(), ranks[rank], inject, HC, packets=descriptor)
                    left_want = hcr.leave(h.clone(), rank_ordered(ranks), inject, HC)
                with self.subTest(rows=rows, rank=rank):
                    self.assertTrue(all(torch.equal(a, b) for a, b in zip(got, want)))
                    self.assertTrue(torch.equal(left, left_want))

    def test_the_fixture_sees_the_order(self):
        """A negative control: rank 2 folding its own packet first (the local-first order the consumer's canonical order
        replaced) gives another sum on this fixture -- 1 where rank order gives 2 -- so the test above would fail it."""
        ranks = packets(2, 16)
        local_first = ((ranks[2].float() + ranks[0].float()) + ranks[1].float() + ranks[3].float()).bfloat16()
        self.assertEqual(float(rank_ordered(ranks)[0, 0]), 2.0)
        self.assertEqual(float(local_first[0, 0]), 1.0)


@unittest.skipUnless(READY, "torch and triton required")
class LeaveArgumentTests(unittest.TestCase):
    def test_packets_are_four_device_addresses_beside_a_packed_packet(self):
        from engine.kernels import gated_residual as hcr
        h, own = torch.zeros(2, HC * 8).bfloat16(), torch.zeros(2, 8).bfloat16()
        inject, w = torch.zeros(2, HC).bfloat16(), torch.zeros(HC * 8).bfloat16()
        with self.assertRaises(ValueError):                                  # the host path cannot fold addresses
            hcr.leave_norm(h, own, inject, w, EPS, HC, packets=torch.zeros(4, dtype=torch.int64))
        with mock.patch.object(torch.Tensor, "is_cuda", property(lambda tensor: True)):
            for bad in (torch.zeros(3, dtype=torch.int64), torch.zeros(4, dtype=torch.int32)):
                with self.assertRaises(ValueError):
                    hcr.leave_norm(h, own, inject, w, EPS, HC, packets=bad)

    def test_the_kernel_reads_the_descriptor_after_the_wait(self):
        from engine.kernels import gated_residual as hcr
        source = inspect.getsource(hcr._leave_norm.fn)
        self.assertGreater(source.index("o = _rank_sum(DESC"), source.index("tl.extra.cuda.gdc_wait()"))
        fold = inspect.getsource(hcr._rank_sum.fn)
        self.assertIn("for k in tl.static_range(1, 4):", fold)                # ranks 1, 2, 3 onto rank 0, in order
        self.assertIn("return acc.to(tl.bfloat16).to(tl.float32)", fold)     # rounded once, as the consumer's store


class Packet:
    """RankPackets' contract as the lanes and the net see it: consumed once, by the leave, with (own, descriptor)."""
    def __init__(self, transport, own):
        self.transport, self.own = transport, own

    def consume(self, fn):
        if self.transport.pending is not self:
            raise RuntimeError("stale packet")
        self.transport.pending = None
        self.transport.events.append("consume")
        return fn(self.own, "descriptor")


class Transport:
    def __init__(self):
        self.pending, self.events = None, []

    @staticmethod
    def eligible(t):
        return True

    def exchange(self, t):
        if self.pending is not None:
            raise RuntimeError("the previous rank-packet consumer did not complete")
        self.events.append("exchange")
        self.pending = Packet(self, t)
        return self.pending

    def assert_consumed(self):
        if self.pending is not None:
            raise RuntimeError("unconsumed packet")


@unittest.skipUnless(READY, "torch and triton required")
class NetTests(unittest.TestCase):
    def comm(self, transport):
        from engine.base.comm import Comm
        reduced = []

        def all_reduce(t):
            if transport is not None and transport.pending is not None:
                raise RuntimeError("consume rank packets before the next device collective")
            reduced.append(tuple(t.shape))
            return t
        return SimpleNamespace(transport=transport, _settled=Comm._settled, all_reduce=all_reduce), reduced

    def test_a_sum_is_exchanged_only_inside_a_direct_step(self):
        from engine.profiles.qwen38.net import Qwen38Net
        transport = Transport()
        comm, reduced = self.comm(transport)
        net = SimpleNamespace(comm=comm, F=SimpleNamespace(hidden=8), _direct=False)
        t = torch.zeros(4, 8).bfloat16()
        self.assertIs(Qwen38Net._sum(net, t), t)
        self.assertEqual(reduced, [(4, 8)])
        net._direct = True
        self.assertIsInstance(Qwen38Net._sum(net, t), Packet)
        transport.pending = None
        self.assertEqual(tuple(Qwen38Net._sum(net, torch.zeros(65, 8).bfloat16()).shape), (65, 8))   # past 64 rows
        self.assertEqual(tuple(Qwen38Net._sum(net, torch.zeros(4, 16).bfloat16()).shape), (4, 16))   # off the width
        self.assertEqual(transport.events, ["exchange"])
        self.assertEqual(reduced, [(4, 8), (65, 8), (4, 16)])

    def test_packets_need_the_transport_and_leaves_that_fold_them(self):
        from engine.profiles.qwen38.net import Qwen38Net
        for rank_packets, folds, transport, live in ((True, True, Transport(), True), (False, True, Transport(), False),
                                                     (True, False, Transport(), False), (True, True, None, False)):
            net = SimpleNamespace(rank_packets=rank_packets, lanes=SimpleNamespace(packets=folds),
                                  comm=SimpleNamespace(transport=transport))
            with self.subTest(rank_packets=rank_packets, folds=folds, transport=transport is not None):
                self.assertEqual(Qwen38Net._packets_live(net), live)

    def test_every_packet_of_a_captured_step_is_consumed_by_the_next_leave(self):
        """Qwen38Net.forward over stand-in sublayers: three layers, the PLE site before the second; each sublayer's sum
        goes through `_sum`, every leave takes what it was handed. The exchange/consume order is strict, and the
        embedding's and the PLE's sums stay reduced."""
        from engine.profiles.qwen38.net import Qwen38Net
        transport = Transport()
        comm, reduced = self.comm(transport)
        rows, hidden = 2, 8
        leaves = []

        def leave(h, out, inject, hc, **kw):
            leaves.append("leave")
            return out.consume(lambda own, d: h) if hasattr(out, "consume") else h

        def leave_norm(h, out, inject, w, eps, hc, prefetch=None):
            leaves.append("leave_norm")
            h = out.consume(lambda own, d: h) if hasattr(out, "consume") else h
            return h, h

        from collections import defaultdict
        lanes = SimpleNamespace(packets=True, hc_leave=leave, hc_leave_norm=leave_norm, hc_norm=lambda h, *a: h)
        F = SimpleNamespace(hc=HC, rms_eps=EPS, hidden=hidden, ple_layers=(1,), is_qsa=lambda L: L == 2)
        net = SimpleNamespace(F=F, p=defaultdict(lambda: None), lanes=lanes, comm=comm, rank_packets=True,
                              _direct=False, layers=[0, 1, 2], _hc_projections={})
        sum_ = lambda t: Qwen38Net._sum(net, t)

        def ple(L, h, *args):
            comm.all_reduce(torch.zeros(rows, hidden).bfloat16())               # the PLE rows' own sum, reduced
            return torch.zeros_like(h)

        net.step_meta = lambda step, caches: None
        net.embed = lambda ids: comm.all_reduce(torch.zeros(rows, hidden).bfloat16())
        net._ple_inject_rows = net._ple_inject = ple
        net._gdn = lambda L, x, step, caches: comm.all_reduce(torch.ones(rows, hidden).bfloat16())
        net._mixer_weight = lambda prefix, down: None
        net._mix = lambda prefix, normed, down, inject: (torch.zeros(rows, hidden).bfloat16(), None)
        net._site = lambda prefix, h, out, inject: Qwen38Net._site(net, prefix, h, out, inject)
        net._qsa = lambda L, x, step, meta, caches: sum_(torch.ones(rows, hidden).bfloat16())
        net._gdn_rows = lambda L, x, step, caches: sum_(torch.ones(rows, hidden).bfloat16())
        net._moe = lambda n, x, compact: sum_(torch.ones(rows, hidden).bfloat16())
        net._packets_live = lambda: Qwen38Net._packets_live(net)
        step = SimpleNamespace(ids=torch.zeros(rows, dtype=torch.int64), captured=True)
        Qwen38Net.forward(net, step, None)
        self.assertEqual(transport.events, ["exchange", "consume"] * 6)            # three layers, two sums each
        self.assertEqual(leaves.count("leave"), 1)                                  # the PLE site's
        self.assertEqual(reduced, [(rows, hidden), (rows, hidden)])                 # the embedding and the PLE rows
        self.assertFalse(net._direct)
        transport.events.clear()
        Qwen38Net.forward(net, SimpleNamespace(ids=step.ids, captured=False, segments=()), None)
        self.assertEqual(transport.events, [])                                       # an eager step reduces


class KnobTests(unittest.TestCase):
    def test_the_knob_reaches_the_net_from_the_launcher(self):
        fleet = (ROOT / "engine/profiles/qwen38/fleet.py").read_text(encoding="utf-8")
        self.assertIn('ap.add_argument("--no-rank-packets", action="store_true",', fleet)
        self.assertIn("rank_packets=not a.no_rank_packets)", fleet)
        self.assertIn("shared_overlap=shared_overlap, rank_packets=rank_packets)", fleet)
        self.assertIn('print("  sums: "', fleet)
        launcher = (ROOT / "launchers/start-st-qwen38.sh").read_text(encoding="utf-8")
        self.assertIn('case "${ST_RANK_PACKETS:-1}" in', launcher)
        self.assertIn('0) PACKETS_ARG="--no-rank-packets" ;;', launcher)
        self.assertIn("$SEQS_ARG $PACKETS_ARG $DRAFTER_ARG", launcher)

    def test_the_transport_names_any_bound_width(self):
        source = (ROOT / "engine/kernels/oneshot/dsv4_oneshot_ar.cu").read_text(encoding="utf-8")
        packets = source[source.index("static at::Tensor py_oneshot_packets"):]
        packets = packets[:packets.index("auto addresses")]
        self.assertIn("input.size(1) > 0 && input.size(1) % 8 == 0", packets)
        self.assertNotIn("input.size(1) == 4096", packets)
        init = (ROOT / "engine/kernels/oneshot/__init__.py").read_text(encoding="utf-8")
        self.assertIn("PACKET_ROWS = 64", init)
        self.assertIn("or t.shape[0] > PACKET_ROWS):", init)


if __name__ == "__main__":
    unittest.main()
