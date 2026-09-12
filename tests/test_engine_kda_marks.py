"""Block-level prefix (45차 §23): cutting the KDA prefill recurrence at a step's marks changes nothing but yields the
states at the cuts -- the reference lanes on CPU, one KDA layer of the tiny facts, world-1 collectives."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from engine.base.comm import Comm, LocalTP  # noqa: E402
from engine.profiles.glm53 import lanes, specs  # noqa: E402
from engine.profiles.glm53.net import Glm53Net, Step  # noqa: E402
from tests.test_engine_glm53 import tiny_facts  # noqa: E402


class Caches:
    """One slot's KDA rings and a recorder for what the marks wrote."""
    def __init__(self, F, layers):
        self.F = F
        self.rings = {L: (torch.zeros(3 * F.kda_heads_local * F.kda_dim, F.conv - 1 + F.spec_k, dtype=torch.bfloat16),
                          torch.zeros(F.spec_k + 1, F.kda_heads_local, F.kda_dim, F.kda_dim)) for L in layers}
        self.marks = {}

    def kda(self, layer, slot):
        return self.rings[layer]

    def mark_kda(self, layer, snap, state, taps):
        self.marks[layer, snap] = (state.clone(), taps.clone())


def kda_net(F, seed=0):
    net = Glm53Net(F, LocalTP(4).rank(0), lanes.reference(), layers=[0])
    net.comm = Comm(1, 0, None)                                          # identity collectives: one rank's arithmetic
    g = torch.Generator().manual_seed(seed)
    net.p = {}
    for s in specs.layer_specs(F, 0):
        if ".kda." in s.name:
            t = torch.randn(s.shape, generator=g) * 0.2
            if s.name.endswith("o_norm"):
                t = torch.ones(s.shape)
            net.p[s.name] = t.to(s.dtype)
    return net


class KdaMarkTests(unittest.TestCase):
    def test_cutting_at_marks_keeps_the_output_and_yields_the_states_at_the_cuts(self):
        F = tiny_facts()
        net = kda_net(F)
        N, marks = 200, ((64, 0), (128, 1))                                # marks sit on the lane's 64-token kernel chunks
        x = (torch.randn(N, F.hidden, generator=torch.Generator().manual_seed(1)) * 0.5).to(torch.bfloat16)
        ids = torch.zeros(N, dtype=torch.int64)
        plain, cut = Caches(F, [0]), Caches(F, [0])
        out_plain = net._kda(0, x, Step.prefill(ids, 0, 1, 1), plain).float()
        out_cut = net._kda(0, x, Step.prefill(ids, 0, 1, 1, marks=marks), cut).float()
        torch.testing.assert_close(out_cut, out_plain, atol=2e-2, rtol=2e-2)
        self.assertEqual(sorted(cut.marks), [(0, 0), (0, 1)])
        # the final ring state is the same either way
        torch.testing.assert_close(cut.rings[0][1][(N - 1) % (F.spec_k + 1)], plain.rings[0][1][(N - 1) % (F.spec_k + 1)], atol=2e-2, rtol=2e-2)
        # the state at a mark is the state of the prefix alone
        for position, snap in marks:
            alone = Caches(F, [0])
            net._kda(0, x[:position], Step.prefill(ids[:position], 0, 1, 1), alone)
            torch.testing.assert_close(cut.marks[0, snap][0], alone.rings[0][1][(position - 1) % (F.spec_k + 1)], atol=2e-2, rtol=2e-2)
            # and the conv taps are the projected inputs of the conv-1 positions before it
            qkv = torch.nn.functional.linear(x, net.p["L0.kda.in_proj"])[:, : 3 * F.kda_heads_local * F.kda_dim]
            torch.testing.assert_close(cut.marks[0, snap][1], qkv[position - (F.conv - 1): position], atol=0, rtol=0)

    def test_a_mark_off_the_kernel_chunk_is_refused(self):
        F = tiny_facts()
        net = kda_net(F)
        x = torch.randn(80, F.hidden).to(torch.bfloat16)
        with self.assertRaises(ValueError):
            net._kda(0, x, Step.prefill(torch.zeros(80, dtype=torch.int64), 0, 1, 1, marks=((16, 0),)), Caches(F, [0]))

    def test_marks_are_validated_by_the_step(self):
        ids = torch.zeros(10, dtype=torch.int64)
        with self.assertRaises(ValueError):
            Step.prefill(ids, 0, 1, 1, marks=((10, 0),))                  # at the end: not inside
        with self.assertRaises(ValueError):
            Step.prefill(ids, 0, 1, 1, marks=((6, 0), (4, 1)))            # must increase
        with self.assertRaises(ValueError):
            Step.decode([(ids[:6], 0, 1, 1), (ids[6:], 0, 2, 2)]).__class__(ids, Step.decode([(ids[:6], 0, 1, 1), (ids[6:], 0, 2, 2)]).segments, (), ((3, 0),))


if __name__ == "__main__":
    unittest.main()
