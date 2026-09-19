"""The text model's interleaved multimodal rotary (mRoPE) for a step with a picture's rows: engine/modules/rotary's tables
from [3, N] positions against vLLM's rule, and the two fused QSA input launches (engine/kernels/qsa.qsa_inputs,
qsa_index_keys) with [3, N] positions and first-member positions held byte for byte to themselves at 1-D positions.

A rotary pair turns by its own angle and nothing else, so a [3, N] launch must equal -- pair by pair -- the 1-D launch at
the axis that pair reads (pair i: t where i % 3 == 0, h where 1, w where 2 for the 32 pairs of sections 11/11/10), and
the unrotated channels must equal every run's. And a text row's three axes are one position: [3, N] with equal axes is
the 1-D launch, byte for byte.

    TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES= python3 -m unittest tests.test_engine_qwen38_mrope
"""
import unittest

import torch

from tests.test_engine_qwen38_kernels import EPS, RUNS, RUNS_REASON, THETA, W, served_kernels
from tests.test_engine_qwen38_qsa_inputs import fresh, layer_case, split

SECTIONS = (11, 11, 10)


def axis_of_pairs(pairs: int, sections=SECTIONS) -> torch.Tensor:
    """The axis each rotary pair reads: vLLM's apply_interleaved_rope masks."""
    i = torch.arange(pairs)
    axis = torch.zeros(pairs, dtype=torch.long)
    axis[(i % 3 == 1) & (i < 3 * sections[1])] = 1
    axis[(i % 3 == 2) & (i < 3 * sections[2])] = 2
    return axis


class TablesTests(unittest.TestCase):
    def test_equal_axes_are_the_text_tables(self):
        from engine.modules.rotary import rope_tables
        pos = torch.tensor([0, 3, 17, 4096, 262143])
        want = rope_tables(pos, 64, THETA)
        got = rope_tables(pos.expand(3, -1), 64, THETA, mrope_section=SECTIONS, interleaved=True)
        self.assertTrue(torch.equal(got[0], want[0]) and torch.equal(got[1], want[1]))

    def test_each_frequency_reads_its_axis(self):
        from engine.modules.rotary import rope_tables
        pos = torch.tensor([[5, 9, 0], [2, 40, 7], [11, 1, 3]])                       # [3 axes, 3 tokens]
        cos, sin = rope_tables(pos, 64, THETA, mrope_section=SECTIONS, interleaved=True)
        axis = axis_of_pairs(32)
        for a in range(3):
            c, s = rope_tables(pos[a], 64, THETA)
            pick = torch.cat([axis == a, axis == a])                                  # both neox halves of a pair
            self.assertTrue(torch.equal(cos[:, pick], c[:, pick]) and torch.equal(sin[:, pick], s[:, pick]))
        self.assertEqual([int((axis == a).sum()) for a in range(3)], list(SECTIONS))

    def test_the_sections_must_be_given(self):
        from engine.modules.rotary import rope_tables
        with self.assertRaises(ValueError):
            rope_tables(torch.zeros(3, 4, dtype=torch.long), 64, THETA)


class StepTests(unittest.TestCase):
    def test_a_step_s_rotary_positions_come_in_pairs_and_shapes(self):
        from engine.profiles.qwen38.net import Step
        ids = torch.zeros(5, dtype=torch.int64)
        rope = torch.arange(15, dtype=torch.int64).view(3, 5)
        Step.prefill(ids, 0, 0, 1)                                                     # text: none
        from engine.profiles.qwen38.net import Segment
        seg = (Segment(0, 1, 0, 0, 5),)
        Step(ids, seg, rope=rope, rope_first=rope - 3)
        Step(ids, seg, rope=rope[0].contiguous(), rope_first=rope[0] - 3)
        for bad in (dict(rope=rope), dict(rope=rope[:2], rope_first=rope[:2]), dict(rope=rope.int(), rope_first=rope.int())):
            with self.subTest(bad=list(bad)), self.assertRaises(ValueError):
                Step(ids, seg, **bad)

    def test_the_host_step_meta_carries_them(self):
        from types import SimpleNamespace
        from engine.profiles.qwen38.net import Qwen38Net, Segment, Step
        ids = torch.zeros(6, dtype=torch.int64)
        rope = torch.stack([torch.arange(6), torch.arange(6) * 2, torch.arange(6) * 3]).to(torch.int64)
        step = Step(ids, (Segment(0, 1, 0, 0, 6),), rope=rope, rope_first=rope - 3)
        net = SimpleNamespace(F=SimpleNamespace(block=64, idx_ratio=4))
        caches = SimpleNamespace(block_table=torch.zeros(1, 4, dtype=torch.int32))
        meta = Qwen38Net.step_meta(net, step, caches)
        self.assertIs(meta.rope, rope)
        self.assertTrue(torch.equal(meta.positions, torch.arange(6)))                  # the caches' positions stay
        text = Qwen38Net.step_meta(net, Step.prefill(ids, 0, 0, 1), caches)
        self.assertIsNone(text.rope)
        self.assertIsNone(text.rope_first)

    def test_the_draft_chain_keeps_the_observation_s_deltas(self):
        from engine.profiles.qwen38.decode_graphs import draft_chain
        from engine.profiles.qwen38.net import DeviceStep
        from tests.test_engine_qwen38_draft_chain import FakeNet
        seen = []
        net = FakeNet()
        forward = net.mtp_forward
        net.mtp_forward = lambda step, *a, **kw: (seen.append(step.deltas), forward(step, *a, **kw))[1]
        deltas = torch.tensor([-7, 0])
        step = DeviceStep(torch.tensor([11, 12, 21, 22]), torch.tensor([10, 20]), torch.tensor([1, 2]),
                          torch.tensor([0, 1]), 2, 4, deltas)
        draft_chain(net, None, step, torch.zeros(4, 2), torch.tensor([1, 3]), torch.tensor([2, 2]), 3)
        self.assertEqual(len(seen), 3)
        self.assertTrue(all(d is deltas for d in seen))


def rows_by_axis(t: torch.Tensor, rotary: int, axis: torch.Tensor, a: int) -> torch.Tensor:
    """The channels of a head [..., D] that pairs reading axis `a` write: i and R2 + i for each such pair."""
    half = rotary // 2
    mask = torch.zeros(t.shape[-1], dtype=torch.bool)
    mask[:half] = axis == a
    mask[half:rotary] = axis == a
    return t[..., mask]


@unittest.skipUnless(RUNS, RUNS_REASON)
class KernelTests(unittest.TestCase):
    def positions3(self, case):
        """Distinct (t, h, w) for every row: t the cache position, h and w moved apart from it and from each other."""
        p = case.meta.positions
        return torch.stack([p, (p * 7 + 3) % 97, (p * 13 + 11) % 89])

    def inputs(self, case, c, positions):
        from engine.kernels import qsa
        m, w = case.meta, case.weights
        qg, k, v, iq_rows, ik = split(case)
        return qsa.qsa_inputs(qg[..., :W.head_dim], k, v, iq_rows, ik, positions, w["q"], w["k"], w["iq"], EPS, THETA,
                              W.rotary, c["k"], c["v"], m.kv_slots, c["ring"], m.ring_slots)

    def index_keys(self, case, c, rope_first):
        from engine.kernels import qsa
        m, w = case.meta, case.weights
        *_, ik = split(case)
        qsa.qsa_index_keys(ik, c["ring"], m.slot_table, m.rows_req, m.starts, m.positions, m.key_slots, W.ratio, w["ik"],
                           EPS, THETA, W.rotary, c["keys"], rope_first=rope_first)

    def test_equal_axes_are_the_text_launch(self):
        case = layer_case(91)
        p = case.meta.positions
        first = p - (W.ratio - 1)
        runs = {}
        with served_kernels():
            for name, pos, rope_first in (("text", p, None), ("text rope", p, first),
                                          ("mrope", p.expand(3, -1).contiguous(), first.expand(3, -1).contiguous())):
                views, storages = fresh(case)
                self.index_keys(case, views, rope_first)
                runs[name] = (self.inputs(case, views, pos), storages)
        (q0, iq0), c0 = runs["text"]
        for name in ("text rope", "mrope"):
            (q, iq), c = runs[name]
            with self.subTest(name):
                self.assertTrue(torch.equal(q, q0) and torch.equal(iq, iq0))
                for cache in ("k", "v", "keys", "ring"):
                    self.assertTrue(torch.equal(c[cache], c0[cache]), cache)

    def test_each_pair_turns_at_its_axis(self):
        case = layer_case(92)
        pos3 = self.positions3(case)
        first3 = pos3 - (W.ratio - 1)                        # any first-member positions: the kernel only reads them
        axis = axis_of_pairs(W.rotary // 2)
        with served_kernels():
            views, got = fresh(case)
            self.index_keys(case, views, first3)
            q3, iq3 = self.inputs(case, views, pos3)
            per_axis = []
            for a in range(3):
                views, storages = fresh(case)
                self.index_keys(case, views, first3[a].contiguous())
                per_axis.append((self.inputs(case, views, pos3[a].contiguous()), storages))
        m = case.meta
        live_k = m.kv_slots.long()
        live_keys = m.key_slots[m.key_slots >= 0].long()

        def cache_rows(storage, slots, view_of):
            return view_of(storage).reshape(-1, view_of(storage).shape[-1])[slots]

        def as_view(name):
            view, _ = case.caches[name]
            return lambda st: st.as_strided(view.shape, view.stride(), view.storage_offset())

        for a in range(3):
            (qa, iqa), ca = per_axis[a]
            with self.subTest(axis=a):
                self.assertTrue(torch.equal(rows_by_axis(q3, W.rotary, axis, a), rows_by_axis(qa, W.rotary, axis, a)))
                self.assertTrue(torch.equal(rows_by_axis(iq3, W.rotary, axis, a), rows_by_axis(iqa, W.rotary, axis, a)))
                self.assertTrue(torch.equal(q3[..., W.rotary:], qa[..., W.rotary:]))
                k3 = cache_rows(got["k"], live_k, as_view("k"))
                ka = cache_rows(ca["k"], live_k, as_view("k"))
                self.assertTrue(torch.equal(rows_by_axis(k3, W.rotary, axis, a), rows_by_axis(ka, W.rotary, axis, a)))
                x3 = cache_rows(got["keys"], live_keys, as_view("keys"))
                xa = cache_rows(ca["keys"], live_keys, as_view("keys"))
                self.assertTrue(torch.equal(rows_by_axis(x3, W.rotary, axis, a), rows_by_axis(xa, W.rotary, axis, a)))
                self.assertTrue(torch.equal(got["v"], ca["v"]) and torch.equal(got["ring"], ca["ring"]))
        # and the axes were different enough to matter: the 3-D launch is none of the 1-D ones
        self.assertFalse(any(torch.equal(q3, qa) for (qa, _), _ in per_axis))

    def test_positions_are_refused_in_other_shapes(self):
        case = layer_case(93)
        p = case.meta.positions
        with served_kernels():
            views, _ = fresh(case)
            with self.assertRaisesRegex(ValueError, r"\[3, N\]"):
                self.inputs(case, views, p.expand(2, -1).contiguous())
            with self.assertRaisesRegex(ValueError, "first-member"):
                self.index_keys(case, views, p[:-1].contiguous())


if __name__ == "__main__":
    unittest.main()
