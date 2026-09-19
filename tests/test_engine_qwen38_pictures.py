"""engine/profiles/qwen38/adapter + net: a sequence with pictures through the served composition -- its rotary layout
(net.pictures, net.rope_rows) against vision.rope_positions, the checks before a row takes pictures, which steps run
eagerly at the rows' own mRoPE positions (a picture's rows, the few after one) and which replay the captured graphs at
the rows' deltas, the pictures encoded once at the piece that reaches them, and the MTP head's steps turned alike.

    python3 -m unittest tests.test_engine_qwen38_pictures
"""
import types
import unittest
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import torch

from engine.profiles.qwen38 import vision
from tests.test_engine_qwen38_vision import facts

RATIO = 4


def fake_net():
    from engine.profiles.qwen38.net import Qwen38Net
    net = SimpleNamespace(pictures={}, F=SimpleNamespace(idx_ratio=RATIO), serves_pictures=True)
    net.rope_rows = types.MethodType(Qwen38Net.rope_rows, net)
    return net


def picture_prompt(V, lead=5, grid=(1, 4, 6), tail=7):
    """ids with one picture after `lead` text tokens, `tail` after it; its record (canvas: zeros of its size)."""
    n = V.tokens(grid)
    ids = [11] * (lead - 1) + [V.vision_start] + [V.image_token] * n + [V.vision_end] + [12] * (tail - 1)
    rec = {"kind": "image", "digest": "p", "positions": list(range(lead, lead + n)), "grid": grid,
           "canvas": np.zeros((1, 3, grid[1] * V.patch, grid[2] * V.patch), np.uint8)}
    return ids, rec


class FakeVision:
    def __init__(self, V):
        self.V, self.encoded = V, []

    def encode(self, canvas, grid):
        self.encoded.append(tuple(grid))
        return torch.arange(self.V.tokens(grid), dtype=torch.float32)[:, None].expand(-1, 3).to(torch.bfloat16)


def composition(net=None):
    from engine.profiles.qwen38.adapter import ServedComposition
    comp = ServedComposition(net or fake_net(), SimpleNamespace(pool=SimpleNamespace(tokens={}), prepare=lambda s: None))
    comp.vision = FakeVision(facts())
    return comp


class RopeRowsTests(unittest.TestCase):
    def test_inside_the_prompt_the_layout_past_it_the_delta(self):
        V = facts()
        ids, rec = picture_prompt(V)
        layout, delta = vision.rope_positions(len(ids), [rec], V.merge)
        net = fake_net()
        net.pictures[0] = (layout, delta, rec["positions"][-1])
        rope, first = net.rope_rows(0, 0, len(ids) + 5, "cpu")
        self.assertTrue(np.array_equal(rope[:, :len(ids)].numpy(), layout))
        past = torch.arange(len(ids), len(ids) + 5)
        self.assertTrue(torch.equal(rope[:, len(ids):], (past + delta).expand(3, -1)))
        for p in range(len(ids) + 5):
            q = max(p - (RATIO - 1), 0)
            self.assertTrue(torch.equal(first[:, p], rope[:, q]), p)
        self.assertIsNone(net.rope_rows(1, 0, 4, "cpu"))                     # a text-only sequence: its cache positions


class CompositionTests(unittest.TestCase):
    def test_checks_refuse_what_the_door_could_not_have_made(self):
        comp = composition()
        V = comp.vision.V
        ids, rec = picture_prompt(V)
        comp.check_media(ids, [rec])
        for bad in (dict(rec, kind="video"), dict(rec, grid=(1, 4, 8)), dict(rec, positions=[p + 1 for p in rec["positions"]])):
            with self.subTest(bad=bad["kind"]), self.assertRaises(ValueError):
                comp.check_media(ids, [bad])

    def test_the_layout_is_adopted_and_forgotten(self):
        comp = composition()
        ids, rec = picture_prompt(comp.vision.V)
        comp.bind_media(3, ids, [rec])
        layout, delta, last = comp.net.pictures[3]
        want, want_delta = vision.rope_positions(len(ids), [rec], comp.vision.V.merge)
        self.assertTrue(np.array_equal(layout, want) and delta == want_delta and last == rec["positions"][-1])
        self.assertTrue(comp.near_picture(3, last + RATIO - 1) and not comp.near_picture(3, last + RATIO))
        self.assertFalse(comp.near_picture(4, 0))
        comp.forget_media(3)
        self.assertNotIn(3, comp.net.pictures)

    def test_a_picture_is_encoded_once_at_the_piece_that_reaches_it(self):
        comp = composition()
        V = comp.vision.V
        ids, rec = picture_prompt(V, lead=5, grid=(1, 4, 6), tail=7)       # rows 5..10
        records = [rec]
        first = comp._patches(0, records, 0, 8)                             # rows 5, 6, 7
        self.assertEqual([p.tolist() for p, _ in first], [[5, 6, 7]])
        second = comp._patches(0, records, 8, 16)                           # rows 8, 9, 10 -- the rest
        self.assertEqual([p.tolist() for p, _ in second], [[0, 1, 2]])
        self.assertEqual(second[0][1][:, 0].tolist(), [3.0, 4.0, 5.0])     # the same encoding's later rows
        self.assertEqual(comp.vision.encoded, [(1, 4, 6)])
        self.assertIsNone(rec["canvas"])                                     # its rows are in the caches now
        self.assertEqual(comp._patches(0, records, 16, 20), ())


class PatchedForwardTests(unittest.TestCase):
    def test_picture_rows_reach_the_host_forward_without_changing_its_step_kind(self):
        """Run the real layer loop, including the PLE branch, instead of replacing net.forward with a recorder.
        The image tensors must replace embeddings, never the boolean selecting host versus captured steps."""
        from engine.profiles.qwen38.net import Qwen38Net, Step
        from engine.profiles.qwen38.lanes import reference
        net = object.__new__(Qwen38Net)
        hidden, hc = 3, 2
        net.F = SimpleNamespace(hc=hc, rms_eps=1e-6, ple_layers=(0,), is_qsa=lambda layer: False)
        net.layers = [0]
        net.p = {"close.norm": torch.zeros(hc * hidden), "close.down": torch.zeros(1, hc * hidden)}
        net._hc_projections = {}
        net.lanes = reference()
        net.step_meta = lambda step, caches: None
        net._site = lambda prefix, h, out, inject: (h[:, :hidden], torch.ones(len(h), hc), h)
        net._mix = lambda prefix, h, name, inject: (h.reshape(-1, hc, hidden).mean(1), None)
        net._ple_inject = lambda layer, h, step, meta, caches: torch.zeros_like(h)
        net._gdn = lambda layer, x, step, caches: x

        def moe(prefix, x, *, compact):
            self.assertTrue(compact)
            return x * .25

        net._moe = moe
        observed = []
        net.head_observer = lambda rows, mask: observed.append(rows.clone())
        embeddings = torch.arange(15, dtype=torch.float32).reshape(5, hidden) / 10
        replacements = torch.tensor([[3., 2., 1.], [-1., -2., -3.]])
        at = torch.tensor([1, 3])
        step = Step.prefill(torch.arange(5), 0, 0, 1)
        net.embed = lambda ids: embeddings.clone()
        step = replace(step, patches=((at, replacements),))
        got = net.forward(step, None, last_hidden_only=True, streams=True)
        self.assertEqual(observed[0].shape, (5, hidden))   # calibration gets the whole prompt, not only its last row
        torch.testing.assert_close(got[0], observed[0][-1:], rtol=0, atol=0)
        expected = embeddings.clone()
        expected[at] = replacements
        net.embed = lambda ids: expected.clone()
        step = replace(step, patches=())
        want = net.forward(step, None, last_hidden_only=True, streams=True)
        for actual, reference in zip(got, want):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)


class StepTests(unittest.TestCase):
    def forward(self, comp, step, *, media=None, admits=True):
        """ServedComposition.forward over fakes: which path ran and the net.Step it built."""
        ran = {}

        class Graphs:
            def admits(self, served, pool):
                return admits

            def run(self, served, known=None):
                ran["graph"] = served
                n = served.ids.numel()
                return torch.zeros(n, 5), torch.zeros(n, 2), None, 1

        def net_forward(served, caches, streams=False):
            ran["eager"] = served
            return torch.zeros(served.ids.numel(), 2), torch.zeros(served.ids.numel(), 2)
        comp.graphs = Graphs()
        comp.net.forward = net_forward
        comp.net.head = lambda out: torch.zeros(out.shape[0], 5)
        store = SimpleNamespace(check=lambda s: None, commit=lambda s: None, slot_of={0: 1, 1: 2})
        comp.forward(step, store, media=media)
        return ran

    def base_step(self, chunks):
        from engine.base.composition import Step
        return Step.of([(seq, ctx, torch.tensor(ids, dtype=torch.int64)) for seq, ctx, ids in chunks])

    def test_which_path_a_step_takes(self):
        comp = composition()
        V = comp.vision.V
        ids, rec = picture_prompt(V, lead=5, grid=(1, 4, 6), tail=7)       # 18 tokens, the last picture row 10
        comp.bind_media(0, ids, [rec])
        # the prefill piece holding the picture: eager, its rows patched, [3, N] rotary positions from the layout
        ran = self.forward(comp, self.base_step([(0, 0, ids[:12])]), media=[rec])
        served = ran["eager"]
        self.assertEqual(served.rope.shape, (3, 12))
        self.assertEqual([p.tolist() for p, _ in served.patches], [list(range(5, 11))])
        # a decode row right after the picture: its group-first members reach back into it -- eager
        ran = self.forward(comp, self.base_step([(0, 12, [3])]))
        self.assertIn("eager", ran)
        self.assertEqual(ran["eager"].rope.shape, (3, 1))
        # far past it: the captured graph (at the row's delta), beside a text-only row
        ran = self.forward(comp, self.base_step([(0, 14, [3]), (1, 40, [4])]))
        self.assertIn("graph", ran)
        # a text-only step never builds a layout
        ran = self.forward(comp, self.base_step([(1, 40, [4])]), admits=False)
        self.assertIsNone(ran["eager"].rope)
        self.assertEqual(ran["eager"].patches, ())

    def test_the_mtp_head_turns_at_the_row_s_positions(self):
        from engine.profiles.qwen38.adapter import ServedMTP
        net = fake_net()
        V = facts()
        ids, rec = picture_prompt(V)
        layout, delta = vision.rope_positions(len(ids), [rec], V.merge)
        net.pictures[0] = (layout, delta, rec["positions"][-1])
        seen = []

        def mtp_forward(step, given, caches, last_hidden_only=True):
            seen.append(step)
            return torch.zeros(1, 2), torch.zeros(1, 2)
        net.mtp_forward = mtp_forward
        net.draft_tokens = lambda h: torch.tensor([9])
        head = ServedMTP(net, SimpleNamespace(prepare=lambda s: None), SimpleNamespace(slot_of={0: 1, 1: 2}), 1)
        head._head(0, 3, torch.tensor(ids[4:10]), torch.zeros(6, 2))
        self.assertTrue(np.array_equal(seen[-1].rope.numpy(), layout[:, 3:9]))
        head._head(1, 3, torch.tensor([1, 2]), torch.zeros(2, 2))
        self.assertIsNone(seen[-1].rope)


class FleetTests(unittest.TestCase):
    def test_the_boot_serves_pictures_when_every_rank_has_the_tower(self):
        from pathlib import Path
        fleet = (Path(__file__).resolve().parents[1] / "engine/profiles/qwen38/fleet.py").read_text()
        self.assertIn('ap.add_argument("--vision", choices=("auto", "on", "off"), default="auto"', fleet)
        self.assertIn("vision=a.vision", fleet)
        build = fleet[fleet.index("def build("):fleet.index("def main(")]
        # the ranks agree before any rank loads, and the net knows before any graph is captured
        self.assertLess(build.index("0 < seeing < facts.TP"), build.index('recorder.phase("load")'))
        self.assertLess(build.index("net.serves_pictures = VF is not None"), build.index("capture(model"))
        self.assertLess(build.index('recorder.phase("qualify vision")'), build.index('recorder.phase("capture decode")'))
        self.assertIn("vision=eyes.Door(vision.V) if comm.rank == 0 and vision is not None else None", fleet)


if __name__ == "__main__":
    unittest.main()
