"""A served Qwen3.8 prefill chunk is one forward: the prefix cache's block boundaries ride it as marks (carry P3).

base/composed.ComposedModel.prefill cut a step at every boundary the runner named, and adapter.ServedComposition
dropped the marks its net already takes (net.Step.marks -> caches.mark_gdn, mark_ple), so a 32,256-token chunk ran the
target 42 times and the MTP head 42 times. The composition now says which boundaries its forward takes on the way
(`takes_mark`); the step is cut only where it declines, and the piece after a cut starts on the block grid, so every
later boundary of the step is taken.

On the CPU with a recording net: what is held here is the steps the net is handed, the snapshots the store still copies
itself, and that the reference composition (no `takes_mark`) is cut exactly as before.
"""
import importlib.util
import unittest
from types import SimpleNamespace

torch = None
if importlib.util.find_spec("torch") is not None:
    import torch

BLOCK, HIDDEN, VOCAB = 768, 8, 32


class RecordingNet:
    """net.Qwen38Net's surface the adapter reads: every step it is handed, and the real `takes_mark` rule."""

    def __init__(self, ple_conv: int = 4, ngram_size: int = 3):
        from engine.profiles.qwen38.net import Qwen38Net
        self.F = SimpleNamespace(ple_conv=ple_conv, ngram_size=ngram_size, ple_layers=(1,))
        self.layers = [0, 1]
        self.takes_mark = lambda offset: Qwen38Net.takes_mark(self, offset)
        self.steps, self.heads = [], []

    def forward(self, step, caches, *, streams=False):
        self.steps.append(step)
        n = step.ids.numel()
        return torch.zeros(n, HIDDEN), torch.zeros(n, 4 * HIDDEN)

    def head(self, out):
        return torch.zeros(out.shape[0], VOCAB)

    def mtp_forward(self, step, given, caches, *, last_hidden_only=True):
        self.heads.append((step, given.shape[0]))
        return torch.zeros(1, HIDDEN), torch.zeros(1, 4 * HIDDEN)

    def head_tokens(self, hidden):
        return torch.ones(hidden.shape[0], dtype=torch.int64)

    def draft_tokens(self, hidden):
        return self.head_tokens(hidden)


class RecordingCaches:
    def __init__(self, spec_k: int = 1):
        self.F = SimpleNamespace(spec_k=spec_k)
        self.pool, self.device = SimpleNamespace(tokens={}), "cpu"
        self.slots = SimpleNamespace(num_slots=4)
        self.checkpoints = []

    def reset_slot(self, slot):
        pass

    def prepare(self, step):
        pass

    def checkpoint(self, slot, position, snap):
        self.checkpoints.append((position, snap))


@unittest.skipUnless(torch is not None, "requires torch")
class ServedPrefillMarksTests(unittest.TestCase):
    def model(self, *, drafter: bool = False):
        from engine.base.composed import ComposedModel
        from engine.profiles.qwen38.adapter import ServedComposition, ServedMTP, ServedStore
        net, caches = RecordingNet(), RecordingCaches()
        store = ServedStore(caches)
        mtp = ServedMTP(net, caches, store, 1) if drafter else None
        model = ComposedModel(ServedComposition(net, caches), store, vocab=VOCAB, eos_ids=[0], max_new=4,
                              temperature=0.0, drafter=mtp)
        return model, net, caches

    def prefill(self, model, prompt: int, start: int, tokens: int, marks: dict):
        if 0 not in model.tokens:
            model.add(0, [1] * prompt)
            model.open(0, 1)
        return model.prefill(0, start, tokens, None, 1, marks=marks)

    def test_a_chunk_of_whole_blocks_is_one_forward_carrying_its_marks(self):
        model, net, caches = self.model()
        tokens = 42 * BLOCK
        marks = {b * BLOCK: 100 + b for b in range(1, 42)}
        self.prefill(model, tokens + 5, 0, tokens, marks)
        self.assertEqual(len(net.steps), 1)
        (step,) = net.steps
        self.assertEqual(step.ids.numel(), tokens)
        self.assertEqual(step.marks, tuple((b * BLOCK, 100 + b) for b in range(1, 42)))
        self.assertEqual(caches.checkpoints, [])              # every snapshot was the forward's own
        self.assertEqual(model.context(0), tokens)

    def test_marks_are_counted_from_the_segments_start(self):
        model, net, _ = self.model()
        self.prefill(model, 4 * BLOCK, 0, BLOCK, {})
        self.prefill(model, 4 * BLOCK, BLOCK, 3 * BLOCK, {2 * BLOCK: 7, 3 * BLOCK: 8})
        self.assertEqual(net.steps[1].segments[0].ctx, BLOCK)
        self.assertEqual(net.steps[1].marks, ((BLOCK, 7), (2 * BLOCK, 8)))

    def test_a_step_that_starts_off_the_grid_is_cut_once_and_keeps_every_boundary(self):
        model, net, caches = self.model()
        start = BLOCK + 100                                     # a continued conversation resumes wherever it stopped
        self.prefill(model, 5 * BLOCK, 0, start, {BLOCK: 1})
        del net.steps[:], caches.checkpoints[:]
        self.prefill(model, 5 * BLOCK, start, 5 * BLOCK - start, {2 * BLOCK: 2, 3 * BLOCK: 3, 4 * BLOCK: 4})
        self.assertEqual([s.ids.numel() for s in net.steps], [BLOCK - 100, 3 * BLOCK])
        self.assertEqual(net.steps[0].marks, ())
        self.assertEqual(net.steps[1].marks, ((BLOCK, 3), (2 * BLOCK, 4)))
        self.assertEqual(caches.checkpoints, [(2 * BLOCK, 2)])  # the cut's boundary: the rings hold it, the store copies it

    def test_a_mark_at_the_steps_end_is_the_stores(self):
        model, net, caches = self.model()
        self.prefill(model, 3 * BLOCK, 0, 2 * BLOCK, {BLOCK: 1, 2 * BLOCK: 2})
        self.assertEqual(len(net.steps), 1)
        self.assertEqual(net.steps[0].marks, ((BLOCK, 1),))
        self.assertEqual(caches.checkpoints, [(2 * BLOCK, 2)])

    def test_the_mtp_head_observes_the_chunk_once(self):
        model, net, _ = self.model(drafter=True)
        tokens = 4 * BLOCK
        self.prefill(model, tokens + 5, 0, tokens, {b * BLOCK: b for b in range(1, 4)})
        self.assertEqual(len(net.steps), 1)
        self.assertEqual([(step.ids.numel(), rows) for step, rows in net.heads], [(tokens, tokens)])

    def test_the_net_declines_what_its_kernels_cannot_take(self):
        net = RecordingNet()
        self.assertTrue(net.takes_mark(BLOCK))
        self.assertTrue(net.takes_mark(64))
        self.assertFalse(net.takes_mark(BLOCK - 100))           # off the GDN chunk kernel's 64-token grid
        self.assertFalse(net.takes_mark(0))
        wide = RecordingNet(ple_conv=40, ngram_size=3)          # PLE's conv taps (117) would reach before the segment
        self.assertFalse(wide.takes_mark(64))
        self.assertTrue(wide.takes_mark(128))
        wide.layers = [0]                                       # no PLE layer among the net's: only the GDN grid
        self.assertTrue(wide.takes_mark(64))


class GdnCaches:
    """One slot's GDN rings and a recorder for what the marks wrote (tests/test_engine_kda_marks's, for Qwen3.8)."""

    def __init__(self, F):
        self.conv = torch.zeros(F.qkv_local, F.conv - 1 + F.spec_k, dtype=torch.bfloat16)
        self.rec = torch.zeros(F.spec_k + 1, F.v_heads_local, F.k_dim, F.v_dim)
        self.marks = {}

    def gdn(self, layer, slot):
        return self.conv, self.rec

    def mark_gdn(self, layer, snap, state, taps):
        self.marks[snap] = (state.clone(), taps.clone())


@unittest.skipUnless(torch is not None, "requires torch")
class GdnMarkTests(unittest.TestCase):
    """What rides the uncut forward is what the cut one left in the rings: the reference lanes on the CPU, one GDN layer
    of the tiny facts, one rank's arithmetic."""

    def net(self):
        from engine.base.comm import Comm, LocalTP
        from engine.profiles.qwen38 import lanes, specs
        from engine.profiles.qwen38.net import Qwen38Net
        from tests.test_engine_qwen38_preshard import facts_of, nvidia_config
        F = facts_of(nvidia_config())
        net = Qwen38Net(F, LocalTP(4).rank(0), lanes.reference(), layers=[0], mtp=False)
        net.comm = Comm(1, 0, None)                             # identity collectives
        g = torch.Generator().manual_seed(0)
        net.p = {}
        for s in specs.layer_specs(F, 0, routed=False):
            if ".gdn." in s.name:
                t = torch.ones(s.shape) if s.name.endswith("norm") else torch.randn(s.shape, generator=g) * 0.2
                net.p[s.name] = t.to(s.dtype)
        return F, net

    def test_a_mark_holds_the_state_and_taps_the_cut_forward_left_in_the_rings(self):
        from engine.profiles.qwen38.net import Step
        F, net = self.net()
        N, marks = 200, ((64, 0), (128, 1))
        x = (torch.randn(N, F.hidden, generator=torch.Generator().manual_seed(1)) * 0.5).to(torch.bfloat16)
        ids = torch.zeros(N, dtype=torch.int64)
        plain, uncut = GdnCaches(F), GdnCaches(F)
        out_plain = net._gdn(0, x, Step.prefill(ids, 0, 1, 1), plain).float()
        out_uncut = net._gdn(0, x, Step.prefill(ids, 0, 1, 1, marks=marks), uncut).float()
        torch.testing.assert_close(out_uncut, out_plain, atol=2e-2, rtol=2e-2)
        last = (N - 1) % (F.spec_k + 1)
        torch.testing.assert_close(uncut.rec[last], plain.rec[last], atol=2e-2, rtol=2e-2)
        self.assertEqual(sorted(uncut.marks), [0, 1])
        qkv = torch.nn.functional.linear(x, net.p["L0.gdn.in_proj"])[:, :F.qkv_local]
        pieces, at = GdnCaches(F), 0                            # the cut prefill: a forward a piece, the rings between
        for position, snap in marks:
            net._gdn(0, x[at:position], Step.prefill(ids[at:position], at, 1, 1), pieces)
            at = position
            state, taps = uncut.marks[snap]
            torch.testing.assert_close(state, pieces.rec[(position - 1) % (F.spec_k + 1)], atol=2e-2, rtol=2e-2)
            torch.testing.assert_close(taps, qkv[position - (F.conv - 1):position], atol=0, rtol=0)

    def test_a_mark_the_net_would_decline_is_refused_by_the_layer(self):
        from engine.profiles.qwen38.net import Step
        F, net = self.net()
        self.assertFalse(net.takes_mark(16))
        x = torch.randn(80, F.hidden).to(torch.bfloat16)
        with self.assertRaises(ValueError):
            net._gdn(0, x, Step.prefill(torch.zeros(80, dtype=torch.int64), 0, 1, 1, marks=((16, 0),)), GdnCaches(F))


@unittest.skipUnless(torch is not None, "requires torch")
class ReferencePrefillMarksTests(unittest.TestCase):
    """A composition that does not declare `takes_mark` (base/composition's reference) is cut at every mark, and is
    never handed the keyword."""

    def test_the_reference_composition_is_still_cut_at_every_mark(self):
        from engine.base.composed import ComposedModel

        class Composition:
            def __init__(self):
                self.lengths = []

            def forward(self, step, store, *, logits="last", hidden=False, given=None):
                self.lengths.append(step.ids.numel())
                store.contexts[0] += step.ids.numel()
                return torch.zeros(1, VOCAB)

        class Store:
            device, ring = "cpu", 0

            def __init__(self):
                self.contexts, self.checkpoints = {0: 0}, []

            def checkpoint(self, seq, position, snap):
                self.checkpoints.append((position, snap))

        composition, store = Composition(), Store()
        model = ComposedModel(composition, store, vocab=VOCAB, eos_ids=[0], max_new=4, temperature=0.0)
        model.add(0, [1] * 30)
        model.prefill(0, 0, 24, None, 1, marks={8: 1, 16: 2})
        self.assertEqual(composition.lengths, [8, 8, 8])
        self.assertEqual(store.checkpoints, [(8, 1), (16, 2)])


if __name__ == "__main__":
    unittest.main()
