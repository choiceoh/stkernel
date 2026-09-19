"""The MTP head's drafts end where the head doubts them, and the verify step is as wide as what was proposed
(LibraSpec's rule, arXiv 2608.08721; fleet --draft-threshold P, --narrow-rows N, --draft-ledger; off by default).

A K=3 step verifies K+1 positions a row whatever the head thought of its picks; the chain's third pick is kept less
than half the time (the 2026-09-19 window: 51.6% a slot). With a threshold:

    modules/vocab.argmax_probability   each pick's softmax probability over the whole vocabulary -- the MAX packet
                                       carries the logit, each rank sums its columns, one all-gather of the sums --
                                       the same bits on every rank, so each rank's host cuts alike
    decode_graphs.draft_chain /        the draft graphs return (picks, probabilities)
      DraftGraphs(probability=True)
    adapter.ServedMTP.propose          a row's drafts end before its first pick under the threshold
    decode_graphs.TargetGraphs         steps of up to `narrow_rows` rows replay a graph as wide as their longest row:
      (narrow_rows)                    every width 1..K+1 captured for those row counts
    adapter.ServedMTP.record           the ledger: every pick, its probability, proposed, kept, committed

Held here on the CPU: the probability against softmax over the gathered row on four LocalTP ranks, bit-equal on every
rank; the chain's probabilities; the cut and the ledger; the widths, the admission and the replay's padding at a
narrow width; the flags.
"""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tests.test_engine_qwen38_draft_chain import FakeNet, fake_caches, torch

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(torch is not None, "requires torch")
class ProbabilityTests(unittest.TestCase):
    def test_four_ranks_agree_on_the_softmax_of_the_whole_row(self):
        from engine.base.comm import LocalTP
        from engine.modules.vocab import argmax_probability
        gen = torch.Generator().manual_seed(7)
        rows, vocab = 5, 64
        logits = torch.randn(rows, vocab, generator=gen) * 3
        logits[2] = 0.0                                                  # a flat row: 1/vocab
        logits[3, 41] = 40.0                                             # a sure pick
        width = vocab // 4

        def rank(comm):
            local = logits[:, comm.rank * width:(comm.rank + 1) * width].to(torch.bfloat16)
            return argmax_probability(local, comm, comm.rank * width)

        results = LocalTP(4, timeout_s=20).run(rank)
        whole = logits.to(torch.bfloat16).float()
        want = torch.softmax(whole, dim=-1)
        for ids, probs in results:
            self.assertTrue(torch.equal(ids, whole.argmax(-1)))
            self.assertTrue(torch.allclose(probs, want.max(-1).values, rtol=1e-5, atol=1e-7))
            self.assertTrue(torch.equal(probs, results[0][1]), "every rank holds the same bits")
        self.assertAlmostEqual(float(results[0][1][2]), 1 / vocab, places=6)
        self.assertGreater(float(results[0][1][3]), 0.999)

    def test_argmax_is_unchanged(self):
        from engine.modules.vocab import argmax, argmax_probability
        comm = SimpleNamespace(all_reduce_max=lambda t: t, all_gather=lambda t, dim=-1: t)
        logits = torch.tensor([[1.0, 5.0, 5.0, -2.0], [0.0, 0.0, 0.0, 0.0]])
        self.assertEqual(argmax(logits.clone(), comm, 0).tolist(), [1, 0])
        ids, probs = argmax_probability(logits.clone(), comm, 0)
        self.assertEqual(ids.tolist(), [1, 0])
        self.assertAlmostEqual(float(probs[1]), 0.25, places=6)


class ProbNet(FakeNet):
    """FakeNet whose picks carry a probability: the token's last digit over ten."""

    def draft_tokens(self, h, *, probability=False):
        picks = super().draft_tokens(h)
        if not probability:
            return picks
        return picks, (picks % 10).to(torch.float32) / 10


@unittest.skipUnless(torch is not None, "requires torch")
class ChainTests(unittest.TestCase):
    def test_the_chain_returns_each_picks_probability(self):
        from engine.profiles.qwen38.decode_graphs import draft_chain
        from engine.profiles.qwen38.net import DeviceStep
        step = DeviceStep(torch.tensor([11, 12, 13, 14]), torch.tensor([10]), torch.tensor([1]), torch.tensor([0]), 4, 4)
        picks, probs = draft_chain(ProbNet(), None, step, torch.zeros(4, 2), torch.tensor([3]), torch.tensor([4]), 3,
                                   probability=True)
        self.assertEqual(picks.tolist(), [[141, 1411, 14111]])
        self.assertEqual(probs.dtype, torch.float32)
        self.assertTrue(torch.allclose(probs, torch.tensor([[0.1, 0.1, 0.1]])))
        plain = draft_chain(ProbNet(), None, step, torch.zeros(4, 2), torch.tensor([3]), torch.tensor([4]), 3)
        self.assertTrue(torch.equal(plain, picks))


class ProbGraphs:
    """DraftGraphs' surface with probabilities: fixed picks and the probabilities a test gives."""

    probability = True

    def __init__(self, k, probs):
        from engine.profiles.qwen38.decode_graphs import DraftGraphs
        self.k, self.tokens, self.probs, self.ran = k, k + 1, probs, []
        self.extent = lambda observed: DraftGraphs.extent(self, observed)

    def run(self, rows):
        self.ran.append(len(rows))
        return ([[100 + 10 * i + j for j in range(self.k)] for i in range(len(rows))],
                [list(self.probs) for _ in rows])


@unittest.skipUnless(torch is not None, "requires torch")
class CutTests(unittest.TestCase):
    def drafter(self, probs, threshold, ledger=None):
        from engine.profiles.qwen38.adapter import ServedMTP
        mtp = ServedMTP(FakeNet(), fake_caches([0] * 7 + [18], (-1, 7, -1)), SimpleNamespace(slot_of={7: 1}), 3,
                        threshold=threshold, ledger=ledger)
        mtp.graphs = ProbGraphs(3, probs)
        return mtp

    def proposed(self, probs, threshold):
        mtp = self.drafter(probs, threshold)
        mtp.observe(7, 10, [1, 2, 3, 4], torch.zeros(4, 2))
        return mtp.propose([7])[0]

    def test_the_drafts_end_before_the_first_the_head_doubts(self):
        self.assertEqual(self.proposed([0.9, 0.8, 0.7], 0.5), [100, 101, 102])
        self.assertEqual(self.proposed([0.9, 0.4, 0.95], 0.5), [100])          # the chain after a doubt goes too
        self.assertEqual(self.proposed([0.3, 0.9, 0.9], 0.5), [])              # nothing: a plain decode step
        self.assertEqual(self.proposed([0.3, 0.2, 0.1], 0.0), [100, 101, 102])  # 0 cuts nothing
        self.assertEqual(self.proposed([0.3, 0.2, 0.1], None), [100, 101, 102])

    def test_an_eager_chain_carries_no_probability_and_is_not_cut(self):
        from engine.profiles.qwen38.adapter import ServedMTP
        mtp = ServedMTP(FakeNet(), fake_caches([0] * 7 + [40], (-1, 7, -1)), SimpleNamespace(slot_of={7: 1}), 3,
                        threshold=0.99)
        mtp.graphs = ProbGraphs(3, [0.0, 0.0, 0.0])
        mtp.observe(7, 0, [1, 2, 3, 4, 5, 6], torch.zeros(6, 2))              # a prompt: the head eagerly
        self.assertEqual(mtp.propose([7]), [[61, 611, 6111]])

    def test_the_ledger_holds_every_pick_and_what_was_kept(self):
        records = []
        mtp = self.drafter([0.9, 0.4, 0.95], 0.5, ledger=records.append)
        mtp.observe(7, 10, [1, 2, 3, 4], torch.zeros(4, 2))
        self.assertEqual(mtp.propose([7]), [[100]])
        mtp.record(7, 14, 1, 1, 2)
        self.assertEqual(records, [{"seq": 7, "ctx": 14, "picks": [100, 101, 102], "probs": [0.9, 0.4, 0.95],
                                    "proposed": 1, "matched": 1, "committed": 2}])
        mtp.record(7, 16, 0, 0, 1)                                              # nothing proposed since: no picks
        self.assertEqual(records[-1]["picks"], [])

    def test_a_step_wider_than_the_narrow_rows_is_not_cut(self):
        """More rows than the narrow graphs serve replay the full width: a cut draft is padded back and only its chance
        of being kept is lost -- no cut there."""
        from engine.profiles.qwen38.adapter import ServedMTP
        mtp = ServedMTP(FakeNet(), fake_caches([0] * 7 + [18] * 3, (-1, 7, 8, 9)),
                        SimpleNamespace(slot_of={7: 1, 8: 2, 9: 3}), 3, threshold=0.5)
        mtp.graphs = ProbGraphs(3, [0.9, 0.4, 0.95])
        mtp.narrow_rows = 2
        for seq, slot in ((7, 1), (8, 2), (9, 3)):
            mtp.observe(seq, 10, [1, 2, 3, 4], torch.zeros(4, 2))
        self.assertEqual(mtp.propose([7, 8, 9]), [[100, 101, 102], [110, 111, 112], [120, 121, 122]])
        for seq in (7, 8):
            mtp.observe(seq, 14, [5], torch.zeros(1, 2))
        self.assertEqual(mtp.propose([7, 8]), [[100], [110]])                  # two rows: the narrow graphs, cut

    def test_a_threshold_is_a_probability(self):
        for bad in (-0.1, 1.0, 2.0):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.drafter([0.5] * 3, bad)

    def test_the_graphs_report_probabilities_where_a_threshold_or_a_ledger_reads_them(self):
        from engine.profiles.qwen38.adapter import ServedMTP
        make = lambda **kw: ServedMTP(FakeNet(), fake_caches([0], [-1]), SimpleNamespace(slot_of={}), 3, **kw)
        self.assertFalse(make().probability)
        self.assertTrue(make(threshold=0.3).probability)
        self.assertTrue(make(ledger=lambda r: None).probability)


class RecordingGraphs:
    """base/graphs.DecodeGraphs' surface TargetGraphs builds on: the shapes, and a run that fills the shape's inputs
    and answers logits and streams as wide as the shape."""

    def __init__(self, forward, make_inputs, shapes, **kwargs):
        self.shapes, self.make_inputs, self.inputs = list(shapes), make_inputs, {}
        for shape in self.shapes:
            self.inputs[shape] = make_inputs(*shape)

    def run(self, shape, fill):
        fill(self.inputs[shape])
        n, t, _ = shape
        return torch.arange(n * t, dtype=torch.float32)[:, None], torch.zeros(n * t, 2)

    def close(self):
        pass


@unittest.skipUnless(torch is not None, "requires torch")
class NarrowWidthTests(unittest.TestCase):
    def target(self, max_seqs=3, narrow_rows=2, reserved=64):
        from engine.profiles.qwen38 import decode_graphs
        F = SimpleNamespace(block=16, spec_k=3)
        net = SimpleNamespace(F=F, comm=SimpleNamespace(graph_capture_safe=True), ple_stage=None,
                              lanes=SimpleNamespace(graph_resources=None))
        caches = fake_caches([reserved] * 8, [-1] * (max_seqs + 1))
        caches.block_table, caches.device, caches.reset = torch.zeros(4, 8, dtype=torch.int32), "cpu", lambda: None
        empty = torch.empty
        unpinned = lambda *a, pin_memory=False, **k: empty(*a, **k)            # the CPU has no pinned allocator
        with mock.patch.object(decode_graphs, "DecodeGraphs", RecordingGraphs), mock.patch.object(torch, "empty", unpinned):
            return decode_graphs.TargetGraphs(net, caches, max_seqs, 4, ceiling=100, narrow_rows=narrow_rows), caches

    def step(self, lengths, ctx=10):
        from engine.profiles.qwen38.net import Segment, Step
        segments, start = [], 0
        for i, n in enumerate(lengths):
            segments.append(Segment(i, i + 1, ctx, start, n))
            start += n
        return Step(torch.arange(100, 100 + start), tuple(segments))

    def test_narrow_rows_capture_every_width_the_rest_one(self):
        g, _ = self.target()
        widths = {(n, t) for n, t, _ in g.graphs.shapes}
        self.assertEqual(widths, {(3, 4), (2, 4), (2, 3), (2, 2), (2, 1), (1, 4), (1, 3), (1, 2), (1, 1)})
        self.assertEqual(g.graphs.shapes[0][:2], (3, 4), "the widest first: it sizes the pool the rest share")
        self.assertEqual([g.width(1, n) for n in (1, 2, 3, 4)], [1, 2, 3, 4])
        self.assertEqual([g.width(3, n) for n in (1, 2, 3, 4)], [4, 4, 4, 4])
        plain, _ = self.target(narrow_rows=0)
        self.assertEqual({t for _, t, _ in plain.graphs.shapes}, {4})

    def test_a_step_replays_its_longest_rows_width_and_pads_the_rest(self):
        g, caches = self.target()
        logits, streams, rows, t = g.run(self.step([2, 1]))
        self.assertEqual(t, 2)
        self.assertEqual(caches.prepared[-1], [(0, 10, 2), (1, 10, 2)])          # each row published its width
        self.assertEqual(rows, [0, 1, 2])                                        # the short row's pad is not read
        inputs = g.graphs.inputs[(2, 2, g.buckets[0])]
        self.assertEqual(inputs.ids.tolist(), [100, 101, 102, 102])              # padded with its last token
        logits, _, rows, t = g.run(self.step([4, 1, 1]))                         # three rows: the full width
        self.assertEqual((t, rows), (4, [0, 1, 2, 3, 4, 8]))

    def test_admission_asks_the_reservation_of_the_width_it_replays(self):
        g, _ = self.target(reserved=12)
        pool = SimpleNamespace(tokens=[12] * 4)
        self.assertTrue(g.admits(self.step([2]), pool))                         # 10 + 2 <= 12
        self.assertFalse(g.admits(self.step([3]), pool))                        # 10 + 3 > 12
        self.assertFalse(g.admits(self.step([5]), pool))                        # wider than the graphs


@unittest.skipUnless(torch is not None, "requires torch")
class FleetTests(unittest.TestCase):
    def test_the_threshold_flag_parses_and_refuses(self):
        from engine.profiles.qwen38.fleet import draft_threshold
        self.assertIsNone(draft_threshold(None))
        self.assertIsNone(draft_threshold("off"))
        self.assertEqual(draft_threshold("0.3"), 0.3)
        self.assertEqual(draft_threshold("0"), 0.0)
        for bad in ("1", "1.5", "-0.1", "x"):
            with self.subTest(bad=bad), self.assertRaises(SystemExit):
                draft_threshold(bad)

    def test_the_ledger_writes_one_line_a_record(self):
        from engine.profiles.qwen38.fleet import DraftLedger
        with tempfile.TemporaryDirectory() as d:
            ledger = DraftLedger(Path(d) / "draft-ledger", every=2)
            ledger({"seq": 1, "matched": 2})
            ledger({"seq": 1, "matched": 0})
            ledger.file.close()
            lines = ledger.path.read_text().splitlines()
            self.assertEqual([json.loads(l)["matched"] for l in lines], [2, 0])
            self.assertTrue(all("t" in json.loads(l) for l in lines))

    def test_the_flags_reach_the_build_and_the_launcher(self):
        fleet = (ROOT / "engine/profiles/qwen38/fleet.py").read_text()
        self.assertIn("draft_threshold=draft_threshold(a.draft_threshold),", fleet)
        self.assertIn("narrow_rows=narrow_rows if draft_threshold else 0", fleet)
        self.assertIn("draft_ledger() if comm.rank == 0 else (lambda record: None)", fleet)
        launcher = (ROOT / "launchers/start-st-qwen38.sh").read_text()
        exec_line = next(l for l in launcher.splitlines() if "-m engine.profiles.qwen38.fleet " in l)
        self.assertLess(exec_line.index("$ADAPT_ARG"), exec_line.index("--port"))
        self.assertIn('ADAPT_ARG="--draft-threshold $ST_DRAFT_THRESHOLD"', launcher)
        self.assertIn('ADAPT_ARG="$ADAPT_ARG --narrow-rows $ST_NARROW_ROWS"', launcher)
        self.assertIn('0) ADAPT_ARG="$ADAPT_ARG --no-draft-ledger" ;;', launcher)

    def test_the_fleet_cuts_and_keeps_the_ledger_by_default(self):
        """The operator's decision of 2026-09-19 ("전부 켜"): drafts cut below 0.1 with narrow widths to two rows, the
        ledger written; `off` and --no-draft-ledger roll them back, and the draft index asks for both."""
        from engine.profiles.qwen38.fleet import DRAFT_THRESHOLD, NARROW_ROWS
        self.assertEqual((DRAFT_THRESHOLD, NARROW_ROWS), (0.1, 2))
        fleet = (ROOT / "engine/profiles/qwen38/fleet.py").read_text()
        self.assertIn('ap.add_argument("--draft-threshold", default=str(DRAFT_THRESHOLD), metavar="P|off",', fleet)
        self.assertIn('ap.add_argument("--narrow-rows", type=int, default=NARROW_ROWS,', fleet)
        self.assertIn('ap.add_argument("--draft-ledger", action=argparse.BooleanOptionalAction, default=True,', fleet)
        self.assertIn("pass --draft-threshold off --no-draft-ledger with it", fleet)
        adapter = (ROOT / "engine/profiles/qwen38/adapter.py").read_text()
        self.assertIn("model.drafter.narrow_rows = model.composition.graphs.narrow_rows", adapter)
        for probe in ("probes/engine_qwen38_step.py", "probes/engine_qwen38_mtp_window.py"):
            self.assertIn("probability=True)", (ROOT / probe).read_text(), probe)


if __name__ == "__main__":
    unittest.main()
