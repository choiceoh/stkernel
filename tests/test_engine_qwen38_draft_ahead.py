"""A greedy verify step's draft step follows it on the device (fleet --draft-ahead, on by default).

The synchronous step reads the target's picks, commits them, and only then launches the MTP head's draft step for the
next one -- whose drafts it reads back before the verify step after it. The GPU idles through both reads. For rows whose
pick is the argmax (greedy and plain) nothing in the draft step's inputs needs the host:

    decode_graphs.greedy_verdict    the kept drafts, counted on the device from the picks and the ids the graph was fed
    DraftGraphs.run_after           the head's observation (the kept positions' next ids and streams, padded with the
                                    last) and its chain, gathered from the verify graph's own outputs
    ServedMTP.chain / adopt         the drafts copied into a pinned row set behind an event; a row takes them once the
                                    host's own count of kept positions agrees, and `propose` reads them
    ServedModel._verify_ahead       all of it launched before the host reads the picks

Held here: the verdict; run_after feeds the draft graph what `run` feeds it for the same observation; and the served
model decodes the same tokens, keeps the same drafts, records the same ledger and tap rows and hands the head the same
inputs with draft-ahead as without -- one row and two, rows that finish inside a step, a sampled row (which keeps the
synchronous step), and narrow verify widths under a draft threshold. The world is the real adapter and graph classes
over a small net whose target and head are arithmetic, on the CPU with the graphs run eagerly, and on a CUDA device
with the graphs captured when there is one (an sm_120 desktop card here, not a GB10).
"""
import importlib.util
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

torch = None
if importlib.util.find_spec("torch") is not None:
    import torch

ROOT = Path(__file__).resolve().parents[1]
V, PAD = 64, 6                                  # the vocabulary, and the head's padded columns past it


class FakeNet:
    """The target and its MTP head over tokens and positions. The target's next token after x at position p is x + 1,
    or x + 2 where p is a multiple of five; the head always guesses x + 1 -- so a chain is kept up to the next multiple
    of five, and a verify step keeps anywhere from one position to all of them. `record`: each head call's (ids,
    positions, given rows), which reads the device -- the CPU's eager graphs only."""

    def __init__(self, k: int, *, record: bool = True):
        self.F = SimpleNamespace(block=16, spec_k=k, vocab=V, max_position=1000, ngram_size=3, hc=1, hidden=2)
        self.comm = SimpleNamespace(graph_capture_safe=True)
        self.lanes = SimpleNamespace(graph_resources=None)
        self.ple_stage = None
        self.record, self.heads = record, []

    def takes_mark(self, offset):
        return False

    @staticmethod
    def positions(step):
        dev = step.ids.device
        if getattr(step, "captured", False):
            return (step.contexts[:, None] + torch.arange(step.tokens, device=dev)).reshape(-1)
        return torch.cat([torch.arange(s.ctx, s.ctx + s.length, device=dev) for s in step.segments])

    def forward(self, step, caches, *, last_hidden_only=False, streams=False):
        hidden = torch.stack([step.ids.float(), self.positions(step).float()], dim=1)
        return (hidden, (hidden + 0.5).to(torch.bfloat16)) if streams else hidden

    def head(self, hidden):
        ids, pos = hidden[:, 0].long(), hidden[:, 1].long()
        nxt = (ids + 1 + (pos % 5 == 0).long()) % V
        logits = torch.zeros(hidden.shape[0], V + PAD, device=hidden.device)
        return logits.scatter_(1, nxt[:, None], 1.0)

    def mtp_forward(self, step, given, caches, *, last_hidden_only=True, rows=None):
        pos = self.positions(step)
        if self.record:
            self.heads.append((step.ids.tolist(), pos.tolist(), given.float().tolist()))
        hidden = torch.stack([step.ids.float(), pos.float()], dim=1)
        streams = (hidden * 2).to(given.dtype)
        if rows is None and last_hidden_only and not getattr(step, "captured", False):
            rows = torch.tensor([s.start + s.length - 1 for s in step.segments], device=hidden.device)
        if rows is not None:
            return hidden.index_select(0, rows), streams.index_select(0, rows)
        return hidden, streams

    def draft_tokens(self, h, *, probability=False):
        picks = (h[:, 0].long() + 1) % V
        if not probability:
            return picks
        return picks, ((picks % 4) + 1).to(torch.float32) / 5          # 0.2 .. 0.8 by the pick


class EagerGraphs:
    """base/graphs.DecodeGraphs on the CPU: each shape's inputs made once; `run` fills them and runs the step."""

    def __init__(self, step_fn, make_inputs, shapes, **kwargs):
        self.step_fn, self.shapes = step_fn, list(shapes)
        self.inputs = {shape: make_inputs(*shape) for shape in self.shapes}

    def run(self, shape, fill):
        fill(self.inputs[shape])
        return self.step_fn(self.inputs[shape])

    def close(self):
        pass


def build(*, ahead: bool, k: int = 3, rows: int = 2, device: str = "cpu", eos=(), max_new: int = 200,
          threshold=None, ledger=None, narrow_rows: int = 0, tap=None):
    """The served model over FakeNet and caches that hold only what the adapter reads, its graphs captured."""
    from engine.profiles.qwen38 import decode_graphs
    from engine.profiles.qwen38.adapter import build_model, capture
    dev = torch.device(device)
    net = FakeNet(k, record=dev.type == "cpu")
    caches = SimpleNamespace(F=net.F, device=dev, pool=SimpleNamespace(tokens=[0] * rows),
                             slots=SimpleNamespace(owner=[-1] * (rows + 1), num_slots=rows + 1),
                             block_table=torch.zeros(rows, 64, dtype=torch.int32, device=dev),
                             prepare=lambda step: None, reset=lambda: None, reset_slot=lambda slot: None)
    model, _ = build_model(net, caches, net.F, eos_ids=list(eos), max_new=max_new, temperature=0.0, top_p=1.0, seed=0,
                           draft_threshold=threshold, draft_ledger=ledger, draft_ahead=ahead)
    model.drafter.inputs_tap = tap
    with ExitStack() as stack:
        if dev.type == "cpu":
            stack.enter_context(mock.patch.object(decode_graphs, "DecodeGraphs", EagerGraphs))
            empty = torch.empty
            unpinned = lambda *a, pin_memory=False, **kw: empty(*a, **kw)     # the CPU has no pinned allocator
            stack.enter_context(mock.patch.object(torch, "empty", unpinned))
        capture(model, rows, narrow_rows=narrow_rows)
    return model, net, caches


def serve(model, caches, prompts, *, steps: int, temps=None):
    """The runner's part: open each row, prefill it, then decode every live row together, reserving each step's
    horizon first; a finished row is collected, forgotten and released."""
    slots, out = {}, {}
    for seq, prompt in enumerate(prompts):
        slot = slots[seq] = seq + 1
        caches.slots.owner[slot] = seq
        model.add(seq, list(prompt), temperature=(temps or {}).get(seq, 0.0))
        model.open(seq, slot)
        caches.pool.tokens[seq] = len(prompt)
        model.prefill(seq, 0, len(prompt), None, slot)
    live = list(slots)
    for _ in range(steps):
        if not live:
            break
        for seq in live:
            caches.pool.tokens[seq] = max(caches.pool.tokens[seq], model.horizon(seq))
        done = model.decode(live, [None] * len(live), [slots[s] for s in live])
        for seq, finished in zip(list(live), done):
            if finished:
                out[seq] = list(model.tokens[seq])
                live.remove(seq)
                model.forget(seq)
                model.close(seq)
                caches.slots.owner[slots[seq]] = -1
    for seq in live:                                                    # and what the next step would verify
        caches.pool.tokens[seq] = max(caches.pool.tokens[seq], model.horizon(seq))
    for seq, drafts in zip(live, model.drafter.propose(live) if live else []):
        out[seq] = (list(model.tokens[seq]), drafts)
    return out


def stats(model):
    return (model.steps, model.drafts_total, model.drafted_total, model.accepted_total)


@unittest.skipUnless(torch is not None, "requires torch")
class VerdictTests(unittest.TestCase):
    def test_the_leading_drafts_equal_to_the_pick_before_them(self):
        from engine.profiles.qwen38.decode_graphs import greedy_verdict
        picks = torch.tensor([[5, 6, 7, 8], [5, 9, 7, 8], [4, 6, 7, 8], [5, 6, 7, 8], [5, 6, 6, 6]])
        drafts = torch.tensor([[5, 6, 7], [5, 6, 7], [5, 6, 7], [5, 6, 6], [5, 6, 7]])
        lengths = torch.tensor([3, 3, 3, 2, 3])
        # all kept; the second rejected; the first rejected; two proposed, both kept (the pad equals the pick: not
        # read); the last draft differs
        self.assertEqual(greedy_verdict(picks, drafts, lengths).tolist(), [3, 1, 0, 2, 2])

    def test_no_drafts_keeps_none(self):
        from engine.profiles.qwen38.decode_graphs import greedy_verdict
        self.assertEqual(greedy_verdict(torch.tensor([[3], [4]]), torch.zeros(2, 0, dtype=torch.int64),
                                        torch.tensor([0, 0])).tolist(), [0, 0])
        self.assertEqual(greedy_verdict(torch.tensor([[3, 4]]), torch.tensor([[3]]), torch.tensor([0])).tolist(), [0])


class Snapshots:
    """DecodeGraphs' surface whose run keeps a copy of the shape's inputs after the fill and answers fixed picks."""

    def __init__(self, step_fn, make_inputs, shapes, **kwargs):
        self.inputs = {shape: make_inputs(*shape) for shape in shapes}
        self.seen = []

    def run(self, shape, fill):
        fill(self.inputs[shape])
        step, given, last, counts, sampler = self.inputs[shape]
        self.seen.append((shape, step.ids.tolist(), step.contexts.tolist(), step.seqs.tolist(), step.slots.tolist(),
                          given.float().tolist(), last.tolist(), counts.tolist(),
                          None if sampler is None else [s.tolist() for s in sampler]))
        n = shape[0]
        picks = torch.arange(n * 3).view(n, 3)
        if sampler is None:
            return picks
        # the sampled chain's answer (#1266): picks, their probabilities, the candidates and the distributions over them
        return picks, picks.float() / 8, torch.zeros(n, 3, 2, dtype=torch.int64), torch.zeros(n, 3, 2)

    def close(self):
        pass


@unittest.skipUnless(torch is not None, "requires torch")
class RunAfterTests(unittest.TestCase):
    def draft_graphs(self, **kwargs):
        from engine.profiles.qwen38 import decode_graphs
        net = SimpleNamespace(F=SimpleNamespace(block=16, spec_k=3, hc=1, hidden=2),
                              comm=SimpleNamespace(graph_capture_safe=True), lanes=SimpleNamespace(graph_resources=None))
        caches = SimpleNamespace(pool=SimpleNamespace(tokens=[64, 64]), slots=SimpleNamespace(owner=[-1, -1, -1]),
                                 block_table=torch.zeros(2, 8, dtype=torch.int32), device=torch.device("cpu"),
                                 reset=lambda: None, prepare=lambda step: None)
        empty = torch.empty
        unpinned = lambda *a, pin_memory=False, **kw: empty(*a, **kw)
        with mock.patch.object(decode_graphs, "DecodeGraphs", Snapshots), mock.patch.object(torch, "empty", unpinned):
            return decode_graphs.DraftGraphs(net, caches, 2, 4, k=3, ceiling=100, **kwargs)

    def test_the_draft_graph_is_fed_what_the_host_would_feed_it(self):
        g = self.draft_graphs()
        picks = torch.tensor([[11, 12, 13, 14], [21, 22, 23, 24]])
        streams = torch.arange(16, dtype=torch.float32).view(8, 2).to(torch.bfloat16)
        kept = torch.tensor([4, 2])
        rows = [(0, 1, 10), (1, 2, 20)]
        g.run_after(rows, picks, streams, kept, width=4)
        g.run([(0, 1, 10, [11, 12, 13, 14], streams[0:4]), (1, 2, 20, [21, 22], streams[4:6])])
        ahead, host = g.graphs.seen
        self.assertEqual(ahead[1:], host[1:])
        self.assertEqual(ahead[1], [11, 12, 13, 14, 21, 22, 22, 22])       # the kept ids, the last repeated
        self.assertEqual(ahead[6:8], ([3, 5], [4, 2]))                      # each row's last kept row, its count
        self.assertEqual(ahead[0], (2, 4, g.buckets[0]))

    def test_a_narrow_verify_width_reads_its_own_rows(self):
        g = self.draft_graphs()
        picks = torch.tensor([[11, 12], [21, 22]])
        streams = torch.arange(8, dtype=torch.float32).view(4, 2).to(torch.bfloat16)
        g.run_after([(0, 1, 10), (1, 2, 20)], picks, streams, torch.tensor([2, 1]), width=2)
        g.run([(0, 1, 10, [11, 12], streams[0:2]), (1, 2, 20, [21], streams[2:3])])
        ahead, host = g.graphs.seen
        self.assertEqual(ahead[1:], host[1:])

    def test_a_sampled_chain_behind_a_greedy_step_draws_the_argmax(self):
        """The sampled draft graph (#1266, `candidates`) behind a greedy verify step: run_after hands the chain
        temperature 0 whatever a sampled step's `run` left in its inputs, and answers (picks, probabilities) as
        ServedMTP.chain reads them."""
        g = self.draft_graphs(candidates=2)
        streams = torch.arange(16, dtype=torch.float32).view(8, 2).to(torch.bfloat16)
        drawn = (0.5, 5, 0.75, [0.25, 0.5, 0.75])
        g.run([(0, 1, 10, [11, 12, 13, 14], streams[0:4]), (1, 2, 20, [21, 22], streams[4:6])], [drawn, drawn])
        out = g.run_after([(0, 1, 10), (1, 2, 20)], torch.tensor([[11, 12, 13, 14], [21, 22, 23, 24]]), streams,
                          torch.tensor([4, 2]), width=4)
        self.assertEqual(len(out), 2)
        host, ahead = g.graphs.seen
        self.assertEqual(host[8], [[0.5, 0.5], [5, 5], [0.75, 0.75], [[0.25, 0.5, 0.75]] * 2])
        self.assertEqual(ahead[8], [[0.0, 0.0], [0, 0], [1.0, 1.0], [[0.0, 0.0, 0.0]] * 2])
        self.assertEqual(ahead[1:8], host[1:8])

    def test_every_row_holds_the_widest_observation(self):
        g = self.draft_graphs()
        g.caches.pool.tokens = [15, 64]                                      # 10 + extent(4) = 16 > 15
        with self.assertRaisesRegex(ValueError, "passes its reservation"):
            g.run_after([(0, 1, 10), (1, 2, 20)], torch.zeros(2, 4, dtype=torch.int64),
                        torch.zeros(8, 2, dtype=torch.bfloat16), torch.tensor([1, 1]), width=4)


class EquivalenceMixin:
    """The same requests served with draft-ahead and without: equal tokens, draft counts, ledgers and tap rows, and
    on the CPU the same head inputs."""

    device = "cpu"

    def pair(self, prompts, *, steps=12, heads=True, temps=None, **kwargs):
        runs = {}
        for ahead in (False, True):
            ledger, tapped = [], []
            tap = lambda seq, ctx, ids, hidden, decoded: tapped.append(
                (seq, ctx, list(ids), hidden.float().tolist(), decoded))
            model, net, caches = build(ahead=ahead, device=self.device, ledger=ledger.append, tap=tap, **kwargs)
            out = serve(model, caches, prompts, steps=steps, temps=temps)
            runs[ahead] = (out, stats(model), ledger, tapped, net.heads, model)
        (sync, ahead) = runs[False], runs[True]
        self.assertEqual(ahead[0], sync[0], "the same tokens")
        self.assertEqual(ahead[1], sync[1], "the same steps, drafts and kept drafts")
        self.assertEqual(ahead[2], sync[2], "the same ledger")
        self.assertEqual(ahead[3], sync[3], "the same tap rows")
        if heads and self.device == "cpu":
            self.assertEqual(ahead[4], sync[4], "the head observed the same ids, positions and streams")
        self.assertEqual(sync[5].ahead_steps, 0)
        self.assertEqual(ahead[5].ahead_misses, 0)
        return sync, ahead

    def test_one_row(self):
        sync, ahead = self.pair([[3, 9, 17, 4, 30]])
        self.assertEqual(ahead[5].ahead_steps, 12, "every step: the prompt's drafts come from the eager head")
        self.assertGreater(sync[1][3], 0)
        self.assertLess(sync[1][3], sync[1][2], "some drafts kept, some not")

    def test_two_rows(self):
        _, ahead = self.pair([[3, 9, 17, 4, 30], [8, 1, 2, 60, 5, 44, 7]])
        self.assertEqual(ahead[5].ahead_steps, 12)

    def test_rows_that_finish_inside_a_step(self):
        # 40 ends a row wherever it is committed, and the limits cut the rows at different steps
        sync, ahead = self.pair([[3, 9, 17, 4, 30], [8, 1, 2, 60, 5, 44, 7]], steps=30, heads=False, eos=(40,),
                                max_new=25)
        self.assertEqual(sorted(sync[0]), [0, 1])
        self.assertEqual(sync[0][0][-1], 40, "the first row ends at its end token")
        self.assertEqual(len(sync[0][1]), 7 + 25, "the second at its limit")

    def test_a_sampled_row_keeps_the_synchronous_step(self):
        sync, ahead = self.pair([[3, 9, 17, 4, 30], [8, 1, 2, 60, 5, 44, 7]], steps=8, temps={1: 0.7})
        self.assertEqual(ahead[5].ahead_steps, 0)

    def test_narrow_widths_under_a_threshold(self):
        # the head's probability is 0.2..0.8 by its pick: a cut at 0.5 proposes 0..3 drafts, the verify step as wide
        sync, ahead = self.pair([[3, 9, 17, 4, 30], [8, 1, 2, 60, 5, 44, 7]], steps=16, threshold=0.5, narrow_rows=2)
        self.assertEqual(ahead[5].ahead_steps, 16)
        self.assertLess(sync[1][2], 16 * 2 * 3, "some drafts were cut")

    def test_k1(self):
        _, ahead = self.pair([[3, 9, 17, 4, 30]], k=1)
        self.assertEqual(ahead[5].ahead_steps, 12)


@unittest.skipUnless(torch is not None, "requires torch")
class CPUEquivalenceTests(EquivalenceMixin, unittest.TestCase):
    pass


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "requires a CUDA device")
class CUDAEquivalenceTests(EquivalenceMixin, unittest.TestCase):
    device = "cuda"


@unittest.skipUnless(torch is not None, "requires torch")
class HorizonTests(unittest.TestCase):
    def test_a_row_reserves_the_widest_draft_step_behind_its_verify_step(self):
        for ahead, k, want in ((False, 3, 4), (True, 3, 6), (False, 1, 2), (True, 1, 2)):
            with self.subTest(ahead=ahead, k=k):
                model, _, caches = build(ahead=ahead, k=k, rows=1)
                caches.slots.owner[1] = 0
                model.add(0, [1, 2, 3], temperature=0.0)
                model.open(0, 1)
                self.assertEqual(model.horizon(0), want)

    def test_a_row_short_of_the_reach_keeps_the_synchronous_step(self):
        model, _, caches = build(ahead=True, rows=1)
        serve(model, caches, [[3, 9, 17, 4, 30]], steps=2)
        self.assertEqual(model.ahead_steps, 2)
        seq, before = 0, model.ahead_steps
        caches.pool.tokens[seq] = model.context(seq) + model.k + 1          # a reservation of the verify step only
        model.decode([seq], [None], [1])
        self.assertEqual(model.ahead_steps, before)


class FlagTests(unittest.TestCase):
    def test_the_flag_reaches_the_build_and_the_launcher(self):
        fleet = (ROOT / "engine/profiles/qwen38/fleet.py").read_text(encoding="utf-8")
        self.assertIn('ap.add_argument("--draft-ahead", action=argparse.BooleanOptionalAction, default=True,', fleet)
        self.assertIn("draft_ahead=a.draft_ahead)", fleet)
        self.assertIn("draft_ahead=draft_ahead)", fleet)
        launcher = (ROOT / "launchers/start-st-qwen38.sh").read_text(encoding="utf-8")
        self.assertIn('0) ADAPT_ARG="$ADAPT_ARG --no-draft-ahead" ;;', launcher)
        exec_line = next(l for l in launcher.splitlines() if "-m engine.profiles.qwen38.fleet " in l)
        self.assertLess(exec_line.index("$ADAPT_ARG"), exec_line.index("--port"))
        defaults = (ROOT / "engine/SERVING_DEFAULTS.md").read_text(encoding="utf-8")
        self.assertIn("`ST_DRAFT_AHEAD=1`", defaults)


if __name__ == "__main__":
    unittest.main()
