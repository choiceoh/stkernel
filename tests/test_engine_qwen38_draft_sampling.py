"""Sampled MTP drafts kept by block verification (fleet --draft-candidates C, default 20; 0 = the argmax drafts).

The served verify kept a draft only where the target's pick landed on the head's argmax. At a sampled row that is the
exact match of a draw: a draft the target would make half the time is kept half the time, however well the head knows
the target's distribution. With candidates:

    net.draft_sample                 the head's C largest logits over the whole vocabulary (modules/vocab.topk), the
                                     row's own sampler over them (base/sampler.rows), one keyed DRAFT uniform
    decode_graphs.draft_chain        the chain continues from the drawn token; the graphs also return the candidates
      (sampled) / DraftGraphs         and the distribution each draft was drawn from
      (candidates)
    adapter.ServedMTP                a sampled row's drafts and their distribution -- never cut by the threshold: a cut
                                     decided on the drawn token biases what the verification divides by
    adapter ServedModel._verify      block verification (base/sampler.block_verify_batch) and rank 0's verdict
      (_block_verify)                 (modules/draft_agreement.agree_verdict); a greedy or rich row keeps the exact match

Held here on the CPU: four LocalTP ranks draw the same candidates, distribution and pick, and the distribution is the
row's sampler over the candidates; the chain continues from the drawn token; the drafter's bookkeeping; the verify
step's first token is distributed as the target's (a chi-square over keyed draws); the flags.
"""
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from tests.test_engine_qwen38_draft_chain import FakeNet, fake_caches, torch

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(torch is not None, "requires torch")
class DraftSampleTests(unittest.TestCase):
    def test_four_ranks_draw_the_same_candidates_distribution_and_pick(self):
        from engine.base.comm import LocalTP
        from engine.base.sampler import distribution
        from engine.profiles.qwen38.net import Qwen38Net
        gen = torch.Generator().manual_seed(3)
        rows, vocab, C = 4, 64, 8
        logits = (torch.randn(rows, vocab, generator=gen) * 3).to(torch.bfloat16).float()
        width = vocab // 4
        temps = torch.tensor([1.0, 0.7, 0.0, 1.0])
        top_ks = torch.tensor([5, 0, 0, 8], dtype=torch.int32)
        top_ps = torch.tensor([1.0, 0.9, 1.0, 0.95])
        uniforms = torch.tensor([0.1, 0.5, 0.9, 0.99])

        def rank(comm):
            local = logits[:, comm.rank * width:(comm.rank + 1) * width].to(torch.bfloat16)
            net = SimpleNamespace(draft_index=None, comm=comm, rank=comm.rank, vp=width, draft_logits=lambda h: local)
            return Qwen38Net.draft_sample(net, torch.zeros(rows, 2), temps, top_ks, top_ps, uniforms, candidates=C)

        results = LocalTP(4, timeout_s=20).run(rank)
        for got in results[1:]:
            for a, b in zip(results[0], got):
                self.assertTrue(torch.equal(a, b), "every rank holds the same bits")
        picks, p, cand, dist = results[0]
        self.assertEqual((tuple(cand.shape), tuple(dist.shape)), ((rows, C), (rows, C)))
        self.assertTrue(torch.equal(cand.sort(-1).values, logits.topk(C, dim=-1).indices.sort(-1).values))
        for i in range(rows):
            # the row's sampler over its candidates' logits: one definition of temperature, top-k and top-p
            want = distribution(logits[i, cand[i]], float(temps[i]), int(top_ks[i]) or None, float(top_ps[i]))
            self.assertTrue(torch.allclose(dist[i], want, atol=1e-6), i)
            self.assertAlmostEqual(float(dist[i].sum()), 1.0, places=5)
            at = int((cand[i] == picks[i]).nonzero()[0])
            self.assertGreater(float(dist[i, at]), 0.0)
            if temps[i] > 0:                              # a greedy row reports the head's whole-row probability (below)
                self.assertEqual(float(p[i]), float(dist[i, at]))
        # top-k no wider than the candidates: the candidates hold the whole nucleus, so it is the target's own law
        for i in (0, 3):
            full = distribution(logits[i], float(temps[i]), int(top_ks[i]), float(top_ps[i]))
            self.assertTrue(torch.allclose(full[cand[i]], dist[i], atol=1e-6))
            self.assertAlmostEqual(float(full[cand[i]].sum()), 1.0, places=5)
        self.assertEqual(int(picks[2]), int(logits[2].argmax()))            # temperature 0: the argmax, all its mass
        self.assertEqual(float(dist[2].max()), 1.0)
        # ... and the head's probability of it over the whole vocabulary, not that distribution's 1: what the threshold
        # cuts a greedy row on and the ledger records -- draft_tokens(probability=True)'s, to the bit, on every rank
        self.assertAlmostEqual(float(p[2]), float(torch.softmax(logits[2], -1).max()), places=6)
        self.assertLess(float(p[2]), 1.0)

    def test_a_greedy_rows_probability_is_argmax_probabilitys(self):
        from engine.base.comm import LocalTP
        from engine.modules.vocab import argmax_probability
        from engine.profiles.qwen38.net import Qwen38Net
        gen = torch.Generator().manual_seed(5)
        rows, vocab, C = 3, 64, 8
        logits = (torch.randn(rows, vocab, generator=gen) * 2).to(torch.bfloat16).float()
        logits[1] = 0.0                                                   # a flat row: 1/vocab
        width = vocab // 4
        zeros = torch.zeros(rows)

        def rank(comm):
            local = logits[:, comm.rank * width:(comm.rank + 1) * width].to(torch.bfloat16)
            net = SimpleNamespace(draft_index=None, comm=comm, rank=comm.rank, vp=width, draft_logits=lambda h: local)
            picks, p, _, _ = Qwen38Net.draft_sample(net, torch.zeros(rows, 2), zeros, zeros.to(torch.int32),
                                                    torch.ones(rows), zeros, candidates=C)
            return picks, p, argmax_probability(local, comm, comm.rank * width)

        for picks, p, (want_ids, want) in LocalTP(4, timeout_s=20).run(rank):
            self.assertTrue(torch.equal(p, want), "the same bits as the argmax drafts' probability")
            self.assertTrue(torch.equal(picks[[0, 2]], want_ids[[0, 2]]))
        self.assertAlmostEqual(float(p[1]), 1 / vocab, places=6)

    def test_a_draft_index_is_refused(self):
        from engine.profiles.qwen38.net import Qwen38Net
        net = SimpleNamespace(draft_index=object())
        with self.assertRaises(ValueError):
            Qwen38Net.draft_sample(net, None, None, None, None, None, candidates=4)


class SampleNet(FakeNet):
    """FakeNet whose draw is its argmax moved by the uniform's first decimal, with two candidates."""

    def draft_sample(self, h, temperature, top_k, top_p, uniform, *, candidates):
        picks = self.head_tokens(h) + (uniform * 10).to(torch.int64)
        cand = torch.stack([picks, picks + 1000], dim=1)
        dist = torch.tensor([[0.75, 0.25]]).expand(h.shape[0], 2).clone()
        return picks, torch.full((h.shape[0],), 0.75), cand, dist


@unittest.skipUnless(torch is not None, "requires torch")
class ChainTests(unittest.TestCase):
    def test_the_sampled_chain_continues_from_the_drawn_token(self):
        from engine.profiles.qwen38.decode_graphs import draft_chain
        from engine.profiles.qwen38.net import DeviceStep
        step = DeviceStep(torch.tensor([11, 12, 13, 14]), torch.tensor([10]), torch.tensor([1]), torch.tensor([0]), 4, 4)
        sampled = (torch.ones(1), torch.zeros(1, dtype=torch.int32), torch.ones(1), torch.tensor([[0.1, 0.2, 0.3]]), 2)
        picks, probs, cand, dist = draft_chain(SampleNet(), None, step, torch.zeros(4, 2), torch.tensor([3]),
                                               torch.tensor([4]), 3, sampled=sampled)
        # 14 -> 141 + 1; its draw 142 -> 1421 + 2; 1423 -> 14231 + 3: each depth reads the token drawn before it
        self.assertEqual(picks.tolist(), [[142, 1423, 14234]])
        self.assertEqual((tuple(cand.shape), tuple(dist.shape)), ((1, 3, 2), (1, 3, 2)))
        self.assertEqual(cand[0, :, 0].tolist(), [142, 1423, 14234])
        self.assertTrue(torch.allclose(probs, torch.full((1, 3), 0.75)))

    def test_a_greedy_chain_reports_what_the_argmax_chain_does(self):
        """The served net's draft_sample and draft_tokens under the chain, four ranks: at temperature 0 the sampled
        chain (the served default, --draft-candidates 20) picks and reports what the argmax chain (--draft-candidates
        0) does -- so the draft threshold cuts a greedy row alike under both, and the ledger records the same."""
        from engine.base.comm import LocalTP
        from engine.profiles.qwen38.decode_graphs import draft_chain
        from engine.profiles.qwen38.net import DeviceStep, Qwen38Net
        vocab = 64
        table = (torch.randn(vocab, vocab, generator=torch.Generator().manual_seed(9)) * 2).to(torch.bfloat16)

        class HeadNet(FakeNet):
            draft_index = draft_tap = None
            draft_tokens, draft_sample = Qwen38Net.draft_tokens, Qwen38Net.draft_sample

            def __init__(self, comm):
                super().__init__()
                self.comm, self.rank, self.vp = comm, comm.rank, vocab // 4

            def draft_logits(self, h):                       # the head's rows: a fixed row a token, this rank's columns
                return table[h[:, 0].long() % vocab, self.rank * self.vp:(self.rank + 1) * self.vp]

        step = DeviceStep(torch.tensor([11, 12, 13, 14, 21, 22, 22, 22]), torch.tensor([10, 20]), torch.tensor([1, 2]),
                          torch.tensor([0, 1]), 4, 4)
        last, counts = torch.tensor([3, 5]), torch.tensor([4, 2])

        def rank(comm):
            net = HeadNet(comm)
            argmax = draft_chain(net, None, step, torch.zeros(8, 2), last, counts, 3, probability=True)
            greedy = (torch.zeros(2), torch.zeros(2, dtype=torch.int32), torch.ones(2), torch.zeros(2, 3), 20)
            sampled = draft_chain(net, None, step, torch.zeros(8, 2), last, counts, 3, sampled=greedy)
            return argmax, sampled[:2]

        for (picks, probs), (drawn, reported) in LocalTP(4, timeout_s=20).run(rank):
            self.assertTrue(torch.equal(drawn, picks))
            self.assertTrue(torch.equal(reported, probs))
            self.assertTrue(bool((probs < 1).all()))


class SampleGraphs:
    """DraftGraphs' surface with candidates: fixed picks the head doubts (0.05 each) and two candidates a draft."""

    probability = True
    candidates = 2

    def __init__(self, k):
        from engine.profiles.qwen38.decode_graphs import DraftGraphs
        self.k, self.tokens, self.ran = k, k + 1, []
        self.extent = lambda observed: DraftGraphs.extent(self, observed)

    def run(self, rows, sampling=None):
        self.ran.append(list(sampling))
        n = len(rows)
        picks = [[100 + 10 * i + j for j in range(self.k)] for i in range(n)]
        cand = torch.tensor(picks).unsqueeze(-1).repeat(1, 1, 2)
        cand[..., 1] += 1
        dist = torch.tensor([0.6, 0.4]).expand(n, self.k, 2).clone()
        return picks, [[0.05] * self.k for _ in rows], cand, dist


@unittest.skipUnless(torch is not None, "requires torch")
class DrafterTests(unittest.TestCase):
    def drafter(self, ledger=None):
        from engine.profiles.qwen38.adapter import ServedMTP
        mtp = ServedMTP(FakeNet(), fake_caches([0] * 7 + [18], (-1, 7, -1)), SimpleNamespace(slot_of={7: 1}), 3,
                        threshold=0.5, ledger=ledger, candidates=2)
        mtp.graphs = SampleGraphs(3)
        return mtp

    def test_a_sampled_row_keeps_all_its_drafts_and_what_they_were_drawn_from(self):
        mtp = self.drafter()
        mtp.observe(7, 10, [1, 2, 3, 4], torch.zeros(4, 2))
        setting = (1.0, 20, 0.95, [0.1, 0.2, 0.3])
        self.assertEqual(mtp.propose([7], sampling={7: setting}), [[100, 101, 102]])     # 0.05 < 0.5, not cut
        self.assertEqual(mtp.graphs.ran, [[setting]])
        cand, dist = mtp.distribution(7)
        self.assertEqual(cand[:, 0].tolist(), [100, 101, 102])
        self.assertEqual(tuple(dist.shape), (3, 2))
        mtp.forget(7)
        self.assertIsNone(mtp.distribution(7))

    def test_a_greedy_row_is_the_argmax_the_threshold_cuts(self):
        mtp = self.drafter()
        mtp.observe(7, 10, [1, 2, 3, 4], torch.zeros(4, 2))
        self.assertEqual(mtp.propose([7]), [[]])                        # the head doubts the first: nothing proposed
        self.assertEqual(mtp.graphs.ran, [[None]])
        self.assertIsNone(mtp.distribution(7))

    def test_the_ledger_marks_a_sampled_row(self):
        records = []
        mtp = self.drafter(ledger=records.append)
        mtp.observe(7, 10, [1, 2, 3, 4], torch.zeros(4, 2))
        mtp.propose([7], sampling={7: (1.0, 0, 1.0, [0.5] * 3)})
        mtp.record(7, 10, 3, 2, 3)
        self.assertEqual((records[0]["sampled"], records[0]["proposed"], records[0]["matched"]), (True, 3, 2))

    def test_the_candidates_are_a_count(self):
        from engine.profiles.qwen38.adapter import ServedMTP
        with self.assertRaises(ValueError):
            ServedMTP(FakeNet(), None, None, 3, candidates=-1)


def _inverse_cdf(weights, u):
    total, walk = sum(weights), 0.0
    for i, w in enumerate(weights):
        walk += w
        if u * total < walk:
            return i
    return max(i for i, w in enumerate(weights) if w > 0)


@unittest.skipUnless(torch is not None, "requires torch")
class VerifyTests(unittest.TestCase):
    """ServedModel._block_verify on a row whose drafts are drawn as the draft graphs draw them (the row's DRAFT
    uniforms through the drafter's distribution) against fixed target logits: whatever the drafts, the first token
    the step commits is distributed as the target's own pick there -- block verification is lossless."""

    V, K = 6, 2

    def model(self, dist, temperature=1.0):
        from engine.profiles.qwen38.adapter import _served_model_class
        Model = _served_model_class()
        model = Model.__new__(Model)
        model.limits = {5: (100, temperature)}
        model.options = {5: {}}
        model.top_p, model.vocab, model.seed, model.seeds, model.nonces = 1.0, self.V, 11, {}, {5: 0}
        model.generated_count = lambda seq: 3
        model.drafter = SimpleNamespace(distribution=lambda seq: dist)
        model.composition = SimpleNamespace(net=SimpleNamespace(comm=SimpleNamespace(world_size=1)))
        return model

    def test_the_first_committed_token_is_the_targets_draw(self):
        from engine.base import draws
        target = torch.tensor([[2.0, 1.0, 0.5, 0.0, -1.0, 0.3],
                               [0.0, 1.5, 0.2, 0.9, 0.1, 0.0],
                               [1.0, 0.0, 0.0, 2.0, 0.5, 0.2]])
        # the head's distribution at each depth over three candidates -- deliberately not the target's
        cand = torch.tensor([[1, 0, 3], [3, 1, 4]])
        dist = torch.tensor([[0.5, 0.3, 0.2], [0.6, 0.3, 0.1]])
        model = self.model((cand, dist))
        segment = [SimpleNamespace(start=0, length=self.K + 1)]
        counts, n = [0] * self.V, 6000
        for nonce in range(n):
            model.nonces[5] = nonce
            key = draws.row_key(model.seed, nonce, 3)
            us = draws.uniforms(key, draws.DRAFT, self.K)
            drafts = [int(cand[j, _inverse_cdf(dist[j].tolist(), us[j])]) for j in range(self.K)]
            tokens = model._block_verify([5], [drafts], segment, target)[0]
            self.assertTrue(1 <= len(tokens) <= self.K + 1)
            self.assertEqual(tokens[:len(tokens) - 1], drafts[:len(tokens) - 1])    # a kept prefix of the drafts
            counts[tokens[0]] += 1
        want = torch.softmax(target[0], dim=0).tolist()
        chi2 = sum((c - n * w) ** 2 / (n * w) for c, w in zip(counts, want))
        self.assertLess(chi2, 25.7, (counts, [round(n * w) for w in want]))   # df 5: p = 1e-4

    def test_a_greedy_or_rich_row_is_not_sampled(self):
        model = self.model(None, temperature=0.0)
        model._rich = lambda seq: False
        model.k = 3
        self.assertEqual(model._draft_sampling([5]), {})
        model.limits[5] = (100, 1.0)
        model._rich = lambda seq: True
        self.assertEqual(model._draft_sampling([5]), {})
        model._rich = lambda seq: False
        model.options[5] = {"top_k": 20, "top_p": 0.95}
        (temperature, top_k, top_p, uniforms), = model._draft_sampling([5]).values()
        self.assertEqual((temperature, top_k, top_p, len(uniforms)), (1.0, 20, 0.95, 3))

    def test_the_verdict_is_rank_zeros(self):
        import inspect
        from engine.profiles.qwen38 import adapter
        source = inspect.getsource(adapter._served_model_class)
        self.assertIn("accepted, tokens = agree_verdict(self.composition.net.comm, accepted, tokens)", source)


class FleetTests(unittest.TestCase):
    def test_sampled_drafts_are_on_by_default_and_reach_the_build_and_the_launcher(self):
        from engine.profiles.qwen38.fleet import DRAFT_CANDIDATES
        self.assertEqual(DRAFT_CANDIDATES, 20)
        fleet = (ROOT / "engine/profiles/qwen38/fleet.py").read_text()
        self.assertIn('ap.add_argument("--draft-candidates", type=int, default=DRAFT_CANDIDATES, metavar="C",', fleet)
        self.assertIn("draft_candidates=a.draft_candidates)", fleet)
        self.assertIn("--no-draft-ledger --draft-candidates 0 with it", fleet)
        adapter = (ROOT / "engine/profiles/qwen38/adapter.py").read_text()
        self.assertIn("candidates=draft_candidates)", adapter)
        launcher = (ROOT / "launchers/start-st-qwen38.sh").read_text()
        self.assertIn('ADAPT_ARG="$ADAPT_ARG --draft-candidates $ST_DRAFT_CANDIDATES"', launcher)


if __name__ == "__main__":
    unittest.main()
