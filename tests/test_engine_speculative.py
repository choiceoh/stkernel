"""engine/base/composed with a drafter: verification by position over any composition, on the CPU, with the small
Qwen3.8-shaped composition of tests/test_engine_composed.py (GDN recurrent and conv state, the n-gram memory's context
and conv, QSA rows) -- the features whose per-sequence values a rejected draft would corrupt.

Two oracles, neither a model: accepting n provisional tokens is feeding those n tokens (the store answers the next step
the same), and a drafter changes how many tokens a step yields, never which (the run with drafts is the run without,
token for token, greedy and sampled, whatever the drafts)."""
import importlib.util
import unittest

torch = None
if importlib.util.find_spec("torch") is not None:
    import torch

from tests.test_engine_composed import TINY, prompt, tiny_composition


def close(case, got, want, tol=1e-5):
    case.assertEqual(tuple(got.shape), tuple(want.shape))
    diff = float((got.float() - want.float()).abs().max())
    case.assertLessEqual(diff, tol, f"max |got - want| = {diff}")


@unittest.skipUnless(torch is not None, "requires torch")
class VerifyStateTests(unittest.TestCase):
    """A verify step keeps the value after each provisional token; accept(n) is having fed n."""

    def setUp(self):
        self.comp, self.cfg = tiny_composition()

    def stores(self):
        from engine.base.composed import store_for
        from engine.base.composition import State
        store, pool, slots, _ = store_for(self.comp, kv_gib=0.02, max_seqs=2, block_tokens=4, ring=5)
        store.open(1, slots.take(1))
        pool.reserve(1, 32)
        return [("reference", State()), ("position", store)]

    def test_accepting_n_answers_like_feeding_n(self):
        from engine.base.composition import State, Step
        ids, provisional, after = prompt(1, 11), prompt(2, 5), prompt(3, 2)
        for n in (1, 2, 5):
            for name, (plain_name, plain), (verify_name, verified) in [
                    (k, a, b) for k, a, b in zip(("reference", "position"), self.stores(), self.stores())]:
                with self.subTest(n=n, store=name), torch.no_grad():
                    for state in (plain, verified):
                        self.comp.forward(Step.of([(1, 0, torch.tensor(ids))]), state)
                    self.comp.forward(Step.of([(1, 11, torch.tensor(provisional[:n]))]), plain)
                    logits = self.comp.forward(Step.of([(1, 11, torch.tensor(provisional), True)]), verified, logits="all")
                    self.assertEqual(verified.contexts[1], 11)                   # nothing kept until accept
                    verified.accept(1, n)
                    self.assertEqual(verified.contexts[1], 11 + n)
                    want = self.comp.forward(Step.of([(1, 11 + n, torch.tensor(after))]), plain, logits="all")
                    got = self.comp.forward(Step.of([(1, 11 + n, torch.tensor(after))]), verified, logits="all")
                    close(self, got, want)
                    # the provisional step's own logits are the plain step's at the kept positions
                    ref = State()
                    self.comp.forward(Step.of([(1, 0, torch.tensor(ids))]), ref)
                    plain_logits = self.comp.forward(Step.of([(1, 11, torch.tensor(provisional))]), ref, logits="all")
                    close(self, logits, plain_logits)

    def test_the_protocol_refuses_what_would_corrupt_a_row(self):
        from engine.base.composed import store_for
        from engine.base.composition import Composition, Layer, Plan, State, Step, put_state
        from engine.modules.residual import PreNorm
        for name, state in self.stores():
            with self.subTest(store=name), torch.no_grad():
                self.comp.forward(Step.of([(1, 0, torch.tensor(prompt(4, 6)))]), state)
                self.comp.forward(Step.of([(1, 6, torch.tensor([5, 6, 7]), True)]), state)
                with self.assertRaisesRegex(ValueError, "waiting for accept"):
                    self.comp.forward(Step.of([(1, 6, torch.tensor([5]))]), state)
                for bad in (0, 4):
                    with self.assertRaisesRegex(ValueError, "accept keeps 1..3"):
                        state.accept(1, bad)
                state.accept(1, 3)
                with self.assertRaisesRegex(ValueError, "no verify step"):
                    state.accept(1, 1)
        # a feature that keeps only the final value cannot be verified
        class LastOnly:
            def __call__(self, layer, x, step, state):
                for s in step.segments:
                    state.put(layer, "counter", s.seq, torch.zeros(1))
                return x
        comp = Composition(Plan((Layer("mix", "mlp"),)), embed=lambda ids: torch.zeros(ids.numel(), 4),
                           residual=PreNorm(eps=1e-6, weights=lambda l, s, n: torch.ones(4)),
                           features={"mix": LastOnly(), "mlp": lambda layer, x, step, state: x}, head=lambda h: h)
        with self.assertRaisesRegex(ValueError, "keeps the value after each"):
            comp.forward(Step.of([(0, 0, torch.tensor([1, 2]), True)]), State())
        with self.assertRaisesRegex(ValueError, "needs the value after each"):
            put_state(State(), 0, "counter", Step.of([(0, 0, torch.tensor([1, 2]), True)]).segments[0], torch.zeros(1))
        # a store without a ring refuses a verify segment before it computes anything
        store, pool, slots, _ = store_for(self.comp, kv_gib=0.02, max_seqs=2, block_tokens=4)
        store.open(1, slots.take(1)); pool.reserve(1, 16)
        with torch.no_grad():
            self.comp.forward(Step.of([(1, 0, torch.tensor([3, 4]))]), store)
            with self.assertRaisesRegex(ValueError, "needs a ring of 2"):
                self.comp.forward(Step.of([(1, 2, torch.tensor([3, 4]), True)]), store)


class Scripted:
    """A drafter that knows the answer: `truth[seq]` is the whole token list the run without drafts produced. It
    proposes the next k tokens of it, the first `right` of them correct and the rest nudged wrong (None: all right),
    and records every observation."""

    def __init__(self, k, model, truth, right=None, vocab=512):
        self.k, self.model, self.truth, self.right, self.vocab = k, model, truth, right, vocab
        self.observed, self.forgotten = {}, []

    def observe(self, seq, ctx, next_ids, hidden):
        self.observed.setdefault(seq, []).append((ctx, list(next_ids), tuple(hidden.shape)))

    def propose(self, seqs):
        out = []
        for seq in seqs:
            at = len(self.model.tokens[seq])
            draft = list(self.truth[seq][at:at + self.k])
            if self.right is not None:
                draft = [t if i < self.right else (t + 1) % self.vocab for i, t in enumerate(draft)]
            out.append(draft)
        return out

    def forget(self, seq):
        self.forgotten.append(seq)


class Random(Scripted):
    def __init__(self, k, model, seed=0):
        super().__init__(k, model, {})
        self.generator = torch.Generator().manual_seed(seed)

    def propose(self, seqs):
        return [torch.randint(1, 512, (self.k,), generator=self.generator).tolist() for _ in seqs]


@unittest.skipUnless(torch is not None, "requires torch")
class SpeculativeRunnerTests(unittest.TestCase):
    """The runner drives a ComposedModel with a drafter: every run equals the run without one."""

    K = 3

    def build(self, *, drafter=None, rows=2, keep_idle=False, prefix=None, max_new=8, temperature=0.0, seed=0):
        from engine.base.composed import ComposedModel, store_for
        from engine.base.record import Ring
        from engine.base.runner import STEP_RECORD, Runner
        from engine.base.scheduler import Contract
        comp, cfg = tiny_composition(0)
        k = self.K if drafter else 0
        store, pool, slots, _ = store_for(comp, kv_gib=0.03, max_seqs=rows, block_tokens=4, ring=k + 1,
                                          snapshots=prefix.snapshots if prefix else 0)
        model = ComposedModel(comp, store, vocab=cfg["vocab_size"], eos_ids=[cfg["eos_token_id"]], max_new=max_new,
                              temperature=temperature, seed=seed)
        if drafter:
            model.drafter = None
            made = drafter(model)
            model.__init__(comp, store, vocab=cfg["vocab_size"], eos_ids=[cfg["eos_token_id"]], max_new=max_new,
                           temperature=temperature, seed=seed, drafter=made)
        runner = Runner(model, Contract(chunk_align=4, token_budget=16, draft_slots=k, max_wait_s=0.0, max_running=rows),
                        pool, slots, Ring(64, STEP_RECORD.size), keep_idle=keep_idle, prefix=prefix)
        return model, runner

    def run_all(self, runner, limit=400):
        kinds = []
        for _ in range(limit):
            step = runner.step(now=0.0)
            if step is None:
                return kinds
            kinds.append(step.kind)
        raise AssertionError("the runner did not finish")

    def serve(self, requests, **build):
        """requests: [(seq, ids, max_new, temperature, options)] -> (model, runner, kinds)."""
        model, runner = self.build(**build)
        for seq, ids, max_new, temperature, options in requests:
            model.add(seq, ids, max_new=max_new, temperature=temperature, options=options)
            runner.submit(seq, len(ids), ids=ids, now=0.0)
        return model, runner, self.run_all(runner)

    def plain_and_drafted(self, requests, drafter_of, **build):
        plain, _, plain_kinds = self.serve(requests, **build)
        truth = {seq: list(plain.tokens[seq]) for seq, *_ in requests}
        drafted, runner, kinds = self.serve(requests, drafter=lambda model: drafter_of(model, truth), **build)
        for seq, *_ in requests:
            self.assertEqual(drafted.generated(seq), plain.generated(seq), f"row {seq}")
        return plain, drafted, plain_kinds, kinds

    def two_rows(self, temperature=0.0, options=None):
        return [(0, prompt(1, 13), 9, temperature, options or {}), (1, prompt(2, 6), 7, temperature, options or {})]

    def test_perfect_drafts_are_all_accepted_and_take_fewer_steps(self):
        plain, drafted, plain_kinds, kinds = self.plain_and_drafted(
            self.two_rows(), lambda model, truth: Scripted(self.K, model, truth))
        self.assertLess(kinds.count("decode"), plain_kinds.count("decode"))
        self.assertEqual(drafted.accepted_total, drafted.drafted_total)
        self.assertGreater(drafted.drafted_total, 0)

    def test_wrong_and_partly_right_drafts_change_nothing(self):
        for right in (0, 1, 2):
            with self.subTest(right=right):
                _, drafted, _, _ = self.plain_and_drafted(
                    self.two_rows(), lambda model, truth, right=right: Scripted(self.K, model, truth, right=right))
                self.assertLessEqual(drafted.accepted_total, right * drafted.drafts_total)
        _, drafted, _, _ = self.plain_and_drafted(self.two_rows(), lambda model, truth: Random(self.K, model))
        self.assertLess(drafted.accepted_total, drafted.drafted_total)

    def test_sampled_rows_draw_the_same_tokens(self):
        rows = self.two_rows(temperature=0.9, options={"seed": 7})
        plain, drafted, _, _ = self.plain_and_drafted(rows, lambda model, truth: Scripted(self.K, model, truth), seed=3)
        self.assertEqual(drafted.accepted_total, drafted.drafted_total)
        greedy, _, _ = self.serve(self.two_rows())
        self.assertNotEqual(plain.generated(0), greedy.generated(0))           # the rows really were sampled
        self.plain_and_drafted(rows, lambda model, truth: Scripted(self.K, model, truth, right=1), seed=3)

    def test_an_end_inside_an_accepted_run_stops_there(self):
        plain, _, _ = self.serve([(0, prompt(4, 6), 9, 0.0, {})], rows=1)
        stop = plain.generated(0)[4]
        rows = [(0, prompt(4, 6), 9, 0.0, {"stop_token_ids": [stop]})]
        stopped, drafted, _, _ = self.plain_and_drafted(rows, lambda model, truth: Scripted(self.K, model, truth), rows=1)
        self.assertEqual(stopped.generated(0), plain.generated(0)[:5])
        held = [(0, prompt(4, 6), 9, 0.0, {"stop_token_ids": [stop]})]
        model, runner = self.build(rows=1)
        model.add(0, held[0][1], max_new=9, temperature=0.0, min_new=7, options=held[0][4])
        runner.submit(0, 6, now=0.0)
        self.run_all(runner)
        truth = {0: list(model.tokens[0])}
        spec, runner = self.build(rows=1, drafter=lambda m: Scripted(self.K, m, truth))
        spec.add(0, held[0][1], max_new=9, temperature=0.0, min_new=7, options=held[0][4])
        runner.submit(0, 6, now=0.0)
        self.run_all(runner)
        self.assertEqual(spec.generated(0), model.generated(0))

    def test_the_drafter_observes_every_kept_position_once_in_order(self):
        _, drafted, _, _ = self.plain_and_drafted(self.two_rows(), lambda model, truth: Scripted(self.K, model, truth, right=1))
        drafter = drafted.drafter
        for seq in (0, 1):
            tokens, seen = drafted.tokens[seq], 0
            for ctx, next_ids, shape in drafter.observed[seq]:
                self.assertEqual(ctx, seen)
                self.assertEqual(next_ids, tokens[ctx + 1:ctx + 1 + len(next_ids)])
                self.assertEqual(shape[0], len(next_ids))
                seen += len(next_ids)
            self.assertEqual(seen, drafted.context(seq) if seq in drafted.store.contexts else len(tokens) - 1)

    def test_a_second_turn_and_a_cached_prefix_carry_on(self):
        from engine.base.prefix import PrefixCache
        a, more = prompt(5, 9), prompt(6, 4)
        runs = []
        for drafter in (None, "scripted"):
            model, runner = self.build(rows=1, keep_idle=True,
                                       drafter=(lambda m: Scripted(self.K, m, truth)) if drafter else None)
            model.add(0, a, max_new=4, temperature=0.0)
            runner.submit(0, len(a), now=0.0)
            self.run_all(runner)
            pending = model.extend(0, more, max_new=4, temperature=0.0)
            runner.extend(0, pending, now=0.0)
            self.run_all(runner)
            runs.append(list(model.tokens[0]))
            truth = {0: runs[0]}
        self.assertEqual(runs[1], runs[0])
        b = prompt(7, 13)
        c = b[:10] + prompt(8, 3)                                              # shares two whole blocks with b
        base, base_runner = self.build(prefix=PrefixCache(4, 8, 4))
        base.add(0, b, max_new=3, temperature=0.0); base_runner.submit(0, len(b), ids=b, now=0.0); self.run_all(base_runner)
        base.add(1, c, max_new=5, temperature=0.0); base_runner.submit(1, len(c), ids=c, now=0.0); self.run_all(base_runner)
        truth = {0: list(base.tokens[0]), 1: list(base.tokens[1])}
        model, runner = self.build(prefix=PrefixCache(4, 8, 4), drafter=lambda m: Scripted(self.K, m, truth))
        model.add(0, b, max_new=3, temperature=0.0); runner.submit(0, len(b), ids=b, now=0.0); self.run_all(runner)
        model.add(1, c, max_new=5, temperature=0.0); runner.submit(1, len(c), ids=c, now=0.0)
        self.assertEqual(runner.reused_tokens, 8)
        self.run_all(runner)
        self.assertEqual(model.generated(1), base.generated(1))

    def test_a_boundary_inside_an_accepted_step_is_checkpointed_from_the_ring(self):
        """A decode step that crosses a block boundary: the snapshot the runner takes there restores a row that answers
        like the row that computed the prefix plainly."""
        from engine.base.composed import store_for
        from engine.base.composition import State, Step
        comp, _ = tiny_composition(0)
        store, pool, slots, _ = store_for(comp, kv_gib=0.02, max_seqs=3, block_tokens=4, ring=4, snapshots=1)
        ids, provisional = prompt(9, 6), prompt(10, 4)
        with torch.no_grad():
            store.open(1, slots.take(1)); pool.reserve(1, 16)
            comp.forward(Step.of([(1, 0, torch.tensor(ids))]), store)
            comp.forward(Step.of([(1, 6, torch.tensor(provisional), True)]), store)
            store.accept(1, 3)                                             # stands at 9; the boundary at 8 was crossed
            store.checkpoint(1, 8, snap=0)
            with self.assertRaisesRegex(ValueError, "stands at 9"):
                store.checkpoint(1, 6, snap=0)                             # not inside the accepted step
            store.open(2, slots.take(2))
            pool.adopt(2, list(pool.row(1)[:2]), 8)
            pool.reserve(2, 4)
            store.restore(2, 8, snap=0)
            got = comp.forward(Step.of([(2, 8, torch.tensor([provisional[2]]))]), store, logits="all")
            ref = State()
            comp.forward(Step.of([(0, 0, torch.tensor(ids + provisional[:2]))]), ref)
            want = comp.forward(Step.of([(0, 8, torch.tensor([provisional[2]]))]), ref, logits="all")
        close(self, got, want)

    def test_a_drafter_needs_a_ring_for_its_verify_step(self):
        from engine.base.composed import ComposedModel, store_for
        comp, cfg = tiny_composition(0)
        store, *_ = store_for(comp, kv_gib=0.02, max_seqs=1, block_tokens=4, ring=2)
        with self.assertRaisesRegex(ValueError, "ring keeps 2"):
            ComposedModel(comp, store, vocab=512, eos_ids=[0], drafter=Scripted(3, None, {}))
        model = ComposedModel(comp, store, vocab=512, eos_ids=[0], drafter=Scripted(1, None, {}))
        self.assertEqual((model.k, model.horizon(0) if 0 in store.contexts else None), (1, None))


if __name__ == "__main__":
    unittest.main()
