"""engine/base/composed: the door's sampling options on any composition -- penalties, logit_bias, min_tokens,
logprobs, a reasoning budget and structured output -- through the same base/sampler and base/grammar functions the
GLM adapter uses. Oracles: a hand-run reference loop applying base/sampler.process_logits itself; the fake xgrammar of
tests/test_engine_grammar.py (the real buffers and walk); and runs with drafts equal to runs without, logprobs too.
Every case is chosen so the option changes the tokens: a check that the options were ignored would fail it."""
import importlib.util
import json
import unittest

torch = None
if importlib.util.find_spec("torch") is not None:
    import torch

from tests.test_engine_composed import TINY, prompt, tiny_composition

VOCAB, END = TINY["vocab_size"], TINY["eos_token_id"]


def reference_rich(comp, ids, count, options, *, min_new=0):
    """The greedy continuation with the options applied by hand: at every position the reference State's logits
    through process_logits over the history so far (min_tokens' forbidden end, the reasoning budget's forced end),
    then argmax. Returns (tokens, processed rows)."""
    from engine.base.composition import State, Step
    from engine.base.sampler import History, process_logits
    state, tokens, rows = State(), list(ids), []
    history = History(VOCAB, "cpu")
    budget, end = options.get("reasoning_budget"), options.get("reasoning_end")
    with torch.no_grad():
        logits = comp.forward(Step.of([(0, 0, torch.tensor(tokens))]), state)[0]
        for g in range(count):
            seen, counts = history.of(0, tokens, len(ids))
            forbid = torch.tensor([END]) if g < min_new else None
            force = end if budget is not None and g >= budget and end not in tokens[len(ids):] else None
            row = process_logits(logits, options, seen, counts, (), VOCAB, forbid=forbid, force=force)
            rows.append(row[:VOCAB])
            token = int(row[:VOCAB].argmax())
            tokens.append(token)
            if token == END and g + 1 >= min_new:
                break
            logits = comp.forward(Step.of([(0, len(tokens) - 1, torch.tensor([token]))]), state)[0]
    return tokens[len(ids):], rows


def grammar_stub(allow):
    """base/grammar.Grammars over the fake xgrammar: every position allows `allow`, nothing ever terminates; its
    `xgr.accepted` is every token a matcher was advanced by."""
    from tests.test_engine_grammar import fake
    g, _ = fake(allow=allow, vocab=VOCAB)
    g.compile = lambda spec: None
    g.resolve = lambda handle: handle
    g.ready = lambda spec: None
    return g


@unittest.skipUnless(torch is not None, "requires torch")
class RichOptionTests(unittest.TestCase):
    def build(self, *, rows=2, drafter=None, grammars=None, temperature=0.0, max_new=12, seed=0, k=3, keep_idle=False):
        from engine.base.composed import ComposedModel, store_for
        from engine.base.record import Ring
        from engine.base.runner import STEP_RECORD, Runner
        from engine.base.scheduler import Contract
        comp, cfg = tiny_composition(0)
        kk = k if drafter else 0
        store, pool, slots, _ = store_for(comp, kv_gib=0.03, max_seqs=rows, block_tokens=4, ring=kk + 1)
        made = dict(vocab=VOCAB, eos_ids=[END], max_new=max_new, temperature=temperature, seed=seed, grammars=grammars)
        model = ComposedModel(comp, store, **made)
        if drafter:
            model.__init__(comp, store, drafter=drafter(model), **made)
        runner = Runner(model, Contract(chunk_align=4, token_budget=16, draft_slots=kk, max_wait_s=0.0, max_running=rows),
                        pool, slots, Ring(64, STEP_RECORD.size), keep_idle=keep_idle)
        return comp, model, runner

    def serve(self, model, runner, requests):
        """requests: [(seq, ids, max_new, temperature, min_new, options)], run to the end."""
        for seq, ids, max_new, temperature, min_new, options in requests:
            model.add(seq, ids, max_new=max_new, temperature=temperature, min_new=min_new, options=dict(options))
            runner.submit(seq, len(ids), now=0.0)
        for _ in range(400):
            if runner.step(now=0.0) is None:
                return
        raise AssertionError("the runner did not finish")

    def served(self, ids, count, options, min_new=0, **build):
        comp, model, runner = self.build(rows=1, **build)
        self.serve(model, runner, [(0, ids, count, build.get("temperature", 0.0), min_new, options)])
        return comp, model

    def test_bias_and_penalties_are_the_reference_loop(self):
        ids = prompt(3, 10)
        comp, model = self.served(ids, 8, {"logit_bias": {7: 2.0}})
        pushed = model.generated(0)
        self.assertEqual(pushed, [7] * 8)                                 # the bias alone repeats its token
        for extra in ({"presence_penalty": 2.0}, {"frequency_penalty": 1.0}, {"repetition_penalty": 2.0},
                      {"presence_penalty": 1.0, "frequency_penalty": 0.5, "repetition_penalty": 1.5}):
            options = dict({"logit_bias": {7: 2.0}}, **extra)
            with self.subTest(options=options):
                _, model = self.served(ids, 8, options)
                want, _ = reference_rich(comp, ids, 8, options)
                self.assertEqual(model.generated(0), want)
                self.assertNotEqual(want, pushed)                         # the penalty broke the repeat

    def test_min_tokens_hold_the_end_back_greedy_and_sampled(self):
        ids = prompt(3, 10)
        ended = {"logit_bias": {END: 50.0}}                               # the end would win every position
        comp, model = self.served(ids, 10, ended, min_new=5)
        self.assertEqual(model.generated(0), reference_rich(comp, ids, 10, ended, min_new=5)[0])
        self.assertEqual((len(model.generated(0)), model.generated(0)[-1]), (6, END))
        ids = prompt(6, 8)
        likely = {"logit_bias": {END: 8.0}, "seed": 11}
        _, free = self.served(ids, 12, likely, temperature=0.9)
        self.assertEqual(free.generated(0), [END])
        _, held = self.served(ids, 12, likely, min_new=6, temperature=0.9)
        self.assertNotIn(END, held.generated(0)[:6])
        self.assertEqual((len(held.generated(0)), held.generated(0)[-1]), (7, END))

    def test_a_reasoning_budget_forces_its_end_once(self):
        ids = prompt(3, 10)
        comp, plain = self.served(ids, 8, {})
        for budget in (2, 3):
            options = {"reasoning_budget": budget, "reasoning_end": 42}
            with self.subTest(budget=budget):
                _, model = self.served(ids, 8, options)
                out = model.generated(0)
                self.assertEqual(out, reference_rich(comp, ids, 8, options)[0])
                self.assertEqual((out[:budget], out[budget], out.count(42)), (plain.generated(0)[:budget], 42, 1))
                self.assertFalse(model.thinking[0])

    def test_logprobs_are_the_processed_rows(self):
        ids = prompt(4, 9)
        options = {"logprobs": 3, "logit_bias": {7: 2.0}, "presence_penalty": 2.0}
        comp, model = self.served(ids, 6, options)
        want, rows = reference_rich(comp, ids, 6, options)
        got = model.logprobs(0)
        self.assertEqual([t for t, _, _ in got], want)
        for (token, lp, top), row in zip(got, rows):
            ref = torch.log_softmax(row.float(), dim=-1)
            self.assertAlmostEqual(lp, float(ref[token]), places=4)
            self.assertEqual([i for i, _ in top], ref.topk(3).indices.tolist())
            for (i, value) in top:
                self.assertAlmostEqual(value, float(ref[i]), places=4)
        _, plain = self.served(ids, 6, {})
        self.assertIsNone(plain.logprobs(0))

    def test_a_grammar_confines_every_token_from_where_it_starts(self):
        ids = prompt(5, 8)
        allow = (5, 9, 21)
        g = grammar_stub(allow)
        _, model = self.served(ids, 8, {"grammar": {"type": "json_object"}}, grammars=g)
        out = model.generated(0)
        self.assertEqual(len(out), 8)
        self.assertTrue(set(out) <= set(allow), out)
        self.assertEqual(g.xgr.accepted, out)                             # the matcher walked every committed token
        comp, _ = tiny_composition(0)
        free, _ = reference_rich(comp, ids, 8, {})
        self.assertFalse(set(free) <= set(allow))
        after = free[1]
        self.assertNotEqual(free[0], after)
        g = grammar_stub(allow)
        _, model = self.served(ids, 8, {"grammar": {"type": "json_object"}, "grammar_after": after}, grammars=g)
        out = model.generated(0)
        self.assertEqual(out[:2], free[:2])                               # unconstrained up to and including `after`
        self.assertTrue(set(out[2:]) <= set(allow), out)
        self.assertEqual((len(out), g.xgr.accepted), (8, out[2:]))
        comp, bare, _ = self.build(rows=1)
        with self.assertRaisesRegex(ValueError, "no grammar compiler"):
            bare.validate_options({"grammar": {"type": "json_object"}})

    def test_drafts_change_nothing_with_every_option_on(self):
        from tests.test_engine_speculative import Scripted
        options = {"presence_penalty": 0.9, "repetition_penalty": 1.3, "logit_bias": {9: 2.0}, "logprobs": 2,
                   "grammar": {"type": "json_object"}, "reasoning_budget": 4, "reasoning_end": 21}
        allow = tuple(range(0, VOCAB, 3)) + (21,)
        for temperature, min_new in ((0.0, 0), (0.8, 3)):
            requests = [(0, prompt(7, 11), 10, temperature, min_new, dict(options, seed=3)),
                        (1, prompt(8, 6), 9, temperature, min_new, dict(options, seed=4))]
            comp, plain, runner = self.build(grammars=grammar_stub(allow), temperature=temperature)
            self.serve(plain, runner, requests)
            truth = {seq: list(plain.tokens[seq]) for seq, *_ in requests}
            for seq, *_ in requests:
                self.assertEqual(plain.generated(seq)[4], 21)             # the budget's end, forced inside the grammar
            for right in (None, 1):
                with self.subTest(temperature=temperature, right=right):
                    comp, drafted, runner = self.build(grammars=grammar_stub(allow), temperature=temperature,
                                                       drafter=lambda m, right=right: Scripted(3, m, truth, right=right))
                    self.serve(drafted, runner, requests)
                    for seq, *_ in requests:
                        self.assertEqual(drafted.generated(seq), plain.generated(seq))
                        got, want = drafted.logprobs(seq), plain.logprobs(seq)
                        self.assertEqual([(t, [i for i, _ in top]) for t, _, top in got],
                                         [(t, [i for i, _ in top]) for t, _, top in want])
                        for (_, a, _), (_, b, _) in zip(got, want):
                            self.assertAlmostEqual(a, b, places=4)
                    self.assertGreater(drafted.accepted_total, 0)
                    self.assertLess(drafted.steps, plain.steps)

    def test_a_parked_record_travels_as_json_and_the_next_turn_keeps_its_options(self):
        comp, model, runner = self.build(rows=2, keep_idle=True, grammars=grammar_stub(tuple(range(VOCAB))))
        ids = prompt(9, 6)
        options = {"logit_bias": {7: 2.0}, "presence_penalty": 2.0, "logprobs": 1, "grammar": {"type": "json_object"}}
        self.serve(model, runner, [(0, ids, 3, 0.0, 0, options)])
        plain = {k: v for k, v in options.items() if k != "grammar"}
        first = model.generated(0)
        self.assertEqual(first, reference_rich(comp, ids, 3, plain)[0])
        slot = runner.slot_of[0]
        held = model.state_bytes(slot).clone()
        record = json.loads(json.dumps(model.park(0)))                    # base/kv_tier writes it as JSON
        self.assertEqual(record["options"], {"logit_bias": {"7": 2.0}, "presence_penalty": 2.0, "logprobs": 1})
        for table in (model.tokens, model.matchers, model.lps, model.thinking, model.history.rows):
            self.assertNotIn(0, table)
        model.state_bytes(2).copy_(held)
        model.resume(0, 2, record)
        self.assertEqual(model.options[0]["logit_bias"], {7: 2.0})
        more = prompt(10, 4)
        turn = {"logit_bias": {9: 2.0}, "frequency_penalty": 2.0}
        fed = model.extend(0, more, max_new=3, temperature=0.0, options=turn)
        start = model.context(0)
        runner.kv.reserve_to((0,), (start + fed + 3,))
        with torch.no_grad():
            done = model.prefill(0, start, fed, None, 2)
            while not done:
                done = model.decode([0], None, None)[0]
        self.assertEqual(model.generated(0), reference_rich(comp, ids + first + more, 3, turn)[0])
        self.assertEqual(model.generated(0)[0], 9)



@unittest.skipUnless(torch is not None, "requires torch")
class DoorTests(unittest.TestCase):
    """The options through base/serve's door on a composition, and the Qwen3.8 boot that binds them."""

    def test_the_door_admits_the_options_and_the_answers_are_the_reference(self):
        from engine.base.comm import Comm
        from engine.base.serve import Server
        allow = (5, 9, 21)
        comp, model, runner = RichOptionTests().build(rows=2, grammars=grammar_stub(allow))
        s = Server(model, runner, Comm(), host="127.0.0.1", port=0)
        a, b = prompt(12, 9), prompt(13, 7)
        biased = {"logit_bias": {7: 2.0}, "presence_penalty": 2.0, "logprobs": 2}
        first = s.submit(a, 6, 0.0, options=biased)
        second = s.submit(b, 6, 0.0, options={"grammar": {"type": "json_object"}})
        for _ in range(400):
            if not s.once() and first[1].is_set() and second[1].is_set():
                break
        self.assertEqual(s.take_result(first[0]), reference_rich(comp, a, 6, biased)[0])
        out = s.take_result(second[0])
        self.assertEqual(len(out), 6)
        self.assertTrue(set(out) <= set(allow), out)

    def test_the_qwen38_boot_binds_grammars_and_speaks_the_doors_thinking_switch(self):
        from engine.profiles.qwen38 import boot
        composition, cfg, _ = boot.tiny()
        g = grammar_stub((5, 9))
        model, runner, _ = boot.build(composition, cfg, [cfg["eos_token_id"]], kv_gib=0.03, max_seqs=2, block_tokens=4,
                                      chunk=4, token_budget=16, snapshots=0, max_new=4, temperature=0.0, grammars=g)
        self.assertIs(model.grammars, g)
        model.validate_options({"grammar": {"type": "json_object"}})
        self.assertEqual(boot.template_kwargs({"thinking": False}), {"thinking": False, "enable_thinking": False})
        self.assertEqual(boot.template_kwargs({"thinking": True, "enable_thinking": True}),
                         {"thinking": True, "enable_thinking": True})
        self.assertEqual(boot.template_kwargs(None), {})
        # the template takes xhigh, medium and low (srv2, 2026-09-14); OpenAI's rungs all land there, aliases on a rung
        self.assertTrue(set(boot.EFFORT_RUNGS.values()) <= {"xhigh", "medium", "low"})
        self.assertTrue({"low", "medium", "high", "max"} <= set(boot.EFFORT_RUNGS))
        self.assertTrue(all(boot.EFFORT_RUNGS[k] == boot.EFFORT_RUNGS[v] for k, v in boot.EFFORT_ALIASES.items()))


if __name__ == "__main__":
    unittest.main()
