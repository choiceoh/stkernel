"""Structured output (45차 §23 A2, §28): the grammar masks over a toy tokenizer, the draft walk and its rollback,
and the step's one bitmask -- filled at each row's own offset, crossed once, applied by the kernel.
The first class runs where xgrammar and transformers are importable (the ST image); the rest run anywhere torch is."""
import sys
import threading
import json
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class GrammarTests(unittest.TestCase):
    """The real xgrammar, over a vocabulary small enough to read."""

    def setUp(self):
        try:
            import torch  # noqa: F401
            import xgrammar  # noqa: F401
            from tokenizers import Tokenizer, models, pre_tokenizers
            from transformers import PreTrainedTokenizerFast
        except ImportError as exc:
            self.skipTest(f"grammar stack unavailable here: {exc}")
        self.vocab = ["{", "}", "a", "1", ":", '"', ",", "<eos>"]
        tok = Tokenizer(models.WordLevel({t: i for i, t in enumerate(self.vocab)}, unk_token="<eos>"))
        tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        self.hf = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<eos>")
        from engine.base.grammar import Grammars
        self.grammars = Grammars(self.hf, len(self.vocab))

    def allowed(self, matcher, drafts):
        """The masks as sets of allowed ids: the step's fill, then the kernel onto logits of zero."""
        import torch
        masks = self.grammars.prepare([("row", matcher, list(drafts))], "cpu")
        live = masks.live("row", len(drafts) + 1)
        logits = torch.zeros(live, len(self.vocab))
        masks.apply("row", logits)
        return [set((~torch.isinf(row)).nonzero().flatten().tolist()) for row in logits]

    def test_json_object_masks_follow_the_grammar_and_drafts_roll_back(self):
        m = self.grammars.matcher({"type": "json_object"}, max_rollback=4)
        open_brace, close_brace, a, one, colon, quote = (self.vocab.index(x) for x in ("{", "}", "a", "1", ":", '"'))
        first = self.allowed(m, [])[0]
        self.assertIn(open_brace, first)
        self.assertNotIn(close_brace, first)                              # an object must open first
        self.assertNotIn(a, first)
        # drafts {, ": the second mask assumes the first draft was accepted; the matcher itself has not moved
        masks = self.allowed(m, [open_brace, quote])
        self.assertEqual(len(masks), 3)
        self.assertIn(close_brace, masks[1])
        self.assertIn(quote, masks[1])
        self.assertNotIn(colon, masks[1])
        self.assertIn(open_brace, self.allowed(m, [])[0])                 # rolled back: still at the start
        m.advance([open_brace, close_brace])
        done = self.allowed(m, [])[0]                                     # {} is complete: only the stop token may follow
        self.assertEqual(done, {self.vocab.index("<eos>")})

    def test_a_draft_the_grammar_refuses_ends_the_row(self):
        m = self.grammars.matcher({"type": "json_object"}, max_rollback=4)
        a = self.vocab.index("a")
        masks = self.grammars.prepare([("row", m, [a, a, a])], "cpu")
        self.assertEqual(masks.live("row", 4), 1)                         # 'a' cannot open an object: positions 1.. are dead
        self.assertIn(self.vocab.index("{"), self.allowed(m, [])[0])      # and the refusal did not move the matcher

    def test_stop_draft_ends_the_walk_and_termination_rolls_back(self):
        """DFlash may propose EOS with more drafts behind it; xgrammar rejects a mask after EOS."""
        eos, a = (self.vocab.index(x) for x in ("<eos>", "a"))
        for trailing in ([], [a, a]):
            m = self.grammars.matcher({"type": "json_object"}, max_rollback=5)
            m.advance([self.vocab.index("{"), self.vocab.index("}")])
            self.assertEqual(self.allowed(m, [eos, *trailing]), [{eos}])
            self.assertFalse(m.matcher.is_terminated())
            self.assertEqual(self.allowed(m, []), [{eos}])
            m.advance([eos])
            self.assertTrue(m.matcher.is_terminated())

    def test_dormant_draft_walk_across_answer_and_eos_rolls_back(self):
        after, brace, close, eos = (self.vocab.index(x) for x in ("a", "{", "}", "<eos>"))
        m = self.grammars.matcher({"type": "json_object"}, max_rollback=5, after=after)
        masks = self.allowed(m, [after, brace, close, eos, brace])
        self.assertEqual(len(masks), 4)
        self.assertEqual(masks[-1], {eos})
        self.assertFalse(m.armed)
        self.assertFalse(m.matcher.is_terminated())
        m.advance([after])
        self.assertIn(brace, self.allowed(m, [])[0])

    def test_a_schema_the_compiler_refuses_is_a_value_error_not_the_compiler_s_own(self):
        """xgrammar's C++ layer raises for a regex it cannot convert and a `$ref` that goes nowhere. Raised
        where a row is admitted, that would abort every live request on every rank; the door answers 400."""
        for schema in ('{"type": "string", "pattern": "(a)\\\\1"}',
                       '{"type": "object", "properties": {"a": {"$ref": "#/definitions/nope"}}}'):
            with self.assertRaises(ValueError) as caught:
                self.grammars.ready({"type": "json_schema", "schema": schema})
            self.assertIn("the grammar cannot be compiled", str(caught.exception))

    def test_the_compile_runs_on_a_thread_and_the_first_mask_waits_for_it(self):
        handle = self.grammars.compile({"type": "json_object"})
        self.assertTrue(hasattr(handle, "result"), "the compile is submitted, not done in the caller")
        self.assertIsNotNone(self.grammars.resolve(handle))
        m = self.grammars.matcher({"type": "json_object"}, max_rollback=2)
        self.assertIsNone(m.m, "the matcher is not built until a mask is asked for")
        self.assertIn(self.vocab.index("{"), self.allowed(m, [])[0])
        self.assertIsNotNone(m.m)

    def test_the_engine_s_end_tokens_are_the_grammar_s_stop_tokens(self):
        """Left to itself xgrammar takes the tokenizer's single eos. A model whose generation config ends on
        something else would finish its JSON on a token the engine does not stop at."""
        from engine.base.grammar import Grammars
        ends = [len(self.vocab) - 1, self.vocab.index("a")]          # pretend the engine also ends on 'a'
        g = Grammars(self.hf, len(self.vocab), stop_token_ids=ends)
        m = g.matcher({"type": "json_object"}, max_rollback=2)
        m.advance([self.vocab.index("{"), self.vocab.index("}")])
        masks = g.prepare([("row", m, [])], "cpu")
        import torch
        logits = torch.zeros(1, len(self.vocab))
        masks.apply("row", logits)
        self.assertEqual(sorted((~torch.isinf(logits[0])).nonzero().flatten().tolist()), sorted(ends))

    def test_the_real_compiler_and_kernel_qualify_together(self):
        self.grammars.qualify("cpu")                       # what boot does, on the tokenizer this test built

    def test_json_schema_is_enforced(self):
        m = self.grammars.matcher({"type": "json_schema", "schema": '{"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}'},
                                  max_rollback=2)
        self.assertIn(self.vocab.index("{"), self.allowed(m, [])[0])
        m.advance([self.vocab.index("{")])
        mask = self.allowed(m, [])[0]
        self.assertIn(self.vocab.index('"'), mask)
        self.assertNotIn(self.vocab.index("}"), mask)                     # "a" is required: the object cannot close yet


class _Xgr:
    """Enough of xgrammar to drive the step buffer where the real one is absent. It answers the way the real one
    was measured to: a fill overwrites the whole row it is given, and the kernel reads the packed words."""

    def __init__(self, words, allow, refuse=(), needed=True):
        self.words, self.allow, self.refuse, self.needed = words, list(allow), set(refuse), needed
        self.accepted, self.rolled, self.fills = [], 0, []

    # -- the compiler's half (what `qualify` reaches for) ---------------------------------
    def compile_builtin_json_grammar(self):
        return "builtin"

    # -- the module's half --------------------------------------------------------------
    def GrammarMatcher(self, compiled, max_rollback_tokens):
        self.max_rollback = max_rollback_tokens
        return self

    def allocate_token_bitmask(self, rows, vocab):
        import torch
        return torch.zeros(rows, self.words, dtype=torch.int32)

    def apply_token_bitmask_inplace(self, logits, bitmask, *, vocab_size=None, indices=None):
        import torch
        assert indices is None, "the step lays its rows out where the logits are: no index list"
        assert logits.shape[0] == bitmask.shape[0], (logits.shape, bitmask.shape)
        shift = torch.arange(32, dtype=torch.int32)
        bits = bitmask.unsqueeze(-1).bitwise_right_shift(shift).bitwise_and_(1).reshape(logits.shape[0], -1)
        logits.masked_fill_(bits[:, : logits.shape[-1]] == 0, float("-inf"))

    # -- the matcher's half -------------------------------------------------------------
    def fill_next_token_bitmask(self, mask, index=0):
        self.fills.append(index)
        mask[index] = 0                                  # the real one writes the whole row; nothing is OR'ed in
        for t in self.allow:
            mask[index, t // 32] |= 1 << (t % 32)
        return self.needed

    def is_terminated(self):
        return False

    def accept_token(self, token):
        if token in self.refuse:
            return False
        self.accepted.append(token)
        return True

    def rollback(self, n):
        self.rolled += n
        del self.accepted[len(self.accepted) - n:]


def fake(allow=(1, 5, 70), vocab=128, refuse=(), needed=True):
    """A `Grammars` with the fake module under it: the real buffers, the real walk, no xgrammar."""
    from concurrent.futures import ThreadPoolExecutor
    from engine.base.grammar import Grammars, Matcher
    g = Grammars.__new__(Grammars)
    g.xgr = _Xgr((vocab + 31) // 32, allow, refuse, needed)
    g.vocab_size, g.words = vocab, (vocab + 31) // 32
    g.staging = g.landing = g.crossed = None
    g._cache, g._lock = {}, threading.Lock()
    g.compiler, g._pool = g.xgr, ThreadPoolExecutor(max_workers=1)
    return g, Matcher(g, None, 5)


class StepBufferTests(unittest.TestCase):
    """One bitmask for the step: every row fills at its own offset, and the whole thing crosses once."""

    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            self.skipTest(str(exc))

    def test_every_position_gets_the_grammar_it_asked_for(self):
        import torch
        g, m = fake()
        masks = g.prepare([("r", m, [3, 4])], "cpu")
        self.assertEqual(masks.live("r", 3), 3)
        logits = torch.zeros(3, g.vocab_size)
        masks.apply("r", logits)
        for row in logits:
            self.assertEqual(sorted((~torch.isinf(row)).nonzero().flatten().tolist()), [1, 5, 70])
        self.assertEqual(g.xgr.rolled, 2, "the walk over the drafts is taken back")

    def test_each_row_owns_a_slice_so_no_mask_can_land_on_another_row(self):
        """vLLM fills compactly and then sorts the rows back into the batch's order, because one kernel call
        covers the batch and the speculative offsets have moved every row. Each row here is handed its own
        slice, so rows of different draft counts cannot cross."""
        import torch
        from engine.base.grammar import Matcher
        g, a = fake(allow=(1, 5))
        b = Matcher(g, None, 5)
        b.m = _Xgr(g.words, allow=(70,))                       # a different grammar, a shorter row
        masks = g.prepare([("a", a, [3, 4]), ("b", b, [7])], "cpu")
        self.assertEqual((masks.filled["a"][:2], masks.filled["b"][:2]), ((0, 3), (3, 2)))
        for key, want in (("a", [1, 5]), ("b", [70])):
            logits = torch.zeros(masks.live(key, 3), g.vocab_size)
            masks.apply(key, logits)
            for row in logits:
                self.assertEqual(sorted((~torch.isinf(row)).nonzero().flatten().tolist()), want)
        self.assertTrue(bool((g.landing[:5] == g.staging[:5]).all()))

    def test_the_buffers_are_kept_between_steps_and_grow_only_upwards(self):
        g, m = fake()
        g.prepare([("r", m, [1])], "cpu")
        staging, landing = g.staging, g.landing
        g.prepare([("r", m, [1])], "cpu")
        self.assertIs(g.staging, staging)
        self.assertIs(g.landing, landing)
        g.prepare([("r", m, [1, 2, 3])], "cpu")
        self.assertIsNot(g.staging, staging)
        wide = g.staging
        g.prepare([("r", m, [1])], "cpu")
        self.assertIs(g.staging, wide)

    def test_a_refused_draft_ends_the_row_and_the_dead_positions_are_never_filled(self):
        g, m = fake(refuse=(4,))
        masks = g.prepare([("r", m, [3, 4, 3])], "cpu")
        self.assertEqual(masks.live("r", 4), 2)                  # position 0 and 1; draft 4 is refused at 1
        self.assertEqual(g.xgr.fills, [0, 1], "nothing behind the refused draft is filled")
        self.assertEqual(g.xgr.rolled, 1, "only the draft that was accepted is taken back")

    def test_a_row_without_a_grammar_keeps_all_of_its_positions_and_is_not_masked(self):
        import torch
        g, m = fake()
        masks = g.prepare([("r", m, [1])], "cpu")
        self.assertEqual(masks.live("other", 6), 6)
        self.assertFalse(masks.has("other"))
        logits = torch.zeros(6, g.vocab_size)
        masks.apply("other", logits)
        self.assertEqual(int(torch.isinf(logits).sum()), 0)

    def test_a_mask_that_refuses_nothing_is_not_applied_at_all(self):
        import torch
        g, m = fake(needed=False)
        masks = g.prepare([("r", m, [])], "cpu")
        logits = torch.zeros(1, g.vocab_size)
        masks.apply("r", logits)                                 # the fake would mask everything but 1, 5, 70
        self.assertEqual(int(torch.isinf(logits).sum()), 0)

    def test_the_mask_lands_on_top_of_what_the_sampler_already_forbade(self):
        import torch
        g, m = fake()
        masks = g.prepare([("r", m, [])], "cpu")
        logits = torch.zeros(1, g.vocab_size)
        logits[0, 5] = float("-inf")                             # min_tokens' end ids, written before the mask
        masks.apply("r", logits)
        self.assertEqual(sorted((~torch.isinf(logits[0])).nonzero().flatten().tolist()), [1, 70])

    def test_the_crossing_happens_once_after_the_walk_not_inside_it(self):
        source = (Path(__file__).resolve().parents[1] / "engine/base/grammar.py").read_text()
        code = source[source.index("from __future__"):]          # the prose above says what the code must not do
        body = code[code.index("    def prepare(self"):code.index("    def qualify(self")]
        self.assertLess(body.index("for key, matcher, drafts in rows"), body.index(".copy_("))
        self.assertEqual(body.count(".copy_("), 1, "one transfer for the step, not one per row")
        # the kernel writes -inf; we never build a vocabulary of bool, and a fill overwrites its row, so no reset
        self.assertEqual(code.count("masked_fill"), 0, "the mask is the kernel's, not a bool vocabulary's")
        self.assertEqual(code.count("reset_token_bitmask("), 0, "a fill overwrites its row: a reset is a second memset")


class QualifyTests(unittest.TestCase):
    """What cannot be masked does not boot (D3): the kernel's verdict on the device is compared with the same
    packed words expanded on the host, and a mask that allows everything or nothing fails too."""

    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            self.skipTest(str(exc))

    def test_a_kernel_that_agrees_with_the_bitmask_qualifies(self):
        g, _ = fake()
        g.qualify("cpu")                                   # no raise

    def test_a_kernel_that_masks_the_wrong_ids_fails_the_boot(self):
        g, _ = fake()
        real = g.xgr.apply_token_bitmask_inplace

        def wrong(logits, bitmask, *, vocab_size=None, indices=None):
            real(logits, bitmask, vocab_size=vocab_size, indices=indices)
            logits[0, 1] = float("-inf")                   # one id the words said was allowed
        g.xgr.apply_token_bitmask_inplace = wrong
        with self.assertRaises(RuntimeError) as caught:
            g.qualify("cpu")
        self.assertIn("disagrees with the bitmask", str(caught.exception))

    def test_a_mask_that_allows_everything_fails_the_boot(self):
        """All-true at a JSON start means the head and the tokenizer disagree about the vocabulary -- every
        later mask would inherit that silently."""
        g, _ = fake(allow=range(128))
        with self.assertRaises(RuntimeError) as caught:
            g.qualify("cpu")
        self.assertIn("do not agree on the vocabulary", str(caught.exception))


class ReasoningGateTests(unittest.TestCase):
    """A grammar that starts inside the model's reasoning forbids the block's own end token: the block never
    closes, and the whole answer comes back as reasoning. `after` is the token it waits for."""

    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            self.skipTest(str(exc))

    def gate(self, after=7, **kw):
        from engine.base.grammar import Matcher
        g, _ = fake(**kw)
        return g, Matcher(g, None, 5, after=after)

    def allowed(self, g, masks, key, rows):
        import torch
        logits = torch.zeros(rows, g.vocab_size)
        masks.apply(key, logits)
        return [sorted((~torch.isinf(r)).nonzero().flatten().tolist()) for r in logits]

    def test_a_dormant_row_costs_nothing_and_refuses_nothing(self):
        g, m = self.gate()
        masks = g.prepare([("r", m, [1, 2, 3])], "cpu")
        self.assertEqual(masks.live("r", 4), 4, "every position is still reachable: nothing is constrained")
        self.assertEqual(g.xgr.fills, [], "a step of pure reasoning does not touch the grammar")
        import torch
        logits = torch.zeros(4, g.vocab_size)
        masks.apply("r", logits)
        self.assertEqual(int(torch.isinf(logits).sum()), 0)

    def test_a_draft_that_ends_the_reasoning_arms_the_rest_of_the_step_and_is_taken_back(self):
        g, m = self.gate()
        masks = g.prepare([("r", m, [1, 7, 2])], "cpu")
        self.assertEqual(g.xgr.fills, [2, 3], "only the positions behind the end token are filled")
        rows = self.allowed(g, masks, "r", 4)
        self.assertEqual(len(rows[0]), g.vocab_size, "the reasoning positions stay open under the same kernel call")
        self.assertEqual(len(rows[1]), g.vocab_size)
        self.assertEqual(rows[2], [1, 5, 70])
        self.assertEqual(rows[3], [1, 5, 70])
        self.assertEqual(g.xgr.rolled, 1, "the draft the grammar accepted is taken back")
        self.assertFalse(m.armed, "a draft is not a commit: the row is still dormant")

    def test_only_a_committed_token_arms_it(self):
        g, m = self.gate()
        m.advance([1, 2])
        self.assertFalse(m.armed)
        g.prepare([("r", m, [])], "cpu")
        self.assertEqual(g.xgr.fills, [])
        m.advance([7])
        self.assertTrue(m.armed)
        masks = g.prepare([("r", m, [])], "cpu")
        self.assertEqual(g.xgr.fills, [0])
        self.assertEqual(self.allowed(g, masks, "r", 1), [[1, 5, 70]])

    def test_the_tokens_after_the_end_token_are_fed_to_the_grammar(self):
        g, m = self.gate()
        m.advance([1, 7, 5, 70])
        self.assertTrue(m.armed)
        self.assertEqual(g.xgr.accepted, [5, 70], "only what came after the reasoning end is the answer")


class PickRichTests(unittest.TestCase):
    """The adapter's rich pick over a grammar row: the mask decides what may be picked, and a draft the grammar
    refuses ends the row before the positions behind it cost anything."""

    def engine(self, vocab=128, k=3, allow=(1, 5, 70), refuse=(), rows=(0,)):
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            self.skipTest(str(exc))
        from types import SimpleNamespace
        from engine.profiles.glm53.adapter import Glm53Engine
        g, m = fake(allow=allow, vocab=vocab, refuse=refuse)
        e = Glm53Engine.__new__(Glm53Engine)                 # the methods, none of the boot
        e.options, e.limits, e.gens, e.gen = {r: {} for r in rows}, {r: (16, 0.0) for r in rows}, {}, torch.Generator().manual_seed(0)
        e.matchers, e.grammars = {0: m}, g
        e.tokens, e.prompt_len, e.min_new, e.ends, e._ends_tensor = {r: [9] for r in rows}, {r: 1 for r in rows}, {}, {}, {}
        e.sampling_history, e.decodable, e._rich_stage = None, None, None
        e.drafter, e.eos = SimpleNamespace(k=k), set()
        e.top_p, e.caches = 1.0, SimpleNamespace(pool=SimpleNamespace(max_seqs=1))
        return e, g, m

    def logits(self, rank):
        """[rows, vocab] raw logits whose argmax is outside the grammar, and whose best allowed id is `rank[i]`."""
        import torch
        out = torch.zeros(len(rank), 128)
        out[:, 0] = 100.0                                    # id 0 is not in the grammar: the mask has to beat it
        for i, want in enumerate(rank):
            out[i, want] = 10.0
        return out

    def test_the_mask_decides_the_pick_and_a_refused_draft_ends_the_row(self):
        import torch
        e, g, m = self.engine(refuse=(4,))
        rows = self.logits([5, 70, 1, 1])
        raw = rows.clone()
        masks = g.prepare([(0, m, [5, 4, 3])], "cpu")
        self.assertEqual(masks.live(0, 4), 2, "the grammar refuses draft 4: positions 2.. are dead")
        # the row is handed over already trimmed to its live positions, as the step's gather does
        (accepted, new, lps), = e._pick_rich([(0, rows[:2], [5, 4, 3], None)], masks)
        self.assertEqual(new, [5, 70], "the mask's best allowed id at each live position, not id 0")
        self.assertEqual(accepted, 1, "the first draft was picked; the row stops at the refused one")
        self.assertIsNone(lps)
        self.assertEqual(g.xgr.fills, [0, 1], "the dead positions were never filled")
        self.assertTrue(torch.equal(rows, raw), "the gathered logits are read, not masked in place")

    def test_the_row_writes_its_positions_into_one_kept_buffer(self):
        """The step's block is the buffer: the mask kernel wants consecutive rows and so does the sampler."""
        e, g, m = self.engine()
        masks = g.prepare([(0, m, [5, 5])], "cpu")
        e._pick_rich([(0, self.logits([5, 70, 1]), [5, 5], None)], masks)
        kept = e._rich_stage
        self.assertEqual(tuple(kept[0].shape), (4, 128), "k + 1 positions, the widest step a row can take")
        e._pick_rich([(0, self.logits([5, 70, 1]), [5, 5], None)], masks)
        self.assertIs(e._rich_stage, kept)

    def test_two_grammar_rows_in_one_call_keep_their_own_span_of_the_block(self):
        """The premise the mask kernel rests on: a row's positions are CONSECUTIVE rows of the step's block,
        so `apply(seq, block[at:at+live])` covers that row and cannot reach into its neighbour's. Batching the
        pick did not change that -- the block is filled row by row, each at its own offset."""
        import torch
        from engine.base.grammar import Matcher
        e, g, a = self.engine(allow=(1, 5), rows=(0, 1))
        b = Matcher(g, None, 5)
        b.m = _Xgr(g.words, allow=(70,), refuse=(), needed=True)          # a different grammar, a shorter row
        e.matchers[1] = b
        masks = g.prepare([(0, a, [5, 1]), (1, b, [70])], "cpu")
        answers = e._pick_rich([(0, self.logits([5, 1, 1]), [5, 1], None),
                                (1, self.logits([70, 70]), [70], None)], masks)
        self.assertEqual([new for _, new, _ in answers], [[5, 1, 1], [70, 70]])
        block = e._rich_stage[0]
        self.assertEqual(sorted((~torch.isinf(block[0])).nonzero().flatten().tolist()), [1, 5])
        self.assertEqual(sorted((~torch.isinf(block[3])).nonzero().flatten().tolist()), [70],
                         "the second row's span starts where the first one ended")
        self.assertEqual(block[:5].data_ptr(), block.data_ptr(), "the spans are views, not copies")

    def test_a_row_with_no_step_to_ride_along_with_fills_its_own(self):
        """The prompt's first token is picked outside any decode step: it still gets its mask."""
        e, g, m = self.engine()
        (accepted, new, lps), = e._pick_rich([(0, self.logits([70]), [], None)])
        self.assertEqual(new, [70])
        self.assertEqual(g.xgr.fills, [0])



class CacheCeilingTests(unittest.TestCase):
    """A cache without a ceiling is not a decision (45차 §53's rule, §63's numbers)."""

    def cache(self, kept=3):
        from engine.base.grammar import Grammars
        from collections import OrderedDict
        import threading
        g = Grammars.__new__(Grammars)
        g._cache, g._lock, g.KEPT = OrderedDict(), threading.Lock(), kept
        g.compiles = g.cache_hits = g.cache_evictions = 0
        built = []

        class Pool:
            def submit(self, fn, spec):
                built.append(spec["schema"]["title"])
                return spec["schema"]["title"]
        g._pool = Pool()
        return g, built

    def spec(self, i):
        return {"type": "json_schema", "schema": {"title": f"s{i}", "type": "object"}}

    def test_the_oldest_compiled_grammar_leaves_at_the_ceiling(self):
        g, built = self.cache(kept=3)
        for i in range(5):
            g.compile(self.spec(i))
        self.assertEqual(len(g._cache), 3)
        self.assertEqual((g.compiles, g.cache_evictions, g.cache_hits), (5, 2, 0))
        self.assertEqual(built, ["s0", "s1", "s2", "s3", "s4"])

    def test_a_hit_is_not_a_compile_and_refreshes_the_entry(self):
        g, built = self.cache(kept=2)
        g.compile(self.spec(0))
        g.compile(self.spec(1))
        self.assertEqual(g.compile(self.spec(0)), "s0", "the same handle came back")
        g.compile(self.spec(2))                          # s1 is now the oldest, not s0
        self.assertEqual(sorted(k for k in g._cache), sorted(json.dumps(self.spec(i), sort_keys=True) for i in (0, 2)))
        self.assertEqual((g.compiles, g.cache_hits, g.cache_evictions), (3, 1, 1))

    def test_the_declared_ceiling_is_a_number_somebody_chose(self):
        from engine.base.grammar import Grammars
        self.assertEqual(Grammars.KEPT, 256)             # ~44 MiB at the measured 173 KiB json_schema

if __name__ == "__main__":
    unittest.main()
