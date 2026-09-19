"""Qwen3.8's fleet serves grammars: the compiler is built off the prelude on every rank and bound to the served model
(fleet.door_host_half -> fleet.bind_grammars), so a request that carries `tools` is admitted.

Found in an MTP data window on the Qwen3.8 fleet (2026-09-19, main a8e3c4de): every /v1/chat/completions request with
`tools` was answered 400 "structured output (response_format) is not served: no grammar compiler is bound". The door
arms a lazy tool-call grammar for every such request (base/serve), and fleet.build built the served model
(adapter.build_model) without the compiler the Qwen3.8 CPU boot and GLM-5.3's fleet bind, so Deneb's agentic traffic
could not be served by Qwen3.8 at all.

Held here on the CPU over a served model built the way fleet.build builds it -- adapter.build_model over a stand-in net
and caches, so the real ServedStore, ServedComposition, ServedMTP and ServedModel -- and the fake xgrammar of
tests/test_engine_grammar.py under the real Grammars (its compile, buffers, walk and qualify):

    PreludeTests     the compiler is built on every rank, not only the one that renders, off the door's tokenizer and
                     with no device: the mask kernel is the main thread's
    BindTests        bind_grammars proves the mask on the device it is given, then binds; a compiler built for another
                     vocabulary or other end tokens, or one whose kernel disagrees, is not bound; no xgrammar binds
                     nothing
    BootOrderTests   fleet.build binds it after the prelude joins and before the capture
    DoorTests        a chat request with tools: 400 from the unbound model (the bug), served by the bound one, the lazy
                     tool grammar on its row
    VerifyTests      the served verify keeps a grammar row a rich row -- the head's argmax drafts, no block
                     verification, every position picked through its mask, greedy or sampled, beside a plain row that
                     keeps its batched draw; a lazy grammar arms at its marker inside a verify step; the draft-ahead
                     verify (the device's argmax, no mask) never takes a step with a grammar row
"""
import ast
import concurrent.futures
import importlib.util
import json
import tempfile
import unittest
import urllib.error
import urllib.request
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

torch = None
if importlib.util.find_spec("torch") is not None:
    import torch

ROOT = Path(__file__).resolve().parents[1]
FLEET = ROOT / "engine/profiles/qwen38/fleet.py"

V, END, K = 512, 400, 3             # the vocabulary, the end token, drafts a step
S = 300                             # the tool-call marker: one token, as <tool_call> is
ALLOW = (5, 9)                      # what the stand-in grammar lets an armed row write
NEXT = {20: 21, 21: S, S: 22, 22: 23}
TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {
    "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]


def favourite(t: int) -> int:
    """The stand-in target's unconstrained next token after `t` -- never one the grammar allows."""
    return NEXT.get(t, 30 + t % 50)


class Net:
    """net.Qwen38Net's surface the served composition and the MTP head read, with a fixed rule behind it: after token t
    the logits favour favourite(t) (4.0), then 5 (2.0), then 9 (1.0), everything else -5. A hidden row carries its
    position's token, and the head drafts the favourite -- what a head trained on the target proposes, and what a
    grammar row's mask refuses."""

    def __init__(self):
        self.F = SimpleNamespace(ngram_size=3)
        self.comm = SimpleNamespace(world_size=1, rank=0)

    def forward(self, step, caches, *, streams=False):
        rows = step.ids.to(torch.float32)[:, None]
        return rows, rows.repeat(1, 4)

    def head(self, out):
        ids = out[:, 0].to(torch.int64).tolist()
        logits = torch.full((len(ids), V), -5.0)
        logits[:, ALLOW[0]], logits[:, ALLOW[1]] = 2.0, 1.0
        for i, t in enumerate(ids):
            logits[i, favourite(t)] = 4.0
        return logits

    def mtp_forward(self, step, given, caches, *, last_hidden_only=True, rows=None):
        last = step.ids[-1:].to(torch.float32)[:, None]
        return last, last.repeat(1, 4)

    def draft_tokens(self, hidden):
        return torch.tensor([favourite(int(t)) for t in hidden[:, 0]])


class Caches:
    """Qwen38Caches' surface the served store, the MTP head and the runner read, with nothing behind it."""

    def __init__(self, rows: int = 2, blocks: int = 64, block: int = 4):
        from engine.base.kv import BlockPool, SlotPool
        self.F = SimpleNamespace(spec_k=K)
        self.pool = BlockPool(blocks, block, rows, blocks)
        self.slots = SlotPool(rows + 1)
        self.device = torch.device("cpu")

    def reset_slot(self, slot):
        pass

    def prepare(self, step):
        pass


def fleet_model(*, candidates: int = 0, draft_ahead: bool = False):
    """The served model as fleet.build's `engine` row builds it -- adapter.build_model, the MTP head drafting K a step
    (eagerly: nothing is captured on the CPU) -- over the stand-ins -> (model, caches)."""
    from engine.profiles.qwen38.adapter import build_model
    caches = Caches()
    F = SimpleNamespace(spec_k=K, vocab=V, max_position=4096)
    model, _ = build_model(Net(), caches, F, eos_ids=[END], max_new=64, temperature=0.0, top_p=1.0, drafter=True,
                           draft_candidates=candidates, draft_ahead=draft_ahead)
    return model, caches


def compiler():
    """base/grammar.Grammars over the fake xgrammar: an armed row may write only ALLOW (committing anything else raises
    in Matcher.advance), an EBNF compiles to its own text, qualify runs for real. `armed`: (spec, max_rollback, after)
    of every matcher built."""
    from tests.test_engine_grammar import fake
    g, _ = fake(allow=ALLOW, vocab=V, refuse=set(range(V)) - set(ALLOW))
    g._cache = OrderedDict()                                     # the real compile's cache (fake() hands a dict)
    g.xgr.Grammar = SimpleNamespace(from_ebnf=lambda text: text)
    g.xgr.compile_grammar = lambda grammar: ("ebnf", grammar)
    g.armed, matcher = [], g.matcher

    def recorded(spec, max_rollback, after=None):
        g.armed.append((spec, max_rollback, after))
        return matcher(spec, max_rollback, after)
    g.matcher = recorded
    return g


def prelude(grammars, vocab: int = V, stops=(END,)) -> dict:
    """The grammar half of what door_host_half hands build."""
    return {"grammars": grammars, "grammar_vocab": vocab, "grammar_stops": list(stops)}


class PreludeTests(unittest.TestCase):
    def half(self, renderer: bool):
        from unittest import mock
        from engine.base import grammar, serve, tool_formats
        from engine.profiles.qwen38 import boot, facts, fleet
        tok, built = object(), []
        with tempfile.TemporaryDirectory() as meta:
            (Path(meta) / "generation_config.json").write_text(json.dumps({"eos_token_id": [248046, 248044]}))
            with mock.patch.object(boot, "tokenizer", return_value=tok), \
                    mock.patch.object(boot, "chat_renderer", return_value=lambda *a, **k: ""), \
                    mock.patch.object(boot, "generation_defaults", return_value={}), \
                    mock.patch.object(facts, "load", return_value=SimpleNamespace(vocab=248320, config={})), \
                    mock.patch.object(serve, "reasoning_marks", return_value=(None, ())), \
                    mock.patch.object(serve, "effort_rungs_checked", return_value=None), \
                    mock.patch.object(tool_formats, "detect", return_value=None), \
                    mock.patch.object(grammar, "for_checkpoint",
                                      side_effect=lambda *a, **k: built.append((a, k)) or "compiler"):
                door = fleet.door_host_half(meta, renderer=renderer)
        return door, built, tok, meta

    def test_every_rank_builds_the_compiler_off_the_doors_tokenizer(self):
        """Rank 0 renders; ranks 1-3 do not, and each of them still runs a matcher for every grammar row it follows."""
        for renderer in (False, True):
            with self.subTest(renderer=renderer):
                door, built, tok, meta = self.half(renderer)
                # the checkpoint's vocabulary and end tokens, the door's own tokenizer, no device: the compiler alone
                self.assertEqual(built, [((meta, 248320, None, [248046, 248044]), {"tokenizer": tok})])
                self.assertEqual(door["grammars"], "compiler")
                self.assertEqual((door["grammar_vocab"], door["grammar_stops"]), (248320, [248046, 248044]))
                self.assertEqual(door["chat"] is not None, renderer)


@unittest.skipUnless(torch is not None, "requires torch")
class BindTests(unittest.TestCase):
    def test_the_compiler_is_proven_on_the_device_it_is_given_then_bound(self):
        from engine.profiles.qwen38.fleet import bind_grammars
        model, _ = fleet_model()
        with self.assertRaisesRegex(ValueError, "no grammar compiler is bound"):         # the fleet's model before
            model.validate_options({"grammar": {"type": "json_object"}})
        g, seen = compiler(), []
        qualify = g.qualify
        g.qualify = lambda device: (seen.append(device), qualify(device))
        bind_grammars(model, prelude(g), "cpu")
        self.assertIs(model.grammars, g)
        self.assertEqual((seen, g.xgr.fills), (["cpu"], [0]))     # a real mask filled and checked against the kernel
        for spec in ({"type": "json_object"}, {"type": "ebnf", "grammar": "root ::= call"}):
            model.validate_options({"grammar": spec})
            model.prepare_options({"grammar": spec})               # the door's compile, before admission

    def test_a_compiler_built_for_another_table_is_not_bound(self):
        from engine.profiles.qwen38.fleet import bind_grammars
        for vocab, stops in ((V + 64, (END,)), (V, (END, 7)), (V, (7,))):
            with self.subTest(vocab=vocab, stops=stops):
                model, _ = fleet_model()
                g = compiler()
                with self.assertRaisesRegex(RuntimeError, "built a grammar compiler for"):
                    bind_grammars(model, prelude(g, vocab, stops), "cpu")
                self.assertIsNone(model.grammars)
                self.assertEqual(g.xgr.fills, [])                  # refused before the kernel ran

    def test_a_kernel_that_disagrees_is_not_bound(self):
        from engine.profiles.qwen38.fleet import bind_grammars
        model, _ = fleet_model()
        g = compiler()
        real = g.xgr.apply_token_bitmask_inplace

        def wrong(logits, bitmask, *, vocab_size=None, indices=None):
            real(logits, bitmask, vocab_size=vocab_size, indices=indices)
            logits[0, ALLOW[0]] = float("-inf")                    # an id the words allowed
        g.xgr.apply_token_bitmask_inplace = wrong
        with self.assertRaisesRegex(RuntimeError, "disagrees with the bitmask"):
            bind_grammars(model, prelude(g), "cpu")
        self.assertIsNone(model.grammars)

    def test_without_xgrammar_nothing_is_bound(self):
        from engine.profiles.qwen38.fleet import bind_grammars
        model, _ = fleet_model()
        bind_grammars(model, prelude(None), "cpu")
        self.assertIsNone(model.grammars)


class BootOrderTests(unittest.TestCase):
    def test_build_binds_the_compiler_after_the_prelude_and_before_the_capture(self):
        """The prelude is joined before the capture (its thread would take the GIL from it); the mask kernel is proven
        right after, on the main thread, before the capture and the memory ledger's `ready` row."""
        from tests.test_engine_qwen38_boot import function
        build = ast.get_source_segment(FLEET.read_text(encoding="utf-8"), function(FLEET, "build"))
        order = [build.index(text) for text in (
            "model, _store = build_model(", 'with recorder.phase("wait for the prelude")', "door = prelude.take()",
            'with recorder.phase("qualify grammar")', "bind_grammars(model, door, caches.device)",
            'with recorder.phase("capture decode")', 'memory.checkpoint("ready")')]
        self.assertEqual(order, sorted(order))


def post(base: str, path: str, body: dict) -> dict:
    request = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=10) as r:
        return json.load(r)


def get(base: str, path: str) -> dict:
    with urllib.request.urlopen(base + path, timeout=10) as r:
        return json.load(r)


@unittest.skipUnless(torch is not None, "requires torch")
class DoorTests(unittest.TestCase):
    """base/serve's door over the fleet's model, with Qwen3.8's tool format (FUNCTION_XML) and its marker token."""

    BODY = {"messages": [{"role": "user", "content": "weather in Seoul?"}], "max_tokens": 4, "tools": TOOLS}

    def door(self, model, caches):
        from engine.base.comm import Comm
        from engine.base.record import Ring
        from engine.base.runner import STEP_RECORD, Runner
        from engine.base.scheduler import Contract
        from engine.base.serve import Server
        from engine.base.tool_formats import FUNCTION_XML
        from tests.test_engine_serve import Tokenizer
        runner = Runner(model, Contract(chunk_align=4, token_budget=16, draft_slots=model.k, max_wait_s=0.0,
                                        max_running=2), caches.pool, caches.slots, Ring(64, STEP_RECORD.size))

        def render(messages, kwargs, *, generation_prompt=True, continue_final=False):
            return "".join(m.get("content") or "" for m in messages)
        return Server(model, runner, Comm(), host="127.0.0.1", port=0, tokenizer=Tokenizer(), chat=render,
                      model_name="qwen3.8-flash-next", tool_parser=FUNCTION_XML.parse, tool_stream=FUNCTION_XML.partial,
                      tool_grammar=FUNCTION_XML.grammar, tool_call_start=S)

    def serve(self, s, fn):
        from tests.test_engine_serve import drive
        httpd = s._serve_http()
        try:
            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                return drive(s, pool.submit(fn, f"http://127.0.0.1:{httpd.server_port}"))
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_unbound_a_request_with_tools_is_refused(self):
        """The bug as the fleet served it: the door armed the tool grammar and the model had nothing to hold it with."""
        model, caches = fleet_model()
        s = self.door(model, caches)
        with self.assertRaises(urllib.error.HTTPError) as refused:
            self.serve(s, lambda base: post(base, "/v1/chat/completions", self.BODY))
        self.assertEqual(refused.exception.code, 400)
        self.assertIn("no grammar compiler is bound", refused.exception.read().decode())
        self.assertFalse(s.pending or s.results)

    def test_bound_a_request_with_tools_is_served_with_the_lazy_tool_grammar(self):
        from engine.profiles.qwen38.fleet import bind_grammars
        model, caches = fleet_model()
        g = compiler()
        bind_grammars(model, prelude(g), "cpu")
        s = self.door(model, caches)
        out, card = self.serve(s, lambda base: (post(base, "/v1/chat/completions", self.BODY),
                                                get(base, "/v1/models/qwen3.8-flash-next")))
        self.assertEqual(out["choices"][0]["finish_reason"], "length")
        self.assertEqual(out["usage"]["completion_tokens"], 4)
        # the row's matcher: the door's EBNF for the declared tool, armed at the marker, rolling back a verify step
        (spec, rollback, after), = [a for a in g.armed if a[0]["type"] == "ebnf"]
        self.assertIn('"get_weather"', spec["grammar"])
        self.assertEqual((rollback, after), (K + 2, S))
        self.assertTrue(card["capabilities"]["structured_output"])
        self.assertTrue(card["capabilities"]["tool_grammar"])


@unittest.skipUnless(torch is not None, "requires torch")
class VerifyTests(unittest.TestCase):
    """adapter.ServedModel._verify over grammar rows, the MTP head proposing the target's unconstrained favourites."""

    def bound(self, candidates: int = 0, draft_ahead: bool = False):
        from engine.profiles.qwen38.fleet import bind_grammars
        model, _ = fleet_model(candidates=candidates, draft_ahead=draft_ahead)
        g = compiler()
        bind_grammars(model, prelude(g), "cpu")
        handed, propose = [], model.drafter.propose

        def recorded(seqs, sampling=None):
            handed.append(None if sampling is None else dict(sampling))
            return propose(seqs) if sampling is None else propose(seqs, sampling=sampling)
        model.drafter.propose = recorded
        return model, g, handed

    def run_rows(self, model, rows: dict, *, steps: "int | None" = None) -> None:
        """rows: seq -> (prompt, max_new, temperature, options), prefilled into slots 1.., then decoded together to
        the end (or for `steps` steps)."""
        with torch.no_grad():
            for slot, (seq, (ids, max_new, temperature, options)) in enumerate(rows.items(), start=1):
                model.add(seq, ids, max_new=max_new, temperature=temperature, options=dict(options))
                model.open(seq, slot)
                model.prefill(seq, 0, len(ids), None, slot)
            live = list(rows)
            while live and steps != 0:
                done = model.decode(live, None, None)
                live = [seq for seq, d in zip(live, done) if not d]
                steps = None if steps is None else steps - 1

    def test_a_greedy_grammar_row_is_picked_through_its_mask_at_every_position(self):
        """Its drafts are the head's favourites, which the mask refuses: each step keeps the masked pick at the first
        position and nothing past it. A plain greedy row beside it takes its drafts, drawn ahead in one call."""
        model, g, _ = self.bound()
        grammar = {"grammar": {"type": "ebnf", "grammar": "root ::= call"}}
        self.run_rows(model, {1: ([11, 12, 13], 8, 0.0, grammar), 2: ([14, 15, 16], 8, 0.0, {})})
        self.assertEqual(model.generated(1), [ALLOW[0]] * 8)       # the mask's best, never the favourite
        self.assertEqual(g.xgr.accepted, model.generated(1))      # the matcher moved by exactly what was committed
        chain = [favourite(16)]
        while len(chain) < 8:
            chain.append(favourite(chain[-1]))
        self.assertEqual(model.generated(2), chain)
        self.assertGreater(model.accepted_total, 0)               # the plain row kept its drafts

    def test_a_sampled_grammar_row_is_never_drawn_as_a_plain_one(self):
        """At a temperature a plain row is handed to the head for drawn drafts (block-verified where the captured
        graphs draw them; the eager head here keeps its argmax) and its positions are drawn ahead in one call; a
        grammar row gets neither -- the head's argmax drafts, each position picked through its mask. Were it drawn as
        a plain row, a favourite (86% of the unmasked mass here) would be committed and the matcher would raise."""
        model, g, handed = self.bound(candidates=2)
        grammar = {"grammar": {"type": "ebnf", "grammar": "root ::= call"}}     # no seed: a seed makes a row rich
        self.run_rows(model, {1: ([11, 12, 13], 10, 0.9, grammar), 2: ([14, 15, 16], 10, 0.9, {})})
        self.assertTrue(handed and all(sampling is not None and 1 not in sampling for sampling in handed))
        self.assertIn(2, handed[0])                                # the plain row is sampled beside it
        self.assertEqual(len(model.generated(1)), 10)
        self.assertLessEqual(set(model.generated(1)), set(ALLOW))
        self.assertEqual(g.xgr.accepted, model.generated(1))
        self.assertTrue(set(model.generated(2)) - set(ALLOW))      # unmasked, the net writes other things

    def test_a_lazy_grammar_arms_at_its_marker_inside_a_verify_step(self):
        """The door's tool grammar waits for the call marker (`grammar_after`). The head drafts the marker and what the
        unconstrained model would write after it; the marker is kept, it arms the grammar, and the next position of
        the same verify step is already picked through the mask."""
        model, g, _ = self.bound()
        tool = {"grammar": {"type": "ebnf", "grammar": "root ::= call"}, "grammar_after": S}
        self.run_rows(model, {1: ([11, 12, 20], 8, 0.0, tool)}, steps=1)
        self.assertEqual(model.generated(1), [21, S, ALLOW[0]])    # prompt 20 -> 21 (free); drafts [S, 22, 23]: S kept
        self.assertEqual(g.xgr.accepted, [ALLOW[0]])               # the marker arms; only what follows it is the call
        with torch.no_grad():
            while not model.decode([1], None, None)[0]:
                pass
        self.assertEqual(model.generated(1), [21, S] + [ALLOW[0]] * 6)
        self.assertEqual(g.xgr.accepted, [ALLOW[0]] * 6)

    def test_the_draft_ahead_verify_never_takes_a_grammar_row(self):
        """fleet --draft-ahead (on by default): behind a step whose rows are all greedy and plain, the picks are the
        gathered logits' argmax on the device, with no mask (ServedModel._verify_ahead). A grammar row -- the tool
        grammar still asleep before its marker included -- keeps the whole step on the masked path."""
        from engine.profiles.qwen38.decode_graphs import DraftGraphs
        model, _, _ = self.bound(draft_ahead=True)
        graphs = SimpleNamespace(tokens=K + 1, k=K)
        graphs.extent = lambda observed: DraftGraphs.extent(graphs, observed)
        model.composition.graphs, model.drafter.graphs = object(), graphs          # as if captured
        tool = {"grammar": {"type": "ebnf", "grammar": "root ::= call"}, "grammar_after": S}
        model.add(0, [11, 12, 13], max_new=8, temperature=0.0, options={})
        model.add(1, [14, 15, 16], max_new=8, temperature=0.0, options=tool)
        model.store.pool.tokens[0] = model.store.pool.tokens[1] = 64             # reserved past the widest draft step

        def step(*seqs):
            return SimpleNamespace(segments=[SimpleNamespace(seq=seq, ctx=3) for seq in seqs])
        self.assertTrue(model._ahead_ready([0], step(0)))           # a greedy plain row: the device's argmax is its pick
        self.assertFalse(model._ahead_ready([1], step(1)))
        self.assertFalse(model._ahead_ready([0, 1], step(0, 1)))


if __name__ == "__main__":
    unittest.main()
