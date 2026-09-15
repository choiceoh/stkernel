"""The door's host half runs beside the load, and it is joined before the capture.

`qualify grammar` cost rank 0 6.75 and 10.18 s on the two warm boots of main `3acae017`, and the `door`
row a second transformers tokenizer on top of it. None of that work reads anything the engine produces,
so it belongs on a thread under `load` -- with two lines it must not cross: no CUDA on the thread, and no
thread still running when the capture starts, because capture spends the GIL (boot-time study 5-h).
"""
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "requires PyTorch")
class PreludeTests(unittest.TestCase):
    def boot(self):
        from engine.profiles.glm53 import boot
        return boot

    def wired(self, boot, *, vocab=128, stop=(1, 2), fail=None, grammars=None):
        """Patch the four checkpoint readers the prelude calls; record what it asked them for."""
        seen = {}

        def built(ckpt, v, device=None, stop_token_ids=None, tokenizer=None):
            seen.update(ckpt=ckpt, vocab=v, device=device, stop=stop_token_ids, tokenizer=tokenizer)
            if fail is not None:
                raise fail
            return grammars

        return seen, patch.multiple(
            boot, tokenizer=lambda ckpt: NS(name="tok", ckpt=ckpt), eos_ids=lambda ckpt: list(stop),
            grammars=built, chat_renderer=lambda ckpt: NS(name="renderer"),
            facts=NS(load=lambda ckpt: NS(vocab=vocab), CKPT="/meta"))

    def test_the_thread_builds_the_compiler_and_the_renderer_and_never_asks_for_a_device(self):
        boot = self.boot()
        compiled = NS(name="grammars")
        seen, patched = self.wired(boot, grammars=compiled)
        with patched:
            prelude = boot.Prelude("/meta", renderer=True).start()
            tok, built, renderer = prelude.take(128, {1, 2})
        self.assertEqual((tok.name, built, renderer.name), ("tok", compiled, "renderer"))
        self.assertIsNone(seen["device"], "the mask kernel is qualified on the main thread, not here")
        self.assertEqual((seen["ckpt"], seen["vocab"], seen["stop"]), ("/meta", 128, {1, 2}))
        self.assertIs(seen["tokenizer"], tok)             # one tokenizer, read once, shared with the door
        self.assertGreater(prelude.seconds, 0)

    def test_a_rank_that_does_not_render_builds_no_renderer(self):
        boot = self.boot()
        _, patched = self.wired(boot)
        with patched, patch.object(boot, "chat_renderer", side_effect=AssertionError("not this rank")):
            _, _, renderer = boot.Prelude("/meta", renderer=False).start().take(128, {1, 2})
        self.assertIsNone(renderer)

    def test_a_thread_that_raised_raises_where_the_boot_can_vote_on_it(self):
        """The failure has to land on the main thread inside its phase, where the peers' ledger vote is."""
        boot = self.boot()
        _, patched = self.wired(boot, fail=RuntimeError("xgrammar is broken"))
        with patched:
            prelude = boot.Prelude("/meta", renderer=False).start()
            with self.assertRaises(RuntimeError) as caught:
                prelude.take(128, {1, 2})
        self.assertIn("xgrammar is broken", str(caught.exception))

    def test_halves_that_disagree_about_the_vocabulary_or_the_stops_do_not_open_the_door(self):
        boot = self.boot()
        for vocab, stop in ((129, {1, 2}), (128, {1, 3})):
            _, patched = self.wired(boot)
            with patched, self.assertRaises(RuntimeError) as caught:
                boot.Prelude("/meta", renderer=False).start().take(vocab, stop)
            self.assertIn("the boot prelude built for", str(caught.exception))


class WiringTests(unittest.TestCase):
    """Where the prelude starts and where it is joined -- the two facts that make it safe."""

    def setUp(self):
        source = (ROOT / "engine/profiles/glm53/boot.py").read_text(encoding="utf-8")
        self.fleet = source[source.index("def fleet(a)"):]

    def at(self, needle):
        return self.fleet.index(needle)

    def test_it_starts_before_the_load_and_is_joined_before_the_capture(self):
        start = self.at("prelude = Prelude(a.ckpt_meta, renderer=comm.rank == 0).start()")
        self.assertLess(start, self.at("F, net, caches, engine, runner = build("))
        taken = self.at("tok, prebuilt_grammars, renderer = prelude.take(F.vocab, engine.eos)")
        self.assertLess(taken, self.at("engine.capture_decode(MAX_SEQS)"))

    def test_the_main_thread_still_qualifies_the_mask_on_the_device(self):
        self.assertIn("engine.grammars.qualify(caches.device)", self.fleet)
        self.assertNotIn("chat_renderer(a.ckpt_meta)", self.fleet)     # the prelude has it


if __name__ == "__main__":
    unittest.main()
