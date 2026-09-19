"""engine/base/composed: pictures through the composed model (the module docstring's media hooks) -- model-free, over a
composition that declares it sees them and records what it is handed: the records kept with absolute positions (a
continued turn's re-based past the history), checked before the row changes and adopted after, handed to every prefill
piece of their row and no other, marks for the prefix cache and the door, marks and grids through a park, dropped with
the row. A composition that does not see pictures serves text only, as before.
"""
import unittest

from tests.test_engine_composed import prompt, tiny_composition, torch

IMG = 7


class Seeing:
    """A composition that sees pictures: the tiny Qwen3.8 composition's arithmetic, every media hook recorded."""
    sees_media = True

    def __init__(self, comp, refuse=False):
        self.comp, self.refuse, self.calls = comp, refuse, []

    def __getattr__(self, name):
        return getattr(self.comp, name)

    def check_media(self, ids, records):
        self.calls.append(("check", len(ids), [r["positions"] for r in records]))
        if self.refuse:
            raise ValueError("refused")

    def bind_media(self, seq, ids, records):
        self.calls.append(("bind", seq, len(ids), [(r["positions"], r["canvas"] is not None) for r in records]))

    def forget_media(self, seq):
        self.calls.append(("forget", seq))

    def forward(self, step, state, **kw):
        media = kw.pop("media", None)
        self.calls.append(("forward", tuple(s.seq for s in step.segments), None if media is None else
                           [r["positions"] for r in media]))
        return self.comp.forward(step, state, **kw)


def record(positions, digest="d", grid=(1, 2, 2)):
    return {"kind": "image", "digest": digest, "positions": list(positions), "canvas": object(), "grid": grid}


@unittest.skipUnless(torch is not None, "requires torch")
class ComposedMediaTests(unittest.TestCase):
    def build(self, *, seeing=True, refuse=False, rows=2):
        from engine.base.composed import ComposedModel, store_for
        from engine.base.record import Ring
        from engine.base.runner import STEP_RECORD, Runner
        from engine.base.scheduler import Contract
        comp, cfg = tiny_composition(0)
        comp = Seeing(comp, refuse) if seeing else comp
        store, pool, slots, _ = store_for(comp, kv_gib=0.02, max_seqs=rows, block_tokens=4)
        model = ComposedModel(comp, store, vocab=cfg["vocab_size"], eos_ids=[cfg["eos_token_id"]], max_new=3, temperature=0.0)
        runner = Runner(model, Contract(chunk_align=4, token_budget=8, draft_slots=0, max_wait_s=0.0, max_running=rows),
                        pool, slots, Ring(64, STEP_RECORD.size), keep_idle=True)
        return comp, model, runner

    def ids_with_picture(self, seed, length, at):
        ids = prompt(seed, length)
        for p in at:
            ids[p] = IMG
        return ids

    def drain(self, runner, limit=100):
        for _ in range(limit):
            if runner.step(now=0.0) is None:
                return

    def test_a_text_only_composition_serves_text_only(self):
        _, model, _ = self.build(seeing=False)
        with self.assertRaisesRegex(ValueError, "text only"):
            model.add(0, self.ids_with_picture(1, 9, (2, 3, 4, 5)), media=[record(range(2, 6))])
        self.assertNotIn(0, model.tokens)

    def test_records_are_checked_then_adopted_and_ride_their_row_s_prefill(self):
        comp, model, runner = self.build()
        a = self.ids_with_picture(1, 13, (3, 4, 5, 6))
        b = prompt(2, 6)
        model.add(0, a, media=[record(range(3, 7))])
        model.add(1, b)
        self.assertEqual([c[0] for c in comp.calls], ["check", "bind"])
        self.assertEqual(model.media_marks(0), [(3, "d")])
        self.assertEqual(model.media_marks(1), [])
        runner.submit(0, len(a), now=0.0)
        runner.submit(1, len(b), now=0.0)
        self.drain(runner)
        forwards = [c for c in comp.calls if c[0] == "forward"]
        with_row0 = [c for c in forwards if c[1] == (0,) and c[2] is not None]
        self.assertTrue(with_row0 and all(c[2] == [[3, 4, 5, 6]] for c in with_row0))   # the prefill pieces of row 0
        self.assertTrue(all(c[2] is None for c in forwards if 1 in c[1]))                    # never another row's
        self.assertTrue(all(c[2] is None for c in forwards if len(c[1]) > 1))                # nor a decode step

    def test_a_continued_turn_s_pictures_are_re_based_and_a_refusal_leaves_the_row(self):
        comp, model, runner = self.build(rows=1)
        a = self.ids_with_picture(1, 9, (2, 3, 4, 5))
        model.add(0, a, media=[record(range(2, 6))])
        runner.submit(0, len(a), now=0.0)
        self.drain(runner)
        history = list(model.history_ref(0))
        tail = self.ids_with_picture(3, 8, (1, 2, 3, 4))
        model.extend(0, tail, media=[record(range(1, 5), digest="e")])
        base = len(history)
        self.assertEqual(model.media_marks(0), [(2, "d"), (base + 1, "e")])
        check, bind = comp.calls[-2:]
        self.assertEqual(check, ("check", base + len(tail), [[2, 3, 4, 5], list(range(base + 1, base + 5))]))
        self.assertEqual(bind[0:3], ("bind", 0, base + len(tail)))
        # a refused turn: the row keeps its tokens and its pictures
        comp.refuse = True
        before, marks = list(model.history_ref(0)), model.media_marks(0)
        with self.assertRaisesRegex(ValueError, "refused"):
            model.extend(0, self.ids_with_picture(4, 6, (1, 2, 3, 4)), media=[record(range(1, 5), digest="f")])
        self.assertEqual((list(model.history_ref(0)), model.media_marks(0)), (before, marks))

    def test_a_park_keeps_marks_and_grids_and_a_forget_drops_them(self):
        comp, model, runner = self.build(rows=1)
        a = self.ids_with_picture(1, 9, (2, 3, 4, 5))
        model.add(0, a, media=[record(range(2, 6), grid=(1, 2, 2))])
        runner.submit(0, len(a), now=0.0)
        self.drain(runner)
        slot = model.store.slot_of[0]
        parked = model.park(0)
        self.assertEqual(parked["media"], [["image", "d", 2, 4, [1, 2, 2]]])
        self.assertIn(("forget", 0), comp.calls)
        self.assertEqual(model.media_marks(0), [])
        model.resume(0, slot, parked)
        self.assertEqual(model.media_marks(0), [(2, "d")])
        self.assertEqual(comp.calls[-1], ("bind", 0, len(parked["tokens"]), [([2, 3, 4, 5], False)]))   # no canvas
        model.close(0)
        model.forget(0)
        self.assertEqual(comp.calls[-1], ("forget", 0))


if __name__ == "__main__":
    unittest.main()
