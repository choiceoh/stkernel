"""Decode steps launched ahead of their results (45차 §23 B3): the runner keeps up to `depth` in flight, resolves the
oldest before launching further, drains before a prefill or a synchronous step, ignores a finished row's ghost step,
and never releases a row with a step ahead of it. A fake model stands in for the device."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.base import scheduler as sched  # noqa: E402
from engine.base.kv import BlockPool, SlotPool  # noqa: E402
from engine.base.prefix import PrefixCache  # noqa: E402
from engine.base.record import Ring  # noqa: E402
from engine.base.runner import Runner, STEP_RECORD  # noqa: E402
from engine.base.scheduler import Contract  # noqa: E402


class Pending:
    def __init__(self, model, seqs):
        self.model, self.seqs, self.resolved = model, seqs, False

    def resolve(self):
        self.resolved = True
        self.model.log.append(("resolve", tuple(self.seqs)))
        out = []
        for seq in self.seqs:
            if not self.model.alive.get(seq, False):           # a ghost: the device made it inert, nothing moves
                out.append(True)
                continue
            self.model.ctx[seq] += 1
            self.model.left[seq] -= 1
            done = self.model.left[seq] == 0
            if done:
                self.model.alive[seq] = False
            out.append(done)
        return out


class Model:
    """Sync prefill/decode as the runner self-check's fake; decode_async launches now, moves state at resolve."""
    def __init__(self, ready=True):
        self.log, self.left, self.ctx, self.alive = [], {}, {}, {}
        self.ready, self.ahead = ready, {}

    def open(self, seq, slot):
        self.left[seq] = 3; self.alive[seq] = True

    def close(self, seq):
        self.left.pop(seq, None); self.alive.pop(seq, None)

    def horizon(self, seq):
        return self.ctx[seq] + 1 + self.ahead.get(seq, 0)

    def context(self, seq):
        return self.ctx[seq]

    def prefill(self, seq, start, tokens, blocks, slot, marks=None):
        self.log.append(("prefill", seq, start, tokens)); self.ctx[seq] = start + tokens

    def decode(self, seqs, blocks, slots):
        self.log.append(("decode", tuple(seqs)))
        out = []
        for s in seqs:
            self.ctx[s] += 1; self.left[s] -= 1; out.append(self.left[s] == 0)
        return out

    def async_ready(self, seqs):
        return self.ready

    def decode_async(self, seqs, blocks, slots):
        self.log.append(("launch", tuple(seqs)))
        return Pending(self, list(seqs))


def runner(model=None, blocks=64):
    c = Contract(chunk_align=16, token_budget=64, draft_slots=0, max_wait_s=20.0, max_running=8)
    return Runner(model or Model(), c, BlockPool(blocks, 16, 8, 32), SlotPool(9), Ring(64, STEP_RECORD.size))


class AsyncRunnerTests(unittest.TestCase):
    def test_parked_row_ghost_does_not_read_the_closed_context(self):
        class ParkingModel(Model):
            def open(self, seq, slot):
                super().open(seq, slot)
                self.left[seq] = 100

            def history(self, seq):
                return list(range(seq * 100, seq * 100 + self.ctx[seq]))

            def checkpoint(self, seq, position, snap):
                self.log.append(("checkpoint", seq, position))

            def park(self, seq):
                record = dict(context=self.ctx.pop(seq), pending=0)
                self.close(seq)
                return record

            def state_bytes(self, slot):
                return b"state"

        m = ParkingModel()
        r = runner(m)
        r.keep_idle = True
        r.prefix = PrefixCache(16, 64, 8)
        r.prefix.bind(r.kv)
        parked = []
        r.tiered = SimpleNamespace(
            is_parked=lambda key: False,
            park_begin=lambda seq, key, **kw: parked.append((seq, key, kw)))
        for seq in range(4):
            ids = list(range(seq * 100, seq * 100 + 15))
            r.submit(seq, len(ids), now=0, ids=ids)
        while r.state.waiting or r.state.in_prefill is not None:
            r.step(now=25)
        m.left[0] = 1
        r.step(now=100); r.step(now=100)
        r.resolve_oldest()                                  # row 0 ends; a four-row ghost still runs ahead
        self.assertIn(0, r.idle)
        survivors = [m.ctx[s] for s in (1, 2, 3)]
        checkpoints = [c for c in m.log if c[0] == "checkpoint" and c[1] == 0]
        r.park_begin(0, key=99)                             # closes model context before disk releases the slot
        self.assertNotIn(0, m.ctx)
        self.assertIn(0, r.slot_of)
        self.assertIn(0, r._chain)
        r.resolve_oldest()
        self.assertEqual([m.ctx[s] for s in (1, 2, 3)], [ctx + 1 for ctx in survivors])
        self.assertEqual([c for c in m.log if c[0] == "checkpoint" and c[1] == 0],
                         checkpoints)
        self.assertEqual(parked[0][:2], (0, 99))
        self.assertFalse(r.inflight)
        r.prefix.check()

    def test_cancel_does_not_wait_for_later_batches_without_that_row(self):
        r = runner()
        r.keep_idle = True
        r.submit(1, 16, now=0); r.submit(2, 16, now=0)
        r.step(now=0); r.step(now=25)
        r.model.left[1] = 1
        r.step(now=25)
        r.resolve_oldest()                                 # row 1 leaves before the next batch
        r.step(now=25)
        before = list(r.inflight)
        r.cancel(1)
        self.assertEqual(r.inflight, before)
        self.assertEqual(r.inflight[0][0].seqs, (2,))
        r.settle()

    def test_reused_row_lands_its_old_ghost_before_the_new_request_is_opened(self):
        r = runner()
        r.submit(1, 16, now=0); r.step(now=0)
        r.model.left[1] = 1
        r.step(now=0); r.step(now=0)
        r.resolve_oldest()                                 # first step completes, second is an inert ghost
        self.assertTrue(r.inflight)
        r.submit(1, 16, now=1)
        self.assertFalse(r.inflight)
        r.step(now=1); r.step(now=1); r.drain()
        self.assertEqual(r.model.left[1], 2)

    def test_two_steps_run_ahead_and_the_third_launch_resolves_the_first(self):
        r = runner()
        r.submit(1, 20, now=0.0)
        self.assertEqual(r.step(now=0.0).kind, "prefill")
        for _ in range(2):
            self.assertEqual(r.step(now=0.0).kind, "decode")
        self.assertEqual(len(r.inflight), 2)
        self.assertEqual([e for e in r.model.log if e[0] in ("launch", "resolve")], [("launch", (1,)), ("launch", (1,))])
        r.step(now=0.0)                                            # depth reached: the oldest lands first
        self.assertEqual([e for e in r.model.log if e[0] in ("launch", "resolve")][2:], [("resolve", (1,)), ("launch", (1,))])
        self.assertEqual(len(r.inflight), 2)
        self.assertEqual(r.async_steps, 3)

    def test_a_finished_row_leaves_when_its_result_lands_and_its_ghost_step_is_ignored(self):
        r = runner()
        r.submit(1, 20, now=0.0)
        r.step(now=0.0)                                            # prefill
        kinds = []
        while (s := r.step(now=0.0)) is not None:
            kinds.append(s.kind)
        self.assertEqual(kinds, ["decode"] * 4)                    # three real steps and one ghost, launched before the third landed
        self.assertEqual(r.inflight, [])
        self.assertEqual(r.state.running, [])
        self.assertEqual(r.kv.available, 64)                       # the row is gone, blocks and slot returned
        self.assertEqual(r.slots.available, 8)
        self.assertEqual(r.ring.count, 5)                          # every step has its record, written when it landed

    def test_a_prefill_drains_the_steps_ahead_and_a_synchronous_decode_does_too(self):
        r = runner()
        r.submit(1, 20, now=0.0)
        r.step(now=0.0)
        r.step(now=0.0); r.step(now=0.0)                           # two decodes in flight
        r.submit(2, 20, now=0.0)
        s = r.step(now=25.0)                                       # the valve opens: a prefill needs everything landed
        self.assertEqual(s.kind, "prefill")
        self.assertEqual(r.inflight, [])
        r.model.ready = False                                      # rows that cannot run ahead: the plain path, after draining
        r.step(now=25.0)
        self.assertEqual(r.model.log[-1][0], "decode")
        self.assertEqual(r.inflight, [])

    def test_cancel_and_settle_land_the_steps_ahead_first(self):
        r = runner()
        r.submit(1, 20, now=0.0)
        r.step(now=0.0); r.step(now=0.0); r.step(now=0.0)
        self.assertEqual(len(r.inflight), 2)
        r.cancel(1)
        self.assertEqual(r.inflight, [])
        self.assertEqual(r.model.log[-2:], [("resolve", (1,)), ("resolve", (1,))])
        self.assertEqual(r.kv.available, 64)
        r.submit(3, 20, now=0.0); r.step(now=0.0); r.step(now=0.0)
        r.settle()
        self.assertEqual(r.inflight, [])

    def test_horizons_cover_the_steps_ahead(self):
        m = Model()
        r = runner(m)
        r.submit(1, 30, now=0.0)
        r.step(now=0.0)
        m.ahead[1] = 2                                             # the model says: two steps of growth may be pending
        r.step(now=0.0)
        self.assertGreaterEqual(r.kv.tokens[1], 30 + 3)


if __name__ == "__main__":
    unittest.main()
