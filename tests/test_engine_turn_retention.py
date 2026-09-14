"""A turn nobody will continue is released when it finishes, not kept or parked (45차, 2026-09-13).

The supervisor's health check -- "ping", four tokens -- was parked to the NVMe tier every thirty
seconds: a slot's whole recurrent state, 256 MiB a rank, for a 17-token history, and each park
pushed a real conversation further down the tier's LRU. A request can now say `retain: false`,
and a server can set `park_min_tokens` below which a finished turn is released. Both inputs are
replicated (the options ride the step's broadcast, the context is the row's), so every rank
releases the same turn; a four-rank run pins that too.
"""
import ast
import importlib.util
import sys
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import test_engine_serve as T                                          # noqa: E402
from test_engine_tier import MemoryTier                                # noqa: E402
from engine.base.serve import RequestError, sampling_options           # noqa: E402

TRANSIENT = {"_transient": True}


def settle(s, steps=300):
    for _ in range(steps):
        ran = s.once()
        if not ran and not s._waiting and not s._retiring and not s._resuming and not s._restoring:
            return
        threading.Event().wait(0.001)
    raise AssertionError("the server did not settle")


class RetentionTests(unittest.TestCase):
    def test_a_turn_that_asks_not_to_be_retained_is_released_and_the_next_one_is_parked(self):
        tier = MemoryTier()
        s = T.server(rows=2, keep_idle=True, tier=tier)
        ping, _ = s.submit([3], 1, 0, options=dict(TRANSIENT))
        settle(s)
        self.assertEqual(s.take_result(ping), [3], "the answer is unchanged")
        self.assertEqual(tier.keys(), [], "nothing was written to the tier")
        self.assertEqual(sorted(s._free_rows), [0, 1])
        self.assertFalse(s._conversations or s._conversation_of or s._transient)
        self.assertEqual(s.turns_not_retained, {"asked": 1})
        chat, _ = s.submit([5], 1, 0)
        settle(s)
        self.assertEqual(s.take_result(chat), [5])
        self.assertEqual(tier.keys(), [chat], "an ordinary turn is still parked")
        self.assertIn('st:turns_not_retained_total{engine="st",reason="asked"} 1', s.metrics())

    def test_a_turn_shorter_than_the_floor_is_released_and_one_at_the_floor_is_parked(self):
        tier = MemoryTier()
        s = T.server(rows=2, keep_idle=True, tier=tier)
        s.park_min_tokens = 4
        short, _ = s.submit([3], 1, 0)
        settle(s)
        self.assertEqual((tier.keys(), s.turns_not_retained), ([], {"short": 1}))
        long, _ = s.submit([1, 2, 3, 4], 1, 0)
        settle(s)
        self.assertEqual(tier.keys(), [long])
        self.assertEqual(s.turns_not_retained, {"short": 1})
        self.assertEqual((s.take_result(short), s.take_result(long)), ([3], [4]))

    def test_without_a_tier_the_released_turn_does_not_stay_resident_either(self):
        s = T.server(rows=2, keep_idle=True)
        ping, _ = s.submit([3], 1, 0, options=dict(TRANSIENT))
        settle(s)
        self.assertEqual(sorted(s._free_rows), [0, 1])
        self.assertFalse(s._idle_order or s._conversations)
        kept, _ = s.submit([5], 1, 0)
        settle(s)
        self.assertEqual(list(s._idle_order), [s._conversations[kept]], "an ordinary turn stays resident")

    def test_a_released_conversation_cannot_be_continued_by_id(self):
        s = T.server(rows=2, keep_idle=True, tier=MemoryTier())
        ping, _ = s.submit([3], 1, 0, options=dict(TRANSIENT))
        settle(s)
        s.take_result(ping)
        turn, _ = s.submit([4], 1, 0, conversation=ping)
        settle(s)
        with self.assertRaises(RequestError) as caught:
            s.take_result(turn)
        self.assertEqual(caught.exception.status, 409)

    def test_no_floor_and_no_request_keeps_every_turn_as_before(self):
        tier = MemoryTier()
        s = T.server(rows=2, keep_idle=True, tier=tier)
        first, _ = s.submit([3], 1, 0)
        settle(s)
        self.assertEqual((tier.keys(), s.turns_not_retained), ([first], {}))
        fresh = T.server(rows=2)
        from engine.base.serve import Server
        with self.assertRaisesRegex(ValueError, "park_min_tokens must be a nonnegative integer"):
            Server(fresh.engine, fresh.runner, T.Comm(), host="127.0.0.1", port=0, park_min_tokens=-1)


class DoorTests(unittest.TestCase):
    def test_retain_false_becomes_the_internal_marker_and_anything_else_is_refused_or_ignored(self):
        self.assertEqual(sampling_options({"retain": False})[1], {"_transient": True})
        self.assertEqual(sampling_options({"retain": True})[1], {})
        self.assertEqual(sampling_options({})[1], {})
        for bad in ("no", 0, 1.0):
            with self.assertRaisesRegex(RequestError, "retain must be a boolean"):
                sampling_options({"retain": bad})

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "the sampler module imports torch")
    def test_the_engine_accepts_the_marker_only_as_a_boolean(self):
        from engine.base.sampler import OPTION_KEYS, validate_options
        self.assertIn("_transient", OPTION_KEYS)
        validate_options({"_transient": True})
        with self.assertRaisesRegex(ValueError, "not-retained marker must be boolean"):
            validate_options({"_transient": 1})


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "requires PyTorch for LocalTP")
class FourRankTests(unittest.TestCase):
    def test_every_rank_releases_the_same_turns(self):
        from engine.base.comm import LocalTP

        def rank_main(comm, _):
            tier = MemoryTier(delay=0.002 * (comm.rank + 1))
            s = T.server(comm=comm, rows=2, keep_idle=True, tier=tier)
            s.park_min_tokens = 3
            if comm.rank == 0:
                s.submit([3], 1, 0, options=dict(TRANSIENT))       # asked
                s.submit([7], 1, 0)                                # short
                s.submit([1, 2, 3, 4], 1, 0)                       # kept
            for _ in range(200):
                s.once()
                threading.Event().wait(0.002)
            if comm.rank == 0:
                s.alive = False
            s.once()
            return tier.keys(), dict(s.turns_not_retained), sorted(s._free_rows)
        out = LocalTP(4).run(rank_main, None)
        self.assertTrue(all(row == out[0] for row in out), out)
        self.assertEqual(out[0][1], {"asked": 1, "short": 1})
        self.assertEqual(len(out[0][0]), 1, "only the long turn was parked, on every rank")


class ContractTests(unittest.TestCase):
    def test_the_production_boot_sets_the_floor_and_the_supervisor_ping_asks(self):
        boot = (ROOT / "engine/profiles/glm53/boot.py").read_text()
        self.assertIn("PARK_MIN_TOKENS = 128", boot)
        server = next(n for n in ast.walk(ast.parse(boot)) if isinstance(n, ast.Call)
                      and isinstance(n.func, ast.Name) and n.func.id == 'Server')
        keywords = {k.arg: ast.unparse(k.value) for k in server.keywords}
        self.assertEqual(keywords['lease'], 'lease')
        self.assertEqual(keywords['park_min_tokens'], 'PARK_MIN_TOKENS')
        supervisor = (ROOT / "launchers/st-glm53-supervisor.sh").read_text()
        body = supervisor[supervisor.index("chat_ok(){"):supervisor.index("}", supervisor.index("chat_ok(){") + 200)]
        self.assertIn('\\"retain\\":false', body)

    def test_both_retire_sites_pass_the_request(self):
        serve = (ROOT / "engine/base/serve.py").read_text()
        self.assertEqual(serve.count("self._retire(row, request)"), 2)
        self.assertNotIn("self._retire(row)\n", serve)


if __name__ == "__main__":
    unittest.main()
