"""Ranks that meet at different collectives die saying so, on every rank alike (45차, 2026-09-13).

Four ranks run here as four threads over a fake control group that sums what they post, the way
gloo does. The contract under test: a tagged vote or exchange completes even when the ranks stand
at different sites (fixed length), every rank then reads the same table, and every rank raises the
same `CollectiveDivergence` naming who stood where -- nobody waits. The serving loop's use of it,
the boot's, the engine's and the fail-fast paths around the boot votes are pinned in the sources.
"""
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.base.tripwire import (SLOTS, TAG, CollectiveDivergence, Tripwire, classify, death_note,   # noqa: E402
                                  pack, peers_agree, site_id, unpack)


class Board:
    """A control group for threads: every rank posts its vector, the sum comes back to all."""

    def __init__(self, world, timeout=5.0):
        self.world, self.timeout = world, timeout
        self.lock = threading.Lock()
        self.posted = {}
        self.arrive = threading.Barrier(world, timeout=timeout)
        self.leave = threading.Barrier(world, timeout=timeout)
        self.objects = {}

    def all_reduce(self, rank, vector):
        with self.lock:
            self.posted[rank] = list(vector)
        self.arrive.wait()
        out = [sum(col) for col in zip(*(self.posted[r] for r in range(self.world)))]
        self.leave.wait()
        return out

    def all_gather(self, rank, obj):
        with self.lock:
            self.objects[rank] = obj
        self.arrive.wait()
        out = [self.objects[r] for r in range(self.world)]
        self.leave.wait()
        return out


class Rank:
    def __init__(self, board, rank):
        self.board, self.rank, self.world_size = board, rank, board.world

    def all_reduce_host(self, values):
        return self.board.all_reduce(self.rank, values)

    def gather_objects(self, obj):
        return self.board.all_gather(self.rank, obj)


def run(world, body, timeout=5.0):
    """Every rank runs `body(wire, rank)`; returns each rank's result or exception."""
    board = Board(world, timeout)
    out = [None] * world

    def main(r):
        wire = Tripwire.of(Rank(board, r))
        try:
            out[r] = ("ok", body(wire, r))
        except BaseException as exc:                     # noqa: BLE001 -- the exception is the result under test
            out[r] = ("raised", exc)
    threads = [threading.Thread(target=main, args=(r,), daemon=True) for r in range(world)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout * 4)
    return out


class TripwireTests(unittest.TestCase):
    def test_votes_sum_and_exchanges_keep_every_ranks_values_by_rank(self):
        def body(wire, r):
            sums = wire.vote("settle:done", [1, r % 2])
            rows = wire.exchange("gather:rows", [10 + r, 100 + r])
            agreed = wire.agree("boot:seed", [7])
            return sums, rows, agreed, wire.calls
        for status, result in run(4, body):
            self.assertEqual(status, "ok", result)
            sums, rows, agreed, calls = result
            self.assertEqual(sums, [4, 2])
            self.assertEqual(rows, [[10, 100], [11, 101], [12, 102], [13, 103]])
            self.assertEqual(agreed, [7])
            self.assertEqual(calls, 3, "every rank counted the same three")

    def test_ranks_at_different_sites_all_raise_the_same_attributed_error_and_none_waits(self):
        def body(wire, r):
            if r == 2:
                wire.vote("admit:fits", [1])            # one branch the others did not take
            return wire.vote("settle:done", [1])
        results = run(4, body)
        messages = set()
        for status, exc in results:
            self.assertEqual(status, "raised", exc)
            self.assertIsInstance(exc, CollectiveDivergence)
            messages.add(str(exc).split("\n", 1)[1])                       # the table: the same on every rank
            self.assertIn("rank2: #1 'admit:fits' (vote, 1 values)", str(exc))
            self.assertIn("rank0: #1 'settle:done' (vote, 1 values)", str(exc))
        self.assertEqual(len(messages), 1, "the same table on every rank")

    def test_a_rank_that_skipped_a_collective_is_one_out_of_step_and_the_sequence_says_so(self):
        def body(wire, r):
            if r != 3:
                wire.vote("settle:done", [1])           # rank 3 returned early before this one
            return wire.vote("settle:outcome", [1, 0])
        for status, exc in run(4, body):
            self.assertEqual(status, "raised", exc)                    # at the FIRST pairing, not a step later
            self.assertIn("rank3: #1 'settle:outcome' (vote, 2 values)", str(exc))
            self.assertIn("rank0: #1 'settle:done' (vote, 1 values)", str(exc))

    def test_disagreement_on_an_agreed_value_names_every_rank(self):
        def body(wire, r):
            return wire.agree("boot:seed", [0 if r < 3 else 1])
        for status, exc in run(4, body):
            self.assertEqual(status, "raised")
            self.assertIn("disagree at 'boot:seed': rank0=[0], rank1=[0], rank2=[0], rank3=[1]", str(exc))

    def test_the_step_broadcast_is_stamped_by_rank_0_and_checked_by_the_followers(self):
        wire0, wire2 = Tripwire.of(Rank(Board(1), 0)), Tripwire.of(Rank(Board(1), 2))
        stamp = wire0.stamp()
        wire2.expect(stamp)                                                     # in step
        self.assertEqual((wire0.calls, wire2.calls), (1, 1))
        wire2.calls += 1                                                        # the follower made one more collective
        with self.assertRaises(CollectiveDivergence) as caught:
            wire2.expect(wire0.stamp())
        self.assertIn("rank 2 is at collective #3 ('step') but rank 0 broadcast #2", str(caught.exception))
        with self.assertRaises(CollectiveDivergence):
            Tripwire.of(Rank(Board(1), 1)).expect(None)

    def test_world_one_costs_nothing_and_a_fixed_shape_has_a_limit(self):
        wire = Tripwire.of(Rank(Board(1), 0))
        self.assertEqual(wire.vote("x", [1, 0, 1]), [1, 0, 1])
        self.assertEqual(wire.exchange("x", [5]), [[5]])
        self.assertEqual(wire.calls, 0)
        with self.assertRaises(ValueError):
            pack(4, 0, 1, 2, [0] * (SLOTS + 1), exchange=False)
        with self.assertRaises(ValueError):
            pack(4, 0, 1, 2, [0] * (SLOTS // 4 + 1), exchange=True)
        with self.assertRaises(ValueError):
            unpack([0] * 3, 4)
        self.assertEqual(len(pack(4, 1, 9, site_id("s"), [3, 4], True)), TAG * 4 + SLOTS)
        tags, region = unpack(pack(4, 1, 9, site_id("s"), [3, 4], True), 4)
        self.assertEqual(tags[1], (9, site_id("s"), 2, 1))
        self.assertEqual(region[16:18], [3, 4], "an exchange puts rank 1's values in its block")

    def test_peers_agree_models_a_fleet_that_stands_where_this_rank_stands(self):
        vote = pack(2, 0, 4, site_id("settle:done"), [1, 0], False)
        tags, region = unpack(peers_agree(vote, 2), 2)
        self.assertEqual(tags, [(4, site_id("settle:done"), 2, 0)] * 2)
        self.assertEqual(region[:2], [2, 0])
        ex = pack(2, 0, 4, site_id("gather:rows"), [7], True)
        tags, region = unpack(peers_agree(ex, 2), 2)
        self.assertEqual(tags, [(4, site_id("gather:rows"), 1, 1)] * 2)
        self.assertEqual((region[0], region[SLOTS // 2]), (7, 7))

    def test_one_tripwire_per_comm_so_the_server_and_the_engine_share_a_sequence(self):
        comm = Rank(Board(1), 0)
        self.assertIs(Tripwire.of(comm), Tripwire.of(comm))
        self.assertIsNot(Tripwire.of(comm), Tripwire.of(Rank(Board(1), 0)))


class DeathNoteTests(unittest.TestCase):
    def test_a_death_is_classified_and_written_where_the_container_cannot_erase_it(self):
        self.assertEqual(classify(CollectiveDivergence("x")), "divergence")
        self.assertEqual(classify(RuntimeError("[c10d] recvValue failed ... Connection closed by peer")), "peer-left")
        self.assertEqual(classify(RuntimeError("NCCL error: unhandled system error")), "peer-left")
        self.assertEqual(classify(ValueError("checkpoint/rank weight layout mismatch")), "local")
        said = []
        with tempfile.TemporaryDirectory() as tmp:
            note = death_note(tmp, 2, RuntimeError("Connection reset by peer"), phase="step 2299", calls=(5518, "settle:done"),
                              say=lambda *a, **k: said.append(a[0]))
            self.assertEqual((note["rank"], note["kind"], note["phase"], note["calls"]), (2, "peer-left", "step 2299", (5518, "settle:done")))
            saved = json.loads(Path(note["path"]).read_text())
            self.assertEqual(saved["kind"], "peer-left")
            self.assertIn("a peer died or stalled first", saved["meaning"])
            self.assertTrue(Path(note["path"]).name.startswith("death-rank2-"))
        self.assertIn("[serve] death rank=2 kind=peer-left phase='step 2299'", said[0])
        self.assertEqual(death_note(None, 0, ValueError("x"), say=lambda *a, **k: None)["kind"], "local")


class ContractTests(unittest.TestCase):
    def test_the_serving_loop_votes_through_the_tripwire_and_stamps_its_broadcast(self):
        serve = (ROOT / "engine/base/serve.py").read_text()
        self.assertIn("self.tripwire = Tripwire.of(comm)", serve)
        self.assertIn('return self.tripwire.vote(site, [int(bool(f)) for f in flags])', serve)
        self.assertNotIn("return self.comm.all_reduce_host([int(bool(f)) for f in flags])", serve)
        for site in ("settle:done", "settle:outcome", "admit:restore", "admit:fits"):
            self.assertIn(f'"{site}"', serve, site)
        once = serve[serve.index("    def once(self) -> bool:"):serve.index("    def _death_note(self")]
        self.assertIn("self._yield_asked(), self.tripwire.stamp()) if self.comm.rank == 0 else None)", once)
        self.assertIn("if self.comm.rank != 0:\n                self.tripwire.expect(stamp)", once)
        self.assertIn("self._death_note(exc)", once)

    def test_the_engine_gather_and_the_boot_seed_ride_the_same_wire(self):
        adapter = (ROOT / "engine/profiles/glm53/adapter.py").read_text()
        self.assertIn('Tripwire.of(comm).exchange("gather:rows", [rows])', adapter)
        self.assertIn('Tripwire.of(comm).exchange("gather:detail", flat)', adapter)
        self.assertNotIn("for seq, span, digest in (detail or ()):\n            spans = comm.all_reduce_host", adapter)
        boot = (ROOT / "engine/profiles/glm53/boot.py").read_text()
        self.assertIn('Tripwire.of(comm).agree("boot:seed", [int(a.seed)])', boot)

    def test_a_rank_that_fails_before_a_boot_vote_still_casts_it(self):
        memory = (ROOT / "engine/base/runtime_memory.py").read_text()
        self.assertIn('def checkpoint(self, phase, failed: "str | None" = None, *, release_cache: bool = False):', memory)
        self.assertIn("except Exception as exc:                    # noqa: BLE001 -- the vote below must still be cast", memory)
        graphs = (ROOT / "engine/base/graphs.py").read_text()
        self.assertIn('memory.checkpoint(f"{label}/failed", failed=f"{type(exc).__name__}: {str(exc)[:300]}")', graphs)
        adapter = (ROOT / "engine/profiles/glm53/adapter.py").read_text()
        self.assertIn('self.memory.checkpoint("capture_decode/failed", failed=', adapter)
        boot = (ROOT / "engine/profiles/glm53/boot.py").read_text()
        self.assertIn('comm.wait_prepared(f"failed: rank {comm.rank}: {type(exc).__name__}: {str(exc)[:200]}", timeout_s=60., final=True)', boot)
        self.assertIn('memory.checkpoint("boot/failed", failed=', boot)
        self.assertIn('death_note(getattr(a, "dump_dir", None), comm.rank, exc, phase="boot")', boot)
        self.assertLess(boot.index("serving = True"), boot.index("server.loop()"))

    def test_the_one_shot_lane_is_chosen_from_what_every_rank_shares(self):
        comm = (ROOT / "engine/base/comm.py").read_text()
        self.assertIn("def _settled(t):", comm)
        self.assertIn("if t.numel() and t.data_ptr() % 16:\n            t = t.clone()", comm)
        for name in ("def all_reduce(self, t):", "def all_reduce_max(self, t):"):
            body = comm[comm.index(name):]
            body = body[:body.index("\n    def ", 10)]
            self.assertIn("t = self._settled(t)", body, name)
            self.assertLess(body.index("t = self._settled(t)"), body.index("eligible"), name)


if __name__ == "__main__":
    unittest.main()
