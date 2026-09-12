"""Serving ends and the box gets its memory back, before the process happens to die.

The arena is one allocation and every weight, block and slot is a VIEW of it, so a
shutdown that drops references frees nothing: at that moment `build`'s caller still
holds `net` and `caches` on the stack. These pin the behaviour that makes the release
decisive instead of hopeful -- the storage goes, whoever is still pointing at it -- and
the one ordering that matters, graphs before the memory they replay over.
"""
import unittest

import torch

from engine.base.arena import GIB, Arena
from engine.profiles.glm53.adapter import Glm53Engine


class ArenaReleaseTests(unittest.TestCase):
    def arena(self, nbytes=1 << 20):
        return Arena(nbytes, device="cpu", expandable=False)

    def test_release_frees_the_storage_while_a_carved_view_is_still_held(self):
        arena = self.arena()
        weight = arena.carve(4096, "weights/L0")            # the shutdown case: someone still holds it
        weight.fill_(7)
        self.assertEqual(weight.untyped_storage().nbytes(), 1 << 20)

        self.assertEqual(arena.release(), 1 << 20)

        self.assertEqual(weight.untyped_storage().nbytes(), 0, "a live view must not keep the arena alive")
        with self.assertRaises(RuntimeError):                # D3: using it afterwards is a mistake, not garbage
            int(weight[0])

    def test_release_is_idempotent_and_reports_only_the_bytes_it_actually_gave_back(self):
        arena = self.arena()
        self.assertEqual(arena.release(), 1 << 20)
        self.assertEqual(arena.release(), 0)

    def test_a_carve_after_release_is_refused_rather_than_allocated(self):
        arena = self.arena()
        arena.carve(1024, "weights/L0")
        arena.release()
        with self.assertRaisesRegex(MemoryError, "released"):
            arena.carve(1024, "weights/L1")

    def test_the_table_still_reads_after_release_without_naming_freed_regions(self):
        arena = self.arena()
        arena.carve(1024, "weights/L0")
        arena.release()
        table = arena.table()
        self.assertIn("free 0.00", table)
        self.assertNotIn("weights/L0", table)


class Stub:
    """A stand-in engine: `release` is unbound, so only the contract is under test."""

    def __init__(self, arena):
        self.arena, self.order = arena, []
        self.sampling_history, self.grammars, self.staged = object(), object(), {1: 2}

    def close_decode(self):
        self.order.append("close_decode")


class EngineReleaseTests(unittest.TestCase):
    def release(self, arena):
        stub = Stub(arena)
        return stub, Glm53Engine.release(stub)

    def test_graphs_close_before_the_memory_they_replay_over_is_freed(self):
        arena = Arena(1 << 20, device="cpu", expandable=False)
        held = arena.carve(4096, "kv")
        stub, report = self.release(arena)
        self.assertEqual(stub.order, ["close_decode"])
        self.assertEqual(held.untyped_storage().nbytes(), 0)
        self.assertEqual(report["arena_bytes"], 1 << 20)

    def test_every_holder_outside_the_arena_is_dropped_too(self):
        stub, _ = self.release(Arena(1 << 20, device="cpu", expandable=False))
        self.assertIsNone(stub.sampling_history)            # penalty tensors
        self.assertIsNone(stub.grammars)                    # xgrammar bitmasks
        self.assertIsNone(stub.arena)
        self.assertEqual(stub.staged, {})

    def test_release_without_an_arena_still_closes_the_graphs_and_says_nothing_came_back(self):
        stub = Stub(None)
        report = Glm53Engine.release(stub)
        self.assertEqual(stub.order, ["close_decode"])
        self.assertEqual(report["arena_bytes"], 0)

    def test_a_second_release_is_harmless(self):
        arena = Arena(1 << 20, device="cpu", expandable=False)
        stub = Stub(arena)
        first = Glm53Engine.release(stub)
        second = Glm53Engine.release(stub)
        self.assertEqual(first["arena_bytes"], 1 << 20)
        self.assertEqual(second["arena_bytes"], 0)
        self.assertEqual(stub.order, ["close_decode", "close_decode"])

    def test_the_report_carries_the_only_honest_proof_the_bytes_landed(self):
        # reserved_before/after is what a workspace nobody freed would show up in: the number
        # is the caller's evidence, not the arena's own claim.
        _, report = self.release(Arena(1 << 20, device="cpu", expandable=False))
        self.assertEqual(set(report), {"arena_bytes", "reserved_before", "reserved_after", "returned"})
        self.assertEqual(report["returned"], report["reserved_before"] - report["reserved_after"])

    @unittest.skipUnless(torch.cuda.is_available(), "the device number is the point")
    def test_on_the_device_the_allocator_actually_hands_the_bytes_back(self):
        torch.cuda.synchronize(); torch.cuda.empty_cache()
        arena = Arena(256 << 20)
        held = arena.carve(64 << 20, "weights")             # still held across the release
        _, report = self.release(arena)
        self.assertEqual(report["arena_bytes"], 256 << 20)
        self.assertGreaterEqual(report["returned"], 256 << 20,
                                f"{report['returned'] / GIB:.3f} GiB came back of 0.250 GiB declared")
        self.assertEqual(held.untyped_storage().nbytes(), 0)


if __name__ == "__main__":
    unittest.main()
