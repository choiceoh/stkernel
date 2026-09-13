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
from engine.base.runtime_memory import live_device_blocks
from engine.profiles.glm53 import boot
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


class Caches:
    def __init__(self):
        self._id_ring = [("pinned", "event")] * 16          # 16 pinned host buffers and their events


class Vision:
    def __init__(self):
        self._rope = {(8, 8): "cos/sin"}                    # one table per distinct picture grid


class Stub:
    """A stand-in engine carrying the real `release`, so only its contract is under test."""

    release = Glm53Engine.release

    def __init__(self, arena, live=()):
        self.arena, self.order = arena, []
        self.caches, self.vision = Caches(), Vision()
        self.tokens = {seq: [1, 2] for seq in live}
        self.slot = {seq: seq for seq in live}
        self.forgotten = []
        self.sampling_history, self.grammars = object(), object()
        self._ids_stage, self._rich_stage = object(), object()
        for name in ("_ends_tensor", "matchers", "embeds", "seeds", "staged", "inflight", "lps", "media"):
            setattr(self, name, {seq: object() for seq in live} or {0: object()})

    def close(self, seq):
        self.order.append(("close", seq))
        self.slot.pop(seq, None)

    def forget(self, seq):
        if seq in self.slot:
            raise ValueError(f"seq {seq} is still live")     # the real engine's guard, so the order is under test
        self.order.append(("forget", seq))
        self.forgotten.append(seq)
        self.tokens.pop(seq, None)

    def close_decode(self):
        self.order.append("close_decode")


class EngineReleaseTests(unittest.TestCase):
    def release(self, arena, live=()):
        stub = Stub(arena, live)
        return stub, Glm53Engine.release(stub)

    def test_graphs_close_before_the_memory_they_replay_over_is_freed(self):
        arena = Arena(1 << 20, device="cpu", expandable=False)
        held = arena.carve(4096, "kv")
        stub, report = self.release(arena)
        self.assertIn("close_decode", stub.order)
        self.assertEqual(held.untyped_storage().nbytes(), 0)
        self.assertEqual(report["arena_bytes"], 1 << 20)

    def test_every_live_request_is_closed_before_it_is_forgotten(self):
        # `forget` refuses a request that still holds a slot, so a release that forgot first
        # would raise here instead of dropping the matcher and the picture rows it was holding.
        stub, _ = self.release(Arena(1 << 20, device="cpu", expandable=False), live=(3, 7))
        self.assertEqual(stub.order[:4], [("close", 3), ("forget", 3), ("close", 7), ("forget", 7)])
        self.assertEqual(stub.forgotten, [3, 7])
        self.assertEqual(stub.tokens, {})

    def test_every_holder_outside_the_arena_is_dropped_too(self):
        stub, _ = self.release(Arena(1 << 20, device="cpu", expandable=False))
        self.assertIsNone(stub.sampling_history)            # penalty tensors
        self.assertIsNone(stub.grammars)                    # xgrammar bitmasks
        self.assertIsNone(stub._ids_stage)                  # pinned upload staging
        self.assertIsNone(stub._rich_stage)                 # the rich sampler's two fp32 planes
        self.assertIsNone(stub.caches._id_ring)             # 16 pinned host buffers
        self.assertEqual(stub.vision._rope, {})             # rope tables, one per picture grid
        self.assertIsNone(stub.arena)
        for name in ("_ends_tensor", "matchers", "embeds", "seeds", "staged", "inflight", "lps", "media"):
            self.assertEqual(getattr(stub, name), {}, name)

    def test_release_without_an_arena_still_closes_the_graphs_and_says_nothing_came_back(self):
        stub = Stub(None)
        report = Glm53Engine.release(stub)
        self.assertIn("close_decode", stub.order)
        self.assertEqual(report["arena_bytes"], 0)

    def test_a_second_release_is_harmless(self):
        arena = Arena(1 << 20, device="cpu", expandable=False)
        stub = Stub(arena)
        first = Glm53Engine.release(stub)
        second = Glm53Engine.release(stub)
        self.assertEqual(first["arena_bytes"], 1 << 20)
        self.assertEqual(second["arena_bytes"], 0)

    def test_the_report_carries_the_only_honest_proof_the_bytes_landed(self):
        # reserved_before/after is what a workspace nobody freed would show up in, and
        # allocated_after is what a holder this method failed to find would show up in.
        _, report = self.release(Arena(1 << 20, device="cpu", expandable=False))
        self.assertEqual(set(report), {"arena_bytes", "reserved_before", "reserved_after",
                                       "returned", "allocated_after", "still_held"})
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


class ReleaseReportTests(unittest.TestCase):
    """What the door prints is the operator's only evidence, so it is pinned too."""

    def report(self, allocated, held=(), **kw):
        base = dict(returned=2 << 30, arena_bytes=2 << 30, tier_staging_bytes=192 << 20,
                    reserved_after=0, allocated_after=allocated, still_held=list(held))
        base.update(kw)
        return base

    def test_a_clean_release_says_what_came_back_and_nothing_else(self):
        line = boot.release_line(self.report(1 << 20), rank=2)
        self.assertIn("rank 2 gave back 2.00 GiB of 2.00 GiB arena plus 192 MiB of tier memory", line)
        self.assertNotIn("NOT come back clean", line)

    def test_a_holder_the_release_did_not_find_is_named_with_its_block_sizes(self):
        line = boot.release_line(self.report(200 << 20, held=(100 << 20, 64 << 20)), rank=1)
        self.assertIn("did NOT come back clean", line)
        self.assertIn("200 MiB is still held by live tensors", line)
        self.assertIn("largest blocks 100 MiB, 64 MiB", line)

    def test_the_floor_is_what_a_cuda_context_keeps_not_a_tolerance_for_leaks(self):
        self.assertEqual(boot.CLEAN_RELEASE_BYTES, 64 << 20)
        self.assertNotIn("NOT come back clean", boot.release_line(self.report(boot.CLEAN_RELEASE_BYTES)))
        self.assertIn("NOT come back clean", boot.release_line(self.report(boot.CLEAN_RELEASE_BYTES + 1)))


class Tier:
    def __init__(self, given=64 << 20, fail=False):
        self.given, self.fail, self.closed = given, fail, 0

    def close(self):
        self.closed += 1
        if self.fail:
            raise OSError("the disk went away")
        return self.given


class Runner:
    def __init__(self, tiered=None, prefix_tier=None):
        self.tiered, self.prefix_tier = tiered, prefix_tier


class ReleaseAllTests(unittest.TestCase):
    def test_the_tiers_close_before_the_engine_because_their_staging_is_outside_the_arena(self):
        stub = Stub(Arena(1 << 20, device="cpu", expandable=False))
        conversations, prefix = Tier(64 << 20), Tier(32 << 20)
        report = boot.release_all(stub, Runner(conversations, prefix))
        self.assertEqual(report["tier_staging_bytes"], 96 << 20)
        self.assertEqual((conversations.closed, prefix.closed), (1, 1))
        self.assertEqual(report["arena_bytes"], 1 << 20)

    def test_a_tier_that_cannot_close_does_not_keep_the_arena(self):
        stub = Stub(Arena(1 << 20, device="cpu", expandable=False))
        report = boot.release_all(stub, Runner(Tier(fail=True), Tier(32 << 20)))
        self.assertEqual(report["tier_staging_bytes"], 32 << 20)
        self.assertEqual(report["arena_bytes"], 1 << 20, "the arena must go whatever the disk did")

    def test_a_door_without_a_runner_releases_the_engine_alone(self):
        stub = Stub(Arena(1 << 20, device="cpu", expandable=False))
        self.assertEqual(boot.release_all(stub)["tier_staging_bytes"], 0)


class LiveBlockTests(unittest.TestCase):
    def test_without_a_device_there_is_nothing_to_name(self):
        self.assertEqual(live_device_blocks(), [])

    @unittest.skipUnless(torch.cuda.is_available(), "the device number is the point")
    def test_a_tensor_nobody_freed_shows_up_largest_first(self):
        keep = torch.empty(8 << 20, dtype=torch.uint8, device="cuda")
        blocks = live_device_blocks()
        self.assertTrue(blocks and blocks[0] >= (8 << 20), blocks)
        self.assertEqual(blocks, sorted(blocks, reverse=True))
        del keep


if __name__ == "__main__":
    unittest.main()
