"""Boundaries crossed while generating (45차 §23): a decode step that crosses a block boundary parks the KDA state and
conv taps at that boundary in the caches' stage (eager form here; the Triton kernel mirrors it), and a checkpoint
from the stage equals a checkpoint taken from the rings while they still hold it. CPU arena, tiny facts."""
import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from engine.base.arena import Arena  # noqa: E402
from engine.profiles.glm53.caches import Glm53Caches, layout, snapshot_layout, stage_bytes  # noqa: E402
from tests.test_engine_glm53 import tiny_facts  # noqa: E402


class BoundaryStageTests(unittest.TestCase):
    def caches(self, F, max_seqs=2, snapshots=2, device="cpu", draft=None):
        p = layout(F, range(F.layers), draft)
        arena = Arena(4096 + p.nbytes(2, max_seqs) + snapshots * snapshot_layout(F, range(F.layers), draft)[0]
                      + stage_bytes(F, range(F.layers), max_seqs, draft), device=device)
        return Glm53Caches(arena, F, range(F.layers), 2, max_seqs, draft=draft, snapshots=snapshots, stage=True)

    @staticmethod
    def write_draft(c, slot, positions):
        """What an observe leaves in the ring: position p's key and value in cell p mod window, tagged by p."""
        ring = c.draft_ring(slot)
        for p in positions:
            ring[:, 0, p % ring.shape[2]] = float(p + 1)
            ring[:, 1, p % ring.shape[2]] = -float(p + 1)

    def test_a_snapshot_past_its_boundary_puts_back_the_drafter_cells_written_after_it(self):
        """The drafter ring files position p in the cell of p - window. A boundary crossed while generating is
        snapshotted after positions past it were written over cells the snapshot still needs -- read back, they are
        another request's keys, rotated past the boundary, sitting where the oldest context should be. The stage
        keeps those cells from before the crossing step's observe and the snapshot puts them back: it is the ring as
        it stood when the context was exactly the boundary, for the async chain two steps ahead and for a
        synchronous step alike."""
        from engine.profiles.glm53.caches import draft_stash_cells
        F = tiny_facts()                                                      # block 16, spec_k 5: steps of up to 6
        draft = (2, 40, 1, 4)                                                 # a 40-cell window: positions past 48 wrap
        dev = "cpu"
        for synchronous in (False, True):
            with self.subTest(synchronous=synchronous):
                c = self.caches(F, draft=draft)
                self.assertEqual(c._stage["draft", -1].shape[1:], (2, 2, draft_stash_cells(F), 1, 4))
                self.write_draft(c, 1, range(48))
                c.checkpoint(1, 48, 1)                                        # the reference: the context is the boundary
                c.reset_slot(1)
                self.write_draft(c, 1, range(45))
                if synchronous:
                    c.stash_draft(1, 48)                                      # adapter.decode, before its observe
                    self.write_draft(c, 1, range(45, 51))                     # 45 -> 51 crosses 48
                    c.checkpoint(1, 48, 0, past=3)
                else:
                    c.stage_boundaries(torch.tensor([1], device=dev), torch.tensor([45], device=dev), torch.tensor([6], device=dev))
                    self.write_draft(c, 1, range(45, 63))                     # the crossing step, and two more ahead of the host
                    c.checkpoint_from_stage(1, 0, 48)
                self.assertTrue(torch.equal(c._snap["draft", -1][0], c._snap["draft", -1][1]))
                live = c.draft_ring(1)
                self.assertFalse(torch.equal(live, c._snap["draft", -1][1]), "the live ring did move past the boundary")
        with self.assertRaises(ValueError):
            c.checkpoint(1, 48, 0, past=draft_stash_cells(F) + 1)
        with self.assertRaises(ValueError):
            c.checkpoint_from_stage(1, 0)                                     # a drafter ring is put back at its boundary
        with self.assertRaises(ValueError):
            self.caches(F, draft=(2, draft_stash_cells(F), 1, 4))             # a window no wider than the stash

    def test_a_crossing_step_parks_the_boundary_and_a_non_crossing_one_leaves_the_stage_alone(self):
        F = tiny_facts()                                                      # block 16, conv 4, spec_k 5
        c = self.caches(F)
        g = torch.Generator().manual_seed(0)
        for L in F.kda_layers:
            conv, rec = c.kda(L, 1)
            conv.copy_(torch.randn(conv.shape, generator=g).to(conv.dtype)); rec.copy_(torch.randn(rec.shape, generator=g))
        dev = c.device
        c.stage_boundaries(torch.tensor([1], device=dev), torch.tensor([14], device=dev), torch.tensor([3], device=dev))   # 14 -> 17 crosses 16
        c.checkpoint_from_stage(1, 0)
        c.checkpoint(1, 16, 1)                                                # the rings still hold position 15 and the taps 13..15
        for L in F.kda_layers:
            self.assertTrue(torch.equal(c._snap["rec", L][0], c._snap["rec", L][1]))
            self.assertTrue(torch.equal(c._snap["conv", L][0], c._snap["conv", L][1]))
        # the rings move on; a step that crosses nothing leaves the parked boundary as it was
        for L in F.kda_layers:
            conv, rec = c.kda(L, 1)
            conv.add_(1); rec.add_(1)
        c.stage_boundaries(torch.tensor([1], device=dev), torch.tensor([17], device=dev), torch.tensor([4], device=dev))   # 17 -> 21: no boundary
        for L in F.kda_layers:
            self.assertTrue(torch.equal(c._stage["rec", L][1], c._snap["rec", L][1]))
        c.stage_boundaries(torch.tensor([1], device=dev), torch.tensor([30], device=dev), torch.tensor([2], device=dev))   # 30 -> 32 crosses 32
        for L in F.kda_layers:
            conv, rec = c.kda(L, 1)
            self.assertTrue(torch.equal(c._stage["rec", L][1], rec[31 % (F.spec_k + 1)]))

    def test_the_stage_is_declared_with_the_arena(self):
        F = tiny_facts()
        self.assertEqual(stage_bytes(F, range(F.layers), 2), 3 * snapshot_layout(F, range(F.layers))[0])
        with self.assertRaises(RuntimeError):
            p = layout(F, range(F.layers))
            Glm53Caches(Arena(4096 + p.nbytes(2, 2) + snapshot_layout(F, range(F.layers))[0], device="cpu"), F, range(F.layers), 2, 2,
                        snapshots=1, stage=False).checkpoint_from_stage(1, 0)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class CudaBoundaryStageTests(unittest.TestCase):
    def test_fp32_and_fp16_graph_staging_restore_match_cpu_bytes(self):
        # The larger odd dimensions require several iterations per CTA for
        # both recurrent cells and conv channels, with a masked final tile.
        for dtype, heads, dim in (("fp32", 8, 33), ("fp16", 8, 33),
                                  ("fp32", 32, 65), ("fp16", 32, 65)):
            F = replace(tiny_facts(), kda_state_dtype=dtype, kinds=("kda", "dsa", "kda"),
                        kda_heads=heads, kda_dim=dim, spec_k=6, block=768)
            host = BoundaryStageTests().caches(F, max_seqs=4, draft=(2, 64, 2, 8))
            dev = BoundaryStageTests().caches(F, max_seqs=4, device="cuda", draft=(2, 64, 2, 8))
            # Include arbitrary floating bit patterns and padding, so this
            # copy contract also detects NaN canonicalization and byte overrun.
            rng = torch.Generator().manual_seed(492)
            host.state.copy_(torch.randint(0, 256, host.state.shape, dtype=torch.uint8, generator=rng))
            dev.state.copy_(host.state)
            host.stage_store.fill_(173); dev.stage_store.copy_(host.stage_store)
            host.snapshot_store.zero_(); dev.snapshot_store.zero_()
            slots = torch.tensor([1, 2, 3, 4], device="cuda")
            before = torch.tensor([766, 767, 1534, 766], device="cuda")
            counts = torch.tensor([3, 0, 6, 1], device="cuda")
            dev.stage_boundaries(slots, before, counts)  # build tables/JIT before capture
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                dev.stage_boundaries(slots, before, counts)
            try:
                dev.stage_store.copy_(host.stage_store)
                for ss, bb, nn in (([1, 2, 3, 4], [766, 767, 1534, 766], [3, 0, 6, 1]),
                                   ([4, 3, 2, 1], [1535, 768, 767, 2301], [1, 6, 1, 6]),
                                   ([2, 4, 1, 3], [0, 767, 768, 1535], [0, 0, 7, 0])):
                    cpu_inputs = [torch.tensor(x) for x in (ss, bb, nn)]
                    for target, value in zip((slots, before, counts), cpu_inputs):
                        target.copy_(value)
                    host.stage_boundaries(*cpu_inputs)
                    graph.replay()
                    self.assertTrue(torch.equal(dev.stage_store.cpu(), host.stage_store), dtype)
                    self.assertTrue(torch.equal(dev.state.cpu(), host.state), dtype)
                    for slot, ctx, count in zip(ss, bb, nn):
                        boundary = ((ctx + count) // F.block) * F.block
                        if count and boundary > ctx:
                            for cache in (host, dev):
                                cache.checkpoint_from_stage(slot, 0, boundary)
                                cache.restore(1, boundary, 0)
                            self.assertTrue(torch.equal(dev.snapshot_store.cpu(), host.snapshot_store), dtype)
                            self.assertTrue(torch.equal(dev.state.cpu(), host.state), dtype)
            finally:
                graph.reset()


if __name__ == "__main__":
    unittest.main()
