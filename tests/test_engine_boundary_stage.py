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
    def caches(self, F, max_seqs=2, snapshots=2, device="cpu"):
        p = layout(F, range(F.layers))
        arena = Arena(4096 + p.nbytes(2, max_seqs) + snapshots * snapshot_layout(F, range(F.layers))[0] + stage_bytes(F, range(F.layers), max_seqs),
                      device=device)
        return Glm53Caches(arena, F, range(F.layers), 2, max_seqs, draft=None, snapshots=snapshots, stage=True)

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
        for dtype in ("fp32", "fp16"):
            F = replace(tiny_facts(), kda_state_dtype=dtype, kinds=("kda", "dsa", "kda"),
                        kda_heads=8, kda_dim=33, spec_k=6, block=768)
            host = BoundaryStageTests().caches(F, max_seqs=4)
            dev = BoundaryStageTests().caches(F, max_seqs=4, device="cuda")
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
                                   ([4, 3, 2, 1], [1535, 768, 767, 2301], [1, 6, 1, 6])):
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
                                cache.checkpoint_from_stage(slot, 0)
                                cache.restore(1, boundary, 0)
                            self.assertTrue(torch.equal(dev.snapshot_store.cpu(), host.snapshot_store), dtype)
                            self.assertTrue(torch.equal(dev.state.cpu(), host.state), dtype)
            finally:
                graph.reset()


if __name__ == "__main__":
    unittest.main()
