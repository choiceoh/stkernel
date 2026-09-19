"""A parked conversation keeps its slot's live state, not the whole slot (engine/modules/state_rings `live_bytes`).

Each delta-rule layer's recurrent ring holds K+1 states, one per draft position, so a rejected draft rolls back by index;
a step reads the one at (context-1) % (K+1) and writes every position it computes before anything reads it. A
conversation that stopped after `context` tokens needs that one cell of each ring and the rest of the slot whole: 48 of
a GLM-5.3 slot's 286 MiB. These pin the pieces -- which bytes (CPU arena, tiny facts); that the KDA ring kernels take the
same step on a ring holding only those (the real Triton kernels, on a GPU or under TRITON_INTERPRET=1); how a window
copies across the pieces' seams; and that a server parks and resumes them, and still resumes a slot an earlier process
parked whole.

    docker exec -e TRITON_INTERPRET=1 -w <repo> stk-test python3 -m unittest tests.test_engine_park_live_state
"""
import importlib.util
import sys
import unittest
from dataclasses import replace
from math import prod
from pathlib import Path

HERE = Path(__file__).resolve().parent
for path in (HERE.parent, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

torch = None
if importlib.util.find_spec("torch") is not None:
    import torch

from engine.base.kv_tier import Segments  # noqa: E402


@unittest.skipUnless(torch is not None, "requires PyTorch")
class SegmentsTests(unittest.TestCase):
    def test_a_window_copies_across_the_seams_as_if_they_were_one_view(self):
        buf = torch.arange(200, dtype=torch.uint8)
        views = [buf[5:12], buf[40:40], buf[50:83], buf[100:101], buf[150:190]]      # the empty one is dropped
        segs = Segments(views)
        whole = torch.cat([v for v in views if v.numel()])
        self.assertEqual(segs.numel(), 81)
        for off, n in ((0, 81), (3, 10), (6, 2), (39, 3), (40, 41), (80, 1)):
            out = torch.zeros(n, dtype=torch.uint8)
            segs.gather(out, off, n)
            self.assertTrue(torch.equal(out, whole[off:off + n]), (off, n))
        target = torch.zeros_like(buf)
        into = [target[5:12], target[50:83], target[100:101], target[150:190]]
        for off in range(0, 81, 16):                                               # a staging window of 16 bytes
            Segments(into).scatter(whole[off:off + 16], off, min(16, 81 - off))
        self.assertTrue(torch.equal(torch.cat(into), whole))
        self.assertEqual(int(target.count_nonzero()), int(whole.count_nonzero()), "nothing outside the views is touched")
        self.assertIs(Segments.of(segs), segs)
        with self.assertRaises(ValueError):
            Segments([buf[0:0]])


@unittest.skipUnless(torch is not None, "requires PyTorch")
class LiveBytesTests(unittest.TestCase):
    DRAFT = (2, 16, 1, 8)

    def caches(self, F):
        from engine.base.arena import Arena
        from engine.profiles.glm53.caches import Glm53Caches, layout
        arena = Arena(4096 + layout(F, range(F.layers), self.DRAFT).nbytes(2, 2), device="cpu")
        return Glm53Caches(arena, F, range(F.layers), 2, 2, draft=self.DRAFT)

    @staticmethod
    def facts():
        from tests.test_engine_glm53 import tiny_facts
        return replace(tiny_facts(), layers=4, kinds=("kda", "dsa", "kda", "dsa"))          # two rings, two tails

    def fields(self, c):
        """(field, byte offset in the slot, bytes, cells: K+1 for a recurrent ring, else 1), in slot order."""
        from engine.base.slot_caches import SIZES
        rings = {("rec", L) for L in c.ring_layers()}
        return [(f, f.offset, prod(f.shape) * SIZES[f.dtype], f.shape[0] if (f.name, f.layer) in rings else 1)
                for f in sorted(c.layout.fields, key=lambda f: f.offset)]

    def test_the_live_bytes_are_the_slot_but_the_draft_cells_of_each_recurrent_ring(self):
        from test_engine_tier import extra_bytes
        F = self.facts()
        c = self.caches(F)
        self.assertEqual(len(c.ring_layers()), 2)
        whole = c.slot_bytes(1)
        whole.copy_(torch.randint(0, 256, whole.shape, dtype=torch.uint8))
        fields = self.fields(c)
        names = {f.name for f, *_ in fields}
        self.assertEqual(names, {"conv", "rec", "tail", "draft"})
        for position in (1, 2, F.spec_k + 1, F.spec_k + 2, 1000):
            live = c.live_bytes(1, position)
            want = []
            for f, at, size, cells in fields:
                cell = size // cells
                at += (position - 1) % cells * cell if cells > 1 else 0
                want.append(whole[at:at + cell].numpy().tobytes())
            self.assertEqual(extra_bytes(live), b"".join(want), position)
            rec = sum(size for f, _, size, cells in fields if cells > 1)
            self.assertEqual(live.numel(), sum(size for _, _, size, _ in fields) - rec * F.spec_k // (F.spec_k + 1))

    def test_a_slot_given_its_live_bytes_back_holds_the_live_state_and_zeros(self):
        from test_engine_tier import extra_bytes, fill_extra
        F = self.facts()
        c = self.caches(F)
        whole, position = c.slot_bytes(1), 13
        whole.copy_(torch.randint(1, 256, whole.shape, dtype=torch.uint8))
        before = whole.clone()
        parked = extra_bytes(c.live_bytes(1, position))                           # what the tier writes
        whole.copy_(torch.randint(1, 256, whole.shape, dtype=torch.uint8))       # the slot serves another conversation
        c.reset_slot(1)                                                          # a resume clears it first (resume_bytes)
        fill_extra(c.live_bytes(1, position), parked)
        for f, at, size, cells in self.fields(c):
            cell = size // cells
            for i in range(cells):
                got = whole[at + i * cell:at + (i + 1) * cell]
                if cells == 1 or i == (position - 1) % cells:
                    self.assertTrue(torch.equal(got, before[at + i * cell:at + (i + 1) * cell]), (f.name, f.layer, i))
                else:
                    self.assertFalse(bool(got.any()), f"{f.name} {f.layer} cell {i}: a draft position's state, cleared")

    def test_only_a_real_slot_with_a_computed_position_has_live_bytes(self):
        c = self.caches(self.facts())
        with self.assertRaises(ValueError):
            c.live_bytes(1, 0)
        with self.assertRaises(IndexError):
            c.live_bytes(0, 5)


@unittest.skipUnless(torch is not None, "requires PyTorch")
class QwenLiveBytesTests(unittest.TestCase):
    """Qwen3.8 parks the same way (#1298 gave it the tier): its served store names the live state -- one GDN state of
    K+1 per layer, every other field whole, PLE's id ring among them."""

    def test_a_served_slot_keeps_one_gdn_state_and_its_ple_ring_comes_back(self):
        from engine.base.arena import Arena
        from engine.base.slot_caches import SIZES
        from engine.profiles.qwen38.adapter import ServedStore
        from engine.profiles.qwen38.caches import Qwen38Caches, layout
        from probes.engine_qwen38_cells import facts
        from test_engine_tier import extra_bytes, fill_extra
        F = replace(facts(), spec_k=3)                                            # the served K
        gdn = [L for L in range(F.layers) if not F.is_qsa(L)]
        qsa = [L for L in range(F.layers) if F.is_qsa(L)]
        layers = tuple(sorted({gdn[0], gdn[1], qsa[0], *tuple(F.ple_layers)[:1]}))  # the served checkpoint's layers, a few
        arena = Arena(4096 + layout(F, layers, mtp=True).nbytes(1, 1), device="cpu")
        c = Qwen38Caches(arena, F, layers, 1, 1, mtp=True)
        store = ServedStore(c)
        names = {f.name for f in c.layout.fields}
        self.assertTrue({"rec", "conv", "keys", "ple_ids", "ple_conv"} <= names, names)
        whole, position = c.slot_bytes(1), 10
        whole.copy_(torch.randint(1, 256, whole.shape, dtype=torch.uint8))
        before = whole.clone()
        parked = extra_bytes(store.live_bytes(1, position))
        whole.copy_(torch.randint(1, 256, whole.shape, dtype=torch.uint8))       # another conversation used the slot
        store.clear(1)
        self.assertTrue(bool((c._fields["ple_ids", -1][1] == -1).all()), "cleared as open leaves it: PLE's ids DEAD")
        fill_extra(store.live_bytes(1, position), parked)
        rec = 0
        for f in c.layout.fields:
            size = prod(f.shape) * SIZES[f.dtype]
            cells = f.shape[0] if f.name == "rec" else 1
            rec += size if cells > 1 else 0
            cell = size // cells
            for i in range(cells):
                got = whole[f.offset + i * cell:f.offset + (i + 1) * cell]
                was = before[f.offset + i * cell:f.offset + (i + 1) * cell]
                if cells == 1 or i == (position - 1) % cells:
                    self.assertTrue(torch.equal(got, was), (f.name, f.layer, i))
                else:
                    self.assertFalse(bool(got.any()), f"{f.name} {f.layer} cell {i}")
        total = sum(prod(f.shape) * SIZES[f.dtype] for f in c.layout.fields)
        self.assertEqual(len(parked), total - rec * F.spec_k // (F.spec_k + 1))

    def test_the_composed_model_parks_what_its_store_names_else_the_slot(self):
        from types import SimpleNamespace
        from engine.base.composed import ComposedModel
        slot, cleared = object(), []
        served = SimpleNamespace(slot_bytes=lambda s: slot, live_bytes=lambda s, position: ("live", s, position),
                                 clear=cleared.append)
        model = SimpleNamespace(store=served)
        self.assertEqual(ComposedModel.park_bytes(model, 1, 9), ("live", 1, 9))
        self.assertEqual(cleared, [])
        self.assertEqual(ComposedModel.resume_bytes(model, 1, 9), ("live", 1, 9))
        self.assertEqual(cleared, [1])
        reference = SimpleNamespace(store=SimpleNamespace(slot_bytes=lambda s: slot))  # base/composed.PositionStore
        self.assertIs(ComposedModel.park_bytes(reference, 1, 9), slot)
        self.assertIs(ComposedModel.resume_bytes(reference, 1, 9), slot)


KDA_RUNS = False
if torch is not None and importlib.util.find_spec("triton") is not None:
    from tests.test_engine_kernel_glue import INTERPRET, KDA_DEVICE, KDA_RUNS, kda_kernels


@unittest.skipUnless(KDA_RUNS, "requires Triton with CUDA, or TRITON_INTERPRET=1")
class KdaRingTests(unittest.TestCase):
    """The ring kernels a GLM-5.3 decode runs, on the ring as parked (every cell) and as resumed (the live cell, zeros)."""

    def setUp(self):
        from engine.base import kernel_shape as ks
        from engine.base.kernel_shape import MEASURED, LinearAttention
        torch.manual_seed(20260919)
        ks.reset()
        self.addCleanup(ks.reset)
        self.h, self.kd = (2, 16) if INTERPRET else (16, 128)                     # GLM-5.3's per-rank KDA cell on a GPU
        ks.bind(replace(MEASURED, linear=LinearAttention(heads=self.h, v_heads=self.h, k_dim=self.kd, v_dim=self.kd,
                                                         conv=4)))

    def step(self, t):
        g = lambda *shape: torch.randn(*shape, device=KDA_DEVICE)                 # noqa: E731
        h, kd = self.h, self.kd
        return g(1, t, h, kd), g(1, t, h, kd), g(1, t, h, kd), g(1, t, h, kd), g(1, t, h), g(h) * .2, g(h * kd) * .1

    def test_a_step_on_the_resumed_ring_is_the_step_on_the_parked_one(self):
        from engine.kernels.kda.ring import recurrent_kda_ring
        cells = 8                                                                 # GLM-5.3: K = 7
        for t in ((1, 3) if INTERPRET else (1, 8)):
            for context in (1, 7, 8, 13):
                with self.subTest(tokens=t, context=context):
                    q, k, v, raw, beta, a_log, bias = self.step(t)
                    parked = torch.randn(3, cells, self.h, self.kd, self.kd, device=KDA_DEVICE) * .1
                    resumed = torch.zeros_like(parked)
                    live = (context - 1) % cells
                    resumed[1, live] = parked[1, live]
                    with kda_kernels():
                        want = recurrent_kda_ring(q, k, v, raw, beta, a_log, bias, parked, 1, context, -5.0)
                        got = recurrent_kda_ring(q, k, v, raw, beta, a_log, bias, resumed, 1, context, -5.0)
                    self.assertTrue(torch.equal(got, want))
                    for i in range(t):
                        cell = (context + i) % cells
                        self.assertTrue(torch.equal(resumed[1, cell], parked[1, cell]), f"position {context + i}")

    def test_a_decode_step_s_rows_on_resumed_rings_are_the_rows_on_parked_ones(self):
        from engine.kernels.kda.ring import recurrent_kda_ring_rows
        cells, t = 8, 2
        q, k, v, raw, beta, a_log, bias = self.step(2 * t)
        slots = torch.tensor([2, 1], device=KDA_DEVICE, dtype=torch.int32)
        contexts = torch.tensor([5, 12], device=KDA_DEVICE, dtype=torch.int32)
        parked = torch.randn(3, cells, self.h, self.kd, self.kd, device=KDA_DEVICE) * .1
        resumed = torch.zeros_like(parked)
        for slot, context in zip(slots.tolist(), contexts.tolist()):
            resumed[slot, (context - 1) % cells] = parked[slot, (context - 1) % cells]
        with kda_kernels():
            want = recurrent_kda_ring_rows(q, k, v, raw, beta, a_log, bias, parked, slots, contexts, -5.0)
            got = recurrent_kda_ring_rows(q, k, v, raw, beta, a_log, bias, resumed, slots, contexts, -5.0)
        self.assertTrue(torch.equal(got, want))


@unittest.skipUnless(KDA_RUNS, "requires Triton with CUDA, or TRITON_INTERPRET=1")
class GdnRingTests(unittest.TestCase):
    """The ring kernel a Qwen3.8 decode runs (GatedDeltaNet's gate), on the ring as parked and as resumed."""

    def setUp(self):
        from engine.base import kernel_shape as ks
        from engine.base.kernel_shape import LinearAttention
        from tests.test_engine_kernel_glue import HEAD_DECAY
        torch.manual_seed(20260919)
        ks.reset()
        self.addCleanup(ks.reset)
        self.h, self.hv, self.kd = (2, 4, 16) if INTERPRET else (4, 12, 128)       # Qwen3.8's per-rank cell on a GPU
        ks.bind(replace(HEAD_DECAY, linear=LinearAttention(heads=self.h, v_heads=self.hv, k_dim=self.kd, v_dim=self.kd,
                                                          conv=4, decay="head")))
        self.dtype = torch.float32 if INTERPRET else torch.bfloat16

    def test_a_step_on_the_resumed_ring_is_the_step_on_the_parked_one(self):
        from engine.kernels.kda.ring import recurrent_gdn_ring
        from tests.test_engine_gdn_ring_gate import ring_kernels
        cells = 4                                                                 # served Qwen3.8: K = 3
        g = lambda *shape: torch.randn(*shape, device=KDA_DEVICE, dtype=self.dtype)   # noqa: E731
        for t in (1, 3):
            for context in (1, 3, 4, 9):
                with self.subTest(tokens=t, context=context):
                    q, k, v = g(1, t, self.h, self.kd), g(1, t, self.h, self.kd), g(1, t, self.hv, self.kd)
                    a, b = g(t, self.hv), g(t, self.hv)
                    A_log, dt_bias = torch.randn(self.hv, device=KDA_DEVICE) * .5, torch.randn(self.hv, device=KDA_DEVICE)
                    parked = torch.randn(3, cells, self.hv, self.kd, self.kd, device=KDA_DEVICE) * .1
                    resumed = torch.zeros_like(parked)
                    live = (context - 1) % cells
                    resumed[1, live] = parked[1, live]
                    with ring_kernels():
                        want = recurrent_gdn_ring(q, k, v, a[None], b[None], A_log, dt_bias, parked, 1, context)
                        got = recurrent_gdn_ring(q, k, v, a[None], b[None], A_log, dt_bias, resumed, 1, context)
                    self.assertTrue(torch.equal(got, want))
                    for i in range(t):
                        cell = (context + i) % cells
                        self.assertTrue(torch.equal(resumed[1, cell], parked[1, cell]), f"position {context + i}")


def ring_engines():
    """The serve tests' engine with a 12-byte slot as a torch tensor: 4 bytes of other state and a recurrent ring of two
    4-byte cells addressed by position -- the cell of (context-1) % 2 is the conversation's, the other a draft's."""
    import test_engine_serve as T

    class WholeRingEngine(T.Engine):
        def __init__(self, slots):
            super().__init__(slots)
            self.state = torch.zeros(slots, 12, dtype=torch.uint8)
            self.resumed = []                                   # the slot's bytes as each resume found them

        def state_bytes(self, slot):
            return self.state[slot]

        def open(self, seq, slot):
            self.ctx[seq] = 0
            self.opened.append(seq)
            self.state[slot] = torch.tensor([slot, seq, 1, 1] + [10 + seq] * 4 + [20 + seq] * 4, dtype=torch.uint8)

        def resume(self, seq, slot, record):
            self.resumed.append(self.state[slot].numpy().tobytes())
            super().resume(seq, slot, record)

    class RingEngine(WholeRingEngine):
        def park_bytes(self, slot, context):
            cell = 4 + (context - 1) % 2 * 4
            return Segments([self.state[slot, :4], self.state[slot, cell:cell + 4]])

        def resume_bytes(self, slot, context):
            self.state[slot].zero_()
            return self.park_bytes(slot, context)

    return WholeRingEngine, RingEngine


def ring_server(engine, tier, rows=1, blocks=16):
    """test_engine_serve.server with this engine and tier: one row, so every conversation takes the same slot."""
    import test_engine_serve as T
    from engine.base.kv import BlockPool, SlotPool
    from engine.base.record import Ring
    from engine.base.runner import STEP_RECORD, Runner
    from engine.base.scheduler import Contract
    from engine.base.serve import Server
    from engine.base.tiered_kv import TieredKV
    from test_engine_tier import Storage
    runner = Runner(engine, Contract(4, 8, 0, 0, rows), BlockPool(blocks, 4, rows, blocks), SlotPool(rows + 1),
                    Ring(16, STEP_RECORD.size), keep_idle=True)
    runner.kv.attach_storage(Storage(blocks * 4), 4)
    runner.tiered = TieredKV(runner.kv, tier)
    return Server(engine, runner, T.Comm(), host="127.0.0.1", port=0, max_pending=64)


@unittest.skipUnless(torch is not None, "requires PyTorch")
class ParkLiveStateServeTests(unittest.TestCase):
    def test_a_conversation_parks_its_live_cell_and_reads_back_into_a_cleared_slot(self):
        from test_engine_tier import MemoryTier, quiet
        _, RingEngine = ring_engines()
        tier, engine = MemoryTier(), RingEngine(2)
        s = ring_server(engine, tier)
        a, _ = s.submit([3, 4, 5], 2, 0)
        quiet(s)
        self.assertEqual(s.take_result(a), [5, 5])
        self.assertEqual(tier.records[a]["context"], 4)                        # (4 - 1) % 2: the second cell is live
        self.assertEqual(tier.index[str(a)]["extra"], 8, "4 bytes of state and one cell, not the slot's 12")
        self.assertEqual(tier.extra[a], bytes([1, a, 1, 1] + [20 + a] * 4))
        b, _ = s.submit([7, 8], 1, 0)                                           # the one slot serves b, then b parks
        quiet(s)
        self.assertEqual(s.take_result(b), [8])
        turn, _ = s.submit([6], 2, 0, conversation=a)
        quiet(s)
        self.assertEqual(s.take_result(turn), [6, 6])
        self.assertEqual(engine.resumed[-1], bytes([1, a, 1, 1] + [0] * 4 + [20 + a] * 4),
                         "the draft cell reads zero, not b's")

    def test_a_slot_an_earlier_process_parked_whole_reads_back_whole(self):
        from test_engine_tier import MemoryTier, quiet
        WholeRingEngine, RingEngine = ring_engines()
        tier = MemoryTier()
        s = ring_server(WholeRingEngine(2), tier)                               # a process from before live-state parking
        a, _ = s.submit([3, 4, 5], 2, 0)
        quiet(s)
        s.take_result(a)
        self.assertEqual(tier.index[str(a)]["extra"], 12)
        engine = RingEngine(2)
        s = ring_server(engine, tier)                                           # the next process, over the same tier
        turn, _ = s.submit([6], 2, 0, conversation=a)
        quiet(s)
        self.assertEqual(s.take_result(turn), [6, 6])
        self.assertEqual(engine.resumed[-1], bytes([1, a, 1, 1] + [10 + a] * 4 + [20 + a] * 4), "all twelve, as parked")
        self.assertEqual(tier.index[str(a)]["extra"], 8, "and its next park keeps the live state")


if __name__ == "__main__":
    unittest.main()
