"""What two profiles carried as copies is written once (charter D6: three sessions fixing the same thing is a missing
common place, not a branch). The profiles still name it where their callers look -- `facts.check_box`,
`glm53.modelopt_scales.ModelOptScales`, `Qwen38Caches.prepare` -- and each name resolves to the one implementation.

Held on a CPU: that it is the same object, that the moved arithmetic is the arithmetic it replaced, and that no
profile grows its own copy back.
"""
import ast
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import torch
except ImportError:                     # the identity and source checks below need no torch
    torch = None

PROFILES = ("glm53", "qwen38", "dsv41")
# Written once in engine/base or engine/modules; a profile that defines one of these again has copied it.
SHARED = {"aligned": "engine/base/slot_caches.py", "field_dtype": "engine/base/slot_caches.py",
          "StateField": "engine/base/slot_caches.py", "typed_view": "engine/base/slot_caches.py",
          "check_box": "engine/base/box.py", "BOX": "engine/base/box.py",
          "ModelOptScales": "engine/modules/modelopt_scales.py"}
MOVES = ("reset", "slot_bytes", "reset_slot", "snapshot_bytes", "prepare", "_upload_ids", "_ring_cells")


class OneImplementationTests(unittest.TestCase):
    def test_no_profile_defines_a_shared_name_again(self):
        found = []
        for profile in PROFILES:
            for path in sorted((ROOT / "engine/profiles" / profile).glob("*.py")):
                for node in ast.parse(path.read_text(encoding="utf-8")).body:
                    names = [node.name] if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else \
                        [t.id for t in node.targets if isinstance(t, ast.Name)] if isinstance(node, ast.Assign) else []
                    found += [f"{path.relative_to(ROOT).as_posix()}: {n} (it lives in {SHARED[n]})" for n in names if n in SHARED]
        self.assertEqual(found, [], "import it from where it lives instead of copying it")

    def test_the_box_is_one_fact(self):
        from engine.base import box
        from engine.profiles.glm53 import facts as glm
        from engine.profiles.qwen38 import facts as qwen
        self.assertIs(glm.BOX, box.BOX)
        self.assertIs(qwen.BOX, box.BOX)
        self.assertIs(glm.check_box, box.check_box)
        self.assertIs(qwen.check_box, box.check_box)
        self.assertEqual(box.BOX, {"name": "GB10 (DGX Spark)", "capability": (12, 1), "sms": 48, "devices": 1,
                                   "unified": True})

    @unittest.skipUnless(torch, "the ModelOpt binder imports torch")
    def test_modelopt_scales_is_the_modules_class_under_both_names(self):
        from engine.modules.modelopt_scales import ModelOptScales
        from engine.profiles.glm53 import modelopt_scales as profile
        self.assertIs(profile.ModelOptScales, ModelOptScales)
        # the preshard manifest records this module path as its serving adapter; it must keep resolving
        self.assertIn("serving_adapter='engine.profiles.glm53.modelopt_scales'",
                      (ROOT / "engine/profiles/glm53/preshard_modelopt.py").read_text(encoding="utf-8"))

    def test_the_ple_table_reads_through_the_lookup_table_module(self):
        from engine.modules.lookup_table import MappedTable
        from engine.profiles.qwen38.ple_table import PLETable
        self.assertTrue(issubclass(PLETable, MappedTable))
        for name in ("gather", "close"):
            self.assertIs(getattr(PLETable, name), getattr(MappedTable, name), name)


@unittest.skipUnless(torch, "the caches are torch tensors")
class SlotCachesTests(unittest.TestCase):
    def test_both_profiles_caches_take_the_shared_moves(self):
        from engine.base.slot_caches import SlotCaches
        from engine.profiles.glm53.caches import Glm53Caches
        from engine.profiles.qwen38.caches import Qwen38Caches
        for cls in (Glm53Caches, Qwen38Caches):
            self.assertTrue(issubclass(cls, SlotCaches), cls.__name__)
            for name in MOVES:
                if cls is Qwen38Caches and name in ("reset", "reset_slot"):
                    continue                    # PLE's id ring starts DEAD (-1): extended, through super()
                self.assertIs(getattr(cls, name), getattr(SlotCaches, name), f"{cls.__name__}.{name}")
        source = (ROOT / "engine/profiles/qwen38/caches.py").read_text(encoding="utf-8")
        self.assertIn("super().reset()", source)
        self.assertIn("super().reset_slot(slot)", source)

    def test_typed_view_is_the_arithmetic_it_replaced(self):
        """engine/profiles/glm53/caches.py built each field view inline, three times; this is that code, kept here as
        the oracle, against the shared function on the same bytes."""
        from engine.base.slot_caches import StateField, typed_view, field_dtype
        from math import prod

        def inline(storage, count, stride_bytes, f):
            dtype = field_dtype(f.dtype)
            size = 4 if f.dtype == "f32" else 2
            strides = tuple(prod(f.shape[i + 1:]) for i in range(len(f.shape)))
            base = storage.view(dtype)
            return base.as_strided((count, *f.shape), (stride_bytes // size, *strides), base.storage_offset() + f.offset // size)

        storage = torch.arange(4096, dtype=torch.int32).view(torch.uint8)[64:]     # a region at a nonzero offset
        stride = 1024
        for f in (StateField("conv", 3, (4, 8), "bf16", 0), StateField("rec", 3, (2, 4, 4), "f32", 64),
                  StateField("tail", -1, (8,), "f16", 200), StateField("rec", 7, (3, 5), "f32", 516)):
            got, want = typed_view(storage, 3, stride, f), inline(storage, 3, stride, f)
            self.assertEqual((got.shape, got.stride(), got.storage_offset(), got.dtype),
                             (want.shape, want.stride(), want.storage_offset(), want.dtype), f)
            self.assertTrue(torch.equal(got.view(torch.uint8) if got.dtype != torch.uint8 else got,
                                        want.view(torch.uint8) if want.dtype != torch.uint8 else want), f)

    def test_prepare_uploads_only_a_rows_new_suffix(self):
        """The shared `prepare` on the CPU path: a row's first publication, an unchanged step (no copy), an append
        (its new suffix only), and a segment whose slot the sequence does not own (refused before any copy)."""
        from engine.base.slot_caches import SlotCaches
        from engine.base.kv import BlockPool, SlotPool

        class Caches(SlotCaches):
            def __init__(self):
                self.pool = BlockPool(8, 16, 2, 8)
                self.slots = SlotPool(3)
                self.device = torch.device("cpu")
                self.paged = torch.zeros(8)
                self.state = torch.zeros(3 * 4, dtype=torch.uint8)
                self.block_table = torch.zeros(2, 8, dtype=torch.int32)
                self.snapshots = 0
                self.uploads = []
                self.reset()

            def _upload_ids(self, destination, ids):
                self.uploads.append(list(ids))
                super()._upload_ids(destination, ids)

        class Seg:
            def __init__(self, seq, slot, ctx, length):
                self.seq, self.slot, self.ctx, self.start, self.length = seq, slot, ctx, 0, length

        class Step:
            def __init__(self, *segments):
                self.segments = segments

        c = Caches()
        slot = c.slots.take(0)
        c.pool.reserve(0, 20)                                       # 20 tokens: two blocks of 16
        c.prepare(Step(Seg(0, slot, 0, 20)))
        self.assertEqual(c.uploads, [list(c.pool.row(0)[:2])])
        self.assertEqual(c.block_table[0, :2].tolist(), list(c.pool.row(0)[:2]))
        c.uploads.clear()
        c.prepare(Step(Seg(0, slot, 19, 1)))
        self.assertEqual(c.uploads, [], "an unchanged row does no copy")
        c.pool.reserve(0, 20)                                       # 20 more: a third block
        c.prepare(Step(Seg(0, slot, 20, 20)))
        self.assertEqual(c.uploads, [list(c.pool.row(0)[2:3])], "an append uploads only the new suffix")
        self.assertEqual(c.block_table[0, :3].tolist(), list(c.pool.row(0)[:3]))
        with self.assertRaisesRegex(ValueError, "does not own state slot"):
            c.prepare(Step(Seg(1, slot, 0, 1)))


@unittest.skipUnless(torch, "the recurrence is torch")
class DeltaRuleMarksTests(unittest.TestCase):
    """The reference lanes' chunk-mark loop (glm53 lanes.kda_chunk, qwen38 lanes.gdn_chunk) is
    modules/linear_attention.gated_delta_rule_marked. Each lane carried the loop; this is that loop, kept as the oracle,
    against the shared function, in both delta-rule forms: KDA's per-channel decay and GDN's per-head one."""

    @staticmethod
    def oracle(q, k, v, g, beta, state0, states_at, per_channel):
        from engine.modules.linear_attention import gated_delta_rule
        if not states_at:
            return gated_delta_rule(q, k, v, g, beta, state0, scale=q.shape[-1] ** -0.5, qk_l2norm=True,
                                    decay_per_channel=per_channel)
        outs, states, state, lo = [], [], state0, 0
        for hi in [c * 64 for c in states_at] + [q.shape[1]]:
            if hi > lo:
                o, state = gated_delta_rule(q[:, lo:hi], k[:, lo:hi], v[:, lo:hi], g[:, lo:hi], beta[:, lo:hi], state,
                                            scale=q.shape[-1] ** -0.5, qk_l2norm=True, decay_per_channel=per_channel)
                outs.append(o)
            if len(states) < len(states_at):
                states.append(state[0] if state is not None else
                              torch.zeros(v.shape[2], k.shape[-1], v.shape[-1], device=q.device, dtype=torch.float32))
            lo = hi
        return torch.cat(outs, dim=1), state, torch.stack(states)

    def test_the_shared_loop_is_the_lanes_loop_byte_for_byte(self):
        from engine.modules.linear_attention import gated_delta_rule_marked
        torch.manual_seed(0)
        t, h, dk, dv = 200, 3, 16, 8
        for per_channel in (True, False):
            q, k = torch.randn(1, t, h, dk), torch.randn(1, t, h, dk)
            v, beta = torch.randn(1, t, h, dv), torch.rand(1, t, h)
            g = -torch.rand(1, t, h, dk) if per_channel else -torch.rand(1, t, h)
            for state0 in (None, torch.randn(1, h, dk, dv)):
                for marks in (None, (), (1, 2), (0, 1, 3), (3,)):
                    want = self.oracle(q, k, v, g, beta, state0, marks, per_channel)
                    got = gated_delta_rule_marked(q, k, v, g, beta, state0, scale=q.shape[-1] ** -0.5, qk_l2norm=True,
                                                  decay_per_channel=per_channel, marks=marks)
                    self.assertEqual(len(got), len(want), (per_channel, marks))
                    for a, b in zip(got, want):
                        self.assertTrue(torch.equal(a, b), (per_channel, state0 is None, marks))

    def test_both_reference_lanes_take_it(self):
        for path, lane in (("engine/profiles/glm53/lanes.py", "kda_chunk"), ("engine/profiles/qwen38/lanes.py", "gdn_chunk")):
            tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
            fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == lane)
            calls = {c.func.id for c in ast.walk(fn) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
            self.assertIn("gated_delta_rule_marked", calls, path)
            self.assertFalse(any(isinstance(n, ast.For) for n in ast.walk(fn)), f"{path}: {lane} carries a loop again")


if __name__ == "__main__":
    unittest.main()
