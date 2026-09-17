"""A Qwen3.8 boot retires its dense projections' BF16 sources into their packs (engine/QWEN38_CARRY.md P4).

engine/profiles/qwen38/fleet.py prepared the dense lanes without `consume_weights`: every projection kept its BF16 source
in the arena beside its W4 and FP8 packs in allocations of their own. Now each lane's packs move into its source's arena
region and the source is dropped -- where they fit: engine/kernels/dense.packed_nbytes (the resident bound) against the
source's bytes. They fit for every projection but the shared expert's down projection, whose 160 columns pack at 256.

Over meta tensors of every served spec (the MTP head's included), with recording lanes: which sources are consumed, which
are kept and why; and the boot asks for it.
"""
import contextlib
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
torch = None
if importlib.util.find_spec("torch") is not None:
    import torch


@contextlib.contextmanager
def recording_lanes(consumed):
    from engine.kernels import dense

    class Recorder:
        def __init__(self, weight, **options):
            self.shape = tuple(weight.shape)

        def consume_weight(self, storage):
            consumed.append((self.label, tuple(storage.shape)))

    stand_ins = {label: type(label, (Recorder,), {"label": label})
                 for label in ("DenseLinear", "PaddedDenseLinear", "FP8Linear")}
    with contextlib.ExitStack() as stack:
        for label, cls in stand_ins.items():
            stack.enter_context(patch.object(dense, label, cls))
        yield


@unittest.skipUnless(torch is not None, "requires torch")
class ConsumeDenseTests(unittest.TestCase):
    def prepared(self, consume):
        from engine.profiles.qwen38 import specs
        from engine.profiles.qwen38.net import Qwen38Net
        from probes.engine_qwen38_cells import facts
        net = object.__new__(Qwen38Net)
        net.p = {s.name: torch.empty(s.shape, dtype=s.dtype, device="meta") for s in specs.all_specs(facts(), mtp=True)}
        net.hc_fp8 = False
        consumed = []
        with recording_lanes(consumed):
            net.prepare_dense(None, consume_weights=consume)
        return net, consumed

    def test_every_source_the_packs_fit_is_consumed(self):
        from engine.kernels.dense import packed_nbytes
        net, consumed = self.prepared(True)
        lanes = {key: lane for key, lane in net.dense.items() if key != "head"}
        kept = [key for key in lanes if net.p[key] is not None]
        self.assertTrue(kept)
        self.assertEqual(sorted(kept), sorted(net.retained_sources))
        self.assertEqual({key.split(".", 1)[1] if key.startswith("L") else key.split(".", 2)[2] for key in kept},
                         {"moe.sh_down"})                                          # the target's and the MTP head's
        self.assertEqual(len(consumed), len(lanes) - len(kept))
        for key, lane in lanes.items():
            with self.subTest(key=key):
                rows, cols = lane.shape
                if key in kept:
                    self.assertEqual(lane.label, "PaddedDenseLinear")
                    self.assertGreater(packed_nbytes(rows, 256), rows * cols * 2)     # 1,034,496 > 819,200
                else:
                    self.assertIsNone(net.p[key])
                    self.assertLessEqual(packed_nbytes(rows, cols), rows * cols * 2)
        self.assertIn(("DenseLinear", (320, 2560)), consumed)                           # the tightest fit
        self.assertIsNotNone(net.p["head"])                                             # the head is not a dense lane here

    def test_without_consume_every_source_stays(self):
        net, consumed = self.prepared(False)
        self.assertEqual(consumed, [])
        self.assertTrue(all(net.p[key] is not None for key in net.dense if key != "head"))

    def test_the_boot_consumes(self):
        source = (ROOT / "engine/profiles/qwen38/fleet.py").read_text()
        self.assertIn("net.prepare_dense(store, consume_weights=True)", source)


if __name__ == "__main__":
    unittest.main()
