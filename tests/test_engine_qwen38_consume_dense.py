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
            self.options = options

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
        # the target's shared-expert down projection: its W4 packs at 256 columns outgrow the 160-column source; the
        # MTP head's projections have no lane at all (mtp_precision "bf16", the default: torch's matmul over the source)
        self.assertEqual({key.split(".", 1)[1] for key in kept}, {"moe.sh_down"})
        self.assertTrue(all(key.startswith("L") for key in kept))
        self.assertFalse(any(key.startswith("mtp.") for key in net.dense))
        self.assertTrue(all(net.p[key] is not None for key in net.dense_names(net.p) if key.startswith("mtp.")))
        self.assertEqual(len(consumed), len(lanes) - len(kept))
        for key, lane in lanes.items():
            with self.subTest(key=key):
                rows, cols = lane.shape
                w4 = lane.options.get("decode_precision", "w4") == "w4"
                self.assertEqual(w4, not key.startswith("mtp."))
                if key in kept:
                    self.assertEqual(lane.label, "PaddedDenseLinear")
                    self.assertGreater(packed_nbytes(rows, 256), rows * cols * 2)     # 1,034,496 > 819,200
                else:
                    self.assertIsNone(net.p[key])
                    padded = cols if cols % 128 == 0 else 256
                    self.assertLessEqual(packed_nbytes(rows, padded, decode_w4=w4), rows * cols * 2)
        self.assertIn(("DenseLinear", (320, 2560)), consumed)                           # the tightest fit
        self.assertIsNotNone(net.p["head"])                                             # the head is not a dense lane here

    def test_the_mtp_head_s_precision(self):
        """fp8 (default): FP8 lanes whose decode rows take fp8_rows; bf16: no lane, torch's matmul over the kept
        source; w4: the target layers' lanes."""
        from engine.profiles.qwen38 import specs
        from engine.profiles.qwen38.net import Qwen38Net
        from probes.engine_qwen38_cells import facts
        for precision in ("fp8", "bf16", "w4"):
            net = object.__new__(Qwen38Net)
            net.p = {s.name: torch.empty(s.shape, dtype=s.dtype, device="meta") for s in specs.all_specs(facts(), mtp=True)}
            net.hc_fp8, net.mtp_precision = False, precision
            with recording_lanes([]):
                net.prepare_dense(None, consume_weights=True)
            mtp = {key: lane for key, lane in net.dense.items() if key.startswith("mtp.")}
            with self.subTest(precision=precision):
                if precision == "bf16":
                    self.assertEqual(mtp, {})
                    self.assertTrue(all(net.p[key] is not None for key in net.dense_names(net.p) if key.startswith("mtp.")))
                else:
                    self.assertEqual(len(mtp), 4)
                    want = dict(decode_precision="fp8", fp8_decode_rows=True) if precision == "fp8" else {}
                    self.assertTrue(all({k: v for k, v in lane.options.items() if k in want} == want for lane in mtp.values()))
                    if precision == "w4":
                        self.assertTrue(all("decode_precision" not in lane.options for lane in mtp.values()))
        with self.assertRaises(ValueError):
            Qwen38Net.__init__(object.__new__(Qwen38Net), facts(), type("C", (), {"world_size": 4, "rank": 0})(), None,
                               mtp_precision="fp4")

    def test_without_consume_every_source_stays(self):
        net, consumed = self.prepared(False)
        self.assertEqual(consumed, [])
        self.assertTrue(all(net.p[key] is not None for key in net.dense if key != "head"))

    def test_the_boot_consumes(self):
        source = (ROOT / "engine/profiles/qwen38/fleet.py").read_text()
        self.assertIn("net.prepare_dense(store, consume_weights=True)", source)


if __name__ == "__main__":
    unittest.main()
