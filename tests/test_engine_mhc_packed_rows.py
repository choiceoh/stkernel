"""MK mHC coefficient storage by row count, on the CPU: which weight and which native form each launch gets.

K=7 verify steps at C=1 (8 rows) and C=2 (16 rows) read the lossless BF16 pack through the consumer kernels, at
packet and at ordinary (all-reduce) boundaries alike. GPU numerics and timing are probes/engine_mhc_c2_packed.py
(measurements/st_c2_mhc_packed_20260915).
"""
import importlib.util
import unittest
from unittest.mock import patch

torch = None
if importlib.util.find_spec("torch") is not None:
    import torch


class Recorder:
    def __init__(self):
        self.calls = []

    def run_mhc(self, ptrs, scalars, ints, bf16_fn, ar_consumer):
        self.calls.append(("ordinary", ints[0], ptrs[4], bf16_fn, ar_consumer))

    def run_mhc_packets(self, ptrs, scalars, ints, packets, bf16_fn):
        self.calls.append(("packets", ints[0], ptrs[4], bf16_fn, False))


def owner(fn, key, **kwargs):
    from engine.kernels.dense import mhc
    ext = Recorder()
    with patch.object(mhc, "extension", lambda: ext):
        return mhc.MHC({key: fn}, **kwargs), ext


def launch(layer, key, rows, packets):
    x = torch.zeros(rows, 4096, dtype=torch.bfloat16)
    res = torch.zeros(rows, 4, 4096, dtype=torch.bfloat16)
    post, comb = torch.zeros(rows, 4, 1), torch.zeros(rows, 4, 4)
    scale, base, norm = torch.ones(3), torch.zeros(24), torch.ones(4096, dtype=torch.bfloat16)
    descriptor = torch.zeros(4, dtype=torch.int64) if packets else None
    return layer(key, x, res, post, comb, scale, base, norm, 1e-5, 1e-6, 2., 20, packets=descriptor)


@unittest.skipUnless(torch is not None, "requires torch")
class PackedRowsTests(unittest.TestCase):
    KEY = "L7.hc.ffn_fn"

    def setUp(self):
        from engine.base import kernel_shape
        kernel_shape.reset()
        self.addCleanup(kernel_shape.reset)

    def test_consumers_read_the_pack_through_sixteen_rows(self):
        fn = (torch.randn(24, 16384) * .006).bfloat16().float()
        for packed_rows in (16, 8):
            kwargs = {} if packed_rows == 16 else dict(packed_rows=8)
            layer, ext = owner(fn, self.KEY, **kwargs)
            fp32, packed = layer.weights[self.KEY]
            self.assertEqual(tuple(packed.shape), (24, 4096, 4))       # the consumers' [output, hidden, stream] layout
            for rows in range(1, 65):
                with self.subTest(packed_rows=packed_rows, rows=rows):
                    small = rows <= packed_rows
                    weight = (packed if small else fp32).data_ptr()
                    launch(layer, self.KEY, rows, packets=True)
                    self.assertEqual(ext.calls[-1], ("packets", rows, weight, small, False))
                    launch(layer, self.KEY, rows, packets=False)
                    self.assertEqual(ext.calls[-1], ("ordinary", rows, weight, small, small))

    def test_lossy_coefficients_stay_fp32_at_every_width(self):
        fn = torch.randn(24, 16384) * .006
        fn[0, 0] = 1. + 2. ** -20                                          # not a BF16 value
        layer, ext = owner(fn, self.KEY)
        fp32, packed = layer.weights[self.KEY]
        self.assertIsNone(packed)
        for rows in (1, 8, 9, 16, 17, 64):
            launch(layer, self.KEY, rows, packets=True)
            self.assertEqual(ext.calls[-1], ("packets", rows, fp32.data_ptr(), False, False))
            launch(layer, self.KEY, rows, packets=False)
            self.assertEqual(ext.calls[-1], ("ordinary", rows, fp32.data_ptr(), False, rows <= 16))

    def test_only_the_probe_control_is_accepted(self):
        fn = (torch.randn(24, 16384) * .006).bfloat16().float()
        for rows in (0, 9, 12, 32, 16., True, None):
            with self.subTest(rows=rows), self.assertRaisesRegex(ValueError, "packed rows"):
                owner(fn, self.KEY, packed_rows=rows)


@unittest.skipUnless(torch is not None, "requires torch")
class ProbeAdapterTests(unittest.TestCase):
    """The GPU probe's arms swap only the weight pointer and the launch flags of the owner's own arguments, and its
    statement of the served dispatch is the owner's."""

    def setUp(self):
        from engine.base import kernel_shape
        kernel_shape.reset()
        self.addCleanup(kernel_shape.reset)

    def test_arms_rewrite_only_weight_and_flags(self):
        from probes.engine_mhc_c2_packed import Arm
        fn = (torch.randn(24, 16384) * .006).bfloat16().float()
        layer, ext = owner(fn, "k")
        fp32, packed = layer.weights["k"]
        ptrs = list(range(18))
        for source in (fp32, packed):
            ptrs[4] = source.data_ptr()
            Arm(layer, packed=False).run_mhc_packets(ptrs, [], [16, 20], None, True)
            self.assertEqual(ext.calls[-1], ("packets", 16, fp32.data_ptr(), False, False))
            Arm(layer, packed=True).run_mhc_packets(ptrs, [], [16, 20], None, False)
            self.assertEqual(ext.calls[-1], ("packets", 16, packed.data_ptr(), True, False))
            Arm(layer, packed=True, consumer=True).run_mhc(ptrs, [], [16, 20], False, False)
            self.assertEqual(ext.calls[-1], ("ordinary", 16, packed.data_ptr(), True, True))
            Arm(layer, packed=False, consumer=False).run_mhc(ptrs, [], [8, 20], True, True)
            self.assertEqual(ext.calls[-1], ("ordinary", 8, fp32.data_ptr(), False, False))
        self.assertEqual(ptrs[5:], list(range(5, 18)))

    def test_records_match_the_stated_dispatch(self):
        from probes.engine_mhc_c2_packed import Record, served_dispatch
        fn = (torch.randn(24, 16384) * .006).bfloat16().float()
        for packed_rows in (8, 16):
            kwargs = {} if packed_rows == 16 else dict(packed_rows=8)
            layer, _ = owner(fn, "k", **kwargs)
            record = Record(layer)
            layer.ext = record
            for rows in (1, 8, 9, 16, 17, 32):
                for packets in (True, False):
                    launch(layer, "k", rows, packets)
                    self.assertEqual(record.calls[-1], served_dispatch(rows, packets=packets, packed_rows=packed_rows))


if __name__ == "__main__":
    unittest.main()
