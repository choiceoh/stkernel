"""MK mHC coefficient storage by row count, on the CPU: which weight and which native form each launch gets.

C=1 (8 rows) and C=2 (16 rows) packet consumers read the lossless BF16 pack; above 8 rows its coefficients expand
once per block. The ordinary (all-reduce) boundary keeps the 8-row consumer. GPU numerics and timing are
probes/engine_mhc_c2_packed.py (measurements/st_c2_mhc_packed_20260915).
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
        self.calls.append(("ordinary", ints[0], ptrs[4], bf16_fn, ar_consumer, False))

    def run_mhc_packets(self, ptrs, scalars, ints, packets, bf16_fn, expand_once):
        self.calls.append(("packets", ints[0], ptrs[4], bf16_fn, False, expand_once))


@unittest.skipUnless(torch is not None, "requires torch")
class PackedRowsTests(unittest.TestCase):
    KEY = "L7.hc.ffn_fn"

    def setUp(self):
        from engine.base import kernel_shape
        kernel_shape.reset()
        self.addCleanup(kernel_shape.reset)

    def owner(self, fn, **kwargs):
        from engine.kernels.dense import mhc
        ext = Recorder()
        with patch.object(mhc, "extension", lambda: ext):
            return mhc.MHC({self.KEY: fn}, **kwargs), ext

    def launch(self, owner, rows, packets):
        x = torch.zeros(rows, 4096, dtype=torch.bfloat16)
        res = torch.zeros(rows, 4, 4096, dtype=torch.bfloat16)
        post, comb = torch.zeros(rows, 4, 1), torch.zeros(rows, 4, 4)
        scale, base, norm = torch.ones(3), torch.zeros(24), torch.ones(4096, dtype=torch.bfloat16)
        descriptor = torch.zeros(4, dtype=torch.int64) if packets else None
        owner(self.KEY, x, res, post, comb, scale, base, norm, 1e-5, 1e-6, 2., 20, packets=descriptor)

    def test_packet_consumers_read_the_pack_through_sixteen_rows(self):
        fn = (torch.randn(24, 16384) * .006).bfloat16().float()
        for packet_rows in (16, 8):
            kwargs = {} if packet_rows == 16 else dict(packet_rows=8)
            owner, ext = self.owner(fn, **kwargs)
            fp32, packed = owner.weights[self.KEY]
            self.assertEqual(tuple(packed.shape), (24, 4096, 4))       # the consumers' [output, hidden, stream] layout
            for rows in range(1, 65):
                with self.subTest(packet_rows=packet_rows, rows=rows):
                    self.launch(owner, rows, packets=True)
                    lossless = rows <= packet_rows
                    self.assertEqual(ext.calls[-1], ("packets", rows, (packed if lossless else fp32).data_ptr(),
                                                     lossless, False, lossless and rows > 8))
                    self.launch(owner, rows, packets=False)
                    small = rows <= 8
                    self.assertEqual(ext.calls[-1], ("ordinary", rows, (packed if small else fp32).data_ptr(),
                                                     small, small, False))

    def test_lossy_coefficients_stay_fp32_at_every_width(self):
        fn = torch.randn(24, 16384) * .006
        fn[0, 0] = 1. + 2. ** -20                                          # not a BF16 value
        owner, ext = self.owner(fn)
        fp32, packed = owner.weights[self.KEY]
        self.assertIsNone(packed)
        for rows in (1, 8, 9, 16, 17, 64):
            self.launch(owner, rows, packets=True)
            self.assertEqual(ext.calls[-1], ("packets", rows, fp32.data_ptr(), False, False, False))
            self.launch(owner, rows, packets=False)
            self.assertEqual(ext.calls[-1], ("ordinary", rows, fp32.data_ptr(), False, rows <= 8, False))

    def test_only_the_probe_control_is_accepted(self):
        fn = (torch.randn(24, 16384) * .006).bfloat16().float()
        for rows in (0, 9, 12, 32, 16., True, None):
            with self.subTest(rows=rows), self.assertRaisesRegex(ValueError, "packet rows"):
                self.owner(fn, packet_rows=rows)


@unittest.skipUnless(torch is not None, "requires torch")
class ProbeAdapterTests(unittest.TestCase):
    """The GPU probe's arms swap only the weight pointer and the launch flags of the owner's own arguments."""

    def test_arms_and_records(self):
        from probes.engine_mhc_c2_packed import Arm, Record, served_dispatch
        from engine.base import kernel_shape
        from engine.kernels.dense import mhc
        kernel_shape.reset()
        self.addCleanup(kernel_shape.reset)
        ext = Recorder()
        fn = (torch.randn(24, 16384) * .006).bfloat16().float()
        with patch.object(mhc, "extension", lambda: ext):
            owner = mhc.MHC({"k": fn})
        fp32, packed = owner.weights["k"]
        ptrs = list(range(18))
        for source in (fp32, packed):
            ptrs[4] = source.data_ptr()
            Arm(owner, packed=False).run_mhc_packets(ptrs, [], [16, 20], None, True, True)
            self.assertEqual(ext.calls[-1], ("packets", 16, fp32.data_ptr(), False, False, False))
            Arm(owner, packed=True, expand=True).run_mhc_packets(ptrs, [], [16, 20], None, False, False)
            self.assertEqual(ext.calls[-1], ("packets", 16, packed.data_ptr(), True, False, True))
            Arm(owner, packed=True, consumer=True).run_mhc(ptrs, [], [16, 20], False, False)
            self.assertEqual(ext.calls[-1], ("ordinary", 16, packed.data_ptr(), True, True, False))
            Arm(owner, packed=False, consumer=False).run_mhc(ptrs, [], [8, 20], True, True)
            self.assertEqual(ext.calls[-1], ("ordinary", 8, fp32.data_ptr(), False, False, False))
        record = Record(owner)
        owner.ext = record
        rows = torch.zeros(16, 4096, dtype=torch.bfloat16)
        owner("k", rows, torch.zeros(16, 4, 4096, dtype=torch.bfloat16), torch.zeros(16, 4, 1), torch.zeros(16, 4, 4),
              torch.ones(3), torch.zeros(24), torch.ones(4096, dtype=torch.bfloat16), 1e-5, 1e-6, 2., 20,
              packets=torch.zeros(4, dtype=torch.int64))
        self.assertEqual(record.calls, [served_dispatch(16, packets=True, packet_rows=16)])
        self.assertEqual(served_dispatch(16, packets=True, packet_rows=16), (16, True, True, False, True))
        self.assertEqual(served_dispatch(16, packets=True, packet_rows=8), (16, False, False, False, False))
        self.assertEqual(served_dispatch(8, packets=False, packet_rows=16), (8, True, True, True, False))


if __name__ == "__main__":
    unittest.main()
