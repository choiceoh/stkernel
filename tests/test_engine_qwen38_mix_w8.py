"""Opt-in precision routing, prefetch extent, and calibration isolation."""
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch
from engine.profiles.qwen38.net import Qwen38Net


class MixerW8Tests(unittest.TestCase):
    def net(self):
        net = Qwen38Net.__new__(Qwen38Net)
        net.F = NS(hc=4, hc_rank=320, rms_eps=1e-6)
        q = torch.empty(384, 10240, dtype=torch.float8_e4m3fn)
        net._hc_w8 = {"site.": ((q, None), (None, None))}
        net._hc_projections = {}
        net.p = {"site.down_inject": object(), "site.up": object(), "site.norm": object()}
        net.lanes = NS(hc_mix=Mock(return_value="bf16"), hc_site=Mock(return_value="whole"))
        return net

    def test_small_rows_take_packed_weights_and_wide_rows_keep_originals(self):
        net = self.net()
        with patch("engine.kernels.gated_residual_w8.mix", return_value="w8") as mix:
            for rows in (1, 4, 8, 16):
                self.assertEqual(net._mix("site.", torch.zeros(rows, 10240).bfloat16(), "down_inject", inject=True), "w8")
            self.assertEqual(mix.call_count, 4)
            self.assertEqual(net._mix("site.", torch.zeros(17, 10240).bfloat16(), "down_inject", inject=True), "bf16")
        self.assertEqual(net.lanes.hc_mix.call_args.args[1:3], (net.p["site.down_inject"], net.p["site.up"]))

    def test_prefetch_view_covers_exactly_the_packed_bytes(self):
        net = self.net()
        packed = net._hc_w8["site."][0][0]
        view = net._mixer_weight("site.", "down_inject", 4)
        self.assertEqual(view.data_ptr(), packed.data_ptr())
        self.assertEqual(view.numel() * view.element_size(), packed.numel())
        self.assertEqual(view.dtype, torch.bfloat16)
        self.assertIs(net._mixer_weight("site.", "down_inject", 17), net.p["site.down_inject"])

    def test_small_sites_reach_the_selected_mixer_and_wide_sites_keep_prefill_fusion(self):
        net = self.net()
        h = torch.zeros(4, 10240).bfloat16()
        self.assertIsNone(net._whole_site("site.", h, None, None, "down_inject", injects=True))
        self.assertEqual(net._whole_site("site.", h.repeat(5, 1), None, None, "down_inject", injects=True), "whole")
        net._hc_w8.clear()
        self.assertEqual(net._whole_site("site.", h, None, None, "down_inject", injects=True), "whole")

    def test_w8_calibration_cannot_read_the_bf16_or_old_fp8_identity(self):
        from engine.profiles.qwen38.calibration import identity
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "rank"
            file.write_bytes(b"same weights")
            args = ({}, [file], {})
            base = identity(*args, hc_fp8=False)
            self.assertEqual(base, identity(*args, hc_fp8=False, hc_w8a16=False))
            cand = identity(*args, hc_fp8=False, hc_w8a16=True)
            self.assertNotEqual(base, cand)
            self.assertNotEqual(identity(*args, hc_fp8=True), cand)

    def test_conflicting_precision_modes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            Qwen38Net(NS(), NS(world_size=4), NS(), hc_fp8=True, hc_w8a16=True)


if __name__ == "__main__":
    unittest.main()
