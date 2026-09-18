"""The served MoE activation is GLM's clamped SiLU-GLU, whatever the kernel calls it.

The b12x lanes name their activation `swigluoai_uninterleave` -- gpt-oss's
`gate * sigmoid(alpha * gate) * (up + beta)` with clamps, whose kernel-side defaults are
alpha 1.702 and beta 1.0. GLM-5.3 (transformers glm5_next `Glm5NextTextExperts._apply_gate`)
is `silu(min(gate, limit)) * clamp(up, -limit, limit)`: the same function only at alpha 1,
beta 0. These pin that every served call site passes those constants and that the two
formulas then agree, so a kernel default can never leak into the served model.
"""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ServedCallSitesTests(unittest.TestCase):
    def test_every_served_swigluoai_launch_pins_alpha_one_and_beta_zero(self):
        lanes = (ROOT / "engine/profiles/glm53/lanes.py").read_text()
        self.assertIn('activation="swigluoai_uninterleave", swiglu_alpha=1.0, swiglu_beta=0.0, swiglu_limit=float(limit)', lanes)
        self.assertIn('activation="silu",\n                    swiglu_alpha=1.0,\n                    swiglu_beta=0.0,', lanes)
        packets = (ROOT / "engine/kernels/b12x/moe_packet_input.py").read_text()
        self.assertIn("activation='swigluoai_uninterleave', swiglu_alpha=1., swiglu_beta=0., swiglu_limit=10.", packets)
        dispatch = (ROOT / "engine/kernels/b12x/moe_dispatch.py").read_text()
        self.assertIn('("swigluoai_uninterleave", 1.0, 0.0, 10.0, "nvfp4")', dispatch)   # the static lanes refuse anything else


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "requires PyTorch")
class FormulaTests(unittest.TestCase):
    @staticmethod
    def swigluoai(g, u, *, alpha, beta, limit):
        """gated_activation_f32's swigluoai branch (engine/kernels/b12x/moe_activation.py), in torch."""
        import torch
        g = torch.minimum(g.float(), torch.tensor(limit))
        u = torch.maximum(torch.minimum(u.float(), torch.tensor(limit)), torch.tensor(-limit))
        return g * torch.sigmoid(alpha * g) * (u + beta)

    def test_alpha_one_beta_zero_is_glms_clamped_silu_glu_and_the_kernel_defaults_are_not(self):
        import torch
        from engine.profiles.glm53.lanes import swiglu_clamped
        g = torch.randn(64, 32, generator=torch.Generator().manual_seed(0)) * 8
        u = torch.randn(64, 32, generator=torch.Generator().manual_seed(1)) * 8
        glm = swiglu_clamped(g, u, 10.0).float()
        torch.testing.assert_close(self.swigluoai(g, u, alpha=1.0, beta=0.0, limit=10.0).bfloat16().float(), glm)
        self.assertGreater((self.swigluoai(g, u, alpha=1.702, beta=1.0, limit=10.0) - glm).abs().max(), 1.0)


if __name__ == "__main__":
    unittest.main()
