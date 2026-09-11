"""Judge the composed KDA output against the model's gated-norm module.

Small core activations expose an incorrect epsilon even when recurrence
and cache tests agree with each other. No GPU or model weights are needed.
"""
import importlib.util
from types import SimpleNamespace
import unittest
from dataclasses import replace


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("triton"),
                     "requires PyTorch and the vendored norm module's Triton imports")
class KdaOutputNormTests(unittest.TestCase):
    def test_composition_matches_default_glm_gated_norm_at_small_core_magnitudes(self):
        import torch
        from engine.kernels.kda.kda import FusedRMSNormGated
        from engine.profiles.glm53 import lanes
        from engine.profiles.glm53.net import Glm53Net, Step
        from test_engine_glm53 import tiny_facts

        F = tiny_facts()
        comm = SimpleNamespace(rank=0, world_size=4, all_reduce=lambda x: x)
        net = Glm53Net(F, comm, lanes.reference(), layers=[0])
        net.p = {s.name: torch.zeros(s.shape, dtype=s.dtype) for s in net.specs()
                 if s.name.startswith("L0.kda.")}
        width = F.kda_heads_local * F.kda_dim
        net.p["L0.kda.o_proj"] = torch.eye(F.hidden, width, dtype=torch.bfloat16)
        net.p["L0.kda.o_norm"] = torch.linspace(.1, .8, F.kda_dim).bfloat16()
        # This is the class GLM constructs without specifying an epsilon.
        oracle = FusedRMSNormGated(F.kda_dim, activation="sigmoid", dtype=torch.bfloat16)
        oracle.weight.data.copy_(net.p["L0.kda.o_norm"])
        for tokens in (1, 6, 7):  # decode, speculative verify, chunk prefill
            for magnitude in (1e-5, 1e-3, 1e-1):
                with self.subTest(tokens=tokens, magnitude=magnitude):
                    core = (torch.linspace(-magnitude, magnitude, F.kda_dim)
                            .repeat(1, tokens, F.kda_heads_local, 1).bfloat16())
                    state = torch.zeros(F.kda_heads_local, F.kda_dim, F.kda_dim)
                    net.lanes = replace(lanes.reference(),
                        kda_chunk=lambda *args: (core, state[None]),
                        kda_recurrent=lambda *args: (core, state.repeat(tokens, 1, 1, 1)))
                    conv = torch.zeros(3*width, net.conv_ring, dtype=torch.bfloat16)
                    rec = torch.zeros(net.rec_ring, *state.shape)
                    cache = SimpleNamespace(kda=lambda *args: (conv, rec))
                    x = torch.zeros(tokens, F.hidden, dtype=torch.bfloat16)
                    step = Step.prefill(torch.zeros(tokens, dtype=torch.int64), 0, 0, 1)
                    got = net._kda(0, x, step, cache)
                    normed = oracle.forward_native(core, torch.zeros_like(core))
                    expected = torch.nn.functional.linear(normed.reshape(tokens, width), net.p["L0.kda.o_proj"])
                    torch.testing.assert_close(got, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
