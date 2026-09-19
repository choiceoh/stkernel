"""The MTP head's experts in the export's own FP8 (engine/profiles/qwen38/mtp_fp8.py side files, kernels/moe_fp8_rows):
the side file's specs, the net's specs and bind with them, the eager step's routes made local, and the kernel held to
its torch form.

    TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES= python3 -m unittest tests.test_engine_qwen38_mtp_fp8
"""
import importlib.util
import os
import unittest
from unittest import mock

torch = None
if importlib.util.find_spec("torch"):
    import torch
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
READY = torch is not None and importlib.util.find_spec("triton") is not None


@unittest.skipUnless(torch is not None, "torch required")
class SpecTests(unittest.TestCase):
    def facts(self):
        from probes.engine_qwen38_cells import facts
        import dataclasses
        return dataclasses.replace(facts(), mtp_experts="fp8_block", mtp_block=128)

    def test_the_side_file_holds_the_export_s_fp8(self):
        from engine.profiles.qwen38 import specs
        F = self.facts()
        side = specs.mtp_fp8_specs(F)
        self.assertEqual([s.name for s in side], list(specs.MTP_FP8))
        E, I, H = F.experts_local, F.moe_inter, F.hidden
        self.assertEqual([tuple(s.shape) for s in side], [(E, 2 * I, H), (E, 2 * I // 128, H // 128), (E, H, I),
                                                          (E, H // 128, I // 128)])
        self.assertEqual([s.dtype for s in side], [torch.float8_e4m3fn, torch.float32, torch.float8_e4m3fn, torch.float32])

    def test_a_bf16_export_has_no_fp8_to_serve(self):
        import dataclasses
        from engine.profiles.qwen38 import specs
        with self.assertRaises(ValueError):
            specs.mtp_fp8_specs(dataclasses.replace(self.facts(), mtp_experts="bf16", mtp_block=0))

    def test_the_net_binds_the_side_file_in_place_of_the_nvfp4_experts(self):
        from engine.profiles.qwen38 import specs
        from engine.profiles.qwen38.net import Qwen38Net
        F = self.facts()
        net = object.__new__(Qwen38Net)
        net.F, net.layers, net.mtp, net.mtp_experts = F, [0], True, "fp8"
        names = {s.name for s in net.specs()}
        self.assertFalse(names & set(specs.MTP_NVFP4))
        self.assertTrue(set(specs.MTP_FP8) <= names)
        self.assertEqual({s.name for s in net.side_specs()}, set(specs.MTP_FP8))
        net.mtp_experts = "nvfp4"
        self.assertEqual(net.side_specs(), [])
        self.assertTrue(set(specs.MTP_NVFP4) <= {s.name for s in net.specs()})

    def test_an_eager_step_s_routes_are_made_this_rank_s(self):
        from engine.profiles.qwen38.net import Qwen38Net
        net = object.__new__(Qwen38Net)
        net.first_expert = 128
        seen = {}

        def run(x, ids, weights, *rest):
            seen["ids"], seen["weights"] = ids.clone(), weights.clone()
            return x
        net.lanes = mock.Mock(moe_fp8=run)
        w13 = torch.empty(128, 2, 2)
        ids = torch.tensor([[130, 5, 255]], dtype=torch.int32)
        weights = torch.tensor([[0.5, 0.3, 0.2]])
        net._moe_fp8(torch.zeros(1, 4), ids, weights, w13=w13, s13=None, w2=None, s2=None, compact=True)
        self.assertEqual(seen["ids"].tolist(), [[2, 0, 127]])
        self.assertTrue(torch.equal(seen["weights"], torch.tensor([[0.5, 0.0, 0.2]])))
        net._moe_fp8(torch.zeros(1, 4), ids, weights, w13=w13, s13=None, w2=None, s2=None, local=True)
        self.assertEqual(seen["ids"].tolist(), ids.tolist())                     # a captured step's are already local


@unittest.skipUnless(READY and (INTERPRET or (torch is not None and torch.cuda.is_available())),
                     "CUDA or Triton interpreter required")
class KernelTests(unittest.TestCase):
    def test_the_kernel_is_its_torch_form(self):
        from engine.kernels import moe_fp8_rows
        got = moe_fp8_rows.qualify("cpu" if INTERPRET else "cuda")
        self.assertEqual(set(got), {1, 4})


if __name__ == "__main__":
    unittest.main()
