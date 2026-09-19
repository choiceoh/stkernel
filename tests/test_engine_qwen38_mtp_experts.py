"""The MTP head's experts from side files (engine/profiles/qwen38/mtp_side.py, kernels/moe_rows): the checkpoint's
original BF16 by default (the operator's rule of 2026-09-19), the export's FP8 on request -- the side files' specs, the
net's specs and bind with them, the eager step's routes made local, the kernel held to its torch form at both, and the
fleet and launcher defaults.

    TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES= python3 -m unittest tests.test_engine_qwen38_mtp_experts
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

    def test_the_bf16_side_file_slices_the_fused_experts(self):
        import dataclasses
        from engine.profiles.qwen38 import specs
        F = dataclasses.replace(self.facts(), mtp_experts="bf16", mtp_block=0)
        side = specs.mtp_bf16_specs(F)
        E, I, H = F.experts_local, F.moe_inter, F.hidden
        self.assertEqual([(s.name, tuple(s.shape), s.dtype) for s in side],
                         [(specs.MTP_BF16[0], (E, 2 * I, H), torch.bfloat16), (specs.MTP_BF16[1], (E, H, I), torch.bfloat16)])
        full = {"mtp.layers.0.mlp.experts.gate_up_proj": torch.arange(4 * E, dtype=torch.float32).view(4 * E, 1, 1),
                "mtp.layers.0.mlp.experts.down_proj": torch.arange(4 * E, dtype=torch.float32).view(4 * E, 1, 1)}
        lo, hi = F.expert_range(2)
        self.assertEqual(side[0].build(full, 2, 4).flatten().tolist(), list(range(lo, hi)))   # rank 2's experts, in order
        from engine.profiles.qwen38 import mtp_side
        with self.assertRaises(ValueError):
            mtp_side.check_source(self.facts(), "bf16")             # the FP8 export keeps no BF16 experts to write from
        mtp_side.check_source(F, "bf16")

    def test_the_served_facts_bind_either_side_file(self):
        """The rank files' facts are the NVIDIA export's ("fp8_block"); a boot binds the BF16 side file all the same --
        the check is the writer's (the first fleet default of #1235 raised here)."""
        from engine.profiles.qwen38 import specs
        from engine.profiles.qwen38.net import Qwen38Net
        for precision, names in (("bf16", specs.MTP_BF16), ("fp8", specs.MTP_FP8)):
            net = object.__new__(Qwen38Net)
            net.F, net.layers, net.mtp, net.mtp_experts = self.facts(), [0], True, precision
            self.assertEqual({s.name for s in net.side_specs()}, set(names), precision)
            self.assertTrue(set(names) <= {s.name for s in net.specs()}, precision)

    def test_the_fleet_and_the_launcher_serve_bf16_by_default(self):
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        fleet = (root / "engine/profiles/qwen38/fleet.py").read_text()
        self.assertIn('ap.add_argument("--mtp-experts", choices=("bf16", "fp8", "nvfp4"), default="bf16",', fleet)
        self.assertIn('mtp_experts: str = "bf16", mtp_experts_dir', fleet)
        launcher = (root / "launchers/start-st-qwen38.sh").read_text()
        self.assertIn("MTP_EXPERTS=${ST_MTP_EXPERTS:-bf16}", launcher)
        self.assertIn('test -f $EXPERTS_DIR/mtp-$MTP_EXPERTS-r${r}of4.safetensors', launcher)
        from engine.profiles.qwen38 import mtp_side
        self.assertEqual(mtp_side.DIRS["bf16"], "/home/choiceoh/models/st-qwen38-mtp-bf16")
        self.assertEqual(mtp_side.path("/d", 3, "bf16").name, "mtp-bf16-r3of4.safetensors")

    def test_a_bf16_copy_has_no_fp8_to_write(self):
        import dataclasses
        from engine.profiles.qwen38 import mtp_side
        with self.assertRaises(ValueError):
            mtp_side.check_source(dataclasses.replace(self.facts(), mtp_experts="bf16", mtp_block=0), "fp8")

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
        import dataclasses
        net.F, net.mtp_experts = dataclasses.replace(F, mtp_experts="bf16", mtp_block=0), "bf16"
        names = {s.name for s in net.specs()}
        self.assertFalse(names & set(specs.MTP_NVFP4))
        self.assertEqual({s.name for s in net.side_specs()}, set(specs.MTP_BF16))
        self.assertTrue(set(specs.MTP_BF16) <= names)
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
        net.lanes = mock.Mock(moe_rows=run)
        w13 = torch.empty(128, 2, 2)
        ids = torch.tensor([[130, 5, 255]], dtype=torch.int32)
        weights = torch.tensor([[0.5, 0.3, 0.2]])
        net._moe_rows(torch.zeros(1, 4), ids, weights, w13=w13, s13=None, w2=None, s2=None, compact=True)
        self.assertEqual(seen["ids"].tolist(), [[2, 0, 127]])
        self.assertTrue(torch.equal(seen["weights"], torch.tensor([[0.5, 0.0, 0.2]])))
        net._moe_rows(torch.zeros(1, 4), ids, weights, w13=w13, s13=None, w2=None, s2=None, local=True)
        self.assertEqual(seen["ids"].tolist(), ids.tolist())                     # a captured step's are already local


@unittest.skipUnless(READY and (INTERPRET or (torch is not None and torch.cuda.is_available())),
                     "CUDA or Triton interpreter required")
class KernelTests(unittest.TestCase):
    def test_the_kernel_is_its_torch_form_at_both_precisions(self):
        from engine.kernels import moe_rows
        for precision in ("bf16", "fp8"):
            # the interpreter truncates a BF16 cast where a GPU rounds it: twice the boot's band there
            got = moe_rows.qualify("cpu" if INTERPRET else "cuda", precision=precision,
                                   band=2.0 ** -5 if INTERPRET else 2.0 ** -6)
            self.assertEqual(set(got), {1, 4}, precision)


if __name__ == "__main__":
    unittest.main()
