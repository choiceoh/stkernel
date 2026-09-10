"""CPU contracts for generalized MHC and two intentionally distinct math seams."""
import importlib.util
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = load(ROOT / "probes/mk_mhc_geometry_bench.py", "mhc_geometry_probe")
try:
    import torch
except ImportError:
    torch = None


class MetadataTests(unittest.TestCase):
    def test_exact_geometry_and_boundaries_without_torch(self):
        for hidden in (4096, 5120):
            for tokens in probe.TOKENS:
                self.assertTrue(probe.geometry_eligible(tokens, 4, hidden))
        for args in ((0, 4, 4096), (33, 4, 5120), (6, 3, 5120),
                     (6, 4, 8192), (True, 4, 5120), (6, 4, 5120.0)):
            self.assertFalse(probe.geometry_eligible(*args))

    def test_exact_metadata_rejects_shape_and_dtype_aliases(self):
        shapes = dict(zip(("x", "residual", "post", "comb", "fn", "scale", "base", "norm"),
                          ((6, 5120), (6, 4, 5120), (6, 4), (6, 4, 4),
                           (24, 20480), (3,), (24,), (5120,))))
        dtypes = {k: "torch.bfloat16" if k in ("x", "residual", "norm") else "torch.float32"
                  for k in shapes}
        self.assertEqual(probe.validate_metadata(shapes, dtypes), (6, 5120))
        for key in shapes:
            changed = dict(shapes, **{key: (1,)})
            with self.assertRaises(ValueError):
                probe.validate_metadata(changed, dtypes)
            with self.assertRaises(ValueError):
                probe.validate_metadata(shapes, dict(dtypes, **{key: "torch.float16"}))

    def test_invalid_arithmetic_parameters_fail_closed(self):
        values = [1e-20, 1e-20, 1e-6, 1e-6, 2.0, 20]
        probe._parameters(*values)
        for position in range(5):
            for bad in (0, -1, math.nan, math.inf, True):
                changed = values.copy()
                changed[position] = bad
                with self.assertRaises(ValueError):
                    probe._parameters(*changed)
        for bad in (0, -1, True, 20.0):
            with self.assertRaises(ValueError):
                probe._parameters(*values[:-1], bad)

    def test_probe_import_does_not_load_model_or_cuda(self):
        name = "mhc_geometry_import_only"
        before = set(sys.modules)
        loaded = load(ROOT / "probes/mk_mhc_geometry_bench.py", name)
        added = set(sys.modules) - before
        self.assertFalse(any(m.startswith(("vllm", "tilelang")) for m in added))
        self.assertEqual(loaded.HIDDENS, (4096, 5120))


class Device:
    def __init__(self, value="cuda:0"):
        if isinstance(value, Device):
            self.type, self.index = value.type, value.index
        else:
            self.type, _, index = str(value).partition(":")
            self.index = int(index) if index else None

    def __eq__(self, other):
        return isinstance(other, Device) and (self.type, self.index) == (other.type, other.index)


class Tensor:
    next_ptr = 4096

    def __init__(self, shape, dtype="bf16", device="cuda:0", contiguous=True):
        self.shape, self.dtype = tuple(shape), dtype
        self.device = Device(device)
        self.is_cuda = self.device.type == "cuda"
        self.contiguous = contiguous
        self.ptr = Tensor.next_ptr
        Tensor.next_ptr += 4096

    def data_ptr(self):
        return self.ptr

    def is_contiguous(self):
        return self.contiguous


class DriverTests(unittest.TestCase):
    def setUp(self):
        self.driver = load(ROOT / "overlay/modules/glm53_megakernel/glm53_megakernel.py",
                           "mhc_geometry_driver_test")
        self.capture, self.current = False, 0
        self.calls = []
        self.fake = SimpleNamespace(
            device=Device, bfloat16="bf16", float32="f32", int32="i32",
            cuda=SimpleNamespace(is_current_stream_capturing=lambda: self.capture,
                                 current_device=lambda: self.current),
            zeros=lambda *shape, dtype="f32", device=None: Tensor(shape, dtype, device),
            empty=lambda *shape, dtype="f32", device=None: Tensor(shape, dtype, device),
            empty_like=lambda t: Tensor(t.shape, t.dtype, t.device))
        self.driver._EXT = SimpleNamespace(
            run_mhc=lambda *args: self.calls.append(("legacy", args)),
            run_mhc_v41=lambda *args: self.calls.append(("v41", args)))
        self.driver._ar_note = lambda *_: None
        self.driver.ENABLE_AR_CONSUMER = self.driver.ENABLE_MHC_BF16 = False

    def values(self, hidden=5120, tokens=6):
        return (Tensor((tokens, hidden)), Tensor((tokens, 4, hidden)),
                Tensor((tokens, 4), "f32"), Tensor((tokens, 4, 4), "f32"),
                Tensor((24, 4 * hidden), "f32"), Tensor((3,), "f32"),
                Tensor((24,), "f32"), Tensor((hidden,)))

    def launch(self, values, contract="legacy", pre=None):
        with patch.dict(sys.modules, {"torch": self.fake}):
            return self.driver._mhc_call(*values, values[0].shape[0],
                                         1e-20, 1e-6, 1e-6, 2., 1e-20, 20,
                                         _contract=contract, _collapse_pre_mix=pre)

    def test_original_hook_stays_4096_and_other_segments_keep_geometry(self):
        d = self.driver
        self.assertTrue(d._mk_mhc_eligible(6, 4, 4096))
        self.assertFalse(d._mk_mhc_eligible(6, 4, 5120))
        self.assertEqual((d.HIDDEN, d.NCHUNK, d.KDA_H, d.KDA_D, d.MLA_D, d.MLA_H),
                         (4096, 16, 16, 128, 512, 16))
        self.assertFalse(d._mk_gemm_eligible(6, 5120, 5120))

    def test_exact_legacy_and_v41_pointer_abis(self):
        for hidden, mode, arity, pointers in ((4096, "legacy", 5, 18),
                                             (5120, "legacy", 6, 18),
                                             (4096, "v41", 4, 20),
                                             (5120, "v41", 4, 20)):
            values = self.values(hidden)
            pre = Tensor((6, 4), "f32") if mode == "v41" else None
            out = self.launch(values, mode, pre)
            actual_mode, args = self.calls[-1]
            self.assertEqual(actual_mode, mode)
            self.assertEqual((len(args), len(args[0])), (arity, pointers))
            self.assertEqual(args[0][:8], [v.data_ptr() for v in values])
            self.assertEqual(args[1], [1e-20, 1e-6, 1e-6, 2., 1e-20])
            self.assertEqual(args[2], [6, 20])
            if mode == "v41":
                self.assertEqual(len(out), 5)
                self.assertEqual(args[0][-2:], [pre.data_ptr(), out[-1].data_ptr()])
                self.assertEqual(args[-1], hidden)
            else:
                self.assertEqual(len(out), 4)
                self.assertEqual(args[3:5], (False, False))
                if hidden == 5120:
                    self.assertEqual(args[-1], 5120)

    def test_workspace_retains_pointers_and_separates_geometry_and_contract(self):
        with patch.dict(sys.modules, {"torch": self.fake}):
            old = self.driver._ensure_mhc_workspace("cuda:0", 4096)
            large = self.driver._ensure_mhc_workspace("cuda:0", 5120)
            v41 = self.driver._ensure_mhc_workspace("cuda:0", 5120, "v41")
            self.assertIs(old, self.driver._WS)
            self.assertIs(large, self.driver._ensure_mhc_workspace("cuda", 5120))
            self.assertEqual(large["yp"].shape, (20 * 32 * 24,))
            self.assertEqual(large["rp"].shape, (20 * 32,))
            self.assertEqual(large["ol_stash"].shape, (32 * 5120,))
            self.assertTrue({v.ptr for v in large.values()}.isdisjoint({v.ptr for v in v41.values()}))
            self.capture = True
            self.assertIs(large, self.driver._ensure_mhc_workspace("cuda:0", 5120))
            with self.assertRaises(RuntimeError):
                self.driver._ensure_mhc_workspace("cuda:0", 4096, "v41")

    def test_invalid_metadata_or_missing_pre_never_launches(self):
        for index in range(8):
            values = list(self.values())
            values[index].dtype = "wrong"
            with self.assertRaises(ValueError):
                self.launch(values)
        values = list(self.values())
        values[4].contiguous = False
        with self.assertRaises(ValueError):
            self.launch(values)
        values = list(self.values())
        values[0].device = Device("cuda:1")
        with self.assertRaises(ValueError):
            self.launch(values)
        with self.assertRaises(ValueError):
            self.launch(self.values(), "v41")
        with self.assertRaises(ValueError):
            self.launch(self.values(), "legacy", Tensor((6, 4), "f32"))
        self.current = 1
        with self.assertRaises(ValueError):
            self.launch(self.values())
        self.assertEqual(self.calls, [])


@unittest.skipIf(torch is None, "Torch CPU runtime is required for arithmetic contracts")
class ArithmeticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_both_hidden_sizes_have_correct_shapes_and_no_input_mutation(self):
        for hidden in probe.HIDDENS:
            values = probe.fixture(1, hidden, seed=hidden)
            originals = tuple(v.clone() for v in values)
            legacy = probe.legacy_fused_reference(*values)
            previous = torch.tensor([[.11, .29, .61, .83]], dtype=torch.float32)
            v41 = probe.v41_component_reference(*values, previous)
            for output in (legacy, v41):
                self.assertEqual([tuple(v.shape) for v in output[:4]],
                                 [(1, 4, hidden), (1, 4), (1, 4, 4), (1, hidden)])
                self.assertTrue(all(torch.isfinite(v).all() for v in output))
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(values, originals)))
            self.assertEqual(tuple(v41[4].shape), (1, 4))

    def test_sinkhorn_first_row_epsilon_and_twenty_rounds(self):
        # Independent scalar 4x4 reference, deliberately nonuniform logits.
        values = [math.sin(i) * 3 for i in range(16)]
        mixes = torch.tensor([[0.] * 8 + values], dtype=torch.float32)
        scale, base = torch.ones(3), torch.zeros(24)
        for iterations in (1, 20):
            _, post, got = probe.split_sinkhorn(mixes, scale, base,
                                                sinkhorn_iters=iterations)
            matrix = [values[i:i + 4] for i in range(0, 16, 4)]
            matrix = [[math.exp(v - max(row)) for v in row] for row in matrix]
            matrix = [[v / sum(row) + 1e-6 for v in row] for row in matrix]
            for step in range(iterations):
                if step:
                    matrix = [[v / (sum(row) + 1e-6) for v in row] for row in matrix]
                cols = [sum(row[j] for row in matrix) for j in range(4)]
                matrix = [[v / (cols[j] + 1e-6) for j, v in enumerate(row)] for row in matrix]
            self.assertTrue(torch.allclose(got[0], torch.tensor(matrix), atol=2e-7, rtol=1e-6))
            self.assertTrue(torch.equal(post, torch.ones(1, 4)))

    def test_legacy_projection_uses_unrounded_post(self):
        values = list(probe.fixture(1, 4096, seed=9))
        # Cancellation makes the rounding seam observable without nonfinite data.
        values[2].fill_(.333)
        values[3].fill_(.071)
        result = probe.legacy_fused_reference(*values)
        rounded_mixes = probe._projection(result[0], values[4], 1e-6)
        _, wrong_post, _ = probe.split_sinkhorn(rounded_mixes, values[5], values[6])
        self.assertFalse(torch.equal(result[1], wrong_post))

    def test_legacy_rms_uses_unrounded_weighted_sum_but_bf16_numerator(self):
        values = list(probe.fixture(1, 4096, seed=31))
        values[4].zero_()  # coefficients known from base, independent of projection
        got = probe.legacy_fused_reference(*values)
        pre, _, _ = probe.split_sinkhorn(torch.zeros(1, 24), values[5], values[6])
        weighted = torch.zeros_like(values[0], dtype=torch.float32)
        for k in range(4):
            weighted += pre[:, k:k + 1] * got[0][:, k].float()
        bf = weighted.to(torch.bfloat16).float()
        expected = (bf * torch.rsqrt(weighted.square().mean(-1, keepdim=True) + 1e-6)
                    * values[-1].float()).to(torch.bfloat16)
        self.assertTrue(torch.equal(got[3], expected))
        wrong = (bf * torch.rsqrt(bf.square().mean(-1, keepdim=True) + 1e-6)
                 * values[-1].float()).to(torch.bfloat16)
        self.assertFalse(torch.equal(got[3], wrong))

    def test_v41_previous_pre_is_independent_of_new_mix(self):
        values = probe.fixture(1, 5120, seed=77)
        previous = torch.tensor([[.1, .3, .6, .9]])
        first = probe.v41_component_reference(*values, previous)
        changed = list(values)
        changed[4] = changed[4] * 2.7
        second = probe.v41_component_reference(*changed, previous)
        self.assertTrue(torch.equal(first[3], second[3]))
        self.assertFalse(torch.equal(first[4], second[4]))
        third = probe.v41_component_reference(*values, previous.flip(-1))
        self.assertFalse(torch.equal(first[3], third[3]))
        self.assertTrue(torch.equal(first[4], third[4]))

    def test_v41_both_bf16_boundaries_and_norm_epsilon(self):
        values = probe.fixture(1, 5120, seed=12)
        previous = torch.tensor([[.13, .31, .57, .89]])
        got = probe.v41_component_reference(*values, previous)
        # Literal released model equations independent of legacy-fused oracle.
        x, residual, post, comb, fn, scale, base, norm = values
        r = (post.unsqueeze(-1) * x.unsqueeze(1) +
             torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(2), dim=1)).to(x.dtype)
        flat = r.flatten(1).float()
        mixes = torch.nn.functional.linear(flat, fn) * torch.rsqrt(
            flat.square().mean(-1, keepdim=True) + 1e-20)
        pre, pm, cm = probe.split_sinkhorn(mixes, scale, base)
        y = torch.sum(previous.unsqueeze(-1) * r.float(), dim=1).to(x.dtype).float()
        output = (norm.float() * (y * torch.rsqrt(
            y.square().mean(-1, keepdim=True) + 1e-20))).to(x.dtype)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(got, (r, pm, cm, output, pre))))
        tiny = [v.clone() for v in values]
        tiny[0].mul_(1e-8)
        tiny[1].mul_(1e-8)
        a = probe.v41_component_reference(*tiny, previous)
        b = probe.v41_component_reference(*tiny, previous, norm_eps=1e-6)
        self.assertFalse(torch.equal(a[3], b[3]))

    def test_nonfinite_or_metadata_mismatch_cannot_pass(self):
        values = probe.fixture(1, 4096)
        reference = probe.legacy_fused_reference(*values)
        self.assertTrue(all(r["passed"] for r in probe.compare_outputs(reference, reference)))
        bad = [v.clone() for v in reference]
        bad[0].flatten()[0] = math.nan
        self.assertFalse(probe.compare_outputs(bad, reference)[0]["passed"])
        huge = tuple(torch.full((32, 4), 3e38) for _ in range(4))
        self.assertFalse(any(row["passed"] for row in probe.compare_outputs(huge, huge)))
        with self.assertRaises(ValueError):
            probe.compare_outputs(reference[:-1], reference)
        with self.assertRaises(ValueError):
            probe.compare_outputs(reference, reference, tolerance=.01)
        with self.assertRaises(ValueError):
            probe.v41_component_reference(*values, torch.zeros(1, 3))

    def test_worst_token_and_zero_reference_cannot_hide_in_pooled_error(self):
        refs = tuple(torch.ones(32, 4) for _ in range(4))
        got = tuple(v.clone() for v in refs)
        got[0][0] *= 1.004
        row = probe.compare_outputs(got, refs)[0]
        self.assertLess(row["relative_l2"], probe.TOL)
        self.assertGreater(row["worst_token_relative_l2"], probe.TOL)
        self.assertFalse(row["passed"])
        zeros = tuple(torch.zeros(32, 4) for _ in range(4))
        changed = tuple(v.clone() for v in zeros)
        changed[0][7, 1] = 1e-12
        self.assertFalse(probe.compare_outputs(changed, zeros)[0]["passed"])
        self.assertTrue(all(r["passed"] for r in probe.compare_outputs(zeros, zeros)))

    def test_v41_zero_and_near_zero_remain_finite_at_real_epsilon(self):
        for factor in (0., 1e-8):
            values = probe.fixture(1, 5120, seed=18)
            values[0].mul_(factor)
            values[1].mul_(factor)
            result = probe.v41_component_reference(*values, torch.ones(1, 4))
            self.assertTrue(all(torch.isfinite(v).all() for v in result))
            if factor == 0:
                self.assertEqual(torch.count_nonzero(result[3]).item(), 0)

    def test_compile_only_exports_isolation_and_loader_restoration(self):
        import torch.utils.cpp_extension as ce
        ext = SimpleNamespace(run_mhc=lambda: None, run_mhc_v41=lambda: None)
        driver = SimpleNamespace(_EXT=None, _build=lambda: ce.load(
            sources=["pinned.cu"], extra_cuda_cflags=["arch=compute_121a,code=sm_121a"]))
        env = {"MK_PROBE_NO_GPU": "1", "NVIDIA_VISIBLE_DEVICES": "void", "CUDA_VISIBLE_DEVICES": ""}
        with patch.dict(os.environ, env), patch.object(probe.glob, "glob", return_value=[]), \
                patch.object(torch.cuda, "is_initialized", return_value=False), \
                patch.object(ce, "load", return_value=ext) as loader:
            result = probe.compile_extension(driver)
            self.assertIs(ce.load, loader)
            self.assertEqual(result["exports"], ["run_mhc", "run_mhc_v41"])
            self.assertIn("-Xptxas=--warn-on-spills", result["load"]["cuda_flags"])
            del ext.run_mhc_v41
            with self.assertRaises(RuntimeError):
                probe.compile_extension(driver)
            self.assertIs(ce.load, loader)
        with patch.dict(os.environ, env), patch.object(probe.glob, "glob", return_value=["/dev/nvidia0"]):
            with self.assertRaises(RuntimeError):
                probe.compile_extension(driver)


if __name__ == "__main__":
    unittest.main()
