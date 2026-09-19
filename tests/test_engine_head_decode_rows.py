"""The W8A16 decode-row lane of engine/kernels/dense.FP8Linear (`decode_rows="w8a16"`), as GLM-5.3's head declares it:
1..16 BF16 rows of a decode step read the FP8 weight in one launch (fp8_rows.project_bf16) ahead of a prepared
cuBLASLt reader, which keeps larger batches; the boot holds the lane to its exact product and to the reader
(`qualify_decode_rows`), and the execution report requires that the declared lane served.

GB10 glm53-head-0919a (measurements/glm53_decode_rows_20260919): 731-743 us a call against the cuBLASLt reader's
850-929 at GLM's head shape, with a third less error against the BF16 product."""
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

torch = None
if importlib.util.find_spec("torch"):
    import torch

KERNEL = (torch is not None and importlib.util.find_spec("triton") is not None
          and (os.environ.get("TRITON_INTERPRET") == "1" or torch.cuda.is_available()))


class FakeReader:
    """The cuBLASLt reader's surface: a call and the producer form, each recording what executed."""

    split_decode = False

    def __init__(self, n, value=None):
        self.n, self.value, self.executed, self.calls = n, value, set(), []

    def __call__(self, x, *, out=None, decode=False, **_):
        self.executed.add("direct")
        self.calls.append(("direct", x.shape[0]))
        return self.value(x) if self.value else torch.zeros(x.shape[0], self.n, dtype=torch.bfloat16)

    def project_mx(self, q, scale, *, out=None):
        self.executed.update(("direct", "producer_mx"))
        self.calls.append(("producer_mx", q.shape[0]))
        return torch.zeros(q.shape[0], self.n, dtype=torch.bfloat16)

    def report(self):
        return {"executed": sorted(self.executed)}


@unittest.skipUnless(torch is not None, "torch required")
class RoutingTests(unittest.TestCase):
    """Which rows take the lane: the kernels stood in for, so this runs on any CPU."""

    def head(self, decode_rows="w8a16", reader=True):
        from engine.kernels.dense import FP8Linear
        layer = FP8Linear(torch.zeros(200, 256, dtype=torch.bfloat16),
                          quantized=(torch.zeros(256, 256).to(torch.float8_e4m3fn), torch.ones(2, 2)),
                          decode_rows=decode_rows, name="head")
        if reader:
            layer.cublas = FakeReader(256)
        return layer

    def rows_lane(self):
        from engine.kernels.dense import fp8_rows
        return mock.patch.object(fp8_rows, "project_bf16",
                                 side_effect=lambda x, w, out=None: torch.ones(x.shape[0], w[0].shape[0], dtype=torch.bfloat16))

    def test_decode_rows_take_the_lane_ahead_of_the_reader(self):
        head = self.head()
        seen = []
        head.observer = lambda x, ok: seen.append(x.shape[0])
        with self.rows_lane() as lane:
            for m in (1, 7, 16):
                out = head(torch.zeros(m, 256, dtype=torch.bfloat16))
                self.assertEqual(tuple(out.shape), (m, 200), "cut to the weight's rows")
        self.assertEqual(lane.call_count, 3)
        self.assertEqual(head.cublas.calls, [], "the reader served none of them")
        self.assertTrue(head.decode_rows_executed)
        self.assertEqual(seen, [1, 7, 16], "calibration still sees every row the head reads")

    def test_a_larger_batch_stays_on_the_reader(self):
        head = self.head()
        with self.rows_lane() as lane:
            head(torch.zeros(17, 256, dtype=torch.bfloat16))
            head(torch.zeros(4, 256, dtype=torch.bfloat16), normalization=None)
        self.assertEqual(lane.call_count, 1)
        self.assertEqual(head.cublas.calls, [("direct", 17)])

    def test_the_producer_form_reads_its_bf16_rows(self):
        head = self.head()
        hidden = torch.zeros(14, 256, dtype=torch.bfloat16)
        q = torch.zeros(14, 256).to(torch.float8_e4m3fn)
        with self.rows_lane() as lane:
            out = head.project_mx(hidden, q, torch.zeros(1, dtype=torch.uint8))
            self.assertIs(lane.call_args.args[0], hidden, "the producer's BF16 rows, not its MX rows")
        self.assertEqual(tuple(out.shape), (14, 200))
        self.assertEqual(head.cublas.calls, [])
        big = torch.zeros(21, 256, dtype=torch.bfloat16)
        head.project_mx(big, torch.zeros(21, 256).to(torch.float8_e4m3fn), torch.zeros(1, dtype=torch.uint8))
        self.assertEqual(head.cublas.calls, [("producer_mx", 21)], "past 16 rows the reader takes the MX rows")

    def test_a_layer_that_did_not_declare_the_lane_never_takes_it(self):
        for mode in (False, True):
            head = self.head(decode_rows=mode)
            with self.rows_lane() as lane:
                head(torch.zeros(4, 256, dtype=torch.bfloat16))
            self.assertEqual(lane.call_count, 0, mode)
            self.assertFalse(head.decode_rows_executed)
        with self.assertRaisesRegex(ValueError, "decode_rows is one of"):
            self.head(decode_rows="bf16")

    def test_rows_the_kernel_cannot_read_are_not_offered_to_it(self):
        head = self.head()
        self.assertTrue(head.bf16_rows(torch.zeros(16, 256, dtype=torch.bfloat16)))
        self.assertFalse(head.bf16_rows(torch.zeros(4, 256, dtype=torch.float32)))
        self.assertFalse(head.bf16_rows(torch.zeros(4, 512, dtype=torch.bfloat16)[:, :256]), "not contiguous")
        self.assertFalse(head.bf16_rows(torch.zeros(0, 256, dtype=torch.bfloat16)))

    def test_prepared_scales_must_be_powers_of_two(self):
        """engine/SM121_INTAKE.md U20: a checkpoint's own FP32 block scales are refused at bind -- the readers take
        UE8M0 exponents, and DeepGEMM on sm_121 faults or asserts on anything else (vllm#54125, sglang#39482)."""
        from engine.kernels.dense import FP8Linear
        q = torch.zeros(256, 256).to(torch.float8_e4m3fn)
        for scale in (torch.full((2, 2), 2.0 ** -9), torch.tensor([[1.0, 0.5], [4.0, 2.0 ** -20]])):
            FP8Linear(torch.zeros(200, 256, dtype=torch.bfloat16), quantized=(q, scale), name="head")
        with self.assertRaisesRegex(ValueError, "head: prepared FP8 block scales must be powers of two"):
            FP8Linear(torch.zeros(200, 256, dtype=torch.bfloat16), quantized=(q, torch.full((2, 2), 0.3)), name="head")

    def test_glms_head_declares_the_lane_and_nothing_else_does(self):
        users = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "engine/profiles").rglob("*.py")
                       if 'decode_rows="w8a16"' in p.read_text(encoding="utf-8"))
        self.assertEqual(users, ["engine/profiles/glm53/net.py"])


@unittest.skipUnless(torch is not None, "torch required")
class BootTests(unittest.TestCase):
    """What GLM's boot does with the lane: qualify it on the head's weight, and require it to have served."""

    def test_the_execution_report_requires_the_declared_lane_to_have_served(self):
        from engine.profiles.glm53 import cublas
        head = RoutingTests.head(RoutingTests())
        head.cublas.executed.update(("direct", "producer_mx"))
        net = types.SimpleNamespace(cublas_readers={"head": head}, cublas_head_producer_required=True)
        with self.assertRaisesRegex(RuntimeError, "declared W8A16 decode-row lane was not executed: head"):
            cublas.execution_report(net)
        head.decode_rows_executed = True
        self.assertEqual(cublas.execution_report(net)["head"]["decode_rows"], "w8a16")

    def test_prepare_qualifies_the_head_before_anything_is_captured(self):
        source = (ROOT / "engine/profiles/glm53/cublas.py").read_text(encoding="utf-8")
        self.assertIn("net.dense['head'].qualify_decode_rows(producer=drafter is not None)", source)

    def test_qualification_does_not_count_as_serving(self):
        from engine.kernels.dense import fp8_rows, mxfp8
        head = RoutingTests.head(RoutingTests())
        exact = lambda x, w, out=None: x.float().matmul(torch.zeros(256, 256).t()).to(torch.bfloat16)
        with mock.patch.object(fp8_rows, "project_bf16", side_effect=exact), \
                mock.patch.object(mxfp8, "quantize", side_effect=lambda x, num_warps=1: (x, None)):
            report = head.qualify_decode_rows(rows=(1, 16), columns=256, producer=True)
        self.assertEqual(sorted(report), [1, 16])
        self.assertEqual(sorted(report[16]), ["exact", "mx", "reader"])
        self.assertEqual(head.cublas.executed, {"direct", "producer_mx"}, "the qualification ran both reader paths")
        self.assertFalse(head.decode_rows_executed, "and it is not the lane serving")

    def test_a_lane_far_from_the_reader_stops_the_boot(self):
        from engine.kernels.dense import fp8_rows
        head = RoutingTests.head(RoutingTests())
        head.cublas = FakeReader(256, value=lambda x: torch.full((x.shape[0], 256), 5.0, dtype=torch.bfloat16))
        zeros = lambda x, w, out=None: torch.zeros(x.shape[0], 256, dtype=torch.bfloat16)
        with mock.patch.object(fp8_rows, "project_bf16", side_effect=zeros), \
                self.assertRaisesRegex(RuntimeError, "from the cuBLASLt reader -- beyond rounding"):
            head.qualify_decode_rows(rows=(4,), columns=256)


@unittest.skipUnless(KERNEL, "CUDA or the Triton interpreter")
class KernelTests(unittest.TestCase):
    """The lane itself, through FP8Linear, against the exact product of its dequantized weight."""

    def test_the_lane_is_its_exact_product(self):
        from engine.kernels.dense import FP8Linear
        device = "cpu" if os.environ.get("TRITON_INTERPRET") == "1" else "cuda"
        gen = torch.Generator().manual_seed(3)
        n, k = 256, 512
        w = torch.randn(n, k, generator=gen) * 0.05
        blocks = w.view(n // 128, 128, k // 128, 128).abs().amax((1, 3)).clamp_min(1e-4)
        ws = torch.exp2(torch.ceil(torch.log2(blocks / 448.0)))
        wq = (w / ws.repeat_interleave(128, 0).repeat_interleave(128, 1)).to(torch.float8_e4m3fn)
        head = FP8Linear(w.bfloat16().to(device), quantized=(wq.to(device), ws.to(device)), decode_rows="w8a16", name="head")
        report = head.qualify_decode_rows(rows=(1, 7, 16), columns=256)
        self.assertTrue(all(r["exact"] <= 2.0 ** -7 for r in report.values()), report)
        self.assertFalse(head.decode_rows_executed)
        x = torch.randn(7, k, generator=gen).bfloat16().to(device)
        exact = x.float() @ (wq.float() * ws.repeat_interleave(128, 0).repeat_interleave(128, 1)).to(device).t()
        got = head(x).float()
        self.assertLessEqual(float((got - exact).abs().max() / exact.abs().max()), 2.0 ** -7)
        self.assertTrue(head.decode_rows_executed)


if __name__ == "__main__":
    unittest.main()
