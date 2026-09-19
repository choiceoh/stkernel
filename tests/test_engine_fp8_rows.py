"""engine/kernels/dense/fp8_rows: a decode step's FP8 head rows in one launch, and FP8Linear handing them to it only
where a profile opted in (`decode_rows`). The arithmetic is held on a GPU (the recipe's exact FP32 value, and
deep_gemm's output); the routing on the CPU with both kernels stood in for."""
import importlib.util
import os
import sys
import types
import unittest
from unittest import mock

torch = None
if importlib.util.find_spec("torch"):
    import torch


@unittest.skipUnless(torch is not None, "torch required")
class RoutingTests(unittest.TestCase):
    def lane(self, decode_rows):
        from engine.kernels.dense import FP8Linear
        q = torch.zeros(256, 256).to(torch.float8_e4m3fn)
        scale = torch.ones(2, 2)
        return FP8Linear(torch.zeros(200, 256, dtype=torch.bfloat16), quantized=(q, scale), decode_rows=decode_rows)

    def run_rows(self, lane, rows):
        from engine.kernels.dense import fp8_rows
        calls = []
        fake = types.ModuleType("deep_gemm")
        fake.fp8_gemm_nt = lambda a, w, out: calls.append("deep_gemm")
        init = types.ModuleType("engine.kernels.deep_gemm")
        init._initialize = lambda: None
        q = torch.zeros(rows, 256).to(torch.float8_e4m3fn)
        s = torch.ones(rows, 2)
        with mock.patch.dict(sys.modules, {"deep_gemm": fake, "engine.kernels.deep_gemm": init}), \
                mock.patch.object(fp8_rows, "project", side_effect=lambda *a, **k: calls.append("fp8_rows")):
            out = lane.project_quantized(q, s)
        self.assertEqual(tuple(out.shape), (rows, 200), "the output is cut to the weight's rows")
        return calls

    def test_opted_in_decode_rows_take_the_one_launch_kernel(self):
        from engine.kernels.dense import fp8_rows
        lane = self.lane(True)
        for rows in (1, 4, fp8_rows.MAX_ROWS):
            self.assertEqual(self.run_rows(lane, rows), ["fp8_rows"], rows)
        self.assertEqual(self.run_rows(lane, fp8_rows.MAX_ROWS + 1), ["deep_gemm"])

    def test_a_lane_that_did_not_opt_in_stays_on_deep_gemm(self):
        self.assertEqual(self.run_rows(self.lane(False), 4), ["deep_gemm"])

    def test_only_qwen38s_head_opts_in(self):
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        users = [p for p in (root / "engine").rglob("*.py") if "decode_rows=True" in p.read_text(encoding="utf-8")]
        self.assertEqual([p.relative_to(root).as_posix() for p in users], ["engine/profiles/qwen38/net.py"])

    def test_one_tile_for_every_decode_row_count(self):
        from engine.kernels.dense.fp8_rows import MAX_ROWS, tile
        self.assertEqual({tile(r) for r in range(1, MAX_ROWS + 1)}, {(32, 4, 4)})


@unittest.skipUnless(torch is not None and importlib.util.find_spec("triton") is not None
                     and (os.environ.get("TRITON_INTERPRET") == "1" or torch.cuda.is_available()),
                     "CUDA or Triton interpreter required")
class DraftHeadTests(unittest.TestCase):
    def weight(self, n=256, k=512, device="cpu"):
        gen = torch.Generator().manual_seed(1)
        w = torch.randn(n, k, generator=gen) * 0.05
        blocks = w.view(n // 128, 128, k // 128, 128).abs().amax((1, 3)).clamp_min(1e-4)
        ws = torch.exp2(torch.ceil(torch.log2(blocks / 448.0)))
        wq = (w / ws.repeat_interleave(128, 0).repeat_interleave(128, 1)).to(torch.float8_e4m3fn)
        return wq.to(device), ws.to(device)

    def test_the_rows_stay_bf16_against_the_fp8_weight(self):
        from engine.kernels.dense import fp8_rows
        device = "cpu" if os.environ.get("TRITON_INTERPRET") == "1" else "cuda"
        wq, ws = self.weight(device=device)
        exact = wq.float() * ws.repeat_interleave(128, 0).repeat_interleave(128, 1)
        for m in (1, 5, 16):
            x = torch.randn(m, 512, generator=torch.Generator().manual_seed(m)).bfloat16().to(device)
            got = fp8_rows.project_bf16(x, (wq, ws)).float()
            ref = x.float() @ exact.t()
            self.assertLessEqual(float((got - ref).abs().max() / ref.abs().max()), 2.0 ** -7, m)
            self.assertTrue(torch.equal(got.argmax(1), ref.argmax(1)), m)

    def test_the_drafter_reads_the_head_through_it(self):
        from unittest import mock
        from engine.kernels.dense import FP8Linear, fp8_rows
        from engine.profiles.qwen38.net import Qwen38Net
        head = object.__new__(FP8Linear)
        head.cublas, head.weight = None, self.weight()
        net = object.__new__(Qwen38Net)
        net.dense, net.vp = {"head": head}, 200
        h = mock.Mock(is_cuda=True, shape=(3, 512), dtype=torch.bfloat16)
        h.contiguous.return_value = h
        with mock.patch.object(fp8_rows, "project_bf16", return_value=torch.zeros(3, 256)) as rows:
            self.assertEqual(tuple(net.draft_logits(h).shape), (3, 200))
            rows.assert_called_once()
        with mock.patch.object(Qwen38Net, "head_local", return_value=torch.zeros(20, 256)) as verify:
            h.shape = (20, 512)                                   # past 16 rows: the verify step's head
            self.assertEqual(tuple(net.draft_logits(h).shape), (20, 200))
            verify.assert_called_once()

    def test_a_pick_is_the_same_with_its_probability(self):
        from unittest import mock
        from engine.profiles.qwen38.net import Qwen38Net
        net = object.__new__(Qwen38Net)
        net.draft_index, net.draft_tap, net.rank, net.vp = None, None, 0, 10
        net.comm = mock.Mock(all_reduce_max=lambda t: t, all_gather=lambda t, dim: t)
        logits = torch.zeros(2, 10)
        logits[0, 7], logits[1, 2] = 3.0, 3.0
        with mock.patch.object(Qwen38Net, "draft_logits", return_value=logits) as draft:
            picks = net.draft_tokens(torch.zeros(2, 4))
            same, probs = net.draft_tokens(torch.zeros(2, 4), probability=True)
        self.assertEqual(picks.tolist(), [7, 2])
        self.assertEqual(same.tolist(), [7, 2])
        self.assertEqual(draft.call_count, 2)                     # both paths read the drafter's head, not the verify step's
        self.assertTrue(torch.all(probs > 0.5))


@unittest.skipUnless(torch is not None and torch.cuda.is_available() and importlib.util.find_spec("deep_gemm")
                     is not None, "a GPU with deep_gemm")
class ArithmeticTests(unittest.TestCase):
    def test_the_recipe_and_deep_gemm(self):
        from engine.kernels.dense import FP8Linear, fp8_rows
        self.assertTrue(all(e <= 2.0 ** -7 for e in fp8_rows.qualify("cuda").values()))
        gen = torch.Generator().manual_seed(3)
        w = (torch.randn(2048, 2560, generator=gen) * 0.02).bfloat16().cuda()
        ours, theirs = FP8Linear(w, decode_rows=True), FP8Linear(w)
        for rows in (1, 3, 16):
            x = torch.randn(rows, 2560, generator=gen).bfloat16().cuda()
            a, b = ours(x).float(), theirs(x).float()
            self.assertLessEqual(float((a - b).abs().max() / b.abs().max()), 2.0 ** -7, rows)
            self.assertTrue(torch.equal(a.argmax(-1), b.argmax(-1)), rows)


if __name__ == "__main__":
    unittest.main()
