"""The strided Q/K norm serves Qwen3.8's 4 key heads a rank as it serves GLM-5.3's 16 (engine/QWEN38_CARRY.md K4).

kda._glm53_qk_l2norm_strided normalises a prefill chunk's q and k straight from their strided views in one launch; any
other shape took two contiguous copies and two l2norm_fwd launches (both GDN prefill lanes, chunk_decay and kda, call
it). It admitted (16, 128) only. The kernel reduces each row over the head dim in 32-row programs; the head count only
maps a row to its (token, head) load, so 4 x 128 is the same arithmetic on the same programs -- three launches fewer on
each GDN layer of a prefill chunk, and 16 x 128 compiles the kernel it did.

Under TRITON_INTERPRET=1 the strided kernel is held to l2norm_fwd_kernel2 over contiguous copies in FP32 at both head
counts and both served layouts; on a GPU the served entry is held to l2norm_fwd in BF16, bit for bit, as
probes/qk_norm_strided_check.py holds 16 heads (probes/engine_qwen38_cells runs the GPU case).

    docker exec -e TRITON_INTERPRET=1 -w <repo> stk-test python3 -m unittest tests.test_engine_qk_norm_strided_heads
"""
import importlib.util
import json
import os
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
torch = None
if importlib.util.find_spec("torch") is not None:
    import torch
TRITON = importlib.util.find_spec("triton") is not None
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
CUDA = torch is not None and torch.cuda.is_available() and not INTERPRET


def views(layout, tokens, heads, dim=128, device="cpu", dtype=None):
    """q and k [1, T, H, D] as a GDN layer hands them over: slices of the conv output row y [T, q | k | v], which is
    token-major (a packed row) or the conv kernel's channel-major layout transposed."""
    dtype = dtype or torch.float32
    width = 3 * heads * dim
    if layout == "channel":
        y = torch.randn(width, tokens, device=device).to(dtype).t()
    else:
        y = torch.randn(tokens, width, device=device).to(dtype)
    q, k, _ = y.split([heads * dim, heads * dim, heads * dim], dim=-1)
    return q.reshape(1, tokens, heads, dim), k.reshape(1, tokens, heads, dim)


@unittest.skipUnless(torch is not None and TRITON and INTERPRET, "runs the kernels under TRITON_INTERPRET=1")
class StridedKernelTests(unittest.TestCase):
    def test_the_strided_kernel_is_kernel2_over_contiguous_copies_at_both_head_counts(self):
        import triton
        from engine.kernels.kda import kda, l2norm
        torch.manual_seed(20260917)
        for heads in (4, 16):
            for layout in ("token", "channel"):
                for tokens in (1, 9, 33):
                    with self.subTest(heads=heads, layout=layout, tokens=tokens):
                        q, k = views(layout, tokens, heads)
                        if tokens > 1:
                            self.assertFalse(q.is_contiguous() or k.is_contiguous())
                        rows = tokens * heads
                        q_out, k_out = torch.empty(q.shape), torch.empty(k.shape)
                        kda._glm53_qk_l2norm_strided_kernel[(triton.cdiv(rows, 32), 2)](
                            q, k, q_out, k_out, 1e-6, rows, QT=q.stride(1), QH=q.stride(2), QD=q.stride(3),
                            KT=k.stride(1), KH=k.stride(2), KD=k.stride(3), H=heads, N=128, BD=128, MBLOCK=32)
                        for got, x in ((q_out, q), (k_out, k)):
                            flat = x.contiguous().view(-1, 128)
                            want = torch.empty_like(flat)
                            l2norm.l2norm_fwd_kernel2[(triton.cdiv(rows, 32),)](flat, want, 1e-6, rows, 128, 128, 32)
                            self.assertTrue(torch.equal(got.view(-1, 128), want))


@unittest.skipUnless(torch is not None and TRITON, "the KDA modules import Triton")
class CellTests(unittest.TestCase):
    def test_the_cells_are_glm53_s_and_qwen38_s_ranks(self):
        from engine.base.kernel_shape import MEASURED
        from engine.kernels.kda import kda
        from engine.profiles.qwen38 import shapes
        config = json.loads((ROOT / "probes" / "qwen38_config.json").read_text())["text_config"]
        qwen = shapes.kernel_shape(config).linear
        self.assertIn((qwen.heads, qwen.k_dim), kda._QK_L2NORM_STRIDED_CELLS)
        self.assertEqual((qwen.heads, qwen.k_dim), (4, 128))
        self.assertIn((MEASURED.linear.heads, MEASURED.linear.k_dim), kda._QK_L2NORM_STRIDED_CELLS)
        self.assertNotIn((8, 128), kda._QK_L2NORM_STRIDED_CELLS)          # qk_norm_strided_check's declined head count

    def test_the_launch_takes_the_input_s_head_count(self):
        source = (ROOT / "engine/kernels/kda/kda.py").read_text()
        self.assertIn("rows = q.shape[1] * heads", source)
        self.assertIn("H=heads, N=128, BD=128, MBLOCK=32", source)
        self.assertIn("tuple(q.shape[2:]) not in _QK_L2NORM_STRIDED_CELLS", source)


@unittest.skipUnless(CUDA and TRITON, "the GPU's BF16 rounding")
class OnTheGpuTests(unittest.TestCase):
    """The served entry against the fallback it replaces, bit for bit, at Qwen3.8's 4 key heads a rank."""

    def test_four_heads_normalise_as_the_contiguous_fallback(self):
        from engine.kernels.kda import kda
        torch.manual_seed(20260917)
        for layout in ("token", "channel"):
            for tokens in (1, 31, 33, 1024, 8192):
                with self.subTest(layout=layout, tokens=tokens):
                    q, k = views(layout, tokens, 4, device="cuda", dtype=torch.bfloat16)
                    inputs = q.clone(), k.clone()
                    got = kda._glm53_qk_l2norm_strided(q, k)
                    self.assertIsNotNone(got)
                    for out, x in zip(got, inputs):
                        want = kda.l2norm_fwd(x.contiguous())
                        self.assertTrue(out.is_contiguous())
                        self.assertTrue(torch.equal(out.view(torch.uint8), want.view(torch.uint8)))
                    self.assertTrue(torch.equal(q, inputs[0]) and torch.equal(k, inputs[1]))
        q, k = views("token", 8, 8, device="cuda", dtype=torch.bfloat16)
        self.assertIsNone(kda._glm53_qk_l2norm_strided(q, k))                # another head count: the fallback


if __name__ == "__main__":
    unittest.main()
