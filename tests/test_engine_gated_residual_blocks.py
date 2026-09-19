"""engine/kernels/gated_residual.mix_block -- a prefill step's mixer in two launches over row blocks -- held byte for byte
to the unfolded mixer (`_gates` and `_mix_mean`) on the same products (a plain block GEMM over `_tile_dot` at the
same tiles), within the oracle's band of the torch form, and where `mix` takes it.

    TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES= python3 -m unittest tests.test_engine_gated_residual_blocks
"""
import importlib.util
import os
import unittest

torch = None
if importlib.util.find_spec("torch"):
    import torch
TRITON = importlib.util.find_spec("triton") is not None
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
READY = torch is not None and TRITON
DEVICE = "cpu" if INTERPRET else "cuda"
HC, HIDDEN, RANK = 4, 2560, 320

if READY:
    import triton
    import triton.language as tl
    from engine.kernels import gated_residual as hcr

    @triton.jit
    def _block_product(X, W, OUT, M, N, K, sX, sW, sO, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                       BLOCK_K: tl.constexpr, FP32_DOT: tl.constexpr):
        # the plain GEMM on mix_block's own dot: the product rounded to BF16 and stored, as a GEMM's output is
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = hcr._tile_dot(X, W, sX, sW, rows, cols, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, FP32_DOT)
        tl.store(OUT + rows[:, None] * sO + cols[None, :], acc.to(OUT.dtype.element_ty),
                 mask=(rows[:, None] < M) & (cols[None, :] < N))


def site(inject: bool, rows: int, device="cpu", seed=0):
    gen = torch.Generator().manual_seed(seed)
    width = HC * HIDDEN
    down = (torch.randn(RANK + (HC if inject else 0), width, generator=gen) * 0.02).bfloat16().to(device)
    up = (torch.randn(width, RANK, generator=gen) * 0.02).bfloat16().to(device)
    normed = torch.randn(rows, width, generator=gen).bfloat16().to(device)
    return normed, down, up


def product(x, w, tile):
    bm, bn, bk, warps, stages = tile
    out = torch.empty(x.shape[0], w.shape[0], dtype=x.dtype, device=x.device)
    _block_product[(triton.cdiv(x.shape[0], bm), triton.cdiv(w.shape[0], bn))](
        x, w, out, x.shape[0], w.shape[0], x.shape[1], x.stride(0), w.stride(0), out.stride(0), BLOCK_M=bm, BLOCK_N=bn,
        BLOCK_K=bk, FP32_DOT=not x.is_cuda, num_warps=warps, num_stages=stages)
    return out


def unfolded(normed, down, up, inject):
    """The five-launch site after the stream norm on mix_block's products: product, `_gates`, product, `_mix_mean`."""
    rows = normed.shape[0]
    di = product(normed, down, hcr.BLOCK_TILES["down"])
    gates = torch.empty(rows, RANK, dtype=normed.dtype, device=normed.device)
    inj = torch.empty(rows, HC, dtype=normed.dtype, device=normed.device) if inject else gates
    hcr._gates[(rows,)](di, gates, inj, di.stride(0), gates.stride(0), inj.stride(0), float(HC), R=RANK,
                        BR=triton.next_power_of_2(RANK), HC=HC, BH=triton.next_power_of_2(HC), WITH_INJECT=inject,
                        num_warps=4)
    weights = product(gates, up, hcr.BLOCK_TILES["up"])
    mixed = torch.empty(rows, HIDDEN, dtype=normed.dtype, device=normed.device)
    hcr._mix_mean[(rows, 1)](weights, normed, mixed, weights.stride(0), normed.stride(0), mixed.stride(0), float(HC),
                             HID=HIDDEN, BD=triton.next_power_of_2(HIDDEN), HC=HC, num_warps=8)
    return mixed, (inj if inject else None)


def torch_form(normed, down, up, inject):
    """engine/modules' gated residual mixer in fp32 over the BF16 inputs: the oracle band's reference."""
    n, d, u = normed.float(), down.float(), up.float()
    di = n @ d.t()
    gates = torch.nn.functional.silu(di[:, :RANK] / HC)
    weights = torch.sigmoid(gates @ u.t()).unflatten(-1, (HC, HIDDEN))
    mixed = (weights * n.unflatten(-1, (HC, HIDDEN))).mean(dim=-2)
    return mixed, (2 * torch.sigmoid(di[:, RANK:] / HC) if inject else None)


@unittest.skipUnless(READY, "torch and triton required")
class BlocksFoldTests(unittest.TestCase):
    def test_only_prefill_rows_fold(self):
        self.assertFalse(hcr.blocks_fold(*site(True, hcr.PREFILL_ROWS)))
        self.assertTrue(hcr.blocks_fold(*site(True, hcr.PREFILL_ROWS + 1)))
        normed, down, up = site(True, hcr.PREFILL_ROWS + 1)
        self.assertFalse(hcr.blocks_fold(normed.float(), down.float(), up.float()))
        self.assertFalse(hcr.blocks_fold(normed, down, up.t().contiguous().t()), "an up weight not packed along its rank")

    def test_mix_takes_the_blocks_after_the_rows_and_before_the_unfolded_site(self):
        import inspect
        body = inspect.getsource(hcr.mix)
        taken = body.index("return mix_block(")
        self.assertLess(body.index("return mix_rows("), taken)
        self.assertIn("if project_down is None and project_up is None and blocks_fold(normed, down_inject, up):",
                      body[:taken])
        self.assertGreater(taken, body.index("if not normed.is_cuda:"), "the CPU form stays torch's")
        self.assertLess(taken, body.index("_gates[(rows,)]"), "the five-launch site is for what neither fold takes")


@unittest.skipUnless(READY and (INTERPRET or (torch is not None and torch.cuda.is_available())),
                     "CUDA or Triton interpreter required")
class MixBlockTests(unittest.TestCase):
    def test_the_fold_is_the_unfolded_mixer_byte_for_byte(self):
        for inject in (True, False):
            for rows in (65, 130):                        # past PREFILL_ROWS; the second ends in a partial block
                normed, down, up = site(inject, rows, DEVICE, seed=rows + 7 * inject)
                mixed, injection = hcr.mix_block(normed, down, up, HC, inject=inject)
                want_mixed, want_injection = unfolded(normed, down, up, inject)
                with self.subTest(inject=inject, rows=rows):
                    self.assertEqual((tuple(mixed.shape), mixed.dtype), ((rows, HIDDEN), torch.bfloat16))
                    self.assertTrue(torch.equal(mixed, want_mixed))
                    if inject:
                        self.assertTrue(torch.equal(injection, want_injection))
                    else:
                        self.assertIsNone(injection)

    def test_it_is_the_torch_form_within_the_oracle_band(self):
        normed, down, up = site(True, 97, DEVICE, seed=3)
        mixed, injection = hcr.mix_block(normed, down, up, HC)
        want_mixed, want_injection = torch_form(normed.cpu(), down.cpu(), up.cpu(), True)
        for got, want in ((mixed, want_mixed), (injection, want_injection)):
            err = float((got.float().cpu() - want).abs().max() / want.abs().max())
            self.assertLess(err, 2.0 ** -6)             # a few BF16 steps: rounding order, not a wrong formula

    def test_mix_takes_it_on_cuda(self):
        if DEVICE != "cuda":
            self.skipTest("mix folds CUDA rows only")
        normed, down, up = site(True, 200, "cuda")
        got = hcr.mix(normed, down, up, HC)
        want = hcr.mix_block(normed, down, up, HC)
        self.assertTrue(torch.equal(got[0], want[0]) and torch.equal(got[1], want[1]))


if __name__ == "__main__":
    unittest.main()
