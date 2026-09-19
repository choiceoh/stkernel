"""engine/kernels/gated_residual.mix_block -- a prefill step's mixer in two launches over row blocks -- held byte for byte
to the unfolded mixer (`_gates` and `_mix_mean`) on the same products (a plain block GEMM over `_tile_dot` at the
same tiles, its narrow last column block and its split included), within the oracle's band of the torch form, and where
`mix` takes it.

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
    def _block_partial(X, W, OUT, M, N, k0, k1, col0, sX, sW, sO, BLOCK_M: tl.constexpr, WIDTH: tl.constexpr,
                       BLOCK_K: tl.constexpr, FP32_DOT: tl.constexpr):
        # one column block of mix_block's own dot over [k0, k1), every row block, stored in OUT's dtype (FP32, or
        # rounded here as the fold rounds -- the interpreter's rounding is not torch's)
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = col0 + tl.arange(0, WIDTH)
        acc = hcr._tile_dot(X, W, sX, sW, rows, cols, M, N, k0, k1, BLOCK_M, WIDTH, BLOCK_K, FP32_DOT)
        tl.store(OUT + rows[:, None] * sO + cols[None, :], acc.to(OUT.dtype.element_ty),
                 mask=(rows[:, None] < M) & (cols[None, :] < N))

    @triton.jit
    def _round(X, OUT, n, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(OUT + i, tl.load(X + i, mask=i < n).to(OUT.dtype.element_ty), mask=i < n)


def site(inject: bool, rows: int, device="cpu", seed=0):
    gen = torch.Generator().manual_seed(seed)
    width = HC * HIDDEN
    down = (torch.randn(RANK + (HC if inject else 0), width, generator=gen) * 0.02).bfloat16().to(device)
    up = (torch.randn(width, RANK, generator=gen) * 0.02).bfloat16().to(device)
    normed = torch.randn(rows, width, generator=gen).bfloat16().to(device)
    return normed, down, up


def product(x, w, tile):
    """The plain GEMM on the fold's dots at `tile` (BLOCK_M, BLOCK_N, BLOCK_K, warps, stages[, split]): each column
    block at the width the fold gives it (the narrow last one too), each split's FP32 partial summed in split order from
    zero, as the fold's last program sums them, and the product rounded to BF16 once, as a GEMM's output is."""
    bm, bn, bk, warps, stages, *split = tile
    split = split[0] if split else 1
    m, k = x.shape
    n = w.shape[0]
    blocks, tail = triton.cdiv(n, bn), hcr.narrow_tail(n, bn)
    span = triton.cdiv(triton.cdiv(k, split), bk) * bk
    total = torch.zeros(m, n, dtype=torch.float32, device=x.device)
    for s in range(split):
        part = torch.empty(m, n, dtype=x.dtype if split == 1 else torch.float32, device=x.device)
        for b in range(blocks):
            _block_partial[(triton.cdiv(m, bm),)](
                x, w, part, m, n, s * span, min(s * span + span, k), b * bn, x.stride(0), w.stride(0), part.stride(0),
                BLOCK_M=bm, WIDTH=tail if tail and b == blocks - 1 else bn, BLOCK_K=bk, FP32_DOT=not x.is_cuda,
                num_warps=warps, num_stages=stages)
        if split == 1:
            return part
        total += part
    out = torch.empty(m, n, dtype=x.dtype, device=x.device)
    _round[(triton.cdiv(m * n, 1024),)](total, out, m * n, BLOCK=1024)
    return out


def unfolded(normed, down, up, inject, *, down_tile=None):
    """The five-launch site after the stream norm on mix_block's products: the down product (cuBLAS's, or the fold's
    own dots at `down_tile`), `_gates`, the up product at the fold's tile, `_mix_mean`."""
    rows = normed.shape[0]
    di = torch.mm(normed, down.t()) if down_tile is None else product(normed, down, down_tile)
    gates = torch.empty(rows, RANK, dtype=normed.dtype, device=normed.device)
    inj = torch.empty(rows, HC, dtype=normed.dtype, device=normed.device) if inject else gates
    hcr._gates[(rows,)](di, gates, inj, di.stride(0), gates.stride(0), inj.stride(0), float(HC), R=RANK,
                        BR=triton.next_power_of_2(RANK), HC=HC, BH=triton.next_power_of_2(HC), WITH_INJECT=inject,
                        num_warps=4)
    weights = product(gates, up, hcr.UP_BLOCK_TILE)
    mixed = torch.empty(rows, HIDDEN, dtype=normed.dtype, device=normed.device)
    hcr._mix_mean[(rows, 1)](weights, normed, mixed, weights.stride(0), normed.stride(0), mixed.stride(0), float(HC),
                             HID=HIDDEN, BD=triton.next_power_of_2(HIDDEN), HC=HC, num_warps=8)
    return mixed, (inj if inject else None)


def kernel_normed(h, out, injection, w, eps):
    """The normalised streams from the served kernels (leave_norm's, in place, or norm_streams' when `out` is None),
    launched here: on the CPU the module's entry points take the torch form."""
    normed = torch.empty_like(h)
    bd = triton.next_power_of_2(HIDDEN)
    grid, rows_first = hcr._stream_grid(h.shape[0], HC)                      # the served order: a row's streams adjacent
    if out is None:
        hcr._norm_streams[grid](h, w, normed, h.stride(0), normed.stride(0), eps, HID=HIDDEN, BD=bd, SCALE_ONLY=False,
                                ROWS_FIRST=rows_first, num_warps=hcr._warps(HIDDEN))
    else:
        hcr._leave_norm[grid](h, out, injection, w, normed, h, h.stride(0), out.stride(0), injection.stride(0),
                              normed.stride(0), eps, 0, HID=HIDDEN, BD=bd, NORM=True, PDL=False, PREFETCH=False,
                              SCALE_ONLY=False, ROWS_FIRST=rows_first, num_warps=hcr._warps(HIDDEN))
    return normed


def streams(rows, device, seed, *, leave):
    """h [rows, hc*H], the norm's weight, and -- with `leave` -- a sublayer's output and its injection."""
    gen = torch.Generator().manual_seed(seed)
    h = torch.randn(rows, HC * HIDDEN, generator=gen).bfloat16().to(device)
    w = (torch.randn(HC * HIDDEN, generator=gen) * 0.1).bfloat16().to(device)
    if not leave:
        return h, w, None, None
    return (h, w, torch.randn(rows, HIDDEN, generator=gen).bfloat16().to(device),
            (torch.rand(rows, HC, generator=gen) * 2).bfloat16().to(device))


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
        self.assertFalse(hcr.blocks_fold(*site(True, hcr.PREFILL_ROWS - 1)))
        self.assertTrue(hcr.blocks_fold(*site(True, hcr.PREFILL_ROWS)))
        normed, down, up = site(True, hcr.PREFILL_ROWS)
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
        """Every way the down projection runs, forced at a few rows: cuBLAS (rows short of DOWN_TILES), the table's
        tiles, a narrow last column block (64-wide blocks: the injection's 4 columns in 16) and a split K (3 splits:
        the last one shorter)."""
        tiles = [None] + [tile for _, tile in hcr.DOWN_TILES] + [(64, 64, 32, 4, 2, 1), (64, 128, 64, 4, 2, 3)]
        for inject in (True, False):
            for rows, tile in [(65, None)] + [(130, t) for t in tiles]:        # 130: a partial last row block
                normed, down, up = site(inject, rows, DEVICE, seed=rows + 7 * inject)
                mixed, injection = hcr.mix_block(normed, down, up, HC, inject=inject,
                                                 tiles={"down": tile, "up": hcr.UP_BLOCK_TILE})
                want_mixed, want_injection = unfolded(normed, down, up, inject, down_tile=tile)
                with self.subTest(inject=inject, rows=rows, tile=tile):
                    self.assertEqual((tuple(mixed.shape), mixed.dtype), ((rows, HIDDEN), torch.bfloat16))
                    self.assertTrue(torch.equal(mixed, want_mixed))
                    if inject:
                        self.assertTrue(torch.equal(injection, want_injection))
                    else:
                        self.assertIsNone(injection)

    def test_the_streams_normalised_as_read_are_the_normalised_streams_byte_for_byte(self):
        """mix_block over the streams and their scales (`stream_scales`: after a leave, or of the streams as they are)
        against mix_block over the kernels' normalised streams: the same bytes, the leave's too -- a narrow last
        column block and a split K among the tiles."""
        eps = 1e-6
        for leave, inject, tile in ((True, True, (64, 64, 64, 4, 2, 1)), (False, False, (64, 128, 64, 4, 2, 3))):
            _, down, up = site(inject, 70, DEVICE, seed=11 + inject)
            h, w, out, injection = streams(70, DEVICE, 5 + leave, leave=leave)
            tiles = {"down": tile, "up": hcr.UP_BLOCK_TILE}
            h_ref = h.clone()
            want = hcr.mix_block(kernel_normed(h_ref, out, injection, w, eps), down, up, HC, inject=inject, tiles=tiles)
            scale = hcr.stream_scales(h, out, injection, eps, HC)
            got = hcr.mix_block(h, down, up, HC, inject=inject, tiles=tiles, norm=(scale, w))
            with self.subTest(leave=leave, inject=inject):
                self.assertEqual((tuple(scale.shape), scale.dtype), ((70, HC), torch.float32))
                self.assertTrue(torch.equal(h, h_ref), "the leave")
                self.assertTrue(torch.equal(got[0], want[0]))
                if inject:
                    self.assertTrue(torch.equal(got[1], want[1]))
                else:
                    self.assertIsNone(got[1])

    def test_the_streams_normalise_only_inside_a_down_fold(self):
        _, down, up = site(True, 20, DEVICE)
        h, w, _, _ = streams(20, DEVICE, 1, leave=False)
        scale = hcr.stream_scales(h, None, None, 1e-6, HC)
        with self.assertRaises(ValueError):
            hcr.mix_block(h, down, up, HC, tiles={"down": None, "up": hcr.UP_BLOCK_TILE}, norm=(scale, w))
        with self.assertRaises(ValueError):
            hcr.mix_block(h, down, up, HC, tiles={"down": (64, 64, 64, 4, 2, 1), "up": hcr.UP_BLOCK_TILE},
                          norm=(scale.bfloat16(), w))

    def test_site_is_the_two_calls_where_it_does_not_fold(self):
        """Rows that mix_block does not take (and the CPU's torch form): leave_norm, or norm_streams, then mix."""
        rows = 3
        for leave, inject in ((True, True), (False, False)):
            _, down, up = site(inject, rows, DEVICE, seed=2)
            h, w, out, injection = streams(rows, DEVICE, 9, leave=leave)
            h_ref = h.clone()
            if leave:
                h_ref, normed = hcr.leave_norm(h_ref, out, injection, w, 1e-6, HC)
            else:
                normed = hcr.norm_streams(h_ref, w, 1e-6, HC)
            want = hcr.mix(normed, down, up, HC, inject=inject)
            got = hcr.site(h, out, injection, w, 1e-6, HC, down, up, inject=inject)
            with self.subTest(leave=leave):
                self.assertTrue(torch.equal(h, h_ref))
                self.assertTrue(torch.equal(got[0], want[0]))
                self.assertTrue(got[1] is None if not inject else torch.equal(got[1], want[1]))

    def test_site_folds_a_prefill_step_on_cuda(self):
        if DEVICE != "cuda":
            self.skipTest("site takes mix_block on CUDA rows only")
        rows = min(r for r, _ in hcr.DOWN_TILES)
        _, down, up = site(True, rows, "cuda")
        h, w, out, injection = streams(rows, "cuda", 4, leave=True)
        h_ref = h.clone()
        h_ref, normed = hcr.leave_norm(h_ref, out, injection, w, 1e-6, HC)
        want = hcr.mix_block(normed, down, up, HC)
        got = hcr.site(h, out, injection, w, 1e-6, HC, down, up)
        # the fused leave (leave_mix_block) leaves the same bytes; its gates associate the sums differently, so the
        # mixer's outputs are within a few BF16 steps of the two launches', not the same bytes
        self.assertTrue(torch.equal(h, h_ref))
        for mine, theirs in zip(got, want):
            err = float((mine.float() - theirs.float()).abs().max() / theirs.float().abs().max())
            self.assertLess(err, 2.0 ** -6)

    def test_it_is_the_torch_form_within_the_oracle_band(self):
        normed, down, up = site(True, 97, DEVICE, seed=3)
        mixed, injection = hcr.mix_block(normed, down, up, HC)
        want_mixed, want_injection = torch_form(normed.cpu(), down.cpu(), up.cpu(), True)
        for got, want in ((mixed, want_mixed), (injection, want_injection)):
            err = float((got.float().cpu() - want).abs().max() / want.abs().max())
            self.assertLess(err, 2.0 ** -6)             # a few BF16 steps: rounding order, not a wrong formula

    def test_the_table_picks_by_rows(self):
        least = min(r for r, _ in hcr.DOWN_TILES)
        self.assertIsNone(hcr.block_tiles(least - 1)["down"])
        for r, tile in hcr.DOWN_TILES:
            self.assertEqual((hcr.block_tiles(r)["down"], hcr.block_tiles(r)["up"]), (tile, hcr.UP_BLOCK_TILE))
        self.assertIsNone(hcr.block_tiles(min(r for r, _ in hcr.LEAVE_DOWN_TILES) - 1)["leave_down"])
        for r, tile in hcr.LEAVE_DOWN_TILES:
            self.assertEqual(hcr.block_tiles(r)["leave_down"], tile)
        self.assertEqual((hcr.narrow_tail(324, 64), hcr.narrow_tail(324, 128), hcr.narrow_tail(320, 64),
                          hcr.narrow_tail(324, 256)), (16, 0, 0, 128))

    def test_mix_takes_it_on_cuda(self):
        if DEVICE != "cuda":
            self.skipTest("mix folds CUDA rows only")
        normed, down, up = site(True, 200, "cuda")
        got = hcr.mix(normed, down, up, HC)
        want = hcr.mix_block(normed, down, up, HC)
        self.assertTrue(torch.equal(got[0], want[0]) and torch.equal(got[1], want[1]))


if __name__ == "__main__":
    unittest.main()
