"""No QSA input launch stores one address twice.

Two stores to one address in one launch are ordered only when the same GPU thread makes both. qsa._norm_rope_partial
(and _norm_rope_into, its twin inside the fused input launches) stored a head's norm over the whole head and then its
rotation over the first rotary channels. At the indexer's cell -- D 128 under four warps -- channels 32..63 belong to
one thread in the 128-wide store and to another in the 32-wide one, and on a GB10 the boot's qualify lost that race 7
times in 2000: one head's second rotated half left un-rotated, different on the next call, the torch reference steady
and equal to the CPU's (measurements/qwen38_qualify_soak_20260919). Through _norm_rope_into the same loss would have
stayed in the K cache and the index keys.

A compiled kernel's race cannot be run here. What can be held under TRITON_INTERPRET=1 is the property that rules it
out: every launch of these kernels writes each address at most once -- the interpreter's store is wrapped and its
addresses counted a launch -- and the launch still writes every address it owes, byte for byte what the two-store
form wrote.

    TRITON_INTERPRET=1 python3 -m unittest tests.test_engine_qwen38_store_once
"""
import contextlib
import unittest
from unittest import mock

from tests.test_engine_qwen38_kernels import EPS, INTERPRET, MAX_POSITION, THETA, W, generator, randn, served_kernels, torch
from tests.test_engine_qwen38_qsa_inputs import fresh, layer_case, nine_launches, two_launches

RUNS = INTERPRET and torch is not None
RUNS_REASON = "counts the interpreter's stores: requires TRITON_INTERPRET=1 with Triton and torch"


@contextlib.contextmanager
def launches():
    """Every launch made inside, as (kernel name, the addresses it stored in order): the interpreter's masked store
    (every tl.store ends there) is wrapped, and a launch is one call of the interpreter's grid executor."""
    import numpy as np
    from triton.runtime import interpreter as ti
    seen, current = [], []
    store, call = ti.InterpreterBuilder.create_masked_store, ti.GridExecutor.__call__

    def create_masked_store(self, ptrs, value, mask, *rest):
        current.extend(np.asarray(ptrs.data)[np.asarray(mask.data, dtype=bool)].ravel().tolist())
        return store(self, ptrs, value, mask, *rest)

    def grid_call(self, *args, **kwargs):
        del current[:]
        try:
            return call(self, *args, **kwargs)
        finally:
            seen.append((getattr(self.fn, "__name__", str(self.fn)), list(current)))

    with mock.patch.object(ti.InterpreterBuilder, "create_masked_store", create_masked_store), \
            mock.patch.object(ti.GridExecutor, "__call__", grid_call):
        yield seen


def twice(addresses) -> int:
    """How many addresses a launch stored more than once."""
    return len(addresses) - len(set(addresses))


@unittest.skipUnless(RUNS, RUNS_REASON)
class StoreOnceTests(unittest.TestCase):
    def test_the_auditor_sees_a_second_store(self):
        import triton
        import triton.language as tl

        @triton.jit
        def two_stores(OUT, N: tl.constexpr, H: tl.constexpr):
            d = tl.arange(0, N)
            tl.store(OUT + d, d.to(tl.float32))
            i = tl.arange(0, H)
            tl.store(OUT + i, -i.to(tl.float32))

        out = torch.zeros(8)
        with served_kernels(), launches() as seen:
            two_stores[(1,)](out, N=8, H=4)
        self.assertEqual([(name, twice(addresses)) for name, addresses in seen], [("two_stores", 4)])

    def test_norm_rope_partial_stores_each_address_once_and_every_address(self):
        from engine.kernels import qsa
        gen = generator(7)
        # the interpreter's widths, and the served indexer's cell itself: four heads of 128, 64 rotated
        for heads, dim, rotary in ((W.heads, W.head_dim, W.rotary), (W.idx_heads, W.idx_dim, W.rotary), (4, 128, 64)):
            for rows in (1, 3):
                x, w = randn(gen, rows, heads, dim), randn(gen, dim, scale=0.1)
                positions = torch.randint(0, MAX_POSITION, (rows,), generator=gen)
                with served_kernels(), launches() as seen:
                    out = qsa.norm_rope_partial(x, w, EPS, positions, THETA, rotary)
                with self.subTest(heads=heads, dim=dim, rows=rows):
                    (name, addresses), = seen
                    self.assertEqual(name, "_norm_rope_partial")
                    self.assertEqual(twice(addresses), 0)
                    self.assertEqual(len(addresses), out.numel())            # and nothing it owes is left unwritten

    def test_it_is_byte_for_byte_the_two_store_form(self):
        import triton
        import triton.language as tl
        from engine.kernels import qsa
        from engine.kernels.common.norm_rope import warm

        @triton.jit
        def two_store_form(X, W_, POS, INV, OUT, sXr, sXh, sO, sP, EPS_, D: tl.constexpr, R2: tl.constexpr,
                           BD: tl.constexpr, BR: tl.constexpr):
            r = tl.program_id(0)
            h = tl.program_id(1)
            base, out = X + r * sXr + h * sXh, OUT + r * sO + h * D
            d = tl.arange(0, BD)
            m = d < D
            x = tl.load(base + d, mask=m, other=0.0).to(tl.float32)
            scale = tl.rsqrt(tl.sum(x * x) / D + EPS_)
            w = tl.load(W_ + d, mask=m, other=0.0).to(tl.float32)
            tl.store(out + d, ((x * scale) * (1.0 + w)).to(OUT.dtype.element_ty), mask=m)
            i = tl.arange(0, BR)
            mi = i < R2
            xl = tl.load(base + i, mask=mi, other=0.0).to(tl.float32)
            xh = tl.load(base + R2 + i, mask=mi, other=0.0).to(tl.float32)
            wl = tl.load(W_ + i, mask=mi, other=0.0).to(tl.float32)
            wh = tl.load(W_ + R2 + i, mask=mi, other=0.0).to(tl.float32)
            lo = ((xl * scale) * (1.0 + wl)).to(OUT.dtype.element_ty).to(tl.float32)
            hi = ((xh * scale) * (1.0 + wh)).to(OUT.dtype.element_ty).to(tl.float32)
            angle = tl.load(POS + r * sP).to(tl.float32) * tl.load(INV + i, mask=mi, other=0.0)
            cos, sin = tl.cos(angle), tl.sin(angle)
            tl.store(out + i, (lo * cos - hi * sin).to(OUT.dtype.element_ty), mask=mi)
            tl.store(out + R2 + i, (lo * sin + hi * cos).to(OUT.dtype.element_ty), mask=mi)

        gen = generator(11)
        rows, heads, dim, rotary = 3, 4, 128, 64
        x, w = randn(gen, rows, heads, dim), randn(gen, dim, scale=0.1)
        positions = torch.randint(0, MAX_POSITION, (rows,), generator=gen)
        want = torch.empty_like(x)
        with served_kernels():
            got = qsa.norm_rope_partial(x, w, EPS, positions, THETA, rotary)
            two_store_form[(rows, heads)](x, w, positions, warm(x.device, rotary, THETA), want, x.stride(0), x.stride(1),
                                          want.stride(0), positions.stride(0), EPS, D=dim, R2=rotary // 2, BD=128, BR=32)
        self.assertTrue(torch.equal(got.view(torch.int16), want.view(torch.int16)))

    def test_a_layers_input_launches_store_each_address_once(self):
        # the fused launches (_norm_rope_into three times a program, straight into K and the index keys) and the nine
        # launches they replaced, over a case whose rows own distinct cache cells
        case = layer_case(81)
        for name, path in (("two launches", two_launches), ("nine launches", nine_launches)):
            views, _ = fresh(case)
            with served_kernels(), launches() as seen:
                path(case, views)
            with self.subTest(path=name):
                self.assertGreaterEqual(len(seen), 2)
                self.assertEqual({kernel: twice(addresses) for kernel, addresses in seen if twice(addresses)}, {})


if __name__ == "__main__":
    unittest.main()
