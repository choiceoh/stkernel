"""The fused normalisations against the torch forms they replace (45차 §69).

`profiles/glm53/drafter.rmsnorm` and `.rope` are the definition; `kernels/norm_rope` is the same arithmetic in
one launch each. These pin what "the same" is allowed to mean: the rotary angle is bit-identical because both
read the same inverse-frequency table, and the normalised value may differ by at most one bf16 step because the
two sum a row in different orders.

Only the drafter reads these. A last-bit difference there moves a draft, never a served token: the target's
logits decide what is emitted and the block verification decides what is kept.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from engine.kernels.common import norm_rope as K  # noqa: E402
from engine.profiles.glm53.drafter import rmsnorm, rope  # noqa: E402

CUDA = torch.cuda.is_available()


def steps(a, b):
    """How many bf16 representations apart two same-shaped bf16 tensors are, element by element."""
    return (a.view(torch.int16).int() - b.view(torch.int16).int()).abs()


class ContractTests(unittest.TestCase):
    """What the wrappers refuse, on any device."""

    def test_the_shapes_must_be_the_ones_the_drafter_passes(self):
        x = torch.zeros(4, 2, 8)
        w = torch.zeros(8)
        pos = torch.zeros(4, dtype=torch.int64)
        with self.assertRaises(ValueError):
            K.norm_rope(x.reshape(8, 8), w, 1e-5, pos, 1e4)          # not [N, heads, D]
        with self.assertRaises(ValueError):
            K.norm_rope(x, torch.zeros(7), 1e-5, pos, 1e4)           # weight is not the head width
        with self.assertRaises(ValueError):
            K.norm_rope(x, w, 1e-5, torch.zeros(3, dtype=torch.int64), 1e4)   # a position per row
        with self.assertRaises(ValueError):
            K.norm(x, torch.zeros(2, 8), 1e-5)                       # the norm weight is one row
        with self.assertRaises(ValueError):
            K.add_norm(x, torch.zeros(4, 2, 7), w, 1e-5)             # the two sides of a join are one shape
        with self.assertRaises(ValueError):
            K.norm_rope(torch.zeros(4, 2, 6), torch.zeros(6), 1e-5, pos, 1e4)  # 6 is not a power of two

    def test_the_table_is_built_once_per_device_dimension_and_theta(self):
        first = K.warm(torch.device("cpu"), 8, 10000.0)
        self.assertIs(first, K.warm(torch.device("cpu"), 8, 10000.0))
        self.assertIsNot(first, K.warm(torch.device("cpu"), 8, 1000000.0))
        self.assertTrue(torch.equal(first, 1.0 / (10000.0 ** (torch.arange(0, 8, 2, dtype=torch.float32) / 8))))


@unittest.skipUnless(CUDA, "the fused kernels are the CUDA path")
class KernelTests(unittest.TestCase):
    """The kernel against the definition, at the shapes the drafter runs."""

    shapes = [(6, 2, 128), (6, 8, 128), (24, 8, 128), (30, 2, 128), (1, 2, 128)]

    def rows(self, rows, heads, dim, seed):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        x = torch.randn(rows, heads, dim, device="cuda", generator=gen).bfloat16()
        w = torch.randn(dim, device="cuda", generator=gen).abs().add(0.5).bfloat16()
        pos = torch.randint(0, 900_000, (rows,), device="cuda", generator=gen)
        return x, w, pos

    def test_the_rotary_is_within_one_bf16_step_of_the_torch_form(self):
        for shape in self.shapes:
            with self.subTest(shape=shape):
                x, w, pos = self.rows(*shape, seed=sum(shape))
                for theta in (10000.0, 1000000.0):
                    gap = steps(rope(rmsnorm(x, w, 1e-5), pos, theta), K.norm_rope(x, w, 1e-5, pos, theta))
                    self.assertLessEqual(int(gap.max()), 1)

    def test_the_norm_is_within_one_bf16_step_of_the_torch_form(self):
        for shape in [(6, 4096), (24, 4096), (6, 2, 128), (30, 12288)]:
            with self.subTest(shape=shape):
                gen = torch.Generator(device="cuda").manual_seed(len(shape) + shape[0])
                x = torch.randn(*shape, device="cuda", generator=gen).bfloat16()
                w = torch.randn(shape[-1], device="cuda", generator=gen).abs().add(0.5).bfloat16()
                self.assertLessEqual(int(steps(rmsnorm(x, w, 1e-5), K.norm(x, w, 1e-5)).max()), 1)

    def test_almost_every_element_is_exactly_the_torch_value(self):
        """The tolerance above is a bound, not the behaviour: the two differ on a handful of elements in a
        hundred thousand, and only where the norm's own summation order lands on the other side of a rounding
        boundary. A regression that moved the arithmetic would blow through this."""
        moved = total = 0
        for trial in range(64):
            x, w, pos = self.rows(6, 2, 128, seed=trial)
            gap = steps(rope(rmsnorm(x, w, 1e-5), pos, 10000.0), K.norm_rope(x, w, 1e-5, pos, 10000.0))
            moved += int((gap > 0).sum())
            total += gap.numel()
        self.assertLess(moved / total, 1e-3)

    def test_the_heads_are_read_where_the_projection_left_them(self):
        """The block hands over one third of a fused qkv, reshaped: rows wider than the heads read here, so
        `stride(0) != heads * D`. Reading it in place is the whole point -- a copy first would put back the
        launch the fusion took out -- and it has to be the same answer as the copy."""
        gen = torch.Generator(device="cuda").manual_seed(21)
        fused = torch.randn(6, (8 + 2 + 2) * 128, device="cuda", generator=gen).bfloat16()
        w = torch.randn(128, device="cuda", generator=gen).abs().add(0.5).bfloat16()
        pos = torch.randint(0, 900_000, (6,), device="cuda", generator=gen)
        q, k, v = fused.split((8 * 128, 2 * 128, 2 * 128), -1)
        for part, heads in ((q, 8), (k, 2), (v, 2)):
            view = part.reshape(6, heads, 128)
            self.assertNotEqual(view.stride(0), heads * 128)
            self.assertTrue(torch.equal(K.norm_rope(view, w, 1e-5, pos, 1e4),
                                        K.norm_rope(view.contiguous(), w, 1e-5, pos, 1e4)))

    def test_the_residual_join_is_the_add_and_the_norm_that_read_it(self):
        """A block writes `res = res + x` and then normalises `res`, twice a layer. Fused it is one launch,
        and it has to be bit for bit what the pair was -- the sum is a residual every later layer reads."""
        for rows, width in ((7, 4096), (28, 4096), (1, 4096), (7, 512), (7, 12288)):
            with self.subTest(rows=rows, width=width):
                gen = torch.Generator(device="cuda").manual_seed(rows + width)
                a = torch.randn(rows, width, device="cuda", generator=gen).bfloat16()
                b = torch.randn(rows, width, device="cuda", generator=gen).bfloat16()
                w = torch.randn(width, device="cuda", generator=gen).abs().add(0.5).bfloat16()
                total, normed = K.add_norm(a, b, w, 1e-5)
                self.assertTrue(torch.equal(total, a + b))
                self.assertTrue(torch.equal(normed, K.norm(a + b, w, 1e-5)))

    def test_a_missing_table_during_capture_is_an_error_not_an_allocation(self):
        K._TABLES.pop(("cuda:0", 64, 12345.0), None)
        graph, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
        x = torch.randn(2, 1, 64, device="cuda").bfloat16()
        w = torch.ones(64, device="cuda", dtype=torch.bfloat16)
        pos = torch.zeros(2, device="cuda", dtype=torch.int64)
        K.norm_rope(x, w, 1e-5, pos, 12345.0)                   # warms it outside capture
        K._TABLES.pop(("cuda:0", 64, 12345.0))
        with torch.cuda.stream(stream):
            graph.capture_begin()
            try:
                with self.assertRaises(RuntimeError):
                    K.norm_rope(x, w, 1e-5, pos, 12345.0)
            finally:
                try:
                    graph.capture_end()
                except Exception:                               # the capture is already invalid; nothing to keep
                    pass
        torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
