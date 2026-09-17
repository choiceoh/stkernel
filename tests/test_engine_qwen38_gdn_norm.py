"""GDN's output norm reads z through its strides, and the bytes are the copy's (engine/QWEN38_CARRY.md K2).

`net._gdn` / `_gdn_rows` hand `gated_norm` the gate as `z.view(N, HV, D)` over the in_proj split, a column slice whose
row stride is the whole projection's width. The wrapper used to flatten it with `reshape(rows * HV, D)`: at one row that
is a view, at more than one it has no flat view and copied -- once on each of the 36 GDN layers of every step with more
than one row (a C=1 verify step is two). The kernel now loads z at (row, head) through both strides.

    docker exec -e TRITON_INTERPRET=1 -w <repo> stk-test python3 -m unittest tests.test_engine_qwen38_gdn_norm
"""
import importlib.util
import os
import unittest
from unittest import mock

torch = None
if importlib.util.find_spec("torch") is not None:
    import torch
TRITON = importlib.util.find_spec("triton") is not None
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
RUNS = torch is not None and TRITON and (INTERPRET or torch.cuda.is_available())
DEVICE = "cpu" if INTERPRET else "cuda"


@unittest.skipUnless(RUNS, "requires Triton with CUDA, or TRITON_INTERPRET=1")
class GatedNormStridesTests(unittest.TestCase):
    HV, D, QKV = 12, 128, 3072                          # Qwen3.8's per-rank value heads and head width; q|k|v ahead of z

    def setUp(self):
        torch.manual_seed(836)
        if INTERPRET:
            patch = mock.patch.object(torch.Tensor, "is_cuda", property(lambda tensor: True))
            patch.start()
            self.addCleanup(patch.stop)

    def split(self, rows):
        """The in_proj row as net._gdn_rows splits it: q|k|v, z, b, a -- z a column slice viewed as [rows, HV, D]."""
        proj = torch.randn(rows, self.QKV + self.HV * self.D + 2 * self.HV, device=DEVICE).to(torch.bfloat16)
        _, z, _, _ = proj.split([self.QKV, self.HV * self.D, self.HV, self.HV], dim=-1)
        return z.view(rows, self.HV, self.D)

    def test_the_norm_reads_the_split_gate_as_its_copy(self):
        from engine.kernels import gdn
        weight = (torch.randn(self.D, device=DEVICE) * .1).to(torch.bfloat16)
        for rows in (1, 2, 4):
            with self.subTest(rows=rows):
                z = self.split(rows)
                core = torch.randn(rows, self.HV, self.D, device=DEVICE).to(torch.bfloat16)
                if rows > 1:
                    self.assertFalse(z.reshape(rows * self.HV, self.D)._base is z._base)   # the copy the wrapper made
                view = gdn.gated_norm(core, z, weight, 1e-6)
                copy = gdn.gated_norm(core, z.contiguous(), weight, 1e-6)
                self.assertTrue(torch.equal(view, copy))

    def test_the_gate_is_read_per_row_and_head(self):
        """A wrong row or head stride would read another head's gate: flipping one head's z changes only that head."""
        from engine.kernels import gdn
        weight = torch.ones(self.D, device=DEVICE, dtype=torch.bfloat16)
        z = self.split(3)
        core = torch.randn(3, self.HV, self.D, device=DEVICE).to(torch.bfloat16)
        base = gdn.gated_norm(core, z, weight, 1e-6).view(3, self.HV, self.D)
        z[2, 5] = -z[2, 5] - 1                                   # writes through the view into the projection row
        moved = gdn.gated_norm(core, z, weight, 1e-6).view(3, self.HV, self.D)
        changed = (moved != base).any(-1)
        expected = torch.zeros(3, self.HV, dtype=torch.bool, device=DEVICE)
        expected[2, 5] = True
        self.assertTrue(torch.equal(changed, expected))

    def test_the_wrapper_no_longer_flattens_z(self):
        import inspect
        from engine.kernels import gdn
        self.assertNotIn("z.reshape", inspect.getsource(gdn.gated_norm))


if __name__ == "__main__":
    unittest.main()
