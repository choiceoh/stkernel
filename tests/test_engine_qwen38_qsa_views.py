"""A QSA layer hands its kernels views, not copies, and the bytes are the same (engine/QWEN38_CARRY.md Q1).

`Qwen38Net._qsa` copied three tensors on every QSA layer of every step: the index keys' column slice (`ik`), the raw
positions it expanded for the compression (which the compression never reads without a rope cache -- only the shape is
checked), and the column of the pooled groups' first positions (which `norm_rope_partial`'s wrapper copied once more,
because its kernel read positions without a stride). Compression and the stores already read rows through their
strides, and the norm now reads positions through theirs. Each case here runs the kernel on the view and on its copy
and requires identical bytes: the copies bought nothing. Three launches a QSA layer, 13 QSA layer calls a C=1 step
(12 in the verify graph, one in the MTP draft graph).

    docker exec -e TRITON_INTERPRET=1 -w <repo> stk-test python3 -m unittest tests.test_engine_qwen38_qsa_views
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
class QsaViewTests(unittest.TestCase):
    D, RATIO, RING = 128, 4, 8

    def setUp(self):
        torch.manual_seed(1917)
        if INTERPRET:                                   # the wrappers take the CUDA path only for CUDA tensors
            patch = mock.patch.object(torch.Tensor, "is_cuda", property(lambda tensor: True))
            patch.start()
            self.addCleanup(patch.stop)

    def rows(self, n, dtype=torch.bfloat16):
        """An in_proj-like row [n, idx_q + D] whose index-key half is a column slice, as net._qsa splits it."""
        proj = torch.randn(n, 4 * self.D + self.D, device=DEVICE).to(dtype)
        return proj[:, 4 * self.D:]

    def test_the_norm_reads_a_column_of_first_positions_as_its_copy(self):
        from engine.kernels import qsa
        x = torch.randn(7, 4, self.D, device=DEVICE).to(torch.bfloat16)
        w = (torch.randn(self.D, device=DEVICE) * .1).to(torch.bfloat16)
        first = torch.randint(0, 262144, (7, 3), device=DEVICE, dtype=torch.int64)
        column = first[:, 0]
        self.assertEqual(column.stride(), (3,))
        view = qsa.norm_rope_partial(x, w, 1e-6, column, 1e7, 64)
        copy = qsa.norm_rope_partial(x, w, 1e-6, column.contiguous(), 1e7, 64)
        self.assertTrue(torch.equal(view, copy))
        # the positions matter: a norm that read the flat storage would rotate rows 1.. by other positions
        flat = qsa.norm_rope_partial(x, w, 1e-6, first.reshape(-1)[:7].contiguous(), 1e7, 64)
        self.assertFalse(torch.equal(view, flat))

    def test_compression_reads_sliced_keys_and_an_expanded_position_view_as_their_copies(self):
        from engine.kernels import qsa
        n = 6
        ik = self.rows(n)
        self.assertNotEqual(ik.stride(), ik.contiguous().stride())
        ring = torch.randn(1, self.RING, 1, self.D, device=DEVICE).to(torch.bfloat16)
        table = torch.zeros(1, 1, device=DEVICE, dtype=torch.int32)
        rows_req = torch.zeros(n, device=DEVICE, dtype=torch.int32)
        starts = torch.tensor([0, n], device=DEVICE, dtype=torch.int32)
        positions = torch.arange(9, 9 + n, device=DEVICE, dtype=torch.int64)
        key_slots = torch.where((positions + 1) % self.RATIO == 0, positions // self.RATIO,
                                torch.full_like(positions, -1)).to(torch.int32)
        raw_positions = positions[:, None, None].expand(n, 1, 3)
        view = qsa.qsa_compress_groups_with_ratio(ik[:, None, :], raw_positions, ring, table, rows_req, starts,
                                                  positions, key_slots, self.RATIO)
        copy = qsa.qsa_compress_groups_with_ratio(ik.contiguous()[:, None, :], raw_positions.contiguous(), ring, table,
                                                  rows_req, starts, positions, key_slots, self.RATIO)
        self.assertTrue(torch.equal(view[0], copy[0]))
        self.assertTrue(torch.equal(view[1], copy[1]))
        self.assertTrue(bool((key_slots >= 0).any()))              # at least one group closes, so a pool was written

    def test_the_stores_write_a_sliced_key_row_as_its_copy(self):
        from engine.kernels import qsa
        ik = self.rows(6)
        slots = torch.tensor([3, -1, 9, 12, 0, 7], device=DEVICE, dtype=torch.int32)
        view = torch.zeros(2, self.RING, 1, self.D, device=DEVICE, dtype=torch.bfloat16)
        copy = view.clone()
        qsa.qsa_store_cache_rows(view, slots, ik)
        qsa.qsa_store_cache_rows(copy, slots, ik.contiguous())
        self.assertTrue(torch.equal(view, copy))
        self.assertTrue(bool(view.any()))

    def test_the_layer_hands_views(self):
        """The call site itself: no `.contiguous()` in the layer; since Q3 the raw keys and the heads go to the two fused
        launches as views of the in_proj row (the compression's positions and first positions are the kernel's own)."""
        from pathlib import Path
        source = (Path(__file__).resolve().parents[1] / "engine/profiles/qwen38/net.py").read_text()
        body = source[source.index("    def _qsa("):source.index("    # -- MoE")]
        self.assertIn("ik = idx[:, idx_q:]\n", body)
        self.assertIn("lanes.qsa_index_keys(ik, ring,", body)
        # the rotary positions: the cache's, or a picture's sequence's mRoPE positions (net.StepMeta.rope)
        self.assertIn("idx[:, :idx_q].view(N, F.idx_heads, F.idx_dim), ik, rope,", body)
        self.assertIn("rope = meta.positions if rope is None else rope", body)
        self.assertNotIn(".contiguous()", body)                  # q is qsa_inputs' own contiguous output


if __name__ == "__main__":
    unittest.main()
