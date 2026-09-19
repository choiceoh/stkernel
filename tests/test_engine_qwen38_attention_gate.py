"""The QSA attention applies the layer's output gate in its final store (engine/QWEN38_CARRY.md Q4).

net._qsa finished each attention layer with (attended.float() * torch.sigmoid(gate.float())).to(bf16): five launches on
each of the 13 QSA layers of a step and three fp32 [rows, 6, 256] temporaries a prefill chunk. The attention kernel's
final store -- the one-split launch or the merge of several -- now rounds the attention to BF16, multiplies it by
sigmoid(gate) in fp32 (Triton's 1 / (1 + exp(-g)), the formula of torch's CUDA sigmoid) and rounds once.

On a GPU the gated launch is held to the layer's torch composition byte for byte; under TRITON_INTERPRET=1 (numpy's exp
beside torch's CPU one) to one BF16 value, with the structure -- which store gates, in which split -- the same.

    docker exec -e TRITON_INTERPRET=1 -w <repo> stk-test python3 -m unittest tests.test_engine_qwen38_attention_gate
"""
from pathlib import Path
import unittest
from unittest import mock

from tests import test_engine_qwen38_qsa_blocks as blocks_test                    # the module: its tests stay there
from tests.test_engine_qwen38_kernels import (DEVICE, INTERPRET, RUNS, RUNS_REASON, W, Launches, generator, randn,
                                              served_kernels, torch)

ROOT = Path(__file__).resolve().parents[1]


def bf16_steps(a, b) -> int:
    """The largest distance between two BF16 tensors in adjacent BF16 values."""
    order = lambda t: torch.where(t.view(torch.int16).to(torch.int32) < 0, -32768 - t.view(torch.int16).to(torch.int32),
                                  t.view(torch.int16).to(torch.int32))
    return int((order(a) - order(b)).abs().max())


@unittest.skipUnless(RUNS, RUNS_REASON)
class AttentionGateTests(unittest.TestCase):
    def test_the_gated_store_is_the_layer_s_gate_over_the_attention(self):
        from engine.kernels import qsa
        gen = generator(91)
        splits = set()
        for budget in ((12, 64) if INTERPRET else (12, W.budget)):
            for kv_heads in (W.kv_heads, 2):
                meta, q, k_cache, v_cache, blocks = blocks_test.BlockAttentionTests().case(gen, budget, kv_heads)
                fused = randn(gen, q.shape[0], q.shape[1], 2 * q.shape[2], scale=3.0)
                gate = fused[..., q.shape[2]:]                                 # the in_proj split's strided view
                attend = Launches(qsa._qsa_sparse_paged_gqa_splitk_kernel)
                merge = Launches(qsa._qsa_merge_splitk_kernel)
                args = (q, k_cache, v_cache, blocks, meta.positions32, meta.lengths, W.ratio, budget, meta.page_table,
                        meta.rows_req)
                with served_kernels(), mock.patch.object(qsa, "_qsa_sparse_paged_gqa_splitk_kernel", attend), \
                        mock.patch.object(qsa, "_qsa_merge_splitk_kernel", merge):
                    plain = qsa.qsa_sparse_paged_attention_blocks(*args)
                    gated = qsa.qsa_sparse_paged_attention_blocks(*args, gate=gate)
                want = (plain.float() * torch.sigmoid(gate.float())).to(torch.bfloat16)
                split_count = attend.grids[-1][2]
                splits.add(split_count)
                with self.subTest(budget=budget, kv_heads=kv_heads, splits=split_count):
                    self.assertEqual(attend.grids[0], attend.grids[1])                 # the same programs either way
                    self.assertEqual(len(merge.grids), 2 * int(split_count > 1))
                    self.assertTrue(bool(plain.any()))
                    if INTERPRET:
                        self.assertLessEqual(bf16_steps(gated, want), 1)
                    else:
                        self.assertTrue(torch.equal(gated, want))
        self.assertTrue(1 in splits and max(splits) > 1, splits)

    def test_the_gate_must_match_the_query(self):
        from engine.kernels import qsa
        gen = generator(92)
        meta, q, k_cache, v_cache, blocks = blocks_test.BlockAttentionTests().case(gen, 12, W.kv_heads)
        args = (q, k_cache, v_cache, blocks, meta.positions32, meta.lengths, W.ratio, 12, meta.page_table, meta.rows_req)
        bad = (randn(gen, q.shape[0], q.shape[1], q.shape[2] - 1),                     # a narrower head
               q.float(),                                                              # not BF16
               randn(gen, q.shape[0], q.shape[2], q.shape[1]).transpose(1, 2))         # strided along the head
        with served_kernels():
            for gate in bad:
                with self.subTest(shape=tuple(gate.shape), dtype=gate.dtype), \
                        self.assertRaisesRegex(ValueError, "output gate"):
                    qsa.qsa_sparse_paged_attention_blocks(*args, gate=gate)


class ServedLaneTests(unittest.TestCase):
    def test_the_layer_hands_the_gate_to_the_attention(self):
        source = (ROOT / "engine/profiles/qwen38/net.py").read_text()
        body = source[source.index("    def _qsa("):source.index("    # -- MoE")]
        self.assertIn("one_request=runs[\"one_request\"], gate=gate)", body)
        self.assertNotIn("torch.sigmoid(gate", body)
        self.assertIn("out = attended.reshape(N, Hq * D)", body)


if __name__ == "__main__":
    unittest.main()
