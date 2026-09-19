"""The compact MoE's combine in one launch (engine/kernels/moe_output.pair_sum).

An eager step dispatches only this rank's (token, route) pairs, one route a pair, and each token's pairs are summed in
FP32 and rounded once. That was an FP32 zero fill, the pairs widened to FP32, an atomic `index_add_` and the rounding:
four launches over an [N, H] FP32 buffer, about 1.2 ms a layer of a 4,096-token prefill chunk on a GB10
(measurements/qwen38_prefill_census_20260919), and a sum whose order the atomics chose. pair_sum reads each row's pairs
in their order -- the CPU's sequential `index_add_` -- so it is held to that byte for byte.

Runs on the served kernels on a GPU, or under TRITON_INTERPRET=1 with tests/test_engine_qwen38_kernels' accommodations.
"""
from pathlib import Path
import unittest

from tests.test_engine_qwen38_kernels import DEVICE, RUNS, RUNS_REASON, generator, served_kernels, torch

ROOT = Path(__file__).resolve().parents[1]


def case(gen, rows: int, hidden: int, counts):
    """pairs BF16 [P, H] and their ascending rows [P]: `counts[r]` pairs for row r (0 for a row no pair names)."""
    token = torch.cat([torch.full((c,), r, dtype=torch.int64) for r, c in enumerate(counts)]) if sum(counts) else \
        torch.empty(0, dtype=torch.int64)
    pairs = (torch.randn(int(token.numel()), hidden, generator=gen) * 3).to(torch.bfloat16)
    return pairs, token


def sequential(pairs, token, rows):
    """The reference: the CPU's index_add_, which adds a row's pairs one after another in their order, in FP32."""
    out = torch.zeros(rows, pairs.shape[1], dtype=torch.float32)
    out.index_add_(0, token, pairs.float())
    return out


@unittest.skipUnless(RUNS, RUNS_REASON)
class PairSumTests(unittest.TestCase):
    def run_sum(self, pairs, token, rows, dtype):
        from engine.kernels import moe_output
        out = torch.empty(rows, pairs.shape[1], dtype=dtype, device=DEVICE)
        with served_kernels():
            return moe_output.pair_sum(pairs.to(DEVICE), token.to(DEVICE), rows, out=out).cpu()

    def test_each_row_is_its_pairs_summed_in_order_and_rounded_once(self):
        gen = generator(7)
        # rows with no pair (first, middle, last), one pair, and the ten routes of a whole row; a tail past the block
        for hidden, counts in ((2560, [0, 1, 3, 0, 10, 2, 0]), (700, [4, 0, 0, 1, 7])):
            pairs, token = case(gen, len(counts), hidden, counts)
            want = sequential(pairs, token, len(counts))
            with self.subTest(hidden=hidden):
                self.assertTrue(torch.equal(self.run_sum(pairs, token, len(counts), torch.float32), want))
                got = self.run_sum(pairs, token, len(counts), torch.bfloat16)
                self.assertTrue(torch.equal(got.view(torch.int16), want.to(torch.bfloat16).view(torch.int16)))

    def test_a_step_with_no_pair_and_rows_past_the_last_are_zeros(self):
        gen = generator(8)
        pairs, token = case(gen, 3, 64, [0, 0, 0])
        self.assertTrue(torch.equal(self.run_sum(pairs, token, 3, torch.float32), torch.zeros(3, 64)))
        pairs, token = case(gen, 2, 64, [2, 1])
        got = self.run_sum(pairs, token, 5, torch.float32)                  # rows 2..4 name no pair
        self.assertTrue(torch.equal(got, sequential(pairs, token, 5)))
        self.assertTrue(torch.equal(got[2:], torch.zeros(3, 64)))

    def test_many_rows_find_their_own_pairs(self):
        gen = generator(9)
        counts = torch.randint(0, 11, (300,), generator=gen).tolist()        # a prefill's rows: 0..10 local routes each
        pairs, token = case(gen, 300, 256, counts)
        self.assertTrue(torch.equal(self.run_sum(pairs, token, 300, torch.float32), sequential(pairs, token, 300)))

    def test_the_entry_refuses_what_it_cannot_read(self):
        from engine.kernels import moe_output
        gen = generator(10)
        pairs, token = case(gen, 2, 64, [1, 1])
        pairs, token = pairs.to(DEVICE), token.to(DEVICE)
        with served_kernels():
            for args in ((pairs.float(), token, 2), (pairs, token.int(), 2), (pairs, token[:1], 2),
                         (pairs.t().contiguous().t(), token, 2), (pairs, token, -1)):
                with self.assertRaises(ValueError):
                    moe_output.pair_sum(*args)


class ServedLaneTests(unittest.TestCase):
    def test_the_compact_path_combines_with_one_launch(self):
        source = (ROOT / "engine/profiles/qwen38/lanes.py").read_text(encoding="utf-8")
        body = source[source.index("        local_ids, w = local_routes(ids, weights, first_expert, E)\n"):
                      source.index("    def on_main(fn):")]
        self.assertIn("return moe_output.pair_sum(pairs, token, x.shape[0])", body)
        self.assertNotIn("index_add_", body)
        self.assertNotIn("dtype=torch.float32", body)


if __name__ == "__main__":
    unittest.main()
