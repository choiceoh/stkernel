"""The selection reference decides set membership, ties and horizons -- test it on the CPU.

`tools/selection_reference.py` exists because the sparse indexer's selection for a
real long prefix has never been compared with a reference recomputed from its own
operands. These tests hold the comparison itself: a selection built by an
independent torch path agrees, a perturbed row is reported with the margin at the
cut, a horizon is respected, and a tie at the cut is reported as a tie rather than
as a wrong score.
"""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("selection_reference", ROOT / "tools/selection_reference.py")
reference = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reference)

ROWS, HEADS, WIDTH, CAND, K = 6, 4, 128, 96, 8


def operands(seed=0, horizons=None):
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn(ROWS, HEADS, WIDTH, generator=generator)
    k = torch.randn(CAND, WIDTH, generator=generator)
    w = torch.rand(ROWS, HEADS, generator=generator) + 0.5
    # the served read: raw fp8 q with the per-token scale folded into the weights,
    # keys dequantised with their per-row fp32 scale
    q8 = q.to(torch.float8_e4m3fn).float()
    scale = (k.abs().amax(dim=1) / 448.0).clamp_min(1e-6)
    keys = (k / scale[:, None]).to(torch.float8_e4m3fn)
    ke = torch.tensor(horizons or [CAND] * ROWS, dtype=torch.int32)
    return dict(layer=3, rows=ROWS, n_cand=CAND, k=K, heads=HEADS, width=WIDTH, prefill=True,
                q8=q8, w_eff=w, keys=keys, scales=scale, ke=ke)


def select_by_torch(dump):
    """An independent path: dequantised keys, fp32 scores, masked top-k, sorted=True."""
    keys_dequant = dump["keys"].float() * dump["scales"][:, None]
    logits = torch.einsum("mhd,nd->mhn", dump["q8"].float(), keys_dequant).relu_()
    logits = torch.einsum("mhn,mh->mn", logits, dump["w_eff"].float())
    horizon = dump["ke"].tolist()
    out = torch.full((ROWS, K), -1, dtype=torch.int32)
    for row in range(ROWS):
        scores = logits[row, :horizon[row]]
        take = min(K, scores.numel())
        out[row, :take] = torch.topk(scores, take, sorted=True).indices.to(torch.int32)
    return out


class SelectionReferenceTests(unittest.TestCase):
    def test_a_selection_built_by_an_independent_path_agrees(self):
        dump = operands(seed=11)
        dump["selected"] = select_by_torch(dump)
        report = reference.compare(dump)
        self.assertEqual(report["compared_rows"], ROWS)
        self.assertEqual(report["differing_rows"], 0)

    def test_a_perturbed_row_is_reported_with_the_margin_at_the_cut(self):
        dump = operands(seed=12)
        selected = select_by_torch(dump)
        row, swap_in, swap_out = 2, CAND - 4, int(selected[2, 7])
        selected[2, 7] = swap_in
        dump["selected"] = selected
        report = reference.compare(dump)
        self.assertEqual(report["differing_rows"], 1)
        reported = report["differing"][0]
        self.assertEqual(reported["row"], row)
        self.assertEqual(reported["only_captured"], [swap_in] if swap_in != swap_out else [])
        self.assertEqual(reported["shared"], K - 1)
        self.assertGreater(reported["tie_gap"], 0.0)                      # a real score gap, not a tie

    def test_a_horizon_shorter_than_the_candidates_is_respected(self):
        horizons = [CAND, CAND, 12, CAND, CAND, 5]
        dump = operands(seed=13, horizons=horizons)
        dump["selected"] = select_by_torch(dump)
        report = reference.compare(dump)
        self.assertEqual(report["differing_rows"], 0)
        within = dump["selected"][2].tolist()
        self.assertTrue(all(x < 12 for x in within if x >= 0))
        # a horizon shorter than k takes every candidate below it: the set, not an order
        self.assertEqual(sorted(x for x in dump["selected"][5].tolist() if x >= 0), [0, 1, 2, 3, 4])

    def test_a_tie_at_the_cut_reads_as_a_tie(self):
        dump = operands(seed=14)
        # a genuine tie has to live in the operands: candidate 9 gets candidate 8's key
        # and scale, so both positions score identically
        dump["keys"][9] = dump["keys"][8]
        dump["scales"][9] = dump["scales"][8]
        selected = select_by_torch(dump)
        self.assertIn(8, selected[0].tolist())
        selected[0][int((selected[0] == 8).nonzero()[0])] = 9     # the captured side took the twin
        dump["selected"] = selected
        report = reference.compare(dump)
        self.assertEqual(report["differing_rows"], 1)
        self.assertLess(report["differing"][0]["tie_gap"], 1e-5)          # the twin, not a wrong score

    def test_the_dump_round_trips_through_torch_save(self):
        dump = operands(seed=15)
        dump["selected"] = select_by_torch(dump)
        with tempfile.TemporaryDirectory() as where:
            path = Path(where) / "selection-L3-rows6.pt"
            torch.save(dump, path)
            report = reference.compare(torch.load(path, map_location="cpu"))
        self.assertEqual(report["differing_rows"], 0)
        self.assertEqual((report["layer"], report["rows"], report["k"]), (3, ROWS, K))


if __name__ == "__main__":
    unittest.main()
