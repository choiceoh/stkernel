"""The prefill indexer selects in bounded passes; every row's top-k is the same as in one pass."""
import unittest.mock
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "requires PyTorch")
class SelectionTests(unittest.TestCase):
    def test_passes_of_select_rows_match_one_pass_and_bound_the_transient(self):
        import torch
        from types import SimpleNamespace
        from engine.profiles.glm53 import net as net_mod
        torch.manual_seed(0)
        rows, n_cand, k = 2600, 700, 128
        calls = []

        def indexer_logits(q8, keys, scales, w, ke):          # a fake lane: scores from the rows' ids, garbage past ke
            calls.append(q8.shape[0])
            base = (q8.float().sum(-1).sum(-1, keepdim=True) * 0.001 + torch.arange(keys.shape[0]).float()[None, :] * 0.37).sin()
            return base

        net = SimpleNamespace(lanes=SimpleNamespace(indexer_logits=indexer_logits))
        q8 = torch.randn(rows, 2, 8)
        w = torch.rand(rows, 2)
        keys, scales = torch.randn(n_cand, 8), torch.rand(n_cand)
        ke = torch.randint(1, n_cand + 1, (rows,), dtype=torch.int32)
        select = net_mod.Glm53Net._select_pools
        chunked = select(net, q8, w, keys, scales, ke, n_cand, k)
        self.assertEqual(calls, [1024, 1024, 552])            # SELECT_ROWS passes, the tail shorter
        calls.clear()
        with unittest.mock.patch.object(net_mod, "SELECT_ROWS", rows):
            single = select(net, q8, w, keys, scales, ke, n_cand, k)
        self.assertEqual(calls, [rows])
        self.assertTrue(torch.equal(chunked, single))
        self.assertTrue(((chunked < ke[:, None]) | (chunked == -1)).all())   # never a pool at or past the row's horizon

    def test_topk_positions_in_place_masks_the_callers_logits(self):
        import torch
        from engine.modules.sparse_indexer import topk_positions
        logits = torch.arange(12.0).view(3, 4)
        copy = logits.clone()
        out = topk_positions(logits, 2, valid=torch.tensor([4, 2, 0]))
        self.assertTrue(torch.equal(logits, copy))            # default: a copy
        out_inplace = topk_positions(logits, 2, valid=torch.tensor([4, 2, 0]), inplace=True)
        self.assertTrue(torch.equal(out, out_inplace))
        self.assertTrue(torch.isinf(logits[1, 2:]).all() and torch.isinf(logits[2]).all())
        self.assertEqual(out.tolist(), [[3, 2], [1, 0], [-1, -1]])


if __name__ == "__main__":
    import unittest.mock
    unittest.main()
