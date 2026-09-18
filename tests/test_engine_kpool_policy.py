"""GLM5Next selects complete pools by score and appends incomplete tails.

Upstream references checked on 2026-09-18:
Transformers modeling_glm5_next.py blob 8efbef9839129b5db0164653c1d9e978ab215666
vLLM sparse_indexer.py blob 1fc42c9e4114bf9ff997dcf49cd88015127b5b3e
"""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipUnless(torch is not None, "requires PyTorch")
class KpoolPolicyTests(unittest.TestCase):
    def select(self, lengths):
        from engine.profiles.glm53.net import Glm53Net

        # The newest complete pool has the worst score; older scores are unique.
        def logits(q, keys, scales, weights, horizon):
            values = torch.arange(12, 0, -1, dtype=torch.float32).repeat(len(q), 1)
            for row, end in enumerate(horizon.tolist()):
                if end:
                    values[row, end - 1] = -10
            return values

        rows = len(lengths)
        net = SimpleNamespace(lanes=SimpleNamespace(indexer_logits=logits))
        return Glm53Net._select_pools(net, torch.zeros(rows, 1, 8), torch.ones(rows, 1),
                                     torch.zeros(12, 8), torch.ones(12), lengths // 4,
                                     12, 4, seq_lens=lengths, pool=4)

    def test_newest_complete_pool_has_no_recency_override(self):
        # Exercise decode width and both sides of the prefill pass boundary.
        for rows in (1, 32, 1025):
            with self.subTest(rows=rows):
                chosen = self.select(torch.full((rows,), 40, dtype=torch.int32))
                for row in chosen.tolist():
                    self.assertEqual(set(row), {0, 1, 2, 3})

    def test_only_incomplete_tail_bypasses_pool_selection(self):
        from engine.modules.sparse_indexer import select_with_tail

        lengths = torch.tensor([40, 41, 42, 43, 44], dtype=torch.int32)
        tokens = select_with_tail(self.select(lengths), lengths, 4)
        for i, length in enumerate(lengths.tolist()):
            expected = set(range(16)) | set(range(length // 4 * 4, length)) | {-1}
            if length % 4 == 3:
                expected.remove(-1)
            self.assertEqual(set(tokens[i].tolist()), expected)

    def test_short_prefix_keeps_every_visible_token(self):
        from engine.modules.sparse_indexer import select_with_tail

        lengths = torch.arange(0, 16, dtype=torch.int32)
        tokens = select_with_tail(self.select(lengths), lengths, 4)
        for length, row in enumerate(tokens.tolist()):
            self.assertEqual({token for token in row if token >= 0}, set(range(length)))


@unittest.skipUnless(torch is not None, "requires PyTorch")
class SelectionReferencePolicyTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location(
            "selection_reference", Path(__file__).resolve().parents[1] / "tools/selection_reference.py")
        self.reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.reference)

    def dump(self):
        # Strictly decreasing positive scores: pool 9 loses by a large margin.
        keys = torch.arange(10, 0, -1, dtype=torch.float32)[:, None].expand(10, 8).contiguous()
        return dict(layer=3, n_cand=10, k=4, heads=1, width=8,
                    q8=torch.ones(1, 1, 8), keys=keys, scales=torch.ones(10),
                    w_eff=torch.ones(1, 1), ke=torch.tensor([10], dtype=torch.int32),
                    seq_lens=torch.tensor([40], dtype=torch.int32), pool=4,
                    selected=torch.tensor([[0, 1, 2, 3]], dtype=torch.int32))

    def test_capture_records_policy_and_matches_upstream_selection(self):
        from engine.modules.selection_capture import SelectionCapture

        dump = self.dump()
        with tempfile.TemporaryDirectory() as directory:
            capture = SelectionCapture(where=directory)
            path = capture(3, q8=dump['q8'], w_eff=dump['w_eff'], keys=dump['keys'],
                           scales=dump['scales'], ke=dump['ke'], n_cand=10, k=4,
                           selected=dump['selected'], prefill=True,
                           seq=dump['seq_lens'], pool=4)
            stored = torch.load(path, map_location='cpu', weights_only=True)
        report = self.reference.compare(stored)
        self.assertEqual(stored['selection_policy'], 'scored_complete_pools')
        self.assertEqual(report['capture_policy'], 'scored_complete_pools')
        self.assertEqual(report['differing_rows'], 0)

    def test_legacy_pin_is_reported_as_a_policy_difference(self):
        dump = self.dump()
        dump['selected'][0, 3] = 9
        report = self.reference.compare(dump)
        self.assertEqual(report['capture_policy'], 'unspecified')
        self.assertEqual(report['differing_rows'], 1)
        self.assertEqual(report['differing'][0]['only_captured'], [9])
        self.assertEqual(report['differing'][0]['only_reference'], [3])
        self.assertGreater(report['differing'][0]['tie_gap'], 0)


if __name__ == '__main__':
    unittest.main()
