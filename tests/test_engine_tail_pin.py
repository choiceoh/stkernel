"""`index_kpool_always_select_tail` is a promise about recency; hold the code to it.

The checkpoint sets the flag and `facts.architecture` asserts it, but the appended
tail covers only `[pool_len * pool_size, seq)` -- empty whenever the sequence
length is a whole number of pools. These tests hold the pin that closes that gap:
which pool it names, that it wins the top-k through the served `_select_pools`,
that other rows are left alone, and that a capture carries the horizon the rule
reads.
"""
import importlib.util
from pathlib import Path
import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError:                                        # the CPU runner without the wheel
    torch = None


@unittest.skipUnless(torch is not None, "requires PyTorch")
class TailPinPoolTests(unittest.TestCase):
    def test_the_pin_names_the_pool_that_just_completed_or_nothing(self):
        from engine.modules.sparse_indexer import tail_pin_pools

        seq = torch.tensor([0, 3, 4, 8, 9, 15, 16, 50004, 50005], dtype=torch.int32)
        # whole pools pin the one that completed; a remainder needs no pin (the appended
        # tail already covers the newest tokens); no complete pool means nothing to pin
        self.assertEqual(tail_pin_pools(seq, 4).tolist(), [-1, -1, 0, 1, -1, -1, 3, 12500, -1])

    def test_a_row_without_a_pin_is_untouched_and_a_pinned_row_clears_its_maximum(self):
        from engine.modules.sparse_indexer import pin_pools_in_logits

        logits = torch.tensor([[1.0, 9.0, 2.0, -4.0], [1.0, 9.0, 2.0, -4.0]])
        before = logits.clone()
        pin = torch.tensor([3, -1], dtype=torch.int32)
        pin_pools_in_logits(logits, pin)
        self.assertGreater(logits[0, 3].item(), logits[0].max().item() - 1.0)
        self.assertGreater(logits[0, 3].item(), before[0].max().item())
        self.assertTrue(torch.equal(logits[1], before[1]))                  # no pin, no change
        pin_pools_in_logits(logits, None)                                   # and the no-op form
        self.assertTrue(torch.equal(logits[1], before[1]))

    def test_a_row_of_negative_infinity_does_not_become_nan(self):
        from engine.modules.sparse_indexer import pin_pools_in_logits

        logits = torch.full((1, 4), float("-inf"))
        pin_pools_in_logits(logits, torch.tensor([2], dtype=torch.int32))
        self.assertTrue(torch.isinf(logits).all() and not torch.isnan(logits).any())


@unittest.skipUnless(torch is not None, "requires PyTorch")
class ServedSelectionTests(unittest.TestCase):
    """The pin has to reach the top-k through `_select_pools`, not beside it."""

    def select(self, ke, *, seq_lens=None, pool=None):
        from types import SimpleNamespace
        from engine.profiles.glm53 import net as net_mod

        # column `ke-1` (each row's newest complete pool) is the LOWEST scorer, so
        # only a pin can put it in the selection
        def indexer_logits(q8, keys, scales, w, ke):
            rows, n = q8.shape[0], keys.shape[0]
            scores = torch.arange(n, dtype=torch.float32)[None, :].repeat(rows, 1)
            for row, horizon in enumerate(ke.tolist()):
                if horizon >= 1:
                    scores[row, horizon - 1] = -10.0
            return scores

        net = SimpleNamespace(lanes=SimpleNamespace(indexer_logits=indexer_logits))
        q8 = torch.randn(2, 2, 8)
        w = torch.rand(2, 2)
        keys, scales = torch.randn(12, 8), torch.rand(12)
        kwargs = {} if seq_lens is None else dict(seq_lens=seq_lens, pool=pool)
        return net_mod.Glm53Net._select_pools(net, q8, w, keys, scales, ke, 12, 4, **kwargs)

    def test_a_whole_number_of_pools_keeps_its_newest_pool_with_the_pin(self):
        # row 0: 40 tokens = 10 pools, so pool 9 just completed and only a pin keeps it;
        # row 1: 15 tokens leaves a 3-token tail, so its newest tokens need no pin
        seq = torch.tensor([40, 15], dtype=torch.int32)
        ke = (seq // 4).to(torch.int32)                                      # [10, 3]
        pinned = self.select(ke, seq_lens=seq, pool=4)
        self.assertIn(9, pinned[0].tolist())                                 # recency kept
        unpinned = self.select(ke, seq_lens=None, pool=None)
        self.assertNotIn(9, unpinned[0].tolist())                            # without the pin it loses
        self.assertEqual(sorted(set(pinned[1].tolist())), sorted(set(unpinned[1].tolist())))

    def test_the_pinned_row_only_gains_a_slot(self):
        seq = torch.tensor([40, 15], dtype=torch.int32)
        ke = (seq // 4).to(torch.int32)
        pinned = self.select(ke, seq_lens=seq, pool=4)
        unpinned = self.select(ke, seq_lens=None, pool=None)
        self.assertEqual(set(pinned[0].tolist()) - set(unpinned[0].tolist()), {9})
        self.assertEqual(set(unpinned[0].tolist()) - set(pinned[0].tolist()), {5})


@unittest.skipUnless(torch is not None, "requires PyTorch")
class CaptureAndReferenceTests(unittest.TestCase):
    def dump(self, tmp):
        """Operands where the newest complete pool is strictly the lowest scorer.

        One head, every key `+q` except the newest pool's `-q`: `relu(q . k)` is 8 for
        the histories and 0 for pool 9, so no top-k can keep it -- only the pin can.
        """
        import importlib.util
        from engine.modules.selection_capture import SelectionCapture
        from engine.modules.sparse_indexer import tail_pin_pools

        spec = importlib.util.spec_from_file_location(
            "selection_reference", Path(__file__).resolve().parents[1] / "tools/selection_reference.py")
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)

        rows, heads, width, n_cand, k, pool = 2, 1, 8, 12, 4, 4
        seq = torch.tensor([40, 15], dtype=torch.int32)
        pin = tail_pin_pools(seq, pool)
        q8 = torch.ones(rows, heads, width, dtype=torch.float8_e4m3fn)
        keys = torch.ones(n_cand, width, dtype=torch.float8_e4m3fn)
        keys[9] = torch.full((width,), -1.0, dtype=torch.float32).to(torch.float8_e4m3fn)   # fp8 has no neg on the CPU lane
        operands = dict(q8=q8, w_eff=torch.ones(rows, heads), keys=keys, scales=torch.ones(n_cand),
                        ke=(seq // pool).to(torch.int32), n_cand=n_cand, k=k, seq_lens=seq, pool=pool)
        selected = reference.select_reference(operands)[1]                   # what the served path pins its way to
        capture = SelectionCapture(where=tmp)
        capture(3, q8=q8, w_eff=operands["w_eff"], keys=keys, scales=operands["scales"],
                ke=operands["ke"], n_cand=n_cand, k=k, selected=selected, prefill=True, seq=seq, pool=pool)
        return next(Path(tmp).glob("selection-*.pt")), pin, reference

    def test_a_capture_carries_the_horizon_the_rule_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, pin, _ = self.dump(tmp)
            stored = torch.load(path, map_location="cpu")
            self.assertEqual(stored["seq_lens"].tolist(), [40, 15])
            self.assertEqual(stored["pool"], 4)
            self.assertEqual(pin.tolist(), [9, -1])

    def test_the_reference_models_the_pin_and_reports_that_it_mattered(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, _, reference = self.dump(tmp)
            report = reference.compare(torch.load(path, map_location="cpu"))
        self.assertEqual(report["differing_rows"], 0)                        # the reference pins too
        self.assertEqual(report["tail_pin"]["pinned_rows"], 1)
        self.assertEqual(report["tail_pin"]["would_have_been_dropped"], 1)   # recency would have lost


if __name__ == "__main__":
    unittest.main()
