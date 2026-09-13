"""Repeated KDA layers must not read the same sequence lengths back from the device."""
import unittest
from unittest.mock import patch

import torch

from engine.kernels.kda.index import (chunk_mark_indices, prepare_chunk_indices,
                                      prepare_chunk_offsets, single_sequence_bounds)


class KdaMetadataTests(unittest.TestCase):
    def test_changing_lengths_preserves_indices_without_a_readback_per_layer(self):
        single_sequence_bounds.cache_clear()
        readbacks = []
        original = torch.Tensor.tolist

        def record(tensor):
            readbacks.append(tensor.numel())
            return original(tensor)

        with patch.object(torch.Tensor, "tolist", record):
            for tokens in (9216, 130, 9216):
                for layer in range(33):
                    bounds = single_sequence_bounds(tokens, torch.device("cpu"))
                    indices = prepare_chunk_indices(bounds, 64)
                    offsets = prepare_chunk_offsets(bounds, 64)
                    chunks = (tokens + 63) // 64
                    self.assertTrue(torch.equal(indices[:, 0], torch.zeros(chunks, dtype=torch.int32)))
                    self.assertTrue(torch.equal(indices[:, 1], torch.arange(chunks, dtype=torch.int32)))
                    self.assertTrue(torch.equal(offsets, torch.tensor([0, chunks])))
        self.assertEqual(readbacks, [1, 1])

    def test_different_mark_patterns_keep_their_own_read_only_values(self):
        device = torch.device("cpu")
        first = chunk_mark_indices((12, 24, 36), device)
        other = chunk_mark_indices((12, 36), device)
        again = chunk_mark_indices((12, 24, 36), device)
        self.assertEqual(first.tolist(), [12, 24, 36])
        self.assertEqual(other.tolist(), [12, 36])
        self.assertIs(first, again)


if __name__ == "__main__":
    unittest.main()
